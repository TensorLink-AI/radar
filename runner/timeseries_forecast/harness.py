"""Frozen training harness for trusted execution.

Usage: python harness.py /workspace/submission.py

Loads a miner's submission module, validates it, runs the training loop,
saves a checkpoint, and prints final metrics. The harness owns the training
loop entirely -- miners only provide architecture, optimizer, and optional
config/loss/scheduler/hooks (init_weights, transform_batch, on_step_end,
configure_amp).
"""

import importlib.util
import io
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from prepare import get_dataloader, validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES

TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 300))
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs("/workspace/checkpoints", exist_ok=True)
os.makedirs("/workspace/logs", exist_ok=True)

_AMP_DTYPE_WHITELIST = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ── Default loss (quantile / pinball) ─────────────────────────────

def default_loss(predictions, targets, quantiles_list):
    """Quantile loss (pinball loss) over all quantiles."""
    target_expanded = targets.unsqueeze(-1)  # (B, P, V, 1)
    errors = target_expanded - predictions    # (B, P, V, Q)
    q = torch.tensor(quantiles_list, device=predictions.device)
    return torch.max(q * errors, (q - 1) * errors).mean()


# ── Submission loader ─────────────────────────────────────────────

def _load_submission(path: str):
    """Load a plaintext submission module."""
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Config clamping ───────────────────────────────────────────────

_DEFAULTS = {"batch_size": 64, "grad_accum_steps": 1, "grad_clip": 1.0, "eval_interval": 200}
_CLAMPS = {
    "batch_size": (1, 512),
    "grad_accum_steps": (1, 16),
    "grad_clip": (0.0, 100.0),
    "eval_interval": (50, 10000),
}


def _read_config(sub) -> dict:
    cfg = dict(_DEFAULTS)
    if hasattr(sub, "training_config") and callable(sub.training_config):
        try:
            user_cfg = sub.training_config()
            if isinstance(user_cfg, dict):
                for k, v in user_cfg.items():
                    if k in _CLAMPS:
                        try:
                            numeric_v = float(v)
                        except (TypeError, ValueError):
                            continue
                        lo, hi = _CLAMPS[k]
                        cfg[k] = type(_DEFAULTS[k])(max(lo, min(hi, numeric_v)))
        except Exception as e:
            print(f"WARNING: training_config() failed: {e}", file=sys.stderr)
    return cfg


# ── AMP config ────────────────────────────────────────────────────

def _read_amp_config(sub) -> dict:
    """Read AMP config from submission's configure_amp() hook.

    Returns {"enabled": bool, "dtype": str} with validated dtype.
    Default: {"enabled": True, "dtype": "bfloat16"}.
    """
    default = {"enabled": True, "dtype": "bfloat16"}
    if not (hasattr(sub, "configure_amp") and callable(sub.configure_amp)):
        return default
    try:
        amp_cfg = sub.configure_amp()
        if not isinstance(amp_cfg, dict):
            print("WARNING: configure_amp() did not return a dict, using default", file=sys.stderr)
            return default
        enabled = amp_cfg.get("enabled", True)
        dtype_str = amp_cfg.get("dtype", "bfloat16")
        if dtype_str not in _AMP_DTYPE_WHITELIST:
            print(f"WARNING: configure_amp() dtype '{dtype_str}' not in {list(_AMP_DTYPE_WHITELIST)}, using bfloat16", file=sys.stderr)
            dtype_str = "bfloat16"
        return {"enabled": bool(enabled), "dtype": dtype_str}
    except Exception as e:
        print(f"WARNING: configure_amp() failed: {e}, using default", file=sys.stderr)
        return default


# ── Batch validation ──────────────────────────────────────────────

def _validate_batch(batch: dict, batch_size: int, label: str = "batch") -> bool:
    """Validate that a batch has correct shapes and no NaN/Inf.

    Returns True if valid, False otherwise (with warning printed).
    """
    if not isinstance(batch, dict) or "context" not in batch or "target" not in batch:
        print(f"WARNING: {label} missing 'context' or 'target' keys", file=sys.stderr)
        return False

    context = batch["context"]
    target = batch["target"]

    expected_ctx = (batch_size, CONTEXT_LEN, NUM_VARIATES)
    expected_tgt = (batch_size, PREDICTION_LEN, NUM_VARIATES)

    if tuple(context.shape) != expected_ctx:
        print(f"WARNING: {label} context shape {tuple(context.shape)} != expected {expected_ctx}", file=sys.stderr)
        return False
    if tuple(target.shape) != expected_tgt:
        print(f"WARNING: {label} target shape {tuple(target.shape)} != expected {expected_tgt}", file=sys.stderr)
        return False

    if torch.isnan(context).any() or torch.isinf(context).any():
        print(f"WARNING: {label} context contains NaN or Inf", file=sys.stderr)
        return False
    if torch.isnan(target).any() or torch.isinf(target).any():
        print(f"WARNING: {label} target contains NaN or Inf", file=sys.stderr)
        return False

    return True


# ── Stdout capture ────────────────────────────────────────────────

class TeeWriter:
    """Write to both original stdout and a capped StringIO buffer."""

    _MAX_BUFFER = 10 * 1024 * 1024  # 10 MB cap

    def __init__(self, original):
        self.original = original
        self.buffer = io.StringIO()
        self._size = 0

    def write(self, text):
        self.original.write(text)
        if self._size < self._MAX_BUFFER:
            self.buffer.write(text)
            self._size += len(text)

    def flush(self):
        self.original.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()


# ── Main ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python harness.py <submission.py>", file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1]

    # Capture stdout
    tee = TeeWriter(sys.stdout)
    sys.stdout = tee

    # Load submission
    try:
        sub = _load_submission(target)
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: failed to load submission: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate required functions
    if not hasattr(sub, "build_model") or not callable(sub.build_model):
        print("ERROR: submission missing callable build_model()", file=sys.stderr)
        sys.exit(1)
    if not hasattr(sub, "build_optimizer") or not callable(sub.build_optimizer):
        print("ERROR: submission missing callable build_optimizer()", file=sys.stderr)
        sys.exit(1)

    # Read config
    cfg = _read_config(sub)
    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum_steps"]
    grad_clip = cfg["grad_clip"]
    eval_interval = cfg["eval_interval"]

    # Read AMP config
    amp_cfg = _read_amp_config(sub)
    amp_enabled = amp_cfg["enabled"]
    amp_dtype = _AMP_DTYPE_WHITELIST[amp_cfg["dtype"]]

    # Build model
    try:
        model = sub.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).cuda()
    except Exception as e:
        print(f"ERROR: build_model() failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Count params before init_weights
    num_params = sum(p.numel() for p in model.parameters())

    # init_weights hook
    has_init_weights = hasattr(sub, "init_weights") and callable(sub.init_weights)
    if has_init_weights:
        try:
            sub.init_weights(model)
            params_after = sum(p.numel() for p in model.parameters())
            if params_after != num_params:
                print(
                    f"ERROR: init_weights() changed param count from {num_params} to {params_after}",
                    file=sys.stderr,
                )
                sys.exit(1)
        except SystemExit:
            raise
        except Exception as e:
            print(f"WARNING: init_weights() failed: {e}", file=sys.stderr)

    print(f"model_params_M: {num_params / 1e6:.1f}")

    # Measure FLOPs-equivalent on CPU — must match Phase C validator measurement.
    try:
        from flops import compute_flops_equivalent
        cpu_model = model.cpu()
        flops_equiv = compute_flops_equivalent(cpu_model, CONTEXT_LEN, NUM_VARIATES, "cpu")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"flops_equivalent_size: {flops_equiv}")
    except Exception as e:
        print(f"WARNING: FLOPs measurement failed: {e}", file=sys.stderr)
        flops_equiv = 0

    # Shape check
    try:
        with torch.no_grad():
            dummy = torch.randn(2, CONTEXT_LEN, NUM_VARIATES).cuda()
            out = model(dummy)
        expected = (2, PREDICTION_LEN, NUM_VARIATES, len(QUANTILES))
        if out.shape != expected:
            print(f"ERROR: wrong output shape: got {tuple(out.shape)}, expected {expected}", file=sys.stderr)
            sys.exit(1)
    except Exception:
        pass  # Skip check on execution error (lazy init, etc.)

    # torch.compile
    if getattr(sub, "COMPILE", False):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"WARNING: torch.compile failed, using eager mode: {e}", file=sys.stderr)

    # Build optimizer
    try:
        optimizer = sub.build_optimizer(model)
    except Exception as e:
        print(f"ERROR: build_optimizer() failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Build scheduler
    total_steps = (TIME_BUDGET * batch_size) // 2  # rough estimate
    scheduler = None
    if hasattr(sub, "build_scheduler") and callable(sub.build_scheduler):
        try:
            scheduler = sub.build_scheduler(optimizer, total_steps)
        except Exception as e:
            print(f"WARNING: build_scheduler() failed: {e}", file=sys.stderr)

    # Loss function
    loss_fn = default_loss
    if hasattr(sub, "compute_loss") and callable(sub.compute_loss):
        loss_fn = sub.compute_loss

    # Detect hooks
    has_transform_batch = hasattr(sub, "transform_batch") and callable(sub.transform_batch)
    has_on_step_end = hasattr(sub, "on_step_end") and callable(sub.on_step_end)
    transform_batch_disabled = False
    transform_batch_slow_count = 0
    transform_batch_fail_count = 0

    # ── Training loop ─────────────────────────────────────────────
    model.train()
    step = 0
    optim_step = 0
    running_loss = 0.0
    loss_curve: list[float] = []
    start = time.time()

    for batch in get_dataloader(batch_size=batch_size):
        if time.time() - start > TIME_BUDGET:
            break

        context = batch["context"].cuda()   # (B, context_len, num_variates)
        targets = batch["target"].cuda()     # (B, prediction_len, num_variates)
        current_batch = {"context": context, "target": targets}

        # transform_batch hook
        if has_transform_batch and not transform_batch_disabled:
            try:
                t0 = time.perf_counter()
                current_batch = sub.transform_batch(current_batch, step, total_steps)
                elapsed_tb = time.perf_counter() - t0
                if elapsed_tb > 0.05:
                    transform_batch_slow_count += 1
                if transform_batch_slow_count >= 3:
                    transform_batch_disabled = True
                    print("WARNING: transform_batch disabled (3 consecutive slow calls >50ms)", file=sys.stderr)
                else:
                    transform_batch_slow_count = 0  # reset on fast call
                if not _validate_batch(current_batch, batch_size, "transform_batch output"):
                    transform_batch_fail_count += 1
                    if transform_batch_fail_count >= 5:
                        transform_batch_disabled = True
                        print("WARNING: transform_batch disabled (5 failures)", file=sys.stderr)
                    current_batch = {"context": context, "target": targets}
                else:
                    transform_batch_fail_count = 0
                    context = current_batch["context"]
                    targets = current_batch["target"]
            except Exception as e:
                transform_batch_fail_count += 1
                if transform_batch_fail_count >= 5:
                    transform_batch_disabled = True
                    print("WARNING: transform_batch disabled (5 failures)", file=sys.stderr)
                else:
                    print(f"WARNING: transform_batch() failed: {e}", file=sys.stderr)
                context = batch["context"].cuda()
                targets = batch["target"].cuda()

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(context)
            loss = loss_fn(predictions, targets, QUANTILES)
            loss = loss / grad_accum

        loss.backward()

        if (step + 1) % grad_accum == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
            optim_step += 1

            # on_step_end hook
            if has_on_step_end:
                try:
                    sub.on_step_end(
                        model=model, optimizer=optimizer,
                        step=optim_step, total_steps=total_steps,
                        loss_value=loss.item() * grad_accum,
                    )
                except Exception as e:
                    print(f"WARNING: on_step_end() failed: {e}", file=sys.stderr)

        step += 1
        running_loss += loss.item() * grad_accum

        if step % eval_interval == 0:
            if hasattr(model, "reset"):
                model.reset()
            model.eval()
            val = validate(model)
            model.train()
            avg = running_loss / eval_interval
            loss_curve.append(avg)
            running_loss = 0.0
            elapsed = time.time() - start
            print(f"step {step} | loss: {avg:.4f} | crps: {val['crps']:.4f} | ncrps: {val['ncrps']:.4f} | mase: {val['mase']:.4f} | time: {elapsed:.0f}s")

    # Flush trailing gradient accumulation
    if step % grad_accum != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optim_step += 1

    # ── Save checkpoint with safetensors (no final eval — validators do Phase C) ──
    save_file(model.state_dict(), "/workspace/checkpoints/model.safetensors")
    elapsed = time.time() - start

    print(f"training_seconds: {elapsed:.1f}")
    print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6:.1f}")
    print(f"num_steps: {step}")
    print(f"num_optim_steps: {optim_step}")
    print(f"num_params_M: {num_params / 1e6:.1f}")
    print(f"flops_equivalent_size: {flops_equiv}")
    print(f"loss_curve: {loss_curve}")
    if amp_cfg != {"enabled": True, "dtype": "bfloat16"}:
        print(f"amp_config: {amp_cfg}")
    if transform_batch_disabled:
        print("transform_batch_disabled: true")

    # Save stdout capture to file for artifact upload
    sys.stdout = tee.original
    stdout_path = "/workspace/logs/stdout.log"
    with open(stdout_path, "w") as f:
        f.write(tee.getvalue())


if __name__ == "__main__":
    main()
