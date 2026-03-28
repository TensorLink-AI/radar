"""Frozen training harness for graph complexity task.

Usage: python harness.py /workspace/submission.py

Loads a miner's submission, validates it, runs the training loop with
cross-entropy loss, saves a checkpoint. Supports both direct and
teacher-forced prediction modes.
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
from prepare import (
    get_dataloader, validate, CONTEXT_LEN, PREDICTION_LEN,
    VOCAB_SIZE, MODALITY, PREDICTION_MODE, MARGINAL_ENTROPY,
)

TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 300))
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs("/workspace/checkpoints", exist_ok=True)
os.makedirs("/workspace/logs", exist_ok=True)

_AMP_DTYPE_WHITELIST = {
    "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
}


# ── Submission loader ────────────────────────────────────────────

def _load_submission(path: str):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Config clamping ──────────────────────────────────────────────

_DEFAULTS = {"batch_size": 64, "grad_accum_steps": 1, "grad_clip": 1.0, "eval_interval": 200}
_CLAMPS = {
    "batch_size": (1, 512), "grad_accum_steps": (1, 16),
    "grad_clip": (0.0, 100.0), "eval_interval": (50, 10000),
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
                            nv = float(v)
                        except (TypeError, ValueError):
                            continue
                        lo, hi = _CLAMPS[k]
                        cfg[k] = type(_DEFAULTS[k])(max(lo, min(hi, nv)))
        except Exception as e:
            print(f"WARNING: training_config() failed: {e}", file=sys.stderr)
    return cfg


def _read_amp_config(sub) -> dict:
    default = {"enabled": True, "dtype": "bfloat16"}
    if not (hasattr(sub, "configure_amp") and callable(sub.configure_amp)):
        return default
    try:
        amp_cfg = sub.configure_amp()
        if not isinstance(amp_cfg, dict):
            return default
        enabled = amp_cfg.get("enabled", True)
        dtype_str = amp_cfg.get("dtype", "bfloat16")
        if dtype_str not in _AMP_DTYPE_WHITELIST:
            dtype_str = "bfloat16"
        return {"enabled": bool(enabled), "dtype": dtype_str}
    except Exception:
        return default


# ── Stdout capture ───────────────────────────────────────────────

class TeeWriter:
    _MAX_BUFFER = 10 * 1024 * 1024

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


# ── Main ─────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python harness.py <submission.py>", file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1]
    tee = TeeWriter(sys.stdout)
    sys.stdout = tee

    try:
        sub = _load_submission(target)
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: failed to load submission: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(sub, "build_model") or not callable(sub.build_model):
        print("ERROR: submission missing callable build_model()", file=sys.stderr)
        sys.exit(1)
    if not hasattr(sub, "build_optimizer") or not callable(sub.build_optimizer):
        print("ERROR: submission missing callable build_optimizer()", file=sys.stderr)
        sys.exit(1)

    cfg = _read_config(sub)
    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum_steps"]
    grad_clip = cfg["grad_clip"]
    eval_interval = cfg["eval_interval"]

    amp_cfg = _read_amp_config(sub)
    amp_enabled = amp_cfg["enabled"]
    amp_dtype = _AMP_DTYPE_WHITELIST[amp_cfg["dtype"]]

    # Build model — pass prediction_mode so arch can adapt
    try:
        model = sub.build_model(
            CONTEXT_LEN, PREDICTION_LEN, VOCAB_SIZE, PREDICTION_MODE,
        ).cuda()
    except Exception as e:
        print(f"ERROR: build_model() failed: {e}", file=sys.stderr)
        sys.exit(1)

    num_params = sum(p.numel() for p in model.parameters())

    # init_weights hook
    if hasattr(sub, "init_weights") and callable(sub.init_weights):
        try:
            sub.init_weights(model)
            if sum(p.numel() for p in model.parameters()) != num_params:
                print("ERROR: init_weights() changed param count", file=sys.stderr)
                sys.exit(1)
        except SystemExit:
            raise
        except Exception as e:
            print(f"WARNING: init_weights() failed: {e}", file=sys.stderr)

    print(f"model_params_M: {num_params / 1e6:.1f}")

    # Measure FLOPs
    try:
        from flops import compute_flops_equivalent
        cpu_model = model.cpu()
        flops_equiv = compute_flops_equivalent(cpu_model, CONTEXT_LEN, 1, "cpu")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"flops_equivalent_size: {flops_equiv}")
    except Exception as e:
        print(f"WARNING: FLOPs measurement failed: {e}", file=sys.stderr)
        flops_equiv = 0

    # torch.compile
    if getattr(sub, "COMPILE", False):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"WARNING: torch.compile failed: {e}", file=sys.stderr)

    # Build optimizer
    try:
        optimizer = sub.build_optimizer(model)
    except Exception as e:
        print(f"ERROR: build_optimizer() failed: {e}", file=sys.stderr)
        sys.exit(1)

    total_steps = (TIME_BUDGET * batch_size) // 2
    scheduler = None
    if hasattr(sub, "build_scheduler") and callable(sub.build_scheduler):
        try:
            scheduler = sub.build_scheduler(optimizer, total_steps)
        except Exception as e:
            print(f"WARNING: build_scheduler() failed: {e}", file=sys.stderr)

    has_transform_batch = hasattr(sub, "transform_batch") and callable(sub.transform_batch)
    has_on_step_end = hasattr(sub, "on_step_end") and callable(sub.on_step_end)
    transform_batch_disabled = False
    transform_batch_slow_count = 0
    transform_batch_fail_count = 0

    # ── Training loop ────────────────────────────────────────────
    model.train()
    step = 0
    optim_step = 0
    running_loss = 0.0
    loss_curve: list[float] = []
    start = time.time()

    for batch in get_dataloader(batch_size=batch_size):
        if time.time() - start > TIME_BUDGET:
            break

        x_tokens = batch["x_tokens"].cuda()  # (B, context_len)
        y_tokens = batch["y_tokens"].cuda()  # (B, prediction_len)

        # transform_batch hook
        if has_transform_batch and not transform_batch_disabled:
            try:
                t0 = time.perf_counter()
                batch_out = sub.transform_batch(
                    {"x_tokens": x_tokens, "y_tokens": y_tokens}, step, total_steps,
                )
                elapsed_tb = time.perf_counter() - t0
                if elapsed_tb > 0.05:
                    transform_batch_slow_count += 1
                if transform_batch_slow_count >= 3:
                    transform_batch_disabled = True
                else:
                    transform_batch_slow_count = 0
                x_tokens = batch_out["x_tokens"]
                y_tokens = batch_out["y_tokens"]
            except Exception:
                transform_batch_fail_count += 1
                if transform_batch_fail_count >= 5:
                    transform_batch_disabled = True

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            if PREDICTION_MODE == "teacher_forced":
                # Input: cat([context, targets[:-1]])
                inp = torch.cat([x_tokens, y_tokens[:, :-1]], dim=1)
                logits = model(inp)  # (B, ctx+pred-1, vocab)
                logits_pred = logits[:, -PREDICTION_LEN:]
            else:
                # Direct mode: model produces all predictions from context
                logits_pred = model(x_tokens)  # (B, pred_len, vocab)

            loss = F.cross_entropy(
                logits_pred.reshape(-1, VOCAB_SIZE),
                y_tokens.reshape(-1),
            )
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
            model.eval()
            val = validate(model)
            model.train()
            avg = running_loss / eval_interval
            loss_curve.append(avg)
            running_loss = 0.0
            elapsed = time.time() - start
            h_m = MARGINAL_ENTROPY if MARGINAL_ENTROPY > 0 else 1.0
            print(
                f"step {step} | loss: {avg:.4f}"
                f" | normalised_ce: {val['normalised_ce']:.4f}"
                f" | raw_ce: {val['raw_ce']:.4f}"
                f" | time: {elapsed:.0f}s"
            )

    # Flush trailing gradients
    if step % grad_accum != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optim_step += 1

    # Save checkpoint
    save_file(model.state_dict(), "/workspace/checkpoints/model.safetensors")
    elapsed = time.time() - start

    # Final eval
    model.eval()
    val = validate(model)

    print(f"normalised_ce: {val['normalised_ce']:.6f}")
    print(f"universal_ce: {val['universal_ce']:.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6:.1f}")
    print(f"num_steps: {step}")
    print(f"num_optim_steps: {optim_step}")
    print(f"num_params_M: {num_params / 1e6:.1f}")
    print(f"flops_equivalent_size: {flops_equiv}")
    print(f"loss_curve: {loss_curve}")

    sys.stdout = tee.original
    with open("/workspace/logs/stdout.log", "w") as f:
        f.write(tee.getvalue())


if __name__ == "__main__":
    main()
