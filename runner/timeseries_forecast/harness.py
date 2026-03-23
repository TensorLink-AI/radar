"""Frozen training harness for trusted execution.

Usage: python harness.py /workspace/submission.py

Loads a miner's submission module, validates it, runs the training loop,
saves a checkpoint, and prints final metrics. The harness owns the training
loop entirely -- miners only provide architecture, optimizer, and optional
config/loss/scheduler.
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
                        lo, hi = _CLAMPS[k]
                        cfg[k] = type(_DEFAULTS[k])(max(lo, min(hi, v)))
        except Exception as e:
            print(f"WARNING: training_config() failed: {e}", file=sys.stderr)
    return cfg


# ── Stdout capture ────────────────────────────────────────────────

class TeeWriter:
    """Write to both original stdout and a StringIO buffer."""

    def __init__(self, original):
        self.original = original
        self.buffer = io.StringIO()

    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)

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

    # Build model
    try:
        model = sub.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).cuda()
    except Exception as e:
        print(f"ERROR: build_model() failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Count params
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model_params_M: {num_params / 1e6:.1f}")

    # Measure FLOPs-equivalent on CPU — must match Phase C validator measurement.
    # GPU wallclock calibration inflates small models due to kernel overhead.
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

    # ── Training loop ─────────────────────────────────────────────
    model.train()
    step = 0
    running_loss = 0.0
    loss_curve: list[float] = []
    start = time.time()

    for batch in get_dataloader(batch_size=batch_size):
        if time.time() - start > TIME_BUDGET:
            break

        context = batch["context"].cuda()   # (B, context_len, num_variates)
        targets = batch["target"].cuda()     # (B, prediction_len, num_variates)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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

    # ── Save checkpoint with safetensors (no final eval — validators do Phase C) ──
    save_file(model.state_dict(), "/workspace/checkpoints/model.safetensors")
    elapsed = time.time() - start

    print(f"training_seconds: {elapsed:.1f}")
    print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6:.1f}")
    print(f"num_steps: {step}")
    print(f"num_params_M: {num_params / 1e6:.1f}")
    print(f"flops_equivalent_size: {flops_equiv}")
    print(f"loss_curve: {loss_curve}")

    # Save stdout capture to file for artifact upload
    sys.stdout = tee.original
    stdout_path = "/workspace/logs/stdout.log"
    with open(stdout_path, "w") as f:
        f.write(tee.getvalue())


if __name__ == "__main__":
    main()
