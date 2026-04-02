"""Generic training harness — the loop that every task shares.

Each task implements a TaskRunner with 4 methods. The harness handles
everything generic: submission loading, seeding, FLOPs measurement,
size gating, optimizer stepping, grad accumulation, AMP, hooks,
checkpointing. Tasks only define what's unique to them.

To add a new task:
  1. Implement TaskRunner (build_model, get_dataloader, default_loss, measure_flops)
  2. Register in runner/server.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

logger = logging.getLogger(__name__)


class TaskRunner(Protocol):
    """What each task must provide. Everything else is generic."""

    def build_model(self, sub: Any, device: str) -> Any:
        """Build model from miner's submission module. Return model on device."""
        ...

    def get_dataloader(self, batch_size: int) -> Iterator:
        """Return an iterator of batches. Each batch is a dict with 'input' and 'target' tensors."""
        ...

    def default_loss(self, predictions: Any, targets: Any) -> Any:
        """Default loss function. Miner can override via compute_loss()."""
        ...

    def measure_flops(self, model: Any, device: str) -> int:
        """Measure FLOPs-equivalent of the model. Return 0 on failure."""
        ...

    def wrap_loss(self, sub_loss_fn: Callable) -> Callable:
        """Wrap a miner's compute_loss to match the generic (preds, targets) signature.

        Override if your task's old harness passed extra args (e.g. quantiles).
        Default: return as-is (assumes 2-arg signature).
        """
        return sub_loss_fn


@dataclass
class TrainingConfig:
    """Generic training config extracted from dispatch payload."""
    seed: int = 42
    round_id: int = 0
    min_flops: int = 0
    max_flops: int = 0
    miner_hotkey: str = "unknown"
    time_budget: int = 300

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def run_training(runner: TaskRunner, architecture_code: str, config: TrainingConfig) -> dict:
    """Generic training pipeline. Task-specific logic comes from the runner.

    1. Seed everything
    2. Load submission module
    3. Build model (task-specific)
    4. Measure FLOPs (task-specific)
    5. Size gate
    6. Run training loop (generic, with task-specific data/loss)
    7. Save checkpoint

    Returns result dict with status, flops, checkpoint_path, etc.
    """
    import random
    import torch

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["SEED"] = str(config.seed)
    os.environ["TIME_BUDGET"] = str(config.time_budget)
    os.environ.setdefault("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")

    # 1. Load submission
    sub = _load_submission(architecture_code)
    if isinstance(sub, dict):
        return sub  # error dict

    if not _has_callable(sub, "build_model"):
        return _fail(config, "build_failed", "Missing build_model()")
    if not _has_callable(sub, "build_optimizer"):
        return _fail(config, "build_failed", "Missing build_optimizer()")

    # 2. Build model (task-specific)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = runner.build_model(sub, device)
    except Exception as e:
        return _fail(config, "build_failed", str(e))

    # 3. Measure FLOPs
    flops_equiv = runner.measure_flops(model, device)

    # 4. Size gate
    gate_result = _check_size_gate(config, flops_equiv)
    if gate_result:
        return gate_result

    # 5. torch.compile (opt-in)
    if getattr(sub, "COMPILE", False):
        try:
            import torch as _torch
            model = _torch.compile(model)
        except Exception as e:
            logger.warning("torch.compile failed, using eager mode: %s", e)

    # 6. Train
    start = time.time()
    num_params = sum(p.numel() for p in model.parameters())

    # init_weights hook
    if _has_callable(sub, "init_weights"):
        try:
            sub.init_weights(model)
            if sum(p.numel() for p in model.parameters()) != num_params:
                return _fail(config, "build_failed", "init_weights() changed param count")
        except Exception as e:
            logger.warning("init_weights() failed: %s", e)

    try:
        step = _training_loop(runner, sub, model, device, config.time_budget, start)
    except Exception as e:
        return _fail(
            config, "failed", str(e),
            flops_equivalent_size=flops_equiv,
            training_time_seconds=time.time() - start,
        )

    # 6. Save checkpoint
    checkpoint_path = "/workspace/checkpoints/model.safetensors"
    os.makedirs("/workspace/checkpoints", exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), checkpoint_path)

    return {
        "round_id": config.round_id,
        "miner_hotkey": config.miner_hotkey,
        "status": "success",
        "flops_equivalent_size": flops_equiv,
        "training_time_seconds": time.time() - start,
        "num_steps": step,
        "num_params_M": num_params / 1e6,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0,
        "checkpoint_path": checkpoint_path,
    }


# ── Generic training loop ────────────────────────────────────────────

# Config clamping (same for all tasks)
_DEFAULTS = {"batch_size": 64, "grad_accum_steps": 1, "grad_clip": 1.0}
_CLAMPS = {"batch_size": (1, 512), "grad_accum_steps": (1, 16), "grad_clip": (0.0, 100.0)}

_AMP_DTYPE_WHITELIST = {"bfloat16": None, "float16": None, "float32": None}  # populated on first use


def _get_amp_dtypes():
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _read_config(sub) -> dict:
    cfg = dict(_DEFAULTS)
    if _has_callable(sub, "training_config"):
        try:
            user_cfg = sub.training_config()
            if isinstance(user_cfg, dict):
                for k, v in user_cfg.items():
                    if k in _CLAMPS:
                        try:
                            lo, hi = _CLAMPS[k]
                            cfg[k] = type(_DEFAULTS[k])(max(lo, min(hi, float(v))))
                        except (TypeError, ValueError):
                            pass
        except Exception:
            pass
    return cfg


def _read_amp_config(sub) -> dict:
    default = {"enabled": True, "dtype": "bfloat16"}
    if not _has_callable(sub, "configure_amp"):
        return default
    try:
        amp_cfg = sub.configure_amp()
        if not isinstance(amp_cfg, dict):
            return default
        dtype_str = amp_cfg.get("dtype", "bfloat16")
        if dtype_str not in _get_amp_dtypes():
            dtype_str = "bfloat16"
        return {"enabled": bool(amp_cfg.get("enabled", True)), "dtype": dtype_str}
    except Exception:
        return default


def _training_loop(runner: TaskRunner, sub, model, device: str, time_budget: int, start: float) -> int:
    """Generic training loop. Returns step count."""
    import torch

    cfg = _read_config(sub)
    optimizer = sub.build_optimizer(model)
    amp_dtypes = _get_amp_dtypes()

    amp_cfg = _read_amp_config(sub)
    amp_enabled = amp_cfg["enabled"] and (device == "cuda")
    amp_dtype = amp_dtypes[amp_cfg["dtype"]]

    total_steps_est = (time_budget * cfg["batch_size"]) // 2
    scheduler = None
    if _has_callable(sub, "build_scheduler"):
        try:
            scheduler = sub.build_scheduler(optimizer, total_steps_est)
        except Exception:
            pass

    # Loss: miner's compute_loss() (wrapped for task compat) or task's default
    loss_fn = runner.default_loss
    if _has_callable(sub, "compute_loss"):
        loss_fn = runner.wrap_loss(sub.compute_loss)

    has_transform = _has_callable(sub, "transform_batch")
    has_on_step = _has_callable(sub, "on_step_end")
    tb_disabled = False
    tb_slow = 0
    tb_fail = 0

    model.train()
    step = 0
    optim_step = 0
    for batch in runner.get_dataloader(batch_size=cfg["batch_size"]):
        if time.time() - start > time_budget:
            break

        inputs = batch["input"].to(device) if "input" in batch else batch["context"].to(device)
        targets = batch["target"].to(device)

        # transform_batch hook
        if has_transform and not tb_disabled:
            try:
                t0 = time.perf_counter()
                tb_result = sub.transform_batch({"input": inputs, "target": targets}, step, total_steps_est)
                elapsed = time.perf_counter() - t0
                if elapsed > 0.05:
                    tb_slow += 1
                if tb_slow >= 3:
                    tb_disabled = True
                elif isinstance(tb_result, dict) and "target" in tb_result:
                    inputs = tb_result.get("input", tb_result.get("context", inputs))
                    targets = tb_result["target"]
                    tb_fail = 0
                    if elapsed <= 0.05:
                        tb_slow = 0
                else:
                    tb_fail += 1
                    if tb_fail >= 5:
                        tb_disabled = True
            except Exception:
                tb_fail += 1
                if tb_fail >= 5:
                    tb_disabled = True

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(inputs)
            loss = loss_fn(predictions, targets) / cfg["grad_accum_steps"]

        loss.backward()
        if (step + 1) % cfg["grad_accum_steps"] == 0:
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
            optim_step += 1

            if has_on_step:
                try:
                    sub.on_step_end(
                        model=model, optimizer=optimizer,
                        step=optim_step, total_steps=total_steps_est,
                        loss_value=loss.item() * cfg["grad_accum_steps"],
                    )
                except Exception as e:
                    logger.warning("on_step_end() failed: %s", e)
        step += 1

    # Flush trailing gradient accumulation
    if step > 0 and step % cfg["grad_accum_steps"] != 0:
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optim_step += 1

    return step


# ── Helpers ──────────────────────────────────────────────────────────

def _load_submission(architecture_code: str):
    """Write and import the miner's submission. Returns module or error dict."""
    submission_path = "/workspace/submission.py"
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    with open(submission_path, "w") as f:
        f.write(architecture_code)
    try:
        spec = importlib.util.spec_from_file_location("submission", submission_path)
        sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sub)
        return sub
    except Exception as e:
        return {"status": "build_failed", "error": str(e)}


def _has_callable(obj, name: str) -> bool:
    return hasattr(obj, name) and callable(getattr(obj, name))


def _check_size_gate(config: TrainingConfig, flops_equiv: int) -> dict | None:
    """Returns error dict if outside gate, None if OK."""
    tol = float(os.environ.get("RADAR_SIZE_GATE_TOLERANCE", "0.10"))
    if config.min_flops > 0 and config.max_flops > 0:
        eff_min = int(config.min_flops * (1 - tol))
        eff_max = int(config.max_flops * (1 + tol))
        if not (eff_min <= flops_equiv <= eff_max):
            return _fail(
                config, "size_violation",
                f"FLOPs {flops_equiv} outside [{config.min_flops}, {config.max_flops}] "
                f"(effective [{eff_min}, {eff_max}] with {tol:.0%} tolerance)",
                flops_equivalent_size=flops_equiv,
            )
    return None


def _fail(config: TrainingConfig, status: str, error: str, **kw) -> dict:
    return {
        "round_id": config.round_id,
        "miner_hotkey": config.miner_hotkey,
        "status": status,
        "error": error,
        **kw,
    }
