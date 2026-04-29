"""Generic training harness — the loop that every task shares.

Each task implements a TaskRunner with 4 methods. The harness handles
everything generic: submission loading, seeding, FLOPs measurement,
size gating, optimizer stepping, grad accumulation, AMP, hooks,
checkpointing. Tasks only define what's unique to them.

To add a new task:
  1. Implement TaskRunner (build_model, get_dataloader, default_loss, measure_flops)
  2. Register in runner/server.py

Submission config (training_config()):
  - batch_size, grad_accum_steps, grad_clip (existing)
  - log_every_n_steps: train loss sample frequency (default 10)
  - val_schedule: "logarithmic" | "fixed" | "none" (default "logarithmic")
  - val_base_step: first val step / fixed interval (default 10)
  - val_growth: log-schedule multiplier (default 2.0)
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from dataclasses import dataclass
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

    def get_val_dataloader(self, batch_size: int) -> Iterator | None:
        """Return a finite, repeatable iterable of val batches (same batches every call).

        Return None if val is unavailable (e.g. no val shard URLs provided) —
        harness will skip val and fall back to last-step checkpoint behaviour.

        Must be finite — harness iterates the full sequence per val check.
        Must be deterministic — same batches in same order on every call.
        """
        return None


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
        return _fail(config, sub.get("status", "build_failed"), sub.get("error", "unknown"))

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
        loop_result = _training_loop(runner, sub, model, device, config.time_budget, start)
    except Exception as e:
        return _fail(
            config, "failed", str(e),
            flops_equivalent_size=flops_equiv,
            training_time_seconds=time.time() - start,
        )

    step = loop_result["step"]
    best_state = loop_result["best_state"]

    # 7. Save checkpoint — prefer best-val state if we tracked one, else current.
    state_to_save = best_state if best_state is not None else model.state_dict()
    checkpoint_path = "/workspace/checkpoints/model.safetensors"
    os.makedirs("/workspace/checkpoints", exist_ok=True)
    from safetensors.torch import save_file
    save_file(state_to_save, checkpoint_path)

    loop_cfg = loop_result.get("cfg", {})
    flops_per_step = float(loop_result.get("flops_per_optim_step", 0) or 0)
    cadence_unit = loop_cfg.get("val_cadence_unit", "step")
    # When FLOPs estimation succeeded we report cadence as flops; otherwise we
    # actually fell back to step cadence regardless of cfg.
    if cadence_unit == "flops" and flops_per_step <= 0:
        cadence_unit = "step"
    val_base = (
        float(loop_cfg.get("val_base_flops", 0.0))
        if cadence_unit == "flops"
        else float(loop_cfg.get("val_base_step", 0))
    )

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
        "train_loss_history": loop_result["train_history"],
        "val_loss_history": loop_result["val_history"],
        "best_val_loss": loop_result["best_val_loss"],
        "best_val_step": loop_result["best_val_step"],
        "val_cadence_unit": cadence_unit,
        "val_base": val_base,
        "val_growth": float(loop_cfg.get("val_growth", 0.0)),
        "val_eval_tokens": 0,
        "flops_per_step_estimate": flops_per_step,
        "reference_eval_loss_history": [],
    }


# ── Generic training loop ────────────────────────────────────────────

# Config clamping (same for all tasks)
_DEFAULTS = {
    "batch_size": 64,
    "grad_accum_steps": 1,
    "grad_clip": 1.0,
    "log_every_n_steps": 10,
    "val_base_step": 10,
    "val_growth": 2.0,
    "val_base_flops": 1e15,
}
_CLAMPS = {
    "batch_size": (1, 512),
    "grad_accum_steps": (1, 16),
    "grad_clip": (0.0, 100.0),
    "log_every_n_steps": (1, 1000),
    "val_base_step": (1, 10000),
    "val_growth": (1.1, 10.0),
    "val_base_flops": (1e12, 1e22),
}
_VAL_SCHEDULES = ("logarithmic", "fixed", "none")
_DEFAULT_VAL_SCHEDULE = "logarithmic"
_VAL_CADENCE_UNITS = ("flops", "step")
_DEFAULT_VAL_CADENCE_UNIT = "flops"

_AMP_DTYPE_WHITELIST = {"bfloat16": None, "float16": None, "float32": None}  # populated on first use


def _get_amp_dtypes():
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _read_config(sub) -> dict:
    cfg = dict(_DEFAULTS)
    cfg["val_schedule"] = _DEFAULT_VAL_SCHEDULE
    cfg["val_cadence_unit"] = _DEFAULT_VAL_CADENCE_UNIT
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
                    elif k == "val_schedule":
                        if isinstance(v, str) and v in _VAL_SCHEDULES:
                            cfg["val_schedule"] = v
                    elif k == "val_cadence_unit":
                        if isinstance(v, str) and v in _VAL_CADENCE_UNITS:
                            cfg["val_cadence_unit"] = v
        except Exception:
            pass
    return cfg


def _next_val_step(current_step: int, base: int, growth: float) -> int:
    """Next optim step at which to run val, given logarithmic schedule."""
    if current_step < base:
        return base
    step = base
    while step <= current_step:
        step = max(step + 1, int(step * growth))
    return step


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


def _training_loop(runner: TaskRunner, sub, model, device: str, time_budget: int, start: float) -> dict:
    """Generic training loop.

    Returns {step, train_history, val_history, best_val_loss, best_val_step,
    best_state, flops_per_optim_step, cumulative_flops}.
    `best_state` is a CPU-cloned state_dict from the lowest-val-loss step,
    or None if val never ran (caller falls back to model.state_dict()).
    """
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

    # Val support: probe whether the runner provides a val dataloader.
    val_loader_factory = None
    try:
        probe = runner.get_val_dataloader(batch_size=cfg["batch_size"])
        if probe is not None:
            val_loader_factory = lambda: runner.get_val_dataloader(batch_size=cfg["batch_size"])
    except Exception as e:
        logger.warning("get_val_dataloader failed, skipping val: %s", e)

    num_params = sum(p.numel() for p in model.parameters())

    best_val_loss = float("inf")
    best_val_step = -1
    best_state = None
    val_history: list[dict] = []
    train_history: list[dict] = []

    # FLOPs accounting. We compute flops_per_optim_step lazily on the first
    # batch (need seq_len), so the FLOPs cadence schedule is initialized
    # after that first batch is observed.
    flops_per_optim_step = 0
    cumulative_flops = 0
    val_use_flops = (
        val_loader_factory is not None
        and cfg["val_schedule"] != "none"
        and cfg["val_cadence_unit"] == "flops"
    )
    next_val_flops = float(cfg["val_base_flops"]) if val_use_flops else -1.0
    next_val = (
        cfg["val_base_step"]
        if val_loader_factory and cfg["val_schedule"] != "none"
        else -1
    )

    model.train()
    step = 0
    optim_step = 0
    nan_streak = 0
    _MAX_NAN_STREAK = 50
    for batch in runner.get_dataloader(batch_size=cfg["batch_size"]):
        if time.time() - start > time_budget:
            break

        inputs = batch["input"].to(device) if "input" in batch else batch["context"].to(device)
        targets = batch["target"].to(device)

        # Compute flops_per_optim_step once, on the first batch we see, so
        # we can key val cadence to FLOPs across runs of different sizes /
        # batch shapes / seq lens.
        if step == 0:
            flops_per_optim_step = _estimate_flops_per_optim_step(
                runner, model, device, inputs,
                num_params=num_params,
                batch_size=int(inputs.shape[0]),
                grad_accum=cfg["grad_accum_steps"],
            )
            if val_use_flops and flops_per_optim_step <= 0:
                # FLOPs estimation failed — log and fall back to step cadence.
                logger.warning(
                    "flops_per_optim_step=0; falling back to step-based val cadence",
                )
                val_use_flops = False
                next_val_flops = -1.0

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

        # Skip backward when loss is non-finite (gradient explosion / bad loss fn)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            nan_streak += 1
            if nan_streak >= _MAX_NAN_STREAK:
                logger.warning("Stopping: %d consecutive non-finite losses (model diverged)", nan_streak)
                break
            step += 1
            continue
        nan_streak = 0

        loss.backward()
        if (step + 1) % cfg["grad_accum_steps"] == 0:
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
            optim_step += 1
            cumulative_flops += flops_per_optim_step

            # Train loss logging (periodic)
            if optim_step % cfg["log_every_n_steps"] == 0:
                entry = {
                    "step": optim_step,
                    "loss": float(loss.item() * cfg["grad_accum_steps"]),
                }
                if flops_per_optim_step > 0:
                    entry["flops"] = cumulative_flops
                train_history.append(entry)

            # Val loss + best checkpoint (scheduled). Runs BEFORE on_step_end so
            # the submission's hook cannot observe val state. val_loss is a local
            # here only — never propagated to any submission hook.
            #
            # FLOPs-keyed cadence (preferred): trigger when cumulative_flops
            # crosses next_val_flops, then advance multiplicatively by val_growth.
            # Step-keyed cadence (legacy / fallback): unchanged.
            triggered = False
            if val_use_flops and flops_per_optim_step > 0:
                if cumulative_flops >= next_val_flops:
                    triggered = True
            elif next_val > 0 and optim_step >= next_val:
                triggered = True

            if triggered:
                val_loss = _run_val(runner, model, val_loader_factory, device, amp_dtype, amp_enabled)
                if val_loss is not None:
                    entry = {"step": optim_step, "loss": val_loss}
                    if flops_per_optim_step > 0:
                        entry["flops"] = cumulative_flops
                    val_history.append(entry)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_step = optim_step
                        # Clone state_dict to CPU to avoid holding GPU memory.
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if val_use_flops and flops_per_optim_step > 0:
                    next_val_flops = max(
                        next_val_flops + 1,
                        int(next_val_flops * cfg["val_growth"]),
                    )
                elif cfg["val_schedule"] == "logarithmic":
                    next_val = _next_val_step(optim_step, cfg["val_base_step"], cfg["val_growth"])
                elif cfg["val_schedule"] == "fixed":
                    next_val = optim_step + cfg["val_base_step"]
                else:
                    next_val = -1

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
        cumulative_flops += flops_per_optim_step

    # Always run val at the final step too (if val is enabled and we did
    # at least one optimizer step), so the best_state captures end-of-run.
    if val_loader_factory and cfg["val_schedule"] != "none" and optim_step > 0:
        if not val_history or val_history[-1]["step"] != optim_step:
            val_loss = _run_val(runner, model, val_loader_factory, device, amp_dtype, amp_enabled)
            if val_loss is not None:
                entry = {"step": optim_step, "loss": val_loss}
                if flops_per_optim_step > 0:
                    entry["flops"] = cumulative_flops
                val_history.append(entry)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = optim_step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
        "step": step,
        "train_history": train_history,
        "val_history": val_history,
        "best_val_loss": (best_val_loss if best_val_step >= 0 else None),
        "best_val_step": best_val_step,
        "best_state": best_state,
        "flops_per_optim_step": flops_per_optim_step,
        "cumulative_flops": cumulative_flops,
        "cfg": cfg,
    }


def _estimate_flops_per_optim_step(
    runner: TaskRunner, model, device: str, inputs,
    *, num_params: int, batch_size: int, grad_accum: int,
) -> int:
    """FLOPs-per-optim-step estimate, used to key val cadence to FLOPs.

    Two paths:
      * If the input tensor has a clear sequence dimension (3D+: B, T, ...),
        use the standard transformer-style approximation
        ``6 * num_params * B * T * grad_accum``.
      * Otherwise, fall back to ``3 * runner.measure_flops(...) * B *
        grad_accum`` — measure_flops already accounts for one forward pass,
        and ``3x`` covers fwd+bwd as a generic non-transformer estimate.

    Returns 0 on failure; callers fall back to step-based cadence.
    """
    try:
        if hasattr(inputs, "shape") and len(inputs.shape) >= 3 and num_params > 0:
            seq_len = int(inputs.shape[1])
            if seq_len > 0:
                return int(6 * num_params * batch_size * seq_len * grad_accum)
        # Fallback: measure_flops gives a per-sample forward FLOP estimate.
        try:
            mf = int(runner.measure_flops(model, device))
        except Exception:
            mf = 0
        if mf > 0:
            return int(3 * mf * batch_size * grad_accum)
    except Exception as e:
        logger.warning("flops_per_optim_step estimation failed: %s", e)
    return 0


def _run_val(runner, model, val_loader_factory, device, amp_dtype, amp_enabled):
    """Compute mean val loss over the full val dataloader using task's default_loss.

    Returns None on any failure (so val failure doesn't kill training).
    """
    import torch
    try:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader_factory():
                inputs = batch["input"].to(device) if "input" in batch else batch["context"].to(device)
                targets = batch["target"].to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    predictions = model(inputs)
                    # ALWAYS use task default_loss — not the submission's compute_loss.
                    loss = runner.default_loss(predictions, targets)
                if torch.isfinite(loss):
                    losses.append(float(loss.item()))
        if not losses:
            return None
        return sum(losses) / len(losses)
    except Exception as e:
        logger.warning("val failed: %s", e)
        return None
    finally:
        model.train()


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
