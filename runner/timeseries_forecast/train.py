"""Time-series forecasting runner — task-specific training logic.

Called by the generalist runner/server.py when task_name is
"ts_forecasting" or "ml_training". Contains model building,
FLOPs measurement, size gating, and the training loop.

All task-specific imports (prepare, harness, flops) live here.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time

logger = logging.getLogger(__name__)


def run_training(architecture_code: str, config: dict) -> dict:
    """Run time-series forecasting training.

    Args:
        architecture_code: Miner's submission.py contents
        config: Dict with seed, round_id, min_flops, max_flops,
                miner_hotkey, time_budget

    Returns:
        Dict with status, flops_equivalent_size, training_time_seconds,
        num_steps, checkpoint_path, etc.
    """
    import random
    import torch

    seed = config["seed"]
    round_id = config["round_id"]
    min_flops = config["min_flops"]
    max_flops = config["max_flops"]
    miner_hotkey = config["miner_hotkey"]
    time_budget = config["time_budget"]

    os.environ["SEED"] = str(seed)
    os.environ["TIME_BUDGET"] = str(time_budget)
    os.environ.setdefault("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Load submission
    submission_path = "/workspace/submission.py"
    with open(submission_path, "w") as f:
        f.write(architecture_code)

    try:
        spec = importlib.util.spec_from_file_location("submission", submission_path)
        sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sub)
    except Exception as e:
        return _fail(round_id, miner_hotkey, "build_failed", str(e))

    if not hasattr(sub, "build_model") or not callable(sub.build_model):
        return _fail(round_id, miner_hotkey, "build_failed", "Missing build_model()")
    if not hasattr(sub, "build_optimizer") or not callable(sub.build_optimizer):
        return _fail(round_id, miner_hotkey, "build_failed", "Missing build_optimizer()")

    # 2. Build model with task-specific constants
    from prepare import CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = sub.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).to(device)
    except Exception as e:
        return _fail(round_id, miner_hotkey, "build_failed", str(e))

    # 3. Measure FLOPs on CPU
    try:
        from flops import compute_flops_equivalent
        cpu_model = model.to("cpu") if device != "cpu" else model
        flops_equiv = compute_flops_equivalent(cpu_model, CONTEXT_LEN, NUM_VARIATES, "cpu")
        if device != "cpu":
            model.to(device)
    except Exception as e:
        logger.warning("FLOPs measurement failed: %s", e)
        flops_equiv = 0

    # 4. Size gate
    tol = float(os.environ.get("RADAR_SIZE_GATE_TOLERANCE", "0.10"))
    if min_flops > 0 and max_flops > 0:
        eff_min = int(min_flops * (1 - tol))
        eff_max = int(max_flops * (1 + tol))
        if not (eff_min <= flops_equiv <= eff_max):
            return _fail(
                round_id, miner_hotkey, "size_violation",
                f"FLOPs {flops_equiv} outside [{min_flops}, {max_flops}] "
                f"(effective [{eff_min}, {eff_max}] with {tol:.0%} tolerance)",
                flops_equivalent_size=flops_equiv,
            )

    # 5. Train
    start = time.time()
    num_params = sum(p.numel() for p in model.parameters())

    # init_weights hook
    if hasattr(sub, "init_weights") and callable(sub.init_weights):
        try:
            sub.init_weights(model)
            if sum(p.numel() for p in model.parameters()) != num_params:
                return _fail(round_id, miner_hotkey, "build_failed", "init_weights() changed param count")
        except Exception as e:
            logger.warning("init_weights() failed: %s", e)

    try:
        step = _run_training_loop(sub, model, device, time_budget, start, QUANTILES)
    except Exception as e:
        return _fail(
            round_id, miner_hotkey, "failed", str(e),
            flops_equivalent_size=flops_equiv,
            training_time_seconds=time.time() - start,
        )

    # 6. Save checkpoint
    checkpoint_path = "/workspace/checkpoints/model.safetensors"
    os.makedirs("/workspace/checkpoints", exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), checkpoint_path)

    training_time = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    return {
        "round_id": round_id,
        "miner_hotkey": miner_hotkey,
        "status": "success",
        "flops_equivalent_size": flops_equiv,
        "training_time_seconds": training_time,
        "num_steps": step,
        "num_params_M": num_params / 1e6,
        "peak_vram_mb": peak_vram,
        "checkpoint_path": checkpoint_path,
    }


def _run_training_loop(sub, model, device: str, time_budget: int, start: float, quantiles) -> int:
    """Execute the training loop. Returns step count."""
    import torch
    from prepare import get_dataloader
    from harness import _read_config, _read_amp_config, _validate_batch, default_loss, _AMP_DTYPE_WHITELIST

    cfg = _read_config(sub)
    optimizer = sub.build_optimizer(model)

    amp_cfg = _read_amp_config(sub)
    amp_enabled = amp_cfg["enabled"] and (device == "cuda")
    amp_dtype = _AMP_DTYPE_WHITELIST[amp_cfg["dtype"]]

    total_steps_est = (time_budget * cfg["batch_size"]) // 2
    scheduler = None
    if hasattr(sub, "build_scheduler") and callable(sub.build_scheduler):
        try:
            scheduler = sub.build_scheduler(optimizer, total_steps_est)
        except Exception:
            pass

    loss_fn = default_loss
    if hasattr(sub, "compute_loss") and callable(sub.compute_loss):
        loss_fn = sub.compute_loss

    has_transform_batch = hasattr(sub, "transform_batch") and callable(sub.transform_batch)
    has_on_step_end = hasattr(sub, "on_step_end") and callable(sub.on_step_end)
    tb_disabled = False
    tb_slow = 0
    tb_fail = 0

    model.train()
    step = 0
    optim_step = 0
    for batch in get_dataloader(batch_size=cfg["batch_size"]):
        if time.time() - start > time_budget:
            break
        context = batch["context"].to(device)
        targets = batch["target"].to(device)

        if has_transform_batch and not tb_disabled:
            try:
                t0 = time.perf_counter()
                tb_result = sub.transform_batch({"context": context, "target": targets}, step, total_steps_est)
                elapsed = time.perf_counter() - t0
                if elapsed > 0.05:
                    tb_slow += 1
                if tb_slow >= 3:
                    tb_disabled = True
                elif _validate_batch(tb_result, cfg["batch_size"], "transform_batch"):
                    context = tb_result["context"]
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
            predictions = model(context)
            loss = loss_fn(predictions, targets, quantiles) / cfg["grad_accum_steps"]

        loss.backward()
        if (step + 1) % cfg["grad_accum_steps"] == 0:
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
            optim_step += 1

            if has_on_step_end:
                try:
                    sub.on_step_end(
                        model=model, optimizer=optimizer,
                        step=optim_step, total_steps=total_steps_est,
                        loss_value=loss.item() * cfg["grad_accum_steps"],
                    )
                except Exception as e:
                    logger.warning("on_step_end() failed: %s", e)
        step += 1

    return step


# ── Eval template for Phase C ────────────────────────────────────────

EVAL_TEMPLATE = '''
import json
import os
import random
import sys

import torch
from safetensors.torch import load_file

from prepare import validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES
from flops import compute_flops_equivalent

random.seed({eval_split_seed})
torch.manual_seed({eval_split_seed})
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

arch_path = "{arch_path}"
checkpoint_path = "{checkpoint_path}"
device = "{device}"

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission", arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "build_model") or not callable(mod.build_model):
        print(json.dumps({{"crps": float("inf"), "mase": float("inf"), "error": "Missing build_model()"}}))
        sys.exit(0)

    model = mod.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).to(device)
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict)

    flops_equiv = 0
    try:
        flops_equiv = compute_flops_equivalent(model, CONTEXT_LEN, NUM_VARIATES, device)
    except Exception:
        pass

    param_count = sum(p.numel() for p in model.parameters())
    if hasattr(model, "reset"):
        model.reset()
    model.eval()

    data_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "")
    metrics = validate(model, seed={eval_split_seed},
                       data_dir=data_dir if data_dir else None)

    result = {{
        "crps": metrics["crps"],
        "ncrps": metrics.get("ncrps", float("inf")),
        "mase": metrics["mase"],
        "flops_equivalent_size": flops_equiv,
        "param_count": param_count,
    }}
    if "n_datasets" in metrics:
        result["n_datasets"] = metrics["n_datasets"]
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"crps": float("inf"), "mase": float("inf"), "error": str(e)}}))
'''


def _fail(round_id: int, miner_hotkey: str, status: str, error: str, **kw) -> dict:
    return {"round_id": round_id, "miner_hotkey": miner_hotkey, "status": status, "error": error, **kw}
