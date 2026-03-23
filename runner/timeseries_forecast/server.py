"""FastAPI server wrapping the frozen training harness.

Runs inside the sanctioned Docker image. Miners deploy this unmodified
on Basilica. Validators dispatch training jobs via POST /train.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import time
import types

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="Radar Trainer")

# R2 client (initialized on first use, only for localnet fallback)
_r2 = None

# ── Metagraph cache ──────────────────────────────────────────────────
_metagraph_cache = None
_metagraph_lock = threading.Lock()
_metagraph_last_refresh: float = 0.0
METAGRAPH_REFRESH_INTERVAL = float(os.getenv("METAGRAPH_REFRESH_INTERVAL", "300"))

# ── Rate limiting ────────────────────────────────────────────────────
_train_semaphore = asyncio.Semaphore(1)  # max 1 concurrent /train
_hotkey_last_request: dict[str, float] = {}
_hotkey_lock = threading.Lock()
HOTKEY_COOLDOWN_SECONDS = float(os.getenv("TRAINER_HOTKEY_COOLDOWN", "60"))


def _get_r2():
    global _r2
    if _r2 is None:
        from shared.r2_audit import R2AuditLog
        _r2 = R2AuditLog()
    return _r2


@app.post("/train")
async def train(request: Request):
    """Execute a training job.

    1. Verify Epistula — sender must be staked validator
    2. Parse: architecture code, seed, training_config, round_id, flops range
    3. Write architecture to submission.py
    4. build_model() -> measure FLOPs-equivalent
    5. Size gate check
    6. Train (existing harness logic)
    7. Save checkpoint.safetensors
    8. Upload to R2
    9. Return training metadata (NOT eval metrics)
    """
    # 0. Concurrency gate — only 1 training job at a time
    if _train_semaphore.locked():
        return JSONResponse(status_code=429, content={"error": "Training job already in progress"})

    body = await request.body()

    # 1. Verify Epistula auth — FAIL CLOSED
    localnet = os.getenv("RADAR_LOCALNET", "").lower() in ("1", "true")
    if localnet:
        logger.info("Localnet mode: skipping auth")
        sender = "localnet"
    else:
        metagraph = _load_metagraph()
        if metagraph is None:
            logger.error("Metagraph unavailable — rejecting request (fail closed)")
            return JSONResponse(status_code=503, content={"error": "Auth unavailable, try again later"})
        try:
            from shared.auth import verify_request
            ok, err, sender = verify_request(dict(request.headers), body, metagraph, require_stake=True)
            if not ok:
                return JSONResponse(status_code=403, content={"error": err})
        except Exception as e:
            logger.error("Auth verification failed: %s", e)
            return JSONResponse(status_code=403, content={"error": "Auth verification error"})

    # 1b. Per-hotkey rate limit
    with _hotkey_lock:
        last = _hotkey_last_request.get(sender, 0.0)
        if time.time() - last < HOTKEY_COOLDOWN_SECONDS:
            remaining = HOTKEY_COOLDOWN_SECONDS - (time.time() - last)
            return JSONResponse(status_code=429, content={
                "error": f"Rate limited. Retry in {remaining:.0f}s",
            })
        _hotkey_last_request[sender] = time.time()

    # 2. Parse request
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    architecture_code = data.get("architecture", "")
    seed = data.get("seed", 42)
    round_id = data.get("round_id", 0)
    min_flops = data.get("min_flops_equivalent", 0)
    max_flops = data.get("max_flops_equivalent", 0)
    miner_hotkey = data.get("miner_hotkey", "unknown")
    time_budget = data.get("time_budget", 300)
    upload_urls = data.get("upload_urls", {})

    if not architecture_code:
        return JSONResponse(status_code=400, content={"error": "Missing architecture code"})

    # Acquire training semaphore for the GPU-intensive work
    async with _train_semaphore:
        return await _execute_training(
            architecture_code, seed, round_id, min_flops, max_flops,
            miner_hotkey, time_budget, upload_urls,
        )


async def _execute_training(
    architecture_code: str, seed: int, round_id: int,
    min_flops: int, max_flops: int, miner_hotkey: str, time_budget: int,
    upload_urls: dict[str, str] | None = None,
):
    """Run the actual training job (called under semaphore)."""

    # 3. Write architecture to submission.py
    submission_path = "/workspace/submission.py"
    with open(submission_path, "w") as f:
        f.write(architecture_code)

    os.environ["SEED"] = str(seed)
    os.environ["TIME_BUDGET"] = str(time_budget)

    # 4-6. Build model, measure FLOPs, size gate, train
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        spec = importlib.util.spec_from_file_location("submission", submission_path)
        sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sub)
    except Exception as e:
        return JSONResponse(content=_result(round_id, miner_hotkey, "build_failed", error=str(e)))

    if not hasattr(sub, "build_model") or not callable(sub.build_model):
        return JSONResponse(content=_result(round_id, miner_hotkey, "build_failed", error="Missing build_model()"))
    if not hasattr(sub, "build_optimizer") or not callable(sub.build_optimizer):
        return JSONResponse(content=_result(round_id, miner_hotkey, "build_failed", error="Missing build_optimizer()"))

    from prepare import CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = sub.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).to(device)
    except Exception as e:
        return JSONResponse(content=_result(round_id, miner_hotkey, "build_failed", error=str(e)))

    # Measure FLOPs on CPU — must match Phase C validator measurement.
    # GPU wallclock calibration inflates small models due to kernel launch
    # overhead, giving inconsistent results across hardware.
    try:
        from flops import compute_flops_equivalent
        cpu_model = model.to("cpu") if device != "cpu" else model
        flops_equiv = compute_flops_equivalent(cpu_model, CONTEXT_LEN, NUM_VARIATES, "cpu")
        if device != "cpu":
            model.to(device)
    except Exception as e:
        logger.warning("FLOPs measurement failed: %s", e)
        flops_equiv = 0

    # 5. Size gate — configurable tolerance for wallclock calibration noise.
    #    Default 50%: wallclock ratios vary 2-4x across CPU hardware for small
    #    models where Python/BLAS overhead dominates actual compute.
    SIZE_GATE_TOLERANCE = float(os.environ.get("RADAR_SIZE_GATE_TOLERANCE", "0.50"))
    if min_flops > 0 and max_flops > 0:
        effective_min = int(min_flops * (1 - SIZE_GATE_TOLERANCE))
        effective_max = int(max_flops * (1 + SIZE_GATE_TOLERANCE))
        if not (effective_min <= flops_equiv <= effective_max):
            return JSONResponse(content=_result(
                round_id, miner_hotkey, "size_violation",
                flops_equivalent_size=flops_equiv,
                error=f"FLOPs {flops_equiv} outside [{min_flops}, {max_flops}] "
                      f"(effective [{effective_min}, {effective_max}] with {SIZE_GATE_TOLERANCE:.0%} tolerance)",
            ))

    # 6. Train (simplified harness logic)
    start = time.time()
    num_params = sum(p.numel() for p in model.parameters())
    try:
        from prepare import get_dataloader, validate
        from harness import _read_config, default_loss

        cfg = _read_config(sub)
        optimizer = sub.build_optimizer(model)

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

        model.train()
        step = 0
        for batch in get_dataloader(batch_size=cfg["batch_size"]):
            if time.time() - start > time_budget:
                break
            context = batch["context"].to(device)
            targets = batch["target"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                predictions = model(context)
                loss = loss_fn(predictions, targets, QUANTILES) / cfg["grad_accum_steps"]

            loss.backward()
            if (step + 1) % cfg["grad_accum_steps"] == 0:
                if cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
            step += 1

    except Exception as e:
        return JSONResponse(content=_result(
            round_id, miner_hotkey, "failed", error=str(e),
            flops_equivalent_size=flops_equiv,
            training_time_seconds=time.time() - start,
        ))

    # 7. Save checkpoint with safetensors
    checkpoint_path = "/workspace/checkpoints/model.safetensors"
    os.makedirs("/workspace/checkpoints", exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), checkpoint_path)
    training_time = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    # 8. Upload to R2 via presigned URLs (preferred) or direct R2 (localnet fallback)
    from shared.artifacts import TrainingMeta
    from shared.artifacts import checkpoint_key as ck_fn, architecture_key as ak_fn

    ck_str = ck_fn(round_id, miner_hotkey)
    ak_str = ak_fn(round_id, miner_hotkey)

    meta = TrainingMeta(
        round_id=round_id,
        miner_hotkey=miner_hotkey,
        status="success",
        flops_equivalent_size=flops_equiv,
        training_time_seconds=training_time,
        num_steps=step,
        num_params_M=num_params / 1e6,
        peak_vram_mb=peak_vram,
    )

    try:
        if upload_urls:
            from shared.artifacts import upload_training_artifacts_presigned
            upload_training_artifacts_presigned(
                presigned_urls=upload_urls,
                checkpoint_path=checkpoint_path,
                architecture_code=architecture_code,
                stdout_log="",
                meta=meta,
            )
        else:
            # Localnet fallback: use direct R2 credentials
            from shared.artifacts import upload_training_artifacts
            r2 = _get_r2()
            upload_training_artifacts(
                r2=r2,
                round_id=round_id,
                miner_hotkey=miner_hotkey,
                checkpoint_path=checkpoint_path,
                architecture_code=architecture_code,
                stdout_log="",
                meta=meta,
            )
    except Exception as e:
        logger.error("R2 upload failed: %s", e)

    # 9. Return training metadata
    return _result(
        round_id, miner_hotkey, "success",
        flops_equivalent_size=flops_equiv,
        training_time_seconds=training_time,
        num_steps=step,
        checkpoint_key=ck_str,
        architecture_key=ak_str,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


def _result(round_id: int, miner_hotkey: str, status: str, **kwargs) -> dict:
    """Build a training result dict."""
    return {
        "round_id": round_id,
        "miner_hotkey": miner_hotkey,
        "status": status,
        **kwargs,
    }


def _load_metagraph():
    """Load metagraph with caching. Returns cached copy if fresh, else refreshes.

    Returns None only if the chain has never been reachable. Callers must
    treat None as "deny" — fail closed.
    """
    global _metagraph_cache, _metagraph_last_refresh
    now = time.time()
    if _metagraph_cache is not None and (now - _metagraph_last_refresh) < METAGRAPH_REFRESH_INTERVAL:
        return _metagraph_cache
    with _metagraph_lock:
        # Double-check after acquiring lock
        if _metagraph_cache is not None and (now - _metagraph_last_refresh) < METAGRAPH_REFRESH_INTERVAL:
            return _metagraph_cache
        try:
            import bittensor as bt
            netuid = int(os.getenv("NETUID", "1"))
            network = os.getenv("SUBTENSOR_NETWORK", "finney")
            subtensor = bt.Subtensor(network=network)
            mg = subtensor.metagraph(netuid)
            _metagraph_cache = mg
            _metagraph_last_refresh = time.time()
            logger.info("Metagraph refreshed (%d neurons)", mg.n)
            return mg
        except Exception as e:
            logger.warning("Metagraph refresh failed: %s", e)
            # Return stale cache if available (better than nothing)
            if _metagraph_cache is not None:
                logger.info("Using stale metagraph cache")
                return _metagraph_cache
            return None


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TRAINER_PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
