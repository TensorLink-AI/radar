"""Generalist training server — one server, all tasks.

Routes to task-specific runners based on task_name in the dispatch payload.
Miners deploy this unmodified on Basilica. Validators dispatch via POST /train.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time

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

# ── Runner registry ─────────────────────────────────────────────────
# Maps task_name -> run_training callable
_RUNNERS: dict[str, object] = {}


def _register_runners():
    """Lazy-load and register all task runners."""
    if _RUNNERS:
        return
    from runner.timeseries_forecast.train import run_training as ts_train
    _RUNNERS["ts_forecasting"] = ts_train
    _RUNNERS["ml_training"] = ts_train  # alias


def _get_r2():
    global _r2
    if _r2 is None:
        from shared.r2_audit import R2AuditLog
        _r2 = R2AuditLog()
    return _r2


@app.post("/train")
async def train(request: Request):
    """Execute a training job, routing to the appropriate task runner."""
    if _train_semaphore.locked():
        return JSONResponse(status_code=429, content={"error": "Training job already in progress"})

    body = await request.body()

    # 1. Auth — FAIL CLOSED
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

    # 2. Per-hotkey rate limit
    with _hotkey_lock:
        last = _hotkey_last_request.get(sender, 0.0)
        if time.time() - last < HOTKEY_COOLDOWN_SECONDS:
            remaining = HOTKEY_COOLDOWN_SECONDS - (time.time() - last)
            return JSONResponse(status_code=429, content={
                "error": f"Rate limited. Retry in {remaining:.0f}s",
            })
        _hotkey_last_request[sender] = time.time()

    # 3. Parse request
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    architecture_code = data.get("architecture", "")
    if not architecture_code:
        return JSONResponse(status_code=400, content={"error": "Missing architecture code"})

    # 4. Route to task runner
    _register_runners()
    task_name = data.get("task_name", "ts_forecasting")
    if task_name not in _RUNNERS:
        return JSONResponse(status_code=400, content={
            "error": f"Unknown task '{task_name}'. Supported: {sorted(_RUNNERS.keys())}",
        })

    # 5. Download GIFT-Eval data if provided
    gift_eval_urls = data.get("gift_eval_urls", {})
    if gift_eval_urls:
        _download_gift_eval_from_urls(gift_eval_urls)

    # 6. Execute training under semaphore
    training_config = {
        "seed": data.get("seed", 42),
        "round_id": data.get("round_id", 0),
        "min_flops": data.get("min_flops_equivalent", 0),
        "max_flops": data.get("max_flops_equivalent", 0),
        "miner_hotkey": data.get("miner_hotkey", "unknown"),
        "time_budget": data.get("time_budget", 300),
    }

    async with _train_semaphore:
        runner_fn = _RUNNERS[task_name]
        result = runner_fn(architecture_code, training_config)

    round_id = training_config["round_id"]
    miner_hotkey = training_config["miner_hotkey"]

    if result.get("status") in ("build_failed", "size_violation", "failed"):
        return JSONResponse(content=result)

    # 7. Upload checkpoint to R2
    upload_urls = data.get("upload_urls", {})
    result = _upload_artifacts(
        result, architecture_code, round_id, miner_hotkey, upload_urls,
    )
    return JSONResponse(content=result)


def _upload_artifacts(
    result: dict, architecture_code: str,
    round_id: int, miner_hotkey: str, upload_urls: dict,
) -> dict:
    """Upload checkpoint + architecture to R2, return updated result."""
    from shared.artifacts import (
        TrainingMeta, checkpoint_key as ck_fn, architecture_key as ak_fn,
    )

    checkpoint_path = result.get("checkpoint_path", "")
    ck_str = ck_fn(round_id, miner_hotkey)
    ak_str = ak_fn(round_id, miner_hotkey)

    meta = TrainingMeta(
        round_id=round_id,
        miner_hotkey=miner_hotkey,
        status="success",
        flops_equivalent_size=result.get("flops_equivalent_size", 0),
        training_time_seconds=result.get("training_time_seconds", 0),
        num_steps=result.get("num_steps", 0),
        num_params_M=result.get("num_params_M", 0),
        peak_vram_mb=result.get("peak_vram_mb", 0),
    )

    upload_ok = True
    try:
        if upload_urls:
            from shared.artifacts import upload_training_artifacts_presigned
            upload_ok = upload_training_artifacts_presigned(
                presigned_urls=upload_urls,
                checkpoint_path=checkpoint_path,
                architecture_code=architecture_code,
                stdout_log="", meta=meta,
            )
        else:
            from shared.artifacts import upload_training_artifacts
            upload_ok = upload_training_artifacts(
                r2=_get_r2(), round_id=round_id, miner_hotkey=miner_hotkey,
                checkpoint_path=checkpoint_path,
                architecture_code=architecture_code,
                stdout_log="", meta=meta,
            )
    except Exception as e:
        logger.error("R2 upload failed: %s", e)
        upload_ok = False

    status = "success" if upload_ok else "upload_failed"
    if not upload_ok:
        logger.error("Artifact upload incomplete for round %d miner %s", round_id, miner_hotkey)

    # Remove internal-only field, add R2 keys
    result.pop("checkpoint_path", None)
    result["status"] = status
    result["checkpoint_key"] = ck_str
    result["architecture_key"] = ak_str
    return result


# ── GIFT-Eval data cache ────────────────────────────────────────────
_gift_eval_cache_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")
_gift_eval_ready = False


def _download_gift_eval_from_urls(urls: dict[str, str]):
    """Download GIFT-Eval data from presigned GET URLs."""
    global _gift_eval_ready
    import httpx
    from pathlib import Path

    cache = Path(_gift_eval_cache_dir)
    downloaded = 0
    for name, url in urls.items():
        local_dir = cache / name
        local_path = local_dir / "data-00000-of-00001.arrow"
        if local_path.exists() and local_path.stat().st_size > 0:
            continue
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            resp = httpx.get(url, timeout=120)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            downloaded += 1
        except Exception as e:
            logger.warning("Failed to download GIFT-Eval %s: %s", name, e)
    if downloaded:
        logger.info("Downloaded %d GIFT-Eval datasets from presigned URLs", downloaded)
    _gift_eval_ready = True


@app.get("/health")
async def health():
    return {"status": "ok", "gift_eval_ready": _gift_eval_ready}


def _load_metagraph():
    """Load metagraph with caching. Returns None if never reachable."""
    global _metagraph_cache, _metagraph_last_refresh
    now = time.time()
    if _metagraph_cache is not None and (now - _metagraph_last_refresh) < METAGRAPH_REFRESH_INTERVAL:
        return _metagraph_cache
    with _metagraph_lock:
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
            if _metagraph_cache is not None:
                logger.info("Using stale metagraph cache")
                return _metagraph_cache
            return None


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TRAINER_PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
