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
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Preload metagraph at startup so /train doesn't block on first request."""
    localnet = os.getenv("RADAR_LOCALNET", "").lower() in ("1", "true")
    if not localnet:
        logger.info("Preloading metagraph at startup...")
        mg = await asyncio.to_thread(_load_metagraph)
        if mg is not None:
            logger.info("Metagraph preloaded (%d neurons)", mg.n)
        else:
            logger.warning("Metagraph preload failed — will retry on first request")
    yield


app = FastAPI(title="Radar Trainer", lifespan=_lifespan)

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
    """Accept a training job and run it in the background.

    Returns 202 Accepted immediately so proxy timeouts don't kill the
    upload pipeline.  Training + R2 upload happen in a background task;
    the validator discovers results via R2 polling.
    """
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

    # 5. Acquire semaphore synchronously to guarantee 429 on overlap,
    #    then kick off training + upload in a background task so the
    #    HTTP response isn't blocked by the full training duration.
    #    The validator discovers results via R2 polling.
    if not _train_semaphore.locked():
        await _train_semaphore.acquire()
    else:
        # Race: another request acquired between the check and here
        return JSONResponse(status_code=429, content={"error": "Training job already in progress"})

    training_config = {
        "seed": data.get("seed", 42),
        "round_id": data.get("round_id", 0),
        "min_flops": data.get("min_flops_equivalent", 0),
        "max_flops": data.get("max_flops_equivalent", 0),
        "miner_hotkey": data.get("miner_hotkey", "unknown"),
        "time_budget": data.get("time_budget", 300),
    }

    runner_fn = _RUNNERS[task_name]
    upload_urls = data.get("upload_urls", {})
    gift_eval_urls = data.get("gift_eval_urls", {})
    pretrain_shard_urls = data.get("pretrain_shard_urls", [])

    asyncio.create_task(_train_and_upload(
        runner_fn, architecture_code, training_config,
        upload_urls, gift_eval_urls, pretrain_shard_urls,
    ))

    logger.info(
        "Training job accepted: round=%d miner=%s task=%s",
        training_config["round_id"], training_config["miner_hotkey"], task_name,
    )
    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "round_id": training_config["round_id"]},
    )


async def _train_and_upload(
    runner_fn,
    architecture_code: str,
    training_config: dict,
    upload_urls: dict,
    gift_eval_urls: dict,
    pretrain_shard_urls: list[str] | None = None,
):
    """Background task: download data, train, upload artifacts, release semaphore."""
    round_id = training_config["round_id"]
    miner_hotkey = training_config["miner_hotkey"]
    logger.info(
        "Starting train+upload: round=%d miner=%s upload_url_keys=%s",
        round_id, miner_hotkey, sorted(upload_urls.keys()) if upload_urls else "(none)",
    )
    try:
        # Download GIFT-Eval data if provided (still needed for validation)
        if gift_eval_urls:
            _download_gift_eval_from_urls(gift_eval_urls)

        # Pass pretrain shard URLs via env var for the training harness
        if pretrain_shard_urls:
            os.environ["RADAR_PRETRAIN_SHARD_URLS"] = json.dumps(pretrain_shard_urls)
            logger.info("Set %d pretrain shard URLs for training", len(pretrain_shard_urls))
        else:
            os.environ.pop("RADAR_PRETRAIN_SHARD_URLS", None)

        # Run training
        t0 = time.time()
        result = await asyncio.to_thread(runner_fn, architecture_code, training_config)
        elapsed = time.time() - t0
        logger.info(
            "Training complete: round=%d miner=%s status=%s elapsed=%.1fs",
            round_id, miner_hotkey, result.get("status", "?"), elapsed,
        )

        if result.get("status") in ("build_failed", "size_violation", "failed"):
            logger.warning(
                "Training failed for round %d miner %s: %s — %s",
                round_id, miner_hotkey, result.get("status"), result.get("error", ""),
            )
            # Upload a failure meta so the validator knows what happened
            _upload_failure_meta(round_id, miner_hotkey, upload_urls, result)
            return

        # Upload artifacts to R2
        _upload_artifacts(result, architecture_code, round_id, miner_hotkey, upload_urls)
    except Exception as e:
        logger.error(
            "Background train+upload failed for round %d miner %s: %s",
            round_id, miner_hotkey, e, exc_info=True,
        )
        try:
            _upload_failure_meta(round_id, miner_hotkey, upload_urls, {
                "status": "failed",
                "error": f"Unhandled exception: {e}",
            })
        except Exception:
            logger.error("Failed to upload failure meta for round %d miner %s", round_id, miner_hotkey)
    finally:
        _train_semaphore.release()
        logger.info("Training semaphore released (round %d miner %s)", round_id, miner_hotkey)


def _upload_failure_meta(
    round_id: int, miner_hotkey: str, upload_urls: dict, result: dict,
):
    """Upload a training_meta.json for a failed run so the validator can see it."""
    from shared.artifacts import TrainingMeta
    meta = TrainingMeta(
        round_id=round_id,
        miner_hotkey=miner_hotkey,
        status=result.get("status", "failed"),
        error=result.get("error", ""),
        flops_equivalent_size=result.get("flops_equivalent_size", 0),
        training_time_seconds=result.get("training_time_seconds", 0),
    )
    if upload_urls.get("meta"):
        try:
            import httpx
            body = json.dumps(meta.to_dict(), indent=2).encode()
            resp = httpx.put(upload_urls["meta"], content=body, timeout=30)
            resp.raise_for_status()
            logger.info("Uploaded failure meta for round %d miner %s", round_id, miner_hotkey)
        except Exception as e:
            logger.error("Failed to upload failure meta: %s", e)


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
