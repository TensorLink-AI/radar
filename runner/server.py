"""Generalist training server -- routes to task-specific runners by task_name.
Training runs in a network-isolated sandbox subprocess (sandbox_runner.py).
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

from runner.sandbox import (
    SANDBOX_DATA_MODE, prefetch_shards, run_sandbox, run_data_proxy,
)
from runner.uploads import (
    upload_failure_meta, upload_artifacts, download_gift_eval,
    is_gift_eval_ready,
)

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
    """Background task: prefetch data, sandbox train, upload artifacts."""
    round_id = training_config["round_id"]
    miner_hotkey = training_config["miner_hotkey"]
    task_name = training_config.get("task_name", "ts_forecasting")
    logger.info(
        "Starting train+upload: round=%d miner=%s upload_url_keys=%s",
        round_id, miner_hotkey, sorted(upload_urls.keys()) if upload_urls else "(none)",
    )
    try:
        # ── Data prefetch (parent process, has network) ──
        local_data_dir = "/workspace/sandbox/data"
        local_shard_paths: list[str] = []

        if gift_eval_urls:
            download_gift_eval(gift_eval_urls, cache_dir=local_data_dir)

        if SANDBOX_DATA_MODE == "prefetch" and pretrain_shard_urls:
            local_shard_paths = await prefetch_shards(pretrain_shard_urls)

        # ── Build sandbox config ──
        sandbox_config = {
            **training_config,
            "architecture_code": architecture_code,
            "task_name": task_name,
            "data_mode": SANDBOX_DATA_MODE,
            "local_data_dir": local_data_dir,
        }

        if SANDBOX_DATA_MODE == "prefetch":
            sandbox_config["local_shard_paths"] = local_shard_paths
        elif SANDBOX_DATA_MODE == "proxy":
            sandbox_config["n_shards"] = len(pretrain_shard_urls or [])
            sandbox_config["proxy_url"] = "http://127.0.0.1:9999"

        config_path = "/workspace/sandbox/train_config.json"
        with open(config_path, "w") as f:
            json.dump(sandbox_config, f)

        # ── Start proxy if needed (future) ──
        proxy_task = None
        if SANDBOX_DATA_MODE == "proxy" and pretrain_shard_urls:
            proxy_task = asyncio.create_task(
                run_data_proxy(pretrain_shard_urls, port=9999)
            )

        # ── Spawn sandbox ──
        t0 = time.time()
        try:
            result = await run_sandbox(config_path, training_config)
        finally:
            if proxy_task:
                proxy_task.cancel()

        elapsed = time.time() - t0
        logger.info(
            "Training complete: round=%d miner=%s status=%s elapsed=%.1fs",
            round_id, miner_hotkey, result.get("status", "?"), elapsed,
        )

        if result.get("status") in ("build_failed", "size_violation", "failed", "timeout"):
            logger.warning(
                "Training failed for round %d miner %s: %s -- %s",
                round_id, miner_hotkey, result.get("status"), result.get("error", ""),
            )
            upload_failure_meta(round_id, miner_hotkey, upload_urls, result)
            return

        # Upload artifacts to R2
        upload_artifacts(result, architecture_code, round_id, miner_hotkey, upload_urls)
    except Exception as e:
        logger.error(
            "Background train+upload failed for round %d miner %s: %s",
            round_id, miner_hotkey, e, exc_info=True,
        )
        try:
            upload_failure_meta(round_id, miner_hotkey, upload_urls, {
                "status": "failed",
                "error": f"Unhandled exception: {e}",
            })
        except Exception:
            logger.error("Failed to upload failure meta for round %d miner %s", round_id, miner_hotkey)
    finally:
        _train_semaphore.release()
        logger.info("Training semaphore released (round %d miner %s)", round_id, miner_hotkey)


@app.get("/health")
async def health():
    return {"status": "ok", "gift_eval_ready": is_gift_eval_ready()}


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
