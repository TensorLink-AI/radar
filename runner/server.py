"""Generalist training server — one server, all tasks.

Routes to task-specific runners based on task_name in the dispatch payload.
Miners deploy this unmodified on Basilica. Validators dispatch via POST /train.

Untrusted miner training code never runs in this process.  Each /train
request prefetches data with the parent's network credentials, then
spawns ``runner/sandbox_runner.py`` in a separate, network-isolated
subprocess via ``runner/sandbox.py::run_sandbox``.  The sandbox child
inherits no R2 / wallet / Basilica secrets and cannot import any
high-level HTTP client.
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
    GIFT_EVAL_DIR,
    SHARD_DIR,
    VAL_SHARD_DIR,
    prefetch_gift_eval,
    prefetch_shards,
    run_sandbox,
)
from runner.uploads import upload_artifacts, upload_failure_meta

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

# ── Task registry ───────────────────────────────────────────────────
# Names accepted on the wire.  The sandbox dispatches to the matching
# task runner module — we do NOT import task runners here so the parent
# server stays minimal and dispatch is data-driven.
_KNOWN_TASKS = frozenset({"ts_forecasting", "ml_training"})


@app.post("/train")
async def train(request: Request):
    """Accept a training job and run it in the background.

    Returns 202 Accepted immediately so proxy timeouts don't kill the
    upload pipeline.  Training + R2 upload happen in a background task;
    the validator discovers results via R2 polling.
    """
    if _train_semaphore.locked():
        return JSONResponse(status_code=429, content={
            "error": "Training job already in progress",
            "reason": "already_running",
        })

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
                signed_by = request.headers.get("x-epistula-signed-by", "?")
                logger.warning(
                    "Auth rejected from %s: %s", signed_by, err,
                )
                return JSONResponse(status_code=403, content={"error": err})
        except Exception as e:
            logger.error("Auth verification failed: %s", e)
            return JSONResponse(status_code=403, content={"error": "Auth verification error"})

    # 2. Per-hotkey rate limit (check only — timestamp recorded after semaphore)
    with _hotkey_lock:
        last = _hotkey_last_request.get(sender, 0.0)
        if time.time() - last < HOTKEY_COOLDOWN_SECONDS:
            remaining = HOTKEY_COOLDOWN_SECONDS - (time.time() - last)
            return JSONResponse(status_code=429, content={
                "error": f"Rate limited. Retry in {remaining:.0f}s",
                "reason": "rate_limited",
                "retry_after": int(remaining) + 1,
            })

    # 3. Parse request
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    architecture_code = data.get("architecture", "")
    if not architecture_code:
        return JSONResponse(status_code=400, content={"error": "Missing architecture code"})

    # 4. Validate task name (the sandbox child does the real dispatch)
    task_name = data.get("task_name", "ts_forecasting")
    if task_name not in _KNOWN_TASKS:
        return JSONResponse(status_code=400, content={
            "error": f"Unknown task '{task_name}'. Supported: {sorted(_KNOWN_TASKS)}",
        })

    # 5. Acquire semaphore synchronously to guarantee 429 on overlap,
    #    then kick off training + upload in a background task so the
    #    HTTP response isn't blocked by the full training duration.
    #    The validator discovers results via R2 polling.
    if not _train_semaphore.locked():
        await _train_semaphore.acquire()
    else:
        # Race: another request acquired between the check and here
        return JSONResponse(status_code=429, content={
            "error": "Training job already in progress",
            "reason": "already_running",
        })

    # Record rate-limit timestamp AFTER semaphore acquired — if the
    # request fails before this point, retries won't be rate-limited.
    with _hotkey_lock:
        _hotkey_last_request[sender] = time.time()

    training_config = {
        "seed": data.get("seed", 42),
        "round_id": data.get("round_id", 0),
        "min_flops": data.get("min_flops_equivalent", 0),
        "max_flops": data.get("max_flops_equivalent", 0),
        # Opaque submission_id from the dispatching validator; the
        # trainer-host never sees the architecture owner's hotkey.
        "submission_id": data.get("submission_id", "unknown"),
        "time_budget": data.get("time_budget", 300),
        "task_name": task_name,
    }

    upload_urls = data.get("upload_urls", {})
    gift_eval_urls = data.get("gift_eval_urls", {})
    pretrain_shard_urls = data.get("pretrain_shard_urls", [])
    pretrain_val_shard_urls = data.get("pretrain_val_shard_urls", [])

    asyncio.create_task(_train_and_upload(
        architecture_code, training_config,
        upload_urls, gift_eval_urls, pretrain_shard_urls,
        pretrain_val_shard_urls,
    ))

    logger.info(
        "Training job accepted: round=%d submission=%s task=%s",
        training_config["round_id"], training_config["submission_id"][:12], task_name,
    )
    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "round_id": training_config["round_id"]},
    )


_LOG_CAPTURE_CAP = 10 * 1024 * 1024  # 10 MB


async def _train_and_upload(
    architecture_code: str,
    training_config: dict,
    upload_urls: dict,
    gift_eval_urls: dict,
    pretrain_shard_urls: list[str] | None = None,
    pretrain_val_shard_urls: list[str] | None = None,
):
    """Background task: prefetch data, sandbox train, upload artifacts.

    Network-touching steps (prefetch + R2 upload) run in *this* process,
    which has the deployment's secrets.  The miner's training code runs
    in a separate ``sandbox_runner.py`` subprocess with no R2 / wallet
    credentials and no network-capable Python imports.
    """
    round_id = training_config["round_id"]
    submission_id = training_config["submission_id"]
    logger.info(
        "Starting train+upload: round=%d submission=%s upload_url_keys=%s",
        round_id, submission_id[:12], sorted(upload_urls.keys()) if upload_urls else "(none)",
    )
    try:
        # ── Prefetch data with the parent's network credentials ──
        gift_dir = ""
        if gift_eval_urls:
            gift_dir = await prefetch_gift_eval(gift_eval_urls, dest_dir=GIFT_EVAL_DIR) or ""

        train_paths: list[str] = []
        if pretrain_shard_urls:
            train_paths = await prefetch_shards(
                pretrain_shard_urls, dest_dir=SHARD_DIR,
            )

        val_paths: list[str] = []
        if pretrain_val_shard_urls:
            val_paths = await prefetch_shards(
                pretrain_val_shard_urls, dest_dir=VAL_SHARD_DIR,
            )

        # ── Build the sandbox config and spawn the child ──
        sandbox_config = {
            **training_config,
            "architecture_code": architecture_code,
            "pretrain_shard_paths": train_paths,
            "pretrain_val_shard_paths": val_paths,
            "gift_eval_dir": gift_dir,
        }

        t0 = time.time()
        result, sandbox_log = await run_sandbox(sandbox_config)
        elapsed = time.time() - t0
        result["stdout_log"] = sandbox_log[-_LOG_CAPTURE_CAP:]

        # Echo the tail of the sandbox stderr to the trainer's own log so
        # operators can see miner traces / security probes in ``docker
        # logs`` without having to wait for the R2 artifact upload.
        if sandbox_log:
            tail = sandbox_log[-4096:]
            logger.info(
                "Sandbox stderr tail (round %d submission %s):\n%s",
                round_id, submission_id[:12], tail,
            )

        logger.info(
            "Training complete: round=%d submission=%s status=%s elapsed=%.1fs",
            round_id, submission_id[:12], result.get("status", "?"), elapsed,
        )

        if result.get("status") in ("build_failed", "size_violation", "failed", "timeout"):
            logger.warning(
                "Training failed for round %d submission %s: %s — %s",
                round_id, submission_id[:12], result.get("status"), result.get("error", ""),
            )
            upload_failure_meta(round_id, submission_id, upload_urls, result)
            return

        upload_artifacts(result, architecture_code, round_id, submission_id, upload_urls)
    except Exception as e:
        logger.error(
            "Background train+upload failed for round %d submission %s: %s",
            round_id, submission_id[:12], e, exc_info=True,
        )
        try:
            upload_failure_meta(round_id, submission_id, upload_urls, {
                "status": "failed",
                "error": f"Unhandled exception: {e}",
            })
        except Exception:
            logger.error(
                "Failed to upload failure meta for round %d submission %s",
                round_id, submission_id[:12],
            )
    finally:
        _train_semaphore.release()
        logger.info(
            "Training semaphore released (round %d submission %s)",
            round_id, submission_id[:12],
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


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
    # Configure the application root logger so INFO messages from
    # ``runner.sandbox`` / ``runner.server`` / ``shared.*`` show up in
    # ``docker logs``.  Without this, only WARNING+ surfaces — making the
    # sandbox dispatch + miner stderr echo invisible to operators.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    port = int(os.getenv("TRAINER_PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
