"""RunPod Serverless handler — same training pipeline, different entry shape.

When the trainer image runs as a RunPod Serverless worker, the
RunPod runtime invokes ``handle(event)`` with ``event["input"]``
holding the dispatch payload the validator built (architecture,
seed, upload_urls, etc — same shape as the body of POST /train on
the FastAPI server).

The handler reuses the existing ``runner.sandbox`` + ``runner.uploads``
machinery so we don't duplicate the training/upload logic. The only
substantive difference vs ``server.py``:

* No HTTP-level auth — the relay (miner listener) verified the
  validator's Epistula signature before submitting the job, and
  RunPod's own API auth gates job submission to the miner's
  account-scoped key. By the time the worker sees the input, the
  payload has already passed both checks.

* Response is returned synchronously to RunPod (which streams it
  back to the submitter), but artifacts still flow through R2 via
  the presigned URLs in the payload — the validator polls R2 for
  completion exactly as in the FastAPI path.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


_LOG_CAPTURE_CAP = 10 * 1024 * 1024  # 10 MB — same as runner/server.py
_KNOWN_TASKS = frozenset({"ts_forecasting", "ml_training"})


def handle(event: dict) -> dict:
    """RunPod Serverless entry point.

    RunPod calls this synchronously per job. Returning early is fine —
    the validator's source of truth is R2, not the handler return
    value. We return enough metadata for telemetry / debugging.
    """
    payload = (event or {}).get("input") or {}
    architecture_code = payload.get("architecture", "")
    if not architecture_code:
        return {"status": "failed", "error": "missing architecture code"}

    task_name = payload.get("task_name", "ts_forecasting")
    if task_name not in _KNOWN_TASKS:
        return {"status": "failed", "error": f"unknown task {task_name!r}"}

    training_config = {
        "seed": int(payload.get("seed", 42)),
        "round_id": int(payload.get("round_id", 0)),
        "min_flops": int(payload.get("min_flops_equivalent", 0)),
        "max_flops": int(payload.get("max_flops_equivalent", 0)),
        "miner_hotkey": payload.get("miner_hotkey", "unknown"),
        "time_budget": int(payload.get("time_budget", 300)),
        "task_name": task_name,
    }
    upload_urls = payload.get("upload_urls", {}) or {}
    gift_eval_urls = payload.get("gift_eval_urls", {}) or {}
    pretrain_shard_urls = payload.get("pretrain_shard_urls", []) or []
    pretrain_val_shard_urls = payload.get("pretrain_val_shard_urls", []) or []

    return asyncio.run(_run(
        architecture_code, training_config, upload_urls,
        gift_eval_urls, pretrain_shard_urls, pretrain_val_shard_urls,
    ))


async def _run(
    architecture_code: str,
    training_config: dict,
    upload_urls: dict,
    gift_eval_urls: dict,
    pretrain_shard_urls: list,
    pretrain_val_shard_urls: list,
) -> dict:
    """Async core — mirrors ``runner.server._train_and_upload``.

    Imports inside the function so the module stays cheap to load
    (RunPod cold-starts call ``handle`` once per worker; module-level
    imports happen at fork time, but training-only deps shouldn't
    block bootstrap-time integrity checks).
    """
    from runner.sandbox import (
        GIFT_EVAL_DIR, SHARD_DIR, VAL_SHARD_DIR,
        prefetch_gift_eval, prefetch_shards, run_sandbox,
    )
    from runner.uploads import upload_artifacts, upload_failure_meta

    round_id = training_config["round_id"]
    miner_hotkey = training_config["miner_hotkey"]
    logger.info(
        "RunPod handler: round=%d miner=%s task=%s",
        round_id, miner_hotkey, training_config["task_name"],
    )

    try:
        gift_dir = ""
        if gift_eval_urls:
            gift_dir = await prefetch_gift_eval(gift_eval_urls, dest_dir=GIFT_EVAL_DIR) or ""
        train_paths = (
            await prefetch_shards(pretrain_shard_urls, dest_dir=SHARD_DIR)
            if pretrain_shard_urls else []
        )
        val_paths = (
            await prefetch_shards(pretrain_val_shard_urls, dest_dir=VAL_SHARD_DIR)
            if pretrain_val_shard_urls else []
        )

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

        logger.info(
            "Training complete: round=%d miner=%s status=%s elapsed=%.1fs",
            round_id, miner_hotkey, result.get("status", "?"), elapsed,
        )

        if result.get("status") in ("build_failed", "size_violation", "failed", "timeout"):
            upload_failure_meta(round_id, miner_hotkey, upload_urls, result)
            return {
                "status": result.get("status"),
                "round_id": round_id,
                "error": result.get("error", ""),
            }

        upload_artifacts(result, architecture_code, round_id, miner_hotkey, upload_urls)
        return {
            "status": result.get("status", "success"),
            "round_id": round_id,
            "flops_equivalent_size": int(result.get("flops_equivalent_size", 0)),
            "training_time_seconds": float(result.get("training_time_seconds", elapsed)),
            "checkpoint_key": result.get("checkpoint_key", ""),
            "architecture_key": result.get("architecture_key", ""),
        }
    except Exception as e:
        logger.error(
            "RunPod handler failed for round %d miner %s: %s",
            round_id, miner_hotkey, e, exc_info=True,
        )
        try:
            upload_failure_meta(round_id, miner_hotkey, upload_urls, {
                "status": "failed",
                "error": f"Unhandled exception: {e}",
            })
        except Exception:
            logger.error("Failed to upload failure meta")
        return {"status": "failed", "round_id": round_id, "error": str(e)}
