"""Training artifact upload helpers for the generalist server.

Handles uploading checkpoints, architecture code, and metadata to R2
via presigned URLs or direct R2 client.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# R2 client (initialized on first use, only for localnet fallback)
_r2 = None

# ── GIFT-Eval data cache ────────────────────────────────────────────
_gift_eval_cache_dir = os.environ.get(
    "RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval",
)
_gift_eval_ready = False


def _get_r2():
    global _r2
    if _r2 is None:
        from shared.r2_audit import R2AuditLog
        _r2 = R2AuditLog()
    return _r2


def upload_failure_meta(
    round_id: int, miner_hotkey: str, upload_urls: dict, result: dict,
):
    """Upload a training_meta.json for a failed run."""
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


def upload_artifacts(
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
        logger.error(
            "Artifact upload incomplete for round %d miner %s",
            round_id, miner_hotkey,
        )

    result.pop("checkpoint_path", None)
    result["status"] = status
    result["checkpoint_key"] = ck_str
    result["architecture_key"] = ak_str
    return result


def download_gift_eval(
    urls: dict[str, str], cache_dir: str | None = None,
):
    """Download GIFT-Eval data from presigned GET URLs."""
    global _gift_eval_ready
    import httpx

    cache = Path(cache_dir or _gift_eval_cache_dir)
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
        logger.info(
            "Downloaded %d GIFT-Eval datasets from presigned URLs", downloaded,
        )
    _gift_eval_ready = True


def is_gift_eval_ready() -> bool:
    """Check if GIFT-Eval data has been downloaded."""
    return _gift_eval_ready
