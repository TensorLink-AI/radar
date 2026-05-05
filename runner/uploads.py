"""Artifact upload helpers for the trainer server.

Split out of ``runner/server.py`` so the request-handling module stays
under the 300-line file cap. All functions assume the caller already
prefetched data and the sandbox already produced a ``result`` dict.

The trainer never sees the architecture owner's hotkey — only the opaque
``submission_id`` minted by the dispatching validator. Bucket paths and
TrainingMeta live entirely in submission_id space until the validator
publishes the post-Phase-C reveal map.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# R2 client (initialised on first use, only for the localnet fallback).
_r2 = None


def _get_r2():
    global _r2
    if _r2 is None:
        from shared.r2_audit import R2AuditLog
        _r2 = R2AuditLog()
    return _r2


def upload_failure_meta(
    round_id: int,
    submission_id: str,
    upload_urls: dict,
    result: dict,
) -> None:
    """Upload a training_meta.json for a failed run so the validator sees it."""
    from shared.artifacts import TrainingMeta

    meta = TrainingMeta(
        round_id=round_id,
        submission_id=submission_id,
        status=result.get("status", "failed"),
        error=result.get("error", ""),
        flops_equivalent_size=result.get("flops_equivalent_size", 0),
        training_time_seconds=result.get("training_time_seconds", 0),
    )
    meta_url = upload_urls.get("meta") if upload_urls else None
    if not meta_url:
        return
    try:
        import httpx
        body = json.dumps(meta.to_dict(), indent=2).encode()
        resp = httpx.put(meta_url, content=body, timeout=30)
        resp.raise_for_status()
        logger.info(
            "Uploaded failure meta for round %d submission %s",
            round_id, submission_id[:12],
        )
    except Exception as e:
        logger.error("Failed to upload failure meta: %s", e)


def upload_artifacts(
    result: dict,
    architecture_code: str,
    round_id: int,
    submission_id: str,
    upload_urls: dict,
) -> dict:
    """Upload checkpoint + architecture + meta. Mutates and returns ``result``."""
    from shared.artifacts import (
        TrainingMeta, checkpoint_key as ck_fn, architecture_key as ak_fn,
    )

    checkpoint_path = result.get("checkpoint_path", "")
    stdout_log = result.get("stdout_log", "")
    ck_str = ck_fn(round_id, submission_id)
    ak_str = ak_fn(round_id, submission_id)

    meta = TrainingMeta(
        round_id=round_id,
        submission_id=submission_id,
        status="success",
        flops_equivalent_size=result.get("flops_equivalent_size", 0),
        training_time_seconds=result.get("training_time_seconds", 0),
        num_steps=result.get("num_steps", 0),
        num_params_M=result.get("num_params_M", 0),
        peak_vram_mb=result.get("peak_vram_mb", 0),
        train_loss_history=result.get("train_loss_history", []),
        val_loss_history=result.get("val_loss_history", []),
        best_val_loss=result.get("best_val_loss"),
        best_val_step=result.get("best_val_step", -1),
        val_cadence_unit=result.get("val_cadence_unit", "step"),
        val_base=result.get("val_base", 0.0),
        val_growth=result.get("val_growth", 0.0),
        val_eval_tokens=result.get("val_eval_tokens", 0),
        flops_per_step_estimate=result.get("flops_per_step_estimate", 0.0),
        reference_eval_loss_history=result.get("reference_eval_loss_history", []),
    )

    upload_ok = True
    try:
        if upload_urls:
            from shared.artifacts import upload_training_artifacts_presigned
            upload_ok = upload_training_artifacts_presigned(
                presigned_urls=upload_urls,
                checkpoint_path=checkpoint_path,
                architecture_code=architecture_code,
                stdout_log=stdout_log,
                meta=meta,
            )
        else:
            from shared.artifacts import upload_training_artifacts
            upload_ok = upload_training_artifacts(
                r2=_get_r2(),
                round_id=round_id,
                submission_id=submission_id,
                checkpoint_path=checkpoint_path,
                architecture_code=architecture_code,
                stdout_log=stdout_log,
                meta=meta,
            )
    except Exception as e:
        logger.error("R2 upload failed: %s", e)
        upload_ok = False

    status = "success" if upload_ok else "upload_failed"
    if not upload_ok:
        logger.error(
            "Artifact upload incomplete for round %d submission %s",
            round_id, submission_id[:12],
        )

    result.pop("checkpoint_path", None)
    result.pop("stdout_log", None)
    result["status"] = status
    result["checkpoint_key"] = ck_str
    result["architecture_key"] = ak_str
    return result
