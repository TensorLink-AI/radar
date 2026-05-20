"""Phase C → substrate publisher.

Glue layer between the validator's Phase C output and the substrate publishing
pipeline. Two functions:

  * `build_phase_c_records` — pure (no IO). Takes the inputs the validator
    already has at the end of Phase C and returns a list of signed
    `PhaseCRecord`s, one per miner that produced an eval result. Testable
    without mocking.
  * `publish_phase_c_records` — best-effort async wrapper that bundles +
    uploads. *Never raises.* Substrate publishing must not be allowed to take
    down weight-setting; on any failure the publisher logs a warning and
    returns ``None``.

The Hippius client is the structured `shared.hippius_client.HippiusClient`
(TEN-242): ``await client.upload_bundle(data, *, app_tag, phase, run_id,
netuid, extra_metadata=None) -> BundleRef``.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

from shared.substrate import (
    SCHEMA_VERSION,
    PhaseCRecord,
    bundle_sha256,
    records_to_bundle,
    sign_record,
)

logger = logging.getLogger(__name__)


# Phase 2 (TEN-242) ships ``BundleRef`` as the canonical upload return type;
# this re-export keeps existing callers' imports stable.
from shared.hippius_client import BundleRef, UploadResult  # noqa: E402,F401


# ── Pure record construction ─────────────────────────────────────────


def _filter_metrics(metrics: dict) -> dict:
    """Keep only JSON-serialisable scalars (primitives that survive the bundle).

    NaN / inf are dropped because plain ``json.dumps`` can't round-trip them
    losslessly across implementations and we never want a record to verify on
    one validator and fail to parse on another.
    """
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            out[k] = v
        elif isinstance(v, str):
            out[k] = v
        elif v is None:
            out[k] = v
        # Anything else (lists, dicts, numpy scalars, tensors) is silently
        # dropped — record bytes must be deterministic across readers.
    return out


_BOOKKEEPING_KEYS = frozenset({
    "passed_size_gate", "flops_verified", "error",
    "flops_equivalent_size",  # diagnostic, not the primary objective
})


def _derive_eval_status(metrics: dict) -> str:
    """Map an eval-result dict to one of the four documented status strings.

    Priority order matches the spec: a size-gate fail short-circuits everything
    else (the metric, even if reported, doesn't count for scoring); an explicit
    error string beats "no_metric" because it tells the operator *why*; finally
    "no_metric" catches the silent-failure case where the runner produced no
    numeric output at all.
    """
    if not metrics.get("passed_size_gate", False):
        return "size_gate_failed"
    if metrics.get("error"):
        return "eval_failed"
    has_metric = any(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
        for k, v in metrics.items()
        if k not in _BOOKKEEPING_KEYS
    )
    if not has_metric:
        return "no_metric"
    return "ok"


def _hotkey_for_uid(metagraph, uid: int) -> str:
    """Look up a hotkey on the metagraph, tolerating short / missing arrays."""
    hotkeys = getattr(metagraph, "hotkeys", None) or []
    if 0 <= uid < len(hotkeys):
        return hotkeys[uid] or ""
    return ""


def _code_hash_for_uid(commitments: dict, uid: int) -> str:
    """Pull `code_hash` off whatever shape the commitments dict carries."""
    c = commitments.get(uid) if commitments else None
    if c is None:
        return ""
    # ImageCommitment dataclass — attribute access. Some call sites may pass
    # a plain dict (older code paths / tests), so fall back to .get.
    return getattr(c, "code_hash", None) or (c.get("code_hash", "") if isinstance(c, dict) else "") or ""


async def build_phase_c_records(
    *,
    wallet,
    challenge,
    eval_results: dict[int, dict],
    training_metas: dict[int, dict],
    commitments: dict,
    metagraph,
    my_uid: int,
    current_block: int,
    task_name: str,
    block_hash: str,
) -> list[PhaseCRecord]:
    """Build one signed PhaseCRecord per miner in `eval_results`.

    Pure — no network, no disk. Async only so callers can `await` it from the
    same site that awaits `publish_phase_c_records`. Iteration order follows
    the eval_results dict (which is already insertion-ordered in CPython),
    giving deterministic record ordering across runs of the same round.

    Records whose miner has no training meta still get a record (with empty
    sha256 fields) — the eval result alone is enough to claim the miner was
    evaluated. Empty sha256 strings are valid in the schema.
    """
    validator_hotkey = wallet.hotkey.ss58_address
    timestamp = time.time()
    round_id = int(getattr(challenge, "round_id", 0))

    records: list[PhaseCRecord] = []
    for uid, metrics in eval_results.items():
        meta = training_metas.get(uid, {}) or {}
        record = PhaseCRecord(
            schema_version=SCHEMA_VERSION,
            round_id=round_id,
            block_hash=block_hash,
            task=task_name,
            miner_uid=int(uid),
            miner_hotkey=_hotkey_for_uid(metagraph, int(uid)),
            code_hash=_code_hash_for_uid(commitments, int(uid)),
            architecture_sha256=str(meta.get("architecture_sha256", "")),
            checkpoint_sha256=str(meta.get("checkpoint_sha256", "")),
            metrics=_filter_metrics(metrics),
            passed_size_gate=bool(metrics.get("passed_size_gate", False)),
            flops_verified=bool(metrics.get("flops_verified", False)),
            eval_status=_derive_eval_status(metrics),
            validator_uid=int(my_uid),
            validator_hotkey=validator_hotkey,
            validator_block_height=int(current_block),
            timestamp=timestamp,
        )
        records.append(sign_record(record, wallet))
    return records


# ── Async upload (best effort) ───────────────────────────────────────


async def publish_phase_c_records(
    hippius,
    records: list[PhaseCRecord],
    round_id: int,
    validator_hotkey: str,
    *,
    netuid: int = 0,
    app_tag: str = "radar",
) -> Optional[UploadResult]:
    """Upload a Phase C record bundle to Hippius. Best-effort — never raises.

    Substrate publishing is *advisory*: it gives operators a portable, signed
    audit trail but is never a precondition for setting weights. If anything
    goes wrong (no client wired up, network flap, SDK regression) we log a
    warning and return ``None`` so the validator's main loop continues.

    Metadata tags exposed on the upload mirror the bundle's own header so a
    Hippius-side discovery query and an in-bundle parse agree on what
    everything is.
    """
    if not records:
        # Nothing to publish — don't burn a no-op upload.
        return None
    if hippius is None:
        logger.warning(
            "Substrate publish skipped: no Hippius client configured "
            "(round_id=%d, %d records)", round_id, len(records),
        )
        return None
    try:
        bundle = records_to_bundle(records)
        digest = bundle_sha256(bundle)
        # The structured args go on the S3 key + tags; the rest of the
        # schema rides as user metadata so /verify can sanity-check it.
        result = await hippius.upload_bundle(
            bundle,
            app_tag=app_tag, phase="phase_c",
            run_id=str(round_id), netuid=netuid,
            extra_metadata={
                "schema_version": SCHEMA_VERSION,
                "validator_hotkey": validator_hotkey,
                "record_count": str(len(records)),
                "bundle_sha256": digest,
            },
        )
    except Exception as e:  # noqa: BLE001 — best-effort by design
        logger.warning(
            "Substrate publish failed for round_id=%d (%d records): %s",
            round_id, len(records), e,
        )
        return None
    if result is not None:
        cid = getattr(result, "cid", None) or getattr(result, "key", "") or (
            result.get("cid", "") if isinstance(result, dict) else ""
        )
        logger.info(
            "Substrate publish ok: round_id=%d records=%d cid=%s",
            round_id, len(records), cid,
        )
    return result


# ── Round-loop integration helper ────────────────────────────────────


async def run_substrate_publish_step(
    *,
    hippius,
    wallet,
    challenge,
    eval_results: dict[int, dict],
    training_metas: dict[int, dict],
    commitments: dict,
    metagraph,
    my_uid: int,
    current_block: int,
    task_name: str,
    block_hash: str,
    netuid: int = 0,
) -> dict[int, str]:
    """Build, sign, and publish a Phase C bundle. Best-effort; never raises.

    Returns ``{miner_uid: bundle_cid}`` (all miners share the round's single
    bundle CID) so Phase 5 can thread CIDs into per-miner DB writes. Returns
    an empty dict when ``hippius`` is None, ``eval_results`` is empty, or
    record building / upload fails.
    """
    if hippius is None or not eval_results:
        return {}
    if wallet is None:
        # Off-chain deploy — Phase C bundles are SR25519-signed by the
        # validator hotkey. Without a wallet there's nothing to sign with,
        # so we skip substrate publishing entirely.
        return {}
    try:
        records = await build_phase_c_records(
            wallet=wallet, challenge=challenge,
            eval_results=eval_results, training_metas=training_metas,
            commitments=commitments, metagraph=metagraph,
            my_uid=my_uid, current_block=current_block,
            task_name=task_name, block_hash=block_hash,
        )
    except Exception as e:  # noqa: BLE001 — best-effort by design
        logger.warning(
            "Substrate publishing failed (non-fatal): record build error: %s", e,
        )
        return {}
    upload_result = await publish_phase_c_records(
        hippius, records,
        round_id=int(getattr(challenge, "round_id", 0)),
        validator_hotkey=wallet.hotkey.ss58_address,
        netuid=netuid,
    )
    if upload_result is None:
        return {}
    cid = getattr(upload_result, "cid", None) or (
        upload_result.get("cid", "") if isinstance(upload_result, dict) else ""
    )
    if not cid:
        return {}
    return {int(uid): cid for uid in eval_results}


__all__ = [
    "UploadResult",
    "build_phase_c_records",
    "publish_phase_c_records",
    "run_substrate_publish_step",
]
