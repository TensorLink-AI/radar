"""Independent verification of Phase C records on Hippius.

Powers the ``GET /experiments/{index}/verify`` endpoint: for each substrate
CID an experiment carries, fetch the bundle, find the record matching this
miner+round, sr25519-verify the signature, and compare the record's metric
dict against what's stored in the DB. Pure-async, no FastAPI dependency, so
this is straightforward to unit-test.

The Hippius client is duck-typed: anything with
``await client.download_bundle(cid: str) -> bytes`` works, which keeps this
module decoupled from the (still in flight) TEN-242 wrapper.
"""

from __future__ import annotations

import logging
from typing import Any

from shared.database import DataElement
from shared.substrate import (
    PhaseCRecord,
    records_from_bundle,
    verify_record,
)

logger = logging.getLogger(__name__)


def _normalise_metrics(d: dict) -> dict:
    """Filter to JSON-comparable scalar entries.

    The publisher already drops non-scalars from records before signing
    (see validator/substrate_publisher._filter_metrics). Mirroring that
    filter here means the comparison is "what was signed" vs. "what was
    stored", not contaminated by columns the DB happened to add (e.g.
    ``loss_curve`` which lives elsewhere on the row).
    """
    out: dict[str, Any] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
    return out


def _diff_metrics(record: PhaseCRecord, element: DataElement) -> list[str]:
    """Return a list of human-readable discrepancies between record + DB row.

    Empty list = clean match. Only keys present in the (signed) record are
    compared; columns the DB carries that aren't in the bundle are ignored,
    since the bundle is the authoritative claim and the DB is the cache.
    """
    discrepancies: list[str] = []
    db_objectives = _normalise_metrics(element.objectives or {})
    rec_metrics = _normalise_metrics(record.metrics or {})
    for key, rec_val in rec_metrics.items():
        if key not in db_objectives:
            discrepancies.append(f"metric {key!r} present in record, missing in DB")
            continue
        if db_objectives[key] != rec_val:
            discrepancies.append(
                f"metric {key!r}: record={rec_val!r} db={db_objectives[key]!r}",
            )
    if record.miner_uid != element.miner_uid:
        discrepancies.append(
            f"miner_uid: record={record.miner_uid} db={element.miner_uid}",
        )
    if record.miner_hotkey and record.miner_hotkey != element.miner_hotkey:
        discrepancies.append(
            f"miner_hotkey: record={record.miner_hotkey!r} "
            f"db={element.miner_hotkey!r}",
        )
    return discrepancies


def _find_record(
    records: list[PhaseCRecord], element: DataElement,
) -> PhaseCRecord | None:
    """Pick the record in the bundle matching this experiment's identity.

    Match key is ``(round_id, miner_uid)`` — a single bundle covers one
    round, one validator, all miners, so that pair is unique within it.
    """
    target_round = int(element.round_id)
    target_uid = int(element.miner_uid)
    for r in records:
        if r.round_id == target_round and r.miner_uid == target_uid:
            return r
    return None


async def verify_experiment(
    *,
    hippius,
    element: DataElement,
) -> list[dict]:
    """Independently verify every CID an experiment claims to be backed by.

    Returns a list of per-CID verification dicts (shape declared in the
    Phase 5 spec). Empty list if the experiment has no substrate_cids.
    Raises nothing: each CID's failure modes (network, missing record,
    bad signature, metric mismatch) appear as flags + a ``discrepancies``
    list on its own dict.
    """
    if not element.substrate_cids:
        return []
    if hippius is None:
        # The endpoint guards on this, but defend in depth.
        raise RuntimeError("Hippius client not configured")

    out: list[dict] = []
    for entry in element.substrate_cids:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("cid", "")
        validator_hotkey = entry.get("validator_hotkey", "")
        result: dict[str, Any] = {
            "validator_hotkey": validator_hotkey,
            "cid": cid,
            "fetchable": False,
            "signature_valid": False,
            "matches_db": False,
            "discrepancies": [],
        }
        if not cid:
            result["discrepancies"].append("cid is empty")
            out.append(result)
            continue

        # 1. Fetch bundle.
        try:
            data = await hippius.download_bundle(cid)
            result["fetchable"] = True
        except Exception as e:  # noqa: BLE001 — surface as flag, not exception
            result["discrepancies"].append(f"fetch failed: {e}")
            out.append(result)
            continue

        # 2. Parse + locate the matching record.
        try:
            records = records_from_bundle(data)
        except Exception as e:  # noqa: BLE001
            result["discrepancies"].append(f"bundle parse failed: {e}")
            out.append(result)
            continue
        record = _find_record(records, element)
        if record is None:
            result["discrepancies"].append(
                f"no record matching round_id={element.round_id} "
                f"miner_uid={element.miner_uid} in bundle"
            )
            out.append(result)
            continue

        # 3. Verify signature against the bundle's claimed validator hotkey
        #    (cross-checked against the audit-list entry's claimed hotkey).
        ok, err = verify_record(record, expected_hotkey=validator_hotkey or None)
        result["signature_valid"] = ok
        if not ok:
            result["discrepancies"].append(f"signature: {err}")

        # 4. Compare against DB. matches_db only goes true if signature was
        #    valid AND the metric/identity comparison was clean.
        diffs = _diff_metrics(record, element)
        if diffs:
            result["discrepancies"].extend(diffs)
        result["matches_db"] = ok and not diffs

        out.append(result)
    return out


__all__ = ["verify_experiment"]
