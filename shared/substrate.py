"""Phase C Substrate record schema, signing, and verification.

A pure-Python contract for what a validator publishes on-chain (or on Hippius)
at the end of Phase C: a `PhaseCRecord` is the structured, signed claim that a
particular checkpoint was evaluated by a particular validator with a particular
result. Every other piece of the substrate pipeline (bundling, IPFS upload,
chain extrinsic) binds to the canonical bytes computed here.

Design constraints:
  * Deterministic — same record always serialises to the same bytes so two
    validators verifying the same record agree on what was signed.
  * Forward compatible — `record_from_json` silently drops unknown fields so a
    future schema can add columns without breaking older readers.
  * Self-contained — no network calls, no Hippius, no chain. Just data + crypto.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from typing import Any

import bittensor as bt

logger = logging.getLogger(__name__)


SCHEMA_VERSION = "radar.substrate.v1"
SIGNATURE_PREFIX = "sr25519:"


@dataclass(frozen=True)
class PhaseCRecord:
    """A single Phase C evaluation claim, signed by the validator.

    `signature` is computed last by `sign_record` and intentionally excluded
    from `canonical_bytes` so that signing and verification compare like with
    like. Callers should treat instances as immutable: use
    `dataclasses.replace(record, ...)` to derive a modified copy.
    """

    # Schema
    schema_version: str

    # Round identity
    round_id: int
    block_hash: str
    task: str

    # Miner
    miner_uid: int
    miner_hotkey: str
    code_hash: str
    architecture_sha256: str
    checkpoint_sha256: str

    # Measurement
    metrics: dict
    passed_size_gate: bool
    flops_verified: bool
    eval_status: str

    # Validator
    validator_uid: int
    validator_hotkey: str
    validator_block_height: int
    timestamp: float

    # Signature (computed last; sr25519:<hex>)
    signature: str = ""


_KNOWN_FIELDS = frozenset(f.name for f in fields(PhaseCRecord))


def _filter_unknown(payload: dict) -> dict:
    """Drop keys not declared on PhaseCRecord (forward-compat)."""
    return {k: v for k, v in payload.items() if k in _KNOWN_FIELDS}


def canonical_bytes(record: PhaseCRecord) -> bytes:
    """Return the deterministic byte string that gets signed and verified.

    Excludes the `signature` field so a signature can be embedded in the
    record without changing the bytes used to verify it. JSON is emitted
    with sorted keys and the most compact separator pair so two callers
    (potentially on different platforms) produce byte-identical output.
    """
    payload = asdict(record)
    payload.pop("signature", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sign_record(record: PhaseCRecord, wallet: "bt.Wallet") -> PhaseCRecord:
    """Sign a record with the wallet hotkey and return a new frozen copy.

    Uses sr25519 via `wallet.hotkey.sign()`. The returned record carries
    the signature in `sr25519:<hex>` form; the original is left untouched.
    Raises ValueError if the record's `validator_hotkey` doesn't match the
    wallet — silently signing under the wrong hotkey would let a record
    pretend to come from another validator.
    """
    if record.validator_hotkey and record.validator_hotkey != wallet.hotkey.ss58_address:
        raise ValueError(
            f"validator_hotkey mismatch: record claims "
            f"{record.validator_hotkey!r}, wallet is "
            f"{wallet.hotkey.ss58_address!r}"
        )
    raw = wallet.hotkey.sign(canonical_bytes(record))
    return dataclasses.replace(record, signature=f"{SIGNATURE_PREFIX}{raw.hex()}")


def verify_record(
    record: PhaseCRecord,
    expected_hotkey: str | None = None,
) -> tuple[bool, str]:
    """Verify a record's sr25519 signature against its canonical bytes.

    Returns `(ok, error_message)`. On success `error_message` is empty. If
    `expected_hotkey` is supplied, the record's `validator_hotkey` must
    match it (caller-supplied identity check, on top of the cryptographic
    check). The function never raises on bad input — malformed records
    return `(False, "...")` so callers can log and reject without a
    try/except wrapper.
    """
    if record.schema_version != SCHEMA_VERSION:
        return False, (
            f"Unsupported schema_version {record.schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if not record.signature.startswith(SIGNATURE_PREFIX):
        return False, (
            f"Signature missing required {SIGNATURE_PREFIX!r} prefix"
        )
    if expected_hotkey and record.validator_hotkey != expected_hotkey:
        return False, (
            f"validator_hotkey mismatch: record has "
            f"{record.validator_hotkey!r}, expected {expected_hotkey!r}"
        )
    hex_sig = record.signature[len(SIGNATURE_PREFIX):]
    try:
        sig_bytes = bytes.fromhex(hex_sig)
    except ValueError:
        return False, "Signature is not valid hex"
    if not record.validator_hotkey:
        return False, "validator_hotkey is empty"
    try:
        keypair = bt.Keypair(ss58_address=record.validator_hotkey)
        if not keypair.verify(canonical_bytes(record), sig_bytes):
            return False, "sr25519 signature does not match canonical bytes"
    except Exception as e:  # bt.Keypair raises various Substrate errors
        return False, f"Signature verification failed: {e}"
    return True, ""


def record_to_json(record: PhaseCRecord) -> str:
    """Serialise a record (including signature) to deterministic JSON."""
    return json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))


def record_from_json(s: str) -> PhaseCRecord:
    """Parse a record from JSON, dropping unknown fields for forward compat."""
    payload = json.loads(s)
    if not isinstance(payload, dict):
        raise ValueError("PhaseCRecord JSON must be an object")
    return PhaseCRecord(**_filter_unknown(payload))


def records_to_bundle(records: list[PhaseCRecord]) -> bytes:
    """Pack a list of records into a gzip-compressed JSON bundle.

    Bundle shape: ``{"schema_version", "count", "records"}``. The bundle
    schema is the same as the per-record schema so a future bump applies
    to both at once. Output is always gzipped — `records_from_bundle`
    accepts plain JSON for human/debug use, but the canonical write path
    is always compressed.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "count": len(records),
        "records": [asdict(r) for r in records],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return gzip.compress(raw)


def records_from_bundle(data: bytes) -> list[PhaseCRecord]:
    """Parse a bundle produced by `records_to_bundle`.

    Accepts either gzip-compressed bytes (the canonical form) or plain
    JSON bytes (debug / hand-rolled). Unknown record fields are silently
    dropped for forward compatibility.
    """
    try:
        raw = gzip.decompress(data)
    except (OSError, gzip.BadGzipFile):
        raw = data
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Bundle JSON must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bundle schema_version "
            f"{payload.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("Bundle 'records' must be a list")
    return [PhaseCRecord(**_filter_unknown(r)) for r in raw_records]


def bundle_sha256(data: bytes) -> str:
    """Return ``sha256:<hex>`` digest of arbitrary bundle bytes."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


__all__ = [
    "SCHEMA_VERSION",
    "SIGNATURE_PREFIX",
    "PhaseCRecord",
    "canonical_bytes",
    "sign_record",
    "verify_record",
    "record_to_json",
    "record_from_json",
    "records_to_bundle",
    "records_from_bundle",
    "bundle_sha256",
]
