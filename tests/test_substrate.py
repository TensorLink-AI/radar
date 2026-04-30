"""Tests for shared/substrate.py — Phase C signed-record contract."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib

import bittensor as bt
import pytest

from shared.substrate import (
    SCHEMA_VERSION,
    SIGNATURE_PREFIX,
    PhaseCRecord,
    bundle_sha256,
    canonical_bytes,
    record_from_json,
    record_to_json,
    records_from_bundle,
    records_to_bundle,
    sign_record,
    verify_record,
)


# ── Test fixtures ────────────────────────────────────────────────────


def _seed_hex(label: str) -> str:
    """Deterministic 32-byte seed → 0x-prefixed hex string."""
    return "0x" + hashlib.sha256(label.encode()).hexdigest()


def _keypair(label: str) -> bt.Keypair:
    """Make a deterministic Keypair for tests."""
    return bt.Keypair.create_from_seed(_seed_hex(label))


class _FakeWallet:
    """Just enough surface area for sign_record (wallet.hotkey)."""

    def __init__(self, keypair: bt.Keypair):
        self.hotkey = keypair


@pytest.fixture
def wallet_a() -> _FakeWallet:
    return _FakeWallet(_keypair("validator-a"))


@pytest.fixture
def wallet_b() -> _FakeWallet:
    return _FakeWallet(_keypair("validator-b"))


def _make_record(wallet: _FakeWallet | None = None, **overrides) -> PhaseCRecord:
    """Build a populated PhaseCRecord with sensible defaults."""
    base = dict(
        schema_version=SCHEMA_VERSION,
        round_id=42,
        block_hash="0xabc123",
        task="ts_forecasting",
        miner_uid=7,
        miner_hotkey="5MinerHotkeyExample",
        code_hash="sha256:deadbeef",
        architecture_sha256="sha256:11" * 1,
        checkpoint_sha256="sha256:22",
        metrics={"crps": 0.123, "mase": 0.456, "size": 1_234_567},
        passed_size_gate=True,
        flops_verified=True,
        eval_status="ok",
        validator_uid=3,
        validator_hotkey=(
            wallet.hotkey.ss58_address if wallet else "5ValidatorHotkeyExample"
        ),
        validator_block_height=100_000,
        timestamp=1_700_000_000.0,
        signature="",
    )
    base.update(overrides)
    return PhaseCRecord(**base)


# ── canonical_bytes ───────────────────────────────────────────────────


def test_canonical_bytes_deterministic(wallet_a):
    record = _make_record(wallet_a)
    a = canonical_bytes(record)
    b = canonical_bytes(record)
    assert a == b
    # Independent reconstruction with the same inputs produces the same bytes.
    record2 = _make_record(wallet_a)
    assert canonical_bytes(record2) == a


def test_canonical_bytes_excludes_signature(wallet_a):
    unsigned = _make_record(wallet_a)
    signed = sign_record(unsigned, wallet_a)
    # Both pre- and post-signing canonical bytes must match — otherwise the
    # signature would change the hash it was computed over.
    assert canonical_bytes(unsigned) == canonical_bytes(signed)
    assert b"signature" not in canonical_bytes(signed)


# ── sign_record / verify_record ──────────────────────────────────────


def test_sign_verify_roundtrip(wallet_a):
    record = _make_record(wallet_a)
    signed = sign_record(record, wallet_a)
    assert signed.signature.startswith(SIGNATURE_PREFIX)
    assert len(bytes.fromhex(signed.signature[len(SIGNATURE_PREFIX):])) == 64
    ok, err = verify_record(signed)
    assert ok, err
    assert err == ""

    # Explicit expected_hotkey check
    ok, err = verify_record(signed, expected_hotkey=wallet_a.hotkey.ss58_address)
    assert ok, err


def test_verify_wrong_hotkey(wallet_a, wallet_b):
    """Signed with A but the record claims hotkey B → verification fails."""
    # Record claims wallet_b but is signed by wallet_a's keypair. We bypass
    # sign_record's hotkey-mismatch guard by signing manually.
    record = _make_record(wallet_b)  # validator_hotkey == B
    raw_sig = wallet_a.hotkey.sign(canonical_bytes(record))
    forged = dataclasses.replace(
        record, signature=f"{SIGNATURE_PREFIX}{raw_sig.hex()}"
    )
    ok, err = verify_record(forged)
    assert not ok
    assert "signature" in err.lower()

    # And the explicit expected_hotkey path complains about identity.
    signed_a = sign_record(_make_record(wallet_a), wallet_a)
    ok, err = verify_record(
        signed_a, expected_hotkey=wallet_b.hotkey.ss58_address,
    )
    assert not ok
    assert "validator_hotkey mismatch" in err


def test_verify_tampered_record(wallet_a):
    signed = sign_record(_make_record(wallet_a), wallet_a)
    tampered = dataclasses.replace(signed, metrics={"crps": 0.001})
    ok, err = verify_record(tampered)
    assert not ok
    assert "signature" in err.lower()


def test_verify_bad_signature(wallet_a):
    record = _make_record(wallet_a)
    garbage = dataclasses.replace(record, signature=f"{SIGNATURE_PREFIX}deadbeef")
    ok, err = verify_record(garbage)
    assert not ok

    not_hex = dataclasses.replace(record, signature=f"{SIGNATURE_PREFIX}zzzz")
    ok, err = verify_record(not_hex)
    assert not ok
    assert "hex" in err.lower()

    no_prefix = dataclasses.replace(record, signature="abc123")
    ok, err = verify_record(no_prefix)
    assert not ok
    assert SIGNATURE_PREFIX in err


def test_verify_bad_schema_version(wallet_a):
    signed = sign_record(_make_record(wallet_a), wallet_a)
    bumped = dataclasses.replace(signed, schema_version="radar.substrate.v2")
    ok, err = verify_record(bumped)
    assert not ok
    assert "schema_version" in err


def test_sign_record_rejects_hotkey_mismatch(wallet_a, wallet_b):
    """Refuse to sign a record whose validator_hotkey is someone else's."""
    record = _make_record(wallet_b)  # claims wallet_b
    with pytest.raises(ValueError, match="validator_hotkey mismatch"):
        sign_record(record, wallet_a)  # but we hold wallet_a


# ── JSON roundtrip ───────────────────────────────────────────────────


def test_record_to_from_json(wallet_a):
    signed = sign_record(_make_record(wallet_a), wallet_a)
    s = record_to_json(signed)
    parsed = record_from_json(s)
    assert parsed == signed
    # And it still verifies after the JSON trip.
    ok, err = verify_record(parsed)
    assert ok, err


def test_record_from_json_drops_unknown_fields(wallet_a):
    signed = sign_record(_make_record(wallet_a), wallet_a)
    s = record_to_json(signed)
    # Splice in a fake forward-compat field — older parser must ignore it.
    spliced = s[:-1] + ',"future_field":"surprise"}'
    parsed = record_from_json(spliced)
    assert parsed == signed


# ── Bundle ────────────────────────────────────────────────────────────


def test_bundle_roundtrip(wallet_a):
    records = [
        sign_record(
            _make_record(wallet_a, round_id=i, miner_uid=i),
            wallet_a,
        )
        for i in range(5)
    ]
    blob = records_to_bundle(records)
    parsed = records_from_bundle(blob)
    assert len(parsed) == 5
    assert parsed == records
    # Each parsed record still verifies cryptographically.
    for r in parsed:
        ok, err = verify_record(r)
        assert ok, err


def test_bundle_is_gzipped(wallet_a):
    records = [sign_record(_make_record(wallet_a), wallet_a)]
    blob = records_to_bundle(records)
    # Gzip magic number
    assert blob[:2] == b"\x1f\x8b"
    # gzip.decompress() succeeds and yields valid JSON.
    decompressed = gzip.decompress(blob)
    assert decompressed.startswith(b"{")


def test_bundle_accepts_plain_json(wallet_a):
    """records_from_bundle should also accept uncompressed JSON for debugging."""
    records = [sign_record(_make_record(wallet_a), wallet_a)]
    gz = records_to_bundle(records)
    plain = gzip.decompress(gz)
    parsed = records_from_bundle(plain)
    assert parsed == records


def test_bundle_sha256_deterministic(wallet_a):
    records = [
        sign_record(_make_record(wallet_a, round_id=i), wallet_a)
        for i in range(3)
    ]
    a = bundle_sha256(records_to_bundle(records))
    b = bundle_sha256(records_to_bundle(records))
    assert a == b
    assert a.startswith("sha256:")
    assert len(a) == len("sha256:") + 64


def test_bundle_rejects_wrong_schema_version(wallet_a):
    records = [sign_record(_make_record(wallet_a), wallet_a)]
    blob = records_to_bundle(records)
    # Forge a bundle whose top-level schema_version doesn't match.
    import json
    payload = json.loads(gzip.decompress(blob))
    payload["schema_version"] = "radar.substrate.v999"
    forged = gzip.compress(json.dumps(payload).encode())
    with pytest.raises(ValueError, match="schema_version"):
        records_from_bundle(forged)
