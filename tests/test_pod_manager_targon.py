"""Tests for validator/trainer_verify.py.

Mocks ``targon_client`` so we can drive each branch independently.
The coordinator-level wiring is exercised in
``tests/test_coordinator.py`` (existing) — here we focus on the four
checks composing correctly and on the deterministic reverify offsets.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.protocol import TrainerReady
from shared.targon_attest import AttestationResult
from shared.targon_breaker import TargonUnavailable
from validator.trainer_verify import (
    check_boot_proof,
    fetch_boot_proof,
    reverify_offsets,
    reverify_workload,
    verify_trainer,
)


def _ready(**overrides):
    base = dict(
        round_id=1,
        trainer_url="https://wl.targon.network",
        instance_name="trainer-1",
        miner_hotkey="hk_miner",
        targon_workload_uid="wl_abc",
        cvm_ip="wl.targon.network",
        gpu_class="H200",
        deployed_image_digest="sha256:digest_abc",
    )
    base.update(overrides)
    return TrainerReady(**base)


def _wallet(ss58="vh1"):
    w = MagicMock()
    w.hotkey.ss58_address = ss58
    return w


def _ok_attest(gpu_class="H200"):
    return AttestationResult(verified=True, gpu_class=gpu_class, gpu_count=1)


# ── Happy path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_trainer_all_pass():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=True)
    targon.verify_attestation = AsyncMock(return_value=_ok_attest())

    with patch("validator.trainer_verify.fetch_boot_proof",
               AsyncMock(return_value={"proof": {}, "signature": "abcd"})):
        result = await verify_trainer(
            ready=_ready(),
            miner_hotkey="hk_miner",
            expected_image_digest="sha256:digest_abc",
            trainer_url="https://wl.targon.network",
            targon_client=targon,
            wallet=_wallet(),
            require_boot_proof=True,
        )

    assert result.ok is True
    assert result.gpu_class == "H200"


# ── Failure modes ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_trainer_digest_mismatch():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=False)
    targon.verify_attestation = AsyncMock()

    result = await verify_trainer(
        ready=_ready(),
        miner_hotkey="hk_miner",
        expected_image_digest="sha256:digest_abc",
        trainer_url="https://wl.targon.network",
        targon_client=targon,
        wallet=_wallet(),
    )
    assert result.ok is False
    assert result.targon_unavailable is False
    assert "not running expected digest" in result.reason
    targon.verify_attestation.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_trainer_attestation_failed():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=True)
    targon.verify_attestation = AsyncMock(
        return_value=AttestationResult(verified=False, error="TDX quote signature invalid"),
    )

    result = await verify_trainer(
        ready=_ready(),
        miner_hotkey="hk_miner",
        expected_image_digest="sha256:digest_abc",
        trainer_url="https://wl.targon.network",
        targon_client=targon,
        wallet=_wallet(),
    )
    assert result.ok is False
    assert "TDX quote signature invalid" in result.reason


@pytest.mark.asyncio
async def test_verify_trainer_gpu_class_mismatch():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=True)
    targon.verify_attestation = AsyncMock(return_value=_ok_attest(gpu_class="H100"))

    result = await verify_trainer(
        ready=_ready(gpu_class="H200"),
        miner_hotkey="hk_miner",
        expected_image_digest="sha256:digest_abc",
        trainer_url="https://wl.targon.network",
        targon_client=targon,
        wallet=_wallet(),
    )
    assert result.ok is False
    assert "GPU class mismatch" in result.reason


# ── Boot-proof flag behaviour ───────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_trainer_warns_but_passes_when_flag_off():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=True)
    targon.verify_attestation = AsyncMock(return_value=_ok_attest())

    with patch("validator.trainer_verify.fetch_boot_proof", AsyncMock(return_value=None)):
        result = await verify_trainer(
            ready=_ready(),
            miner_hotkey="hk",
            expected_image_digest="sha256:digest_abc",
            trainer_url="https://wl",
            targon_client=targon,
            wallet=_wallet(),
            require_boot_proof=False,
        )
    assert result.ok is True


@pytest.mark.asyncio
async def test_verify_trainer_fails_hard_when_flag_on_and_proof_missing():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=True)
    targon.verify_attestation = AsyncMock(return_value=_ok_attest())

    with patch("validator.trainer_verify.fetch_boot_proof", AsyncMock(return_value=None)):
        result = await verify_trainer(
            ready=_ready(),
            miner_hotkey="hk",
            expected_image_digest="sha256:digest_abc",
            trainer_url="https://wl",
            targon_client=targon,
            wallet=_wallet(),
            require_boot_proof=True,
        )
    assert result.ok is False
    assert "boot_proof" in result.reason


# ── Soft-fail (Targon outage) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_trainer_marks_targon_unavailable_on_breaker():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(side_effect=TargonUnavailable("breaker open"))

    result = await verify_trainer(
        ready=_ready(),
        miner_hotkey="hk",
        expected_image_digest="sha256:digest_abc",
        trainer_url="https://wl",
        targon_client=targon,
        wallet=_wallet(),
    )
    assert result.ok is False
    assert result.targon_unavailable is True
    assert "breaker" in result.reason


# ── reverify_offsets — deterministic + reproducible ─────────────────


class TestReverifyOffsets:
    def test_offsets_within_window(self):
        offsets = reverify_offsets(
            block_hash="abcdef0123456789", round_id=42, miner_uid=7,
            n=3, window_seconds=1800,
        )
        assert len(offsets) == 3
        assert offsets == sorted(offsets)
        for o in offsets:
            assert 0 < o < 1800

    def test_deterministic(self):
        a = reverify_offsets("aabb", 1, 5, n=3, window_seconds=900)
        b = reverify_offsets("aabb", 1, 5, n=3, window_seconds=900)
        assert a == b

    def test_different_seeds_differ(self):
        a = reverify_offsets("aabb", 1, 5, n=3, window_seconds=900)
        b = reverify_offsets("aabb", 1, 6, n=3, window_seconds=900)
        assert a != b


# ── reverify_workload skips attestation ─────────────────────────────


@pytest.mark.asyncio
async def test_reverify_skips_attestation():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=True)
    targon.verify_attestation = AsyncMock()

    with patch("validator.trainer_verify.fetch_boot_proof", AsyncMock(return_value={"proof": {}, "signature": "x"})):
        result = await reverify_workload(
            ready=_ready(),
            expected_image_digest="sha256:digest_abc",
            trainer_url="https://wl",
            targon_client=targon,
            require_boot_proof=True,
        )
    assert result.ok is True
    targon.verify_attestation.assert_not_awaited()


@pytest.mark.asyncio
async def test_reverify_flags_compromised_on_digest_change():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=False)

    result = await reverify_workload(
        ready=_ready(),
        expected_image_digest="sha256:digest_abc",
        trainer_url="https://wl",
        targon_client=targon,
        require_boot_proof=False,
    )
    assert result.ok is False
    assert "digest mismatch" in result.reason


@pytest.mark.asyncio
async def test_reverify_soft_fails_on_breaker():
    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(side_effect=TargonUnavailable("breaker"))

    result = await reverify_workload(
        ready=_ready(),
        expected_image_digest="sha256:digest_abc",
        trainer_url="https://wl",
        targon_client=targon,
        require_boot_proof=False,
    )
    assert result.ok is False
    assert result.targon_unavailable is True


# ── boot_proof helpers ──────────────────────────────────────────────


class TestBootProofHelpers:
    def test_check_boot_proof_missing(self):
        ok, reason = check_boot_proof(None, "sha256:x")
        assert not ok
        assert reason == "missing"

    def test_check_boot_proof_no_signature(self):
        ok, reason = check_boot_proof({"proof": {}}, "sha256:x")
        assert not ok
        assert "signature" in reason

    def test_check_boot_proof_ok(self):
        ok, reason = check_boot_proof(
            {"proof": {"hashes_root_sha256": "abc"}, "signature": "sig"}, "sha256:x",
        )
        assert ok
        assert reason == ""

    @pytest.mark.asyncio
    async def test_fetch_boot_proof_503_returns_none(self):
        from unittest.mock import AsyncMock as AM
        resp = MagicMock(status_code=503)
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(return_value=resp)

        with patch("validator.trainer_verify.httpx.AsyncClient", return_value=client):
            out = await fetch_boot_proof("https://x", timeout=1.0)
        assert out is None
