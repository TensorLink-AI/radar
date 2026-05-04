"""Tests for validator/trainer_verify.py.

Mocks ``targon_client`` so we can drive each branch independently.
The coordinator-level wiring is exercised in
``tests/test_coordinator.py`` (existing) — here we focus on the four
checks composing correctly and on the deterministic reverify offsets.
"""

from __future__ import annotations

import asyncio
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
    """End-to-end verify_trainer with a real (signed) boot proof."""
    try:
        import bittensor as bt
    except ImportError:
        pytest.skip("bittensor not installed")

    kp = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    from validator.trainer_verify import _canonical_json
    proof = {"hashes_root_sha256": "deadbeef", "bootstrap_version": "1"}
    sig = kp.sign(_canonical_json(proof)).hex()
    boot_proof = {"proof": proof, "signature": sig, "signer_hotkey": kp.ss58_address}

    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=True)
    targon.verify_attestation = AsyncMock(return_value=_ok_attest())

    with patch("validator.trainer_verify.fetch_boot_proof",
               AsyncMock(return_value=boot_proof)):
        result = await verify_trainer(
            ready=_ready(),
            miner_hotkey=kp.ss58_address,
            expected_image_digest="sha256:digest_abc",
            trainer_url="https://wl.targon.network",
            targon_client=targon,
            wallet=_wallet(),
            require_boot_proof=True,
        )

    assert result.ok is True, result.reason
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
    try:
        import bittensor as bt
    except ImportError:
        pytest.skip("bittensor not installed")
    kp = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    from validator.trainer_verify import _canonical_json
    proof = {"hashes_root_sha256": "x"}
    sig = kp.sign(_canonical_json(proof)).hex()
    boot_proof = {"proof": proof, "signature": sig, "signer_hotkey": kp.ss58_address}

    targon = MagicMock()
    targon.verify_image_digest = AsyncMock(return_value=True)
    targon.verify_attestation = AsyncMock()

    with patch("validator.trainer_verify.fetch_boot_proof", AsyncMock(return_value=boot_proof)):
        result = await reverify_workload(
            ready=_ready(),
            expected_image_digest="sha256:digest_abc",
            trainer_url="https://wl",
            targon_client=targon,
            require_boot_proof=True,
            expected_signer_hotkey=kp.ss58_address,
        )
    assert result.ok is True, result.reason
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
        ok, reason = check_boot_proof(None)
        assert not ok
        assert reason == "missing"

    def test_check_boot_proof_no_signature(self):
        ok, reason = check_boot_proof({"proof": {}})
        assert not ok
        assert "signature" in reason

    def test_check_boot_proof_rejects_garbage_signature(self):
        # A non-empty signature string is no longer enough — it must verify
        # cryptographically. This was the audit gap that turned the prior
        # check into a no-op.
        ok, reason = check_boot_proof(
            {"proof": {"hashes_root_sha256": "abc"},
             "signature": "deadbeef" * 8,
             "signer_hotkey": "5stub"},
        )
        assert not ok
        assert reason

    def test_check_boot_proof_signer_mismatch(self):
        ok, reason = check_boot_proof(
            {"proof": {}, "signature": "ab", "signer_hotkey": "5wrong"},
            expected_signer_hotkey="5expected",
        )
        assert not ok
        assert "signer mismatch" in reason

    def test_check_boot_proof_signature_not_hex(self):
        ok, reason = check_boot_proof(
            {"proof": {}, "signature": "not-hex!!", "signer_hotkey": "5x"},
        )
        assert not ok
        assert "hex" in reason

    def test_check_boot_proof_real_signature_passes(self):
        """Sign a real proof with a real keypair, confirm verification accepts it."""
        try:
            import bittensor as bt
        except ImportError:
            pytest.skip("bittensor not installed")
        kp = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
        proof = {"hashes_root_sha256": "abc123", "bootstrap_version": "1"}
        from validator.trainer_verify import _canonical_json
        sig = kp.sign(_canonical_json(proof)).hex()
        ok, reason = check_boot_proof(
            {"proof": proof, "signature": sig, "signer_hotkey": kp.ss58_address},
            expected_signer_hotkey=kp.ss58_address,
            expected_hashes_root="abc123",
        )
        assert ok, reason

    def test_check_boot_proof_hashes_root_mismatch(self):
        try:
            import bittensor as bt
        except ImportError:
            pytest.skip("bittensor not installed")
        kp = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
        proof = {"hashes_root_sha256": "actual"}
        from validator.trainer_verify import _canonical_json
        sig = kp.sign(_canonical_json(proof)).hex()
        ok, reason = check_boot_proof(
            {"proof": proof, "signature": sig, "signer_hotkey": kp.ss58_address},
            expected_signer_hotkey=kp.ss58_address,
            expected_hashes_root="expected",
        )
        assert not ok
        assert "hashes_root mismatch" in reason

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


# ── schedule_mid_round_reverify ─────────────────────────────────────


@pytest.mark.asyncio
async def test_schedule_reverify_noop_on_basilica(monkeypatch):
    """No reverify tasks scheduled for Basilica deployments."""
    from validator import coordinator as coord_mod
    monkeypatch.setattr(coord_mod.Config, "HOSTING_BACKEND", "basilica")
    coord = coord_mod.TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(),
        r2=MagicMock(), my_uid=0,
    )
    coord._ready_msgs[1] = {5: _ready()}
    spawned = coord.schedule_mid_round_reverify(
        round_id=1, block_hash="abcd", training_window_seconds=10.0,
    )
    assert spawned == 0


@pytest.mark.asyncio
async def test_schedule_reverify_skips_basilica_uids_in_targon_mode(monkeypatch):
    from validator import coordinator as coord_mod
    monkeypatch.setattr(coord_mod.Config, "HOSTING_BACKEND", "targon")
    coord = coord_mod.TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(),
        r2=MagicMock(), my_uid=0,
    )
    # One Targon miner, one Basilica miner (no targon_workload_uid).
    targon_ready = _ready(targon_workload_uid="wl_1")
    basilica_ready = TrainerReady(
        round_id=1, trainer_url="https://b", instance_name="i",
        miner_hotkey="hk", targon_workload_uid="",
    )
    coord._ready_msgs[1] = {5: targon_ready, 6: basilica_ready}

    # Stub reverify_running so the scheduled task completes instantly.
    async def _fake_reverify(rid, uid, rmsg, bh):
        return True
    coord.reverify_running = _fake_reverify

    spawned = coord.schedule_mid_round_reverify(
        round_id=1, block_hash="abcd",
        training_window_seconds=0.05, n_checkpoints=1,
    )
    assert spawned == 1  # only the Targon miner
    await coord.cancel_mid_round_reverify(1)


@pytest.mark.asyncio
async def test_schedule_reverify_runs_at_offsets_and_marks_compromise(monkeypatch):
    """End-to-end: tasks fire at offsets and a digest mismatch records compromise."""
    from validator import coordinator as coord_mod
    monkeypatch.setattr(coord_mod.Config, "HOSTING_BACKEND", "targon")
    monkeypatch.setattr(coord_mod.Config, "REQUIRE_BOOT_PROOF", False)
    monkeypatch.setattr(coord_mod.Config, "OFFICIAL_TRAINING_IMAGE_DIGEST", "sha256:expected")

    coord = coord_mod.TrainingCoordinator(
        wallet=MagicMock(hotkey=MagicMock(ss58_address="vh")),
        metagraph=MagicMock(hotkeys=["", "", "", "", "", "mh"]),
        r2=MagicMock(), my_uid=0,
    )
    coord._ready_msgs[1] = {5: _ready(targon_workload_uid="wl_x")}

    # Mock the targon client's verify so reverify_workload reports mismatch.
    fake_targon = MagicMock()
    fake_targon.verify_image_digest = AsyncMock(return_value=False)
    coord._targon_client = fake_targon

    # Use a tiny window so the task fires fast.
    coord.schedule_mid_round_reverify(
        round_id=1, block_hash="abcdef0123456789",
        training_window_seconds=0.05, n_checkpoints=1,
    )
    # Wait for tasks to complete.
    tasks = coord._reverify_tasks.get(1, [])
    for t in tasks:
        await t

    assert coord._compromised.get(1, {}).get(5) is True


@pytest.mark.asyncio
async def test_cancel_reverify_stops_in_flight_tasks(monkeypatch):
    from validator import coordinator as coord_mod
    monkeypatch.setattr(coord_mod.Config, "HOSTING_BACKEND", "targon")
    coord = coord_mod.TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["", "", "", "", "", "mh"]),
        r2=MagicMock(), my_uid=0,
    )
    coord._ready_msgs[1] = {5: _ready(targon_workload_uid="wl")}

    # reverify_running blocks long enough for cancel to land.
    async def _slow(rid, uid, rmsg, bh):
        await asyncio.sleep(60)
        return True
    coord.reverify_running = _slow

    coord.schedule_mid_round_reverify(
        round_id=1, block_hash="aa",
        training_window_seconds=10.0, n_checkpoints=2,
    )
    await asyncio.sleep(0.01)
    await coord.cancel_mid_round_reverify(1)

    # All tasks should be done (cancelled).
    tasks = coord._reverify_tasks.get(1, [])
    assert all(t.done() for t in tasks)
    assert 1 not in coord._reverify_tasks  # cleared
