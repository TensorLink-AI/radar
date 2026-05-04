"""Tests for verify_trainer_runpod + scoring's non_attested multiplier."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.protocol import TrainerReady
from shared.scoring import apply_round_metadata
from validator.trainer_verify import verify_trainer_runpod


def _ready(**overrides):
    base = dict(
        round_id=1,
        trainer_url="https://p1-8081.proxy.runpod.net",
        miner_hotkey="hk1",
        backend="runpod",
        runpod_pod_id="p1",
        deployed_image_digest="sha256:abc",
    )
    base.update(overrides)
    return TrainerReady(**base)


class TestVerifyTrainerRunpod:
    @pytest.mark.asyncio
    async def test_happy_path_returns_attested_false(self):
        client = MagicMock()
        client.verify_pod_image = AsyncMock(return_value=(True, ""))
        with patch(
            "validator.trainer_verify.fetch_boot_proof",
            AsyncMock(return_value=None),
        ):
            result = await verify_trainer_runpod(
                ready=_ready(),
                miner_hotkey="hk1",
                expected_image_digest="sha256:abc",
                trainer_url="https://p1-8081.proxy.runpod.net",
                runpod_client=client,
                require_boot_proof=False,
            )
        assert result.ok
        # Non-attested backend — caller routes through NON_ATTESTED_SCORE_MULTIPLIER.
        assert result.attested is False

    @pytest.mark.asyncio
    async def test_missing_pod_id_rejected(self):
        client = MagicMock()
        result = await verify_trainer_runpod(
            ready=_ready(runpod_pod_id=""),
            miner_hotkey="hk1",
            expected_image_digest="sha256:abc",
            trainer_url="x",
            runpod_client=client,
        )
        assert not result.ok
        assert "missing runpod_pod_id" in result.reason

    @pytest.mark.asyncio
    async def test_declared_digest_mismatch_short_circuits(self):
        client = MagicMock()
        # verify_pod_image must NOT be called when declared mismatches.
        client.verify_pod_image = AsyncMock(side_effect=AssertionError("should not call"))
        result = await verify_trainer_runpod(
            ready=_ready(deployed_image_digest="sha256:wrong"),
            miner_hotkey="hk1",
            expected_image_digest="sha256:abc",
            trainer_url="x",
            runpod_client=client,
        )
        assert not result.ok
        assert "declared digest" in result.reason

    @pytest.mark.asyncio
    async def test_runpod_verify_failure_propagates(self):
        client = MagicMock()
        client.verify_pod_image = AsyncMock(return_value=(False, "digest mismatch"))
        result = await verify_trainer_runpod(
            ready=_ready(),
            miner_hotkey="hk1",
            expected_image_digest="sha256:abc",
            trainer_url="x",
            runpod_client=client,
        )
        assert not result.ok
        assert "digest mismatch" in result.reason


class TestNonAttestedMultiplier:
    def test_non_attested_uids_get_multiplier(self):
        # Default Config.NON_ATTESTED_SCORE_MULTIPLIER = 0.6.
        scores = {1: 1.0, 2: 1.0, 3: 1.0}
        out = apply_round_metadata(scores, {"non_attested": [1, 3]})
        assert out[1] == pytest.approx(0.6)
        assert out[2] == 1.0
        assert out[3] == pytest.approx(0.6)

    def test_compromised_overrides_non_attested(self):
        scores = {1: 1.0}
        out = apply_round_metadata(
            scores, {"non_attested": [1], "compromised": [1]},
        )
        assert out[1] == 0.0

    def test_non_attested_and_targon_unavailable_compound(self):
        # Hybrid edge case — operator runs both backends, multipliers
        # multiply. With defaults: 1.0 * 0.5 (targon_unavailable) * 0.6 = 0.30.
        scores = {1: 1.0}
        out = apply_round_metadata(
            scores,
            {"targon_unavailable": [1], "non_attested": [1]},
        )
        assert out[1] == pytest.approx(0.5 * 0.6)

    def test_does_not_mutate_input(self):
        scores = {1: 1.0}
        _ = apply_round_metadata(scores, {"non_attested": [1]})
        assert scores[1] == 1.0
