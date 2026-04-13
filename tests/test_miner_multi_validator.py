"""Tests for miner handling TrainerRequests from multiple validators."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miner.neuron import Miner
from shared.protocol import TrainerRequest


def _make_miner():
    """Create a Miner with mocked bittensor components."""
    with patch("miner.neuron.bt") as mock_bt:
        mock_bt.Wallet.return_value = MagicMock(
            hotkey=MagicMock(ss58_address="5FakeHotkey")
        )
        mock_bt.Subtensor.return_value = MagicMock()
        mock_bt.Subtensor.return_value.metagraph.return_value = MagicMock(n=5)
        mock_bt.Config = dict

        cfg = SimpleNamespace(
            netuid=1,
            agent_dir="agent/",
            trainer_image="test:latest",
            listener_port=8090,
            external_ip="127.0.0.1",
        )
        with patch.object(Miner, "__init__", lambda self, _: None):
            m = Miner(cfg)
        m.wallet = mock_bt.Wallet.return_value
        m.active_deployments = {}
        m._pending_notify = {}
        m.trainer_image = "test:latest"
        m.netuid = 1
        m.subtensor = mock_bt.Subtensor.return_value
        m.metagraph = mock_bt.Subtensor.return_value.metagraph.return_value
        return m


class TestMultiValidatorTrainerReady:
    """The miner must send TrainerReady to every validator that requests it."""

    @pytest.mark.asyncio
    async def test_second_request_same_round_sends_ready(self):
        """When a pod is already deployed and a second validator sends
        TrainerRequest for the same round, the miner must unpack the
        3-tuple (deployment, created_at, ttl) correctly and send
        TrainerReady to the second validator."""
        miner = _make_miner()

        # Simulate a deployment already stored from the first validator
        mock_deployment = SimpleNamespace(
            url="https://pod-1.basilica.ai",
            name="pod-1",
        )
        round_id = 12345
        miner.active_deployments[round_id] = (mock_deployment, 1000.0, 1800)

        second_request = TrainerRequest(
            round_id=round_id,
            challenge_id="test",
            seed=42,
            validator_db_url="http://validator-B:8080",
        )

        with patch.object(miner, "_post_trainer_ready", new_callable=AsyncMock) as mock_post:
            with patch.object(miner, "_teardown_prior_rounds", new_callable=AsyncMock):
                await miner.handle_prepare(second_request)

            # Must have notified the second validator
            mock_post.assert_called_once_with(
                round_id,
                "https://pod-1.basilica.ai",
                "pod-1",
                "http://validator-B:8080",
            )

    @pytest.mark.asyncio
    async def test_pending_deploy_queues_notification(self):
        """When deployment is in progress ('pending'), the second
        validator's URL is queued and notified after deploy completes."""
        miner = _make_miner()
        round_id = 12345
        miner.active_deployments[round_id] = "pending"
        miner._pending_notify[round_id] = []

        second_request = TrainerRequest(
            round_id=round_id,
            challenge_id="test",
            seed=42,
            validator_db_url="http://validator-B:8080",
        )

        with patch.object(miner, "_teardown_prior_rounds", new_callable=AsyncMock):
            await miner.handle_prepare(second_request)

        assert "http://validator-B:8080" in miner._pending_notify[round_id]
