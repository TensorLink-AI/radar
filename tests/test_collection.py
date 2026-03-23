"""Tests for validator.collection — agent pod execution + R2 proposal sharing."""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.commitment import ImageCommitment
from shared.protocol import Proposal
from validator.collection import run_and_collect_agents


def _mock_commitments():
    return {
        0: ImageCommitment(image_url="agent:v1", image_digest="sha256:abc", miner_uid=0),
        1: ImageCommitment(image_url="agent:v2", image_digest="sha256:def", miner_uid=1),
    }


def _identity_assignments(all_uids, validator_uids, my_uid, seed):
    """All UIDs assigned to the single validator."""
    return list(all_uids)


@pytest.mark.asyncio
async def test_collect_empty_commitments():
    """No commitments -> no proposals."""
    mock_r2 = MagicMock()
    proposals, agent_logs = await run_and_collect_agents(
        wallet=MagicMock(),
        metagraph=MagicMock(),
        challenge_json='{"round_id": 1}',
        round_id=1,
        seed=42,
        r2=mock_r2,
        my_uid=0,
        validator_uids=[0],
        commitments={},
        get_my_assignments_fn=_identity_assignments,
    )
    assert proposals == {}
    assert agent_logs == {}


@pytest.mark.asyncio
async def test_dedup_removes_identical_code():
    """Proposals with identical code should be deduped."""
    code = "def build_model(): pass\ndef build_optimizer(): pass"
    mock_r2 = MagicMock()
    # Simulate R2 returning two proposals with same code
    mock_r2.download_json.side_effect = lambda key: {
        "code": code, "name": "dup", "motivation": "same",
    }

    commitments = {
        0: ImageCommitment(image_url="", miner_uid=0),  # no image -> skip
        1: ImageCommitment(image_url="", miner_uid=1),
    }

    with patch("validator.collection.pull_and_verify_image", return_value=True):
        proposals, agent_logs = await run_and_collect_agents(
            wallet=MagicMock(),
            metagraph=MagicMock(),
            challenge_json='{"round_id": 1}',
            round_id=1,
            seed=42,
            r2=mock_r2,
            my_uid=0,
            validator_uids=[0],
            commitments=commitments,
            get_my_assignments_fn=_identity_assignments,
        )

    # Both have no image_url so neither runs locally;
    # both read from R2 with same code -> dedup to 1
    assert len(proposals) == 1


@pytest.mark.asyncio
async def test_r2_proposal_read_on_other_validator_submissions():
    """Proposals uploaded by other validators are read from R2."""
    mock_r2 = MagicMock()

    # UID 0 not assigned to us, UID 1 not assigned to us
    def assign_none(all_uids, validator_uids, my_uid, seed):
        return []

    mock_r2.download_json.side_effect = lambda key: {
        "code": f"code_{key}", "name": "from_r2", "motivation": "other vali",
    }

    commitments = _mock_commitments()

    proposals, agent_logs = await run_and_collect_agents(
        wallet=MagicMock(),
        metagraph=MagicMock(),
        challenge_json='{"round_id": 1}',
        round_id=1,
        seed=42,
        r2=mock_r2,
        my_uid=10,
        validator_uids=[10, 11],
        commitments=commitments,
        get_my_assignments_fn=assign_none,
    )

    # Should have read both from R2
    assert len(proposals) == 2
    assert 0 in proposals
    assert 1 in proposals
