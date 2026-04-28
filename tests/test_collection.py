"""Tests for validator.collection — agent pod execution + R2 proposal sharing."""

import hashlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.commitment import ImageCommitment
from shared.protocol import Proposal
from validator.collection import run_and_collect_agents, _run_single_agent


def _mock_commitments():
    return {
        0: ImageCommitment(code_hash="sha256:abc", miner_uid=0, hotkey="hk0"),
        1: ImageCommitment(code_hash="sha256:def", miner_uid=1, hotkey="hk1"),
    }


def _identity_assignments(all_uids, validator_uids, my_uid, seed):
    """All UIDs assigned to the single validator."""
    return list(all_uids)


def _mock_db_client():
    """DB client that returns no agent code (forces R2 fallback path)."""
    client = AsyncMock()
    client.get_agent_code = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_collect_empty_commitments():
    """No commitments -> no proposals."""
    mock_r2 = MagicMock()
    proposals, agent_meta = await run_and_collect_agents(
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
        db_client=_mock_db_client(),
    )
    assert proposals == {}
    assert agent_meta == {}


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
        0: ImageCommitment(miner_uid=0, hotkey="hk0"),
        1: ImageCommitment(miner_uid=1, hotkey="hk1"),
    }

    proposals, agent_meta = await run_and_collect_agents(
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
        db_client=_mock_db_client(),
    )

    # Both have no agent code from DB, both read from R2 with same code -> dedup to 1
    assert len(proposals) == 1


@pytest.mark.asyncio
async def test_r2_proposal_read_on_other_validator_submissions():
    """Proposals uploaded by other validators are read from R2."""
    mock_r2 = MagicMock()

    # UID 0 not assigned to us, UID 1 not assigned to us
    def assign_none(all_uids, validator_uids, my_uid, seed):
        return []

    mock_r2.download_json.side_effect = lambda key: {
        "code": f"code_{key}",
        "name": "from_r2",
        "motivation": "other vali",
        "reasoning": "shared via R2",
        "tool_calls": [{"tool": "llm", "summary": "asked for ideas"}],
        "agent_log": "log line",
        "agent_behavior": {"wall_clock_s": 12.5,
                            "proxy": {"calls": {"llm": 1}}},
    }

    commitments = _mock_commitments()

    proposals, agent_meta = await run_and_collect_agents(
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
        db_client=_mock_db_client(),
    )

    # Should have read both from R2
    assert len(proposals) == 2
    assert 0 in proposals
    assert 1 in proposals
    # Self-reported metadata is propagated from R2 onto the proposal + meta
    assert proposals[0].reasoning == "shared via R2"
    assert proposals[0].tool_calls == [
        {"tool": "llm", "summary": "asked for ideas"},
    ]
    assert agent_meta[0]["agent_log"] == "log line"
    assert agent_meta[0]["agent_behavior"]["wall_clock_s"] == 12.5


@pytest.mark.asyncio
async def test_run_single_agent_logs_error_result(caplog):
    """Agent returning an error dict should log the error, not silently drop."""
    commitment = ImageCommitment(miner_uid=0, hotkey="hk0")
    bundle = {"files": {"agent.py": "def design_architecture(c,cl): ..."}, "entry_point": "agent.py"}
    mock_r2 = MagicMock()

    error_result = {"error": "Agent load failed: SyntaxError", "stderr": "trace..."}

    with patch("validator.collection.launch_agent_pod") as mock_launch, \
         patch("validator.collection.run_agent_on_pod", new_callable=AsyncMock) as mock_run:
        mock_env = AsyncMock()
        mock_launch.return_value = mock_env
        mock_run.return_value = error_result

        with caplog.at_level(logging.WARNING, logger="validator.collection"):
            proposal, meta = await _run_single_agent(
                uid=0,
                commitment=commitment,
                challenge_json='{"round_id": 1}',
                r2=mock_r2,
                round_id=1,
                allowed_urls="",
                bundle=bundle,
            )

    assert proposal is None
    # Empty metadata bag, all keys present so neuron.py can lookup safely
    assert meta == {
        "agent_log": "",
        "reasoning": "",
        "tool_calls": [],
        "agent_behavior": {},
    }
    assert "UID 0 proposal rejected (no code returned)" in caplog.text
    assert "SyntaxError" in caplog.text


@pytest.mark.asyncio
async def test_run_single_agent_captures_reasoning_and_behavior():
    """Successful run propagates reasoning, tool_calls, and pod wall-clock."""
    commitment = ImageCommitment(miner_uid=7, hotkey="hk7")
    bundle = {"files": {"agent.py": "..."}, "entry_point": "agent.py"}
    mock_r2 = MagicMock()
    success_result = {
        "code": "def build_model(): ...\ndef build_optimizer(): ...",
        "name": "tx",
        "motivation": "frontier",
        "reasoning": "Picked the smallest preset that fits the bucket.",
        "tool_calls": [
            {"tool": "llm", "summary": "asked for ideas"},
            "stringified",
        ],
        "agent_log": "designing\n",
    }

    with patch("validator.collection.launch_agent_pod") as mock_launch, \
         patch("validator.collection.run_agent_on_pod", new_callable=AsyncMock) as mock_run:
        mock_env = AsyncMock()
        mock_launch.return_value = mock_env
        mock_run.return_value = success_result

        proposal, meta = await _run_single_agent(
            uid=7,
            commitment=commitment,
            challenge_json='{"round_id": 1}',
            r2=mock_r2,
            round_id=1,
            allowed_urls="",
            bundle=bundle,
        )

    assert proposal is not None
    assert proposal.reasoning == "Picked the smallest preset that fits the bucket."
    # _normalise_tool_calls coerces non-dict entries to {"value": ...}
    assert proposal.tool_calls == [
        {"tool": "llm", "summary": "asked for ideas"},
        {"value": "stringified"},
    ]
    # _truncate_text only trims at the byte-cap, so trailing whitespace
    # passes through unchanged (the env wrapper strips it before sending).
    assert meta["agent_log"] == "designing\n"
    # wall_clock_s is measured by the validator (>= 0)
    assert "wall_clock_s" in meta["agent_behavior"]
    assert meta["agent_behavior"]["wall_clock_s"] >= 0
    assert meta["agent_behavior"]["exit_status"] == "ok"

    # R2 upload payload includes the new fields so other validators get them
    upload_args = mock_r2.upload_json.call_args
    payload = upload_args[0][1] if upload_args else {}
    assert payload.get("reasoning") == proposal.reasoning
    assert payload.get("tool_calls") == proposal.tool_calls
    assert "agent_behavior" in payload
