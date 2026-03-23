"""Tests for validator.coordinator — job assignment and dispatch."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.protocol import Proposal
from validator.coordinator import (
    compute_assignments, compute_fallback, Job, TrainingCoordinator,
)


def test_deterministic_assignments():
    """Same inputs -> same assignments."""
    submissions = {0: Proposal(code="a"), 1: Proposal(code="b")}
    a1 = compute_assignments("a" * 64, submissions, [0, 1, 2], [10, 11], round_id=1)
    a2 = compute_assignments("a" * 64, submissions, [0, 1, 2], [10, 11], round_id=1)
    assert len(a1) == len(a2)
    for j1, j2 in zip(a1, a2):
        assert j1.arch_owner == j2.arch_owner
        assert j1.trainer_uid == j2.trainer_uid
        assert j1.dispatcher == j2.dispatcher


def test_cross_eval_invariant():
    """arch_owner != trainer_uid (cross-eval)."""
    submissions = {0: Proposal(code="a"), 1: Proposal(code="b"), 2: Proposal(code="c")}
    jobs = compute_assignments("a" * 64, submissions, [0, 1, 2], [10, 11], round_id=1)
    for job in jobs:
        assert job.arch_owner != job.trainer_uid


def test_single_miner_self_train():
    """With only one miner, self-training is allowed."""
    submissions = {0: Proposal(code="a")}
    jobs = compute_assignments("a" * 64, submissions, [0], [10], round_id=1)
    assert len(jobs) == 1
    # With only one miner, trainer_uid must be 0 (self)
    assert jobs[0].trainer_uid == 0


def test_dispatch_round_robin():
    """Jobs are distributed across validators."""
    submissions = {i: Proposal(code=f"code_{i}") for i in range(6)}
    jobs = compute_assignments("a" * 64, submissions, list(range(6)), [10, 11, 12], round_id=1)
    dispatchers = {j.dispatcher for j in jobs}
    assert len(dispatchers) > 1  # At least 2 validators get work


def test_empty_submissions():
    jobs = compute_assignments("a" * 64, {}, [0, 1], [10], round_id=1)
    assert jobs == []


def test_fallback_reassignment():
    """Missing validators' jobs get reassigned."""
    jobs = [
        Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1),
        Job(arch_owner=1, trainer_uid=0, dispatcher=10, round_id=1),
        Job(arch_owner=2, trainer_uid=0, dispatcher=11, round_id=1),
    ]
    reassigned = compute_fallback("a" * 64, [10], jobs, [11, 12])
    assert len(reassigned) == 2  # 2 jobs from missing validator 10
    for j in reassigned:
        assert j.dispatcher in [11, 12]


# ── Attestation tests ────────────────────────────────────────────


@dataclass
class _FakeCommitment:
    pod_attestation_id: str = ""


@pytest.mark.asyncio
async def test_attestation_failed_skips_dispatch():
    """Jobs sent to trainers that fail attestation get status='attestation_failed'."""
    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(), my_uid=10,
    )
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    challenge = MagicMock(seed=42, round_id=1, min_flops_equivalent=0,
                          max_flops_equivalent=1000000, task={"time_budget": 300})
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}
    commitments = {1: _FakeCommitment(pod_attestation_id="attest-bad")}

    with patch("validator.pod_manager.verify_miner_pod", return_value=(False, "Wrong image")):
        results = await coordinator.dispatch_jobs(
            jobs, challenge, submissions, endpoints, commitments=commitments,
        )

    assert len(results) == 1
    assert results[0].status == "attestation_failed"


@pytest.mark.asyncio
async def test_no_attestation_id_passes():
    """Trainers with no attestation_id committed pass (localnet compatibility)."""
    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(), my_uid=10,
    )
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    challenge = MagicMock(seed=42, round_id=1, min_flops_equivalent=0,
                          max_flops_equivalent=1000000, task={"time_budget": 300})
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}
    commitments = {1: _FakeCommitment(pod_attestation_id="")}  # no attestation

    # Mock the HTTP call to avoid actual network
    with patch("httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "success", "flops_equivalent_size": 500000}
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            post=AsyncMock(return_value=mock_resp),
        ))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await coordinator.dispatch_jobs(
            jobs, challenge, submissions, endpoints, commitments=commitments,
        )

    assert len(results) == 1
    assert results[0].status != "attestation_failed"


@pytest.mark.asyncio
async def test_url_without_commitment_passes():
    """Trainers with a URL but no commitment entry should pass verification."""
    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(), my_uid=10,
    )
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    challenge = MagicMock(seed=42, round_id=1, min_flops_equivalent=0,
                          max_flops_equivalent=1000000, task={"time_budget": 300})
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}
    # No commitment for trainer UID 1
    commitments = {}

    with patch("httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "success", "flops_equivalent_size": 500000}
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            post=AsyncMock(return_value=mock_resp),
        ))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await coordinator.dispatch_jobs(
            jobs, challenge, submissions, endpoints, commitments=commitments,
        )

    assert len(results) == 1
    assert results[0].status == "success"
