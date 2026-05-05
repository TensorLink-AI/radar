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


def test_self_training_allowed():
    """Self-training is now allowed — no cross-eval constraint."""
    submissions = {i: Proposal(code=f"code_{i}") for i in range(10)}
    jobs = compute_assignments("b" * 64, submissions, list(range(10)), [10, 11], round_id=1)
    # With shuffled pool and round-robin, self-training can happen
    # Just verify all jobs are valid
    for job in jobs:
        assert job.arch_owner in submissions
        assert job.trainer_uid in list(range(10))


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


def test_shuffled_assignment_varies_by_round():
    """Different block hashes produce different trainer assignments."""
    submissions = {0: Proposal(code="a"), 1: Proposal(code="b"), 2: Proposal(code="c")}
    miners = [0, 1, 2, 3, 4]
    jobs_a = compute_assignments("a" * 64, submissions, miners, [10], round_id=1)
    jobs_b = compute_assignments("b" * 64, submissions, miners, [10], round_id=1)
    trainers_a = [j.trainer_uid for j in jobs_a]
    trainers_b = [j.trainer_uid for j in jobs_b]
    # Very unlikely to be identical with different seeds
    assert trainers_a != trainers_b or len(submissions) == 1


# ── Dispatch tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_no_endpoint():
    """Jobs with no trainer endpoint get status='failed'."""
    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(), my_uid=10,
    )
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    challenge = MagicMock(seed=42, round_id=1, min_flops_equivalent=0,
                          max_flops_equivalent=1000000, task={"time_budget": 300})
    submissions = {0: Proposal(code="code_a")}
    endpoints = {}  # No endpoints

    results = await coordinator.dispatch_jobs(
        jobs, challenge, submissions, endpoints,
    )

    assert len(results) == 1
    assert results[0].status == "failed"


@pytest.mark.asyncio
async def test_dispatch_success():
    """Successful dispatch returns trainer response."""
    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(), my_uid=10,
    )
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    challenge = MagicMock(seed=42, round_id=1, min_flops_equivalent=0,
                          max_flops_equivalent=1000000, task={"time_budget": 300})
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}

    with patch("httpx.AsyncClient") as mock_client, \
         patch("shared.artifacts.generate_upload_urls", return_value={"checkpoint": "http://fake/ckpt"}), \
         patch("shared.auth.sign_request", return_value={"X-Epistula-Signed-By": "hk0"}):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "success", "flops_equivalent_size": 500000}
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            post=AsyncMock(return_value=mock_resp),
        ))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await coordinator.dispatch_jobs(
            jobs, challenge, submissions, endpoints,
        )

    assert len(results) == 1
    assert results[0].status == "success"


@pytest.mark.asyncio
async def test_dispatch_resigns_on_retry():
    """Headers are re-signed on retry so Epistula timestamps stay fresh."""
    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(), my_uid=10,
    )
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    challenge = MagicMock(seed=42, round_id=1, min_flops_equivalent=0,
                          max_flops_equivalent=1000000, task={"time_budget": 300})
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}

    # First call returns 503, second returns 200
    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.json.return_value = {"error": "not ready"}
    resp_503.text = '{"error": "not ready"}'
    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"status": "success", "flops_equivalent_size": 500000}

    sign_call_count = 0
    original_headers = {"X-Epistula-Signed-By": "hk0", "X-Epistula-Timestamp": "1000"}

    def mock_sign(wallet, body):
        nonlocal sign_call_count
        sign_call_count += 1
        return {
            "X-Epistula-Signed-By": "hk0",
            "X-Epistula-Timestamp": str(1000 + sign_call_count),
        }

    mock_post = AsyncMock(side_effect=[resp_503, resp_200])

    with patch("httpx.AsyncClient") as mock_client, \
         patch("shared.artifacts.generate_upload_urls", return_value={"checkpoint": "http://fake/ckpt"}), \
         patch("validator.coordinator.sign_request", side_effect=mock_sign), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            post=mock_post,
        ))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await coordinator.dispatch_jobs(
            jobs, challenge, submissions, endpoints,
        )

    assert len(results) == 1
    assert results[0].status == "success"
    # sign_request called fresh every attempt (initial + retry)
    assert sign_call_count >= 2


@pytest.mark.asyncio
async def test_dispatch_retries_on_403():
    """403 (stale timestamp) triggers retry with fresh signature."""
    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(), my_uid=10,
    )
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    challenge = MagicMock(seed=42, round_id=1, min_flops_equivalent=0,
                          max_flops_equivalent=1000000, task={"time_budget": 300})
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}

    resp_403 = MagicMock()
    resp_403.status_code = 403
    resp_403.json.return_value = {"error": "Timestamp too old or too far in future"}
    resp_403.text = '{"error": "Timestamp too old or too far in future"}'
    resp_202 = MagicMock()
    resp_202.status_code = 202
    resp_202.json.return_value = {"status": "accepted"}

    mock_post = AsyncMock(side_effect=[resp_403, resp_202])

    with patch("httpx.AsyncClient") as mock_client, \
         patch("shared.artifacts.generate_upload_urls", return_value={"checkpoint": "http://fake/ckpt"}), \
         patch("validator.coordinator.sign_request", return_value={"X-Epistula-Signed-By": "hk0"}), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            post=mock_post,
        ))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await coordinator.dispatch_jobs(
            jobs, challenge, submissions, endpoints,
        )

    assert len(results) == 1
    assert results[0].status == "accepted"
    # POST called twice: first attempt (403) + retry (202)
    assert mock_post.call_count == 2


# ── Submission-ID anonymity tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_payload_omits_miner_hotkey_and_carries_submission_id():
    """Trainer-host must never see the architecture owner's hotkey on the wire.

    The dispatch payload must carry an opaque ``submission_id`` instead.
    The coordinator mints a fresh sid per job at dispatch time.
    """
    import json as _json

    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(), my_uid=10,
    )
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    challenge = MagicMock(
        seed=42, round_id=1, min_flops_equivalent=0,
        max_flops_equivalent=1000000, task={"time_budget": 300},
    )
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}

    captured_payload: dict = {}

    async def fake_post(url, content=None, headers=None):
        captured_payload.update(_json.loads(content))
        resp = MagicMock()
        resp.status_code = 202
        resp.json.return_value = {"status": "accepted"}
        return resp

    with patch("httpx.AsyncClient") as mock_client, \
         patch("shared.artifacts.generate_upload_urls", return_value={
             "checkpoint": "http://fake/ckpt",
             "architecture": "http://fake/arch",
             "meta": "http://fake/meta",
             "stdout": "http://fake/std",
         }), \
         patch("validator.coordinator.sign_request", return_value={"X-Epistula-Signed-By": "hk0"}):
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            post=AsyncMock(side_effect=fake_post),
        ))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await coordinator.dispatch_jobs(
            jobs, challenge, submissions, endpoints,
        )

    # Payload privacy invariants
    assert "miner_hotkey" not in captured_payload
    assert captured_payload.get("submission_id"), "submission_id must be set"
    sid = captured_payload["submission_id"]
    assert len(sid) >= 16  # 16 random bytes hex = 32 chars
    # The job and result both carry the minted sid
    assert jobs[0].submission_id == sid
    assert len(results) == 1
    assert results[0].submission_id == sid


@pytest.mark.asyncio
async def test_dispatch_record_includes_submission_reveal():
    """write_dispatch_record publishes (submission_id, miner_hotkey) so other
    validators can resolve checkpoints minted by this dispatcher in Phase C."""
    from validator.coordinator import TrainingResult

    r2 = MagicMock()
    uploaded: dict = {}

    def fake_upload_json(key, body):
        uploaded[key] = body
        return True

    r2.upload_json = fake_upload_json
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "validator_hk"
    metagraph = MagicMock(hotkeys=["miner_hk_0", "miner_hk_1"])

    coordinator = TrainingCoordinator(
        wallet=wallet, metagraph=metagraph, r2=r2, my_uid=10,
    )
    results = [
        TrainingResult(
            round_id=42, arch_owner=0, trainer_uid=1, dispatcher=10,
            seed=0, status="accepted", submission_id="sid_one",
        ),
        TrainingResult(
            round_id=42, arch_owner=1, trainer_uid=0, dispatcher=10,
            seed=0, status="accepted", submission_id="sid_two",
        ),
    ]
    await coordinator.write_dispatch_record(42, results)

    assert "round_42/dispatch/vali_validator_hk.json" in uploaded
    body = uploaded["round_42/dispatch/vali_validator_hk.json"]
    sids = {j["submission_id"]: j["arch_owner_hotkey"] for j in body["jobs"]}
    assert sids == {"sid_one": "miner_hk_0", "sid_two": "miner_hk_1"}


@pytest.mark.asyncio
async def test_collect_submission_map_unions_dispatch_records():
    """Every dispatcher's record contributes its submission_ids; Phase C
    consumers union the lot to fetch every checkpoint."""
    r2 = MagicMock()
    r2.list_keys = MagicMock(return_value=[
        "round_5/dispatch/vali_a.json",
        "round_5/dispatch/vali_b.json",
        "round_5/dispatch/something_else.txt",  # ignored
    ])
    r2.download_json = MagicMock(side_effect=[
        {"jobs": [
            {"arch_owner": 0, "submission_id": "sid_0_from_a"},
            {"arch_owner": 1, "submission_id": "sid_1_from_a"},
        ]},
        {"jobs": [
            {"arch_owner": 2, "submission_id": "sid_2_from_b"},
            # First-write-wins: another validator can't overwrite arch_owner=0
            {"arch_owner": 0, "submission_id": "sid_0_dup_from_b"},
        ]},
    ])

    coordinator = TrainingCoordinator(
        wallet=MagicMock(), metagraph=MagicMock(hotkeys=[]),
        r2=r2, my_uid=10,
    )
    mapping = await coordinator.collect_submission_map(round_id=5)

    assert mapping == {
        0: "sid_0_from_a",
        1: "sid_1_from_a",
        2: "sid_2_from_b",
    }
