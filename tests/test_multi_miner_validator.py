"""Tests for realistic multi-miner, multi-validator subnet scenarios.

Covers:
- Multiple miners competing in the same round with varied architectures
- Multiple validators splitting work, coordinating training, and scoring
- Cross-eval invariants at scale (no self-training, full coverage)
- Concurrent R2 artifact handling across validators
- EMA weight convergence over multiple rounds
- Scoring fairness with mixed success/failure/penalty combinations
- Pareto frontier evolution across rounds with many miners
- Deduplication across a large miner pool
- Fallback reassignment with multiple validator failures
- DB consistency with high-volume concurrent writes
"""

import math
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.challenge import generate_challenge
from shared.database import DataElement, ExperimentDB
from shared.dedup import code_similarity
from shared.pareto import ParetoFront
from shared.protocol import Challenge, Proposal
from shared.scoring import (
    score_round, scores_to_weights, ema_update, compute_penalties,
    passes_size_gate,
)
from shared.task import Objective
from validator.coordinator import compute_assignments, compute_fallback, Job
from validator.neuron import get_my_assignments


# ── Helpers ──────────────────────────────────────────────────────────


def _objectives():
    return [
        Objective(name="crps", pattern=r"crps:\s*([\d.]+)", lower_is_better=True, primary=True),
        Objective(name="mase", pattern=r"mase:\s*([\d.]+)", lower_is_better=True),
    ]


class _MockChallenge:
    min_flops_equivalent = 100_000
    max_flops_equivalent = 500_000


def _pareto(elements=None):
    pf = ParetoFront(max_size=50)
    for e in (elements or []):
        pf.update(e)
    return pf


def _make_proposal(uid: int) -> Proposal:
    """Generate a unique architecture proposal for a given miner UID."""
    return Proposal(
        code=f"import torch\nclass Arch{uid}(torch.nn.Module):\n"
             f"    def __init__(self): super().__init__()\n"
             f"    layers = {uid + 1}\n"
             f"def build_model(): return Arch{uid}()\n"
             f"def build_optimizer(m): return torch.optim.Adam(m.parameters())\n",
        name=f"arch_{uid}",
        motivation=f"Architecture designed by miner {uid}",
    )


def _make_eval_result(
    uid: int,
    crps: float,
    mase: float = 0.9,
    flops: int = 200_000,
    passed: bool = True,
    flops_verified: bool = True,
) -> dict:
    return {
        "crps": crps,
        "mase": mase,
        "flops_equivalent_size": flops,
        "param_count": flops // 2,
        "passed_size_gate": passed,
        "flops_verified": flops_verified,
    }


# ═══════════════════════════════════════════════════════════════════
# Multi-miner scoring scenarios
# ═══════════════════════════════════════════════════════════════════


class TestMultiMinerScoring:
    """Score rounds with 8-16 miners to test ranking stability."""

    def test_8_miners_relative_ranking(self):
        """8 miners, bootstrapping (empty frontier), ranked by CRPS."""
        eval_results = {}
        for uid in range(8):
            crps = 0.9 - uid * 0.05  # UID 7 has best CRPS (0.55)
            eval_results[uid] = _make_eval_result(uid, crps=crps)

        scores = score_round(eval_results, _MockChallenge(), _pareto(), _objectives(), {})

        assert len(scores) == 8
        # Best CRPS (lowest) should get highest score
        best_uid = max(scores, key=scores.get)
        assert best_uid == 7
        # Worst CRPS should get lowest score
        worst_uid = min(scores, key=scores.get)
        assert worst_uid == 0
        # Monotonic: lower CRPS -> higher score
        for i in range(7):
            assert scores[i] <= scores[i + 1], f"UID {i} should score <= UID {i+1}"

    def test_16_miners_mixed_success_and_size_gate(self):
        """16 miners: some pass size gate, some fail, some have bad CRPS."""
        eval_results = {}
        for uid in range(16):
            if uid % 4 == 0:
                # Fails size gate (FLOPs out of range)
                eval_results[uid] = _make_eval_result(uid, crps=0.3, flops=600_000, passed=False)
            elif uid % 4 == 1:
                # NaN CRPS
                eval_results[uid] = _make_eval_result(uid, crps=float("nan"))
            else:
                # Valid
                eval_results[uid] = _make_eval_result(uid, crps=0.8 - uid * 0.01)

        scores = score_round(eval_results, _MockChallenge(), _pareto(), _objectives(), {})

        # Size-gate failures score zero
        for uid in range(0, 16, 4):
            assert scores[uid] == 0.0, f"UID {uid} should be zero (size gate)"
        # NaN CRPS scores zero
        for uid in range(1, 16, 4):
            assert scores[uid] == 0.0, f"UID {uid} should be zero (NaN CRPS)"
        # Valid UIDs should have positive scores
        valid_uids = [uid for uid in range(16) if uid % 4 >= 2]
        for uid in valid_uids:
            assert scores[uid] > 0.0, f"Valid UID {uid} should have positive score"

    def test_weights_with_many_miners(self):
        """Softmax weights across 12 miners sum to 1 and are well-distributed."""
        scores = {uid: 0.1 + uid * 0.05 for uid in range(12)}
        uids, weights = scores_to_weights(scores, temperature=0.1)
        assert abs(sum(weights) - 1.0) < 1e-6
        assert len(uids) == 12
        # Higher scores -> higher weights
        uid_weight = dict(zip(uids, weights))
        assert uid_weight[11] > uid_weight[0]

    def test_weights_with_many_zeros(self):
        """10 miners, only 3 have positive scores."""
        scores = {uid: 0.0 for uid in range(10)}
        scores[3] = 0.5
        scores[7] = 0.8
        scores[9] = 0.3
        uids, weights = scores_to_weights(scores, temperature=0.1)
        uid_weight = dict(zip(uids, weights))
        # Zero scores stay zero
        for uid in [0, 1, 2, 4, 5, 6, 8]:
            assert uid_weight[uid] == 0.0
        # Positive scores get weight
        assert uid_weight[7] > uid_weight[3] > uid_weight[9]
        assert abs(sum(weights) - 1.0) < 1e-6


class TestMultiMinerPenalties:
    """Penalty computation with many miners in cross-eval roles."""

    def test_penalties_across_8_miners(self):
        """8 miners training each other; failures penalize the trainer, not arch owner."""
        training_metas = {
            0: {"status": "success", "trainer_uid": 1, "flops_equivalent_size": 200_000},
            1: {"status": "failed", "trainer_uid": 2},    # Miner 2 fails as trainer
            2: {"status": "success", "trainer_uid": 3, "flops_equivalent_size": 200_000},
            3: {"status": "timeout", "trainer_uid": 4},   # Miner 4 times out
            4: {"status": "success", "trainer_uid": 5, "flops_equivalent_size": 200_000},
            5: {"status": "build_failed", "trainer_uid": 6},  # Miner 6 build fail
            6: {"status": "success", "trainer_uid": 7, "flops_equivalent_size": 200_000},
            7: {"status": "success", "trainer_uid": 0, "flops_equivalent_size": 200_000},
        }
        eval_results = {
            uid: {"flops_verified": True} for uid in range(8)
        }

        penalties = compute_penalties(training_metas, eval_results)

        # Trainers that failed get penalized
        assert penalties.get(2, 0) > 0, "Miner 2 should be penalized (failed as trainer)"
        assert penalties.get(4, 0) > 0, "Miner 4 should be penalized (timeout as trainer)"
        assert penalties.get(6, 0) > 0, "Miner 6 should be penalized (build_failed as trainer)"
        # Successful trainers should not be penalized
        assert penalties.get(1, 0) == 0
        assert penalties.get(3, 0) == 0
        assert penalties.get(5, 0) == 0

    def test_double_failure_stacks_penalties(self):
        """A miner that fails as trainer for 2 different arch owners gets stacked penalty."""
        training_metas = {
            0: {"status": "failed", "trainer_uid": 5},
            1: {"status": "failed", "trainer_uid": 5},  # Same trainer fails twice
            2: {"status": "success", "trainer_uid": 3, "flops_equivalent_size": 200_000},
        }
        eval_results = {uid: {"flops_verified": True} for uid in range(3)}

        penalties = compute_penalties(training_metas, eval_results)
        assert penalties.get(5, 0) >= 1.0, "Double failure should stack to >= 1.0"

    def test_flops_mismatch_penalty_with_many_miners(self):
        """FLOPs mismatch penalty on trainer, not arch owner."""
        training_metas = {}
        eval_results = {}
        for uid in range(6):
            trainer = (uid + 1) % 6
            training_metas[uid] = {
                "status": "success",
                "trainer_uid": trainer,
                "flops_equivalent_size": 200_000,
            }
            # UIDs 0, 2, 4 have FLOPs mismatch
            eval_results[uid] = {"flops_verified": uid % 2 == 1}

        penalties = compute_penalties(training_metas, eval_results)

        # Trainers for UIDs 0, 2, 4 should be penalized
        for uid in [0, 2, 4]:
            trainer = (uid + 1) % 6
            assert penalties.get(trainer, 0) > 0, f"Trainer {trainer} should be penalized"


# ═══════════════════════════════════════════════════════════════════
# Multi-validator work splitting and coordination
# ═══════════════════════════════════════════════════════════════════


class TestMultiValidatorWorkSplitting:
    """Work splitting with 5+ validators and 20+ miners."""

    def test_5_validators_20_miners_full_coverage(self):
        """5 validators, 20 miners: every miner assigned to exactly one validator."""
        miners = list(range(20))
        validators = [100, 101, 102, 103, 104]

        all_assigned = []
        for v in validators:
            assigned = get_my_assignments(miners, validators, v, seed=42)
            all_assigned.extend(assigned)

        assert sorted(all_assigned) == sorted(miners)

    def test_5_validators_20_miners_no_overlap(self):
        """No miner assigned to two different validators."""
        miners = list(range(20))
        validators = [100, 101, 102, 103, 104]

        sets = [
            set(get_my_assignments(miners, validators, v, seed=42))
            for v in validators
        ]

        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                overlap = sets[i] & sets[j]
                assert not overlap, f"Validators {validators[i]} and {validators[j]} overlap: {overlap}"

    def test_balanced_with_uneven_count(self):
        """17 miners across 5 validators: max imbalance is 1."""
        miners = list(range(17))
        validators = [100, 101, 102, 103, 104]

        counts = [
            len(get_my_assignments(miners, validators, v, seed=42))
            for v in validators
        ]
        # With 17 / 5 = 3r2, each gets 3 or 4
        assert max(counts) - min(counts) <= 1
        assert sum(counts) == 17

    def test_different_seeds_produce_different_splits(self):
        """Work splits change across rounds (different seeds)."""
        miners = list(range(20))
        validators = [100, 101, 102]

        split_a = get_my_assignments(miners, validators, 100, seed=1)
        split_b = get_my_assignments(miners, validators, 100, seed=2)
        assert split_a != split_b

    def test_10_validators_100_miners(self):
        """Stress test: 10 validators, 100 miners."""
        miners = list(range(100))
        validators = list(range(200, 210))

        all_assigned = []
        for v in validators:
            assigned = get_my_assignments(miners, validators, v, seed=42)
            all_assigned.extend(assigned)
            assert len(assigned) == 10  # 100/10 = 10 each

        assert sorted(all_assigned) == sorted(miners)


class TestMultiValidatorCoordination:
    """Cross-eval job assignment with many miners and validators."""

    def test_10_miners_4_validators_cross_eval(self):
        """10 miners, 4 validators: no miner trains their own architecture."""
        submissions = {i: _make_proposal(i) for i in range(10)}
        miner_uids = list(range(10))
        validator_uids = [100, 101, 102, 103]

        jobs = compute_assignments("a" * 64, submissions, miner_uids, validator_uids, round_id=1)

        assert len(jobs) == 10
        for job in jobs:
            assert job.arch_owner != job.trainer_uid, (
                f"Self-training detected: UID {job.arch_owner}"
            )

    def test_dispatch_distributed_across_validators(self):
        """Jobs are spread across all validators, not concentrated on one."""
        submissions = {i: _make_proposal(i) for i in range(12)}
        miner_uids = list(range(12))
        validator_uids = [100, 101, 102, 103]

        jobs = compute_assignments("b" * 64, submissions, miner_uids, validator_uids, round_id=1)

        dispatcher_counts = {}
        for job in jobs:
            dispatcher_counts[job.dispatcher] = dispatcher_counts.get(job.dispatcher, 0) + 1

        # Each validator should get 3 jobs (12 / 4)
        assert len(dispatcher_counts) == 4
        assert all(c == 3 for c in dispatcher_counts.values())

    def test_all_archs_covered(self):
        """Every submitted architecture gets assigned to some trainer."""
        submissions = {i: _make_proposal(i) for i in range(8)}
        miner_uids = list(range(8))
        validator_uids = [100, 101, 102]

        jobs = compute_assignments("c" * 64, submissions, miner_uids, validator_uids, round_id=1)

        trained_archs = {j.arch_owner for j in jobs}
        assert trained_archs == set(submissions.keys())

    def test_deterministic_across_validators(self):
        """All validators compute the same job assignments."""
        submissions = {i: _make_proposal(i) for i in range(6)}
        miner_uids = list(range(6))
        validator_uids = [100, 101, 102]
        block_hash = "d" * 64

        # Each validator calls compute_assignments independently
        results = []
        for _ in range(3):
            jobs = compute_assignments(block_hash, submissions, miner_uids, validator_uids, round_id=1)
            results.append([(j.arch_owner, j.trainer_uid, j.dispatcher) for j in jobs])

        assert results[0] == results[1] == results[2]


class TestMultiValidatorFallback:
    """Fallback reassignment when multiple validators go offline."""

    def test_2_of_4_validators_fail(self):
        """2 validators drop; their jobs reassigned to remaining 2."""
        jobs = [
            Job(arch_owner=i, trainer_uid=(i + 1) % 8, dispatcher=100 + (i % 4), round_id=1)
            for i in range(8)
        ]
        missing_valis = [100, 101]  # 2 of 4 validators fail
        remaining_valis = [102, 103]

        reassigned = compute_fallback("e" * 64, missing_valis, jobs, remaining_valis)

        # Jobs from validators 100, 101 should be reassigned
        orphaned_count = sum(1 for j in jobs if j.dispatcher in missing_valis)
        assert len(reassigned) == orphaned_count

        # All reassigned to remaining validators
        for j in reassigned:
            assert j.dispatcher in remaining_valis

    def test_all_but_one_validator_fails(self):
        """Worst case: only 1 validator left, gets all orphaned work."""
        jobs = [
            Job(arch_owner=i, trainer_uid=(i + 1) % 6, dispatcher=100 + (i % 3), round_id=1)
            for i in range(6)
        ]
        missing = [100, 101]
        remaining = [102]

        reassigned = compute_fallback("f" * 64, missing, jobs, remaining)

        # All orphaned jobs go to the sole remaining validator
        for j in reassigned:
            assert j.dispatcher == 102

    def test_fallback_preserves_arch_and_trainer(self):
        """Reassignment only changes dispatcher, not arch_owner or trainer_uid."""
        original_jobs = [
            Job(arch_owner=0, trainer_uid=1, dispatcher=100, round_id=1),
            Job(arch_owner=2, trainer_uid=3, dispatcher=100, round_id=1),
            Job(arch_owner=4, trainer_uid=5, dispatcher=101, round_id=1),
        ]
        reassigned = compute_fallback("a" * 64, [100], original_jobs, [101, 102])

        # Only dispatcher 100's jobs get reassigned
        assert len(reassigned) == 2
        for j in reassigned:
            assert j.arch_owner in [0, 2]
            # arch_owner and trainer_uid preserved
            original = next(o for o in original_jobs if o.arch_owner == j.arch_owner)
            assert j.trainer_uid == original.trainer_uid


# ═══════════════════════════════════════════════════════════════════
# Multi-round EMA convergence
# ═══════════════════════════════════════════════════════════════════


class TestMultiRoundEMAConvergence:
    """EMA weight evolution over multiple rounds with many miners."""

    def test_ema_converges_with_consistent_performer(self):
        """A consistently good miner's EMA score rises over 10 rounds."""
        ema = {}
        all_uids = list(range(8))

        for round_num in range(10):
            round_scores = {}
            for uid in all_uids:
                if uid == 3:
                    round_scores[uid] = 0.9  # Consistently best
                else:
                    round_scores[uid] = 0.1 + (uid * 0.05)
            ema = ema_update(ema, round_scores, all_uids, alpha=0.3)

        # UID 3 should have highest EMA
        assert ema[3] == max(ema.values())
        # Should be close to 0.9 (converges from repeated 0.9 scores)
        assert ema[3] > 0.8

    def test_ema_decay_for_inactive_miner(self):
        """A miner that stops submitting sees EMA decay toward 0."""
        ema = {0: 1.0, 1: 1.0, 2: 1.0}
        all_uids = [0, 1, 2]

        # UID 0 scores 0 for 10 rounds
        for _ in range(10):
            ema = ema_update(ema, {0: 0.0, 1: 0.5, 2: 0.5}, all_uids, alpha=0.3)

        assert ema[0] < 0.05  # Should have decayed close to 0
        assert ema[1] > 0.4   # Should be near 0.5

    def test_ema_with_new_miners_joining(self):
        """New miners get initialized when they first appear."""
        ema = {}
        # Round 1: 4 miners
        ema = ema_update(ema, {0: 0.5, 1: 0.3, 2: 0.7, 3: 0.1}, [0, 1, 2, 3], alpha=0.3)
        assert len(ema) == 4

        # Round 2: 2 new miners join (UIDs 4, 5)
        ema = ema_update(
            ema, {0: 0.5, 1: 0.3, 2: 0.7, 3: 0.1, 4: 0.8, 5: 0.6},
            [0, 1, 2, 3, 4, 5], alpha=0.3,
        )
        assert len(ema) == 6
        assert ema[4] == 0.8  # New miner, first score is raw
        assert ema[5] == 0.6

    def test_weight_stability_over_rounds(self):
        """Weights don't oscillate wildly with stable performance."""
        ema = {}
        all_uids = list(range(6))
        weight_history = []

        for round_num in range(20):
            round_scores = {uid: 0.3 + uid * 0.1 for uid in all_uids}
            ema = ema_update(ema, round_scores, all_uids, alpha=0.3)
            _, weights = scores_to_weights(ema, temperature=0.1)
            weight_history.append(weights)

        # After 20 rounds of stable scores, weights should stabilize
        last_weights = weight_history[-1]
        second_last = weight_history[-2]
        max_diff = max(abs(a - b) for a, b in zip(last_weights, second_last))
        assert max_diff < 0.01, "Weights should stabilize with consistent performance"


# ═══════════════════════════════════════════════════════════════════
# Multi-miner Pareto frontier evolution
# ═══════════════════════════════════════════════════════════════════


class TestMultiMinerParetoEvolution:
    """Pareto front evolution when many miners submit across rounds."""

    def test_front_grows_with_non_dominated_submissions(self):
        """Multiple non-dominated submissions all join the front."""
        pareto = ParetoFront(max_size=50)

        # 5 submissions with different tradeoffs (CRPS vs exec_time)
        for i in range(5):
            elem = DataElement(
                index=i, name=f"arch_{i}", success=True,
                metric=0.5 + i * 0.1,  # Increasing metric (worse CRPS)
                objectives={
                    "exec_time": 100 - i * 15,  # Decreasing time (better)
                    "memory_mb": 5000,
                },
            )
            pareto.update(elem)

        # All should be on the front (non-dominated tradeoff)
        assert pareto.size == 5

    def test_dominated_submissions_rejected(self):
        """A submission dominated on all objectives is not added."""
        pareto = ParetoFront(max_size=50)

        good = DataElement(
            index=0, name="good", success=True, metric=0.3,
            objectives={"exec_time": 50, "memory_mb": 3000},
        )
        pareto.update(good)

        # Worse on all dimensions
        bad = DataElement(
            index=1, name="bad", success=True, metric=0.5,
            objectives={"exec_time": 100, "memory_mb": 5000},
        )
        added = pareto.update(bad)
        assert not added
        assert pareto.size == 1

    def test_new_dominant_removes_old(self):
        """A new submission that dominates existing members removes them."""
        pareto = ParetoFront(max_size=50)

        # Add 3 mediocre entries
        for i in range(3):
            pareto.update(DataElement(
                index=i, name=f"old_{i}", success=True,
                metric=0.8 + i * 0.01,
                objectives={"exec_time": 200, "memory_mb": 8000},
            ))

        # New entry dominates all
        dominant = DataElement(
            index=10, name="dominant", success=True, metric=0.3,
            objectives={"exec_time": 50, "memory_mb": 2000},
        )
        pareto.update(dominant)
        assert pareto.size == 1
        assert pareto.best.name == "dominant"

    def test_feasible_filtering_with_multiple_buckets(self):
        """Feasible returns only elements in the requested FLOPs bucket."""
        pareto = ParetoFront(max_size=50)

        # Add elements in different size buckets with tradeoffs so none dominate
        # Each has a unique advantage on a different objective
        entries = [
            (150_000, 0.9, 50, 8000),   # Tiny: best exec_time
            (300_000, 0.5, 200, 3000),   # Tiny: best metric + memory
            (800_000, 0.7, 80, 6000),    # Small
            (1_500_000, 0.6, 150, 4000), # Small
            (5_000_000, 0.4, 300, 2000), # Medium-small: best metric
        ]
        for i, (flops, metric, exec_t, mem) in enumerate(entries):
            pareto.update(DataElement(
                index=i, name=f"bucket_{i}", success=True,
                metric=metric,
                objectives={
                    "flops_equivalent_size": flops,
                    "exec_time": exec_t,
                    "memory_mb": mem,
                },
            ))

        # Tiny bucket: 100K - 500K
        tiny = pareto.get_feasible(100_000, 500_000)
        tiny_flops = [c.element.objectives["flops_equivalent_size"] for c in tiny]
        assert all(100_000 <= f <= 500_000 for f in tiny_flops)
        assert len(tiny) == 2  # 150K and 300K

        # Small bucket: 500K - 2M
        small = pareto.get_feasible(500_000, 2_000_000)
        assert len(small) == 2  # 800K and 1.5M


# ═══════════════════════════════════════════════════════════════════
# End-to-end multi-round simulation
# ═══════════════════════════════════════════════════════════════════


class TestEndToEndMultiRound:
    """Simulate multiple rounds with many miners and validators."""

    def test_3_rounds_8_miners_3_validators(self):
        """Full simulation: 3 rounds, 8 miners, 3 validators."""
        tmpdir = tempfile.mkdtemp()
        db = ExperimentDB(db_dir=tmpdir)
        pareto = ParetoFront(max_size=50)
        ema = {}
        all_miner_uids = list(range(8))
        validator_uids = [100, 101, 102]

        for round_num in range(3):
            block_hash = f"{round_num:064x}"
            challenge = generate_challenge(block_hash, {"name": "test"})

            # Every miner submits
            submissions = {uid: _make_proposal(uid) for uid in all_miner_uids}

            # Compute job assignments
            jobs = compute_assignments(
                block_hash, submissions, all_miner_uids, validator_uids,
                round_id=challenge.round_id,
            )

            # Verify cross-eval invariant
            for job in jobs:
                if len(all_miner_uids) > 1:
                    assert job.arch_owner != job.trainer_uid

            # Verify all archs get trained
            trained_archs = {j.arch_owner for j in jobs}
            assert trained_archs == set(all_miner_uids)

            # Verify work distributed across validators
            dispatchers = {j.dispatcher for j in jobs}
            assert len(dispatchers) >= 2

            # Simulate Phase C eval results
            eval_results = {}
            for uid in all_miner_uids:
                crps = 0.9 - round_num * 0.1 - uid * 0.02  # Improving each round
                flops = challenge.min_flops_equivalent + 1000 + uid * 100
                if challenge.min_flops_equivalent <= flops <= challenge.max_flops_equivalent:
                    eval_results[uid] = _make_eval_result(uid, crps=max(crps, 0.1), flops=flops)
                else:
                    eval_results[uid] = _make_eval_result(
                        uid, crps=max(crps, 0.1), flops=flops, passed=False,
                    )

            # Score
            penalties = {}
            round_scores = score_round(
                eval_results, challenge, pareto, _objectives(), penalties,
            )

            # Update DB + Pareto
            for uid, metrics in eval_results.items():
                if metrics.get("passed_size_gate"):
                    element = DataElement(
                        name=f"r{round_num}_m{uid}",
                        code=submissions[uid].code,
                        metric=metrics["crps"],
                        success=True,
                        objectives=metrics,
                        miner_uid=uid,
                        generation=round_num,
                    )
                    db.add(element)
                    pareto.update(element)

            # EMA update
            uids, weights = scores_to_weights(round_scores, temperature=0.1)
            normalized = dict(zip(uids, weights))
            ema = ema_update(ema, normalized, all_miner_uids, alpha=0.3)

        # Final assertions
        assert db.size > 0
        assert pareto.size > 0
        assert len(ema) == 8
        assert abs(sum(ema.values()) - sum(ema.values())) < 1e-9  # no NaN

    def test_miner_churn_across_rounds(self):
        """Miners register and deregister across rounds."""
        ema = {}

        # Round 1: miners 0-4
        round1_uids = list(range(5))
        scores_r1 = {uid: 0.5 + uid * 0.1 for uid in round1_uids}
        ema = ema_update(ema, scores_r1, round1_uids, alpha=0.3)

        # Round 2: miners 3-7 (0, 1, 2 deregistered; 5, 6, 7 new)
        round2_uids = list(range(3, 8))
        scores_r2 = {uid: 0.4 + uid * 0.05 for uid in round2_uids}
        ema = ema_update(ema, scores_r2, round2_uids, alpha=0.3)

        # Old miners (0, 1, 2) still have EMA from round 1
        for uid in [0, 1, 2]:
            assert uid in ema
        # New miners (5, 6, 7) have EMA from round 2
        for uid in [5, 6, 7]:
            assert uid in ema


# ═══════════════════════════════════════════════════════════════════
# Multi-miner deduplication
# ═══════════════════════════════════════════════════════════════════


class TestMultiMinerDedup:
    """Deduplication when many miners submit similar architectures."""

    def test_unique_proposals_are_distinct(self):
        """Proposals from _make_proposal have low similarity."""
        proposals = [_make_proposal(uid) for uid in range(10)]
        for i in range(len(proposals)):
            for j in range(i + 1, len(proposals)):
                sim = code_similarity(proposals[i].code, proposals[j].code)
                assert sim < 0.95, f"UIDs {i} and {j} are too similar ({sim:.2f})"

    def test_identical_code_detected(self):
        """Two miners submitting identical code get similarity 1.0."""
        code = "def build_model(): return nn.Linear(10, 1)\ndef build_optimizer(m): pass"
        assert code_similarity(code, code) == 1.0

    def test_minor_modification_below_threshold(self):
        """Trivial changes (whitespace, comments) still detected as similar."""
        code_a = "def build_model():\n    return nn.Linear(10, 1)\ndef build_optimizer(m): pass"
        code_b = "def build_model():\n    # my model\n    return nn.Linear(10, 1)\ndef build_optimizer(m): pass"
        sim = code_similarity(code_a, code_b)
        assert sim > 0.9, "Trivial comment addition should still be very similar"


# ═══════════════════════════════════════════════════════════════════
# Multi-miner DB operations
# ═══════════════════════════════════════════════════════════════════


class TestMultiMinerDatabase:
    """Database operations with many concurrent miner submissions."""

    def test_batch_add_from_many_miners(self):
        """add_batch handles 16 miners in a single save."""
        tmpdir = tempfile.mkdtemp()
        db = ExperimentDB(db_dir=tmpdir)

        elements = []
        for uid in range(16):
            elements.append(DataElement(
                name=f"miner_{uid}_arch",
                code=_make_proposal(uid).code,
                metric=0.5 + uid * 0.01,
                success=True,
                objectives={"crps": 0.5 + uid * 0.01, "flops_equivalent_size": 200_000},
                miner_uid=uid,
            ))

        indices = db.add_batch(elements)
        assert len(indices) == 16
        assert db.size == 16
        assert indices == list(range(16))

    def test_stats_with_mixed_results(self):
        """Stats correctly reflect mixed success/failure from 12 miners."""
        tmpdir = tempfile.mkdtemp()
        db = ExperimentDB(db_dir=tmpdir)

        for uid in range(12):
            success = uid % 3 != 0  # UIDs 0, 3, 6, 9 fail
            db.add(DataElement(
                name=f"miner_{uid}",
                metric=0.5 if success else None,
                success=success,
                miner_uid=uid,
            ))

        stats = db.stats()
        assert stats["total"] == 12
        assert stats["successful"] == 8
        assert stats["failed"] == 4

    def test_get_in_flops_range_with_many_buckets(self):
        """Filter experiments from 20 miners across different size buckets."""
        tmpdir = tempfile.mkdtemp()
        db = ExperimentDB(db_dir=tmpdir)

        flops_values = [
            150_000, 300_000, 450_000,   # Tiny
            700_000, 1_200_000,          # Small
            3_000_000, 8_000_000,        # Medium-small
            15_000_000, 40_000_000,      # Medium
            80_000_000,                  # Large
        ]
        for i, flops in enumerate(flops_values):
            db.add(DataElement(
                name=f"arch_{i}", metric=0.5, success=True,
                objectives={"flops_equivalent_size": flops},
            ))

        tiny = db.get_in_flops_range(100_000, 500_000)
        assert len(tiny) == 3

        small = db.get_in_flops_range(500_000, 2_000_000)
        assert len(small) == 2

        medium = db.get_in_flops_range(10_000_000, 50_000_000)
        assert len(medium) == 2


# ═══════════════════════════════════════════════════════════════════
# Multi-validator R2 artifact coordination
# ═══════════════════════════════════════════════════════════════════


class TestMultiValidatorR2Coordination:
    """R2 artifact upload/download across multiple validators."""

    def test_validators_write_independent_dispatch_records(self, mock_r2):
        """Each validator writes its own dispatch record to R2."""
        round_id = 42
        for vali_uid in [100, 101, 102]:
            hotkey = f"vali_hotkey_{vali_uid}"
            key = f"round_{round_id}/dispatch/vali_{hotkey}.json"
            record = {
                "dispatcher": hotkey,
                "round_id": round_id,
                "jobs": [
                    {"arch_owner": i, "trainer_uid": (i + 1) % 8, "status": "success"}
                    for i in range(vali_uid - 100, vali_uid - 100 + 3)
                ],
            }
            mock_r2.upload_json(key, record)

        # All 3 dispatch records should exist
        for vali_uid in [100, 101, 102]:
            hotkey = f"vali_hotkey_{vali_uid}"
            key = f"round_{round_id}/dispatch/vali_{hotkey}.json"
            data = mock_r2.download_json(key)
            assert data is not None
            assert data["dispatcher"] == hotkey
            assert len(data["jobs"]) == 3

    def test_checkpoint_uploads_from_multiple_miners(self, mock_r2):
        """8 miners upload checkpoints; all validators can read them."""
        round_id = 1
        for uid in range(8):
            hotkey = f"miner_hotkey_{uid}"
            ckpt_key = f"round_{round_id}/miner_{hotkey}/checkpoint.safetensors"
            meta_key = f"round_{round_id}/miner_{hotkey}/training_meta.json"
            mock_r2.upload_text(ckpt_key, f"weights_for_miner_{uid}")
            mock_r2.upload_json(meta_key, {
                "miner_hotkey": hotkey,
                "status": "success",
                "flops_equivalent_size": 200_000 + uid * 10_000,
            })

        # Each validator can read all checkpoints
        for uid in range(8):
            hotkey = f"miner_hotkey_{uid}"
            meta = mock_r2.download_json(f"round_{round_id}/miner_{hotkey}/training_meta.json")
            assert meta["status"] == "success"
            ckpt = mock_r2.download_text(f"round_{round_id}/miner_{hotkey}/checkpoint.safetensors")
            assert f"weights_for_miner_{uid}" in ckpt

    def test_frontier_written_once_read_by_all(self, mock_r2):
        """One validator writes frontier, all validators read the same state."""
        frontier_data = [
            {"name": f"front_{i}", "metric": 0.3 + i * 0.1, "index": i}
            for i in range(5)
        ]
        mock_r2.upload_json("frontier/latest.json", {"frontier": frontier_data})

        # Any validator reading frontier sees the same data
        for _ in range(3):  # Simulate 3 validators reading
            data = mock_r2.download_json("frontier/latest.json")
            assert len(data["frontier"]) == 5
            assert data["frontier"][0]["name"] == "front_0"


# ═══════════════════════════════════════════════════════════════════
# Challenge determinism with multi-validator consensus
# ═══════════════════════════════════════════════════════════════════


class TestMultiValidatorConsensusDeterminism:
    """All validators must independently derive identical challenges."""

    def test_same_block_hash_same_challenge(self):
        """5 validators all generate the exact same challenge."""
        block_hash = "abc123" * 10 + "abcd"
        task = {"name": "ts_forecasting"}

        challenges = [generate_challenge(block_hash, task) for _ in range(5)]

        for c in challenges[1:]:
            assert c.round_id == challenges[0].round_id
            assert c.seed == challenges[0].seed
            assert c.min_flops_equivalent == challenges[0].min_flops_equivalent
            assert c.max_flops_equivalent == challenges[0].max_flops_equivalent
            assert c.eval_split_seed == challenges[0].eval_split_seed

    def test_scoring_determinism_across_validators(self):
        """All validators with the same eval results compute identical scores."""
        eval_results = {uid: _make_eval_result(uid, crps=0.5 + uid * 0.05) for uid in range(6)}
        pareto = _pareto()
        objectives = _objectives()
        penalties = {2: 0.3}

        scores_list = [
            score_round(eval_results, _MockChallenge(), pareto, objectives, penalties)
            for _ in range(5)
        ]

        for scores in scores_list[1:]:
            for uid in scores:
                assert abs(scores[uid] - scores_list[0][uid]) < 1e-10
