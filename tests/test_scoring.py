"""Tests for shared.scoring — Phase C size-gated Pareto frontier scoring."""

import math

from shared.database import DataElement
from shared.pareto import ParetoFront
from shared.scoring import (
    passes_size_gate, score_round, compute_penalties,
    scores_to_weights, ema_update,
)


class _MockChallenge:
    min_flops_equivalent = 100_000
    max_flops_equivalent = 500_000


def _objectives():
    from shared.task import Objective
    return [
        Objective(name="crps", pattern=r"crps:\s*([\d.]+)", lower_is_better=True, primary=True),
        Objective(name="mase", pattern=r"mase:\s*([\d.]+)", lower_is_better=True),
    ]


def _pareto(elements=None):
    pf = ParetoFront(max_size=10)
    for e in (elements or []):
        pf.update(e)
    return pf


# ── passes_size_gate tests ──


def test_size_gate_in_range():
    metrics = {"flops_equivalent_size": 200_000}
    assert passes_size_gate(metrics, _MockChallenge())


def test_size_gate_out_of_range():
    metrics = {"flops_equivalent_size": 900_000}
    assert not passes_size_gate(metrics, _MockChallenge())


def test_size_gate_at_boundary():
    metrics = {"flops_equivalent_size": 100_000}
    assert passes_size_gate(metrics, _MockChallenge())
    metrics = {"flops_equivalent_size": 500_000}
    assert passes_size_gate(metrics, _MockChallenge())


def test_size_gate_tolerance():
    """Models within 10% of bucket boundary should pass (default tolerance)."""
    # Over max (500K * 1.10 = 550K) — should pass
    metrics = {"flops_equivalent_size": 540_000}
    assert passes_size_gate(metrics, _MockChallenge())
    # Under min (100K * 0.90 = 90K) — should pass
    metrics = {"flops_equivalent_size": 92_000}
    assert passes_size_gate(metrics, _MockChallenge())
    # Over 10% tolerance — should fail
    metrics = {"flops_equivalent_size": 600_000}
    assert not passes_size_gate(metrics, _MockChallenge())


def test_size_gate_tolerance_large_bucket():
    """137M FLOPs should pass the large bucket (max 125M) with 10% tolerance."""
    class _LargeBucket:
        min_flops_equivalent = 100_000
        max_flops_equivalent = 125_000_000

    # 125M * 1.10 = 137.5M — should pass
    metrics = {"flops_equivalent_size": 135_000_000}
    assert passes_size_gate(metrics, _LargeBucket())
    # Over 10% tolerance — should fail
    metrics = {"flops_equivalent_size": 150_000_000}
    assert not passes_size_gate(metrics, _LargeBucket())


# ── score_round tests ──


def test_score_round_size_gate_filters():
    """Out-of-range submissions score zero."""
    eval_results = {
        0: {"crps": 0.5, "mase": 0.6, "flops_equivalent_size": 200_000, "passed_size_gate": True},
        1: {"crps": 0.4, "mase": 0.5, "flops_equivalent_size": 900_000, "passed_size_gate": False},
    }
    scores = score_round(eval_results, _MockChallenge(), _pareto(), _objectives(), {})
    assert scores[0] > 0
    assert scores[1] == 0.0


def test_score_round_bootstrapping():
    """With empty frontier, uses relative ranking."""
    eval_results = {
        0: {"crps": 0.5, "mase": 0.6, "flops_equivalent_size": 200_000, "passed_size_gate": True},
        1: {"crps": 0.3, "mase": 0.4, "flops_equivalent_size": 300_000, "passed_size_gate": True},
    }
    scores = score_round(eval_results, _MockChallenge(), _pareto(), _objectives(), {})
    # UID 1 has better CRPS -> higher score
    assert scores[1] > scores[0]


def test_score_round_nan_crps_scores_zero():
    eval_results = {
        0: {"crps": float("nan"), "mase": 0.5, "flops_equivalent_size": 200_000, "passed_size_gate": True},
    }
    scores = score_round(eval_results, _MockChallenge(), _pareto(), _objectives(), {})
    assert scores[0] == 0.0


def test_score_round_with_penalties():
    eval_results = {
        0: {"crps": 0.5, "mase": 0.6, "flops_equivalent_size": 200_000, "passed_size_gate": True},
    }
    penalties = {0: 0.5}
    scores = score_round(eval_results, _MockChallenge(), _pareto(), _objectives(), penalties)
    scores_no_pen = score_round(eval_results, _MockChallenge(), _pareto(), _objectives(), {})
    assert scores[0] < scores_no_pen[0]


def test_score_round_empty_results():
    scores = score_round({}, _MockChallenge(), _pareto(), _objectives(), {})
    assert scores == {}


# ── compute_penalties tests ──


def test_penalties_failed_trainer():
    training_metas = {0: {"status": "failed", "trainer_uid": 1}}
    eval_results = {}
    penalties = compute_penalties(training_metas, eval_results)
    # Penalty is on arch_owner (uid=0), not trainer_uid
    assert penalties.get(0, 0) > 0


def test_penalties_flops_mismatch():
    training_metas = {0: {"status": "success", "flops_equivalent_size": 100_000}}
    eval_results = {0: {"flops_verified": False}}
    penalties = compute_penalties(training_metas, eval_results)
    assert penalties.get(0, 0) > 0


# ── scores_to_weights tests ──


def test_scores_to_weights_sums_to_one():
    scores = {0: 0.3, 1: 0.6, 2: 0.9}
    uids, weights = scores_to_weights(scores, temperature=0.1)
    assert abs(sum(weights) - 1.0) < 1e-6


def test_scores_to_weights_zeros_stay_zero():
    scores = {0: 0.0, 1: 0.5, 2: 0.0, 3: 0.8}
    uids, weights = scores_to_weights(scores, temperature=0.1)
    uid_weight = dict(zip(uids, weights))
    assert uid_weight[0] == 0.0
    assert uid_weight[2] == 0.0
    assert uid_weight[3] > uid_weight[1]


def test_scores_to_weights_all_zero():
    scores = {0: 0.0, 1: 0.0}
    uids, weights = scores_to_weights(scores)
    assert all(w == 0.0 for w in weights)


def test_scores_to_weights_low_temp_winner_take_all():
    scores = {0: 0.5, 1: 0.6, 2: 1.0}
    uids, weights = scores_to_weights(scores, temperature=0.01)
    uid_weight = dict(zip(uids, weights))
    assert uid_weight[2] > 0.95


# ── ema_update tests ──


def test_ema_update_new_uid():
    ema = {}
    ema = ema_update(ema, {0: 0.5}, [0], alpha=0.3)
    assert ema[0] == 0.5


def test_ema_update_existing():
    ema = {0: 1.0}
    ema = ema_update(ema, {0: 0.0}, [0], alpha=0.3)
    assert abs(ema[0] - 0.7) < 1e-6  # 0.3*0.0 + 0.7*1.0
