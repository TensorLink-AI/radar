"""Tests for bugs 1-8 fixes."""

import math

from shared.database import DataElement
from shared.pareto import ParetoFront
from shared.scoring import (
    score_round, compute_penalties, scores_to_weights, ema_update,
)
from shared.challenge import current_phase


class _MockChallenge:
    min_flops_equivalent = 100_000
    max_flops_equivalent = 500_000


def _objectives():
    from shared.task import Objective
    return [
        Objective(name="crps", pattern=r"crps:\s*([\d.]+)", lower_is_better=True, primary=True),
        Objective(name="mase", pattern=r"mase:\s*([\d.]+)", lower_is_better=True),
        Objective(name="exec_time", pattern=r"training_seconds:\s*([\d.]+)", lower_is_better=True, weight=0.2),
        Objective(name="memory_mb", pattern=r"peak_vram_mb:\s*([\d.]+)", lower_is_better=True, weight=0.1),
    ]


def _make_pareto(elements=None):
    from shared.task import load_task
    task = load_task("ml_training")
    objective_fn = lambda elem: task.objective_vector(elem.objectives)
    pf = ParetoFront(max_size=50, objective_fn=objective_fn)
    for e in (elements or []):
        pf.update(e)
    return pf


# ── Bug 1: Double-normalization ──


def test_bug1_ema_on_raw_scores_preserves_differentiation():
    """EMA accumulates raw scores; softmax applied once at the end.

    Verifies:
    1. EMA accumulates raw score values (not softmax-compressed)
    2. A single softmax at the end produces valid weights
    3. All miners with nonzero raw scores get nonzero weight
    """
    round_scores = {0: 1.0, 1: 0.5, 2: 0.1}
    ema = {}
    for _ in range(3):
        ema = ema_update(ema, round_scores, [0, 1, 2], alpha=0.3)

    # EMA should hold raw-scale values (not summing to 1.0)
    assert abs(ema[0] - 1.0) < 1e-6, "EMA should hold raw score for miner 0"
    assert abs(ema[1] - 0.5) < 1e-6, "EMA should hold raw score for miner 1"
    assert abs(ema[2] - 0.1) < 1e-6, "EMA should hold raw score for miner 2"

    # Single softmax at the end
    uids, weights = scores_to_weights(ema, temperature=0.1)
    uid_weight = dict(zip(uids, weights))

    # All miners with positive scores get positive weight
    assert uid_weight[0] > 0
    assert uid_weight[1] > 0
    assert uid_weight[2] > 0
    # Ordering preserved
    assert uid_weight[0] > uid_weight[1] > uid_weight[2]
    # Weights sum to 1.0 (single normalization, not double)
    assert abs(sum(weights) - 1.0) < 1e-6


def test_bug1_ema_preserves_raw_scale():
    """EMA should operate on raw scores, not softmax-normalized ones."""
    ema = {}
    round_scores = {0: 1.0, 1: 0.5, 2: 0.1}
    ema = ema_update(ema, round_scores, [0, 1, 2], alpha=0.3)
    # After first round, EMA should equal raw scores
    assert abs(ema[0] - 1.0) < 1e-6
    assert abs(ema[1] - 0.5) < 1e-6
    assert abs(ema[2] - 0.1) < 1e-6


# ── Bug 2: Pareto dominance bonus fires with training meta ──


def test_bug2_pareto_dominance_bonus_with_training_meta():
    """Pareto dominance bonus fires when training_metas supplies exec_time/memory_mb."""
    frontier_elem = DataElement(
        metric=0.5, success=True,
        objectives={
            "crps": 0.5, "mase": 0.8,
            "exec_time": 100.0, "memory_mb": 500.0,
            "flops_equivalent_size": 200_000,
        },
        task="ml_training",
    )
    pareto = _make_pareto([frontier_elem])

    eval_results = {
        0: {
            "crps": 0.3, "mase": 0.6,
            "flops_equivalent_size": 200_000,
            "passed_size_gate": True,
        },
    }
    training_metas = {
        0: {
            "training_time_seconds": 90.0,
            "peak_vram_mb": 400.0,
        },
    }

    # Without training_metas — dominance bonus won't fire (missing exec_time/memory_mb)
    scores_no_meta = score_round(
        eval_results, _MockChallenge(), pareto, _objectives(), {},
    )

    # With training_metas — dominance bonus should fire (1.5x)
    scores_with_meta = score_round(
        eval_results, _MockChallenge(), pareto, _objectives(), {},
        training_metas=training_metas,
    )

    # Score with meta should be higher due to 1.5x bonus
    assert scores_with_meta[0] > scores_no_meta[0], (
        f"With training_metas ({scores_with_meta[0]}) should be > "
        f"without ({scores_no_meta[0]}) due to 1.5x dominance bonus"
    )


# ── Bug 3: Penalty keyed by arch_owner ──


def test_bug3_penalty_on_arch_owner_not_trainer():
    """Cross-eval: penalty applies to arch owner, not trainer."""
    training_metas = {
        0: {"status": "failed", "trainer_uid": 1},
    }
    eval_results = {}
    penalties = compute_penalties(training_metas, eval_results)
    assert 0 in penalties, "Arch owner 0 should be penalized"
    assert 1 not in penalties, "Trainer 1 should NOT be penalized"
    assert penalties[0] == 0.5


# ── Bug 4: DataElement round_id ──


def test_bug4_data_element_has_round_id():
    """DataElement includes round_id field."""
    elem = DataElement(round_id=42, miner_hotkey="abc123")
    assert elem.round_id == 42
    assert elem.miner_hotkey == "abc123"

    d = elem.to_dict()
    assert d["round_id"] == 42
    assert d["miner_hotkey"] == "abc123"

    restored = DataElement.from_dict(d)
    assert restored.round_id == 42
    assert restored.miner_hotkey == "abc123"


def test_bug4_round_id_in_api_dict():
    """to_api_dict() includes round_id."""
    elem = DataElement(round_id=99)
    api = elem.to_api_dict()
    assert api["round_id"] == 99


def test_bug4_round_id_sqlite_persistence():
    """round_id persists through SQLite round-trip."""
    import tempfile
    import os
    from shared.sqlite_store import SQLiteExperimentStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SQLiteExperimentStore(os.path.join(tmpdir, "test.db"))
        elem = DataElement(
            name="test", code="x=1", success=True, metric=0.5,
            miner_uid=7, miner_hotkey="hotkey_abc",
            round_id=42, task="ml_training",
        )
        idx = db.add(elem)
        loaded = db.get(idx)
        assert loaded.round_id == 42
        assert loaded.miner_hotkey == "hotkey_abc"
        db.close()


# ── Bug 6: current_phase scoring window ──


def test_bug6_scoring_phase_block_226():
    """Block 226 (offset=226 from round_start=0) reports 'scoring'."""
    # submission=50, training=150, eval=25 => eval ends at 225
    # scoring_window=50 => scoring from 225 to 275
    assert current_phase(226, 0) == "scoring"
    assert current_phase(274, 0) == "scoring"


def test_bug6_scoring_phase_default():
    """Default scoring_window=50 covers the idle gap."""
    # Phase transitions at offset 0=submission, 50=training, 200=eval, 225=scoring, 275=idle
    assert current_phase(224, 0) == "evaluation"
    assert current_phase(225, 0) == "scoring"
    assert current_phase(274, 0) == "scoring"
    assert current_phase(275, 0) == "idle"


# ── Bug 8: Penalty capped at 1.0 ──


def test_bug8_penalty_capped_at_one():
    """attestation_failed (1.0) + flops_mismatch (0.3) should cap at 1.0."""
    training_metas = {
        0: {"status": "attestation_failed", "trainer_uid": 1},
    }
    eval_results = {0: {"flops_verified": False}}
    penalties = compute_penalties(training_metas, eval_results)
    # attestation_failed = 1.0, flops_mismatch = 0.3, capped at 1.0
    assert penalties[0] == 1.0, f"Penalty should be capped at 1.0, got {penalties[0]}"


def test_bug8_failed_plus_flops_capped():
    """failed (0.5) + flops_mismatch (0.3) = 0.8, should not exceed 1.0."""
    training_metas = {
        0: {"status": "failed", "trainer_uid": 1},
    }
    eval_results = {0: {"flops_verified": False}}
    penalties = compute_penalties(training_metas, eval_results)
    assert abs(penalties[0] - 0.8) < 1e-6
