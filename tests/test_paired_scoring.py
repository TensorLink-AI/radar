"""Tests for the paired-Δ continuation gate and replicate handling in
local/scoring.py + local/continuation.py."""

from __future__ import annotations

import pytest

from local.continuation import score_continuation
from local.scoring import compute_pareto, score_round


def _pt(name, m):
    return {"name": name, "ncrps": m, "nmase": m}


def _paired(n, n_improved):
    # Shape matches eval_metrics.paired_per_task_delta output.
    import math
    z = (n_improved - n / 2.0) / math.sqrt(n / 4.0)
    return {"n": n, "n_improved": n_improved,
            "frac_improved": n_improved / n,
            "mean_log_ratio": 0.01, "z": z,
            "significant": n >= 10 and z >= 1.645}


def test_score_continuation_paired_significant_passes():
    score, delta = score_continuation(
        metric=0.8, parent_metric=1.0, cumulative_compute=5.0,
        frontier=[], paired=_paired(30, 28),
    )
    assert delta == pytest.approx(0.2)
    assert score > 0


def test_score_continuation_paired_insignificant_zeroes():
    # Scalar delta positive but the per-dataset evidence says noise.
    score, delta = score_continuation(
        metric=0.99, parent_metric=1.0, cumulative_compute=5.0,
        frontier=[], paired=_paired(30, 16),
    )
    assert delta == pytest.approx(0.01)
    assert score == 0.0


def test_score_continuation_small_paired_sample_falls_back():
    # Too few joined tasks for the sign test — scalar path applies.
    score, _ = score_continuation(
        metric=0.8, parent_metric=1.0, cumulative_compute=5.0,
        frontier=[], paired=_paired(4, 1),
    )
    assert score > 0


def test_score_round_stamps_delta_and_reads_paired():
    results = [{
        "success": True, "metric": 0.99, "mode": "continue",
        "parent_metric": 1.0,
        "objectives": {"flops_equivalent_size": 500,
                       "cumulative_compute": 3.0,
                       "paired": _paired(30, 15)},
        "analysis": "",
    }]
    out = score_round(results, min_flops=200, max_flops=2000, frontier=[])
    assert out[0]["score"] == 0.0
    assert out[0]["objectives"]["delta"] == pytest.approx(0.01)
    assert "paired=15/30" in out[0]["analysis"]


def test_score_round_zeroes_replicates():
    results = [{
        "success": True, "metric": 0.5, "mode": "replicate",
        "objectives": {"flops_equivalent_size": 500},
        "analysis": "",
    }]
    out = score_round(results, min_flops=200, max_flops=2000, frontier=[])
    assert out[0]["score"] == 0.0
    assert "replicate noise probe" in out[0]["analysis"]


def test_compute_pareto_excludes_replicates():
    exps = [
        {"success": True, "metric": 0.5, "mode": "new",
         "objectives": {"flops_equivalent_size": 100}},
        # A "luckier" replicate must not enter (or dominate) the frontier.
        {"success": True, "metric": 0.4, "mode": "replicate",
         "objectives": {"flops_equivalent_size": 100}},
    ]
    front = compute_pareto(exps)
    assert len(front) == 1
    assert front[0]["mode"] == "new"
