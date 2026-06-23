"""Tests for local/eval_metrics.py — per-task persistence, canary split,
paired per-dataset comparison. Pure-python, no torch."""

from __future__ import annotations

import math

import pytest

from local.eval_metrics import (
    average_per_task,
    canary_frac,
    compact_per_task,
    eval_seeds,
    finalize_gift_eval,
    geomean,
    is_canary,
    is_proxy,
    paired_per_task_delta,
    proxy_frac,
    task_metric,
)


def _pt(name, ncrps, nmase):
    return {"config_name": name, "normalized_crps": ncrps,
            "normalized_mase": nmase}


def test_geomean_ignores_nonfinite():
    assert geomean([1.0, 4.0]) == pytest.approx(2.0)
    assert geomean([1.0, float("inf"), 4.0]) == pytest.approx(2.0)
    assert geomean([]) == float("inf")


def test_compact_per_task_filters_and_rounds():
    raw = [
        _pt("a", 0.5, 0.8),
        _pt("b", float("nan"), 0.8),       # dropped: non-finite
        {"config_name": "c"},               # dropped: missing values
        {"name": "d", "ncrps": 0.1, "nmase": 0.2},  # alt key shape works
    ]
    out = compact_per_task(raw)
    assert [t["name"] for t in out] == ["a", "d"]
    assert out[0] == {"name": "a", "ncrps": 0.5, "nmase": 0.8}


def test_canary_membership_deterministic_and_fractional():
    names = [f"ds_{i}" for i in range(400)]
    members = {n for n in names if is_canary(n, 0.15)}
    # Deterministic across calls.
    assert members == {n for n in names if is_canary(n, 0.15)}
    # Roughly the requested fraction.
    assert 0.05 < len(members) / len(names) < 0.30
    # frac=0 disables.
    assert not any(is_canary(n, 0.0) for n in names)


def test_finalize_without_per_task_passes_through():
    final = finalize_gift_eval({"crps": 0.5, "mase": 2.0, "n_tasks": 97})
    assert final["crps"] == 0.5 and final["mase"] == 2.0
    assert final["metric"] == pytest.approx(1.0)
    assert final["extras"]["n_tasks"] == 97
    assert "per_task" not in final["extras"]


def test_finalize_nonfinite_returns_none():
    assert finalize_gift_eval({"crps": float("nan"), "mase": 1.0}) is None
    assert finalize_gift_eval({"crps": None, "mase": 1.0}) is None


def test_finalize_persists_per_task_and_recomputes_aggregates():
    per_task = [_pt(f"d{i}", 0.5 * (1 + i % 3), 0.8) for i in range(20)]
    final = finalize_gift_eval(
        {"crps": 999.0, "mase": 999.0, "per_task": per_task},
        canary=0.0, proxy=0.0,
    )
    # Aggregates recomputed from the breakdown, not the (bogus) top level.
    assert final["crps"] < 2.0
    assert final["extras"]["n_tasks"] == 20
    assert len(final["extras"]["per_task"]) == 20


def test_finalize_canary_split_excludes_held_out_tasks():
    per_task = [_pt(f"d{i}", 1.0, 1.0) for i in range(200)]
    final = finalize_gift_eval(
        {"per_task": per_task, "crps": 1.0, "mase": 1.0},
        canary=0.2, proxy=0.0,
    )
    extras = final["extras"]
    assert extras["eval_canary_frac"] == 0.2
    assert extras["n_tasks"] + extras["n_tasks_canary"] == 200
    assert extras["n_tasks_canary"] > 0
    assert extras["canary_metric"] == pytest.approx(1.0)
    # Scored aggregate only covers non-canary tasks.
    assert final["metric"] == pytest.approx(1.0)


def test_proxy_frac_env(monkeypatch):
    monkeypatch.delenv("RADAR_EVAL_PROXY_FRAC", raising=False)
    assert proxy_frac() == 0.15  # default ON
    monkeypatch.setenv("RADAR_EVAL_PROXY_FRAC", "0.25")
    assert proxy_frac() == 0.25
    monkeypatch.setenv("RADAR_EVAL_PROXY_FRAC", "0")
    assert proxy_frac() == 0.0  # explicit opt-out
    monkeypatch.setenv("RADAR_EVAL_PROXY_FRAC", "0.9")
    assert proxy_frac() == 0.5  # clamped
    monkeypatch.setenv("RADAR_EVAL_PROXY_FRAC", "junk")
    assert proxy_frac() == 0.15  # bad value falls back to default


def test_proxy_and_canary_are_disjoint_three_way_split():
    per_task = [_pt(f"d{i}", 1.0, 1.0) for i in range(300)]
    final = finalize_gift_eval(
        {"per_task": per_task, "crps": 1.0, "mase": 1.0},
        canary=0.2, proxy=0.2,
    )
    extras = final["extras"]
    # scored | proxy | canary partition the whole set with no overlap.
    assert (extras["n_tasks"] + extras["n_tasks_proxy"]
            + extras["n_tasks_canary"] == 300)
    assert extras["n_tasks_proxy"] > 0
    assert extras["eval_proxy_frac"] == 0.2
    # Proxy metric is the same sqrt(crps*mase) formula on the held-out slice.
    assert extras["proxy_metric"] == pytest.approx(1.0)
    # The scored metric still ignores both held-out slices.
    assert final["metric"] == pytest.approx(1.0)


def test_proxy_membership_independent_of_canary():
    names = [f"ds_{i}" for i in range(400)]
    proxy_set = {n for n in names if is_proxy(n, 0.2)}
    canary_set = {n for n in names if is_canary(n, 0.2)}
    # Different salts → the two name-hash draws are not identical.
    assert proxy_set != canary_set
    assert proxy_set  # non-empty at this fraction
    assert not any(is_proxy(n, 0.0) for n in names)


def test_proxy_default_on_shrinks_scored_set(monkeypatch):
    monkeypatch.delenv("RADAR_EVAL_PROXY_FRAC", raising=False)
    per_task = [_pt(f"d{i}", 1.0, 1.0) for i in range(200)]
    final = finalize_gift_eval({"per_task": per_task, "crps": 1.0, "mase": 1.0})
    extras = final["extras"]
    # With the default proxy on, some tasks are held out of scoring and a
    # proxy_metric is exposed.
    assert extras["n_tasks"] < 200
    assert extras.get("proxy_metric") is not None


def test_canary_frac_env(monkeypatch):
    monkeypatch.setenv("RADAR_EVAL_CANARY_FRAC", "0.15")
    assert canary_frac() == 0.15
    monkeypatch.setenv("RADAR_EVAL_CANARY_FRAC", "0.9")
    assert canary_frac() == 0.5  # clamped
    monkeypatch.setenv("RADAR_EVAL_CANARY_FRAC", "junk")
    assert canary_frac() == 0.0


def test_paired_delta_requires_both_sides():
    pt = [_pt("a", 1.0, 1.0)]
    assert paired_per_task_delta(None, pt) is None
    assert paired_per_task_delta(pt, None) is None
    assert paired_per_task_delta(pt, [_pt("zzz", 1.0, 1.0)]) is None


def test_paired_delta_significant_improvement():
    parent = [_pt(f"d{i}", 1.0, 1.0) for i in range(30)]
    child = [_pt(f"d{i}", 0.8, 0.8) for i in range(30)]
    paired = paired_per_task_delta(parent, child)
    assert paired["n"] == 30
    assert paired["n_improved"] == 30
    assert paired["frac_improved"] == 1.0
    assert paired["mean_log_ratio"] == pytest.approx(math.log(1.0 / 0.8))
    assert paired["significant"] is True


def test_paired_delta_noise_not_significant():
    # Half improve, half regress — classic noise. Never significant.
    parent = [_pt(f"d{i}", 1.0, 1.0) for i in range(30)]
    child = [
        _pt(f"d{i}", 0.9 if i % 2 == 0 else 1.1, 1.0) for i in range(30)
    ]
    paired = paired_per_task_delta(parent, child)
    assert paired["n_improved"] == 15
    assert paired["significant"] is False


def test_paired_delta_small_n_never_significant():
    parent = [_pt(f"d{i}", 1.0, 1.0) for i in range(5)]
    child = [_pt(f"d{i}", 0.5, 0.5) for i in range(5)]
    paired = paired_per_task_delta(parent, child)
    assert paired["n"] == 5
    assert paired["significant"] is False  # below PAIRED_MIN_TASKS


def test_task_metric_is_geomean_of_pair():
    assert task_metric(4.0, 1.0) == pytest.approx(2.0)


# ── Task 3: multi-seed eval averaging ────────────────────────────────


def test_eval_seeds_env(monkeypatch):
    monkeypatch.delenv("RADAR_EVAL_SEEDS", raising=False)
    assert eval_seeds() == 1
    monkeypatch.setenv("RADAR_EVAL_SEEDS", "3")
    assert eval_seeds() == 3
    monkeypatch.setenv("RADAR_EVAL_SEEDS", "99")
    assert eval_seeds() == 8  # clamped
    monkeypatch.setenv("RADAR_EVAL_SEEDS", "junk")
    assert eval_seeds() == 1


def test_average_per_task_single_run_passthrough():
    run = [_pt("a", 0.5, 0.8), _pt("b", 0.2, 0.4)]
    out = average_per_task([run])
    assert {t["name"] for t in out} == {"a", "b"}
    assert out[0]["ncrps"] == 0.5


def test_average_per_task_means_across_seeds():
    r1 = [_pt("a", 0.4, 0.8), _pt("b", 0.2, 0.6)]
    r2 = [_pt("a", 0.6, 1.2), _pt("b", 0.4, 0.4)]
    out = average_per_task([r1, r2])
    by = {t["name"]: t for t in out}
    assert by["a"]["ncrps"] == pytest.approx(0.5)
    assert by["a"]["nmase"] == pytest.approx(1.0)
    assert by["b"]["ncrps"] == pytest.approx(0.3)


def test_average_per_task_uses_common_datasets_only():
    r1 = [_pt("a", 1.0, 1.0), _pt("b", 1.0, 1.0)]
    r2 = [_pt("a", 1.0, 1.0)]  # b missing from this seed
    out = average_per_task([r1, r2])
    assert [t["name"] for t in out] == ["a"]
