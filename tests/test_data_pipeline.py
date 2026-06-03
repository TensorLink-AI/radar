"""Tests for the ts_data_pipeline task: frozen-arch snapshots, dashboard
endpoints, scoring helpers. The trainer dispatch path needs torch +
GIFT-Eval data and is exercised in integration runs only.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from local.data_pipeline import _compute_aulc
from local.dashboard import _data_pipeline_frontier, _frozen_archs
from local.frozen_arch import FrozenArchStore, maybe_refresh
from local.scoring import passes_size_gate
from local.store import LocalStore
from local.task import (
    TSDataPipelineSpec, TSForecastingSpec, buckets_for, make_spec,
)


def test_make_spec_ts_data_pipeline():
    spec = make_spec("ts_data_pipeline")
    assert isinstance(spec, TSDataPipelineSpec)
    assert spec.context_len == 512
    assert spec.prediction_len == 96


def test_buckets_for_data_pipeline_matches_ts():
    dp = make_spec("ts_data_pipeline")
    ts = make_spec("ts_forecasting")
    assert buckets_for(dp) == buckets_for(ts)


def test_passes_size_gate_disabled_with_zero_bounds():
    # Both bounds 0 = gate disabled (ts_data_pipeline runs the frozen arch
    # whose flops are fixed and can't be re-bucketed).
    assert passes_size_gate({"flops_equivalent_size": 1_234_567}, 0, 0)
    # Lower bound 0 only should still gate on the upper bound.
    assert passes_size_gate({"flops_equivalent_size": 5}, 10, 100) is False


def test_compute_aulc_returns_per_step_area():
    # Linear loss from 1.0 → 0.0 over steps 0 → 10 → AULC = mean = 0.5.
    curve = [{"step": s, "loss": 1.0 - s / 10.0} for s in range(11)]
    aulc = _compute_aulc(curve)
    assert aulc == pytest.approx(0.5, abs=1e-9)


def test_compute_aulc_rewards_fast_descent():
    # Curve that drops fast (steep early decay) should have lower AULC than
    # a curve that drops late, even when both end at the same loss.
    steep = [{"step": s, "loss": math.exp(-s)} for s in range(20)]
    late = [{"step": s, "loss": 1.0 if s < 19 else math.exp(-19)}
            for s in range(20)]
    assert _compute_aulc(steep) < _compute_aulc(late)


def test_compute_aulc_handles_sparse_curve():
    assert _compute_aulc([]) is None
    assert _compute_aulc([{"step": 0, "loss": 1.0}]) is None  # one point
    # Degenerate single-step range → None
    assert _compute_aulc([
        {"step": 5, "loss": 1.0}, {"step": 5, "loss": 0.5}
    ]) is None


def test_frozen_arch_store_bootstrap(tmp_path):
    store = FrozenArchStore(base_dir=str(tmp_path))
    assert store.current_version() == 0
    assert store.current() is None
    assert store.all_versions() == []


def test_frozen_arch_save_and_load(tmp_path):
    store = FrozenArchStore(base_dir=str(tmp_path))
    arch = store.save(
        code="def build_model(*a, **k): pass\n",
        source_experiment_id=42,
        source_metric=0.123,
        source_crps=0.1,
        source_mase=0.2,
        source_flops=1_000_000,
        source_name="tiny_transformer",
    )
    assert arch.version == 1
    assert store.current_version() == 1
    loaded = store.current()
    assert loaded is not None
    assert loaded.source_experiment_id == 42
    assert loaded.source_metric == pytest.approx(0.123)
    versions = store.all_versions()
    assert len(versions) == 1
    assert versions[0]["version"] == 1
    assert "code" not in versions[0]   # all_versions strips code


def test_frozen_arch_bootstraps_from_ts_forecasting(tmp_path):
    db = LocalStore(str(tmp_path / "test.db"))
    db.add_experiment(
        round_id=0, miner_id="m0", name="arch_a", code="ARCH_A_CODE",
        motivation="", reasoning="", tool_calls=[],
        metric=0.5, success=True,
        objectives={"crps": 0.3, "mase": 0.7, "flops_equivalent_size": 5_000_000},
        score=0.8, loss_curve=[], analysis="",
        task="ts_forecasting",
    )
    db.add_experiment(
        round_id=1, miner_id="m1", name="arch_b", code="ARCH_B_CODE",
        motivation="", reasoning="", tool_calls=[],
        metric=0.4, success=True,
        objectives={"crps": 0.25, "mase": 0.6, "flops_equivalent_size": 6_000_000},
        score=0.9, loss_curve=[], analysis="",
        task="ts_forecasting",
    )
    arch_store = FrozenArchStore(base_dir=str(tmp_path / "arch"))
    snap = maybe_refresh(arch_store, db, every_n=50)
    assert snap is not None
    # Lowest metric wins → arch_b.
    assert snap.source_name == "arch_b"
    assert snap.source_metric == pytest.approx(0.4)
    db.close()


def test_frozen_arch_no_refresh_without_threshold(tmp_path):
    db = LocalStore(str(tmp_path / "test.db"))
    db.add_experiment(
        round_id=0, miner_id="m0", name="arch_a", code="ARCH_A",
        motivation="", reasoning="", tool_calls=[],
        metric=0.5, success=True,
        objectives={"crps": 0.3, "mase": 0.7},
        score=0.8, loss_curve=[], analysis="",
        task="ts_forecasting",
    )
    arch_store = FrozenArchStore(base_dir=str(tmp_path / "arch"))
    # First call bootstraps.
    snap1 = maybe_refresh(arch_store, db, every_n=50)
    assert snap1 is not None
    # No new successful data-pipeline rounds → no refresh.
    snap2 = maybe_refresh(arch_store, db, every_n=50)
    assert snap2 is None
    assert arch_store.current_version() == 1
    db.close()


def test_data_pipeline_frontier_filters_by_task(tmp_path):
    db_path = tmp_path / "test.db"
    db = LocalStore(str(db_path))
    # ts_forecasting row (should be ignored).
    db.add_experiment(
        round_id=0, miner_id="m0", name="ts_a", code="",
        motivation="", reasoning="", tool_calls=[],
        metric=0.5, success=True,
        objectives={"crps": 0.3, "mase": 0.7},
        score=0.8, loss_curve=[], analysis="",
        task="ts_forecasting",
    )
    # Two data-pipeline rows with different (aulc, gift) values.
    db.add_experiment(
        round_id=1, miner_id="m1", name="dp_a", code="",
        motivation="", reasoning="", tool_calls=[],
        metric=0.4, success=True,
        objectives={"aulc": 1.0, "gift_metric": 0.5,
                    "frozen_arch_version": 1},
        score=0.7, loss_curve=[], analysis="",
        task="ts_data_pipeline",
    )
    db.add_experiment(
        round_id=2, miner_id="m2", name="dp_b", code="",
        motivation="", reasoning="", tool_calls=[],
        metric=0.3, success=True,
        objectives={"aulc": 0.8, "gift_metric": 0.4,
                    "frozen_arch_version": 1},
        score=0.9, loss_curve=[], analysis="",
        task="ts_data_pipeline",
    )
    db.close()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        result = _data_pipeline_frontier(conn)
    finally:
        conn.close()
    names = {p["name"] for p in result["all"]}
    assert names == {"dp_a", "dp_b"}  # ts_forecasting row excluded
    # dp_b dominates dp_a on both axes.
    assert len(result["frontier"]) == 1
    assert result["frontier"][0]["name"] == "dp_b"
    assert result["versions"] == [1]


def test_frozen_archs_endpoint_lists_versions(tmp_path):
    arch_store = FrozenArchStore(base_dir=str(tmp_path / "arch"))
    arch_store.save(
        code="A", source_experiment_id=1, source_metric=0.5,
        source_crps=0.3, source_mase=0.7, source_flops=1_000_000,
        source_name="v1_arch",
    )
    arch_store.save(
        code="B", source_experiment_id=2, source_metric=0.4,
        source_crps=0.2, source_mase=0.6, source_flops=2_000_000,
        source_name="v2_arch",
    )
    rows = _frozen_archs(str(tmp_path / "arch"))
    assert [r["version"] for r in rows] == [1, 2]
    assert [r["source_name"] for r in rows] == ["v1_arch", "v2_arch"]
    # Code is stripped from the listing.
    assert all("code" not in r for r in rows)


def test_challenge_dict_for_data_pipeline_advertises_pipeline_contract():
    from local.validator import _task_dict
    spec = make_spec("ts_data_pipeline")
    d = _task_dict(spec)
    assert d["name"] == "ts_data_pipeline"
    assert d["runner_dir"] == "ts_data_pipeline"
    # The miner contract is build_pipeline, not build_model.
    assert any("build_pipeline" in c for c in d["constraints"])
    obj_names = {o["name"] for o in d["objectives"]}
    assert "aulc" in obj_names
    assert "gift_metric" in obj_names
