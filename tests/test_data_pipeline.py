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

from local.data_pipeline import (
    REFERENCE_PIPELINE_CODE, _compute_aulc, _exec_pipeline_submission,
    ensure_baseline_aulc,
)
from local.dashboard import (
    _data_pipeline_frontier, _frozen_archs, _synth_frontier,
)
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
    # Baseline is unset until the validator computes it lazily.
    assert loaded.baseline_aulc is None
    versions = store.all_versions()
    assert len(versions) == 1
    assert versions[0]["version"] == 1
    assert "code" not in versions[0]   # all_versions strips code


def test_frozen_arch_update_baseline_aulc_round_trips(tmp_path):
    store = FrozenArchStore(base_dir=str(tmp_path))
    store.save(
        code="x", source_experiment_id=1, source_metric=0.5,
        source_crps=0.3, source_mase=0.7, source_flops=1_000_000,
        source_name="v1",
    )
    updated = store.update_baseline_aulc(1, 0.1234)
    assert updated is not None
    assert updated.baseline_aulc == pytest.approx(0.1234)
    # Survives a fresh store instance (persisted to JSON).
    reloaded = FrozenArchStore(base_dir=str(tmp_path)).current()
    assert reloaded is not None
    assert reloaded.baseline_aulc == pytest.approx(0.1234)


def test_frozen_arch_load_tolerates_legacy_json(tmp_path):
    """Pre-baseline snapshots on disk must still load with baseline_aulc=None."""
    base = tmp_path / "arch"
    base.mkdir()
    legacy = {
        "version": 1, "code": "x", "source_experiment_id": 1,
        "source_metric": 0.5, "source_crps": 0.3, "source_mase": 0.7,
        "source_flops": 1, "source_name": "v1", "created_at": 0.0,
        "unknown_future_field": "ignored",
    }
    (base / "v000001.json").write_text(json.dumps(legacy))
    (base / "manifest.json").write_text(json.dumps({"current": 1, "versions": [1]}))
    arch = FrozenArchStore(base_dir=str(base)).current()
    assert arch is not None
    assert arch.baseline_aulc is None


def test_ensure_baseline_aulc_returns_cached(tmp_path):
    """When baseline is already set the helper must not re-run training."""
    store = FrozenArchStore(base_dir=str(tmp_path))
    arch = store.save(
        code="x", source_experiment_id=1, source_metric=0.5,
        source_crps=0.3, source_mase=0.7, source_flops=1_000_000,
        source_name="v1",
    )
    store.update_baseline_aulc(arch.version, 0.42)
    arch = store.current()  # reload with cached baseline
    # task=None would crash any training path; cache hit must short-circuit.
    out = ensure_baseline_aulc(store, arch, task=None)
    assert out == pytest.approx(0.42)


def test_reference_pipeline_executes_and_yields_correct_shape():
    """The baseline-anchor pipeline must be a valid build_pipeline."""
    torch = pytest.importorskip("torch")
    build_pipeline = _exec_pipeline_submission(REFERENCE_PIPELINE_CODE)
    it = build_pipeline(64, 16, 1, (0.5,))
    batch = next(iter(it))
    assert "input" in batch and "target" in batch
    inp, tgt = batch["input"], batch["target"]
    assert torch.is_tensor(inp) and torch.is_tensor(tgt)
    assert inp.dim() == 3 and tgt.dim() == 3
    assert inp.shape[1] == 64 and tgt.shape[1] == 16
    assert inp.shape[2] == 1 and tgt.shape[2] == 1
    assert inp.shape[0] == tgt.shape[0]
    # Determinism: a second build draws from a freshly seeded generator,
    # so the first batch must match bit-for-bit.
    it2 = build_pipeline(64, 16, 1, (0.5,))
    batch2 = next(iter(it2))
    assert torch.equal(batch["input"], batch2["input"])
    assert torch.equal(batch["target"], batch2["target"])


def test_validating_batch_iter_sanitizes_nonfinite():
    """nan/inf in a miner batch (e.g. float32 overflow) are zeroed and the
    run continues, rather than failing the whole experiment."""
    torch = pytest.importorskip("torch")
    from local.data_pipeline import _ValidatingBatchIter

    inp = torch.zeros(2, 8, 1)
    tgt = torch.zeros(2, 4, 1)
    inp[0, 0, 0] = float("nan")
    inp[1, 1, 0] = float("inf")
    tgt[0, 0, 0] = float("-inf")

    it = _ValidatingBatchIter([{"input": inp, "target": tgt}], {}, 2)
    out = next(it)
    assert torch.isfinite(out["input"]).all()
    assert torch.isfinite(out["target"]).all()
    assert it._nonfinite_seen == 3
    # Shapes are preserved — sanitize, don't drop.
    assert tuple(out["input"].shape) == (2, 8, 1)
    assert tuple(out["target"].shape) == (2, 4, 1)


def test_validating_batch_iter_passes_clean_batches_through():
    torch = pytest.importorskip("torch")
    from local.data_pipeline import _ValidatingBatchIter

    inp = torch.randn(3, 8, 1)
    tgt = torch.randn(3, 4, 1)
    it = _ValidatingBatchIter([{"input": inp, "target": tgt}], {}, 3)
    out = next(it)
    assert it._nonfinite_seen == 0
    assert torch.equal(out["input"], inp)
    assert torch.equal(out["target"], tgt)


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


def test_synth_frontier_filters_by_task_and_lists_versions(tmp_path):
    db_path = tmp_path / "test.db"
    db = LocalStore(str(db_path))
    # ts_forecasting row shares crps/mase but must be excluded from the
    # synth-only frontier.
    db.add_experiment(
        round_id=0, miner_id="m0", name="ts_a", code="",
        motivation="", reasoning="", tool_calls=[],
        metric=0.5, success=True,
        objectives={"crps": 0.3, "mase": 0.7},
        score=0.8, loss_curve=[], analysis="",
        task="ts_forecasting",
    )
    # Two synth rows; sdg_b dominates sdg_a on both axes.
    db.add_experiment(
        round_id=1, miner_id="m1", name="sdg_a", code="",
        motivation="", reasoning="", tool_calls=[],
        metric=0.45, success=True,
        objectives={"crps": 0.4, "mase": 0.6, "synth_arch_version": 2},
        score=0.7, loss_curve=[], analysis="",
        task="synthetic_data_generator",
    )
    db.add_experiment(
        round_id=2, miner_id="m2", name="sdg_b", code="",
        motivation="", reasoning="", tool_calls=[],
        metric=0.35, success=True,
        objectives={"crps": 0.3, "mase": 0.5, "synth_arch_version": 2},
        score=0.9, loss_curve=[], analysis="",
        task="synthetic_data_generator",
    )
    db.close()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        result = _synth_frontier(conn)
    finally:
        conn.close()
    names = {p["name"] for p in result["all"]}
    assert names == {"sdg_a", "sdg_b"}  # ts_forecasting row excluded
    assert len(result["frontier"]) == 1
    assert result["frontier"][0]["name"] == "sdg_b"
    assert result["versions"] == [2]


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


# ── continuation epoch pinning ──────────────────────────────────────


def _add_dp_exp(store, *, frozen_arch_version: int, metric: float = 0.5,
                cumc: float = 0.0, success: bool = True,
                ckpt: bool = True) -> int:
    objs = {
        "flops_equivalent_size": 0,
        "num_params": 0,
        "frozen_arch_version": frozen_arch_version,
        "frozen_arch_source_id": 1,
        "cumulative_compute": cumc,
    }
    eid = store.add_experiment(
        round_id=0, miner_id="m", name="x", code="c", motivation="",
        reasoning="", tool_calls=[], metric=metric, success=success,
        objectives=objs, score=0.0, loss_curve=[],
        task="ts_data_pipeline", cumulative_compute=cumc, mode="new",
    )
    if ckpt and success:
        store.set_checkpoint_ref(eid, f"ckpt:{eid}")
    return eid


def test_eligible_parents_ignores_size_gate_when_disabled(tmp_path):
    """ts_data_pipeline rounds run with min/max flops both 0 (frozen-arch
    FLOPs are fixed and not comparable across miners). The eligible-parent
    query must treat that as 'gate disabled' rather than 'flops must be 0',
    or every scheduled continuation round silently downgrades to 'new'.
    """
    store = LocalStore(tmp_path / "t.db")
    parent_id = _add_dp_exp(store, frozen_arch_version=1)
    # Frozen arch's actual training FLOPs are non-zero.
    store._conn.execute(
        "UPDATE experiments SET objectives_json = ? WHERE id = ?",
        (json.dumps({
            "flops_equivalent_size": 5_000_000,
            "num_params": 0,
            "frozen_arch_version": 1,
            "frozen_arch_source_id": 1,
            "cumulative_compute": 0,
        }), parent_id),
    )
    elig = store.eligible_parents(
        task="ts_data_pipeline", min_flops=0, max_flops=0,
    )
    assert {e["id"] for e in elig} == {parent_id}
    store.close()


def test_prepare_continuation_rejects_cross_epoch_parent(tmp_path):
    from local.checkpoints import CheckpointStore
    from local.continuation import prepare_continuation

    store = LocalStore(tmp_path / "t.db")
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    src = tmp_path / "m.safetensors"
    src.write_bytes(b"w")

    parent_id = _add_dp_exp(store, frozen_arch_version=1)
    cs.save(parent_id, src)
    store.set_checkpoint_ref(parent_id, f"ckpt:{parent_id}")

    # Same-epoch continuation: accepted.
    prep = prepare_continuation(
        store, cs,
        payload={"mode": "continue", "parent_index": parent_id},
        task_name="ts_data_pipeline", min_flops=0, max_flops=0,
        pool=[], shards_per_round=0, seed=1,
        current_epoch={"frozen_arch_version": 1},
    )
    assert prep["mode"] == "continue"
    assert prep["parent_metric"] == 0.5

    # Cross-epoch (parent v1, round v2): rejected, falls back to fresh.
    prep2 = prepare_continuation(
        store, cs,
        payload={"mode": "continue", "parent_index": parent_id},
        task_name="ts_data_pipeline", min_flops=0, max_flops=0,
        pool=[], shards_per_round=0, seed=1,
        current_epoch={"frozen_arch_version": 2},
    )
    assert prep2["mode"] == "new"
    assert "different epoch" in prep2["note"]
    store.close()


def test_current_epoch_for_data_pipeline():
    from local.frozen_arch import FrozenArch
    from local.validator import _current_epoch

    dp = make_spec("ts_data_pipeline")
    ts = make_spec("ts_forecasting")
    arch = FrozenArch(
        version=7, code="x", source_experiment_id=1, source_metric=0.5,
        source_crps=0.3, source_mase=0.7, source_flops=1_000_000,
        source_name="v7", created_at=0.0,
    )
    assert _current_epoch(dp, arch) == {"frozen_arch_version": 7}
    # ts_forecasting has no epoch axis (yet); empty dict = pin is a no-op.
    assert _current_epoch(ts, None) == {}
    assert _current_epoch(dp, None) == {}


# ── interwoven task dispatch ────────────────────────────────────────


def test_parse_task_mixture_round_trip():
    from local.validator import _parse_task_mixture

    assert _parse_task_mixture("ts_forecasting") == [("ts_forecasting", 1.0)]
    assert _parse_task_mixture("ts_forecasting:0.6,ts_data_pipeline:0.4") == [
        ("ts_forecasting", 0.6), ("ts_data_pipeline", 0.4),
    ]
    # Whitespace and trailing commas tolerated.
    assert _parse_task_mixture(" a:1 , b:2 , ") == [("a", 1.0), ("b", 2.0)]


def test_parse_task_mixture_rejects_bad_input():
    from local.validator import _parse_task_mixture

    with pytest.raises(ValueError):
        _parse_task_mixture("")
    with pytest.raises(ValueError):
        _parse_task_mixture("ts_forecasting:nope")
    with pytest.raises(ValueError):
        _parse_task_mixture("ts_forecasting:-1")
    with pytest.raises(ValueError):
        _parse_task_mixture("ts_forecasting:0,ts_data_pipeline:0")


def test_pick_task_name_distribution_matches_weights():
    from local.validator import _pick_task_name

    mixture = [("a", 0.7), ("b", 0.3)]
    picks = [_pick_task_name(r, mixture) for r in range(5000)]
    a_share = picks.count("a") / len(picks)
    assert 0.65 < a_share < 0.75
    # Deterministic for same (round_id, mixture).
    assert _pick_task_name(42, mixture) == _pick_task_name(42, mixture)


def test_pick_task_name_singleton_is_identity():
    from local.validator import _pick_task_name

    assert _pick_task_name(0, [("only", 1.0)]) == "only"
    assert _pick_task_name(999, [("only", 5.0)]) == "only"


def test_parent_in_epoch_filters_eligible():
    from local.validator import _parent_in_epoch

    p1 = {"objectives": {"frozen_arch_version": 3}}
    p2 = {"objectives": {"frozen_arch_version": 4}}
    p3 = {"objectives": {}}
    assert _parent_in_epoch(p1, {"frozen_arch_version": 3})
    assert not _parent_in_epoch(p2, {"frozen_arch_version": 3})
    assert not _parent_in_epoch(p3, {"frozen_arch_version": 3})
    # Empty epoch is a no-op gate.
    assert _parent_in_epoch(p1, {})
    assert _parent_in_epoch(p3, {})


# ── frozen pipeline store + ts_forecasting epoch ────────────────────


def test_frozen_pipeline_store_bootstrap(tmp_path):
    from local.frozen_pipeline import FrozenPipelineStore

    s = FrozenPipelineStore(base_dir=str(tmp_path / "pipes"))
    assert s.current_version() == 0
    assert s.current() is None


def test_frozen_pipeline_save_and_load(tmp_path):
    from local.frozen_pipeline import FrozenPipelineStore

    s = FrozenPipelineStore(base_dir=str(tmp_path / "pipes"))
    pipe = s.save(
        code="def build_pipeline(*a, **k): pass",
        source_experiment_id=42, source_metric=0.4,
        source_crps=0.2, source_mase=0.6, source_flops=1_000_000,
        source_name="pipe_a", frozen_arch_version=3,
        shard_paths=[str(tmp_path / "s.parquet")],
    )
    assert pipe.version == 1 and s.current_version() == 1
    loaded = s.load(1)
    assert loaded is not None
    assert loaded.frozen_arch_version == 3
    assert loaded.shard_paths == [str(tmp_path / "s.parquet")]
    # Listing strips the code.
    rows = s.all_versions()
    assert len(rows) == 1
    assert "code" not in rows[0]


def test_frozen_pipeline_bootstraps_from_data_pipeline_frontier(tmp_path):
    from local.frozen_pipeline import FrozenPipelineStore, maybe_refresh

    store = LocalStore(tmp_path / "t.db")
    store.add_experiment(
        round_id=0, miner_id="m", name="p", code="code-A",
        motivation="", reasoning="", tool_calls=[],
        metric=0.5, success=True,
        objectives={"flops_equivalent_size": 0, "num_params": 0,
                    "crps": 0.4, "mase": 0.6,
                    "frozen_arch_version": 2},
        score=0.0, loss_curve=[], task="ts_data_pipeline",
    )
    pipe_store = FrozenPipelineStore(base_dir=str(tmp_path / "pipes"))
    # num_shards=0 short-circuits the rendering path so the test stays
    # numpy-only — we're verifying the bookkeeping, not pyarrow.
    saved = maybe_refresh(pipe_store, store, every_n=10, num_shards=0)
    assert saved is not None
    assert saved.frozen_arch_version == 2
    assert saved.source_metric == 0.5
    store.close()


def test_current_epoch_for_forecasting_includes_pipeline_version():
    from local.frozen_pipeline import FrozenPipeline
    from local.validator import _current_epoch

    ts = make_spec("ts_forecasting")
    pipe = FrozenPipeline(
        version=4, code="x", source_experiment_id=1, source_metric=0.5,
        source_crps=0.3, source_mase=0.7, source_flops=0,
        source_name="p", frozen_arch_version=2, created_at=0.0,
        shard_paths=["a.parquet"],
    )
    # ts_forecasting with a pipeline → pin the version.
    assert _current_epoch(ts, frozen_pipeline=pipe) == {"frozen_pipeline_version": 4}
    # ts_forecasting without a pipeline (the pure-real control track) →
    # no pin: empty epoch lets it pair with any other pure-real parent.
    assert _current_epoch(ts) == {}
