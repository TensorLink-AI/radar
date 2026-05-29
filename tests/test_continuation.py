"""Tests for continuation-training: checkpoint store, disjoint shards,
two-frontier scoring, lineage helpers, trajectory stitching, and the
agent's continue/new heuristic. All numpy-only — no torch required."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest

from local.checkpoints import CheckpointStore, read_signature
from local.continuation import (
    continuation_frontier,
    is_continuation,
    prepare_continuation,
    score_continuation,
)
from local.scoring import score_round
from local.shards import assign_shards, lineage_shards
from local.store import LocalStore


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path):
    s = LocalStore(tmp_path / "t.db")
    yield s
    s.close()


def _add(store, *, metric, mode="new", parent=None, n_rounds=1,
         cumc=0.0, shards=None, delta=None, flops=100, success=True,
         ckpt=True, task="ts_forecasting", loss_curve=None):
    objs = {"flops_equivalent_size": flops, "num_params": 10,
            "pretrain_shards": shards or []}
    if delta is not None:
        objs["delta"] = delta
    objs["cumulative_compute"] = cumc
    eid = store.add_experiment(
        round_id=n_rounds, miner_id="m", name="x", code="c", motivation="",
        reasoning="", tool_calls=[], metric=metric, success=success,
        objectives=objs, score=0.0, loss_curve=loss_curve or [],
        parent_index=parent, task=task, n_rounds=n_rounds,
        cumulative_compute=cumc, mode=mode,
    )
    if ckpt and success:
        store.set_checkpoint_ref(eid, f"ckpt:{eid}")
    return eid


# ── checkpoint store ─────────────────────────────────────────────────

def test_checkpoint_save_resolve_gc(tmp_path):
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    src = tmp_path / "model.safetensors"
    src.write_bytes(b"weights")
    ref = cs.save(7, src)
    assert ref == "ckpt:7"
    assert Path(cs.resolve(ref)).read_bytes() == b"weights"
    # gc keeps 7, drops 8 (which doesn't exist) — no error, keeps 7.
    cs.save(8, src)
    removed = cs.gc(keep_ids={7})
    assert removed == 1
    assert cs.resolve("ckpt:7") is not None
    assert cs.resolve("ckpt:8") is None


def test_checkpoint_save_missing_source(tmp_path):
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    assert cs.save(1, tmp_path / "nope.safetensors") is None
    assert cs.resolve("ckpt:1") is None
    assert cs.resolve(None) is None
    assert cs.resolve("notaref") is None


def test_read_signature(tmp_path):
    # Hand-build a minimal safetensors file: 8-byte LE header len + JSON.
    header = {"layer.weight": {"dtype": "F32", "shape": [4, 8], "data_offsets": [0, 128]},
              "__metadata__": {"x": "y"}}
    blob = json.dumps(header).encode()
    p = tmp_path / "m.safetensors"
    with p.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00" * 128)
    sig = read_signature(p)
    assert sig == {"layer.weight": [4, 8]}
    assert read_signature(tmp_path / "absent") == {}


# ── shard assignment ─────────────────────────────────────────────────

def test_assign_shards_fresh_deterministic():
    pool = [f"/d/shard-{i:02d}.parquet" for i in range(10)]
    a, reused_a = assign_shards(pool, lineage_used=set(), n=4, seed=1)
    b, reused_b = assign_shards(pool, lineage_used=set(), n=4, seed=1)
    assert a == b and len(a) == 4 and not reused_a and not reused_b


def test_assign_shards_continuation_disjoint():
    pool = [f"/d/shard-{i:02d}.parquet" for i in range(10)]
    used = {"shard-00.parquet", "shard-01.parquet", "shard-02.parquet"}
    chosen, reused = assign_shards(pool, lineage_used=used, n=4, seed=2)
    assert not reused
    assert all(Path(p).name not in used for p in chosen)


def test_assign_shards_exhaustion_allows_reuse():
    pool = [f"/d/shard-{i:02d}.parquet" for i in range(3)]
    used = {"shard-00.parquet", "shard-01.parquet", "shard-02.parquet"}
    chosen, reused = assign_shards(pool, lineage_used=used, n=2, seed=3)
    assert reused and len(chosen) == 2  # fell back to full pool


def test_lineage_shards_union():
    exps = [
        {"objectives": {"pretrain_shards": ["a.parquet", "b.parquet"]}},
        {"objectives": {"pretrain_shards": ["/x/b.parquet", "c.parquet"]}},
    ]
    assert lineage_shards(exps) == {"a.parquet", "b.parquet", "c.parquet"}


# ── continuation frontier + scoring ──────────────────────────────────

def test_is_continuation():
    assert is_continuation({"mode": "continue"})
    assert is_continuation({"n_rounds": 2, "parent_index": 5})
    assert not is_continuation({"mode": "new", "n_rounds": 1})


def test_continuation_frontier_dominance():
    exps = [
        {"success": True, "mode": "continue", "cumulative_compute": 10,
         "objectives": {"delta": 0.5, "cumulative_compute": 10}},
        # dominated: more compute, smaller delta
        {"success": True, "mode": "continue", "cumulative_compute": 20,
         "objectives": {"delta": 0.3, "cumulative_compute": 20}},
        # on frontier: more compute but larger delta
        {"success": True, "mode": "continue", "cumulative_compute": 30,
         "objectives": {"delta": 0.9, "cumulative_compute": 30}},
    ]
    front = continuation_frontier(exps)
    deltas = sorted(e["objectives"]["delta"] for e in front)
    assert deltas == [0.5, 0.9]


def test_score_continuation_no_improvement_scores_zero():
    score, delta = score_continuation(
        metric=1.0, parent_metric=0.8, cumulative_compute=5, frontier=[],
    )
    assert score == 0.0 and delta < 0


def test_score_continuation_rewards_progress():
    score, delta = score_continuation(
        metric=0.6, parent_metric=1.0, cumulative_compute=5, frontier=[],
    )
    assert delta == pytest.approx(0.4) and score > 0.5


def test_score_round_dispatches_continuation():
    cont = {"success": True, "metric": 0.6, "mode": "continue",
            "parent_metric": 1.0,
            "objectives": {"flops_equivalent_size": 100, "cumulative_compute": 5}}
    fresh = {"success": True, "metric": 0.9, "mode": "new",
             "objectives": {"flops_equivalent_size": 100}}
    out = score_round([cont, fresh], min_flops=100, max_flops=100,
                      frontier=[], continuation_frontier=[])
    assert out[0]["score"] > 0  # continuation scored on delta
    assert "continuation" in out[0]["analysis"]
    assert out[1]["score"] >= 0  # fresh scored on absolute frontier


def test_scoring_never_reads_val_loss():
    # A continuation whose only "signal" is val loss must still score on Δ:
    # remove any val loss and confirm score depends purely on metric delta.
    p = {"success": True, "metric": 0.5, "mode": "continue",
         "parent_metric": 1.0,
         "objectives": {"flops_equivalent_size": 100, "cumulative_compute": 1,
                        "best_val_loss": 999.0}}
    out = score_round([p], min_flops=100, max_flops=100, frontier=[],
                      continuation_frontier=[])
    # best_val_loss is huge but score is driven by Δ=0.5 → positive.
    assert out[0]["score"] > 0.5


# ── store lineage / eligible parents ─────────────────────────────────

def test_store_lineage_and_eligible(store):
    e1 = _add(store, metric=1.0, shards=["s1.parquet"])
    e2 = _add(store, metric=0.8, mode="continue", parent=e1, n_rounds=2,
              cumc=5.0, shards=["s2.parquet"], delta=0.2)
    chain = store.lineage(e2)
    assert [e["id"] for e in chain] == [e1, e2]
    elig = store.eligible_parents(task="ts_forecasting", min_flops=100, max_flops=100)
    assert {e["id"] for e in elig} == {e1, e2}
    # A failed / checkpoint-less experiment is not eligible.
    e3 = _add(store, metric=0.7, ckpt=False)
    elig_ids = {e["id"] for e in store.eligible_parents(
        task="ts_forecasting", min_flops=100, max_flops=100)}
    assert e3 not in elig_ids


# ── trajectory stitching ─────────────────────────────────────────────

def test_trajectory_stitches_lineage(store):
    from local import experiments_api as api
    e1 = _add(store, metric=1.0, loss_curve=[1.0, 0.8, 0.6])
    e2 = _add(store, metric=0.7, mode="continue", parent=e1, n_rounds=2,
              cumc=5.0, delta=0.3, loss_curve=[0.5, 0.4])
    traj = api.trajectory(store, e2)
    assert traj["lineage"] == [e1, e2]
    assert len(traj["loss_curve"]) == 5            # 3 + 2 stitched
    assert traj["loss_curve"][0]["experiment_id"] == e1
    assert traj["loss_curve"][-1]["experiment_id"] == e2
    assert [b["start_index"] for b in traj["boundaries"]] == [0, 3]
    assert len(traj["gift_eval"]) == 2


def test_signature_endpoint(store):
    e1 = store.add_experiment(
        round_id=0, miner_id="m", name="x", code="c", motivation="",
        reasoning="", tool_calls=[], metric=1.0, success=True,
        objectives={"flops_equivalent_size": 100,
                    "param_signature": {"w": [4, 8]}},
        score=0.0, loss_curve=[], task="ts_forecasting")
    from local import experiments_api as api
    sig = api.signature(store, e1)
    assert sig["signature"] == {"w": [4, 8]}


# ── prepare_continuation (validator-side) ────────────────────────────

def test_prepare_continuation_valid(store, tmp_path):
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    src = tmp_path / "m.safetensors"
    src.write_bytes(b"w")
    e1 = _add(store, metric=1.0, cumc=5.0, shards=["s1.parquet"])
    cs.save(e1, src)
    store.set_checkpoint_ref(e1, f"ckpt:{e1}")
    pool = [f"/d/shard-{i:02d}.parquet" for i in range(6)]
    prep = prepare_continuation(
        store, cs, payload={"mode": "continue", "parent_index": e1},
        task_name="ts_forecasting", min_flops=100, max_flops=100,
        pool=pool, shards_per_round=2, seed=1,
    )
    assert prep["mode"] == "continue"
    assert prep["parent_metric"] == 1.0
    assert prep["compute_offset"] == 5.0
    assert prep["n_rounds"] == 2
    assert prep["parent_checkpoint_path"] is not None


def test_prepare_continuation_rejects_bad_parent(store, tmp_path):
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    prep = prepare_continuation(
        store, cs, payload={"mode": "continue", "parent_index": 999},
        task_name="ts_forecasting", min_flops=100, max_flops=100,
        pool=[], shards_per_round=0, seed=1,
    )
    assert prep["mode"] == "new"
    assert "rejected" in prep["note"]


# ── agent heuristic ──────────────────────────────────────────────────

def test_tail_descending():
    from local.agent import _tail_descending
    assert _tail_descending([1.0, 0.9, 0.8, 0.5])      # clearly falling
    assert not _tail_descending([0.5, 0.5, 0.5, 0.5])  # flat
    assert not _tail_descending([0.5, 0.6])            # too few points


def test_choose_continuation_picks_descending_parent():
    from local.agent import _choose_continuation
    ch = {
        "continuation_allowed": True,
        "eligible_parents": [
            {"id": 3, "metric": 0.9, "checkpoint_available": True,
             "loss_curve_tail": [1.0, 0.9, 0.8, 0.5]},   # descending
            {"id": 4, "metric": 0.7, "checkpoint_available": True,
             "loss_curve_tail": [0.5, 0.5, 0.5, 0.5]},   # plateaued
        ],
    }
    mode, pid = _choose_continuation(ch)
    assert mode == "continue" and pid == 3


def test_choose_continuation_off_by_default():
    from local.agent import _choose_continuation
    assert _choose_continuation({"continuation_allowed": False}) == ("new", None)


# ── harness offset helper ────────────────────────────────────────────

def test_offset_history():
    from runner.harness import _offset_history
    hist = [{"step": 5, "loss": 1.0, "flops": 100}]
    assert _offset_history(hist, 0, 0) is hist  # no-op identity
    out = _offset_history(hist, 10, 1000)
    assert out[0]["step"] == 15 and out[0]["flops"] == 1100
    assert hist[0]["step"] == 5  # original untouched
