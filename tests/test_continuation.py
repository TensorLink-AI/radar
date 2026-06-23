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
    is_extend_round,
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


def test_checkpoint_optimizer_sidecar_roundtrip(tmp_path):
    # The harness writes optim_state.pt next to model.safetensors; the store
    # must copy it, resolve it back, and gc it alongside the weights.
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    src = workdir / "model.safetensors"
    src.write_bytes(b"weights")
    (workdir / "optim_state.pt").write_bytes(b"optim-moments")

    ref = cs.save(11, src)
    assert ref == "ckpt:11"
    optim_path = cs.resolve_optimizer(ref)
    assert optim_path is not None
    assert Path(optim_path).read_bytes() == b"optim-moments"

    # gc removes both the weights and the sidecar when the id isn't kept.
    removed = cs.gc(keep_ids=set())
    assert removed == 2
    assert cs.resolve("ckpt:11") is None
    assert cs.resolve_optimizer("ckpt:11") is None


def test_checkpoint_optimizer_sidecar_absent_is_none(tmp_path):
    # A parent that predates the sidecar (weights only) resolves to None —
    # the continuation then trains the optimizer from scratch.
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    src = tmp_path / "model.safetensors"
    src.write_bytes(b"w")
    cs.save(3, src)
    assert cs.resolve("ckpt:3") is not None
    assert cs.resolve_optimizer("ckpt:3") is None


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

def test_successful_round_count(store):
    _add(store, metric=1.0)                       # round 1 success
    _add(store, metric=0.9, success=False, ckpt=False)  # round 1 (n_rounds=1) — same round_id
    _add(store, metric=0.8, n_rounds=2)           # round 2 success
    # _add uses n_rounds as round_id; distinct successful rounds = {1, 2}.
    assert store.successful_round_count(task="ts_forecasting") == 2


def test_attempted_round_count_includes_failures(store):
    _add(store, metric=1.0)                                   # round 1 success
    _add(store, metric=0.9, success=False, ckpt=False, n_rounds=2)  # round 2 failure
    _add(store, metric=None, success=False, ckpt=False, n_rounds=3) # round 3 failure
    # Failures advance the cadence clock; successes don't double-count.
    assert store.attempted_round_count(task="ts_forecasting") == 3
    assert store.successful_round_count(task="ts_forecasting") == 1


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


def test_prepare_continuation_resumes_steps_and_optimizer(store, tmp_path):
    # An extend/continuation must carry the parent's cumulative step count
    # (so the LR schedule spans the lineage) and the optimizer sidecar path
    # (so AdamW moments resume) — the fix for Δ≈0 extend rounds.
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    src = workdir / "model.safetensors"
    src.write_bytes(b"w")
    (workdir / "optim_state.pt").write_bytes(b"moments")

    e1 = store.add_experiment(
        round_id=1, miner_id="m", name="x", code="c", motivation="",
        reasoning="", tool_calls=[], metric=1.0, success=True,
        objectives={"flops_equivalent_size": 100, "cumulative_compute": 5.0,
                    "cumulative_steps": 4200},
        score=0.0, loss_curve=[], task="ts_forecasting",
        cumulative_compute=5.0,
    )
    cs.save(e1, src)
    store.set_checkpoint_ref(e1, f"ckpt:{e1}")

    prep = prepare_continuation(
        store, cs, payload={"mode": "continue", "parent_index": e1},
        task_name="ts_forecasting", min_flops=100, max_flops=100,
        pool=[], shards_per_round=0, seed=1,
    )
    assert prep["mode"] == "continue"
    assert prep["step_offset"] == 4200
    assert prep["parent_optimizer_state_path"] is not None
    assert Path(prep["parent_optimizer_state_path"]).read_bytes() == b"moments"


def test_prepare_continuation_optimizer_path_none_when_absent(store, tmp_path):
    # Parent has weights but no optimizer sidecar → path is None, step_offset
    # falls back to 0 (no cumulative_steps recorded).
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    src = tmp_path / "m.safetensors"
    src.write_bytes(b"w")
    e1 = _add(store, metric=1.0, cumc=5.0)
    cs.save(e1, src)
    store.set_checkpoint_ref(e1, f"ckpt:{e1}")
    prep = prepare_continuation(
        store, cs, payload={"mode": "continue", "parent_index": e1},
        task_name="ts_forecasting", min_flops=100, max_flops=100,
        pool=[], shards_per_round=0, seed=1,
    )
    assert prep["mode"] == "continue"
    assert prep["parent_optimizer_state_path"] is None
    assert prep["step_offset"] == 0


def test_prepare_continuation_rejects_bad_parent(store, tmp_path):
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    prep = prepare_continuation(
        store, cs, payload={"mode": "continue", "parent_index": 999},
        task_name="ts_forecasting", min_flops=100, max_flops=100,
        pool=[], shards_per_round=0, seed=1,
    )
    assert prep["mode"] == "new"
    assert "rejected" in prep["note"]


def test_prepare_continuation_force_assigns_parent(store, tmp_path):
    # On a validator-mandated continuation round, a miner that sends
    # mode=new must still be warm-started from a seeded-random eligible
    # parent. The miner only chooses which parent; the validator decides
    # whether the round is a continuation.
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    src = tmp_path / "m.safetensors"
    src.write_bytes(b"w")
    e1 = _add(store, metric=1.0, cumc=5.0)
    cs.save(e1, src)
    store.set_checkpoint_ref(e1, f"ckpt:{e1}")
    e2 = _add(store, metric=0.9, cumc=3.0)
    cs.save(e2, src)
    store.set_checkpoint_ref(e2, f"ckpt:{e2}")

    prep = prepare_continuation(
        store, cs, payload={"mode": "new"},
        task_name="ts_forecasting", min_flops=100, max_flops=100,
        pool=[], shards_per_round=0, seed=1,
        force_continuation=True, eligible_parent_ids=[e1, e2],
        miner_id="alice",
    )
    assert prep["mode"] == "continue"
    assert prep["parent_index"] in {e1, e2}
    assert prep["parent_checkpoint_path"] is not None
    assert "auto-assigned" in prep["note"]


# ── extend vs modify split ───────────────────────────────────────────

def test_is_extend_round_bounds_and_split():
    # Degenerate rates short-circuit.
    assert all(not is_extend_round(r, 0.0) for r in range(50))
    assert all(is_extend_round(r, 1.0) for r in range(50))
    # Deterministic per round_id.
    assert is_extend_round(7, 0.5) == is_extend_round(7, 0.5)
    # ~half over many rounds at 0.5.
    hits = sum(is_extend_round(r, 0.5) for r in range(2000))
    assert 850 < hits < 1150
    # Independent namespace from the new-vs-continuation schedule.
    from local.continuation import is_continuation_round
    same = sum(
        is_extend_round(r, 0.5) == is_continuation_round(r, 1000)
        for r in range(2000)
    )
    assert 850 < same < 1150  # uncorrelated → ~half agree by chance


def _add_with_ckpt(store, cs, tmp_path, **kw):
    src = tmp_path / "m.safetensors"
    if not src.exists():
        src.write_bytes(b"w")
    eid = _add(store, **kw)
    cs.save(eid, src)
    store.set_checkpoint_ref(eid, f"ckpt:{eid}")
    return eid


def test_prepare_continuation_extend_reuses_parent_code(store, tmp_path):
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    e1 = _add_with_ckpt(store, cs, tmp_path, metric=1.0, cumc=5.0,
                        task="synthetic_data_generator")
    prep = prepare_continuation(
        store, cs, payload={"mode": "continue", "parent_index": e1},
        task_name="synthetic_data_generator", min_flops=0, max_flops=0,
        pool=[], shards_per_round=0, seed=1, extend=True,
    )
    assert prep["mode"] == "continue"
    assert prep["continuation_kind"] == "extend"
    # Trains on the parent's own generator code, not the miner's submission.
    assert prep["train_code"] == "c"


def test_prepare_continuation_modify_ignores_parent_code(store, tmp_path):
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    e1 = _add_with_ckpt(store, cs, tmp_path, metric=1.0, cumc=5.0,
                        task="synthetic_data_generator")
    prep = prepare_continuation(
        store, cs, payload={"mode": "continue", "parent_index": e1},
        task_name="synthetic_data_generator", min_flops=0, max_flops=0,
        pool=[], shards_per_round=0, seed=1, extend=False,
    )
    assert prep["continuation_kind"] == "modify"
    assert prep["train_code"] is None


def test_prepare_continuation_force_falls_back_when_miner_picks_bad_parent(
    store, tmp_path,
):
    # Miner sent a bogus parent_index on a forced-continuation round →
    # validator records the rejection and still warm-starts from an
    # eligible alternative.
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    src = tmp_path / "m.safetensors"
    src.write_bytes(b"w")
    e1 = _add(store, metric=1.0, cumc=5.0)
    cs.save(e1, src)
    store.set_checkpoint_ref(e1, f"ckpt:{e1}")

    prep = prepare_continuation(
        store, cs, payload={"mode": "continue", "parent_index": 999},
        task_name="ts_forecasting", min_flops=100, max_flops=100,
        pool=[], shards_per_round=0, seed=2,
        force_continuation=True, eligible_parent_ids=[e1],
        miner_id="bob",
    )
    assert prep["mode"] == "continue"
    assert prep["parent_index"] == e1


def test_prepare_continuation_force_deterministic_per_miner(store, tmp_path):
    # Same (seed, miner_id) must pick the same parent so retries are
    # reproducible; different miners spread across parents.
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    src = tmp_path / "m.safetensors"
    src.write_bytes(b"w")
    ids = []
    for _ in range(5):
        e = _add(store, metric=1.0, cumc=1.0)
        cs.save(e, src)
        store.set_checkpoint_ref(e, f"ckpt:{e}")
        ids.append(e)

    def pick(miner):
        return prepare_continuation(
            store, cs, payload={"mode": "new"},
            task_name="ts_forecasting", min_flops=100, max_flops=100,
            pool=[], shards_per_round=0, seed=42,
            force_continuation=True, eligible_parent_ids=ids,
            miner_id=miner,
        )["parent_index"]

    assert pick("alice") == pick("alice")
    # Across many miners we should see >1 distinct parent picked.
    picks = {pick(f"m{i}") for i in range(20)}
    assert len(picks) > 1


def test_prepare_continuation_force_no_eligible_stays_new(store, tmp_path):
    cs = CheckpointStore(base_dir=tmp_path / "ck")
    prep = prepare_continuation(
        store, cs, payload={"mode": "new"},
        task_name="ts_forecasting", min_flops=100, max_flops=100,
        pool=[], shards_per_round=0, seed=1,
        force_continuation=True, eligible_parent_ids=[],
        miner_id="alice",
    )
    assert prep["mode"] == "new"
    assert prep["parent_index"] is None


# ── agent heuristic ──────────────────────────────────────────────────

def test_tail_descending():
    from local.agent import _tail_descending
    assert _tail_descending([1.0, 0.9, 0.8, 0.5])      # clearly falling
    assert not _tail_descending([0.5, 0.5, 0.5, 0.5])  # flat
    assert not _tail_descending([0.5, 0.6])            # too few points


def test_choose_continuation_picks_descending_parent():
    from local.agent import _choose_continuation
    ch = {
        "round_type": "continuation",
        "eligible_parents": [
            {"id": 3, "metric": 0.9, "checkpoint_available": True,
             "loss_curve_tail": [1.0, 0.9, 0.8, 0.5]},   # descending
            {"id": 4, "metric": 0.7, "checkpoint_available": True,
             "loss_curve_tail": [0.5, 0.5, 0.5, 0.5]},   # plateaued
        ],
    }
    # Prefers the still-descending parent even though #4 has a better metric.
    mode, pid = _choose_continuation(ch)
    assert mode == "continue" and pid == 3


def test_choose_continuation_falls_back_to_best_when_none_descending():
    from local.agent import _choose_continuation
    ch = {
        "round_type": "continuation",
        "eligible_parents": [
            {"id": 3, "metric": 0.9, "checkpoint_available": True,
             "loss_curve_tail": [0.9, 0.9, 0.9, 0.9]},
            {"id": 4, "metric": 0.7, "checkpoint_available": True,
             "loss_curve_tail": [0.7, 0.7, 0.7, 0.7]},
        ],
    }
    # Validator mandated continuation → continue with the best metric (#4).
    assert _choose_continuation(ch) == ("continue", 4)


def test_choose_continuation_new_round_is_fresh():
    from local.agent import _choose_continuation
    ch = {"round_type": "new", "eligible_parents": [
        {"id": 3, "metric": 0.9, "checkpoint_available": True,
         "loss_curve_tail": [1.0, 0.5]}]}
    assert _choose_continuation(ch) == ("new", None)


# ── validator-scheduled cadence ──────────────────────────────────────

def test_continuation_rate_warmup_then_staircase():
    from local.continuation import continuation_rate
    # 0% through the whole warmup window.
    assert continuation_rate(0) == 0.0
    assert continuation_rate(19) == 0.0
    assert continuation_rate(20) == pytest.approx(0.0)   # ramp starts here
    # +2% per 3 attempted rounds after warmup.
    assert continuation_rate(23) == pytest.approx(0.02)
    assert continuation_rate(50) == pytest.approx(0.20)
    # Reaches the 0.70 equilibrium ~105 rounds after warmup, then holds.
    assert continuation_rate(20 + 105) == pytest.approx(0.70)
    assert continuation_rate(20 + 999) == pytest.approx(0.70)


def test_is_continuation_round_warmup_and_tracking():
    from local.continuation import is_continuation_round
    # Inside warmup: never a continuation regardless of round_id.
    assert not any(is_continuation_round(r, 19) for r in range(200))
    # Deterministic per (round_id, attempted_rounds).
    a = is_continuation_round(7, 100 + 350)
    b = is_continuation_round(7, 100 + 350)
    assert a == b
    # Realized frequency at the equilibrium tracks 0.70.
    hits = sum(is_continuation_round(r, 100 + 350) for r in range(2000))
    assert 0.65 < hits / 2000 < 0.75


# ── harness offset helper ────────────────────────────────────────────

def test_offset_history():
    from runner.harness import _offset_history
    hist = [{"step": 5, "loss": 1.0, "flops": 100}]
    assert _offset_history(hist, 0, 0) is hist  # no-op identity
    out = _offset_history(hist, 10, 1000)
    assert out[0]["step"] == 15 and out[0]["flops"] == 1100
    assert hist[0]["step"] == 5  # original untouched
