"""Tests for local/round_types.py — special-round scheduling, target
pickers, and the recipe_only AST freeze."""

from __future__ import annotations

from local.round_types import (
    enforce_recipe_only,
    pick_ablation_target,
    pick_replicate_source,
    pick_transfer_source,
    same_build_model,
    source_card,
    special_round_type,
)


def _exp(eid, metric, flops, *, task="synth_regression", mode="new",
         code="def build_model(i, o):\n    return None\n", success=True):
    return {
        "id": eid, "metric": metric, "mode": mode, "task": task,
        "success": success, "code": code,
        "objectives": {"flops_equivalent_size": flops},
    }


# ── scheduler ────────────────────────────────────────────────────────

def test_special_round_type_deterministic():
    kw = dict(replicate_pct=0.1, ablate_pct=0.1, recipe_pct=0.1,
              transfer_pct=0.1)
    for rid in range(50):
        assert special_round_type(rid, **kw) == special_round_type(rid, **kw)


def test_special_round_type_rates_roughly_track():
    kw = dict(replicate_pct=0.25, ablate_pct=0.0, recipe_pct=0.0,
              transfer_pct=0.0)
    hits = sum(
        1 for rid in range(2000) if special_round_type(rid, **kw) == "replicate"
    )
    assert 0.18 < hits / 2000 < 0.32


def test_special_round_type_zero_disables():
    assert all(
        special_round_type(rid) == "" for rid in range(100)
    )


def test_special_round_type_bands_are_disjoint():
    kw = dict(replicate_pct=0.25, ablate_pct=0.25, recipe_pct=0.25,
              transfer_pct=0.25)
    seen = {special_round_type(rid, **kw) for rid in range(500)}
    assert seen <= {"replicate", "ablate", "recipe_only", "transfer"}
    assert len(seen) == 4


# ── target pickers ───────────────────────────────────────────────────

def test_pick_replicate_source_prefers_frontier_skips_replicates():
    exps = [
        _exp(1, 0.5, 1000),
        _exp(2, 0.4, 1000, mode="replicate"),   # never a source
        _exp(3, 0.9, 1000),                      # dominated, off frontier
    ]
    src = pick_replicate_source(exps, task="synth_regression", round_id=1)
    assert src["id"] == 1


def test_pick_replicate_source_skips_continuations():
    # A warm-started lineage member's metric reflects accumulated
    # compute; re-running its code from scratch is a different
    # configuration, so continuations are never special-round sources.
    cont = _exp(1, 0.3, 1000, mode="continue")
    cont["n_rounds"] = 3
    exps = [cont, _exp(2, 0.5, 1000)]
    src = pick_replicate_source(exps, task="synth_regression", round_id=1)
    assert src["id"] == 2


def test_pick_replicate_source_respects_epoch():
    exps = [_exp(1, 0.5, 1000)]
    exps[0]["objectives"]["frozen_pipeline_version"] = 3
    assert pick_replicate_source(
        exps, task="synth_regression", round_id=1,
        epoch={"frozen_pipeline_version": 4},
    ) is None
    assert pick_replicate_source(
        exps, task="synth_regression", round_id=1,
        epoch={"frozen_pipeline_version": 3},
    ) is not None


def test_gate_off_tasks_find_sources_outside_buckets():
    # Pipeline tasks (fixed arch, size gate off) can report FLOPs outside
    # every TS bucket; gate_off must still surface them as sources.
    exps = [_exp(1, 0.5, 999_000_000, task="synthetic_data_generator")]
    assert pick_replicate_source(
        exps, task="synthetic_data_generator", round_id=1,
    ) is None  # bucketed view misses it
    src = pick_replicate_source(
        exps, task="synthetic_data_generator", round_id=1, gate_off=True,
    )
    assert src["id"] == 1
    t = pick_ablation_target(
        exps, task="synthetic_data_generator", round_id=1,
        min_flops=0, max_flops=0, gate_off=True,
    )
    assert t["id"] == 1


def test_pick_ablation_target_in_bucket():
    exps = [_exp(1, 0.5, 1000), _exp(2, 0.3, 60_000)]
    t = pick_ablation_target(
        exps, task="synth_regression", round_id=0,
        min_flops=200, max_flops=2_000,
    )
    assert t["id"] == 1


def test_pick_transfer_source_from_smaller_bucket():
    buckets = {"tiny": (200, 2_000), "small": (2_000, 10_000)}
    exps = [_exp(1, 0.5, 1000), _exp(2, 0.3, 9_000)]
    src = pick_transfer_source(
        exps, task="synth_regression", round_id=0, min_flops=2_000,
        buckets=buckets,
    )
    assert src["id"] == 1  # only the tiny-bucket member sits below 2_000


def test_source_card_includes_bucket():
    buckets = {"tiny": (200, 2_000)}
    card = source_card(_exp(1, 0.5, 1000), buckets)
    assert card["id"] == 1 and card["source_bucket"] == "tiny"
    assert "code" in card


# ── recipe_only AST freeze ───────────────────────────────────────────

ARCH_A = """
class Net:
    def __init__(self):
        self.w = 3

def build_model(c, p, v, q):
    return Net()

def build_optimizer(model):
    return "sgd"
"""

ARCH_A_NEW_RECIPE = """
class Net:
    def __init__(self):
        self.w = 3

def build_model(c, p, v, q):
    return Net()

def build_optimizer(model):
    return "adamw"

def build_scheduler(opt):
    return "cosine"
"""

ARCH_B = """
class Net:
    def __init__(self):
        self.w = 4

def build_model(c, p, v, q):
    return Net()
"""


def test_same_build_model_ignores_recipe_changes():
    assert same_build_model(ARCH_A, ARCH_A_NEW_RECIPE)


def test_same_build_model_detects_arch_change():
    assert not same_build_model(ARCH_A, ARCH_B)


def test_same_build_model_handles_syntax_errors():
    assert not same_build_model("def broken(:", ARCH_A)


def test_enforce_recipe_only():
    ok, note = enforce_recipe_only(ARCH_A_NEW_RECIPE, ARCH_A)
    assert ok and note == ""
    ok, note = enforce_recipe_only(ARCH_B, ARCH_A)
    assert not ok and "recipe_only violated" in note
