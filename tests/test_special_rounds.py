"""Tests for local/special_rounds.py — replicate round end-to-end on the
numpy task, challenge annotation, and objective stamping."""

from __future__ import annotations

import pytest

from local.special_rounds import (
    annotate_special,
    run_replicate_round,
    stamp_special_objectives,
)
from local.store import LocalStore
from local.task import TaskSpec
from local.trainer import run_training

MLP_CODE = """
class Model:
    hidden_sizes = [16]
    activation = "relu"
    learning_rate = 0.05
    epochs = 5


def build_model(input_dim, output_dim):
    return Model()
"""

ARCH = "class Net:\n    pass\n\ndef build_model(c, p, v, q):\n    return Net()\n"
ARCH_OTHER = "class Net2:\n    pass\n\ndef build_model(c, p, v, q):\n    return Net2()\n"


@pytest.fixture()
def store(tmp_path):
    s = LocalStore(tmp_path / "t.db")
    yield s
    s.close()


def _add(store, *, metric=0.5, flops=322, code=MLP_CODE,
         task="synth_regression", mode="new", success=True):
    return store.add_experiment(
        round_id=1, miner_id="m", name="src", code=code, motivation="",
        reasoning="", tool_calls=[], metric=metric, success=success,
        objectives={"flops_equivalent_size": flops, "num_params": 10},
        score=0.5, loss_curve=[], task=task, mode=mode,
    )


def test_replicate_round_end_to_end(store):
    src_id = _add(store)

    def train_fn(code, **kw):
        return run_training(code, **kw)

    consumed = run_replicate_round(
        store, TaskSpec(), 7, epoch={}, train_fn=train_fn,
    )
    assert consumed is True
    rows = store.recent_experiments(n=10)
    repl = next(e for e in rows if e["mode"] == "replicate")
    assert repl["objectives"]["replicate_of"] == src_id
    assert repl["success"] is True
    assert repl["score"] == 0.0
    assert "replicate_delta" in repl["objectives"]
    # Challenge row exists but was never open to miners.
    assert store.open_challenge() is None
    # The lab report for the replicate was written.
    assert store.get_lab_report(repl["id"]) is not None


def test_replicate_round_downgrades_without_source(store):
    consumed = run_replicate_round(
        store, TaskSpec(), 7, epoch={}, train_fn=lambda c, **k: {},
    )
    assert consumed is False


def _challenge(lo=200, hi=2000):
    return {
        "round_type": "new",
        "min_flops_equivalent": lo, "max_flops_equivalent": hi,
        "continuation_allowed": False, "eligible_parents": [],
        "continuation_kind": "",
    }


def test_annotate_ablate_sets_target(store):
    src_id = _add(store)
    ch = _challenge()
    annotate_special(ch, "ablate", store, TaskSpec(), round_id=3, epoch={})
    assert ch["round_type"] == "ablate"
    assert ch["ablation_target"]["id"] == src_id
    assert ch["continuation_allowed"] is False


def test_annotate_downgrades_without_target(store):
    ch = _challenge()
    annotate_special(ch, "ablate", store, TaskSpec(), round_id=3, epoch={})
    assert ch["round_type"] == "new"
    assert ch["downgrade_reason"] == "no_ablate_target"


def test_annotate_transfer_needs_smaller_bucket(store):
    _add(store, flops=322)  # tiny bucket
    ch = _challenge(lo=2_000, hi=10_000)  # small bucket round
    annotate_special(ch, "transfer", store, TaskSpec(), round_id=3, epoch={})
    assert ch["round_type"] == "transfer"
    assert ch["transfer_source"]["source_bucket"] == "tiny"


def test_stamp_ablation_objective():
    result = {"objectives": {}}
    challenge = {"round_type": "ablate", "ablation_target": {"id": 11}}
    stamp_special_objectives(result, {"code": "x"}, challenge)
    assert result["objectives"]["ablation_of"] == 11


def test_stamp_recipe_only_enforces_arch_freeze():
    challenge = {"round_type": "recipe_only",
                 "recipe_base": {"id": 5, "code": ARCH}}
    ok_result = {"objectives": {}}
    stamp_special_objectives(
        ok_result, {"code": ARCH + "\ndef build_optimizer(m):\n    return 1\n"},
        challenge,
    )
    assert ok_result["objectives"]["recipe_only_of"] == 5

    bad_result = {"objectives": {}}
    stamp_special_objectives(bad_result, {"code": ARCH_OTHER}, challenge)
    assert "recipe_only_of" not in bad_result["objectives"]
    assert "recipe_only violated" in bad_result["continuation_note"]


def test_stamp_transfer_objective():
    result = {"objectives": {}}
    challenge = {"round_type": "transfer", "transfer_source": {"id": 4}}
    stamp_special_objectives(result, {"code": "x"}, challenge)
    assert result["objectives"]["transfer_of"] == 4
