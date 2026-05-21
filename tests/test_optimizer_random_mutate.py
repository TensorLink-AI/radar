"""Tests for the random_mutate baseline optimizer."""

from __future__ import annotations

import pytest

from miner_template.optimizers import ResultRow
from miner_template.optimizers.random_mutate import optimize
from miner_template.prompts import Prompt


def _pop(*ids_and_templates: tuple[str, str]) -> list[Prompt]:
    return [Prompt(id=i, template=t, generation=0)
            for i, t in ids_and_templates]


def _result(prompt_id: str, score: float, round_id: int = 1) -> ResultRow:
    return ResultRow(
        round_id=round_id,
        submission_id=f"sub-{round_id}-{prompt_id}",
        task_name="ts_forecasting",
        prompt_id=prompt_id,
        architecture_code="class M: pass",
        scores={"raw_score": score},
    )


def test_empty_population_returns_empty():
    assert optimize([], [], {}) == []


def test_no_results_clones_top_of_population():
    pop = _pop(("a", "A"), ("b", "B"), ("c", "C"))
    new = optimize([], pop, {"population": 4, "elite_k": 2})
    assert len(new) == 4
    # Both elites must appear at the head (order = ranking, stable).
    assert {p.id for p in new[:2]} == {"a", "b"}


def test_ranks_by_mean_score_and_promotes_winners():
    pop = _pop(("a", "A"), ("b", "B"), ("c", "C"))
    results = [
        _result("a", 0.1),
        _result("b", 0.9), _result("b", 0.9),
        _result("c", 0.5),
    ]
    new = optimize(results, pop, {"population": 4, "elite_k": 2})
    assert len(new) == 4
    elite_ids = [p.id for p in new if p.parent_id is None]
    assert elite_ids[0] == "b"  # highest mean score
    assert elite_ids[1] == "c"


def test_children_have_parent_pointing_at_elite():
    pop = _pop(("a", "A"), ("b", "B"))
    results = [_result("a", 0.9), _result("b", 0.1)]
    new = optimize(results, pop, {"population": 4, "elite_k": 1})
    children = [p for p in new if p.parent_id is not None]
    assert len(children) == 3
    assert all(c.parent_id == "a" for c in children)
    assert all(c.template.startswith("A") for c in children)
    assert all(c.generation == 1 for c in children)


def test_population_size_enforced():
    pop = _pop(("a", "A"), ("b", "B"))
    new = optimize([], pop, {"population": 8})
    assert len(new) == 8


def test_population_below_one_raises():
    pop = _pop(("a", "A"))
    with pytest.raises(ValueError):
        optimize([], pop, {"population": 0})


def test_deterministic_perturbation():
    """Same inputs -> same outputs (modulo new UUIDs).  We can't compare
    ids, but the chosen perturbation suffix is deterministic in the
    parent + child_idx, so child templates should match across runs."""
    pop = _pop(("a", "Alpha"))
    a = optimize([], pop, {"population": 4, "elite_k": 1})
    b = optimize([], pop, {"population": 4, "elite_k": 1})
    assert [p.template for p in a] == [p.template for p in b]


def test_unknown_prompt_ids_in_results_ignored():
    pop = _pop(("a", "A"))
    results = [_result("ghost", 999.0)]  # not in population
    new = optimize(results, pop, {"population": 2, "elite_k": 1})
    assert len(new) == 2
    assert new[0].id == "a"


def test_custom_score_key():
    pop = _pop(("a", "A"), ("b", "B"))
    rows = [
        ResultRow(round_id=1, submission_id="x", task_name="t",
                  prompt_id="a", architecture_code="",
                  scores={"raw_score": 0.0, "frontier_bonus": 1.0}),
        ResultRow(round_id=1, submission_id="y", task_name="t",
                  prompt_id="b", architecture_code="",
                  scores={"raw_score": 1.0, "frontier_bonus": 0.0}),
    ]
    new = optimize(rows, pop, {"population": 2, "elite_k": 1,
                                "score_key": "frontier_bonus"})
    # Top elite should now be 'a' (high frontier_bonus) instead of 'b'.
    assert new[0].id == "a"
