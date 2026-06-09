"""Tests for local/screening.py — candidate triage with a stub trainer."""

from __future__ import annotations

import os

from local.screening import screen_candidates, screening_card
from local.task import TSForecastingSpec


def _stub_train_fn(scores):
    """Trainer stub: metric looked up by code string."""
    calls = []

    def fn(code, *, seed, task, min_flops, max_flops):
        calls.append({"code": code, "budget": task.time_budget_seconds,
                      "max_tasks": os.environ.get("RADAR_GIFT_EVAL_MAX_TASKS")})
        metric = scores.get(code)
        if metric is None:
            return {"success": False, "metric": None, "error": "boom",
                    "workdir": ""}
        return {"success": True, "metric": metric, "workdir": ""}

    fn.calls = calls
    return fn


def test_screening_card_shape():
    card = screening_card(max_candidates=4, budget_seconds=300,
                          eval_max_tasks=10)
    assert card["enabled"] and card["max_candidates"] == 4


def test_screen_picks_best_and_uses_short_budget():
    task = TSForecastingSpec(time_budget_seconds=3600)
    fn = _stub_train_fn({"a": 0.9, "b": 0.4, "c": 0.7})
    winner, summaries = screen_candidates(
        [{"name": "A", "code": "a"}, {"name": "B", "code": "b"},
         {"name": "C", "code": "c"}],
        task=task, seed=1, min_flops=0, max_flops=10,
        budget_seconds=120, eval_max_tasks=9, train_fn=fn,
    )
    assert winner["code"] == "b"
    assert [s["winner"] for s in summaries if s.get("winner")] == [True]
    # Screening ran on a short-budget copy; the original spec is untouched.
    assert all(c["budget"] == 120 for c in fn.calls)
    assert task.time_budget_seconds == 3600
    # Truncated eval was active during screening, restored afterwards.
    assert all(c["max_tasks"] == "9" for c in fn.calls)
    assert os.environ.get("RADAR_GIFT_EVAL_MAX_TASKS") is None


def test_screen_requires_two_candidates():
    task = TSForecastingSpec()
    winner, summaries = screen_candidates(
        [{"name": "A", "code": "a"}],
        task=task, seed=1, min_flops=0, max_flops=10,
        budget_seconds=60, eval_max_tasks=5,
        train_fn=_stub_train_fn({"a": 0.5}),
    )
    assert winner is None and summaries == []


def test_screen_all_failures_returns_no_winner():
    task = TSForecastingSpec()
    winner, summaries = screen_candidates(
        [{"name": "A", "code": "a"}, {"name": "B", "code": "b"}],
        task=task, seed=1, min_flops=0, max_flops=10,
        budget_seconds=60, eval_max_tasks=5,
        train_fn=_stub_train_fn({}),
    )
    assert winner is None
    assert all(not s["success"] for s in summaries)


def test_screen_skips_malformed_candidates():
    task = TSForecastingSpec()
    fn = _stub_train_fn({"a": 0.5, "b": 0.6})
    winner, summaries = screen_candidates(
        ["junk", {"code": ""}, {"name": "A", "code": "a"},
         {"name": "B", "code": "b"}],
        task=task, seed=1, min_flops=0, max_flops=10,
        budget_seconds=60, eval_max_tasks=5, train_fn=fn,
    )
    assert winner["code"] == "a"
    assert len(summaries) == 2
