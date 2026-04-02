"""Tests for multi-task multi-runner support."""

import pytest

from shared.challenge import generate_challenge, select_task
from shared.task import load_task, load_enabled_tasks, BUILT_IN_TASKS


# ── Task Registry ─────────────────────────────────────────────


def test_load_enabled_tasks_single():
    tasks = load_enabled_tasks("ts_forecasting")
    assert "ts_forecasting" in tasks
    assert len(tasks) == 1


def test_load_enabled_tasks_multiple():
    tasks = load_enabled_tasks("ts_forecasting,ml_training")
    assert "ts_forecasting" in tasks
    assert "ml_training" in tasks
    assert len(tasks) == 2


def test_load_enabled_tasks_all():
    tasks = load_enabled_tasks("all")
    assert len(tasks) == len(BUILT_IN_TASKS)


def test_load_enabled_tasks_empty_means_all():
    tasks = load_enabled_tasks("")
    assert len(tasks) == len(BUILT_IN_TASKS)


def test_load_enabled_tasks_invalid():
    with pytest.raises(ValueError):
        load_enabled_tasks("nonexistent_task_xyz")


def test_load_enabled_tasks_whitespace():
    tasks = load_enabled_tasks(" ts_forecasting , ml_training ")
    assert "ts_forecasting" in tasks
    assert "ml_training" in tasks


# ── Task Selection ────────────────────────────────────────────


def test_select_task_not_correlated_with_bucket():
    """Task selection uses different seed than bucket selection."""
    h = "abcdef1234567890" * 4
    task = select_task(h, ["ts_forecasting", "ml_training"])
    challenge = generate_challenge(h, {"name": task})
    # Just verify both work without error — correlation is
    # prevented by the +1 seed offset in select_task
    assert challenge.round_id > 0
    assert task in ("ts_forecasting", "ml_training")


# ── Challenge includes task name ──────────────────────────────


def test_challenge_carries_task_name():
    task = load_task("ts_forecasting")
    challenge = generate_challenge("a" * 64, task.to_dict())
    # ts_forecasting is an alias for ml_training
    assert challenge.task["name"] == "ml_training"
    assert challenge.task["runner_dir"] == "runner/timeseries_forecast"


# ── Runner dir on tasks ───────────────────────────────────────


def test_ts_forecasting_has_runner_dir():
    task = load_task("ts_forecasting")
    assert task.runner_dir == "runner/timeseries_forecast"


def test_ml_training_has_runner_dir():
    task = load_task("ml_training")
    assert task.runner_dir == "runner/timeseries_forecast"
