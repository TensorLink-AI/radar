"""Tests for local/train_subprocess.py — isolation policy, crash and
timeout containment. Uses the numpy synth task only (no torch)."""

from __future__ import annotations

import pytest

from local.task import TaskSpec
from local.train_subprocess import run_training_isolated

GOOD_CODE = """
class Model:
    hidden_sizes = [8]
    activation = "relu"
    learning_rate = 0.05
    epochs = 3


def build_model(input_dim, output_dim):
    return Model()
"""

CRASH_CODE = """
import os


def build_model(input_dim, output_dim):
    os._exit(3)  # simulate a native crash — no exception to catch
"""

HANG_CODE = """
import time


def build_model(input_dim, output_dim):
    time.sleep(60)
"""


def test_in_process_path(monkeypatch):
    monkeypatch.setenv("RADAR_TRAIN_ISOLATION", "never")
    result = run_training_isolated(GOOD_CODE, seed=0, task=TaskSpec())
    assert result["success"] is True
    assert result["metric"] is not None


def test_auto_keeps_numpy_task_in_process(monkeypatch):
    monkeypatch.delenv("RADAR_TRAIN_ISOLATION", raising=False)
    result = run_training_isolated(GOOD_CODE, seed=0, task=TaskSpec())
    assert result["success"] is True


def test_subprocess_success(monkeypatch):
    monkeypatch.setenv("RADAR_TRAIN_ISOLATION", "always")
    result = run_training_isolated(GOOD_CODE, seed=0, task=TaskSpec())
    assert result["success"] is True
    assert result["metric"] is not None


def test_subprocess_contains_hard_crash(monkeypatch):
    monkeypatch.setenv("RADAR_TRAIN_ISOLATION", "always")
    result = run_training_isolated(CRASH_CODE, seed=0, task=TaskSpec())
    assert result["success"] is False
    assert result["harness_status"] == "subprocess_crash"
    assert "exitcode=3" in result["error"]


@pytest.mark.timeout(60)
def test_subprocess_timeout(monkeypatch):
    monkeypatch.setenv("RADAR_TRAIN_ISOLATION", "always")
    result = run_training_isolated(
        HANG_CODE, seed=0, task=TaskSpec(), timeout_seconds=3,
    )
    assert result["success"] is False
    assert result["harness_status"] == "subprocess_timeout"
