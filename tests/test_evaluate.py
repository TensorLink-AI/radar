"""Tests for the frozen evaluation script (runner/timeseries_forecast/evaluate.py)."""

import ast
import os


def test_evaluate_syntax():
    """Evaluate script parses without syntax errors."""
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "evaluate.py"
    )
    with open(eval_path) as f:
        source = f.read()
    ast.parse(source)


def test_evaluate_uses_submission_path():
    """Evaluate loads from /workspace/submission.py."""
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "evaluate.py"
    )
    with open(eval_path) as f:
        source = f.read()
    assert "/workspace/submission.py" in source


def test_evaluate_loads_checkpoint():
    """Evaluate loads checkpoint from standard path."""
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "evaluate.py"
    )
    with open(eval_path) as f:
        source = f.read()
    assert "/workspace/checkpoints/model.safetensors" in source
    assert "load_state_dict" in source


def test_evaluate_calls_validate():
    """Evaluate calls validate() from prepare module."""
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "evaluate.py"
    )
    with open(eval_path) as f:
        source = f.read()
    assert "validate(model)" in source


def test_evaluate_prints_crps():
    """Evaluate outputs crps metric."""
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "evaluate.py"
    )
    with open(eval_path) as f:
        source = f.read()
    assert "crps:" in source


def test_evaluate_prints_mase():
    """Evaluate outputs mase metric."""
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "evaluate.py"
    )
    with open(eval_path) as f:
        source = f.read()
    assert "mase:" in source


def test_evaluate_handles_reset():
    """Evaluate calls reset() for stateful models."""
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "evaluate.py"
    )
    with open(eval_path) as f:
        source = f.read()
    assert "reset()" in source
