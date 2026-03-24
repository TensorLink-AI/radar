"""Tests for runner/timeseries_forecast/flops.py — hybrid analytical + wallclock FLOPs."""

import sys
import os

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

# Add runner dir to path so we can import flops module
_runner_dir = os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast")
sys.path.insert(0, os.path.abspath(_runner_dir))

from flops import compute_flops_analytical, compute_flops_equivalent


def test_analytical_flops_linear():
    """compute_flops_analytical returns a sensible value for nn.Linear."""
    model = nn.Linear(10, 5)
    result = compute_flops_analytical(model, context_len=32, num_variates=10)
    assert result is not None
    assert result > 0


def test_analytical_flops_preferred():
    """compute_flops_equivalent prefers the analytical result."""
    model = nn.Linear(10, 5)
    result = compute_flops_equivalent(model, context_len=32, num_variates=10, device="cpu")
    analytical = compute_flops_analytical(model, context_len=32, num_variates=10)
    assert analytical is not None
    # When analytical succeeds, compute_flops_equivalent should return the same value
    assert result == analytical


def test_analytical_flops_none_on_empty_model():
    """compute_flops_analytical returns None if model has zero FLOPs."""
    class NoOpModel(nn.Module):
        def forward(self, x):
            return x

    model = NoOpModel()
    result = compute_flops_analytical(model, context_len=32, num_variates=10)
    # A no-op model has 0 FLOPs counted, so should return None
    assert result is None
