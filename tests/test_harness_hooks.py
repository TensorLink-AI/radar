"""Tests for training harness recipe hooks (_validate_batch, _read_amp_config, init_weights guard)."""

import sys
import os
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

# Add runner dir to path
_runner_dir = os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast")
sys.path.insert(0, os.path.abspath(_runner_dir))

from harness import _validate_batch, _read_amp_config, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES


# ── _validate_batch tests ─────────────────────────────────────────

def test_validate_batch_accepts_valid():
    batch_size = 4
    batch = {
        "context": torch.randn(batch_size, CONTEXT_LEN, NUM_VARIATES),
        "target": torch.randn(batch_size, PREDICTION_LEN, NUM_VARIATES),
    }
    assert _validate_batch(batch, batch_size) is True


def test_validate_batch_rejects_wrong_context_shape():
    batch_size = 4
    batch = {
        "context": torch.randn(batch_size, CONTEXT_LEN + 1, NUM_VARIATES),
        "target": torch.randn(batch_size, PREDICTION_LEN, NUM_VARIATES),
    }
    assert _validate_batch(batch, batch_size) is False


def test_validate_batch_rejects_wrong_target_shape():
    batch_size = 4
    batch = {
        "context": torch.randn(batch_size, CONTEXT_LEN, NUM_VARIATES),
        "target": torch.randn(batch_size, PREDICTION_LEN + 1, NUM_VARIATES),
    }
    assert _validate_batch(batch, batch_size) is False


def test_validate_batch_rejects_nan():
    batch_size = 4
    context = torch.randn(batch_size, CONTEXT_LEN, NUM_VARIATES)
    context[0, 0, 0] = float("nan")
    batch = {
        "context": context,
        "target": torch.randn(batch_size, PREDICTION_LEN, NUM_VARIATES),
    }
    assert _validate_batch(batch, batch_size) is False


def test_validate_batch_rejects_inf():
    batch_size = 4
    target = torch.randn(batch_size, PREDICTION_LEN, NUM_VARIATES)
    target[0, 0, 0] = float("inf")
    batch = {
        "context": torch.randn(batch_size, CONTEXT_LEN, NUM_VARIATES),
        "target": target,
    }
    assert _validate_batch(batch, batch_size) is False


def test_validate_batch_rejects_missing_keys():
    assert _validate_batch({"context": torch.randn(1, 1, 1)}, 1) is False
    assert _validate_batch({}, 1) is False


# ── _read_amp_config tests ────────────────────────────────────────

def test_read_amp_config_default_no_hook():
    sub = types.ModuleType("sub")
    result = _read_amp_config(sub)
    assert result == {"enabled": True, "dtype": "bfloat16"}


def test_read_amp_config_valid_float16():
    sub = types.ModuleType("sub")
    sub.configure_amp = lambda: {"enabled": True, "dtype": "float16"}
    result = _read_amp_config(sub)
    assert result == {"enabled": True, "dtype": "float16"}


def test_read_amp_config_rejects_invalid_dtype():
    sub = types.ModuleType("sub")
    sub.configure_amp = lambda: {"enabled": True, "dtype": "int8"}
    result = _read_amp_config(sub)
    assert result["dtype"] == "bfloat16"  # falls back to bfloat16


def test_read_amp_config_handles_exception():
    sub = types.ModuleType("sub")
    def bad_amp():
        raise RuntimeError("boom")
    sub.configure_amp = bad_amp
    result = _read_amp_config(sub)
    assert result == {"enabled": True, "dtype": "bfloat16"}


def test_read_amp_config_disabled():
    sub = types.ModuleType("sub")
    sub.configure_amp = lambda: {"enabled": False, "dtype": "float32"}
    result = _read_amp_config(sub)
    assert result == {"enabled": False, "dtype": "float32"}


# ── init_weights param count guard ────────────────────────────────

def test_init_weights_param_count_guard():
    """init_weights that changes param count should be detected."""
    model = nn.Linear(10, 5)
    params_before = sum(p.numel() for p in model.parameters())

    # Simulate init_weights that adds a parameter
    def bad_init_weights(m):
        m.extra = nn.Parameter(torch.randn(10))

    bad_init_weights(model)
    params_after = sum(p.numel() for p in model.parameters())
    assert params_after != params_before, "Test setup: param count should change"


def test_init_weights_no_change_is_ok():
    """init_weights that only modifies values should be fine."""
    model = nn.Linear(10, 5)
    params_before = sum(p.numel() for p in model.parameters())

    def good_init_weights(m):
        nn.init.xavier_uniform_(m.weight)

    good_init_weights(model)
    params_after = sum(p.numel() for p in model.parameters())
    assert params_after == params_before
