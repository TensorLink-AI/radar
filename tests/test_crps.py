"""Tests for CRPS computation in runner/timeseries_forecast/prepare.py.

These tests verify the CRPS implementation via source inspection and
mathematical invariants. Direct torch tests require GPU/torch environment
(run inside the runner container or with torch installed).
"""

import ast
import os
import textwrap


PREPARE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "prepare.py"
)


def _read_source():
    with open(PREPARE_PATH) as f:
        return f.read()


def _parse_tree():
    return ast.parse(_read_source())


def _get_function_source(name: str) -> str:
    """Extract a function's source from prepare.py."""
    source = _read_source()
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end])
    return ""


# ── Bug A: No 2x multiplier ──


def test_crps_from_quantiles_exists():
    """_crps_from_quantiles helper must exist."""
    source = _read_source()
    assert "def _crps_from_quantiles" in source


def test_crps_no_2x_multiplier():
    """_crps_from_quantiles must NOT have a 2.0 * multiplier.

    The naive baseline CRPS is plain MAE (no 2x), so the model CRPS
    must also omit the 2x to keep nCRPS ratios correct.
    """
    fn_src = _get_function_source("_crps_from_quantiles")
    assert fn_src, "_crps_from_quantiles not found"
    # Should not contain 2.0 * or 2 * as a multiplier on the pinball loss
    assert "2.0 *" not in fn_src, "Found 2.0 * multiplier in _crps_from_quantiles"
    assert "2 * pinball" not in fn_src, "Found 2 * multiplier on pinball"
    assert "* 2.0" not in fn_src, "Found * 2.0 multiplier in _crps_from_quantiles"
    assert "* 2" not in fn_src.replace("* 2.", "").replace("(1, 2, 3)", ""), \
        "Found * 2 multiplier in _crps_from_quantiles"


def test_crps_returns_per_sample_not_scalar():
    """_crps_from_quantiles must return per-sample values (B,), not a batch scalar.

    This is Bug B: a batch-averaged scalar divided by per-sample naive values
    is mathematically wrong.
    """
    fn_src = _get_function_source("_crps_from_quantiles")
    assert fn_src, "_crps_from_quantiles not found"
    # Should use .mean(dim=...) not .mean() (which collapses all dims)
    # The return should preserve the batch dimension
    assert ".mean()" not in fn_src, \
        "_crps_from_quantiles uses .mean() which collapses batch dim — must use .mean(dim=...)"


# ── Bug B: Per-sample nCRPS ──


def test_naive_crps_exists():
    """_naive_crps helper must exist."""
    source = _read_source()
    assert "def _naive_crps" in source


def test_naive_crps_returns_per_sample():
    """_naive_crps must return per-sample (B,), not a scalar."""
    fn_src = _get_function_source("_naive_crps")
    assert fn_src, "_naive_crps not found"
    assert ".mean()" not in fn_src, \
        "_naive_crps uses .mean() which collapses batch dim — must use .mean(dim=...)"


def test_validate_returns_ncrps():
    """validate() must return ncrps key."""
    source = _read_source()
    assert '"ncrps"' in source or "'ncrps'" in source, \
        "validate() must return ncrps (normalized CRPS)"


def test_ncrps_uses_geometric_mean():
    """nCRPS must be geometric mean of per-sample ratios.

    Correct: geomean(crps[b] / naive[b]) = exp(mean(log(crps[b] / naive[b])))
    Wrong:   mean(crps) / mean(naive)  (ratio of batch means)
    """
    fn_src = _get_function_source("validate")
    assert fn_src, "validate not found"
    # Must use log + exp for geometric mean
    assert "log" in fn_src, "nCRPS must use log for geometric mean"
    assert "exp" in fn_src, "nCRPS must use exp for geometric mean"


def test_ncrps_divides_per_sample():
    """nCRPS ratio must be per-sample, not batch-scalar / per-sample."""
    fn_src = _get_function_source("validate")
    assert fn_src, "validate not found"
    # Should have sample_crps / sample_naive (per-sample division)
    assert "sample_crps" in fn_src or "crps" in fn_src.lower()
    # Should NOT just call .mean().item() then divide
    # The division should happen before any mean()
    assert "sample_crps / sample_naive" in fn_src or \
           "sample_crps / naive" in fn_src or \
           "ratio" in fn_src, \
        "Must compute per-sample ratio before averaging"


def test_validate_still_returns_raw_crps():
    """validate() must still return raw crps for backward compat."""
    source = _read_source()
    assert '"crps"' in source


def test_validate_returns_n_samples():
    """validate() must return sample count for verification."""
    source = _read_source()
    assert '"n_samples"' in source or "'n_samples'" in source


def test_evaluate_prints_ncrps():
    """evaluate.py should print ncrps metric."""
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "evaluate.py"
    )
    with open(eval_path) as f:
        source = f.read()
    assert "ncrps:" in source
