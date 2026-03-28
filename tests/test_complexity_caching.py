"""Tests for complexity profile caching."""

import json
import os
import tempfile
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "graph_complexity"))
from complexity import compute_complexity_profile, marginal_entropy


def test_complexity_profile_has_required_keys():
    rng = np.random.RandomState(42)
    tokens = rng.randint(0, 64, (50, 100))
    profile = compute_complexity_profile(tokens)

    assert "multi_horizon_entropy" in profile
    assert "spectral_entropy" in profile
    assert "stationarity" in profile
    assert "cross_series_diversity" in profile
    assert "marginal_entropy" in profile
    assert "composite_difficulty" in profile


def test_complexity_profile_serializable():
    """Profile should be JSON-serializable."""
    rng = np.random.RandomState(42)
    tokens = rng.randint(0, 64, (50, 100))
    profile = compute_complexity_profile(tokens)

    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    json_str = json.dumps(profile, default=default)
    loaded = json.loads(json_str)
    assert loaded["marginal_entropy"] == profile["marginal_entropy"]


def test_marginal_entropy_uniform():
    """Uniform distribution -> H = log(V)."""
    V = 64
    tokens = np.tile(np.arange(V), 100)
    h = marginal_entropy(tokens)
    expected = np.log(V)
    assert abs(h - expected) < 0.1


def test_marginal_entropy_single_token():
    """All same token -> H = 0."""
    tokens = np.zeros(1000, dtype=np.int64)
    h = marginal_entropy(tokens)
    assert h < 1e-6


def test_complexity_values_match():
    """Same data -> same profile."""
    rng = np.random.RandomState(42)
    tokens = rng.randint(0, 64, (50, 100))
    p1 = compute_complexity_profile(tokens)
    p2 = compute_complexity_profile(tokens)
    assert p1["marginal_entropy"] == p2["marginal_entropy"]
    assert p1["composite_difficulty"] == p2["composite_difficulty"]
