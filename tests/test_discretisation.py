"""Tests for discretisation — deterministic, correct vocab, uniform-ish."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "graph_complexity"))
from prepare import discretise


def test_deterministic():
    values = np.random.RandomState(42).randn(100, 50)
    t1, e1, c1 = discretise(values, 256)
    t2, e2, c2 = discretise(values, 256)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(e1, e2)


def test_correct_vocab_256():
    values = np.random.RandomState(42).randn(200, 50)
    tokens, edges, centres = discretise(values, 256)
    assert tokens.min() >= 0
    assert tokens.max() < 256


def test_correct_vocab_1024():
    values = np.random.RandomState(42).randn(200, 50)
    tokens, edges, centres = discretise(values, 1024)
    assert tokens.min() >= 0
    assert tokens.max() < 1024


def test_correct_vocab_4096():
    values = np.random.RandomState(42).randn(500, 50)
    tokens, edges, centres = discretise(values, 4096)
    assert tokens.min() >= 0
    assert tokens.max() < 4096


def test_output_shape():
    values = np.random.RandomState(42).randn(100, 50)
    tokens, edges, centres = discretise(values, 256)
    assert tokens.shape == values.shape


def test_bin_centres_length():
    values = np.random.RandomState(42).randn(200, 50)
    _, _, centres = discretise(values, 256)
    assert len(centres) == 256


def test_roughly_uniform_marginals():
    """Percentile binning should produce roughly uniform token distribution."""
    rng = np.random.RandomState(42)
    values = rng.randn(1000, 100)
    tokens, _, _ = discretise(values, 64)
    counts = np.bincount(tokens.ravel(), minlength=64)
    # No bin should have more than 3x the average
    expected = tokens.size / 64
    assert counts.max() < 3 * expected
