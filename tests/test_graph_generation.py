"""Tests for graph generation — deterministic, correct sizes, symmetric."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "graph_complexity"))
from prepare import generate_graph


def test_deterministic_from_seed():
    """Same seed -> identical graph."""
    g1 = generate_graph("er", 50, 200, seed=42)
    g2 = generate_graph("er", 50, 200, seed=42)
    np.testing.assert_array_equal(g1, g2)


def test_different_seed_different_graph():
    g1 = generate_graph("er", 50, 200, seed=42)
    g2 = generate_graph("er", 50, 200, seed=99)
    assert not np.array_equal(g1, g2)


def test_symmetric_er():
    g = generate_graph("er", 50, 200, seed=42)
    np.testing.assert_array_equal(g, g.T)


def test_symmetric_ba():
    g = generate_graph("ba", 50, 200, seed=42)
    np.testing.assert_array_equal(g, g.T)


def test_correct_size_er():
    g = generate_graph("er", 100, 500, seed=42)
    assert g.shape == (100, 100)


def test_correct_size_ba():
    g = generate_graph("ba", 100, 500, seed=42)
    assert g.shape == (100, 100)


def test_no_self_loops():
    g = generate_graph("er", 50, 200, seed=42)
    assert np.trace(g) == 0.0


def test_no_isolated_nodes_er():
    g = generate_graph("er", 50, 100, seed=42)
    degrees = g.sum(axis=1)
    assert (degrees > 0).all()


def test_ba_has_edges():
    g = generate_graph("ba", 50, 250, seed=42)
    assert g.sum() > 0
