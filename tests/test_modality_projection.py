"""Tests for modality projection — shapes, value ranges."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "graph_complexity"))
from prepare import generate_graph, graph_walks, ModalityProjection


def _setup(modality, n_nodes=50, n_edges=200):
    adj = generate_graph("er", n_nodes, n_edges, seed=42)
    walks = graph_walks(adj, kappa=1.0, n_walks=10, walk_len=100, seed=42)
    proj = ModalityProjection(adj, modality, seed=42)
    return proj, walks


def test_tokens_shape():
    proj, walks = _setup("tokens")
    result = proj.project(walks)
    assert result.shape == walks.shape


def test_continuous_shape():
    proj, walks = _setup("continuous")
    result = proj.project(walks)
    assert result.shape == walks.shape


def test_waveform_shape():
    proj, walks = _setup("waveform")
    result = proj.project(walks)
    assert result.shape == walks.shape


def test_rms_energy_shape():
    proj, walks = _setup("rms_energy")
    result = proj.project(walks)
    assert result.shape == walks.shape


def test_rms_energy_in_01():
    proj, walks = _setup("rms_energy")
    result = proj.project(walks)
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_tokens_permutation():
    """Token projection should be a permutation of node IDs."""
    proj, walks = _setup("tokens")
    result = proj.project(walks)
    # All values should be integers (node IDs)
    assert np.all(result == result.astype(int))
    # Range should be within [0, n_nodes)
    assert result.min() >= 0
    assert result.max() < 50


def test_continuous_preserves_structure():
    """Continuous projection should have finite values."""
    proj, walks = _setup("continuous")
    result = proj.project(walks)
    assert np.all(np.isfinite(result))


def test_waveform_finite_values():
    proj, walks = _setup("waveform")
    result = proj.project(walks)
    assert np.all(np.isfinite(result))
