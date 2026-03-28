"""Tests for vocab_size variation (256, 1024, 4096)."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "graph_complexity"))
from prepare import generate_graph, graph_walks, ModalityProjection, discretise


def _make_data(vocab_size):
    adj = generate_graph("er", 100, 500, seed=42)
    walks = graph_walks(adj, kappa=1.0, n_walks=50, walk_len=100, seed=42)
    proj = ModalityProjection(adj, "continuous", seed=42)
    values = proj.project(walks)
    tokens, edges, centres = discretise(values, vocab_size)
    return tokens, centres


def test_vocab_256():
    tokens, centres = _make_data(256)
    assert tokens.max() < 256
    assert len(centres) == 256


def test_vocab_1024():
    tokens, centres = _make_data(1024)
    assert tokens.max() < 1024
    assert len(centres) == 1024


def test_vocab_4096():
    tokens, centres = _make_data(4096)
    assert tokens.max() < 4096
    assert len(centres) == 4096


def test_all_token_ids_valid():
    """Token IDs should be in [0, vocab_size) for all sizes."""
    for vs in [256, 1024, 4096]:
        tokens, _ = _make_data(vs)
        assert tokens.min() >= 0
        assert tokens.max() < vs
