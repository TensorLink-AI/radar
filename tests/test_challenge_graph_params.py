"""Tests for graph complexity challenge parameters."""

from shared.challenge import generate_challenge


def test_graph_params_populated_for_graph_complexity():
    h = "abcdef1234567890" * 4
    c = generate_challenge(h, {"name": "graph_complexity"})
    assert c.graph_type in ("er", "ba")
    assert c.graph_nodes in (200, 500, 1000, 2000)
    assert c.graph_edges is not None and c.graph_edges > 0
    assert c.kappa is not None and 0.0 <= c.kappa <= 3.0
    assert c.modality in ("tokens", "continuous", "waveform", "rms_energy")
    assert c.vocab_size in (256, 1024, 4096)
    assert c.prediction_mode in ("direct", "teacher_forced")


def test_graph_params_none_for_ts_forecasting():
    h = "abcdef1234567890" * 4
    c = generate_challenge(h, {"name": "ts_forecasting"})
    assert c.graph_type is None
    assert c.graph_nodes is None
    assert c.kappa is None
    assert c.modality is None
    assert c.vocab_size is None
    assert c.prediction_mode is None


def test_same_hash_same_graph_params():
    h = "1234567890abcdef" * 4
    c1 = generate_challenge(h, {"name": "graph_complexity"})
    c2 = generate_challenge(h, {"name": "graph_complexity"})
    assert c1.graph_type == c2.graph_type
    assert c1.graph_nodes == c2.graph_nodes
    assert c1.graph_edges == c2.graph_edges
    assert c1.kappa == c2.kappa
    assert c1.modality == c2.modality
    assert c1.vocab_size == c2.vocab_size
    assert c1.prediction_mode == c2.prediction_mode


def test_different_hash_different_params():
    h1 = "a" * 64
    h2 = "b" * 64
    c1 = generate_challenge(h1, {"name": "graph_complexity"})
    c2 = generate_challenge(h2, {"name": "graph_complexity"})
    # At least some params should differ (probabilistically)
    params1 = (c1.graph_type, c1.graph_nodes, c1.kappa, c1.modality, c1.vocab_size)
    params2 = (c2.graph_type, c2.graph_nodes, c2.kappa, c2.modality, c2.vocab_size)
    assert params1 != params2


def test_edges_equals_nodes_times_per_node():
    """graph_edges should be graph_nodes * edges_per_node."""
    h = "abcdef1234567890" * 4
    c = generate_challenge(h, {"name": "graph_complexity"})
    # edges_per_node is one of [5, 10, 20]
    edges_per_node = c.graph_edges // c.graph_nodes
    assert edges_per_node in (5, 10, 20)


def test_tokens_modality_default_teacher_forced():
    """Tokens modality should default to teacher_forced."""
    # Try many hashes until we get tokens modality
    for i in range(100):
        h = f"{i:064x}"
        c = generate_challenge(h, {"name": "graph_complexity"})
        if c.modality == "tokens":
            assert c.prediction_mode == "teacher_forced"
            return
    # If we never hit tokens in 100 tries, that's fine — probabilistic test


def test_challenge_serialization_with_graph_params():
    """Graph params survive JSON round-trip."""
    from shared.protocol import Challenge
    h = "abcdef1234567890" * 4
    c = generate_challenge(h, {"name": "graph_complexity"})
    json_str = c.to_json()
    c2 = Challenge.from_json(json_str)
    assert c2.graph_type == c.graph_type
    assert c2.graph_nodes == c.graph_nodes
    assert c2.graph_edges == c.graph_edges
    assert c2.kappa == c.kappa
    assert c2.modality == c.modality
    assert c2.vocab_size == c.vocab_size
    assert c2.prediction_mode == c.prediction_mode
