"""Tests for shared.challenge — deterministic challenge generation."""

from shared.challenge import (
    generate_challenge, round_start_block, current_phase,
    SIZE_BUCKETS,
)


def test_deterministic_same_hash():
    """Same block hash -> same challenge."""
    h = "abcdef1234567890" * 4
    c1 = generate_challenge(h, {"name": "test"})
    c2 = generate_challenge(h, {"name": "test"})
    assert c1.round_id == c2.round_id
    assert c1.seed == c2.seed
    assert c1.eval_split_seed == c2.eval_split_seed
    assert c1.min_flops_equivalent == c2.min_flops_equivalent
    assert c1.max_flops_equivalent == c2.max_flops_equivalent


def test_different_hash_different_challenge():
    """Different block hashes -> different challenges (usually)."""
    h1 = "a" * 64
    h2 = "b" * 64
    c1 = generate_challenge(h1, {"name": "test"})
    c2 = generate_challenge(h2, {"name": "test"})
    assert c1.round_id != c2.round_id


def test_challenge_has_valid_size_bucket():
    h = "1234567890abcdef" * 4
    c = generate_challenge(h, {"name": "test"})
    bucket = (c.min_flops_equivalent, c.max_flops_equivalent)
    assert bucket in SIZE_BUCKETS


def test_challenge_preserves_task():
    task_dict = {"name": "ml_training", "time_budget": 300}
    c = generate_challenge("a" * 64, task_dict)
    assert c.task == task_dict


def test_round_start_block():
    assert round_start_block(0, 275) == 0
    assert round_start_block(274, 275) == 0
    assert round_start_block(275, 275) == 275
    assert round_start_block(550, 275) == 550
    assert round_start_block(600, 275) == 550


def test_current_phase_submission():
    assert current_phase(100, 100) == "submission"
    assert current_phase(149, 100) == "submission"


def test_current_phase_training():
    assert current_phase(150, 100) == "training"
    assert current_phase(299, 100) == "training"


def test_current_phase_evaluation():
    assert current_phase(300, 100) == "evaluation"
    assert current_phase(324, 100) == "evaluation"


def test_current_phase_scoring():
    """Scoring phase covers fallback/scoring window after evaluation."""
    assert current_phase(325, 100) == "scoring"
    assert current_phase(374, 100) == "scoring"


def test_current_phase_no_scoring_zero_window():
    """Without scoring window, evaluation goes straight to idle."""
    assert current_phase(325, 100, scoring_window=0) == "idle"


def test_current_phase_idle():
    assert current_phase(375, 100) == "idle"
    assert current_phase(99, 100) == "idle"
