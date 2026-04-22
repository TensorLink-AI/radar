"""Tests for shared.challenge — deterministic challenge generation."""

import hashlib

from shared.challenge import (
    generate_challenge, round_start_block, round_id_from_block,
    current_phase, select_task, SIZE_BUCKETS,
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


def test_challenge_uses_task_specific_buckets():
    """Tasks declaring `size_buckets` draw from their own list, not globals."""
    custom = [
        [1_000_000_000, 10_000_000_000],
        [10_000_000_000, 100_000_000_000],
    ]
    task_dict = {"name": "nanogpt", "size_buckets": custom}
    # Sample many block hashes; every bucket choice must land in the custom list.
    for i in range(50):
        h = f"{i:064x}"
        c = generate_challenge(h, task_dict)
        bucket = (c.min_flops_equivalent, c.max_flops_equivalent)
        assert bucket in {tuple(b) for b in custom}
    # And at least one hash must fall outside the global SIZE_BUCKETS range,
    # proving it's actually using the task-specific list.
    all_from_globals = all(
        (generate_challenge(f"{i:064x}", task_dict).min_flops_equivalent,
         generate_challenge(f"{i:064x}", task_dict).max_flops_equivalent)
        in SIZE_BUCKETS
        for i in range(50)
    )
    assert not all_from_globals


def test_challenge_falls_back_to_global_when_buckets_empty():
    """Empty size_buckets → fall back to global SIZE_BUCKETS."""
    task_dict = {"name": "task", "size_buckets": []}
    c = generate_challenge("a" * 64, task_dict)
    assert (c.min_flops_equivalent, c.max_flops_equivalent) in SIZE_BUCKETS


def test_challenge_ignores_malformed_buckets():
    """Malformed entries are skipped; if nothing valid remains, fall back."""
    task_dict = {"name": "task", "size_buckets": [["bad"], [0, 0], [10, 5]]}
    c = generate_challenge("a" * 64, task_dict)
    assert (c.min_flops_equivalent, c.max_flops_equivalent) in SIZE_BUCKETS


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


# ── Task Selection ──────────────────────────────────────────────


def test_select_task_single():
    """Single task always selected."""
    result = select_task("a" * 64, ["ts_forecasting"])
    assert result == "ts_forecasting"


def test_select_task_deterministic():
    """Same block hash -> same task."""
    h = "abcdef1234567890" * 4
    t1 = select_task(h, ["ts_forecasting", "nanogpt", "ml_training"])
    t2 = select_task(h, ["ts_forecasting", "nanogpt", "ml_training"])
    assert t1 == t2


def test_select_task_varies_by_hash():
    """Different hashes usually pick different tasks (with enough tasks)."""
    import hashlib
    tasks = ["task_a", "task_b", "task_c", "task_d", "task_e"]
    # Use sha256 of integers so the first 16 hex chars vary widely
    selections = {
        select_task(hashlib.sha256(str(i).encode()).hexdigest(), tasks)
        for i in range(100)
    }
    # With 100 hashes and 5 tasks, we should see more than 1 task
    assert len(selections) > 1


def test_select_task_order_independent():
    """Task list order doesn't matter — sorted internally."""
    h = "f" * 64
    t1 = select_task(h, ["b_task", "a_task", "c_task"])
    t2 = select_task(h, ["c_task", "a_task", "b_task"])
    assert t1 == t2


def test_round_id_from_block_matches_generate_challenge():
    """DB server derivation must produce the same round_id validators use."""
    interval = 275
    for block in (0, 274, 275, 550, 1100, 1_234_567):
        rs = round_start_block(block, interval)
        block_hash = hashlib.sha256(str(rs).encode()).hexdigest()
        expected = generate_challenge(block_hash, {"name": "test"}).round_id
        assert round_id_from_block(block, interval) == expected


def test_round_id_stable_within_window():
    """Every block inside one round maps to the same round_id."""
    interval = 275
    rs = round_start_block(12_345, interval)
    ids = {round_id_from_block(b, interval) for b in range(rs, rs + interval)}
    assert len(ids) == 1


def test_round_id_changes_at_boundary():
    """Crossing the round boundary flips round_id."""
    interval = 275
    rs = round_start_block(50_000, interval)
    assert round_id_from_block(rs + interval - 1, interval) != round_id_from_block(
        rs + interval, interval,
    )
