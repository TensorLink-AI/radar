"""Deterministic per-round challenge generation from block hash.

Each round targets a specific FLOPs-equivalent size bucket. The challenge,
size bucket, and all seeds are deterministic from the block hash so that
every validator independently arrives at the same parameters.
"""

from __future__ import annotations

import os
import random

from shared.protocol import Challenge

SIZE_BUCKETS = [
    (100_000, 500_000),         # tiny
    (500_000, 2_000_000),       # small
    (2_000_000, 10_000_000),    # medium-small
    (10_000_000, 50_000_000),   # medium
    (50_000_000, 125_000_000),  # large
]


def generate_challenge(block_hash: str, base_task: dict) -> Challenge:
    """Generate a deterministic challenge from block hash.

    Picks a random size bucket, derives round_id, eval_split_seed, and seed.
    All validators with the same block_hash produce the same challenge.

    Override bucket with RADAR_MIN_FLOPS / RADAR_MAX_FLOPS env vars (testing).
    """
    seed_int = int(block_hash[:16], 16)
    rng = random.Random(seed_int)

    bucket = rng.choice(SIZE_BUCKETS)
    round_id = seed_int % (2**32)
    eval_split_seed = rng.randint(0, 2**31)
    seed = rng.randint(0, 2**31)

    # Allow env var override for testing — forces a specific size range
    min_override = os.getenv("RADAR_MIN_FLOPS")
    max_override = os.getenv("RADAR_MAX_FLOPS")
    if min_override and max_override:
        bucket = (int(min_override), int(max_override))

    return Challenge(
        challenge_id=f"round_{round_id}",
        seed=seed,
        task=base_task,
        min_flops_equivalent=bucket[0],
        max_flops_equivalent=bucket[1],
        eval_split_seed=eval_split_seed,
        round_id=round_id,
    )


def round_start_block(current_block: int, interval: int = 275) -> int:
    """Compute the start block of the current round."""
    return (current_block // interval) * interval


def current_phase(
    current_block: int,
    round_start: int,
    submission_window: int = 50,
    training_window: int = 150,
    eval_window: int = 25,
    scoring_window: int = 50,
) -> str:
    """Determine the current phase within a round.

    Returns 'submission' | 'training' | 'evaluation' | 'scoring' | 'idle'.
    The scoring window covers fallback/scoring after evaluation ends.
    """
    offset = current_block - round_start
    if offset < 0:
        return "idle"
    if offset < submission_window:
        return "submission"
    if offset < submission_window + training_window:
        return "training"
    if offset < submission_window + training_window + eval_window:
        return "evaluation"
    if offset < submission_window + training_window + eval_window + scoring_window:
        return "scoring"
    return "idle"
