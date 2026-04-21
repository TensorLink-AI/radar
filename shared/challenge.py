"""Deterministic per-round challenge generation from block hash.

Each round targets a specific FLOPs-equivalent size bucket. The challenge,
size bucket, and all seeds are deterministic from the block hash so that
every validator independently arrives at the same parameters.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

from shared.protocol import Challenge

if TYPE_CHECKING:
    from shared.task import TaskSpec

SIZE_BUCKETS = [
    (100_000, 500_000),         # tiny
    (500_000, 2_000_000),       # small
    (2_000_000, 10_000_000),    # medium-small
    (10_000_000, 50_000_000),   # medium
    (50_000_000, 125_000_000),  # large
]


def _resolve_task_buckets(base_task: dict) -> list[tuple[int, int]]:
    """Pick the bucket list to sample from for this task.

    A task may declare its own `size_buckets` list of [min, max] pairs in
    YAML. If it doesn't, or if the declared list is empty/malformed, fall
    back to the global SIZE_BUCKETS.
    """
    raw = base_task.get("size_buckets") or []
    buckets: list[tuple[int, int]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            lo, hi = int(entry[0]), int(entry[1])
            if 0 < lo < hi:
                buckets.append((lo, hi))
    return buckets if buckets else list(SIZE_BUCKETS)


def select_task(block_hash: str, task_names: list[str]) -> str:
    """Deterministically select a task name from the block hash.

    Uses the same block hash so all validators pick the same task.
    Task names are sorted before selection for determinism.
    """
    if len(task_names) == 1:
        return task_names[0]
    seed_int = int(block_hash[:16], 16)
    rng = random.Random(seed_int + 1)  # +1 to avoid correlation with bucket
    return rng.choice(sorted(task_names))


def generate_challenge(
    block_hash: str,
    base_task: dict,
    default_agent_seconds: int = 600,
) -> Challenge:
    """Generate a deterministic challenge from block hash.

    Picks a random size bucket, derives round_id, eval_split_seed, and seed.
    All validators with the same block_hash produce the same challenge.

    Size buckets are resolved from the task dict: `base_task["size_buckets"]`
    (list of [min, max] pairs) is used when present; otherwise the global
    SIZE_BUCKETS list is used as the default. This lets different tasks
    declare their own FLOPs ranges — e.g. a language-modeling task can span
    1M–10B while a small-scale task stays within 100K–125M.

    The agent's Phase A wall-clock budget is resolved in priority order:
      1. base_task["agent_seconds"] if > 0 (per-task override from YAML)
      2. default_agent_seconds (validator-side global, e.g. Config.AGENT_TIMEOUT)

    Override bucket with RADAR_MIN_FLOPS / RADAR_MAX_FLOPS env vars (testing).
    """
    seed_int = int(block_hash[:16], 16)
    rng = random.Random(seed_int)

    bucket = rng.choice(_resolve_task_buckets(base_task))
    round_id = seed_int % (2**32)
    # randint is inclusive on both ends; cap at 2**31 - 1 so seeds stay in
    # signed int32 range for any downstream consumer that narrows them.
    eval_split_seed = rng.randint(0, 2**31 - 1)
    seed = rng.randint(0, 2**31 - 1)

    # Allow env var override for testing — forces a specific size range
    min_override = os.getenv("RADAR_MIN_FLOPS")
    max_override = os.getenv("RADAR_MAX_FLOPS")
    if min_override and max_override:
        bucket = (int(min_override), int(max_override))

    # Agent wall-clock budget: per-task override from YAML wins; otherwise
    # use the caller-supplied default (typically Config.AGENT_TIMEOUT).
    task_agent_seconds = int(base_task.get("agent_seconds") or 0)
    agent_seconds = task_agent_seconds if task_agent_seconds > 0 else int(default_agent_seconds)

    return Challenge(
        challenge_id=f"round_{round_id}",
        seed=seed,
        task=base_task,
        min_flops_equivalent=bucket[0],
        max_flops_equivalent=bucket[1],
        eval_split_seed=eval_split_seed,
        round_id=round_id,
        agent_seconds=agent_seconds,
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
