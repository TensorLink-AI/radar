"""Lineage-aware pretrain shard assignment for continuation training.

A continuation must, where possible, train on shards its lineage has not
already seen — otherwise the GIFT-eval Δ stops being a generalization
signal and rewards memorization. This module computes the disjoint
assignment and reports whether the pool was exhausted (forcing reuse).

Shard identity is the file basename (e.g. ``shard-00007.parquet``), which
is what gets recorded in ``objectives["pretrain_shards"]`` so a child can
exclude every ancestor's shards.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum training shards a run needs before we'd rather reuse than starve.
MIN_TRAIN_SHARDS = 1


def shard_key(path: str) -> str:
    """Canonical shard identity: the file basename."""
    return Path(path).name


def lineage_shards(lineage_exps: list[dict]) -> set[str]:
    """Union of shard keys consumed by every experiment in a lineage."""
    used: set[str] = set()
    for exp in lineage_exps:
        objs = exp.get("objectives", {}) or {}
        for k in objs.get("pretrain_shards", []) or []:
            used.add(shard_key(str(k)))
    return used


def assign_shards(
    all_paths: list[str],
    *,
    lineage_used: set[str],
    n: int,
    seed: int,
    min_needed: int = MIN_TRAIN_SHARDS,
) -> tuple[list[str], bool]:
    """Pick training shard paths for a run.

    * Fresh run (``lineage_used`` empty): deterministic sample of ``n``
      from the full pool.
    * Continuation: sample from the pool minus ``lineage_used``. If fewer
      than ``min_needed`` remain, fall back to the full pool and flag
      ``reused=True`` (the "allow with reuse" exhaustion policy).

    Returns ``(chosen_paths, reused)`` with ``chosen_paths`` sorted for
    determinism. ``n <= 0`` or ``n >= len(pool)`` selects the whole pool.
    """
    pool = sorted(all_paths)
    if not pool:
        return [], False

    reused = False
    if lineage_used:
        available = [p for p in pool if shard_key(p) not in lineage_used]
        if len(available) >= max(min_needed, 1):
            candidate_pool = available
        else:
            logger.warning(
                "continuation shard pool exhausted (%d of %d disjoint); "
                "allowing reuse",
                len(available), len(pool),
            )
            candidate_pool = pool
            reused = True
    else:
        candidate_pool = pool

    if n <= 0 or n >= len(candidate_pool):
        return sorted(candidate_pool), reused
    rng = random.Random(seed)
    chosen = rng.sample(candidate_pool, n)
    return sorted(chosen), reused
