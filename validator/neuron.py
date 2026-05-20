"""Validator neuron — stubbed out.

The chain-coupled validator loop has been removed. This module is kept
only so that imports from `validator.neuron` don't break. The functions
return immediately and `main()` exits with an error.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_live_validator_uids(
    *args,
    miner_uids: Optional[set[int]] = None,
    current_block: Optional[int] = None,
    stale_blocks: int = 600,
    **kwargs,
) -> list[int]:
    """Stub — chain-based validator discovery removed."""
    return []


def get_my_assignments(
    all_uids: list[int],
    validator_uids: list[int],
    my_uid: int,
    seed: int,
) -> list[int]:
    """Deterministic assignment helper kept for unit tests."""
    import random
    if not validator_uids or my_uid not in validator_uids:
        return list(all_uids)
    rng = random.Random(seed)
    shuffled = list(all_uids)
    rng.shuffle(shuffled)
    sorted_validators = sorted(validator_uids)
    my_idx = sorted_validators.index(my_uid)
    return [uid for i, uid in enumerate(shuffled) if i % len(sorted_validators) == my_idx]


class Validator:
    """Stub validator — chain layer has been removed."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Validator neuron has been removed."
        )

    async def run_round(self):
        raise NotImplementedError

    async def run(self):
        raise NotImplementedError


def get_config():
    """Stub — no chain config available."""
    raise NotImplementedError(
        "Validator chain configuration has been removed."
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    raise SystemExit(
        "validator/neuron.py: chain integration removed — nothing to run."
    )


if __name__ == "__main__":
    main()
