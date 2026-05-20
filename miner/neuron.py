"""Miner neuron — stubbed out.

The chain-coupled miner loop has been removed. This module is kept only
so existing imports of `miner.neuron` don't break. `main()` exits with an
error.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Miner:
    """Stub miner — chain layer has been removed."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Miner neuron has been removed."
        )

    async def submit_agent_code(self):
        raise NotImplementedError

    async def handle_prepare(self, request):
        raise NotImplementedError

    async def handle_release(self, round_id: int):
        raise NotImplementedError

    async def run(self):
        raise NotImplementedError


def get_config():
    raise NotImplementedError(
        "Miner chain configuration has been removed."
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    raise SystemExit(
        "miner/neuron.py: chain integration removed — nothing to run."
    )


if __name__ == "__main__":
    main()
