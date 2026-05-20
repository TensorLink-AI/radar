"""Database neuron — stubbed out.

The chain-coupled subnet-owner process has been removed. The FastAPI
database server in `database/server.py` is still usable directly; this
module just no longer wires it to any chain layer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DatabaseNeuron:
    """Stub database neuron — chain layer has been removed."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DatabaseNeuron has been removed. "
            "Run `database/server.py` directly to host the FastAPI app."
        )

    async def run(self):
        raise NotImplementedError


def get_config():
    raise NotImplementedError(
        "Database chain configuration has been removed."
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    raise SystemExit(
        "database/neuron.py: chain integration removed — nothing to run."
    )


if __name__ == "__main__":
    main()
