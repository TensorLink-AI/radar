"""Database neuron — subnet-owner-side peer refresh loop.

The chain-coupled subnet-owner process has been removed. What remains is
a small loop that keeps the in-memory peer cache fresh from
``MINERS_CONFIG_PATH``. The FastAPI database server in
``database/server.py`` is still usable directly; spin that up with
uvicorn alongside this process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from shared.auth import set_auth
from shared.peers import load_peers

logger = logging.getLogger(__name__)

PEER_REFRESH_INTERVAL_SECONDS = int(os.getenv("RADAR_PEER_REFRESH_INTERVAL", "60"))


async def _peer_refresh_loop(interval: int):
    while True:
        try:
            peers = load_peers(force_reload=True)
            logger.debug("peer refresh: %d peers loaded", len(peers))
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("peer refresh failed: %s", e)
        await asyncio.sleep(interval)


def main():
    logging.basicConfig(
        level=os.getenv("RADAR_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_auth()
    peers = load_peers(force_reload=True)
    logger.info("Database neuron starting: %d peers loaded", len(peers))

    loop = asyncio.new_event_loop()
    stopped = asyncio.Event()

    def _stop(*_):
        loop.call_soon_threadsafe(stopped.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:  # pragma: no cover — Windows fallback
            signal.signal(sig, _stop)

    async def _run():
        task = asyncio.create_task(
            _peer_refresh_loop(PEER_REFRESH_INTERVAL_SECONDS),
        )
        await stopped.wait()
        task.cancel()

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
