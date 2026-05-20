"""Validator neuron — lightweight, config-driven peer refresh loop.

The chain-coupled validator that used to live here has been removed.
What remains is a small process that:

  * Periodically reloads ``MINERS_CONFIG_PATH`` so peer membership is
    fresh for the validator (collection, coordinator, evaluator).
  * Exposes helper functions kept for the test suite (work-splitting).
  * Wires HMAC shared-secret auth via ``shared.auth.set_auth``.

To bring up a full validator stack you still need to run the FastAPI
proxy in ``validator/db_proxy.py`` separately; this module just keeps the
periodic peer refresh and is the canonical entry point for
``python validator/neuron.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Optional

from shared.auth import set_auth
from shared.peers import load_peers

logger = logging.getLogger(__name__)

PEER_REFRESH_INTERVAL_SECONDS = int(os.getenv("RADAR_PEER_REFRESH_INTERVAL", "60"))


def compute_live_validator_uids(
    *args,
    miner_uids: Optional[set[int]] = None,
    current_block: Optional[int] = None,
    stale_blocks: int = 600,
    **kwargs,
) -> list[int]:
    """Return UIDs of currently-known validators.

    The old chain-based "is this UID alive on-chain" check has been
    replaced by "is this UID in the static miners.json?". Callers pass in
    the set of known miner UIDs; we intersect that with the configured
    peer list.
    """
    peers = load_peers()
    known = {p.uid for p in peers}
    if miner_uids is None:
        return sorted(known)
    return sorted(known & set(miner_uids))


def get_my_assignments(
    all_uids: list[int],
    validator_uids: list[int],
    my_uid: int,
    seed: int,
) -> list[int]:
    """Deterministic work-split helper kept for unit tests."""
    import random
    if not validator_uids or my_uid not in validator_uids:
        return list(all_uids)
    rng = random.Random(seed)
    shuffled = list(all_uids)
    rng.shuffle(shuffled)
    sorted_validators = sorted(validator_uids)
    my_idx = sorted_validators.index(my_uid)
    return [uid for i, uid in enumerate(shuffled) if i % len(sorted_validators) == my_idx]


async def _peer_refresh_loop(interval: int):
    """Periodically force-reload the peer registry from disk."""
    while True:
        try:
            peers = load_peers(force_reload=True)
            logger.debug("peer refresh: %d peers loaded", len(peers))
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("peer refresh failed: %s", e)
        await asyncio.sleep(interval)


def main():
    """Run the validator peer-refresh side-loop.

    Intended to be invoked as ``python validator/neuron.py``. It does NOT
    start the FastAPI proxy — bring that up separately via uvicorn.
    """
    logging.basicConfig(
        level=os.getenv("RADAR_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_auth()
    peers = load_peers(force_reload=True)
    logger.info("Validator starting: %d peers loaded from miners config", len(peers))

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
