"""Targon-specific lifecycle helpers extracted from ``miner/neuron.py``.

Holds the bits that are coupled to Targon's CVM lifecycle:

  - ``make_targon_client()``    — config-driven lazy construction
  - ``validate_and_reap_orphans(client)`` — startup credential check +
    orphan workload cleanup, run before the miner subscribes to the
    metagraph.
  - ``HealthMonitorRegistry``   — per-round HealthMonitor bookkeeping.
  - ``install_shutdown_handlers(loop, event, teardown_async)`` —
    SIGTERM/SIGINT → teardown → set shutdown event → run loop exits.

The Miner class keeps backend-agnostic state (``active_deployments``)
and the deploy/teardown dispatch. Only Targon-specific orchestration
moved here, so ``miner/neuron.py`` stays under the line cap and the
Targon code is testable in isolation.
"""

from __future__ import annotations

import asyncio
import logging
import signal as signal_mod
from typing import Awaitable, Callable

from config import Config
from miner.health_monitor import HealthMonitor
from miner.hosting import Deployment, teardown_targon_with_retry

logger = logging.getLogger(__name__)


# ── Client factory ──────────────────────────────────────────────────


def make_targon_client():
    """Build a TargonClient using Config defaults. Caller caches the result."""
    from shared.targon_breaker import CircuitBreaker
    from shared.targon_client import TargonClient
    return TargonClient(
        base_url=Config.TARGON_API_BASE_URL,
        tower_url=Config.TARGON_TOWER_URL,
        timeout=Config.TARGON_VERIFICATION_TIMEOUT,
        breaker=CircuitBreaker(
            threshold=Config.TARGON_CIRCUIT_BREAKER_THRESHOLD,
            reset_after=Config.TARGON_CIRCUIT_BREAKER_RESET,
        ),
    )


# ── Startup ─────────────────────────────────────────────────────────


async def validate_and_reap_orphans(client) -> None:
    """Validate credentials then tear down workloads from prior processes.

    Raises RuntimeError on auth failure / unreachable API. Logs but
    does not raise on individual orphan teardown failures — Targon's
    TTL eventually reaps anything we miss.
    """
    try:
        await client.validate_credentials()
        logger.info("Targon credentials validated")
    except Exception as e:
        raise RuntimeError(
            f"Targon credentials invalid or API unreachable at startup: {e}. "
            "Check TARGON_API_KEY at https://docs.targon.com."
        ) from e

    try:
        workloads = await client.list_active_workloads()
    except Exception as e:
        logger.warning("Could not list workloads at startup (continuing): %s", e)
        return
    if not workloads:
        return
    logger.warning(
        "Found %d orphan workloads from prior process — tearing down",
        len(workloads),
    )
    for wl in workloads:
        ok = await teardown_targon_with_retry(client, wl.uid)
        logger.info("ORPHAN_TEARDOWN uid=%s ok=%s", wl.uid, ok)


# ── Health monitor registry ─────────────────────────────────────────


class HealthMonitorRegistry:
    """Per-round HealthMonitor bookkeeping. One instance per Miner."""

    def __init__(self) -> None:
        self._monitors: dict[int, HealthMonitor] = {}

    def start(self, round_id: int, deployment: Deployment) -> None:
        existing = self._monitors.pop(round_id, None)
        if existing is not None:
            asyncio.create_task(existing.stop())
        monitor = HealthMonitor(
            round_id=round_id,
            trainer_url=deployment.url,
            workload_uid=deployment.targon_workload_uid,
            poll_interval_s=Config.TARGON_HEALTH_POLL_INTERVAL_S,
            fail_grace_s=Config.TARGON_HEALTH_FAIL_GRACE_S,
        )
        monitor.start()
        self._monitors[round_id] = monitor

    async def stop(self, round_id: int) -> None:
        monitor = self._monitors.pop(round_id, None)
        if monitor is not None:
            await monitor.stop()

    async def stop_all(self) -> None:
        rids = list(self._monitors.keys())
        for rid in rids:
            await self.stop(rid)


# ── Shutdown handlers ───────────────────────────────────────────────


def install_shutdown_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
    teardown_async: Callable[[], Awaitable[None]],
    *,
    teardown_timeout_s: float = 30.0,
) -> None:
    """Wire SIGTERM/SIGINT to teardown + shutdown_event.set().

    The handler runs ``teardown_async`` with a bounded timeout so a
    Targon outage at shutdown can't block the process forever (leaked
    workloads are caught by the next process's startup orphan reap).
    Setting ``shutdown_event`` lets the run loop exit cleanly.
    """
    async def _shutdown(sig_name: str) -> None:
        logger.info("Shutdown signal %s received — tearing down active workloads", sig_name)
        try:
            await asyncio.wait_for(teardown_async(), timeout=teardown_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "Teardown exceeded %.0fs — leaving remaining workloads to next-startup reap",
                teardown_timeout_s,
            )
        except Exception as e:
            logger.warning("Teardown raised: %s", e)
        shutdown_event.set()

    for sig in (signal_mod.SIGTERM, signal_mod.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: loop.create_task(_shutdown(s.name)))
        except (NotImplementedError, RuntimeError):
            # Non-Unix or already-running-loop tests — fall back to default
            # signal behaviour (KeyboardInterrupt for SIGINT).
            pass
