"""RunPod-specific lifecycle helpers — parallel to ``miner/targon_lifecycle``.

Holds the bits coupled to RunPod's job lifecycle:

  - ``make_runpod_client()``           — config-driven lazy construction.
  - ``validate_credentials(client)``   — startup auth check; no orphan
    teardown because RunPod endpoints persist across processes (only
    in-flight jobs can leak, and they're TTL'd by the endpoint's
    executionTimeoutMs).
  - ``cancel_jobs_for_round(client, endpoint_id, jobs)`` —
    teardown of jobs submitted during a single round.

The Miner class keeps backend-agnostic state and dispatch; this
module only owns RunPod-specific orchestration so neuron.py stays
under the line cap and the RunPod code is testable in isolation.
"""

from __future__ import annotations

import logging
from typing import Iterable

from config import Config
from miner.hosting_runpod import cancel_active_jobs

logger = logging.getLogger(__name__)


# ── Client factory ──────────────────────────────────────────────────


def make_runpod_client():
    """Build a RunpodClient using Config defaults. Caller caches the result."""
    from shared.runpod_breaker import RunpodCircuitBreaker
    from shared.runpod_client import RunpodClient
    return RunpodClient(
        base_url=Config.RUNPOD_API_BASE_URL,
        timeout=Config.RUNPOD_VERIFICATION_TIMEOUT,
        breaker=RunpodCircuitBreaker(
            threshold=Config.RUNPOD_CIRCUIT_BREAKER_THRESHOLD,
            reset_after=Config.RUNPOD_CIRCUIT_BREAKER_RESET,
        ),
    )


# ── Startup ─────────────────────────────────────────────────────────


async def validate_credentials(client) -> None:
    """Validate the RunPod API key at startup. Raises RuntimeError on failure.

    Unlike Targon, there is no orphan-workload reaping step: RunPod
    endpoints persist across processes by design. In-flight jobs from
    a prior crashed miner process get killed by the endpoint's
    ``executionTimeoutMs`` cap (RunPod side), and any leaked job IDs
    are forgotten with no billing impact — a paused job is free.
    """
    try:
        await client.validate_credentials()
        logger.info("RunPod credentials validated")
    except Exception as e:
        raise RuntimeError(
            f"RunPod credentials invalid or API unreachable at startup: {e}. "
            "Check RUNPOD_API_KEY at https://www.runpod.io/console/user/settings."
        ) from e


# ── Per-round teardown ─────────────────────────────────────────────


async def cancel_jobs_for_round(
    client, endpoint_id: str, job_ids: Iterable[str],
) -> None:
    """Cancel all RunPod jobs we submitted for a round.

    Logs but does not raise — endpoint TTL covers anything we miss.
    """
    job_list = [j for j in job_ids if j]
    if not job_list:
        return
    ok = await cancel_active_jobs(client, endpoint_id, job_list)
    if not ok:
        logger.warning(
            "RUNPOD_TEARDOWN_LEAK endpoint=%s jobs=%s — relying on executionTimeoutMs",
            endpoint_id, job_list,
        )
