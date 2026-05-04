"""Background per-round CVM health monitor.

Targon CVMs occasionally degrade mid-round (host preemption,
hardware blip, network partition). The miner-side monitor polls
``/health`` on the trainer URL every 30s while a round is active.
If the endpoint stays unhealthy for >2 consecutive minutes the
round is marked locally compromised and we log loudly — we do NOT
attempt to redeploy mid-round (a new workload UID would need a
fresh TrainerReady round-trip; the round window is too tight).

This is defensive logging. The validator's mid-run re-verification
(``validator/trainer_verify.reverify_workload``) is the authoritative
signal — but the miner sees the failure first and operators benefit
from the head start.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class HealthMonitor:
    """One per active Targon round. Cancellable via ``stop()``."""

    round_id: int
    trainer_url: str
    workload_uid: str
    poll_interval_s: float = 30.0
    fail_grace_s: float = 120.0
    health_path: str = "/health"

    locally_compromised: bool = field(default=False, init=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "HEALTH_MONITOR_START round=%d uid=%s url=%s interval=%.0fs grace=%.0fs",
            self.round_id, self.workload_uid, self.trainer_url,
            self.poll_interval_s, self.fail_grace_s,
        )

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

    async def _loop(self) -> None:
        url = f"{self.trainer_url.rstrip('/')}{self.health_path}"
        first_failure_at: float | None = None
        async with httpx.AsyncClient(timeout=5.0) as http:
            while not self._stop.is_set():
                ok = await self._probe(http, url)
                now = time.monotonic()
                if ok:
                    if first_failure_at is not None:
                        logger.info(
                            "HEALTH_RECOVERED round=%d uid=%s after %.0fs",
                            self.round_id, self.workload_uid, now - first_failure_at,
                        )
                    first_failure_at = None
                else:
                    if first_failure_at is None:
                        first_failure_at = now
                        logger.warning(
                            "HEALTH_DEGRADED round=%d uid=%s url=%s — first miss",
                            self.round_id, self.workload_uid, url,
                        )
                    elif (
                        now - first_failure_at >= self.fail_grace_s
                        and not self.locally_compromised
                    ):
                        self.locally_compromised = True
                        logger.error(
                            "HEALTH_COMPROMISED round=%d uid=%s url=%s — "
                            "unhealthy >%.0fs. Validator's mid-run reverify is "
                            "authoritative; this miner cannot redeploy without "
                            "a fresh TrainerReady round-trip.",
                            self.round_id, self.workload_uid, url, self.fail_grace_s,
                        )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval_s)
                except asyncio.TimeoutError:
                    pass

    async def _probe(self, http: httpx.AsyncClient, url: str) -> bool:
        try:
            resp = await http.get(url)
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Health probe error for round %d: %s", self.round_id, e)
            return False
