"""Circuit breaker for RunPod control-plane calls.

Same shape as ``shared/targon_breaker.py``. Kept parallel so the two
backends raise their own typed ``*Unavailable`` exceptions — callers
that want to distinguish "RunPod is degraded" from "Targon is
degraded" can pattern-match on the class.

Closed → counts consecutive failures.
Open after ``threshold`` failures; refuses calls until ``reset_after``
has elapsed; then half-open (one trial). A success closes it; a
failure re-opens it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional


class RunpodUnavailable(Exception):
    """Circuit breaker open or repeated transport failure on RunPod.

    Validators treat this as a soft failure: the round proceeds with a
    ``runpod_unavailable`` flag and reduced scoring weight (parallel
    to Targon's hybrid-fallback behaviour).
    """


class RunpodCircuitBreaker:
    def __init__(
        self,
        threshold: int = 5,
        reset_after: float = 60.0,
        clock=time.monotonic,
    ):
        self.threshold = threshold
        self.reset_after = reset_after
        self._clock = clock
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._half_open_inflight = False
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at < self.reset_after:
            return "open"
        return "half_open"

    async def before_call(self) -> None:
        async with self._lock:
            state = self.state
            if state == "open":
                raise RunpodUnavailable(
                    f"circuit breaker open ({self._consecutive_failures} consecutive failures)"
                )
            if state == "half_open":
                if self._half_open_inflight:
                    raise RunpodUnavailable("circuit breaker half-open, trial in flight")
                self._half_open_inflight = True

    async def on_success(self) -> None:
        async with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_inflight = False

    async def on_failure(self) -> None:
        async with self._lock:
            self._consecutive_failures += 1
            self._half_open_inflight = False
            if self._consecutive_failures >= self.threshold:
                self._opened_at = self._clock()
