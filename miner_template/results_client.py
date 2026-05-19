"""HTTP client for the miner-feedback endpoints on the database server.

The DB exposes a small set of bearer-auth'd routes under ``/miners/me/*``
that let a miner pull its own scored history without going through
the validator.  This is the feedback channel the optimizer reads.

  GET /miners/me/submissions  — Phase A submissions made by this miner
  GET /miners/me/results      — submissions joined with Phase C scores
  GET /miners/me/summary      — counts + rolling-window stats
  GET /tasks/{task}/frontier  — public per-task Pareto frontier
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from miner_template.optimizers import ResultRow

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


class MinerResultsClient:
    """Pull this miner's scored history from the central DB."""

    def __init__(
        self,
        db_url: str,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ):
        if not db_url:
            raise ValueError("db_url required")
        if not api_key:
            raise ValueError("api_key required")
        self.db_url = db_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._owned_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    # ── Context manager ─────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if self._owned_client and not self._client.is_closed:
            self._client.close()

    # ── Internal ────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get(self, path: str, params: Optional[dict] = None):
        resp = self._client.get(
            f"{self.db_url}{path}", params=params, headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    # ── Public ──────────────────────────────────────────────────────

    def submissions(
        self,
        *,
        since: str = "",
        limit: int = 200,
        task: str = "",
    ) -> list[dict]:
        """Phase A submissions made by this miner.  Returns raw dicts —
        the optimizer doesn't need these; this is for the CLI's
        ``results`` view + custom miner tooling."""
        params: dict = {"limit": limit}
        if since:
            params["since"] = since
        if task:
            params["task"] = task
        payload = self._get("/miners/me/submissions", params=params)
        if isinstance(payload, dict):
            return list(payload.get("submissions", []) or [])
        return list(payload or [])

    def results(
        self,
        *,
        since: str = "",
        limit: int = 200,
        task: str = "",
    ) -> list[ResultRow]:
        """Scored history — submissions joined with Phase C results.
        Returned as ``ResultRow`` so the optimizer can consume directly."""
        params: dict = {"limit": limit}
        if since:
            params["since"] = since
        if task:
            params["task"] = task
        payload = self._get("/miners/me/results", params=params)
        rows = payload.get("results") if isinstance(payload, dict) else payload
        return [ResultRow.from_dict(r) for r in (rows or []) if isinstance(r, dict)]

    def summary(self) -> dict:
        """Rolling stats — total submissions, last round id, mean score
        over the recent window, etc.  Useful for ``--watch`` to detect
        when a new round of results is available."""
        payload = self._get("/miners/me/summary")
        return payload if isinstance(payload, dict) else {}

    def frontier(self, task: str) -> list[dict]:
        """Public per-task frontier (no auth required, but we send the
        bearer anyway — harmless)."""
        if not task:
            raise ValueError("task required")
        payload = self._get(f"/tasks/{task}/frontier")
        if isinstance(payload, dict):
            return list(payload.get("points", []) or [])
        return list(payload or [])
