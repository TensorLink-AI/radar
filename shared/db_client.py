"""HTTP client for validators and miners to talk to the centralized DB server.

Used for writes (POST experiments, provenance, agent code) and frontier
fetches. Requests are signed with the HMAC shared secret
(``RADAR_SHARED_SECRET``); miners must additionally pass their hotkey so
the server can identify the caller for endpoints like ``/agent_code``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from shared.auth import sign_request

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Async HTTP client for the centralized database API."""

    def __init__(
        self,
        db_url: str,
        wallet=None,
        api_key: str = "",
        hotkey: str = "",
    ):
        self.db_url = db_url.rstrip("/")
        self.wallet = wallet
        self.api_key = api_key or ""
        self.hotkey = hotkey or ""
        self._client: Optional[httpx.AsyncClient] = None
        self._auth_hint_logged: bool = False
        if not self.api_key and not os.getenv("RADAR_SHARED_SECRET", ""):
            logger.error(
                "DatabaseClient(%s): neither RADAR_SHARED_SECRET nor an API "
                "key is configured — signed requests will carry empty "
                "signatures and the DB server will reject them with "
                "'missing HMAC headers'. Set RADAR_SHARED_SECRET in the "
                "environment to match the validator/DB server.",
                self.db_url,
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _sign(self, body: bytes) -> dict[str, str]:
        headers = sign_request(self.wallet, body)
        if self.api_key:
            headers["X-Radar-API-Key"] = self.api_key
        if self.hotkey:
            headers["X-Miner-Hotkey"] = self.hotkey
        return headers

    async def _post(self, path: str, json_data: dict) -> Optional[dict]:
        """POST with Epistula signing. Returns JSON response or None."""
        try:
            client = await self._get_client()
            import json
            body = json.dumps(json_data).encode()
            headers = self._sign(body)
            headers["Content-Type"] = "application/json"
            resp = await client.post(
                f"{self.db_url}{path}", content=body, headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:500] if e.response else ""
            logger.warning(
                "DatabaseClient POST %s failed (HTTP %d): %s",
                path, e.response.status_code, detail,
            )
            self._maybe_warn_auth(e.response.status_code, detail)
            return None
        except Exception as e:
            logger.warning("DatabaseClient POST %s failed: %s", path, e)
            return None

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """GET with HMAC signing. Returns JSON response or None."""
        try:
            client = await self._get_client()
            headers = self._sign(b"")
            resp = await client.get(
                f"{self.db_url}{path}", params=params, headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:500] if e.response else ""
            logger.warning(
                "DatabaseClient GET %s failed (HTTP %d): %s",
                path, e.response.status_code, detail,
            )
            self._maybe_warn_auth(e.response.status_code, detail)
            return None
        except Exception as e:
            logger.warning("DatabaseClient GET %s failed: %s", path, e)
            return None

    def _maybe_warn_auth(self, status: int, detail: str) -> None:
        """Emit a one-shot actionable hint when an auth-shaped 4xx comes back."""
        if status not in (401, 403):
            return
        if self._auth_hint_logged:
            return
        body = (detail or "").lower()
        looks_auth = (
            "hmac" in body or "epistula" in body or "signature" in body
            or "auth" in body or "unauthor" in body
        )
        if not looks_auth:
            return
        self._auth_hint_logged = True
        if not os.getenv("RADAR_SHARED_SECRET", "") and not self.api_key:
            logger.error(
                "DatabaseClient auth rejected by %s: RADAR_SHARED_SECRET is "
                "unset in this process, so all requests are being signed "
                "with an empty HMAC. Export RADAR_SHARED_SECRET (matching "
                "the value the DB server is configured with) and restart.",
                self.db_url,
            )
        else:
            logger.error(
                "DatabaseClient auth rejected by %s — the configured "
                "RADAR_SHARED_SECRET / API key does not match the server. "
                "Verify the secret on both sides.",
                self.db_url,
            )

    # ── Public API ───────────────────────────────────────

    async def health(self) -> bool:
        """Check database server health."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.db_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def add_experiment(self, element_data: dict) -> Optional[int]:
        """POST a DataElement dict to the database. Returns new index or None."""
        result = await self._post("/experiments/add", {"data": element_data})
        if result and "index" in result:
            return result["index"]
        return None

    async def get_frontier(self, task: str = "") -> list[dict]:
        """GET the current Pareto frontier."""
        params = {"task": task} if task else None
        result = await self._get("/frontier", params=params)
        return result if isinstance(result, list) else []

    async def update_frontier(self, frontier_data: list[dict], task: str = "") -> bool:
        """POST updated frontier data."""
        result = await self._post("/frontier/update", {
            "frontier": frontier_data, "task": task,
        })
        return result is not None

    async def get_pareto_elements(self, task: str = "") -> list[dict]:
        """GET all pareto-eligible elements."""
        params = {"task": task} if task else None
        result = await self._get("/experiments/pareto", params=params)
        return result if isinstance(result, list) else []

    async def get_challenge(self) -> Optional[dict]:
        """GET the current challenge."""
        return await self._get("/challenge")

    async def set_challenge(self, challenge_data) -> bool:
        """POST challenge data (used by validators to update challenge)."""
        result = await self._post("/challenge/update", {"challenge": challenge_data})
        return result is not None

    async def record_components(self, experiment_id: int, components: list[str]) -> bool:
        """POST detected components for an experiment."""
        result = await self._post("/provenance/record_components", {
            "experiment_id": experiment_id, "components": components,
        })
        return result is not None

    async def record_round_context(
        self, round_id: int, experiment_id: int, context_type: str = "frontier",
    ) -> bool:
        """POST round context (which experiments were shown)."""
        result = await self._post("/provenance/record_context", {
            "round_id": round_id, "experiment_id": experiment_id,
            "context_type": context_type,
        })
        return result is not None

    async def get_experiment(self, index: int) -> Optional[dict]:
        """GET a single experiment by index."""
        return await self._get(f"/experiments/{index}")

    async def get_diff(self, index: int) -> Optional[dict]:
        """GET diff for an experiment vs its parent."""
        return await self._get(f"/experiments/{index}/diff")

    async def get_agent_code(self, hotkey: str) -> Optional[dict]:
        """GET a miner's agent code bundle from the DB server."""
        return await self._get(f"/agent_code/{hotkey}")

    async def submit_agent_code(
        self, files: dict[str, str], entry_point: str = "agent.py",
    ) -> Optional[dict]:
        """POST agent code bundle. Returns {"code_hash", "r2_key"} or None."""
        return await self._post("/agent_code", {
            "files": files, "entry_point": entry_point,
        })

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
