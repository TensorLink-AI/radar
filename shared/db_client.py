"""HTTP client for validators to talk to the centralized database server.

Used for writes (POST experiments, provenance) and frontier fetches.
All methods sign requests with Epistula via the validator's wallet.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from shared.auth import sign_request

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Async HTTP client for the centralized database API."""

    def __init__(self, db_url: str, wallet):
        self.db_url = db_url.rstrip("/")
        self.wallet = wallet
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _sign(self, body: bytes) -> dict[str, str]:
        return sign_request(self.wallet, body)

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
        except Exception as e:
            logger.warning("DatabaseClient POST %s failed: %s", path, e)
            return None

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """GET with Epistula signing. Returns JSON response or None."""
        try:
            client = await self._get_client()
            headers = self._sign(b"")
            resp = await client.get(
                f"{self.db_url}{path}", params=params, headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("DatabaseClient GET %s failed: %s", path, e)
            return None

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
