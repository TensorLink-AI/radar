"""HTTP client for validators / miners to talk to the centralized DB.

Two auth modes selected at construction time:

  * Service HMAC — passes ``service_secret`` (bytes).  Outbound requests
    carry X-Radar-{Signature, Timestamp, Key-Id} headers.  Used by the
    validator dispatcher and any other operator-side process holding
    the shared service key.
  * Bearer token — passes ``api_key`` (str).  Outbound requests carry
    ``Authorization: Bearer <token>``.  Used by miner-side tooling.

Pass exactly ONE of the two.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Async HTTP client for the centralized database API."""

    def __init__(
        self,
        db_url: str,
        *,
        service_secret: Optional[bytes] = None,
        key_id: str = "operator",
        api_key: str = "",
    ):
        self.db_url = db_url.rstrip("/")
        self.service_secret = service_secret
        self.key_id = key_id
        self.api_key = api_key or ""
        if not (self.service_secret or self.api_key):
            raise ValueError(
                "DatabaseClient needs one of service_secret (HMAC) or "
                "api_key (bearer).",
            )
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _sign(self, body: bytes) -> dict[str, str]:
        if self.service_secret:
            from shared.auth import hmac_sign_request
            return hmac_sign_request(
                self.service_secret, body, key_id=self.key_id,
            )
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _post(self, path: str, json_data: dict) -> Optional[dict]:
        try:
            client = await self._get_client()
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
            return None
        except Exception as e:
            logger.warning("DatabaseClient POST %s failed: %s", path, e)
            return None

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
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
            return None
        except Exception as e:
            logger.warning("DatabaseClient GET %s failed: %s", path, e)
            return None

    # ── Public API ───────────────────────────────────────

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.db_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def add_experiment(
        self,
        element_data: dict,
        artifact_cids: Optional[list[dict]] = None,
    ) -> Optional[int]:
        """POST a DataElement dict."""
        payload: dict = {"data": element_data}
        if artifact_cids:
            payload["artifact_cids"] = artifact_cids
        result = await self._post("/experiments/add", payload)
        if result and "index" in result:
            return result["index"]
        return None

    async def get_frontier(self, task: str = "") -> list[dict]:
        params = {"task": task} if task else None
        result = await self._get("/frontier", params=params)
        return result if isinstance(result, list) else []

    async def update_frontier(self, frontier_data: list[dict], task: str = "") -> bool:
        result = await self._post("/frontier/update", {
            "frontier": frontier_data, "task": task,
        })
        return result is not None

    async def get_pareto_elements(self, task: str = "") -> list[dict]:
        params = {"task": task} if task else None
        result = await self._get("/experiments/pareto", params=params)
        return result if isinstance(result, list) else []

    async def get_challenge(self) -> Optional[dict]:
        return await self._get("/challenge")

    async def set_challenge(self, challenge_data) -> bool:
        result = await self._post("/challenge/update", {"challenge": challenge_data})
        return result is not None

    async def record_components(self, experiment_id: int, components: list[str]) -> bool:
        result = await self._post("/provenance/record_components", {
            "experiment_id": experiment_id, "components": components,
        })
        return result is not None

    async def record_round_context(
        self, round_id: int, experiment_id: int, context_type: str = "frontier",
    ) -> bool:
        result = await self._post("/provenance/record_context", {
            "round_id": round_id, "experiment_id": experiment_id,
            "context_type": context_type,
        })
        return result is not None

    async def get_experiment(self, index: int) -> Optional[dict]:
        return await self._get(f"/experiments/{index}")

    async def get_diff(self, index: int) -> Optional[dict]:
        return await self._get(f"/experiments/{index}/diff")

    async def get_agent_code(self, miner_id: str) -> Optional[dict]:
        return await self._get(f"/agent_code/{miner_id}")

    async def get_agent_code_history(
        self, miner_id: str, limit: int = 100,
    ) -> list[dict]:
        result = await self._get(
            f"/agent_code/{miner_id}/history", params={"limit": limit},
        )
        if isinstance(result, dict):
            subs = result.get("submissions")
            if isinstance(subs, list):
                return subs
        return []

    async def get_agent_code_by_hash(self, code_hash: str) -> Optional[dict]:
        return await self._get(f"/agent_code/by_hash/{code_hash}")

    async def submit_agent_code(
        self, files: dict[str, str], entry_point: str = "agent.py",
    ) -> Optional[dict]:
        return await self._post("/agent_code", {
            "files": files, "entry_point": entry_point,
        })

    async def submit_training_meta(
        self, round_id: int, hotkey: str, meta: dict,
    ) -> bool:
        result = await self._post("/training_metas", {
            "round_id": int(round_id),
            "hotkey": hotkey,
            "meta": meta,
        })
        return result is not None

    async def submit_submission_reveal(
        self, round_id: int, entries: list[dict],
    ) -> bool:
        """POST per-round submission_id -> miner_id map.  Idempotent."""
        result = await self._post("/round_submissions/reveal", {
            "round_id": int(round_id),
            "entries": entries,
        })
        return result is not None

    # ── Miner feedback surface (bearer auth) ──────────────────────

    async def my_submissions(
        self, since: str = "", limit: int = 200, task: str = "",
    ) -> list[dict]:
        params: dict = {"limit": limit}
        if since:
            params["since"] = since
        if task:
            params["task"] = task
        result = await self._get("/miners/me/submissions", params=params)
        if isinstance(result, dict):
            return result.get("submissions", []) or []
        return result if isinstance(result, list) else []

    async def my_results(
        self, since: str = "", limit: int = 200, task: str = "",
    ) -> list[dict]:
        params: dict = {"limit": limit}
        if since:
            params["since"] = since
        if task:
            params["task"] = task
        result = await self._get("/miners/me/results", params=params)
        if isinstance(result, dict):
            return result.get("results", []) or []
        return result if isinstance(result, list) else []

    async def my_summary(self) -> dict:
        result = await self._get("/miners/me/summary")
        return result if isinstance(result, dict) else {}

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
