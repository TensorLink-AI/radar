"""
Desearch proxy — HTTP proxy for miners to search arxiv via SN22.

Rate-limited to 20 queries per miner per tempo. Runs as additional routes
on the existing FastAPI DB server.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default SN22 endpoint (Desearch/arxiv search subnet)
DEFAULT_SN22_URL = "https://desearch.ai/api/v1"

# Rate limit: max queries per miner per tempo
MAX_QUERIES_PER_TEMPO = 20

# Tempo duration in seconds (~72 min)
TEMPO_DURATION_SECONDS = 360 * 12


class SearchQuery(BaseModel):
    """Arxiv search request from a miner."""

    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    """Single arxiv paper result."""

    title: str = ""
    authors: list[str] = []
    abstract: str = ""
    arxiv_id: str = ""
    published: str = ""
    url: str = ""


class SearchResponse(BaseModel):
    """Response to a miner search query."""

    results: list[SearchResult] = []
    remaining_queries: int = 0


class DesearchProxy:
    """
    Rate-limited proxy for arxiv searches via SN22.

    Each miner gets MAX_QUERIES_PER_TEMPO queries per tempo window.
    The window resets every TEMPO_DURATION_SECONDS.
    """

    def __init__(
        self,
        sn22_url: str = DEFAULT_SN22_URL,
        max_queries: int = MAX_QUERIES_PER_TEMPO,
        tempo_seconds: int = TEMPO_DURATION_SECONDS,
    ):
        self.sn22_url = sn22_url
        self.max_queries = max_queries
        self.tempo_seconds = tempo_seconds

        # miner_uid -> list of query timestamps in current window
        self._query_counts: dict[int, list[float]] = defaultdict(list)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _prune_old_queries(self, miner_uid: int) -> int:
        """Remove expired timestamps, return current count."""
        cutoff = time.time() - self.tempo_seconds
        timestamps = self._query_counts[miner_uid]
        self._query_counts[miner_uid] = [t for t in timestamps if t > cutoff]
        return len(self._query_counts[miner_uid])

    def remaining_queries(self, miner_uid: int) -> int:
        """How many queries this miner has left in the current tempo."""
        used = self._prune_old_queries(miner_uid)
        return max(0, self.max_queries - used)

    def _record_query(self, miner_uid: int):
        """Record a query for rate limiting."""
        self._query_counts[miner_uid].append(time.time())

    async def search(
        self, miner_uid: int, query: str, max_results: int = 5,
        miner_hotkey: str = "",
    ) -> SearchResponse:
        """
        Search arxiv via the SN22 Desearch endpoint.

        Raises HTTPException(429) if the miner has exhausted their quota.
        """
        remaining = self.remaining_queries(miner_uid)
        if remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_queries} queries per tempo.",
            )

        self._record_query(miner_uid)
        remaining -= 1

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.sn22_url}/search/arxiv",
                json={"query": query, "max_results": max_results},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("SN22 returned error: %s", e.response.status_code)
            raise HTTPException(status_code=502, detail="Desearch upstream error")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning("SN22 request failed: %s", e)
            raise HTTPException(status_code=502, detail="Desearch upstream unreachable")

        results = _parse_sn22_response(data)
        return SearchResponse(results=results[:max_results], remaining_queries=remaining)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def reset_limits(self):
        """Reset all rate limits (e.g., at tempo boundary)."""
        self._query_counts.clear()


def _parse_sn22_response(data: dict | list) -> list[SearchResult]:
    """Parse the SN22 API response into SearchResult objects."""
    papers = data if isinstance(data, list) else data.get("results", data.get("papers", []))
    results = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        results.append(SearchResult(
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            abstract=paper.get("abstract", ""),
            arxiv_id=paper.get("arxiv_id", paper.get("id", "")),
            published=paper.get("published", paper.get("date", "")),
            url=paper.get("url", paper.get("link", "")),
        ))
    return results


# ── FastAPI routes ──────────────────────────────────────────────────────────

# Singleton proxy — set by validator at startup
_proxy: Optional[DesearchProxy] = None


def set_proxy(proxy: DesearchProxy):
    global _proxy
    _proxy = proxy


def get_proxy() -> DesearchProxy:
    if _proxy is None:
        raise HTTPException(status_code=503, detail="Desearch proxy not initialized")
    return _proxy


def register_routes(app: FastAPI):
    """Register desearch proxy routes on the given FastAPI app."""

    @app.post("/desearch/search")
    async def desearch_search(req: SearchQuery, request: Request):
        """Search arxiv papers. Requires X-Miner-UID header."""
        proxy = get_proxy()
        miner_uid = _extract_miner_uid(request)
        miner_hotkey = _extract_miner_hotkey(request)
        return await proxy.search(miner_uid, req.query, req.max_results, miner_hotkey)

    @app.get("/desearch/quota")
    def desearch_quota(request: Request):
        """Check remaining query quota for this miner."""
        proxy = get_proxy()
        miner_uid = _extract_miner_uid(request)
        miner_hotkey = _extract_miner_hotkey(request)
        return {
            "miner_uid": miner_uid,
            "miner_hotkey": miner_hotkey,
            "remaining_queries": proxy.remaining_queries(miner_uid),
        }

    @app.get("/desearch/health")
    def desearch_health():
        return {"status": "ok", "proxy_initialized": _proxy is not None}


def _extract_miner_uid(request: Request) -> int:
    """Extract miner UID from request header."""
    uid_str = request.headers.get("X-Miner-UID", "")
    if not uid_str:
        raise HTTPException(status_code=400, detail="Missing X-Miner-UID header")
    try:
        uid = int(uid_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-Miner-UID header")
    if uid < 0:
        raise HTTPException(status_code=400, detail="Invalid miner UID")
    return uid


def _extract_miner_hotkey(request: Request) -> str:
    """Extract optional miner hotkey from request header."""
    return request.headers.get("X-Miner-Hotkey", "")
