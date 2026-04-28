"""
Desearch proxy — HTTP proxy for miner agents to search arxiv via SN22.

Runs on the subnet owner's database server (NOT on validators).
Validators forward /desearch/* requests here.

All queries are logged to Postgres (proxy_query_log table).
Rate-limited to RADAR_DESEARCH_MAX_QUERIES per miner per tempo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Literal, Optional, get_args

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default Desearch (SN22) endpoint base URL.
# The actual path used is /desearch/ai/search.
DEFAULT_SN22_URL = "https://api.desearch.ai"
DESEARCH_SEARCH_PATH = "/desearch/ai/search"

# Desearch "tools" values — restrict to arxiv/web; miners don't get to
# choose other tools (twitter, etc.) through this proxy.
DESEARCH_TOOL_ARXIV = "arxiv"
DESEARCH_TOOL_WEB = "web"
ToolT = Literal["arxiv", "web"]
ALLOWED_TOOLS = frozenset(get_args(ToolT))

# Allowed values for the Desearch `date_filter` field. Desearch does not
# accept "NONE"; PAST_2_YEARS is the broadest supported window.
DateFilterT = Literal[
    "PAST_24_HOURS",
    "PAST_2_DAYS",
    "PAST_WEEK",
    "PAST_2_WEEKS",
    "PAST_MONTH",
    "PAST_2_MONTHS",
    "PAST_YEAR",
    "PAST_2_YEARS",
]
ALLOWED_DATE_FILTERS = frozenset(get_args(DateFilterT))
DEFAULT_DATE_FILTER: DateFilterT = "PAST_2_YEARS"

# Desearch response shape we request. Gives us a list of link objects plus
# a final summary string.
DESEARCH_RESULT_TYPE = "LINKS_WITH_FINAL_SUMMARY"

# Desearch requires `count >= 10` on /desearch/ai/search. We still let miners
# ask for fewer results — the response is sliced to max_results before return.
DESEARCH_MIN_COUNT = 10

# Rate limit: max queries per miner per tempo
MAX_QUERIES_PER_TEMPO = 20

# Tempo duration in seconds (~72 min)
TEMPO_DURATION_SECONDS = 360 * 12


class SearchQuery(BaseModel):
    """Search request from a miner."""

    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    tool: ToolT = Field(default=DESEARCH_TOOL_ARXIV)
    date_filter: DateFilterT = Field(default=DEFAULT_DATE_FILTER)


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
        pool=None,
        api_key: str = "",
    ):
        self.sn22_url = sn22_url
        self.api_key = api_key
        self.max_queries = max_queries
        self.tempo_seconds = tempo_seconds
        self.pool = pool  # asyncpg pool for query logging

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

    async def _log_query(
        self, miner_uid: int, miner_hotkey: str,
        query: str, num_results: int,
    ):
        """Log query to Postgres (best-effort, never raises)."""
        if not self.pool:
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO proxy_query_log
                    (service, miner_uid, miner_hotkey, query_text,
                     response_summary, timestamp)
                VALUES ($1, $2, $3, $4, $5, extract(epoch from now()))
                """,
                "desearch", miner_uid, miner_hotkey, query[:500],
                f"{num_results} results",
            )
        except Exception as e:
            logger.debug("Desearch query log failed: %s", e)

    async def search(
        self, miner_uid: int, query: str, max_results: int = 5,
        miner_hotkey: str = "", tool: str = DESEARCH_TOOL_ARXIV,
        date_filter: str = DEFAULT_DATE_FILTER,
    ) -> SearchResponse:
        """
        Search via the Desearch /desearch/ai/search endpoint.

        Raises HTTPException(429) if the miner has exhausted their quota.
        """
        if tool not in ALLOWED_TOOLS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported tool '{tool}'. Allowed: {sorted(ALLOWED_TOOLS)}.",
            )
        if date_filter not in ALLOWED_DATE_FILTERS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported date_filter '{date_filter}'.",
            )

        remaining = self.remaining_queries(miner_uid)
        if remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_queries} queries per tempo.",
            )

        self._record_query(miner_uid)
        remaining -= 1

        headers = {"Authorization": self.api_key} if self.api_key else {}
        # Desearch requires count >= DESEARCH_MIN_COUNT; we slice the response
        # back down to max_results before returning to the miner.
        upstream_count = max(DESEARCH_MIN_COUNT, max_results)
        body = {
            "prompt": query,
            "tools": [tool],
            "date_filter": date_filter,
            "result_type": DESEARCH_RESULT_TYPE,
            "count": upstream_count,
        }

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.sn22_url}{DESEARCH_SEARCH_PATH}",
                json=body,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:500]
            logger.warning(
                "Desearch HTTP %d: %s", e.response.status_code, body,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Desearch upstream error: HTTP {e.response.status_code} — {body}".strip(),
            )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning("Desearch request failed: %s: %s", type(e).__name__, e)
            raise HTTPException(
                status_code=502,
                detail=f"Desearch upstream unreachable: {type(e).__name__}",
            )
        except json.JSONDecodeError as e:
            preview = (resp.text or "")[:500]
            logger.warning(
                "Desearch returned %d with non-JSON body (%s): %s",
                resp.status_code, e, preview,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Desearch upstream returned non-JSON body (HTTP {resp.status_code})",
            )

        results = _parse_sn22_response(data)

        # Log to Postgres (best-effort)
        await self._log_query(miner_uid, miner_hotkey, query, len(results))

        return SearchResponse(results=results[:max_results], remaining_queries=remaining)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def reset_limits(self):
        """Reset all rate limits (e.g., at tempo boundary)."""
        self._query_counts.clear()


def _parse_sn22_response(data: dict | list) -> list[SearchResult]:
    """Parse the Desearch API response into SearchResult objects.

    Desearch's /search/links/web endpoint returns a list of link objects
    (or, for some variants, a dict containing one of: links/results/papers).
    """
    if isinstance(data, list):
        papers = data
    else:
        papers = (
            data.get("links")
            or data.get("results")
            or data.get("papers")
            or []
        )
    results = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        results.append(SearchResult(
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            abstract=paper.get("abstract", paper.get("snippet", "")),
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
        """Search arxiv/web via Desearch. Requires X-Miner-UID header."""
        proxy = get_proxy()
        miner_uid = _extract_miner_uid(request)
        miner_hotkey = _extract_miner_hotkey(request)
        return await proxy.search(
            miner_uid, req.query, req.max_results, miner_hotkey,
            req.tool, req.date_filter,
        )

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
