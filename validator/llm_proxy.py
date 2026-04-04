"""LLM proxy — rate-limited proxy for miner agents to query Chutes AI.

Subnet owner configures which models are allowed and the API key.
Validators run this as additional routes on the proxy FastAPI app.
Agents authenticate via the per-round ``X-Agent-Token`` header
(verified by the proxy middleware in db_proxy.py).

Rate-limited to RADAR_LLM_MAX_QUERIES per miner per tempo.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Chutes AI inference endpoint
DEFAULT_CHUTES_URL = "https://chutes-api.com/v1"

# Rate limit: max LLM queries per miner per tempo
MAX_QUERIES_PER_TEMPO = 50

# Tempo duration in seconds (~72 min, matching desearch)
TEMPO_DURATION_SECONDS = 360 * 12


class ChatMessage(BaseModel):
    """Single message in a chat conversation."""

    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1, max_length=50000)


class ChatRequest(BaseModel):
    """Chat completion request from an agent."""

    model: str = Field(..., min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=16384)


class ChatResponse(BaseModel):
    """Proxied chat completion response."""

    model: str = ""
    content: str = ""
    usage: dict = Field(default_factory=dict)
    remaining_queries: int = 0


class LLMProxy:
    """Rate-limited proxy for LLM inference via Chutes AI.

    The subnet owner controls:
      - Which models are available (allowed_models)
      - The API key (chutes_api_key)
      - Rate limits per miner per tempo
    """

    def __init__(
        self,
        chutes_url: str = DEFAULT_CHUTES_URL,
        chutes_api_key: str = "",
        allowed_models: list[str] | None = None,
        max_queries: int = MAX_QUERIES_PER_TEMPO,
        tempo_seconds: int = TEMPO_DURATION_SECONDS,
    ):
        self.chutes_url = chutes_url.rstrip("/")
        self.chutes_api_key = chutes_api_key
        self.allowed_models = allowed_models or []
        self.max_queries = max_queries
        self.tempo_seconds = tempo_seconds

        # miner_uid -> list of query timestamps in current window
        self._query_counts: dict[int, list[float]] = defaultdict(list)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
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
        self._query_counts[miner_uid].append(time.time())

    def _validate_model(self, model: str) -> None:
        """Raise if model is not in the allowed list."""
        if not self.allowed_models:
            return  # empty list = all models allowed
        if model not in self.allowed_models:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {model!r} not allowed. "
                    f"Allowed models: {self.allowed_models}"
                ),
            )

    async def chat(
        self, miner_uid: int, req: ChatRequest,
    ) -> ChatResponse:
        """Forward a chat completion request to Chutes AI.

        Raises HTTPException(429) on rate limit, 400 on bad model.
        """
        remaining = self.remaining_queries(miner_uid)
        if remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_queries} LLM queries per tempo.",
            )

        self._validate_model(req.model)
        self._record_query(miner_uid)
        remaining -= 1

        headers = {"Content-Type": "application/json"}
        if self.chutes_api_key:
            headers["Authorization"] = f"Bearer {self.chutes_api_key}"

        payload = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.chutes_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Chutes AI returned error: %s %s", e.response.status_code, e.response.text[:200])
            raise HTTPException(status_code=502, detail="LLM upstream error")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning("Chutes AI request failed: %s", e)
            raise HTTPException(status_code=502, detail="LLM upstream unreachable")

        # Extract response content from OpenAI-compatible format
        content = ""
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")

        return ChatResponse(
            model=data.get("model", req.model),
            content=content,
            usage=data.get("usage", {}),
            remaining_queries=remaining,
        )

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def reset_limits(self):
        """Reset all rate limits."""
        self._query_counts.clear()


# ── FastAPI routes ──────────────────────────────────────────────────────────

_proxy: Optional[LLMProxy] = None


def set_proxy(proxy: LLMProxy):
    global _proxy
    _proxy = proxy


def get_proxy() -> LLMProxy:
    if _proxy is None:
        raise HTTPException(status_code=503, detail="LLM proxy not initialized")
    return _proxy


def register_routes(app: FastAPI):
    """Register LLM proxy routes on the given FastAPI app."""

    @app.post("/llm/chat")
    async def llm_chat(req: ChatRequest, request: Request):
        """Chat completion. Requires X-Miner-UID and X-Agent-Token headers."""
        proxy = get_proxy()
        miner_uid = _extract_miner_uid(request)
        return await proxy.chat(miner_uid, req)

    @app.get("/llm/models")
    def llm_models():
        """List allowed models."""
        proxy = get_proxy()
        return {
            "models": proxy.allowed_models,
            "all_allowed": len(proxy.allowed_models) == 0,
        }

    @app.get("/llm/quota")
    def llm_quota(request: Request):
        """Check remaining LLM query quota for this miner."""
        proxy = get_proxy()
        miner_uid = _extract_miner_uid(request)
        return {
            "miner_uid": miner_uid,
            "remaining_queries": proxy.remaining_queries(miner_uid),
        }

    @app.get("/llm/health")
    def llm_health():
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
