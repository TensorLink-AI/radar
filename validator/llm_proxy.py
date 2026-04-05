"""
LLM proxy — HTTP proxy for miners to use a shared LLM endpoint.

Rate-limited per miner per tempo. Runs as additional routes
on the existing FastAPI DB server.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Tempo duration in seconds (~72 min)
TEMPO_DURATION_SECONDS = 360 * 12


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    """Chat completion request from a miner."""

    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=16384)


class ChatResponse(BaseModel):
    """Chat completion response returned to miners."""

    content: str = ""
    model: str = ""
    remaining_requests: int = 0


class LLMProxy:
    """
    Rate-limited proxy for LLM chat completions.

    Each miner gets max_requests queries per tempo window.
    Supports OpenAI-compatible and Anthropic APIs.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4.1",
        api_key: str = "",
        base_url: str = "",
        max_requests: int = 50,
        tempo_seconds: int = TEMPO_DURATION_SECONDS,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or self._default_api_key()
        self.base_url = base_url or self._default_base_url()
        self.max_requests = max_requests
        self.tempo_seconds = tempo_seconds

        self._request_counts: dict[int, list[float]] = defaultdict(list)
        self._client: Optional[httpx.AsyncClient] = None

    def _default_api_key(self) -> str:
        if self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        return os.getenv("OPENAI_API_KEY", "")

    def _default_base_url(self) -> str:
        if self.provider == "anthropic":
            return "https://api.anthropic.com"
        return "https://api.openai.com/v1"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    def _prune_old_requests(self, miner_uid: int) -> int:
        cutoff = time.time() - self.tempo_seconds
        timestamps = self._request_counts[miner_uid]
        self._request_counts[miner_uid] = [t for t in timestamps if t > cutoff]
        return len(self._request_counts[miner_uid])

    def remaining_requests(self, miner_uid: int) -> int:
        used = self._prune_old_requests(miner_uid)
        return max(0, self.max_requests - used)

    def _record_request(self, miner_uid: int):
        self._request_counts[miner_uid].append(time.time())

    async def chat(self, miner_uid: int, req: ChatRequest) -> ChatResponse:
        """Forward a chat completion request to the LLM provider."""
        remaining = self.remaining_requests(miner_uid)
        if remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per tempo.",
            )

        self._record_request(miner_uid)
        remaining -= 1
        model = req.model or self.model

        try:
            if self.provider == "anthropic":
                result = await self._call_anthropic(req, model)
            else:
                result = await self._call_openai(req, model)
        except httpx.HTTPStatusError as e:
            logger.warning("LLM upstream error: %s", e.response.status_code)
            raise HTTPException(status_code=502, detail="LLM upstream error")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning("LLM request failed: %s", e)
            raise HTTPException(status_code=502, detail="LLM upstream unreachable")

        return ChatResponse(
            content=result, model=model, remaining_requests=remaining,
        )

    async def _call_openai(self, req: ChatRequest, model: str) -> str:
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": m.role, "content": m.content} for m in req.messages],
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def _call_anthropic(self, req: ChatRequest, model: str) -> str:
        client = await self._get_client()
        # Extract system message if present
        system = ""
        messages = []
        for m in req.messages:
            if m.role == "system":
                system = m.content
            else:
                messages.append({"role": m.role, "content": m.content})
        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if system:
            body["system"] = system
        resp = await client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def reset_limits(self):
        self._request_counts.clear()


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
        """Chat completion proxy. Requires X-Miner-UID header."""
        proxy = get_proxy()
        miner_uid = _extract_miner_uid(request)
        return await proxy.chat(miner_uid, req)

    @app.get("/llm/quota")
    def llm_quota(request: Request):
        """Check remaining LLM request quota for this miner."""
        proxy = get_proxy()
        miner_uid = _extract_miner_uid(request)
        return {
            "miner_uid": miner_uid,
            "remaining_requests": proxy.remaining_requests(miner_uid),
        }

    @app.get("/llm/health")
    def llm_health():
        return {"status": "ok", "proxy_initialized": _proxy is not None}


def _extract_miner_uid(request: Request) -> int:
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
