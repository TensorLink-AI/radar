"""FastAPI routes for the LLM proxy (Chutes AI passthrough).

Registers OpenAI-compatible endpoints on the database server's FastAPI app:
  POST /llm/chat                — chat completions (legacy)
  POST /llm/v1/chat/completions — chat completions (standard)
  POST /llm/v1/completions      — text completions
  POST /llm/v1/embeddings       — embeddings
  GET  /llm/v1/models           — upstream model list (filtered by allowlist)
  GET  /llm/models              — local allowlist
  GET  /llm/quota               — per-miner rate limit status
  GET  /llm/health              — health check
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from validator.llm_proxy import LLMProxy

logger = logging.getLogger(__name__)


async def _forward_with_disconnect(
    request: Request, proxy: LLMProxy,
    path: str, uid: int, payload: dict, hotkey: str,
):
    """Forward to Chutes AI, but cancel if the downstream client disconnects.

    Prevents zombie Chutes requests from piling up when the agent's
    shorter timeout fires before the Chutes call completes.
    """
    forward_task = asyncio.create_task(
        proxy.forward(path, uid, payload, hotkey),
    )
    while not forward_task.done():
        if await request.is_disconnected():
            forward_task.cancel()
            logger.info("Client disconnected, cancelled Chutes request [miner=%d path=%s]", uid, path)
            raise HTTPException(status_code=499, detail="Client disconnected")
        await asyncio.sleep(0.5)
    return forward_task.result()

_MAX_BODY_BYTES = 256 * 1024
_proxy: Optional[LLMProxy] = None


def set_proxy(proxy: LLMProxy):
    global _proxy
    _proxy = proxy


def get_proxy() -> LLMProxy:
    if _proxy is None:
        raise HTTPException(status_code=503, detail="LLM proxy not initialized")
    return _proxy


def register_routes(app: FastAPI):
    """Register full OpenAI-compatible LLM proxy routes."""

    @app.post("/llm/chat")
    @app.post("/llm/v1/chat/completions")
    async def llm_chat(request: Request):
        proxy = get_proxy()
        uid, hotkey, payload = await _parse_request(request)
        result = await _forward_with_disconnect(request, proxy, "chat/completions", uid, payload, hotkey)
        return _to_response(result)

    @app.post("/llm/v1/completions")
    async def llm_completions(request: Request):
        proxy = get_proxy()
        uid, hotkey, payload = await _parse_request(request)
        result = await _forward_with_disconnect(request, proxy, "completions", uid, payload, hotkey)
        return _to_response(result)

    @app.post("/llm/v1/embeddings")
    async def llm_embeddings(request: Request):
        proxy = get_proxy()
        uid, hotkey, payload = await _parse_request(request)
        result = await _forward_with_disconnect(request, proxy, "embeddings", uid, payload, hotkey)
        return _to_response(result)

    @app.get("/llm/v1/models")
    async def llm_upstream_models():
        """Proxy model list from Chutes AI, filtered by allowlist."""
        proxy = get_proxy()
        try:
            client = await proxy._get_client()
            resp = await client.get(f"{proxy.chutes_url}/models",
                                    headers=proxy.auth_headers(), timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Failed to fetch upstream models: %s", e)
            raise HTTPException(status_code=502, detail="Cannot reach upstream models")
        if proxy.allowed_models:
            data["data"] = [m for m in data.get("data", [])
                            if m.get("id") in proxy.allowed_models]
        return JSONResponse(content=data)

    @app.get("/llm/models")
    def llm_models():
        proxy = get_proxy()
        return {"models": proxy.allowed_models,
                "all_allowed": len(proxy.allowed_models) == 0}

    @app.get("/llm/quota")
    def llm_quota(request: Request):
        proxy = get_proxy()
        uid = _extract_miner_uid(request)
        return {"miner_uid": uid, "remaining_queries": proxy.remaining_queries(uid)}

    @app.get("/llm/health")
    def llm_health():
        return {"status": "ok", "proxy_initialized": _proxy is not None}


def _to_response(result: dict | AsyncIterator[bytes]):
    if isinstance(result, dict):
        return JSONResponse(content=result)
    return StreamingResponse(result, media_type="text/event-stream")


async def _parse_request(request: Request) -> tuple[int, str, dict]:
    uid = _extract_miner_uid(request)
    hotkey = request.headers.get("X-Miner-Hotkey", "")
    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    return uid, hotkey, payload


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
