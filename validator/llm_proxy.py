"""LLM proxy — rate-limited proxy for miner agents to query Chutes AI.

Runs on the subnet owner's database server (NOT on validators).
Validators forward /llm/* requests here. The Chutes API key never
leaves the owner's machine.

Passes through the full OpenAI-compatible request/response so agents
can use tool calls, streaming, JSON mode, etc. Rate limiting and
model validation happen before forwarding.

All queries are logged to Postgres (proxy_query_log table).
Rate-limited to RADAR_LLM_MAX_QUERIES per miner per tempo.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

# Chutes AI inference endpoint
DEFAULT_CHUTES_URL = "https://llm.chutes.ai/v1"

# Rate limit: max LLM queries per miner per tempo
MAX_QUERIES_PER_TEMPO = 50

# Tempo duration in seconds (~72 min, matching desearch)
TEMPO_DURATION_SECONDS = 360 * 12

# Fields agents are NOT allowed to set (security / cost control)
_BLOCKED_REQUEST_FIELDS = {"api_key", "user", "stream"}

# Max request body size (256 KB) to prevent abuse
_MAX_BODY_BYTES = 256 * 1024


class LLMProxy:
    """Rate-limited passthrough proxy for LLM inference via Chutes AI.

    The subnet owner controls:
      - Which models are available (allowed_models)
      - The API key (chutes_api_key)
      - Rate limits per miner per tempo

    Passes through full OpenAI-compatible JSON so agents can use
    tool_calls, response_format, etc.
    """

    def __init__(
        self,
        chutes_url: str = DEFAULT_CHUTES_URL,
        chutes_api_key: str = "",
        allowed_models: list[str] | None = None,
        max_queries: int = MAX_QUERIES_PER_TEMPO,
        tempo_seconds: int = TEMPO_DURATION_SECONDS,
        pool=None,
        timeout: float = 120.0,
    ):
        self.chutes_url = chutes_url.rstrip("/")
        self.chutes_api_key = chutes_api_key
        self.allowed_models = allowed_models or []
        self.max_queries = max_queries
        self.tempo_seconds = tempo_seconds
        self.pool = pool  # asyncpg pool for query logging
        self.timeout = timeout

        # miner_uid -> list of query timestamps in current window
        self._query_counts: dict[int, list[float]] = defaultdict(list)
        self._client: Optional[httpx.AsyncClient] = None

        # Circuit breaker: avoid hammering a dead upstream
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0
        self._CB_THRESHOLD = 3       # failures before opening circuit
        self._CB_COOLDOWN = 60.0     # seconds to wait before retrying

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
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

    async def _log_query(
        self, miner_uid: int, miner_hotkey: str,
        model: str, query_text: str, content: str, tokens: int,
    ):
        """Log query to Postgres (best-effort, never raises)."""
        if not self.pool:
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO proxy_query_log
                    (service, miner_uid, miner_hotkey, query_text, model,
                     response_summary, tokens_used, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, extract(epoch from now()))
                """,
                "llm", miner_uid, miner_hotkey, query_text[:500], model,
                content[:200], tokens,
            )
        except Exception as e:
            logger.debug("LLM query log failed: %s", e)

    async def chat(
        self, miner_uid: int, payload: dict, miner_hotkey: str = "",
    ) -> dict:
        """Forward an OpenAI-compatible chat request to Chutes AI.

        Returns the full upstream JSON response with remaining_queries added.
        Raises HTTPException(429) on rate limit, 400 on bad model.
        """
        remaining = self.remaining_queries(miner_uid)
        if remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_queries} LLM queries per tempo.",
            )

        # Validate model
        model = payload.get("model", "")
        if not model:
            raise HTTPException(status_code=400, detail="Missing 'model' field")
        self._validate_model(model)

        # Strip blocked fields
        for field in _BLOCKED_REQUEST_FIELDS:
            payload.pop(field, None)

        # Enforce max_tokens cap
        if payload.get("max_tokens", 0) > 16384:
            payload["max_tokens"] = 16384

        # Context tag for log messages
        ctx = f"miner={miner_uid} model={model}"

        # Circuit breaker: fail fast if upstream is known-broken
        now = time.time()
        if self._consecutive_failures >= self._CB_THRESHOLD and now < self._circuit_open_until:
            wait_s = self._circuit_open_until - now
            logger.warning("Chutes AI circuit open [%s] (failures=%d, retry in %.0fs)",
                           ctx, self._consecutive_failures, wait_s)
            raise HTTPException(status_code=503, detail=(
                f"LLM upstream unavailable ({self._consecutive_failures} "
                f"failures, retry in {wait_s:.0f}s)"))

        self._record_query(miner_uid)
        remaining -= 1
        headers = {"Content-Type": "application/json"}
        if self.chutes_api_key:
            headers["Authorization"] = f"Bearer {self.chutes_api_key}"

        max_retries, backoff = 3, 2.0
        last_error: Exception | None = None
        t0 = time.time()

        for attempt in range(max_retries + 1):
            now = time.time()
            if (attempt > 0 and self._consecutive_failures >= self._CB_THRESHOLD
                    and now < self._circuit_open_until):
                logger.warning("Chutes AI circuit opened mid-retry [%s] %.1fs in",
                               ctx, now - t0)
                raise HTTPException(status_code=503, detail=(
                    f"LLM upstream unavailable ({self._consecutive_failures} failures)"))

            try:
                client = await self._get_client()
                resp = await client.post(f"{self.chutes_url}/chat/completions",
                                         json=payload, headers=headers,
                                         timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if attempt > 0:
                    logger.info("Chutes AI OK on attempt %d [%s] (%.1fs)",
                                attempt + 1, ctx, time.time() - t0)
                break
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                body = e.response.text[:200] if e.response.text else ""
                if status in (429, 502, 503) and attempt < max_retries:
                    ra = e.response.headers.get("Retry-After")
                    wait = min(float(ra) if ra else backoff * (2 ** attempt), 30.0)
                    logger.warning("Chutes AI HTTP %s [%s]: %s — retry in %.1fs (%d/%d)",
                                   status, ctx, body, wait, attempt + 1, max_retries)
                    await asyncio.sleep(wait)
                    continue
                self._consecutive_failures += 1
                cd = self._CB_COOLDOWN * 2 if status in (429, 503) else self._CB_COOLDOWN
                self._circuit_open_until = time.time() + cd
                logger.error("Chutes AI HTTP %s [%s] after %.1fs: %s "
                             "(failures=%d, circuit %.0fs)",
                             status, ctx, time.time() - t0, body,
                             self._consecutive_failures, cd)
                raise HTTPException(status_code=502,
                                    detail=f"LLM upstream error: HTTP {status} — {body}".strip())
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                etype = type(e).__name__
                if attempt < max_retries:
                    wait = min(backoff * (2 ** attempt), 30.0)
                    logger.warning("Chutes AI %s [%s] (timeout=%.0fs) — retry in %.1fs (%d/%d)",
                                   etype, ctx, self.timeout, wait, attempt + 1, max_retries)
                    await asyncio.sleep(wait)
                    continue
                self._consecutive_failures += 1
                self._circuit_open_until = time.time() + self._CB_COOLDOWN
                elapsed = time.time() - t0
                logger.error("Chutes AI %s [%s] after %.1fs (timeout=%.0fs, "
                             "failures=%d, circuit %.0fs)",
                             etype, ctx, elapsed, self.timeout,
                             self._consecutive_failures, self._CB_COOLDOWN)
                raise HTTPException(status_code=502, detail=(
                    f"LLM unreachable: {etype} after {max_retries + 1} attempts ({elapsed:.0f}s)"))
        else:
            self._consecutive_failures += 1
            cd = self._CB_COOLDOWN * 2
            self._circuit_open_until = time.time() + cd
            elapsed = time.time() - t0
            sc = last_error.response.status_code if isinstance(last_error, httpx.HTTPStatusError) else "?"
            logger.error("Chutes AI exhausted retries [%s] (status=%s, %.1fs, "
                         "failures=%d, circuit %.0fs)",
                         ctx, sc, elapsed, self._consecutive_failures, cd)
            raise HTTPException(status_code=502, detail=(
                f"LLM upstream error (HTTP {sc}) after {max_retries + 1} attempts ({elapsed:.0f}s)"))

        if self._consecutive_failures > 0:
            logger.info("Chutes AI recovered [%s] after %d consecutive failures",
                        ctx, self._consecutive_failures)
        self._consecutive_failures = 0

        # Extract summary for logging (best-effort)
        content = ""
        query_text = ""
        tokens = 0
        try:
            messages = payload.get("messages", [])
            if messages:
                last_msg = messages[-1]
                c = last_msg.get("content", "")
                query_text = c[:500] if isinstance(c, str) else str(c)[:500]
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content") or msg.get("reasoning_content") or ""
                if not isinstance(content, str):
                    content = str(content)
            tokens = data.get("usage", {}).get("total_tokens", 0)
        except Exception:
            pass  # logging extraction should never fail the request

        await self._log_query(miner_uid, miner_hotkey, model, query_text, content, tokens)

        # Inject remaining quota into response for agent convenience
        data["remaining_queries"] = remaining

        return data

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
    async def llm_chat(request: Request):
        """OpenAI-compatible chat completion. Requires X-Miner-UID header.

        Accepts the full OpenAI request body (messages, tools, tool_choice,
        response_format, etc.) and returns the full upstream response.
        """
        proxy = get_proxy()
        miner_uid = _extract_miner_uid(request)
        miner_hotkey = request.headers.get("X-Miner-Hotkey", "")

        # Read raw JSON body (with size limit)
        body = await request.body()
        if len(body) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        data = await proxy.chat(miner_uid, payload, miner_hotkey)
        return JSONResponse(content=data)

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
