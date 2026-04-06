"""LLM proxy — full OpenAI-compatible passthrough to Chutes AI.

Runs on the subnet owner's database server (NOT on validators).
Validators forward /llm/* requests here. The Chutes API key never
leaves the owner's machine.

Supports all OpenAI-compatible endpoints: chat completions, text
completions, embeddings, and model listing — including SSE streaming.
Rate limiting and model validation happen before forwarding.

All queries are logged to Postgres (proxy_query_log table).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import AsyncIterator, Optional

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

DEFAULT_CHUTES_URL = "https://llm.chutes.ai/v1"
MAX_QUERIES_PER_TEMPO = 50
TEMPO_DURATION_SECONDS = 360 * 12
_BLOCKED_FIELDS = {"api_key", "user"}
_MAX_TOKENS_CAP = 16384


class LLMProxy:
    """Rate-limited passthrough proxy for Chutes AI (OpenAI-compatible)."""

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
        self.pool = pool
        self.timeout = timeout
        self._query_counts: dict[int, list[float]] = defaultdict(list)
        self._client: Optional[httpx.AsyncClient] = None
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0
        self._CB_THRESHOLD = 3
        self._CB_COOLDOWN = 60.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _prune_old_queries(self, miner_uid: int) -> int:
        cutoff = time.time() - self.tempo_seconds
        ts = self._query_counts[miner_uid]
        self._query_counts[miner_uid] = [t for t in ts if t > cutoff]
        return len(self._query_counts[miner_uid])

    def remaining_queries(self, miner_uid: int) -> int:
        return max(0, self.max_queries - self._prune_old_queries(miner_uid))

    def _record_query(self, miner_uid: int):
        self._query_counts[miner_uid].append(time.time())

    def _validate_model(self, model: str) -> None:
        if not self.allowed_models:
            return
        if model not in self.allowed_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model!r} not allowed. Allowed: {self.allowed_models}",
            )

    def _check_circuit(self, ctx: str) -> None:
        now = time.time()
        if self._consecutive_failures >= self._CB_THRESHOLD and now < self._circuit_open_until:
            wait_s = self._circuit_open_until - now
            logger.warning("Chutes AI circuit open [%s] (failures=%d, %.0fs)",
                           ctx, self._consecutive_failures, wait_s)
            raise HTTPException(status_code=503, detail=(
                f"LLM unavailable ({self._consecutive_failures} failures, retry {wait_s:.0f}s)"))

    def _open_circuit(self, double: bool = False, timeout: bool = False):
        self._consecutive_failures += 1
        cooldown = self._CB_COOLDOWN * (2 if double else 1)
        self._circuit_open_until = time.time() + cooldown
        # Timeouts are strong evidence the upstream is down — trip the
        # circuit breaker immediately instead of waiting for _CB_THRESHOLD.
        if timeout and self._consecutive_failures < self._CB_THRESHOLD:
            self._consecutive_failures = self._CB_THRESHOLD

    def _reset_circuit(self, ctx: str):
        if self._consecutive_failures > 0:
            logger.info("Chutes AI recovered [%s] after %d failures",
                        ctx, self._consecutive_failures)
        self._consecutive_failures = 0

    def auth_headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.chutes_api_key:
            h["Authorization"] = f"Bearer {self.chutes_api_key}"
        return h

    async def _log_query(self, uid: int, hotkey: str,
                         model: str, query: str, content: str, tokens: int):
        if not self.pool:
            return
        try:
            await self.pool.execute(
                "INSERT INTO proxy_query_log "
                "(service,miner_uid,miner_hotkey,query_text,model,"
                "response_summary,tokens_used,timestamp) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,extract(epoch from now()))",
                "llm", uid, hotkey, query[:500], model, content[:200], tokens,
            )
        except Exception as e:
            logger.debug("LLM query log failed: %s", e)

    def _prepare(self, miner_uid: int, payload: dict) -> tuple[str, int]:
        """Validate, sanitise payload. Returns (model, remaining)."""
        remaining = self.remaining_queries(miner_uid)
        if remaining <= 0:
            raise HTTPException(status_code=429,
                                detail=f"Rate limit exceeded ({self.max_queries}/tempo).")
        model = payload.get("model", "")
        if not model:
            raise HTTPException(status_code=400, detail="Missing 'model' field")
        self._validate_model(model)
        for f in _BLOCKED_FIELDS:
            payload.pop(f, None)
        if payload.get("max_tokens", 0) > _MAX_TOKENS_CAP:
            payload["max_tokens"] = _MAX_TOKENS_CAP
        return model, remaining

    async def forward(
        self, path: str, miner_uid: int, payload: dict, miner_hotkey: str = "",
    ) -> dict | AsyncIterator[bytes]:
        """Forward request to Chutes AI. Returns dict or SSE byte iterator."""
        model, remaining = self._prepare(miner_uid, payload)
        ctx = f"miner={miner_uid} model={model}"
        self._check_circuit(ctx)
        self._record_query(miner_uid)
        remaining -= 1

        streaming = payload.get("stream", False)
        headers = {"Content-Type": "application/json", **self.auth_headers()}
        url = f"{self.chutes_url}/{path.lstrip('/')}"

        if streaming:
            return await self._forward_stream(
                url, payload, headers, ctx, remaining, miner_uid, miner_hotkey, model)

        return await self._forward_json(
            url, payload, headers, ctx, remaining, miner_uid, miner_hotkey, model)

    async def _forward_json(
        self, url: str, payload: dict, headers: dict, ctx: str,
        remaining: int, uid: int, hotkey: str, model: str,
    ) -> dict:
        max_retries, backoff, t0 = 3, 2.0, time.time()
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                self._check_circuit(ctx)
            try:
                client = await self._get_client()
                resp = await client.post(url, json=payload, headers=headers,
                                         timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if attempt > 0:
                    logger.info("Chutes AI OK attempt %d [%s] (%.1fs)",
                                attempt + 1, ctx, time.time() - t0)
                break
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                body = (e.response.text or "")[:200]
                if status in (429, 502, 503) and attempt < max_retries:
                    ra = e.response.headers.get("Retry-After")
                    wait = min(float(ra) if ra else backoff * (2 ** attempt), 30.0)
                    logger.warning("Chutes AI HTTP %s [%s]: %s — retry %.1fs (%d/%d)",
                                   status, ctx, body, wait, attempt + 1, max_retries)
                    await asyncio.sleep(wait)
                    continue
                self._open_circuit(status in (429, 503))
                logger.error("Chutes AI HTTP %s [%s] %.1fs: %s (failures=%d)",
                             status, ctx, time.time() - t0, body, self._consecutive_failures)
                raise HTTPException(status_code=502,
                                    detail=f"LLM error: HTTP {status} — {body}".strip())
            except httpx.TimeoutException as e:
                last_error = e
                # Timeouts are expensive (120s wasted). Only retry once,
                # then open the circuit so subsequent calls fail fast.
                is_timeout = True
                if attempt < 1:  # at most 1 retry for timeouts
                    wait = min(backoff * (2 ** attempt), 30.0)
                    logger.warning("Chutes AI ReadTimeout [%s] (timeout=%.0fs) — retry %.1fs (%d/1)",
                                   ctx, self.timeout, wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                self._open_circuit(False, timeout=True)
                elapsed = time.time() - t0
                logger.error("Chutes AI ReadTimeout [%s] %.1fs — circuit OPEN (failures=%d)",
                             ctx, elapsed, self._consecutive_failures)
                raise HTTPException(status_code=502, detail=(
                    f"LLM unreachable: ReadTimeout ({elapsed:.0f}s)"))
            except httpx.RequestError as e:
                last_error = e
                etype = type(e).__name__
                if attempt < max_retries:
                    wait = min(backoff * (2 ** attempt), 30.0)
                    logger.warning("Chutes AI %s [%s] (timeout=%.0fs) — retry %.1fs (%d/%d)",
                                   etype, ctx, self.timeout, wait, attempt + 1, max_retries)
                    await asyncio.sleep(wait)
                    continue
                self._open_circuit(False)
                elapsed = time.time() - t0
                logger.error("Chutes AI %s [%s] %.1fs (failures=%d)",
                             etype, ctx, elapsed, self._consecutive_failures)
                raise HTTPException(status_code=502, detail=(
                    f"LLM unreachable: {etype} ({elapsed:.0f}s)"))
        else:
            self._open_circuit(True)
            elapsed = time.time() - t0
            sc = last_error.response.status_code if isinstance(
                last_error, httpx.HTTPStatusError) else "?"
            logger.error("Chutes AI retries exhausted [%s] (status=%s, %.1fs)",
                         ctx, sc, elapsed)
            raise HTTPException(status_code=502,
                                detail=f"LLM error (HTTP {sc}) after retries ({elapsed:.0f}s)")

        self._reset_circuit(ctx)
        await self._log_from_response(uid, hotkey, model, payload, data)
        data["remaining_queries"] = remaining
        return data

    async def _forward_stream(
        self, url: str, payload: dict, headers: dict, ctx: str,
        remaining: int, uid: int, hotkey: str, model: str,
    ) -> AsyncIterator[bytes]:
        """SSE streaming passthrough (no retry mid-stream)."""
        client = await self._get_client()
        req = client.build_request("POST", url, json=payload, headers=headers)
        try:
            resp = await client.send(req, stream=True)
        except (httpx.RequestError, httpx.TimeoutException) as e:
            etype = type(e).__name__
            is_timeout = isinstance(e, httpx.TimeoutException)
            self._open_circuit(False, timeout=is_timeout)
            logger.error("Chutes AI stream %s [%s] (failures=%d)",
                         etype, ctx, self._consecutive_failures)
            raise HTTPException(status_code=502,
                                detail=f"LLM unreachable: {etype}")
        if resp.status_code >= 400:
            body = (await resp.aread()).decode(errors="replace")[:200]
            await resp.aclose()
            self._open_circuit(resp.status_code in (429, 503))
            logger.error("Chutes AI stream HTTP %s [%s]: %s",
                         resp.status_code, ctx, body)
            raise HTTPException(status_code=502,
                                detail=f"LLM error: HTTP {resp.status_code} — {body}")
        self._reset_circuit(ctx)

        async def generate() -> AsyncIterator[bytes]:
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await self._log_query(uid, hotkey, model, "", "(stream)", 0)

        return generate()

    async def _log_from_response(self, uid: int, hotkey: str,
                                 model: str, payload: dict, data: dict):
        query, content, tokens = "", "", 0
        try:
            msgs = payload.get("messages", [])
            if msgs:
                c = msgs[-1].get("content", "")
                query = c[:500] if isinstance(c, str) else str(c)[:500]
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content") or msg.get("reasoning_content") or ""
                if not isinstance(content, str):
                    content = str(content)
            tokens = data.get("usage", {}).get("total_tokens", 0)
        except Exception:
            pass
        await self._log_query(uid, hotkey, model, query, content, tokens)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def reset_limits(self):
        self._query_counts.clear()
