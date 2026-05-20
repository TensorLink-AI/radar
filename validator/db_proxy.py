"""Reverse proxy — validators run this for agent pods.

Forwards /experiments/*, /challenge, /frontier, /provenance/* to the
centralized database server. Also proxies /desearch/*, /llm/*.
Rate limits per miner UID per route category.

Agent pods authenticate via a per-round ephemeral token injected into
the challenge JSON (``X-Agent-Token`` header). Miner neuron processes
submit agent code directly to the DB server (not through this proxy).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from shared.auth import sign_request, verify_request_hmac
from shared.protocol import TrainerReady

logger = logging.getLogger(__name__)

app = FastAPI(title="RADAR Validator Proxy")

# Warm-standby trainer readiness: round_id → {uid: TrainerReady}
_trainer_ready: dict[int, dict[int, object]] = {}
_hotkey_to_uid: dict[str, int] = {}

# ── Per-round agent token ──────────────────────────────────────────
# A short-lived random token generated each round by the validator.
# Injected into the challenge JSON and required on all agent-facing
# proxy routes via the ``X-Agent-Token`` header.
_agent_token: str = ""


def rotate_agent_token() -> str:
    """Generate a new agent token for the current round.

    Returns the new token (caller injects it into the challenge JSON).
    """
    global _agent_token
    _agent_token = secrets.token_urlsafe(32)
    logger.info("Agent token rotated (token=%s...)", _agent_token[:8])
    return _agent_token


def get_agent_token() -> str:
    """Return the current agent token."""
    return _agent_token


def get_ready_trainers(round_id: int) -> dict:
    """Called by coordinator to poll for TrainerReady responses."""
    return dict(_trainer_ready.get(round_id, {}))


def clear_ready_trainers(round_id: int):
    """Cleanup after round completes."""
    _trainer_ready.pop(round_id, None)


def set_hotkey_map(mapping: dict[str, int]):
    global _hotkey_to_uid
    _hotkey_to_uid = mapping

# Injected at startup
_db_api_url: str = ""
_wallet = None
_api_key: str = ""
_client: Optional[httpx.AsyncClient] = None

# Per-route-category rate limits: (max_requests, window_seconds).
# Each category gets its own independent bucket so heavy DB reads
# don't starve LLM or search budgets.
# Round duration is ~55 min; round-scoped limits use 3600s (1 hour).
def _build_default_category_limits() -> dict[str, tuple[int, int]]:
    from config import Config
    return {
        "db":         (Config.DB_VALI_RATE_LIMIT, 60),
        "desearch":   (Config.DESEARCH_MAX_QUERIES, 3600),
        "llm":        (Config.LLM_MAX_QUERIES, 3600),
    }

_CATEGORY_LIMITS: dict[str, tuple[int, int]] = {}
_DEFAULT_LIMIT: tuple[int, int] = (5, 60)

# Rate limiter: "category:identity" -> list of timestamps
_rate_window: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()


def _route_category(path: str) -> str:
    """Map a request path to its rate-limit category."""
    if path.startswith("/desearch"):
        return "desearch"
    if path.startswith("/llm"):
        return "llm"
    return "db"


def set_config(
    db_api_url: str, wallet=None, api_key: str = "",
    rate_limits: dict[str, tuple[int, int]] | None = None,
    **kwargs,
):
    """Configure the proxy at startup.

    ``rate_limits`` overrides per-category limits as
    ``{category: (max_requests, window_seconds)}``.
    """
    global _db_api_url, _wallet, _api_key
    _db_api_url = db_api_url.rstrip("/")
    _wallet = wallet
    _api_key = api_key
    if not _CATEGORY_LIMITS:
        _CATEGORY_LIMITS.update(_build_default_category_limits())
    if rate_limits:
        _CATEGORY_LIMITS.update(rate_limits)




async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


# LLM timeout: the llm_proxy layer handles its own retries and circuit
# breaking against Chutes AI.  The db_proxy is a thin passthrough — its
# timeout just needs to cover one llm_proxy round-trip.  Keep well under
# the GatedClient's llm_timeout (120s) so we never do work after the
# agent has already given up.
_LLM_TIMEOUT = httpx.Timeout(90.0, connect=5.0)


def _check_rate_limit(identity: str, category: str) -> bool:
    """Check per-category rate limit for a given identity (hotkey or UID)."""
    max_req, window_secs = _CATEGORY_LIMITS.get(category, _DEFAULT_LIMIT)
    key = f"{category}:{identity}"
    with _rate_lock:
        now = time.time()
        window = _rate_window[key]
        _rate_window[key] = [t for t in window if now - t < window_secs]
        if len(_rate_window[key]) >= max_req:
            return False
        _rate_window[key].append(now)
        return True


def _verify_agent_token(request: Request) -> bool:
    """Check if the request carries a valid agent token."""
    if not _agent_token:
        return False
    token = request.headers.get("X-Agent-Token", "")
    return secrets.compare_digest(token, _agent_token)


# Routes that accept agent token auth (used by agent pods)
_AGENT_TOKEN_PREFIXES = (
    "/experiments", "/challenge", "/frontier",
    "/provenance", "/desearch", "/llm",
)



@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Auth for proxy routes.

    Agent pods authenticate via ``X-Agent-Token`` (ephemeral per-round secret).
    """
    path = request.url.path

    needs_auth = any(path.startswith(p) for p in _AGENT_TOKEN_PREFIXES)
    if not needs_auth:
        response = await call_next(request)
        return response

    category = _route_category(path)

    # Agent token is the primary auth for pod requests
    if _verify_agent_token(request):
        rate_key = request.headers.get("X-Miner-UID", "agent-shared")
        if not _check_rate_limit(f"agent:{rate_key}", category):
            return JSONResponse(status_code=429, content={
                "error": f"Rate limit exceeded for {category}",
            })
        response = await call_next(request)
        logger.info(
            "Agent proxy: miner=%s %s %s -> %d",
            rate_key, request.method, path, response.status_code,
        )
        return response

    return JSONResponse(status_code=403, content={"error": "Invalid agent token"})


def _build_proxy_headers(request: Request, body: bytes) -> dict:
    """Build signed headers with forwarded miner identity."""
    headers = {}
    if _wallet:
        headers = sign_request(_wallet, body)
    if _api_key:
        headers["X-Radar-API-Key"] = _api_key
    if body:
        headers["Content-Type"] = request.headers.get("content-type", "application/json")
    for fwd in ("X-Miner-UID", "X-Miner-Hotkey"):
        val = request.headers.get(fwd, "")
        if val:
            headers[fwd] = val
    if "X-Miner-UID" not in headers:
        headers["X-Miner-UID"] = "0"
    return headers


def _build_target(path: str, request: Request) -> str:
    target = f"{_db_api_url}{path}"
    query = str(request.url.query)
    return f"{target}?{query}" if query else target


async def _proxy_request(request: Request, path: str) -> Response:
    """Forward a request to the database server, signed with validator wallet."""
    if not _db_api_url:
        return JSONResponse(
            status_code=503,
            content={"error": "DB API URL not configured"},
        )

    client = await _get_client()
    body = await request.body()
    headers = _build_proxy_headers(request, body)
    target = _build_target(path, request)
    timeout = _LLM_TIMEOUT if path.startswith("/llm") else None

    # Check if this is a streaming LLM request (agent sent stream: true)
    is_stream_request = False
    if path.startswith("/llm") and body:
        try:
            import json
            is_stream_request = json.loads(body).get("stream", False)
        except Exception:
            pass

    if is_stream_request:
        return await _proxy_stream(client, target, body, headers, timeout)

    is_llm = path.startswith("/llm")
    # No retries at the proxy layer — the llm_proxy has its own retry +
    # circuit breaker.  Retrying here just doubles the timeout burn.
    max_retries = 0
    retry_budget = 35.0
    t0 = time.time()

    for attempt in range(1 + max_retries):
        elapsed = time.time() - t0
        if attempt > 0 and elapsed > retry_budget:
            logger.warning(
                "Proxy retry budget exhausted (%.0fs/%.0fs) for %s",
                elapsed, retry_budget, target,
            )
            return JSONResponse(
                status_code=502,
                content={"error": "Retry budget exhausted"},
            )

        # Re-sign on retries so the Epistula timestamp stays fresh
        if attempt > 0:
            headers = _build_proxy_headers(request, body)
        try:
            if request.method == "GET":
                resp = await client.get(target, headers=headers, timeout=timeout)
            elif request.method == "POST":
                resp = await client.post(target, content=body, headers=headers,
                                         timeout=timeout)
            else:
                resp = await client.request(request.method, target, content=body,
                                            headers=headers, timeout=timeout)

            # Retry transient upstream errors for LLM requests
            if resp.status_code in (502, 503) and attempt < max_retries:
                import asyncio as _aio
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    "Proxy %s returned %d — retry %.0fs (%d/%d)",
                    target, resp.status_code, wait, attempt + 1, max_retries,
                )
                await _aio.sleep(wait)
                continue

            # For error responses, ensure the body is always valid JSON so
            # downstream clients (GatedClient.get_json / post_json) can parse it.
            if resp.status_code >= 400:
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type="application/json",
                    )
                # Non-JSON error body — wrap it in a JSON envelope
                import json as _json
                body_text = resp.content.decode(errors="replace")[:500]
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": body_text or "Unknown error"},
                )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            if attempt < max_retries:
                import asyncio as _aio
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    "Proxy request to %s failed: %s — retry %.0fs (%d/%d)",
                    target, e or type(e).__name__, wait,
                    attempt + 1, max_retries,
                )
                await _aio.sleep(wait)
                continue  # headers re-signed at top of loop
            logger.warning("Proxy request to %s failed: %s", target, e or type(e).__name__)
            return JSONResponse(
                status_code=502,
                content={"error": "Database server unreachable"},
            )


async def _proxy_stream(
    client: httpx.AsyncClient, target: str, body: bytes,
    headers: dict, timeout,
) -> StreamingResponse:
    """Stream an SSE response from the database server back to the caller."""
    try:
        req = client.build_request("POST", target, content=body, headers=headers)
        resp = await client.send(req, stream=True, timeout=timeout)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.warning("Proxy stream to %s failed: %s", target, e or type(e).__name__)
        return JSONResponse(
            status_code=502,
            content={"error": "Database server unreachable"},
        )

    if resp.status_code >= 400:
        error_body = (await resp.aread()).decode(errors="replace")
        await resp.aclose()
        return Response(content=error_body, status_code=resp.status_code,
                        media_type="application/json")

    async def generate():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "proxy": True}


@app.post("/trainer/ready")
async def trainer_ready(request: Request):
    """Record that a miner's trainer pod is up and ready to accept jobs.

    Body: JSON-encoded ``TrainerReady`` payload (round_id, uid, hotkey,
    trainer_url, ...). Authenticated via HMAC over the raw body — caller
    must include ``X-Radar-Signature`` keyed by ``RADAR_SHARED_SECRET``.
    """
    import os as _os
    import json as _json
    from shared.peers import get_peer_by_hotkey

    body = await request.body()

    secret = _os.getenv("RADAR_SHARED_SECRET", "")
    if secret:
        signature = (
            request.headers.get("X-Radar-Signature", "")
            or request.headers.get("x-radar-signature", "")
        )
        if not signature or not verify_request_hmac(body, signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
    else:
        logger.warning(
            "trainer_ready: RADAR_SHARED_SECRET unset — accepting unsigned request (dev mode)",
        )

    try:
        data = _json.loads(body) if body else {}
    except _json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    round_id = int(data.get("round_id", 0))
    hotkey = str(data.get("miner_hotkey", "") or data.get("hotkey", ""))
    trainer_url = str(data.get("trainer_url", ""))
    instance_name = str(data.get("instance_name", ""))

    uid = data.get("miner_uid")
    if uid is None:
        peer = get_peer_by_hotkey(hotkey)
        uid = peer.uid if peer else _hotkey_to_uid.get(hotkey, -1)
    uid = int(uid)

    ready = TrainerReady(
        round_id=round_id,
        miner_hotkey=hotkey,
        trainer_url=trainer_url,
        instance_name=instance_name,
    )
    _trainer_ready.setdefault(round_id, {})[uid] = ready
    logger.info(
        "trainer_ready: round=%d uid=%d hotkey=%s url=%s",
        round_id, uid, hotkey[:16], trainer_url,
    )
    return JSONResponse(status_code=200, content={"status": "ok"})


# ── Proxied routes ───────────────────────────────────────

@app.get("/challenge")
async def proxy_challenge(request: Request):
    return await _proxy_request(request, "/challenge")


@app.get("/frontier")
async def proxy_frontier(request: Request):
    return await _proxy_request(request, "/frontier")


@app.api_route("/experiments/{path:path}", methods=["GET", "POST"])
async def proxy_experiments(request: Request, path: str):
    return await _proxy_request(request, f"/experiments/{path}")


@app.api_route("/provenance/{path:path}", methods=["GET", "POST"])
async def proxy_provenance(request: Request, path: str):
    return await _proxy_request(request, f"/provenance/{path}")


# ── LLM proxy (forwarded to DB server) ────────────────────

@app.api_route("/llm/{path:path}", methods=["GET", "POST"])
async def proxy_llm(request: Request, path: str):
    return await _proxy_request(request, f"/llm/{path}")


# ── Desearch proxy (forwarded to DB server) ────────────────

@app.api_route("/desearch/{path:path}", methods=["GET", "POST"])
async def proxy_desearch(request: Request, path: str):
    return await _proxy_request(request, f"/desearch/{path}")
