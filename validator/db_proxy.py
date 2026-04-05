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
from fastapi.responses import JSONResponse, Response

from shared.auth import sign_request

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
_metagraph = None
_api_key: str = ""
_client: Optional[httpx.AsyncClient] = None

# Per-route-category rate limits: (max_requests, window_seconds).
# Each category gets its own independent bucket so heavy DB reads
# don't starve LLM or search budgets.
# Round duration is ~55 min; round-scoped limits use 3600s (1 hour).
_CATEGORY_LIMITS: dict[str, tuple[int, int]] = {
    "db":         (5, 60),      #  5 req / min
    "desearch":   (10, 3600),   # 10 req / round
    "llm":        (30, 3600),   # 30 req / round
}
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
    db_api_url: str, wallet, metagraph,
    api_key: str = "",
    rate_limits: dict[str, tuple[int, int]] | None = None,
):
    """Configure the proxy at startup.

    ``rate_limits`` overrides per-category limits as
    ``{category: (max_requests, window_seconds)}``.
    """
    global _db_api_url, _wallet, _metagraph, _api_key
    _db_api_url = db_api_url.rstrip("/")
    _wallet = wallet
    _metagraph = metagraph
    _api_key = api_key
    if rate_limits:
        _CATEGORY_LIMITS.update(rate_limits)


def set_metagraph(metagraph):
    """Update metagraph (called on sync)."""
    global _metagraph
    _metagraph = metagraph


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


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
        return response

    # Fall back to Epistula for backward compat
    if _metagraph:
        body = await request.body()
        from shared.auth import verify_request
        ok, err, hotkey = verify_request(dict(request.headers), body, _metagraph)
        if ok:
            request.state.miner_hotkey = hotkey
            if not _check_rate_limit(hotkey, category):
                return JSONResponse(status_code=429, content={
                    "error": f"Rate limit exceeded for {category}",
                })
            response = await call_next(request)
            return response

    return JSONResponse(status_code=403, content={"error": "Invalid agent token or signature"})


async def _proxy_request(request: Request, path: str) -> Response:
    """Forward a request to the database server, signed with validator wallet."""
    if not _db_api_url:
        raise HTTPException(status_code=503, detail="DB API URL not configured")

    client = await _get_client()
    body = await request.body()

    # Sign outgoing request with validator wallet
    headers = {}
    if _wallet:
        headers = sign_request(_wallet, body)
    if _api_key:
        headers["X-Radar-API-Key"] = _api_key
    if body:
        headers["Content-Type"] = request.headers.get("content-type", "application/json")

    # Forward miner identity headers (used by LLM/desearch proxies on DB server)
    # Agent pods don't know their UID, so inject a fallback for routes that
    # require X-Miner-UID (e.g. /llm/chat rate limiting on the DB server).
    for fwd_header in ("X-Miner-UID", "X-Miner-Hotkey"):
        val = request.headers.get(fwd_header, "")
        if val:
            headers[fwd_header] = val
    if "X-Miner-UID" not in headers:
        headers["X-Miner-UID"] = "0"

    # Build target URL with query string
    target = f"{_db_api_url}{path}"
    query = str(request.url.query)
    if query:
        target += f"?{query}"

    try:
        if request.method == "GET":
            resp = await client.get(target, headers=headers)
        elif request.method == "POST":
            resp = await client.post(target, content=body, headers=headers)
        else:
            resp = await client.request(
                request.method, target, content=body, headers=headers,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.warning("Proxy request to %s failed: %s", target, e)
        raise HTTPException(status_code=502, detail="Database server unreachable")


@app.get("/health")
def health():
    return {"status": "ok", "proxy": True}


@app.post("/trainer/ready")
async def trainer_ready(request: Request):
    """Miner POSTs here after spinning up their Basilica pod."""
    if not _metagraph:
        raise HTTPException(status_code=503, detail="Metagraph not configured")

    body = await request.body()
    from shared.auth import verify_request
    ok, err, hotkey = verify_request(dict(request.headers), body, _metagraph)
    if not ok:
        return JSONResponse(status_code=403, content={"error": err})

    from shared.protocol import TrainerReady
    try:
        ready_msg = TrainerReady.from_json(body.decode())
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid TrainerReady: {e}"})

    uid = _hotkey_to_uid.get(hotkey)
    if uid is None:
        return JSONResponse(status_code=404, content={"error": "Unknown hotkey"})

    round_id = ready_msg.round_id
    if round_id not in _trainer_ready:
        _trainer_ready[round_id] = {}
    _trainer_ready[round_id][uid] = ready_msg

    return {"status": "ok", "uid": uid, "round_id": round_id}


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
