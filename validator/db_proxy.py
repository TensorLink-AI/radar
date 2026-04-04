"""Reverse proxy — validators run this for their miners.

Forwards /experiments/*, /challenge, /frontier, /provenance/* to the
centralized database server. Hosts /desearch/* locally. Rate limits
per miner hotkey.
"""

from __future__ import annotations

import logging
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
_client: Optional[httpx.AsyncClient] = None

# Rate limiter: hotkey -> list of timestamps
_rate_window: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
_rate_limit: int = 10


def set_config(db_api_url: str, wallet, metagraph, rate_limit: int = 10):
    """Configure the proxy at startup."""
    global _db_api_url, _wallet, _metagraph, _rate_limit
    _db_api_url = db_api_url.rstrip("/")
    _wallet = wallet
    _metagraph = metagraph
    _rate_limit = rate_limit


def set_metagraph(metagraph):
    """Update metagraph (called on sync)."""
    global _metagraph
    _metagraph = metagraph


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


def _check_rate_limit(hotkey: str) -> bool:
    with _rate_lock:
        now = time.time()
        window = _rate_window[hotkey]
        _rate_window[hotkey] = [t for t in window if now - t < 60]
        if len(_rate_window[hotkey]) >= _rate_limit:
            return False
        _rate_window[hotkey].append(now)
        return True


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Epistula auth for miner requests on proxied routes."""
    path = request.url.path
    needs_auth = (
        path.startswith("/experiments")
        or path.startswith("/challenge")
        or path.startswith("/frontier")
        or path.startswith("/provenance")
        or path.startswith("/agent_code")
    )

    if needs_auth and _metagraph:
        body = await request.body()
        from shared.auth import verify_request
        ok, err, hotkey = verify_request(dict(request.headers), body, _metagraph)
        if not ok:
            return JSONResponse(status_code=403, content={"error": err})
        request.state.miner_hotkey = hotkey
        if not _check_rate_limit(hotkey):
            return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

    response = await call_next(request)
    return response


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
    if body:
        headers["Content-Type"] = request.headers.get("content-type", "application/json")

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


@app.post("/agent_code")
async def proxy_submit_agent_code(request: Request):
    return await _proxy_request(request, "/agent_code")


@app.get("/agent_code/{hotkey}")
async def proxy_get_agent_code(request: Request, hotkey: str):
    return await _proxy_request(request, f"/agent_code/{hotkey}")


@app.get("/agent_code/{hotkey}/meta")
async def proxy_get_agent_code_meta(request: Request, hotkey: str):
    return await _proxy_request(request, f"/agent_code/{hotkey}/meta")
