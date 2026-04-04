"""Centralized DB API server — FastAPI app for the experiment database.

Migrated from validator/db_server.py. All routes preserved with identical
paths and response shapes. New write endpoints for validators.
Auth: Epistula verify, caller must be a validator.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared.pg_access_logger import PgAccessLogger

app = FastAPI(title="RADAR Experiment DB (Centralized)")

logger = logging.getLogger(__name__)

# Injected at startup by database/neuron.py
db = None  # PgExperimentStore
_r2 = None  # R2AuditLog (for agent code storage)
_pool = None  # asyncpg pool (for agent_submissions table)

# Auth middleware reference
_metagraph = None

# Rate limiter: hotkey -> list of timestamps
_rate_window: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
_rate_limit: int = 60  # requests per minute per validator

# IP-based rate limiter for unauthenticated / pre-auth flood protection
_ip_rate_window: dict[str, list[float]] = defaultdict(list)
_ip_rate_lock = threading.Lock()
_IP_RATE_LIMIT: int = 120  # max requests per minute per IP (pre-auth)

# Maximum request body size (5 MB) — reject oversized payloads early
_MAX_BODY_BYTES: int = 5 * 1024 * 1024

# Agent code limits
_MAX_AGENT_FILES: int = 10          # max .py files per submission
_MAX_AGENT_FILE_BYTES: int = 50_000  # 50 KB per file

# Nonce replay protection: track recently seen nonces
_nonce_cache: set[str] = set()
_nonce_timestamps: list[tuple[float, str]] = []  # (time, nonce)
_nonce_lock = threading.Lock()

# Current challenge and frontier (set by database neuron)
_current_challenge = None
_current_frontier = None

# Access logging
_access_logger: Optional[PgAccessLogger] = None
_hotkey_to_uid: dict[str, int] = {}


def set_db(experiment_db):
    global db
    db = experiment_db


def set_r2(r2):
    global _r2
    _r2 = r2


def set_pool(pool):
    global _pool
    _pool = pool


def set_auth(metagraph):
    global _metagraph
    _metagraph = metagraph


def set_rate_limit(limit: int):
    global _rate_limit
    _rate_limit = limit


def set_challenge(challenge):
    global _current_challenge
    _current_challenge = challenge


def set_frontier(frontier_data):
    global _current_frontier
    _current_frontier = frontier_data


def set_access_logger(al: PgAccessLogger):
    global _access_logger
    _access_logger = al


def set_hotkey_map(mapping: dict[str, int]):
    global _hotkey_to_uid
    _hotkey_to_uid = mapping


def _require_db():
    if db is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    return db


def _require_provenance():
    d = _require_db()
    if not hasattr(d, "provenance") or d.provenance is None:
        raise HTTPException(status_code=501, detail="Provenance not available")
    return d.provenance


def _check_rate_limit(hotkey: str) -> bool:
    with _rate_lock:
        now = time.time()
        window = _rate_window[hotkey]
        _rate_window[hotkey] = [t for t in window if now - t < 60]
        if len(_rate_window[hotkey]) >= _rate_limit:
            return False
        _rate_window[hotkey].append(now)
        return True


def _is_validator(hotkey: str) -> bool:
    """Check if a hotkey belongs to a validator (has permit)."""
    if not _metagraph:
        return False
    permits = _metagraph.validator_permit
    hotkeys = _metagraph.hotkeys
    if permits is None or hotkeys is None:
        return False
    for uid in range(_metagraph.n):
        if uid < len(hotkeys) and hotkeys[uid] == hotkey:
            return uid < len(permits) and bool(permits[uid])
    return False


def _check_ip_rate_limit(ip: str) -> bool:
    """Pre-auth IP-based rate limit to mitigate unauthenticated floods."""
    with _ip_rate_lock:
        now = time.time()
        window = _ip_rate_window[ip]
        _ip_rate_window[ip] = [t for t in window if now - t < 60]
        if len(_ip_rate_window[ip]) >= _IP_RATE_LIMIT:
            return False
        _ip_rate_window[ip].append(now)
        return True


class SearchRequest(BaseModel):
    query: str


class AddExperimentRequest(BaseModel):
    """Validator POSTs experiment data after Phase C."""
    data: dict


class RecordComponentsRequest(BaseModel):
    experiment_id: int
    components: list[str]


class RecordContextRequest(BaseModel):
    round_id: int
    experiment_id: int
    context_type: str = "frontier"


class UpdateFrontierRequest(BaseModel):
    frontier: list[dict]
    task: str = ""


# Routes that only validators (hotkeys with permit) may call
_VALIDATOR_ONLY_PREFIXES = ("/experiments/add", "/frontier/update", "/provenance/record")


def _check_nonce(nonce: str) -> bool:
    """Reject replayed nonces within the timestamp tolerance window."""
    from shared.auth import EPISTULA_TIMESTAMP_TOLERANCE
    with _nonce_lock:
        now = time.time()
        # Evict expired nonces
        cutoff = now - EPISTULA_TIMESTAMP_TOLERANCE
        while _nonce_timestamps and _nonce_timestamps[0][0] < cutoff:
            _, old = _nonce_timestamps.pop(0)
            _nonce_cache.discard(old)
        # Check for replay
        if nonce in _nonce_cache:
            return False
        _nonce_cache.add(nonce)
        _nonce_timestamps.append((now, nonce))
        return True


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Epistula auth on protected routes with IP-level flood protection."""
    # ── IP-level rate limit (pre-auth, blocks unauthenticated floods) ──
    client_ip = request.client.host if request.client else "unknown"
    if not _check_ip_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Too many requests"})

    path = request.url.path

    needs_auth = (
        path.startswith("/experiments")
        or path.startswith("/challenge")
        or path.startswith("/frontier")
        or path.startswith("/provenance")
        or path.startswith("/agent_code")
        or path.startswith("/desearch")
        or path.startswith("/llm")
    )
    if needs_auth:
        # ── Reject oversized bodies before reading fully ──
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413, content={"error": "Request body too large"},
            )

        body = await request.body()
        if len(body) > _MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413, content={"error": "Request body too large"},
            )

        # ── Validator API key: trusted proxy, skip Epistula ──
        from config import Config
        if Config.DB_API_KEY:
            import secrets as _secrets
            provided = request.headers.get("X-Radar-API-Key", "")
            if provided and _secrets.compare_digest(provided, Config.DB_API_KEY):
                # Trusted validator proxy — rate-limit by IP instead of hotkey
                if not _check_rate_limit(client_ip):
                    return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
                response = await call_next(request)
                return response

        # ── Epistula auth: miners and validators without API key ──
        if _metagraph:
            from shared.auth import verify_request
            ok, err, hotkey = verify_request(dict(request.headers), body, _metagraph)
            if not ok:
                return JSONResponse(status_code=403, content={"error": err})

            nonce = request.headers.get("x-epistula-nonce", "")
            if nonce and not _check_nonce(nonce):
                return JSONResponse(status_code=403, content={"error": "Replayed request"})

            request.state.caller_hotkey = hotkey
            if not _check_rate_limit(hotkey):
                return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

            # ── Role enforcement ──
            is_vali_route = any(path.startswith(p) for p in _VALIDATOR_ONLY_PREFIXES)
            if is_vali_route and not _is_validator(hotkey):
                return JSONResponse(status_code=403, content={"error": "Validators only"})

    response = await call_next(request)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/challenge")
def get_challenge():
    if _current_challenge is None:
        raise HTTPException(status_code=404, detail="No active challenge")
    return _current_challenge


@app.get("/frontier")
def get_frontier():
    if _current_frontier is None:
        raise HTTPException(status_code=404, detail="No frontier available")
    return _current_frontier


# ── Read endpoints (same paths as old db_server.py) ──────

@app.get("/experiments/pareto")
async def get_pareto(request: Request, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    elements = await d.get_pareto_elements(**kw)
    return [e.to_api_dict() for e in elements]


@app.get("/experiments/recent")
async def get_recent(request: Request, n: int = 20, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return [e.to_api_dict() for e in await d.get_recent(n, **kw)]


@app.get("/experiments/failures")
async def get_failures(request: Request, n: int = 10, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return [e.to_api_dict() for e in await d.get_failures(n, **kw)]


@app.get("/experiments/stats")
async def get_stats(task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return await d.stats(**kw)


@app.get("/experiments/tasks")
async def get_tasks():
    d = _require_db()
    return {"tasks": await d.get_tasks()}


@app.get("/experiments/stats/by_task")
async def get_stats_by_task():
    d = _require_db()
    return await d.stats_by_task()


@app.get("/experiments/families")
async def get_families(task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return await d.get_family_summary(**kw)


@app.get("/experiments/diff/{index_a}/{index_b}")
async def get_diff_between(index_a: int, index_b: int):
    d = _require_db()
    diff = await d.get_diff_between(index_a, index_b)
    if diff is None:
        raise HTTPException(status_code=404, detail="One or both experiments not found")
    return {"index_a": index_a, "index_b": index_b, "diff": diff}


@app.get("/experiments/{index}/diff")
async def get_diff(index: int):
    d = _require_db()
    diff = await d.get_diff(index)
    if diff is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return {"index": index, "diff": diff}


@app.get("/experiments/{index}/lineage_diffs")
async def get_lineage_diffs(index: int):
    d = _require_db()
    elem = await d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return await d.get_lineage_diffs(index)


@app.get("/experiments/lineage/{index}")
async def get_lineage(index: int):
    d = _require_db()
    elem = await d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return [e.to_api_dict() for e in await d.get_lineage(index)]


@app.post("/experiments/search")
async def search_experiments(request: Request, req: SearchRequest, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return [e.to_api_dict() for e in await d.search(req.query, **kw)]


@app.get("/experiments/{index}")
async def get_experiment(index: int):
    d = _require_db()
    elem = await d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return elem.to_api_dict()


# ── Write endpoints (new — validators only) ──────────────

@app.post("/experiments/add")
async def add_experiment(req: AddExperimentRequest):
    """Validator writes a DataElement after Phase C."""
    d = _require_db()
    from shared.database import DataElement
    element = DataElement.from_dict(req.data)
    idx = await d.add(element)
    return {"index": idx}


@app.post("/frontier/update")
async def update_frontier(req: UpdateFrontierRequest):
    """Validator pushes frontier data."""
    global _current_frontier
    _current_frontier = req.frontier
    return {"status": "ok"}


# ── Provenance endpoints ─────────────────────────────────

@app.post("/provenance/record_components")
async def record_components(req: RecordComponentsRequest):
    prov = _require_provenance()
    await prov.record_components(req.experiment_id, req.components)
    return {"status": "ok"}


@app.post("/provenance/record_context")
async def record_context(req: RecordContextRequest):
    prov = _require_provenance()
    await prov.record_round_context(
        req.round_id, req.experiment_id, req.context_type,
    )
    return {"status": "ok"}


@app.get("/provenance/{experiment_id}/influences")
async def get_influences(experiment_id: int):
    prov = _require_provenance()
    return await prov.get_influences(experiment_id)


@app.get("/provenance/{experiment_id}/impact")
async def get_impact(experiment_id: int):
    prov = _require_provenance()
    return await prov.get_impact(experiment_id)


@app.get("/provenance/{experiment_id}/similar")
async def get_similar(experiment_id: int, top_k: int = 10):
    prov = _require_provenance()
    return await prov.get_similar(experiment_id, top_k=top_k)


@app.get("/provenance/components")
async def get_component_experiments(component: str):
    prov = _require_provenance()
    return {"component": component, "experiment_ids": await prov.get_component_experiments(component)}


@app.get("/provenance/component_stats")
async def get_component_stats():
    prov = _require_provenance()
    return await prov.get_component_stats()


@app.get("/provenance/dead_ends")
async def get_dead_ends(task: str = ""):
    prov = _require_provenance()
    return {"dead_ends": await prov.get_dead_ends(task=task)}


@app.get("/provenance/{experiment_id}/graph")
async def get_experiment_graph(experiment_id: int, depth: int = 3):
    prov = _require_provenance()
    return await prov.get_experiment_graph(experiment_id, depth=depth)


# ── Agent code endpoints ───────────────────────────────────


class SubmitAgentCodeRequest(BaseModel):
    """Miner POSTs their agent code bundle."""
    files: dict[str, str]          # filename -> source code
    entry_point: str = "agent.py"  # which file has design_architecture()


@app.post("/agent_code")
async def submit_agent_code(request: Request, req: SubmitAgentCodeRequest):
    """Miner submits agent code. Validated, stored in R2, recorded in Postgres."""
    if _r2 is None:
        raise HTTPException(status_code=503, detail="R2 not configured")
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB pool not configured")

    hotkey = getattr(request.state, "caller_hotkey", "")
    if not hotkey:
        raise HTTPException(status_code=403, detail="Auth required")

    miner_uid = _hotkey_to_uid.get(hotkey, -1)

    # ── Size limits ──
    if len(req.files) > _MAX_AGENT_FILES:
        return JSONResponse(status_code=400, content={
            "error": f"Too many files ({len(req.files)}), max {_MAX_AGENT_FILES}",
        })
    for fname, code in req.files.items():
        if len(code.encode()) > _MAX_AGENT_FILE_BYTES:
            return JSONResponse(status_code=400, content={
                "error": f"File {fname!r} too large ({len(code.encode())} bytes), "
                         f"max {_MAX_AGENT_FILE_BYTES}",
            })

    from shared.agent_code import compute_code_hash, validate_bundle

    bundle = {
        "files": req.files,
        "entry_point": req.entry_point,
    }

    # Validate structure, syntax, entry point
    ok, err = validate_bundle(bundle)
    if not ok:
        return JSONResponse(status_code=400, content={"error": err})

    code_hash = compute_code_hash(req.files)
    bundle["code_hash"] = code_hash

    # Upload to R2
    r2_key = f"agents/{hotkey}/latest.json"
    try:
        _r2.upload_json(r2_key, bundle)
    except Exception as e:
        logger.error("R2 upload failed for agent code %s: %s", hotkey[:16], e)
        raise HTTPException(status_code=500, detail="Failed to store agent code")

    # Upsert into Postgres
    try:
        await _pool.execute(
            """
            INSERT INTO agent_submissions
                (hotkey, miner_uid, code_hash, entry_point, r2_key, timestamp)
            VALUES ($1, $2, $3, $4, $5, extract(epoch from now()))
            ON CONFLICT (hotkey) DO UPDATE SET
                miner_uid = $2,
                code_hash = $3,
                entry_point = $4,
                r2_key = $5,
                timestamp = extract(epoch from now())
            """,
            hotkey, miner_uid, code_hash, req.entry_point, r2_key,
        )
    except Exception as e:
        logger.error("Postgres write failed for agent code %s: %s", hotkey[:16], e)
        raise HTTPException(status_code=500, detail="Failed to record submission")

    logger.info(
        "Agent code submitted: hotkey=%s uid=%d hash=%s files=%s",
        hotkey[:16], miner_uid, code_hash[:24], sorted(req.files.keys()),
    )
    return {"status": "ok", "code_hash": code_hash, "r2_key": r2_key}


@app.get("/agent_code/{hotkey}")
async def get_agent_code(hotkey: str):
    """Validator fetches a miner's latest agent code bundle."""
    if _r2 is None:
        raise HTTPException(status_code=503, detail="R2 not configured")

    r2_key = f"agents/{hotkey}/latest.json"
    try:
        bundle = _r2.download_json(r2_key)
    except Exception:
        bundle = None

    if not bundle:
        raise HTTPException(status_code=404, detail="No agent code for this hotkey")

    return bundle


@app.get("/agent_code/{hotkey}/meta")
async def get_agent_code_meta(hotkey: str):
    """Get metadata about a miner's agent submission (no code)."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB pool not configured")

    row = await _pool.fetchrow(
        "SELECT code_hash, entry_point, timestamp FROM agent_submissions WHERE hotkey = $1",
        hotkey,
    )
    if not row:
        raise HTTPException(status_code=404, detail="No agent submission for this hotkey")

    return {
        "hotkey": hotkey,
        "code_hash": row["code_hash"],
        "entry_point": row["entry_point"],
        "timestamp": row["timestamp"],
    }
