"""Centralized DB API server — FastAPI app for the experiment database.

Migrated from validator/db_server.py. All routes preserved with identical
paths and response shapes. New write endpoints for validators.
Auth: Epistula verify, caller must be a validator.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from config import Config
from shared.access_logger import extract_ids_from_body
from shared.pg_access_logger import PgAccessLogger
from shared.pg_events import PgEventStore

app = FastAPI(title="RADAR Experiment DB (Centralized)")

# Validator-surface routes (Epistula-authed): /experiments/*, /challenge,
# /frontier, /provenance/*, /agent_code/*. Included in modes {validator, all}
# via include_validator_routes(); dashboard-only processes never mount these.
validator_router = APIRouter()

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

# Cap body size for provenance extraction — beyond this a response is a bulk
# dump (e.g. /experiments/pareto with thousands of rows) and not useful as
# per-query provenance evidence. Measured on the response side so oversized
# payloads still reach the client; we just skip ID extraction.
_ACCESS_LOG_BODY_CAP: int = 256 * 1024

# Paths whose JSON responses carry experiment IDs worth logging against the
# caller's access record. Everything else (agent_code blobs, desearch,
# /health) is skipped to avoid unnecessary body buffering.
_PROVENANCE_CAPTURE_PREFIXES = (
    "/experiments", "/frontier", "/challenge",
)

# Agent code limits
_MAX_AGENT_FILES: int = 25          # max .py files per submission
_MAX_AGENT_FILE_BYTES: int = 100_000  # 100 KB per file

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

# Validator event stream (wandb-style log + metric tail). Set by
# database/neuron.py during init; left as None in test contexts.
_event_store: Optional[PgEventStore] = None

# Maximum events accepted in a single POST /events batch — defends the DB
# server from outsized writes regardless of the validator's local config.
_MAX_EVENTS_PER_BATCH: int = 1000


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


def get_current_challenge():
    """Accessor for the active challenge (used by the dashboard)."""
    return _current_challenge


def get_current_frontier():
    """Accessor for the current frontier (used by the dashboard)."""
    return _current_frontier


def set_access_logger(al: PgAccessLogger):
    global _access_logger
    _access_logger = al


def set_event_store(es: Optional[PgEventStore]):
    global _event_store
    _event_store = es


def get_event_store() -> Optional[PgEventStore]:
    return _event_store


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


class SubmitTrainingMetaRequest(BaseModel):
    """Validator caches a training_meta.json blob in Postgres so the public
    dashboard can render loss curves without R2 credentials."""
    round_id: int
    hotkey: str
    meta: dict


class PostEventsRequest(BaseModel):
    """Validator pushes a batch of log/metric events for the public tail."""
    events: list[dict]


# Routes that only validators (hotkeys with permit) may call
_VALIDATOR_ONLY_PREFIXES = (
    "/experiments/add", "/frontier/update", "/provenance/record",
    "/training_metas", "/events",
)


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
    """Epistula auth on protected routes with IP-level flood protection.

    In ``dashboard`` mode the validator routes are not mounted at all, so
    auth/rate-limit/nonce checks are skipped entirely — the dashboard
    process only serves the public JSON API and never sees Epistula
    traffic.
    """
    # Dashboard mode: no Epistula, no nonce cache, no IP rate limit.
    # The public JSON API is open by design.
    if Config.NEURON_MODE == "dashboard":
        return await call_next(request)

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
        or path.startswith("/events")
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
        proxy_authed = False
        if Config.DB_API_KEY:
            import secrets as _secrets
            provided = request.headers.get("X-Radar-API-Key", "")
            if provided and _secrets.compare_digest(provided, Config.DB_API_KEY):
                # Trusted validator proxy — rate-limit by IP instead of hotkey
                if not _check_rate_limit(client_ip):
                    return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
                proxy_authed = True

        # ── Epistula auth: miners and validators without API key ──
        if not proxy_authed and _metagraph:
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

    # Log successful authenticated requests to Postgres for audit
    if _access_logger and needs_auth and response.status_code < 400:
        exp_ids: list[int] = []
        if any(path.startswith(p) for p in _PROVENANCE_CAPTURE_PREFIXES):
            response, exp_ids = await _buffer_and_extract_ids(response)

        miner_hotkey = request.headers.get("X-Miner-Hotkey", "")
        miner_uid_str = request.headers.get("X-Miner-UID", "-1")
        try:
            miner_uid = int(miner_uid_str)
        except ValueError:
            miner_uid = -1
        asyncio.create_task(_access_logger.log_access(
            hotkey=miner_hotkey or "validator",
            miner_uid=miner_uid,
            endpoint=path,
            experiment_ids=exp_ids,
            method=request.method,
        ))

    return response


async def _buffer_and_extract_ids(response) -> tuple[Response, list[int]]:
    """Drain a streaming response into a buffered Response + extracted IDs.

    The original ``response.body_iterator`` can only be consumed once, so we
    read it here and hand the client a new ``Response`` carrying the same
    bytes. Bodies larger than ``_ACCESS_LOG_BODY_CAP`` are passed through
    unparsed (no extraction) to avoid holding bulk dumps in memory.
    """
    chunks: list[bytes] = []
    total = 0
    over_cap = False
    async for chunk in response.body_iterator:
        chunks.append(chunk)
        total += len(chunk)
        if total > _ACCESS_LOG_BODY_CAP:
            over_cap = True
    body = b"".join(chunks)

    headers = dict(response.headers)
    headers.pop("content-length", None)
    rebuilt = Response(
        content=body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
    )
    if over_cap:
        return rebuilt, []
    content_type = response.headers.get("content-type", "")
    return rebuilt, extract_ids_from_body(body, content_type)


@app.get("/health")
def health():
    return {"status": "ok"}


@validator_router.get("/challenge")
def get_challenge():
    if _current_challenge is None:
        raise HTTPException(status_code=404, detail="No active challenge")
    return _current_challenge


@validator_router.get("/frontier")
def get_frontier():
    if _current_frontier is None:
        raise HTTPException(status_code=404, detail="No frontier available")
    return _current_frontier


# ── Read endpoints (same paths as old db_server.py) ──────

@validator_router.get("/experiments/pareto")
async def get_pareto(request: Request, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    try:
        elements = await d.get_pareto_elements(**kw)
        return [e.to_api_dict() for e in elements]
    except Exception:
        logger.exception("GET /experiments/pareto failed (task=%r)", task)
        raise


@validator_router.get("/experiments/recent")
async def get_recent(request: Request, n: int = 20, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return [e.to_api_dict() for e in await d.get_recent(n, **kw)]


@validator_router.get("/experiments/failures")
async def get_failures(request: Request, n: int = 10, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return [e.to_api_dict() for e in await d.get_failures(n, **kw)]


@validator_router.get("/experiments/stats")
async def get_stats(task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return await d.stats(**kw)


@validator_router.get("/experiments/tasks")
async def get_tasks():
    d = _require_db()
    return {"tasks": await d.get_tasks()}


@validator_router.get("/experiments/stats/by_task")
async def get_stats_by_task():
    d = _require_db()
    return await d.stats_by_task()


@validator_router.get("/experiments/families")
async def get_families(task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return await d.get_family_summary(**kw)


@validator_router.get("/experiments/diff/{index_a}/{index_b}")
async def get_diff_between(index_a: int, index_b: int):
    d = _require_db()
    diff = await d.get_diff_between(index_a, index_b)
    if diff is None:
        raise HTTPException(status_code=404, detail="One or both experiments not found")
    return {"index_a": index_a, "index_b": index_b, "diff": diff}


@validator_router.get("/experiments/{index}/diff")
async def get_diff(index: int):
    d = _require_db()
    diff = await d.get_diff(index)
    if diff is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return {"index": index, "diff": diff}


@validator_router.get("/experiments/{index}/lineage_diffs")
async def get_lineage_diffs(index: int):
    d = _require_db()
    elem = await d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return await d.get_lineage_diffs(index)


@validator_router.get("/experiments/lineage/{index}")
async def get_lineage(index: int):
    d = _require_db()
    elem = await d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return [e.to_api_dict() for e in await d.get_lineage(index)]


@validator_router.post("/experiments/search")
async def search_experiments(request: Request, req: SearchRequest, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return [e.to_api_dict() for e in await d.search(req.query, **kw)]


@validator_router.get("/experiments/{index}")
async def get_experiment(index: int):
    d = _require_db()
    elem = await d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return elem.to_api_dict()


# ── Write endpoints (new — validators only) ──────────────

@validator_router.post("/experiments/add")
async def add_experiment(req: AddExperimentRequest):
    """Validator writes a DataElement after Phase C."""
    d = _require_db()
    from shared.database import DataElement
    try:
        element = DataElement.from_dict(req.data)
        idx = await d.add(element)
    except Exception as e:
        logger.error("add_experiment failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")
    return {"index": idx}


@validator_router.post("/training_metas")
async def submit_training_meta(req: SubmitTrainingMetaRequest):
    """Validator caches a training_meta.json blob keyed by (round_id, hotkey).

    Idempotent — repeated submissions for the same key overwrite, which is
    fine since fallback dispatchers may produce a fresher meta after the
    primary trainer timed out.
    """
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB pool not configured")
    if not isinstance(req.meta, dict) or not req.meta:
        raise HTTPException(status_code=400, detail="meta must be a non-empty object")

    import json as _json

    try:
        await _pool.execute(
            """
            INSERT INTO training_metas (round_id, hotkey, meta)
            VALUES ($1, $2, $3::jsonb)
            ON CONFLICT (round_id, hotkey) DO UPDATE SET
                meta = EXCLUDED.meta,
                created_at = extract(epoch from now())
            """,
            int(req.round_id), req.hotkey, _json.dumps(req.meta),
        )
    except Exception as e:
        logger.error(
            "training_meta cache failed: round=%s hotkey=%s err=%s",
            req.round_id, req.hotkey[:16], e,
        )
        raise HTTPException(status_code=500, detail="Failed to cache training meta")
    return {"status": "ok"}


@validator_router.post("/events")
async def post_events(request: Request, req: PostEventsRequest):
    """Validator flushes a batch of log/metric events.

    The caller's hotkey is taken from the Epistula signature
    (``request.state.caller_hotkey``); validators cannot post events on
    behalf of another hotkey. Events with unknown ``kind`` are silently
    dropped — the validator-side library only ever produces valid kinds,
    and we don't want a misbehaving client to noisily fail the whole
    batch.
    """
    if _event_store is None:
        raise HTTPException(status_code=503, detail="Event store not configured")
    hotkey = getattr(request.state, "caller_hotkey", "")
    if not hotkey:
        # Trusted-proxy mode skips Epistula; fall back to an explicit
        # header so deployments using DB_API_KEY can still attribute
        # events to a specific validator.
        hotkey = request.headers.get("X-Validator-Hotkey", "")
    if not hotkey:
        raise HTTPException(status_code=403, detail="Hotkey required")

    events = req.events or []
    if len(events) > _MAX_EVENTS_PER_BATCH:
        events = events[-_MAX_EVENTS_PER_BATCH:]

    try:
        n = await _event_store.insert_batch(hotkey, events)
    except Exception as e:
        logger.error("post_events insert failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Event insert failed")
    return {"status": "ok", "inserted": n}


@validator_router.post("/frontier/update")
async def update_frontier(req: UpdateFrontierRequest):
    """Validator pushes frontier data."""
    global _current_frontier
    from shared.pg_schema import _sanitize_for_json
    _current_frontier = _sanitize_for_json(req.frontier)
    return {"status": "ok"}


# ── Provenance endpoints ─────────────────────────────────

@validator_router.post("/provenance/record_components")
async def record_components(req: RecordComponentsRequest):
    prov = _require_provenance()
    await prov.record_components(req.experiment_id, req.components)
    return {"status": "ok"}


@validator_router.post("/provenance/record_context")
async def record_context(req: RecordContextRequest):
    prov = _require_provenance()
    await prov.record_round_context(
        req.round_id, req.experiment_id, req.context_type,
    )
    return {"status": "ok"}


@validator_router.get("/provenance/{experiment_id}/influences")
async def get_influences(experiment_id: int):
    prov = _require_provenance()
    return await prov.get_influences(experiment_id)


@validator_router.get("/provenance/{experiment_id}/impact")
async def get_impact(experiment_id: int):
    prov = _require_provenance()
    return await prov.get_impact(experiment_id)


@validator_router.get("/provenance/{experiment_id}/similar")
async def get_similar(experiment_id: int, top_k: int = 10):
    prov = _require_provenance()
    return await prov.get_similar(experiment_id, top_k=top_k)


@validator_router.get("/provenance/components")
async def get_component_experiments(component: str):
    prov = _require_provenance()
    return {"component": component, "experiment_ids": await prov.get_component_experiments(component)}


@validator_router.get("/provenance/component_stats")
async def get_component_stats():
    prov = _require_provenance()
    try:
        return await prov.get_component_stats()
    except Exception:
        logger.exception("GET /provenance/component_stats failed")
        raise HTTPException(status_code=500, detail="Component stats query failed")


@validator_router.get("/provenance/dead_ends")
async def get_dead_ends(task: str = ""):
    prov = _require_provenance()
    return {"dead_ends": await prov.get_dead_ends(task=task)}


@validator_router.get("/provenance/{experiment_id}/graph")
async def get_experiment_graph(experiment_id: int, depth: int = 3):
    prov = _require_provenance()
    return await prov.get_experiment_graph(experiment_id, depth=depth)


# ── Agent code endpoints ───────────────────────────────────


class SubmitAgentCodeRequest(BaseModel):
    """Miner POSTs their agent code bundle."""
    files: dict[str, str]          # filename -> source code
    entry_point: str = "agent.py"  # which file has design_architecture()


@validator_router.post("/agent_code")
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

    # Content-addressed blob (immutable) + mutable "latest" pointer for
    # readers that still want the current code without a DB round trip.
    immutable_key = f"agents/{hotkey}/{code_hash}.json"
    latest_key = f"agents/{hotkey}/latest.json"
    try:
        _r2.upload_json(immutable_key, bundle)
        _r2.upload_json(latest_key, bundle)
    except Exception as e:
        logger.error("R2 upload failed for agent code %s: %s", hotkey[:16], e)
        raise HTTPException(status_code=500, detail="Failed to store agent code")

    # Best-effort current round — populated when a challenge is active.
    round_submitted = -1
    if isinstance(_current_challenge, dict):
        try:
            round_submitted = int(_current_challenge.get("round_id", -1))
        except (TypeError, ValueError):
            round_submitted = -1

    # Upsert into registry + append to history + cache the bundle JSON in one
    # transaction so we never end up with a live registry row that has no
    # matching audit entry, and the public dashboard can serve the bundle
    # straight from Postgres without reaching back into R2.
    import json as _json

    bundle_json = _json.dumps(bundle)
    try:
        async with _pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO agent_submissions
                        (hotkey, miner_uid, code_hash, entry_point, r2_key,
                         round_submitted, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, extract(epoch from now()))
                    ON CONFLICT (hotkey) DO UPDATE SET
                        miner_uid = $2,
                        code_hash = $3,
                        entry_point = $4,
                        r2_key = $5,
                        round_submitted = $6,
                        timestamp = extract(epoch from now())
                    """,
                    hotkey, miner_uid, code_hash, req.entry_point,
                    immutable_key, round_submitted,
                )
                await conn.execute(
                    """
                    INSERT INTO agent_submission_history
                        (hotkey, miner_uid, code_hash, entry_point, r2_key,
                         round_submitted, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, extract(epoch from now()))
                    """,
                    hotkey, miner_uid, code_hash, req.entry_point,
                    immutable_key, round_submitted,
                )
                # Content-addressed; ON CONFLICT DO NOTHING because identical
                # bytes always hash to the same code_hash.
                await conn.execute(
                    """
                    INSERT INTO agent_bundles (code_hash, bundle)
                    VALUES ($1, $2::jsonb)
                    ON CONFLICT (code_hash) DO NOTHING
                    """,
                    code_hash, bundle_json,
                )
    except Exception as e:
        logger.error("Postgres write failed for agent code %s: %s", hotkey[:16], e)
        raise HTTPException(status_code=500, detail="Failed to record submission")

    logger.info(
        "Agent code submitted: hotkey=%s uid=%d hash=%s round=%d files=%s",
        hotkey[:16], miner_uid, code_hash[:24], round_submitted,
        sorted(req.files.keys()),
    )
    return {
        "status": "ok",
        "code_hash": code_hash,
        "r2_key": immutable_key,
        "round_submitted": round_submitted,
    }


@validator_router.get("/agent_code/{hotkey}")
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


@validator_router.get("/agent_code/{hotkey}/meta")
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


@validator_router.get("/agent_code/{hotkey}/history")
async def get_agent_code_history(hotkey: str, limit: int = 100):
    """Full submission timeline for a hotkey (most recent first)."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB pool not configured")

    limit = max(1, min(int(limit), 500))
    rows = await _pool.fetch(
        """
        SELECT code_hash, entry_point, r2_key, round_submitted, timestamp
        FROM agent_submission_history
        WHERE hotkey = $1
        ORDER BY timestamp DESC
        LIMIT $2
        """,
        hotkey, limit,
    )
    return {
        "hotkey": hotkey,
        "submissions": [
            {
                "code_hash": r["code_hash"],
                "entry_point": r["entry_point"],
                "r2_key": r["r2_key"],
                "round_submitted": int(r["round_submitted"]),
                "timestamp": float(r["timestamp"]),
            }
            for r in rows
        ],
    }


@validator_router.get("/agent_code/by_hash/{code_hash}")
async def get_agent_code_by_hash(code_hash: str):
    """Fetch an immutable agent bundle by its content hash.

    Lets callers replay the exact bytes that were active for a given round
    even after the miner has uploaded a new version.
    """
    if _r2 is None:
        raise HTTPException(status_code=503, detail="R2 not configured")
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB pool not configured")

    row = await _pool.fetchrow(
        "SELECT r2_key FROM agent_submission_history "
        "WHERE code_hash = $1 ORDER BY timestamp DESC LIMIT 1",
        code_hash,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Unknown code_hash")

    try:
        bundle = _r2.download_json(row["r2_key"])
    except Exception:
        bundle = None
    if not bundle:
        raise HTTPException(status_code=404, detail="Bundle missing in R2")
    return bundle


def include_validator_routes(target_app: FastAPI) -> None:
    """Mount the Epistula-authed validator/miner surface onto ``target_app``.

    Idempotent — safe to call multiple times. Called from ``database.neuron``
    in modes ``validator`` and ``all``. Dashboard-only processes never call
    this, so /experiments/*, /challenge, /frontier, /provenance/*, and
    /agent_code/* return 404 there.
    """
    already_mounted = any(
        getattr(r, "path", None) == "/challenge" for r in target_app.routes
    )
    if already_mounted:
        return
    target_app.include_router(validator_router)


# Legacy auto-mount: keep the validator surface on ``app`` at import time
# unless this process is running in ``dashboard`` mode. Preserves backward
# compatibility for tests that import ``app`` directly and expect every
# route to be reachable.
if Config.NEURON_MODE != "dashboard":
    include_validator_routes(app)
