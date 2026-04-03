"""Centralized DB API server — FastAPI app for the experiment database.

Migrated from validator/db_server.py. All routes preserved with identical
paths and response shapes. New write endpoints for validators.
Auth: Epistula verify, caller must be a validator.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared.access_logger import _extract_experiment_ids
from shared.pg_access_logger import PgAccessLogger

app = FastAPI(title="RADAR Experiment DB (Centralized)")

logger = logging.getLogger(__name__)

# Injected at startup by database/neuron.py
db = None  # PgExperimentStore

# Auth middleware reference
_auth_verify = None
_metagraph = None

# Rate limiter: hotkey -> list of timestamps
_rate_window: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
_rate_limit: int = 60  # requests per minute per validator

# Current challenge and frontier (set by database neuron)
_current_challenge = None
_current_frontier = None

# Access logging
_access_logger: Optional[PgAccessLogger] = None
_hotkey_to_uid: dict[str, int] = {}


def set_db(experiment_db):
    global db
    db = experiment_db


def set_auth(metagraph, verify_fn=None):
    global _auth_verify, _metagraph
    _metagraph = metagraph
    _auth_verify = verify_fn


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


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Epistula auth on protected routes."""
    path = request.url.path
    needs_auth = (
        path.startswith("/experiments")
        or path.startswith("/challenge")
        or path.startswith("/frontier")
        or path.startswith("/provenance")
    )
    if needs_auth and _auth_verify and _metagraph:
        body = await request.body()
        from shared.auth import verify_request
        ok, err, hotkey = verify_request(dict(request.headers), body, _metagraph)
        if not ok:
            return JSONResponse(status_code=403, content={"error": err})
        request.state.caller_hotkey = hotkey
        if not _check_rate_limit(hotkey):
            return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

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
