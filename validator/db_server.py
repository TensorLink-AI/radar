"""FastAPI server that exposes the experiment DB to miners.

Read-only for miners. Only the validator writes to the DB.
Includes Epistula auth middleware, rate limiting, and access logging.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared.access_logger import AccessLogger

app = FastAPI(title="RADAR Experiment DB")

# Injected at startup by the validator (ExperimentDB or SQLiteExperimentStore)
db = None

# Auth middleware reference (set by validator if auth enabled)
_auth_verify = None
_metagraph = None

# Rate limiter: hotkey -> list of timestamps
_rate_window: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
_rate_limit: int = 10  # requests per minute per miner

# Current challenge and frontier (set by validator each round)
_current_challenge = None
_current_frontier = None

# Access logging
_access_logger: Optional[AccessLogger] = None
_hotkey_to_uid: dict[str, int] = {}


def set_db(experiment_db):
    global db
    db = experiment_db


def set_auth(metagraph, verify_fn=None):
    """Enable Epistula auth on experiment routes."""
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


def set_access_logger(logger: AccessLogger):
    global _access_logger
    _access_logger = logger


def set_hotkey_map(mapping: dict[str, int]):
    global _hotkey_to_uid
    _hotkey_to_uid = mapping


def _require_db():
    if db is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    return db


def _require_provenance():
    d = _require_db()
    if not hasattr(d, "provenance"):
        raise HTTPException(status_code=501, detail="Provenance not available")
    return d.provenance  # ProvenanceQuery instance


def _check_rate_limit(hotkey: str) -> bool:
    """Thread-safe sliding-window rate limiter. Returns True if allowed."""
    with _rate_lock:
        now = time.time()
        window = _rate_window[hotkey]
        _rate_window[hotkey] = [t for t in window if now - t < 60]
        if len(_rate_window[hotkey]) >= _rate_limit:
            return False
        _rate_window[hotkey].append(now)
        return True


def _extract_experiment_ids_from_path(path: str) -> list[int]:
    """Parse experiment IDs from URL patterns."""
    patterns = [
        r"/experiments/(\d+)",
        r"/experiments/lineage/(\d+)",
        r"/experiments/diff/(\d+)/(\d+)",
    ]
    ids: list[int] = []
    for pat in patterns:
        for m in re.finditer(pat, path):
            ids.extend(int(g) for g in m.groups())
    return ids


def _log_miner_access(hotkey: str, path: str, method: str):
    """Log a miner's API access if the access logger is configured."""
    if _access_logger is None:
        return
    exp_ids = _extract_experiment_ids_from_path(path)
    uid = _hotkey_to_uid.get(hotkey, -1)
    _access_logger.log_request(
        hotkey=hotkey, endpoint=path, method=method,
        miner_uid=uid,
    )


def _log_response_ids(request: Request, response_data):
    """Log experiment IDs from response body for list endpoints."""
    if _access_logger is None:
        return
    hotkey = getattr(request.state, "miner_hotkey", None)
    if not hotkey:
        return
    uid = _hotkey_to_uid.get(hotkey, -1)
    _access_logger.log_request(
        hotkey=hotkey, endpoint=request.url.path,
        method=request.method, response_data=response_data,
        miner_uid=uid,
    )


class SearchRequest(BaseModel):
    query: str


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Epistula auth on /experiments/* routes if auth is enabled."""
    path = request.url.path
    needs_auth = (
        path.startswith("/experiments")
        or path.startswith("/challenge")
        or path.startswith("/frontier")
    )

    if needs_auth and _auth_verify and _metagraph:
        body = await request.body()
        from shared.auth import verify_request
        ok, err, hotkey = verify_request(dict(request.headers), body, _metagraph)
        if not ok:
            return JSONResponse(status_code=403, content={"error": err})
        request.state.miner_hotkey = hotkey
        if not _check_rate_limit(hotkey):
            return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

    response = await call_next(request)

    # Log access after successful response
    if needs_auth and hasattr(request.state, "miner_hotkey"):
        _log_miner_access(request.state.miner_hotkey, path, request.method)

    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/challenge")
def get_challenge():
    """Return the current round's challenge."""
    if _current_challenge is None:
        raise HTTPException(status_code=404, detail="No active challenge")
    return _current_challenge


@app.get("/frontier")
def get_frontier():
    """Return the current Pareto frontier."""
    if _current_frontier is None:
        raise HTTPException(status_code=404, detail="No frontier available")
    return _current_frontier


# NOTE: Literal paths MUST come before {index} parameterized route
@app.get("/experiments/pareto")
def get_pareto(request: Request, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    elements = d.get_pareto_elements(**kw)
    result = [e.to_api_dict() for e in elements]
    _log_response_ids(request, result)
    return result


@app.get("/experiments/recent")
def get_recent(request: Request, n: int = 20, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    result = [e.to_api_dict() for e in d.get_recent(n, **kw)]
    _log_response_ids(request, result)
    return result


@app.get("/experiments/failures")
def get_failures(request: Request, n: int = 10, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    result = [e.to_api_dict() for e in d.get_failures(n, **kw)]
    _log_response_ids(request, result)
    return result


@app.get("/experiments/stats")
def get_stats(task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    return d.stats(**kw)


@app.get("/experiments/tasks")
def get_tasks():
    """List all tasks that have experiments in the DB."""
    d = _require_db()
    if not hasattr(d, "get_tasks"):
        raise HTTPException(status_code=501, detail="Task listing not supported by this DB backend")
    return {"tasks": d.get_tasks()}


@app.get("/experiments/stats/by_task")
def get_stats_by_task():
    """Return stats for each task."""
    d = _require_db()
    if not hasattr(d, "stats_by_task"):
        raise HTTPException(status_code=501, detail="Per-task stats not supported by this DB backend")
    return d.stats_by_task()


@app.get("/experiments/families")
def get_families(task: str = ""):
    """Return summary of architectural families, optionally filtered by task."""
    d = _require_db()
    kw = {"task": task} if task else {}
    return d.get_family_summary(**kw)


@app.get("/experiments/diff/{index_a}/{index_b}")
def get_diff_between(index_a: int, index_b: int):
    """Return unified diff between any two experiments."""
    d = _require_db()
    diff = d.get_diff_between(index_a, index_b)
    if diff is None:
        raise HTTPException(status_code=404, detail="One or both experiments not found")
    return {"index_a": index_a, "index_b": index_b, "diff": diff}


@app.get("/experiments/{index}/diff")
def get_diff(index: int):
    """Return unified diff between experiment and its parent."""
    d = _require_db()
    diff = d.get_diff(index)
    if diff is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return {"index": index, "diff": diff}


@app.get("/experiments/{index}/lineage_diffs")
def get_lineage_diffs(index: int):
    """Return full ancestry chain with diffs at each step."""
    d = _require_db()
    elem = d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return d.get_lineage_diffs(index)


@app.get("/experiments/lineage/{index}")
def get_lineage(index: int):
    d = _require_db()
    elem = d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return [e.to_api_dict() for e in d.get_lineage(index)]


@app.post("/experiments/search")
def search_experiments(request: Request, req: SearchRequest, task: str = ""):
    d = _require_db()
    kw = {"task": task} if task else {}
    result = [e.to_api_dict() for e in d.search(req.query, **kw)]
    _log_response_ids(request, result)
    return result


@app.get("/experiments/{index}")
def get_experiment(index: int):
    d = _require_db()
    elem = d.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return elem.to_api_dict()


# ── Provenance endpoints ─────────────────────────────────

@app.get("/provenance/{experiment_id}/influences")
def get_influences(experiment_id: int):
    prov = _require_provenance()
    return prov.get_influences(experiment_id)


@app.get("/provenance/{experiment_id}/impact")
def get_impact(experiment_id: int):
    prov = _require_provenance()
    return prov.get_impact(experiment_id)


@app.get("/provenance/{experiment_id}/similar")
def get_similar(experiment_id: int, top_k: int = 10):
    prov = _require_provenance()
    return prov.get_similar(experiment_id, top_k=top_k)


@app.get("/provenance/components")
def get_component_experiments(component: str):
    prov = _require_provenance()
    return {"component": component, "experiment_ids": prov.get_component_experiments(component)}


@app.get("/provenance/component_stats")
def get_component_stats():
    prov = _require_provenance()
    return prov.get_component_stats()


@app.get("/provenance/dead_ends")
def get_dead_ends(task: str = ""):
    prov = _require_provenance()
    return {"dead_ends": prov.get_dead_ends(task=task)}


@app.get("/provenance/{experiment_id}/graph")
def get_experiment_graph(experiment_id: int, depth: int = 3):
    prov = _require_provenance()
    return prov.get_experiment_graph(experiment_id, depth=depth)
