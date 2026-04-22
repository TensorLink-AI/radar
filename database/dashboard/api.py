"""JSON endpoints the dashboard UI fetches from the browser.

These routes are PUBLIC. The SPA on radarnet.io/dashboard/ calls them with
``credentials: 'omit'`` and no login — every field returned here is
world-readable. Jinja pages rendered server-side also fetch these endpoints
while holding a cookie; dropping the auth dependency lets both audiences
hit the same router without divergent behavior.

Do not add fields to any response here that shouldn't be visible to the
entire internet.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from config import Config
from database.dashboard import logs as log_helpers
from database.dashboard.app import get_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard/api")

# Max points sent to the browser for a loss curve — Chart.js starts
# struggling past a few thousand points.
_MAX_LOSS_POINTS = 500


def _downsample(points: list[float], limit: int = _MAX_LOSS_POINTS) -> list[float]:
    if not points:
        return []
    if len(points) <= limit:
        return [float(p) for p in points if p is not None]
    # Stride-based downsample, always keep first and last
    step = max(1, len(points) // limit)
    sampled = list(points[::step])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return [float(p) for p in sampled if p is not None]


@router.get("/loss_curve/{index}.json")
async def loss_curve(index: int) -> dict:
    state = get_state()
    row = await state.pool.fetchrow(
        "SELECT loss_curve FROM experiments WHERE id = $1", index,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="experiment not found")

    from shared.pg_schema import _decode_jsonb
    curve = _decode_jsonb(row["loss_curve"], [])
    return {"index": index, "points": _downsample(curve)}


@router.get("/pareto.json")
async def pareto_json(
    task: str = "",
    min_flops: int | None = None,
    max_flops: int | None = None,
) -> dict:
    """Scatter data — every successful experiment with a flops+metric, flagged
    as dominated vs on-frontier via ``shared.pareto.ParetoFront``.
    """
    state = get_state()
    kw = {"task": task} if task else {}
    elements = await state.store.get_pareto_elements(**kw)

    # Apply optional FLOPs window
    def _flops(e):
        return int(e.objectives.get("flops_equivalent_size") or 0)

    if min_flops is not None:
        elements = [e for e in elements if _flops(e) >= min_flops]
    if max_flops is not None:
        elements = [e for e in elements if _flops(e) <= max_flops]

    # Build a Pareto front to identify frontier members
    from shared.pareto import ParetoFront

    def _objective(elem):
        metric = elem.metric if elem.metric is not None else float("inf")
        flops = _flops(elem) or float("inf")
        return (metric, flops)

    pf = ParetoFront(max_size=len(elements) + 1, objective_fn=_objective)
    for e in elements:
        pf.update(e)
    frontier_ids = {c.element.index for c in pf.candidates}

    points = []
    for e in elements:
        flops = _flops(e)
        if not flops or e.metric is None:
            continue
        points.append({
            "id": e.index,
            "name": e.name,
            "task": e.task,
            "flops": flops,
            "metric": float(e.metric),
            "miner_hotkey": e.miner_hotkey,
            "on_frontier": e.index in frontier_ids,
        })
    return {"task": task, "points": points}


@router.get("/stats.json")
async def stats_json(task: str = "") -> dict:
    state = get_state()
    kw = {"task": task} if task else {}
    return await state.store.stats(**kw)


@router.get("/provenance/miner_rounds.json")
async def provenance_miner_rounds(rounds: int = 30) -> dict:
    """Miner × round heatmap — unique experiments queried per round."""
    from database.dashboard import queries as q

    state = get_state()
    rounds = max(1, min(int(rounds), 100))
    return await q.miner_round_activity(state.pool, rounds=rounds)


@router.get("/provenance/top_experiments.json")
async def provenance_top_experiments(top_k: int = 20) -> dict:
    """Miner × top-K experiments heatmap — raw query counts per pair."""
    from database.dashboard import queries as q

    state = get_state()
    top_k = max(1, min(int(top_k), 100))
    return await q.top_experiments_activity(state.pool, top_k=top_k)


# ── SPA summary endpoints ─────────────────────────────────────

@router.get("/tasks_stats.json")
async def tasks_stats_json() -> dict:
    """Per-task stats, keyed by task name."""
    state = get_state()
    return await state.store.stats_by_task()


@router.get("/recent.json")
async def recent_json(n: int = 20, task: str = "") -> list[dict]:
    """Most recent experiments (newest first)."""
    state = get_state()
    n = max(1, min(int(n), 200))
    kw = {"task": task} if task else {}
    elements = await state.store.get_recent(n=n, **kw)
    return [e.to_api_dict() for e in elements]


@router.get("/tasks.json")
async def tasks_json() -> list[str]:
    """Distinct task names present in the DB."""
    state = get_state()
    return await state.store.get_tasks()


@router.get("/miners.json")
async def miners_json(limit: int = 200) -> list[dict]:
    """Per-hotkey aggregates for the miners page."""
    from database.dashboard import queries as q

    state = get_state()
    limit = max(1, min(int(limit), 500))
    return await q.miner_stats(state.pool, limit=limit)


@router.get("/challenge.json")
async def challenge_json() -> dict:
    """Active challenge + current frontier size (SPA reads both at once)."""
    state = get_state()
    challenge = state.get_challenge() or {}
    frontier = state.get_frontier() or []
    frontier_size = len(frontier) if isinstance(frontier, list) else 0
    return {**challenge, "frontier_size": frontier_size}


@router.get("/rounds.json")
async def rounds_json(limit: int = 30) -> list[int]:
    """Distinct round IDs seen in experiments (most recent first)."""
    from database.dashboard import queries as q

    state = get_state()
    limit = max(1, min(int(limit), 200))
    return await q.distinct_rounds(state.pool, limit=limit)


def _parse_bool(s: str) -> Optional[bool]:
    if s in ("", None):
        return None
    if s.lower() in ("1", "true", "yes", "success"):
        return True
    if s.lower() in ("0", "false", "no", "failed"):
        return False
    return None


def _parse_int(s: str) -> Optional[int]:
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


@router.get("/browse.json")
async def browse_json(
    request: Request,
    task: str = "",
    round_id: str = "",
    miner_hotkey: str = "",
    success: str = "",
    min_flops: str = "",
    max_flops: str = "",
    q_: str = "",
    page: int = 0,
    page_size: int = 0,
) -> dict:
    """Paginated browse — mirrors the filters on the Jinja /experiments page."""
    from database.dashboard import queries as q

    state = get_state()
    text = q_ or request.query_params.get("q", "")
    filters = q.BrowseFilters(
        task=task,
        round_id=_parse_int(round_id),
        miner_hotkey=miner_hotkey,
        success=_parse_bool(success),
        min_flops=_parse_int(min_flops),
        max_flops=_parse_int(max_flops),
        q=text,
    )
    if page_size <= 0:
        page_size = Config.DASHBOARD_PAGE_SIZE
    page_size = max(1, min(int(page_size), 200))
    result = await q.browse(state.pool, filters, page=max(0, int(page)), page_size=page_size)
    return {
        "rows": [e.to_api_dict() for e in result["items"]],
        "total": int(result["total"]),
        "page": int(result["page"]),
        "page_size": int(result["page_size"]),
    }


@router.get("/experiments/{index}.json")
async def experiment_json(index: int) -> dict:
    """Full experiment detail by index."""
    state = get_state()
    elem = await state.store.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return elem.to_api_dict()


@router.get("/miners/{hotkey}/submissions.json")
async def miner_submissions_json(hotkey: str, limit: int = 100) -> dict:
    """Per-hotkey submission list + agent code history."""
    from database.dashboard import queries as q

    state = get_state()
    limit = max(1, min(int(limit), 500))
    submissions = await q.miner_submissions(state.pool, hotkey, limit=limit)
    agent_history = await q.miner_agent_history(state.pool, hotkey, limit=50)
    if not submissions and not agent_history:
        raise HTTPException(status_code=404, detail="No submissions for this hotkey")
    return {
        "hotkey": hotkey,
        "submissions": [e.to_api_dict() for e in submissions],
        "agent_history": agent_history,
    }


@router.get("/experiments/{index}/lineage.json")
async def experiment_lineage_json(index: int) -> dict:
    """Lineage chain with per-step diffs."""
    state = get_state()
    elem = await state.store.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    diffs = await state.store.get_lineage_diffs(index)
    return {
        "root": elem.to_api_dict(),
        "diffs": diffs,
    }


# ── Training logs (public — same payload shape as cookie-gated Jinja route) ──

@router.get("/logs/{round_id}/{hotkey}/meta.json")
async def logs_meta_json(round_id: int, hotkey: str):
    state = get_state()
    meta = log_helpers.fetch_meta(state.r2, round_id, hotkey)
    if meta is None:
        raise HTTPException(status_code=404, detail="No training_meta.json in R2")
    return JSONResponse(meta)


@router.get("/logs/{round_id}/{hotkey}/stdout.json")
async def logs_stdout_json(round_id: int, hotkey: str, direct: int = 0):
    state = get_state()
    if direct:
        url = log_helpers.presigned_stdout_url(state.r2, round_id, hotkey)
        if not url:
            raise HTTPException(status_code=503, detail="R2 presign unavailable")
        return RedirectResponse(url=url, status_code=302)
    payload = log_helpers.fetch_stdout(state.r2, round_id, hotkey)
    if payload is None:
        raise HTTPException(status_code=404, detail="No stdout.log in R2")
    return JSONResponse(payload)


@router.get("/logs/{round_id}/{hotkey}/architecture.json")
async def logs_architecture_json(round_id: int, hotkey: str, direct: int = 0):
    state = get_state()
    if direct:
        url = log_helpers.presigned_architecture_url(state.r2, round_id, hotkey)
        if not url:
            raise HTTPException(status_code=503, detail="R2 presign unavailable")
        return RedirectResponse(url=url, status_code=302)
    payload = log_helpers.fetch_architecture(state.r2, round_id, hotkey)
    if payload is None:
        raise HTTPException(status_code=404, detail="No architecture.py in R2")
    return JSONResponse(payload)


# ── Agent code bundles (public JSON mirror of the cookie-gated Jinja view) ──

@router.get("/agent_code/{code_hash}.json")
async def agent_code_json(code_hash: str) -> dict:
    """Full agent bundle (files + entry point) plus related history.

    Matches the data the Jinja ``/dashboard/agent_code/{hash}`` view renders,
    but returns JSON so the public SPA can display bundles without cookies.
    """
    from database.dashboard import queries as q

    state = get_state()
    record = await q.agent_bundle_record(state.pool, code_hash)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown code_hash")
    if state.r2 is None:
        raise HTTPException(status_code=503, detail="R2 not configured")
    try:
        bundle = state.r2.download_json(record["r2_key"])
    except Exception:
        bundle = None
    if not bundle:
        raise HTTPException(status_code=404, detail="Bundle missing in R2")
    history = await q.miner_agent_history(state.pool, record["hotkey"], limit=50)
    return {
        "record": record,
        "entry_point": bundle.get("entry_point", record["entry_point"]),
        "files": bundle.get("files") or {},
        "history": [h for h in history if h["code_hash"] != code_hash],
    }
