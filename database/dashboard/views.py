"""HTML views: overview, browse, experiment detail, pareto, lineage, miners, logs.

All routes require a valid dashboard session cookie. Pages use Jinja2
templates under ``database/dashboard/templates/``.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from config import Config
from database.dashboard import logs as log_helpers
from database.dashboard import queries as q
from database.dashboard.app import get_state, get_templates, require_session

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dashboard",
    dependencies=[Depends(require_session)],
    tags=["dashboard"],
)


def _html(name: str, request: Request, **ctx) -> HTMLResponse:
    return get_templates().TemplateResponse(request, name, ctx)


# ── Overview ───────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    state = get_state()
    stats = await state.store.stats()
    tasks_stats = await state.store.stats_by_task()
    recent = await state.store.get_recent(n=20)
    tasks = await state.store.get_tasks()
    challenge = state.get_challenge()
    frontier = state.get_frontier()
    return _html(
        "overview.html",
        request,
        stats=stats,
        tasks_stats=tasks_stats,
        recent=[e.to_api_dict() for e in recent],
        tasks=tasks,
        challenge=challenge,
        frontier_size=len(frontier) if isinstance(frontier, list) else 0,
    )


# ── Browse ────────────────────────────────────────────────────

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


@router.get("/experiments", response_class=HTMLResponse)
async def browse_experiments(
    request: Request,
    task: str = "",
    round_id: str = "",
    miner_hotkey: str = "",
    success: str = "",
    min_flops: str = "",
    max_flops: str = "",
    q_: str = "",
    page: int = 0,
):
    # Accept both ?q= and ?q_= (FastAPI rejects ?q= as a reserved kw here)
    text = q_ or request.query_params.get("q", "")

    state = get_state()
    filters = q.BrowseFilters(
        task=task,
        round_id=_parse_int(round_id),
        miner_hotkey=miner_hotkey,
        success=_parse_bool(success),
        min_flops=_parse_int(min_flops),
        max_flops=_parse_int(max_flops),
        q=text,
    )
    page_size = Config.DASHBOARD_PAGE_SIZE
    result = await q.browse(state.pool, filters, page=page, page_size=page_size)
    tasks = await state.store.get_tasks()
    rounds = await q.distinct_rounds(state.pool, limit=30)
    return _html(
        "browse.html",
        request,
        items=[e.to_api_dict() for e in result["items"]],
        total=result["total"],
        page=page,
        page_size=page_size,
        pages=max(1, (result["total"] + page_size - 1) // page_size),
        filters={
            "task": task, "round_id": round_id, "miner_hotkey": miner_hotkey,
            "success": success, "min_flops": min_flops, "max_flops": max_flops,
            "q": text,
        },
        tasks=tasks,
        rounds=rounds,
    )


# ── Experiment detail ─────────────────────────────────────────

@router.get("/experiments/{index}", response_class=HTMLResponse)
async def experiment_detail(request: Request, index: int):
    state = get_state()
    elem = await state.store.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    diff = await state.store.get_diff(index)
    # Short lineage chain (parents only) for the sidebar
    lineage = await state.store.get_lineage(index)
    return _html(
        "experiment.html",
        request,
        elem=elem.to_api_dict(),
        diff=diff or "",
        lineage=[e.to_api_dict() for e in lineage],
    )


@router.get("/experiments/{index}/diff", response_class=HTMLResponse)
async def experiment_diff_partial(request: Request, index: int):
    """HTMX partial — inline diff swap used by the detail view."""
    state = get_state()
    diff = await state.store.get_diff(index)
    if diff is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _html("_partials/diff.html", request, diff=diff, index=index)


@router.get("/experiments/{index}/lineage", response_class=HTMLResponse)
async def experiment_lineage(request: Request, index: int):
    state = get_state()
    elem = await state.store.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    diffs = await state.store.get_lineage_diffs(index)
    return _html(
        "lineage.html",
        request,
        root=elem.to_api_dict(),
        diffs=diffs,
    )


# ── Pareto ───────────────────────────────────────────────────

@router.get("/pareto", response_class=HTMLResponse)
async def pareto_view(request: Request, task: str = ""):
    state = get_state()
    tasks = await state.store.get_tasks()
    return _html("pareto.html", request, task=task, tasks=tasks)


# ── Miners ───────────────────────────────────────────────────

@router.get("/miners", response_class=HTMLResponse)
async def miners_list(request: Request):
    state = get_state()
    stats = await q.miner_stats(state.pool, limit=200)
    return _html("miners.html", request, miners=stats)


@router.get("/miners/{hotkey}", response_class=HTMLResponse)
async def miner_detail(request: Request, hotkey: str):
    state = get_state()
    submissions = await q.miner_submissions(state.pool, hotkey, limit=200)
    if not submissions:
        raise HTTPException(status_code=404, detail="No submissions for this hotkey")
    return _html(
        "miner_detail.html",
        request,
        hotkey=hotkey,
        submissions=[e.to_api_dict() for e in submissions],
    )


# ── Provenance activity heatmaps ──────────────────────────────

@router.get("/provenance", response_class=HTMLResponse)
async def provenance_view(request: Request):
    return _html("provenance.html", request)


# ── Training logs (R2) ────────────────────────────────────────

@router.get("/logs/{round_id}/{hotkey}", response_class=HTMLResponse)
async def logs_view(request: Request, round_id: int, hotkey: str):
    state = get_state()
    meta = log_helpers.fetch_meta(state.r2, round_id, hotkey)
    stdout = log_helpers.fetch_stdout(state.r2, round_id, hotkey)
    architecture = log_helpers.fetch_architecture(state.r2, round_id, hotkey)
    return _html(
        "logs.html",
        request,
        round_id=round_id,
        hotkey=hotkey,
        meta=meta,
        stdout=stdout,
        architecture=architecture,
    )


@router.get("/logs/{round_id}/{hotkey}/meta")
async def logs_meta(round_id: int, hotkey: str):
    state = get_state()
    meta = log_helpers.fetch_meta(state.r2, round_id, hotkey)
    if meta is None:
        raise HTTPException(status_code=404, detail="No training_meta.json in R2")
    return JSONResponse(meta)


@router.get("/logs/{round_id}/{hotkey}/stdout")
async def logs_stdout(round_id: int, hotkey: str, direct: int = 0):
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


@router.get("/logs/{round_id}/{hotkey}/architecture")
async def logs_architecture(round_id: int, hotkey: str, direct: int = 0):
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
