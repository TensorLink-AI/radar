"""JSON endpoints the dashboard UI fetches from the browser.

All routes require a valid dashboard session cookie.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from database.dashboard.app import get_state, require_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard/api", dependencies=[Depends(require_session)])

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
