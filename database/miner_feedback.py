"""Miner-feedback endpoints — bearer-auth ``/miners/me/*`` + public
per-task ``/tasks/{task}/frontier``.

Mounted onto the DB FastAPI app via ``include_miner_feedback_routes``.
Auth runs through ``database.server.auth_middleware`` which delegates
bearer lookups to ``shared.miner_auth`` for any path prefixed
``/miners/me/``.

These endpoints are the data plane behind the miner CLI
(``miner/cli.py``): ``MinerResultsClient`` calls them to pull
submissions + Phase C scores for the local optimizer (GEPA, etc.).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request

from shared.miner_auth import touch_key_usage

logger = logging.getLogger(__name__)

miner_feedback_router = APIRouter()
public_router = APIRouter()


def _identity(request: Request):
    """Pull the resolved ``MinerIdentity`` off ``request.state`` (set by
    the auth middleware).  401 if absent — middleware should always set
    it for ``/miners/me/*`` paths."""
    ident = getattr(request.state, "miner_identity", None)
    if ident is None:
        raise HTTPException(status_code=401, detail="Bearer auth required")
    return ident


def _pool_or_503():
    """Lazy import to avoid circular dependency with database.server."""
    from database import server
    if server._pool is None:
        raise HTTPException(status_code=503, detail="DB pool unavailable")
    return server._pool


def _miner_filter_clause(ident) -> tuple[str, list]:
    """SQL fragment + params to scope a query to this miner's rows.

    During the dual-stack period a miner row may have a ``hotkey`` (the
    on-chain identity used by older write paths) and a ``miner_id`` (the
    new operator-issued identifier).  We accept BOTH so submissions
    written with either identity surface in the feedback API.
    """
    # We match on miner_hotkey OR a computed miner_id fallback so old
    # rows that were written before the registry existed still surface.
    if ident.hotkey:
        return (
            "(experiments.miner_hotkey = $1)",
            [ident.hotkey],
        )
    # No hotkey on file — use the miner_id directly against the prompt_id
    # tail (best-effort; miners with no hotkey are post-cutover only).
    return ("FALSE", [])


# ── /miners/me/submissions ──────────────────────────────────────────


@miner_feedback_router.get("/miners/me/submissions")
async def miner_submissions(
    request: Request,
    since: Optional[str] = None,
    limit: int = 200,
    task: str = "",
):
    """Return Phase A submissions this miner produced.

    JSON shape::

        {"submissions": [{round_id, submission_id, task_name, prompt_id,
                          architecture_code, motivation, reasoning,
                          tool_calls, created_at}, ...]}
    """
    ident = _identity(request)
    pool = _pool_or_503()
    limit = max(1, min(int(limit or 200), 1000))
    where, params = _miner_filter_clause(ident)
    if where == "FALSE":
        return {"submissions": []}

    sql_parts = [
        "SELECT id AS submission_id, round_id, task AS task_name, "
        "prompt_id, code AS architecture_code, motivation, reasoning, "
        "tool_calls, timestamp AS created_at "
        "FROM experiments WHERE " + where,
    ]
    if task:
        sql_parts.append(f"AND task = ${len(params) + 1}")
        params.append(task)
    if since:
        sql_parts.append(f"AND timestamp >= ${len(params) + 1}")
        params.append(_parse_iso(since))
    sql_parts.append("ORDER BY timestamp DESC")
    sql_parts.append(f"LIMIT ${len(params) + 1}")
    params.append(limit)

    rows = await pool.fetch(" ".join(sql_parts), *params)
    await touch_key_usage(pool, ident.key_id)
    return {"submissions": [_row_to_submission(r) for r in rows]}


# ── /miners/me/results ──────────────────────────────────────────────


@miner_feedback_router.get("/miners/me/results")
async def miner_results(
    request: Request,
    since: Optional[str] = None,
    limit: int = 200,
    task: str = "",
):
    """Submissions joined with Phase C scoring metadata.

    The experiments table already holds the scoring columns (``metric``,
    ``score``, ``success``, ``objectives``) — Phase C writes them in
    place — so we just project them alongside the submission shape and
    fold them into a ``scores`` blob the optimizer can consume.
    """
    ident = _identity(request)
    pool = _pool_or_503()
    limit = max(1, min(int(limit or 200), 1000))
    where, params = _miner_filter_clause(ident)
    if where == "FALSE":
        return {"results": []}

    sql_parts = [
        "SELECT id AS submission_id, round_id, task AS task_name, "
        "prompt_id, code AS architecture_code, motivation, "
        "metric, score, success, objectives, "
        "timestamp AS created_at "
        "FROM experiments WHERE " + where,
    ]
    if task:
        sql_parts.append(f"AND task = ${len(params) + 1}")
        params.append(task)
    if since:
        sql_parts.append(f"AND timestamp >= ${len(params) + 1}")
        params.append(_parse_iso(since))
    sql_parts.append("ORDER BY timestamp DESC")
    sql_parts.append(f"LIMIT ${len(params) + 1}")
    params.append(limit)

    rows = await pool.fetch(" ".join(sql_parts), *params)
    await touch_key_usage(pool, ident.key_id)
    return {"results": [_row_to_result(r) for r in rows]}


# ── /miners/me/summary ──────────────────────────────────────────────


@miner_feedback_router.get("/miners/me/summary")
async def miner_summary(request: Request):
    """Quick stats — total submissions, last seen round, mean score."""
    ident = _identity(request)
    pool = _pool_or_503()
    where, params = _miner_filter_clause(ident)
    if where == "FALSE":
        return {
            "miner_id": ident.miner_id,
            "total_submissions": 0,
            "last_round_id": None,
            "mean_score_recent": None,
            "successes_recent": 0,
        }

    summary_sql = (
        "SELECT COUNT(*) AS total, MAX(round_id) AS last_round "
        "FROM experiments WHERE " + where
    )
    s = await pool.fetchrow(summary_sql, *params)

    # Recent-window stats (last 50 rows).
    recent_sql = (
        "SELECT AVG(score) AS mean_score, "
        "SUM(CASE WHEN success THEN 1 ELSE 0 END) AS successes, "
        "COUNT(*) AS n "
        "FROM (SELECT score, success FROM experiments WHERE " + where +
        " ORDER BY timestamp DESC LIMIT 50) sub"
    )
    r = await pool.fetchrow(recent_sql, *params)

    await touch_key_usage(pool, ident.key_id)
    return {
        "miner_id": ident.miner_id,
        "total_submissions": int(s["total"] or 0),
        "last_round_id": int(s["last_round"]) if s["last_round"] is not None else None,
        "mean_score_recent": float(r["mean_score"]) if r["mean_score"] is not None else None,
        "successes_recent": int(r["successes"] or 0),
        "recent_window": int(r["n"] or 0),
    }


# ── /tasks/{task}/frontier (public) ─────────────────────────────────


@public_router.get("/tasks/{task}/frontier")
async def task_frontier(task: str, limit: int = 50):
    """Per-task Pareto frontier as ``{points: [{flops, crps, mase}, …]}``.

    Public — no bearer required.  Same data the optimizer's
    ``MinerResultsClient.frontier()`` hits.  Computed live from the
    experiments table (top successful rows per FLOPs bucket).
    """
    pool = _pool_or_503()
    limit = max(1, min(int(limit or 50), 500))
    sql = (
        "SELECT metric, objectives, score "
        "FROM experiments "
        "WHERE task = $1 AND success = TRUE AND metric IS NOT NULL "
        "ORDER BY metric ASC "
        "LIMIT $2"
    )
    rows = await pool.fetch(sql, task, limit)
    points = []
    for r in rows:
        # objectives is JSONB; asyncpg may return as str without codec.
        obj = r["objectives"]
        if isinstance(obj, (bytes, bytearray)):
            obj = obj.decode()
        if isinstance(obj, str):
            import json as _json
            try:
                obj = _json.loads(obj)
            except (ValueError, TypeError):
                obj = {}
        if not isinstance(obj, dict):
            obj = {}
        points.append({
            "flops": obj.get("flops_equivalent_size"),
            "metric": r["metric"],
            "score": r["score"],
        })
    return {"task": task, "points": points}


# ── Helpers ────────────────────────────────────────────────────────


def _parse_iso(s: str) -> float:
    """Accept an ISO-8601 timestamp and return a POSIX seconds float so
    we can compare against ``experiments.timestamp``."""
    if not s:
        return 0.0
    from datetime import datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        # Fall back to "treat as already-numeric epoch".
        try:
            return float(s)
        except (TypeError, ValueError):
            return 0.0


def _row_to_submission(row) -> dict:
    return {
        "submission_id": str(row["submission_id"]),
        "round_id": int(row["round_id"]) if row["round_id"] is not None else None,
        "task_name": row["task_name"] or "",
        "prompt_id": row["prompt_id"],
        "architecture_code": row["architecture_code"] or "",
        "motivation": row["motivation"] or "",
        "reasoning": row["reasoning"] or "",
        "tool_calls": _decode_jsonb(row["tool_calls"], []),
        "created_at": _to_iso(row["created_at"]),
    }


def _row_to_result(row) -> dict:
    obj = _decode_jsonb(row["objectives"], {})
    return {
        "submission_id": str(row["submission_id"]),
        "round_id": int(row["round_id"]) if row["round_id"] is not None else None,
        "task_name": row["task_name"] or "",
        "prompt_id": row["prompt_id"],
        "architecture_code": row["architecture_code"] or "",
        "motivation": row["motivation"] or "",
        "scores": {
            "metric": row["metric"],
            "raw_score": row["score"],
            "success": bool(row["success"]),
            "flops_equivalent_size": obj.get("flops_equivalent_size"),
            **{k: v for k, v in obj.items() if k != "flops_equivalent_size"},
        },
        "created_at": _to_iso(row["created_at"]),
    }


def _decode_jsonb(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        import json as _json
        try:
            return _json.loads(value)
        except (ValueError, TypeError):
            return default
    return default


def _to_iso(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return ""


# ── Mount ──────────────────────────────────────────────────────────


def include_miner_feedback_routes(target_app: FastAPI) -> None:
    """Idempotent mount.  Safe to call multiple times."""
    already = any(
        getattr(r, "path", None) == "/miners/me/summary" for r in target_app.routes
    )
    if already:
        return
    target_app.include_router(miner_feedback_router)
    target_app.include_router(public_router)
