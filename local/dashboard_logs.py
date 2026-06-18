"""Dashboard helpers for agent-event log + checkpoint browsers.

Kept separate from ``local/dashboard.py`` so the core experiment-centric
read path doesn't grow past the project's per-file budget. These helpers
are stateless (they open their own read-only sqlite cursor through the
caller) so the HTTP handler can call them directly.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Per-event payload preview cap. The raw rows can be up to 4MB each
# (see RADAR_AGENT_EVENT_MAX_BYTES); the list view only needs a glance.
_PREVIEW_CHARS = 160


def _maybe_json(s: Optional[str]) -> Any:
    if s is None:
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s


def _preview(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= _PREVIEW_CHARS else s[:_PREVIEW_CHARS] + "…"


def event_summary(row: sqlite3.Row) -> dict[str, Any]:
    """Compact event row for the list view — no full JSON bodies."""
    req = row["request_json"]
    res = row["response_json"]
    return {
        "id": row["id"],
        "ts": row["ts"],
        "round_id": row["round_id"],
        "miner_id": row["miner_id"],
        "kind": row["kind"],
        "endpoint": row["endpoint"],
        "status": row["status"],
        "latency_ms": row["latency_ms"],
        "error": row["error"],
        "request_bytes": len(req or ""),
        "response_bytes": len(res or ""),
        "request_preview": _preview(req),
        "response_preview": _preview(res),
    }


def event_detail(row: sqlite3.Row) -> dict[str, Any]:
    """Full event including parsed request/response JSON bodies."""
    out = event_summary(row)
    out["request"] = _maybe_json(row["request_json"])
    out["response"] = _maybe_json(row["response_json"])
    return out


def _task_round_ids(conn: sqlite3.Connection, task: str) -> Optional[list[int]]:
    """Round ids that produced an experiment for ``task``.

    ``agent_events`` carries no task column, so a task-scoped service log is
    built by mapping events back to the rounds that ran that task. Returns
    ``None`` when the experiments table is unavailable (pre-engine DB)."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT round_id FROM experiments WHERE task = ?",
            (task,),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    return [int(r[0]) for r in rows if r[0] is not None]


def list_events(
    conn: sqlite3.Connection,
    *,
    round_id: Optional[int] = None,
    miner_id: Optional[str] = None,
    kind: Optional[str] = None,
    endpoint_q: Optional[str] = None,
    task: Optional[str] = None,
    only_errors: bool = False,
    before_id: Optional[int] = None,
    since_id: Optional[int] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    clauses: list[str] = ["1=1"]
    params: list[Any] = []
    if task:
        rids = _task_round_ids(conn, task)
        if not rids:
            # No rounds for this task (or no experiments table) → nothing to
            # scope to. Return empty rather than the whole unscoped log.
            return []
        clauses.append(f"round_id IN ({','.join('?' * len(rids))})")
        params.extend(rids)
    if round_id is not None:
        clauses.append("round_id = ?")
        params.append(int(round_id))
    if miner_id:
        clauses.append("miner_id = ?")
        params.append(miner_id)
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if endpoint_q:
        clauses.append("endpoint LIKE ?")
        params.append(f"%{endpoint_q}%")
    if only_errors:
        clauses.append("(error IS NOT NULL OR (status IS NOT NULL AND status >= 400))")
    if before_id is not None:
        clauses.append("id < ?")
        params.append(int(before_id))
    if since_id is not None:
        clauses.append("id > ?")
        params.append(int(since_id))
    sql = (
        "SELECT id, ts, round_id, miner_id, kind, endpoint, status, "
        " latency_ms, error, request_json, response_json "
        "FROM agent_events WHERE " + " AND ".join(clauses) +
        " ORDER BY id DESC LIMIT ?"
    )
    params.append(int(max(1, min(limit, 500))))
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        logger.debug("agent_events not available: %s", e)
        return []
    return [event_summary(r) for r in rows]


def get_event(conn: sqlite3.Connection, event_id: int) -> Optional[dict[str, Any]]:
    try:
        r = conn.execute(
            "SELECT id, ts, round_id, miner_id, kind, endpoint, status, "
            " latency_ms, error, request_json, response_json "
            "FROM agent_events WHERE id = ?",
            (int(event_id),),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return event_detail(r) if r else None


def event_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Aggregate counts for filter chips + headline stats."""
    try:
        head = conn.execute(
            "SELECT COUNT(*) AS n, MIN(id) AS first_id, MAX(id) AS last_id, "
            " MIN(ts) AS first_ts, MAX(ts) AS last_ts "
            "FROM agent_events"
        ).fetchone()
    except sqlite3.OperationalError:
        return {"total": 0, "by_kind": [], "by_miner": [], "errors": 0}
    by_kind = [
        {
            "kind": r["kind"], "n": r["n"],
            "avg_ms": r["avg_ms"], "max_ms": r["max_ms"],
            "errors": r["errors"],
        }
        for r in conn.execute(
            "SELECT kind, COUNT(*) AS n, "
            " AVG(latency_ms) AS avg_ms, MAX(latency_ms) AS max_ms, "
            " SUM(CASE WHEN error IS NOT NULL OR "
            "          (status IS NOT NULL AND status >= 400) "
            "          THEN 1 ELSE 0 END) AS errors "
            "FROM agent_events GROUP BY kind ORDER BY n DESC"
        )
    ]
    by_miner = [
        {"miner_id": r["miner_id"], "n": r["n"]}
        for r in conn.execute(
            "SELECT miner_id, COUNT(*) AS n FROM agent_events "
            "GROUP BY miner_id ORDER BY n DESC LIMIT 20"
        )
    ]
    errors = conn.execute(
        "SELECT COUNT(*) AS n FROM agent_events "
        "WHERE error IS NOT NULL OR (status IS NOT NULL AND status >= 400)"
    ).fetchone()["n"] or 0
    return {
        "total": head["n"] or 0,
        "first_id": head["first_id"],
        "last_id": head["last_id"],
        "first_ts": head["first_ts"],
        "last_ts": head["last_ts"],
        "by_kind": by_kind,
        "by_miner": by_miner,
        "errors": errors,
    }


# ─── Checkpoints ────────────────────────────────────────────────

def _checkpoint_dir(explicit: str = "") -> Path:
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("RADAR_CHECKPOINT_DIR", "local/checkpoints"))


def _exp_meta(conn: sqlite3.Connection, exp_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    ids = list({int(i) for i in exp_ids})
    if not ids:
        return {}
    # SQLite doesn't take a tuple param for IN() directly via execute; build
    # placeholders. The id set is bounded by what's on disk so it's small.
    qmarks = ",".join("?" * len(ids))
    try:
        rows = conn.execute(
            f"SELECT id, miner_id, round_id, name, task, metric, score, "
            f" success, n_rounds, mode, parent_index, cumulative_compute, "
            f" timestamp "
            f"FROM experiments WHERE id IN ({qmarks})",
            ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        out[int(r["id"])] = {
            "miner_id": r["miner_id"],
            "round_id": r["round_id"],
            "name": r["name"],
            "task": r["task"],
            "metric": r["metric"],
            "score": r["score"],
            "success": bool(r["success"]),
            "n_rounds": r["n_rounds"],
            "mode": r["mode"],
            "parent_index": r["parent_index"],
            "cumulative_compute": r["cumulative_compute"],
            "timestamp": r["timestamp"],
        }
    return out


def list_checkpoints(
    conn: sqlite3.Connection,
    checkpoint_dir: str = "",
) -> dict[str, Any]:
    """List local safetensors checkpoints, joined with experiment metadata.

    Returns ``{dir, items}``. Each item carries ``exp_id``, ``size_bytes``,
    ``mtime`` plus the experiment fields (or ``meta=null`` when the row was
    pruned but the file lingers).
    """
    d = _checkpoint_dir(checkpoint_dir)
    items: list[dict[str, Any]] = []
    if not d.is_dir():
        return {"dir": str(d), "exists": False, "items": items}
    files: list[tuple[int, Path]] = []
    for f in d.glob("*.safetensors"):
        try:
            files.append((int(f.stem), f))
        except ValueError:
            continue
    meta = _exp_meta(conn, (eid for eid, _ in files))
    for exp_id, f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        items.append({
            "exp_id": exp_id,
            "path": str(f),
            "size_bytes": st.st_size,
            "mtime": st.st_mtime,
            "meta": meta.get(exp_id),
        })
    items.sort(key=lambda e: e["mtime"], reverse=True)
    return {"dir": str(d), "exists": True, "items": items}


def checkpoint_signature(
    exp_id: int,
    checkpoint_dir: str = "",
) -> Optional[dict[str, Any]]:
    """Read safetensors header for an experiment's checkpoint.

    Returns ``{exp_id, path, size_bytes, tensors:[{name, shape, n_params}],
    total_params}`` or ``None`` when the file is missing.
    """
    d = _checkpoint_dir(checkpoint_dir)
    f = d / f"{int(exp_id)}.safetensors"
    if not f.is_file():
        return None
    try:
        from local.checkpoints import read_signature
    except Exception as e:  # noqa: BLE001
        logger.warning("checkpoints module unavailable: %s", e)
        return None
    sig = read_signature(f)
    tensors: list[dict[str, Any]] = []
    total = 0
    for name, shape in sig.items():
        n = 1
        for d_ in shape:
            n *= int(d_)
        total += n
        tensors.append({"name": name, "shape": shape, "n_params": n})
    tensors.sort(key=lambda t: t["n_params"], reverse=True)
    try:
        size = f.stat().st_size
    except OSError:
        size = 0
    return {
        "exp_id": int(exp_id),
        "path": str(f),
        "size_bytes": size,
        "tensors": tensors,
        "total_params": total,
    }
