"""Experiment-engine queries for the dashboard: lab reports, noise
floors, round-kind classification and special-round counts.

Split out of ``dashboard.py`` (read-only SQLite helpers, no HTTP) the
same way ``dashboard_logs.py`` carries the events/checkpoints tabs.
All functions tolerate pre-experiment-engine DBs: a missing
``lab_reports`` table or absent objectives stamps degrade to empty
results, never a 500.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Optional

logger = logging.getLogger(__name__)

# objectives stamp → display kind, in precedence order (mirrors
# local/lab_reports.py::_round_kind).
_STAMPS = (
    ("replicate_of", "replicate"),
    ("ablation_of", "ablation"),
    ("recipe_only_of", "recipe_only"),
    ("transfer_of", "transfer"),
)


def round_kind(mode: str, objectives: dict) -> str:
    """Display label for the validator-owned round type of a row."""
    for key, label in _STAMPS:
        if (objectives or {}).get(key) is not None:
            return label
    if mode == "continue":
        kind = (objectives or {}).get("continuation_kind")
        return f"continuation:{kind}" if kind else "continuation"
    if mode == "replicate":
        return "replicate"
    return "new"


def list_lab_reports(conn: sqlite3.Connection, n: int = 50,
                     task: Optional[str] = None) -> list[dict[str, Any]]:
    try:
        if task:
            rows = conn.execute(
                "SELECT * FROM lab_reports WHERE task=? "
                "ORDER BY id DESC LIMIT ?", (task, max(1, n)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM lab_reports ORDER BY id DESC LIMIT ?",
                (max(1, n),),
            ).fetchall()
    except sqlite3.OperationalError:
        return []  # pre-engine DB without the table
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            report = json.loads(r["report_json"] or "{}")
        except json.JSONDecodeError:
            continue
        report["_created_at"] = r["created_at"]
        out.append(report)
    return out


def get_lab_report(conn: sqlite3.Connection,
                   exp_id: int) -> Optional[dict[str, Any]]:
    try:
        r = conn.execute(
            "SELECT report_json FROM lab_reports WHERE experiment_id=?",
            (int(exp_id),),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if r is None:
        return None
    try:
        return json.loads(r["report_json"] or "{}")
    except json.JSONDecodeError:
        return None


def _slim_experiments(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Minimal experiment dicts for ``local.noise`` (id, metric, mode,
    success, task, objectives)."""
    try:
        rows = conn.execute(
            "SELECT id, metric, mode, success, task, objectives_json "
            "FROM experiments"
        ).fetchall()
    except sqlite3.OperationalError:
        # Pre-continuation DBs lack the ``mode`` column (the validator's
        # additive migration never ran on a read-only snapshot); retry
        # without it so the row loop's default kicks in.
        try:
            rows = conn.execute(
                "SELECT id, metric, success, task, objectives_json "
                "FROM experiments"
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # no experiments table at all
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            objs = json.loads(r["objectives_json"] or "{}")
        except json.JSONDecodeError:
            objs = {}
        out.append({
            "id": r["id"], "metric": r["metric"],
            "mode": r["mode"] if "mode" in r.keys() else "new",
            "success": bool(r["success"]), "task": r["task"],
            "objectives": objs,
        })
    return out


def noise_floors(conn: sqlite3.Connection) -> dict[str, Any]:
    """Replicate-derived noise floors: overall + per task."""
    from local.noise import noise_floor

    exps = _slim_experiments(conn)
    by_task: dict[str, Any] = {}
    for task in sorted({e["task"] for e in exps if e["task"]}):
        floor = noise_floor([e for e in exps if e["task"] == task])
        if floor.get("n_pairs"):
            by_task[task] = floor
    return {"overall": noise_floor(exps), "by_task": by_task}


def special_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """How many experiments each special round kind produced."""
    counts = {label: 0 for _, label in _STAMPS}
    for e in _slim_experiments(conn):
        kind = round_kind(e["mode"], e["objectives"])
        if kind in counts:
            counts[kind] += 1
    return {f"n_{k}": v for k, v in counts.items()}
