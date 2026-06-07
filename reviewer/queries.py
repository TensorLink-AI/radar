"""Read-only SQL queries the reviewer issues against ``radar_local.db``.

All callers receive plain dicts (or lists thereof) so the reviewer
prompt template doesn't have to think about ``sqlite3.Row``. Free-text
fields are wrapped/clipped via ``sanitize`` before they reach the
prompt.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from reviewer.sanitize import sanitize_experiment_text


def round_window(conn: sqlite3.Connection, rounds_back: int) -> tuple[int, int]:
    row = conn.execute(
        "SELECT MIN(round_id) AS lo, MAX(round_id) AS hi FROM experiments"
    ).fetchone()
    if row is None or row["hi"] is None:
        return 0, 0
    hi = int(row["hi"])
    lo = max(int(row["lo"] or 0), hi - rounds_back + 1)
    return lo, hi


def aggregate_summary(conn: sqlite3.Connection, lo: int, hi: int) -> dict:
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM experiments WHERE round_id BETWEEN ? AND ?",
        (lo, hi),
    ).fetchone()["n"]
    success = conn.execute(
        "SELECT COUNT(*) AS n FROM experiments "
        "WHERE round_id BETWEEN ? AND ? AND success = 1",
        (lo, hi),
    ).fetchone()["n"]
    top_miners = [
        dict(r) for r in conn.execute(
            "SELECT miner_id, COUNT(*) AS n, MIN(metric) AS best "
            "FROM experiments WHERE round_id BETWEEN ? AND ? AND success = 1 "
            "GROUP BY miner_id ORDER BY n DESC LIMIT 20",
            (lo, hi),
        )
    ]
    event_kinds = [
        dict(r) for r in conn.execute(
            "SELECT kind, COUNT(*) AS n, "
            "SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors "
            "FROM agent_events WHERE round_id BETWEEN ? AND ? "
            "GROUP BY kind ORDER BY n DESC",
            (lo, hi),
        )
    ]
    tasks = [
        dict(r) for r in conn.execute(
            "SELECT task, COUNT(*) AS n FROM experiments "
            "WHERE round_id BETWEEN ? AND ? GROUP BY task",
            (lo, hi),
        )
    ]
    return {
        "round_lo": lo, "round_hi": hi,
        "experiments_total": total,
        "experiments_success": success,
        "experiments_failure": total - success,
        "tasks": tasks, "top_miners": top_miners,
        "event_kinds": event_kinds,
    }


def self_scorecard(conn: sqlite3.Connection) -> list[dict]:
    """Performance of every ``rev_*`` prompt variant the reviewer has
    authored. Read first each session so retirement precedes new
    speculation."""
    return [dict(r) for r in conn.execute(
        "SELECT prompt_id, AVG(score) AS avg_score, AVG(metric) AS avg_metric, "
        "       COUNT(*) AS n, MAX(round_id) AS last_seen "
        "FROM experiments WHERE prompt_id LIKE 'rev_%' AND success = 1 "
        "GROUP BY prompt_id ORDER BY avg_score DESC"
    )]


def recent_failures(conn: sqlite3.Connection, lo: int, hi: int,
                    limit: int = 40) -> list[dict]:
    rows = conn.execute(
        "SELECT id, round_id, miner_id, task, motivation, reasoning, analysis "
        "FROM experiments WHERE success = 0 "
        "AND round_id BETWEEN ? AND ? ORDER BY round_id DESC LIMIT ?",
        (lo, hi, limit),
    ).fetchall()
    return [sanitize_experiment_text(dict(r)) for r in rows]


def new_experiments_since(conn: sqlite3.Connection,
                          round_id: int) -> int:
    return int(conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE round_id > ?",
        (round_id,),
    ).fetchone()[0])


def new_experiments_since_or_none(conn: sqlite3.Connection,
                                  round_id: Optional[int]) -> Optional[int]:
    if round_id is None:
        return None
    return new_experiments_since(conn, round_id)
