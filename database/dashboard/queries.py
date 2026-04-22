"""Paginated browse + miner aggregate queries for the dashboard.

Uses the shared asyncpg pool directly instead of going through
``PgExperimentStore`` because the browse view needs combined filter
support (task + round + hotkey + size bucket + free-text) and server-side
LIMIT/OFFSET that the existing helpers don't expose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from shared.database import DataElement
from shared.pg_schema import row_to_element

# Simple per-process cache for expensive miner aggregates
_miner_cache: dict[tuple, tuple[float, list[dict]]] = {}
_CACHE_TTL = 60.0


@dataclass
class BrowseFilters:
    task: str = ""
    round_id: Optional[int] = None
    miner_hotkey: str = ""
    success: Optional[bool] = None  # None = all, True/False = filter
    min_flops: Optional[int] = None
    max_flops: Optional[int] = None
    q: str = ""  # free-text search against search_vector

    def is_empty(self) -> bool:
        return (
            not self.task
            and self.round_id is None
            and not self.miner_hotkey
            and self.success is None
            and self.min_flops is None
            and self.max_flops is None
            and not self.q
        )


def _build_where(f: BrowseFilters) -> tuple[str, list]:
    """Return (WHERE ..., params) for the filters. Empty filters → ('', [])."""
    clauses: list[str] = []
    params: list = []

    def _p(value) -> str:
        params.append(value)
        return f"${len(params)}"

    if f.task:
        clauses.append(f"task = {_p(f.task)}")
    if f.round_id is not None:
        clauses.append(f"round_id = {_p(f.round_id)}")
    if f.miner_hotkey:
        clauses.append(f"miner_hotkey = {_p(f.miner_hotkey)}")
    if f.success is not None:
        clauses.append(f"success = {_p(bool(f.success))}")
    if f.min_flops is not None:
        clauses.append(
            "CAST((objectives->>'flops_equivalent_size') AS BIGINT) >= "
            f"{_p(int(f.min_flops))}"
        )
    if f.max_flops is not None:
        clauses.append(
            "CAST((objectives->>'flops_equivalent_size') AS BIGINT) <= "
            f"{_p(int(f.max_flops))}"
        )
    if f.q:
        clauses.append(
            f"search_vector @@ plainto_tsquery('english', {_p(f.q)})"
        )

    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), params


async def browse(
    pool,
    filters: BrowseFilters,
    page: int = 0,
    page_size: int = 50,
) -> dict:
    """List experiments matching filters with pagination.

    Returns ``{items: [DataElement...], total, page, page_size}``.
    """
    where, params = _build_where(filters)
    count_sql = f"SELECT COUNT(*) FROM experiments {where}"
    total = await pool.fetchval(count_sql, *params) or 0

    offset = max(0, page) * max(1, page_size)
    list_sql = (
        f"SELECT * FROM experiments {where} "
        f"ORDER BY id DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
    )
    rows = await pool.fetch(list_sql, *params, page_size, offset)
    return {
        "items": [row_to_element(r) for r in rows],
        "total": int(total),
        "page": page,
        "page_size": page_size,
    }


async def distinct_rounds(pool, limit: int = 50) -> list[int]:
    rows = await pool.fetch(
        "SELECT DISTINCT round_id FROM experiments "
        "WHERE round_id IS NOT NULL ORDER BY round_id DESC LIMIT $1",
        limit,
    )
    return [int(r["round_id"]) for r in rows]


async def distinct_hotkeys(pool, limit: int = 200) -> list[str]:
    rows = await pool.fetch(
        "SELECT DISTINCT miner_hotkey FROM experiments "
        "WHERE miner_hotkey != '' ORDER BY miner_hotkey LIMIT $1",
        limit,
    )
    return [r["miner_hotkey"] for r in rows]


async def miner_stats(pool, limit: int = 100) -> list[dict]:
    """Per-hotkey aggregates: submissions, successes, best metric, latest ts.

    Cached for ``_CACHE_TTL`` seconds per-process to keep the /miners page
    cheap on large tables.
    """
    now = time.time()
    key = ("miner_stats", int(limit))
    cached = _miner_cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    rows = await pool.fetch(
        """
        SELECT miner_hotkey,
               COUNT(*) AS total,
               SUM(CASE WHEN success THEN 1 ELSE 0 END) AS successes,
               MIN(CASE WHEN success THEN metric END) AS best_metric,
               MAX(timestamp) AS last_seen,
               MAX(miner_uid) AS last_uid
        FROM experiments
        WHERE miner_hotkey != ''
        GROUP BY miner_hotkey
        ORDER BY successes DESC NULLS LAST, total DESC
        LIMIT $1
        """,
        limit,
    )
    result = [
        {
            "miner_hotkey": r["miner_hotkey"],
            "miner_uid": int(r["last_uid"] or -1),
            "total": int(r["total"] or 0),
            "successes": int(r["successes"] or 0),
            "best_metric": (
                float(r["best_metric"]) if r["best_metric"] is not None else None
            ),
            "last_seen": float(r["last_seen"] or 0.0),
        }
        for r in rows
    ]
    _miner_cache[key] = (now, result)
    return result


async def miner_submissions(
    pool, hotkey: str, limit: int = 100,
) -> list[DataElement]:
    rows = await pool.fetch(
        "SELECT * FROM experiments WHERE miner_hotkey = $1 "
        "ORDER BY id DESC LIMIT $2",
        hotkey, limit,
    )
    return [row_to_element(r) for r in rows]


async def miner_round_activity(
    pool, rounds: int = 30, max_miners: int = 40,
) -> dict:
    """Heatmap data: unique experiments each miner queried per round.

    Returns ``{miners, rounds, matrix}`` where ``matrix[i][j]`` is the
    unique-experiment count for ``miners[i]`` in ``rounds[j]``. The miner
    axis is sorted by total activity (descending) and capped at
    ``max_miners``; the round axis is the most recent ``rounds`` rounds
    present in ``miner_access_log``.
    """
    rows = await pool.fetch(
        """
        SELECT hotkey, round_id,
               COUNT(DISTINCT ref) AS unique_queried
        FROM (
            SELECT hotkey, round_id,
                   jsonb_array_elements_text(experiment_ids) AS ref
            FROM miner_access_log
            WHERE round_id >= 0
              AND jsonb_typeof(experiment_ids) = 'array'
        ) t
        WHERE round_id IN (
            SELECT DISTINCT round_id FROM miner_access_log
            WHERE round_id >= 0
            ORDER BY round_id DESC LIMIT $1
        )
        GROUP BY hotkey, round_id
        """,
        rounds,
    )
    if not rows:
        return {"miners": [], "rounds": [], "matrix": []}

    round_ids = sorted({int(r["round_id"]) for r in rows}, reverse=True)[:rounds]
    round_set = set(round_ids)
    totals: dict[str, int] = {}
    cells: dict[tuple[str, int], int] = {}
    for r in rows:
        rid = int(r["round_id"])
        if rid not in round_set:
            continue
        hk = r["hotkey"]
        cnt = int(r["unique_queried"] or 0)
        totals[hk] = totals.get(hk, 0) + cnt
        cells[(hk, rid)] = cnt

    miners = sorted(totals, key=lambda h: -totals[h])[:max_miners]
    matrix = [
        [cells.get((hk, rid), 0) for rid in round_ids]
        for hk in miners
    ]
    return {"miners": miners, "rounds": round_ids, "matrix": matrix}


async def top_experiments_activity(
    pool, top_k: int = 20, max_miners: int = 40,
) -> dict:
    """Heatmap data: per-miner query counts against the top-K referenced experiments.

    Returns ``{miners, experiments, matrix}`` where ``experiments`` is a list
    of ``{id, name, references}`` for the ``top_k`` most-queried experiment
    IDs in ``miner_access_log`` and ``matrix[i][j]`` is the number of times
    ``miners[i]`` queried ``experiments[j]``.
    """
    top_rows = await pool.fetch(
        """
        WITH refs AS (
            SELECT (jsonb_array_elements_text(experiment_ids))::int AS exp_id
            FROM miner_access_log
            WHERE jsonb_typeof(experiment_ids) = 'array'
        )
        SELECT exp_id, COUNT(*) AS cnt
        FROM refs
        GROUP BY exp_id
        ORDER BY cnt DESC
        LIMIT $1
        """,
        top_k,
    )
    if not top_rows:
        return {"miners": [], "experiments": [], "matrix": []}

    top_ids = [int(r["exp_id"]) for r in top_rows]
    top_counts = {int(r["exp_id"]): int(r["cnt"]) for r in top_rows}

    name_rows = await pool.fetch(
        "SELECT id, name FROM experiments WHERE id = ANY($1::int[])",
        top_ids,
    )
    names = {int(r["id"]): (r["name"] or "") for r in name_rows}

    matrix_rows = await pool.fetch(
        """
        WITH refs AS (
            SELECT hotkey,
                   (jsonb_array_elements_text(experiment_ids))::int AS exp_id
            FROM miner_access_log
            WHERE jsonb_typeof(experiment_ids) = 'array'
        )
        SELECT hotkey, exp_id, COUNT(*) AS cnt
        FROM refs
        WHERE exp_id = ANY($1::int[])
        GROUP BY hotkey, exp_id
        """,
        top_ids,
    )

    totals: dict[str, int] = {}
    cells: dict[tuple[str, int], int] = {}
    for r in matrix_rows:
        hk = r["hotkey"]
        eid = int(r["exp_id"])
        cnt = int(r["cnt"] or 0)
        totals[hk] = totals.get(hk, 0) + cnt
        cells[(hk, eid)] = cnt

    miners = sorted(totals, key=lambda h: -totals[h])[:max_miners]
    experiments = [
        {"id": eid, "name": names.get(eid, ""), "references": top_counts[eid]}
        for eid in top_ids
    ]
    matrix = [
        [cells.get((hk, eid), 0) for eid in top_ids]
        for hk in miners
    ]
    return {"miners": miners, "experiments": experiments, "matrix": matrix}


async def miner_agent_history(
    pool, hotkey: str, limit: int = 50,
) -> list[dict]:
    """Agent code submission timeline for a hotkey (most recent first)."""
    rows = await pool.fetch(
        """
        SELECT code_hash, entry_point, r2_key, round_submitted, timestamp
        FROM agent_submission_history
        WHERE hotkey = $1
        ORDER BY timestamp DESC
        LIMIT $2
        """,
        hotkey, limit,
    )
    return [
        {
            "code_hash": r["code_hash"],
            "entry_point": r["entry_point"],
            "r2_key": r["r2_key"],
            "round_submitted": int(r["round_submitted"]),
            "timestamp": float(r["timestamp"]),
        }
        for r in rows
    ]


async def agent_bundle_record(pool, code_hash: str) -> Optional[dict]:
    """Most recent history row for ``code_hash`` — resolves hotkey + R2 key."""
    row = await pool.fetchrow(
        """
        SELECT hotkey, code_hash, entry_point, r2_key, round_submitted, timestamp
        FROM agent_submission_history
        WHERE code_hash = $1
        ORDER BY timestamp DESC LIMIT 1
        """,
        code_hash,
    )
    if row is None:
        return None
    return {
        "hotkey": row["hotkey"],
        "code_hash": row["code_hash"],
        "entry_point": row["entry_point"],
        "r2_key": row["r2_key"],
        "round_submitted": int(row["round_submitted"]),
        "timestamp": float(row["timestamp"]),
    }


__all__ = [
    "BrowseFilters",
    "agent_bundle_record",
    "browse",
    "distinct_hotkeys",
    "distinct_rounds",
    "miner_agent_history",
    "miner_round_activity",
    "miner_stats",
    "miner_submissions",
    "top_experiments_activity",
]
