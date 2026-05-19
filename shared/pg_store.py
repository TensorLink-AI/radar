"""Async Postgres experiment store.

Drop-in async replacement for SQLiteExperimentStore.
All methods are async, backed by asyncpg connection pool.
"""

import json
import logging
import re
import time
from collections import Counter
from typing import Optional

import asyncpg

from shared.database import DataElement
from shared.pg_schema import (
    SCHEMA_TABLE_DDL, SCHEMA_INDEX_DDL, FTS_FUNCTION_DDL, FTS_TRIGGER_DDL,
    MINER_REGISTRY_SCHEMA,
    INSERT_SQL, row_to_element, element_to_params, compute_diff, _finite_or,
)

logger = logging.getLogger(__name__)


# Schema identifiers are embedded in `SET search_path` and CANNOT be
# parameterised (Postgres treats identifiers as keywords, not values), so the
# only protection against SQL injection is a strict allowlist. This regex is
# intentionally narrow: lowercase start, lowercase/digit/underscore body, max
# 63 chars (Postgres' NAMEDATALEN-1 identifier limit).
_SCHEMA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def _validate_schema_name(schema: str) -> str:
    """Return ``schema`` if it matches the allowlist, else raise ValueError.

    This is the single choke point for validating any schema name that will
    be embedded in a SQL identifier. Every code path that splices a schema
    name into SQL must route through this helper.
    """
    if not isinstance(schema, str) or not _SCHEMA_NAME_RE.match(schema):
        raise ValueError(
            f"Invalid Postgres schema name {schema!r}: must match "
            f"{_SCHEMA_NAME_RE.pattern} (lowercase start, "
            "lowercase/digit/underscore only, <=63 chars)"
        )
    return schema


async def ensure_schema_exists(conn: asyncpg.Connection, schema: str) -> None:
    """Create ``schema`` if it doesn't exist.

    Call this ONCE, before any DDL, on a bare connection (NOT one obtained
    from a pool that sets ``search_path`` to the same schema — asking a
    schema to create itself is a bootstrap paradox).

    The schema name is validated via ``_validate_schema_name`` and then
    embedded directly in the DDL. There is no parameter binding because
    Postgres identifiers cannot be parameterised.
    """
    schema = _validate_schema_name(schema)
    await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')


async def _register_json_codecs(conn: asyncpg.Connection) -> None:
    """Register JSON/JSONB codecs so asyncpg returns Python dicts/lists.

    Without this, asyncpg returns JSONB columns as raw strings, which breaks
    ``row_to_element`` (objectives/loss_curve/generated_samples end up as
    ``str`` instead of ``dict``/``list``) and causes ``to_api_dict`` to raise
    ``AttributeError: 'str' object has no attribute 'items'``.
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def create_pg_pool(
    dsn: str,
    schema: Optional[str] = None,
    **kwargs,
) -> asyncpg.Pool:
    """Create an asyncpg pool with JSON/JSONB codecs pre-registered.

    All radar components should use this helper instead of
    ``asyncpg.create_pool`` directly so JSONB columns are decoded to Python
    objects consistently.

    If ``schema`` is provided, every connection borrowed from the pool has
    its ``search_path`` set to ``"<schema>", public`` before it's handed to
    the caller — so every unqualified reference (``SELECT ... FROM
    experiments``) transparently resolves inside that schema, with
    ``public`` kept as a fallback for cross-schema utilities (e.g. built-in
    functions). The schema name is validated at pool creation time and will
    raise ``ValueError`` for anything outside the allowlist, so callers
    fail fast rather than discovering a mis-set search_path mid-transaction.

    Why ``setup`` and not ``init`` for search_path: asyncpg's pool runs
    ``RESET ALL;`` on every connection release to scrub session state. That
    wipes ``search_path`` (and any other SET parameter). ``init`` runs once
    per physical connection — great for client-side things like codecs that
    survive ``RESET ALL`` — but useless for SET parameters the pool will
    reset right after. ``setup`` runs on every acquire, so the SET is
    replayed each time the connection is handed out, which is what we need.
    """
    user_init = kwargs.pop("init", None)
    user_setup = kwargs.pop("setup", None)

    validated_schema = _validate_schema_name(schema) if schema else None

    async def _init(conn: asyncpg.Connection) -> None:
        # Runs once when the connection is first created. Used for things
        # that survive ``RESET ALL`` (client-side codecs).
        await _register_json_codecs(conn)
        if user_init is not None:
            await user_init(conn)

    async def _setup(conn: asyncpg.Connection) -> None:
        # Runs every time a connection is acquired from the pool. Must be
        # used for any SET parameter the pool's RESET ALL would otherwise
        # wipe on release (notably search_path).
        if validated_schema is not None:
            await conn.execute(
                f'SET search_path TO "{validated_schema}", public'
            )
        if user_setup is not None:
            await user_setup(conn)

    return await asyncpg.create_pool(
        dsn, init=_init, setup=_setup, **kwargs,
    )


class PgExperimentStore:
    """Async Postgres experiment store with FTS and diff API."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.provenance = None  # Set after init_schema()

    async def init_schema(self):
        """Create tables, indexes, FTS trigger. Safe to call repeatedly."""
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA_TABLE_DDL)
            await conn.execute(SCHEMA_INDEX_DDL)
            await conn.execute(FTS_FUNCTION_DDL)
            await conn.execute(FTS_TRIGGER_DDL)
            await conn.execute(MINER_REGISTRY_SCHEMA)
        from shared.pg_provenance import PgProvenanceQuery
        self.provenance = PgProvenanceQuery(self.pool)
        await self.provenance.init_schema()

    def _task_clause(self, task: str, prefix: str = "AND", start: int = 1) -> tuple[str, list, int]:
        """Return (SQL fragment, params list, next_param_index)."""
        if task:
            return f" {prefix} task = ${start}", [task], start + 1
        return "", [], start

    # ── Core API ─────────────────────────────────────────

    @property
    async def size(self) -> int:
        return await self.pool.fetchval("SELECT COUNT(*) FROM experiments")

    async def get_size(self) -> int:
        """Non-property async size accessor."""
        return await self.pool.fetchval("SELECT COUNT(*) FROM experiments")

    async def add(self, element: DataElement) -> int:
        next_id = await self.pool.fetchval(
            "SELECT COALESCE(MAX(id), -1) + 1 FROM experiments"
        )
        element.index = next_id
        element.timestamp = time.time()
        params = element_to_params(element, next_id)
        await self.pool.execute(INSERT_SQL, *params)
        return next_id

    async def add_batch(self, elements: list[DataElement]) -> list[int]:
        indices = []
        now = time.time()
        next_id = await self.pool.fetchval(
            "SELECT COALESCE(MAX(id), -1) + 1 FROM experiments"
        )
        for elem in elements:
            elem.index = next_id
            elem.timestamp = now
            params = element_to_params(elem, next_id)
            await self.pool.execute(INSERT_SQL, *params)
            indices.append(next_id)
            next_id += 1
        return indices

    async def get(self, index: int) -> Optional[DataElement]:
        row = await self.pool.fetchrow(
            "SELECT * FROM experiments WHERE id = $1", index
        )
        return row_to_element(row) if row else None

    async def get_successful(self, task: str = "") -> list[DataElement]:
        tc, tp, _ = self._task_clause(task)
        rows = await self.pool.fetch(
            "SELECT * FROM experiments WHERE success = TRUE AND metric IS NOT NULL"
            f"{tc} ORDER BY metric ASC", *tp
        )
        return [row_to_element(r) for r in rows]

    async def get_best(self, n: int = 1, task: str = "") -> list[DataElement]:
        tc, tp, next_p = self._task_clause(task)
        rows = await self.pool.fetch(
            "SELECT * FROM experiments WHERE success = TRUE AND metric IS NOT NULL"
            f"{tc} ORDER BY metric ASC LIMIT ${next_p}", *tp, n
        )
        return [row_to_element(r) for r in rows]

    async def get_recent(self, n: int = 5, task: str = "") -> list[DataElement]:
        tc, tp, next_p = self._task_clause(task, prefix="WHERE")
        sql = f"SELECT * FROM experiments{tc} ORDER BY id DESC LIMIT ${next_p}"
        rows = await self.pool.fetch(sql, *tp, n)
        return [row_to_element(r) for r in rows]

    async def get_failures(self, n: int = 10, task: str = "") -> list[DataElement]:
        tc, tp, next_p = self._task_clause(task)
        rows = await self.pool.fetch(
            "SELECT * FROM experiments WHERE success = FALSE"
            f"{tc} ORDER BY id DESC LIMIT ${next_p}", *tp, n
        )
        return [row_to_element(r) for r in rows]

    async def get_children(self, parent_index: int, task: str = "") -> list[DataElement]:
        tc, tp, _ = self._task_clause(task, start=2)
        rows = await self.pool.fetch(
            f"SELECT * FROM experiments WHERE parent_index = $1{tc}",
            parent_index, *tp,
        )
        return [row_to_element(r) for r in rows]

    async def get_lineage(self, index: int) -> list[DataElement]:
        lineage = []
        row = await self.pool.fetchrow(
            "SELECT * FROM experiments WHERE id = $1", index
        )
        while row is not None:
            elem = row_to_element(row)
            lineage.append(elem)
            if elem.parent is not None:
                row = await self.pool.fetchrow(
                    "SELECT * FROM experiments WHERE id = $1", elem.parent
                )
            else:
                row = None
        return list(reversed(lineage))

    async def search(self, query: str, top_k: int = 10, task: str = "") -> list[DataElement]:
        query = query.strip()
        if not query:
            return []
        tc, tp, next_p = self._task_clause(task)
        rows = await self.pool.fetch(
            "SELECT * FROM experiments "
            "WHERE search_vector @@ plainto_tsquery('english', $1)"
            f"{tc} "
            f"ORDER BY ts_rank(search_vector, plainto_tsquery('english', $1)) DESC "
            f"LIMIT ${next_p}",
            query, *tp, top_k,
        )
        return [row_to_element(r) for r in rows]

    async def search_failures(self, query: str, top_k: int = 10, task: str = "") -> list[DataElement]:
        query = query.strip()
        if not query:
            return []
        tc, tp, next_p = self._task_clause(task)
        rows = await self.pool.fetch(
            "SELECT * FROM experiments "
            "WHERE search_vector @@ plainto_tsquery('english', $1) "
            f"AND success = FALSE{tc} "
            f"ORDER BY ts_rank(search_vector, plainto_tsquery('english', $1)) DESC "
            f"LIMIT ${next_p}",
            query, *tp, top_k,
        )
        return [row_to_element(r) for r in rows]

    async def get_component_stats(
        self, patterns: Optional[dict[str, str]] = None, task: str = "",
    ) -> dict:
        if not patterns:
            return {}
        tc, tp, _ = self._task_clause(task)
        rows = await self.pool.fetch(
            f"SELECT code FROM experiments WHERE success = TRUE{tc}", *tp
        )
        stats: dict[str, Counter] = {k: Counter() for k in patterns}
        for row in rows:
            for category, pattern in patterns.items():
                for match in re.findall(pattern, row["code"], re.IGNORECASE):
                    stats[category][match] += 1
        return {k: dict(v.most_common(10)) for k, v in stats.items()}

    async def get_pareto_elements(self, task: str = "") -> list[DataElement]:
        tc, tp, _ = self._task_clause(task)
        rows = await self.pool.fetch(
            "SELECT * FROM experiments WHERE success = TRUE AND metric IS NOT NULL"
            + tc, *tp
        )
        return [row_to_element(r) for r in rows]

    async def count_in_flops_range(
        self, min_flops: int, max_flops: int, task: str = "",
    ) -> int:
        tc, tp, next_p = self._task_clause(task, start=3)
        row = await self.pool.fetchval(
            "SELECT COUNT(*) FROM experiments "
            "WHERE success = TRUE AND metric IS NOT NULL "
            "AND CAST((objectives->>'flops_equivalent_size') AS BIGINT) "
            f"BETWEEN $1 AND $2{tc}",
            min_flops, max_flops, *tp,
        )
        return row

    async def get_in_flops_range(
        self, min_flops: int, max_flops: int, task: str = "",
    ) -> list[DataElement]:
        tc, tp, _ = self._task_clause(task, start=3)
        rows = await self.pool.fetch(
            "SELECT * FROM experiments "
            "WHERE success = TRUE AND metric IS NOT NULL "
            "AND CAST((objectives->>'flops_equivalent_size') AS BIGINT) "
            f"BETWEEN $1 AND $2{tc}",
            min_flops, max_flops, *tp,
        )
        return [row_to_element(r) for r in rows]

    async def stats(self, task: str = "") -> dict:
        w = " WHERE task = $1" if task else ""
        p = [task] if task else []
        row = await self.pool.fetchrow(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN success = TRUE THEN 1 ELSE 0 END) as successful, "
            f"MAX(generation) as max_gen FROM experiments{w}", *p,
        )
        total = row["total"]
        successful = row["successful"] or 0
        max_gen = row["max_gen"] if row["max_gen"] is not None else 0
        mw = f"{w}{' AND' if task else ' WHERE'} success = TRUE AND metric IS NOT NULL"
        mrow = await self.pool.fetchrow(
            f"SELECT MIN(metric) as best, MAX(metric) as worst, "
            f"AVG(metric) as mean FROM experiments{mw}", *p,
        )
        return {
            "total": total, "successful": successful,
            "failed": total - successful,
            "best_metric": _finite_or(mrow["best"], None),
            "worst_metric": _finite_or(mrow["worst"], None),
            "mean_metric": _finite_or(mrow["mean"], None),
            "max_generation": max_gen,
        }

    # ── Task discovery ──────────────────────────────────

    async def get_tasks(self) -> list[str]:
        rows = await self.pool.fetch(
            "SELECT DISTINCT task FROM experiments WHERE task != '' ORDER BY task"
        )
        return [r[0] for r in rows]

    async def stats_by_task(self) -> dict[str, dict]:
        tasks = await self.get_tasks()
        return {task: await self.stats(task=task) for task in tasks}

    # ── Diff API ────────────────────────────────────────

    async def get_diff(self, index: int) -> Optional[str]:
        row = await self.pool.fetchrow(
            "SELECT * FROM experiments WHERE id = $1", index
        )
        if row is None:
            return None
        elem = row_to_element(row)
        parent = None
        if elem.parent is not None:
            prow = await self.pool.fetchrow(
                "SELECT * FROM experiments WHERE id = $1", elem.parent
            )
            parent = row_to_element(prow) if prow else None
        return compute_diff(parent, elem)

    async def get_diff_between(self, index_a: int, index_b: int) -> Optional[str]:
        a_row = await self.pool.fetchrow(
            "SELECT * FROM experiments WHERE id = $1", index_a
        )
        b_row = await self.pool.fetchrow(
            "SELECT * FROM experiments WHERE id = $1", index_b
        )
        if not a_row or not b_row:
            return None
        return compute_diff(row_to_element(a_row), row_to_element(b_row))

    async def get_lineage_diffs(self, index: int) -> list[dict]:
        lineage = await self.get_lineage(index)
        return [
            {"index": e.index, "name": e.name, "metric": e.metric,
             "motivation": e.motivation,
             "diff": compute_diff(lineage[i - 1] if i > 0 else None, e)}
            for i, e in enumerate(lineage)
        ]

    async def get_family_summary(self, task: str = "") -> list[dict]:
        tc, tp, _ = self._task_clause(task)
        roots = await self.pool.fetch(
            f"SELECT id, name FROM experiments WHERE parent_index IS NULL{tc}",
            *tp,
        )
        result = []
        for root in roots:
            row = await self.pool.fetchrow(
                "WITH RECURSIVE family AS ("
                "  SELECT id, metric, generation FROM experiments WHERE id = $1 "
                "  UNION ALL "
                "  SELECT e.id, e.metric, e.generation "
                "  FROM experiments e JOIN family f ON e.parent_index = f.id"
                ") "
                "SELECT COUNT(*) as cnt, MIN(metric) as best, "
                "MAX(id) as latest, MAX(generation) as max_gen FROM family",
                root["id"],
            )
            result.append({
                "root_index": root["id"], "root_name": root["name"],
                "num_descendants": row["cnt"],
                "best_metric": _finite_or(row["best"], None),
                "latest_index": row["latest"],
                "max_generation": row["max_gen"] or 0,
            })
        return result

    async def close(self):
        await self.pool.close()
