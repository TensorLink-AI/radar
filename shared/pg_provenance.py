"""Async Postgres provenance queries.

Same public API as ProvenanceQuery but all methods are async,
backed by asyncpg connection pool.
"""

import logging
import math
import time
from typing import Optional

import asyncpg

from shared.dedup import code_similarity
from shared.pg_schema import PROVENANCE_SCHEMA
from shared.provenance import compute_similarity

logger = logging.getLogger(__name__)


def _safe_float(v):
    """Replace inf/nan floats with None for JSON compliance."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    return v


class PgProvenanceQuery:
    """Query provenance from raw observation tables (async Postgres)."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute(PROVENANCE_SCHEMA)

    # ── Record facts ─────────────────────────────────────

    async def record_round_context(
        self, round_id: int, experiment_id: int,
        context_type: str = "frontier", timestamp: Optional[float] = None,
    ):
        """Record that an experiment was shown in a round's challenge."""
        await self.pool.execute(
            "INSERT INTO round_context (round_id, experiment_id, context_type, timestamp) "
            "VALUES ($1, $2, $3, $4)",
            round_id, experiment_id, context_type, timestamp or time.time(),
        )

    async def record_components(self, experiment_id: int, components: list[str]):
        """Store detected components for an experiment."""
        for comp in components:
            try:
                await self.pool.execute(
                    "INSERT INTO code_components (experiment_id, component) "
                    "VALUES ($1, $2)",
                    experiment_id, comp,
                )
            except asyncpg.UniqueViolationError:
                continue

    # ── Query: influences ────────────────────────────────

    async def get_influences(self, experiment_id: int) -> list[dict]:
        """All evidence of what influenced this experiment."""
        results: list[dict] = []
        exp = await self._get_experiment(experiment_id)
        if exp is None:
            return results

        miner_hotkey = exp["miner_hotkey"]
        round_id = exp["round_id"]

        # 1. Access log: what experiments did the miner's agent query?
        if miner_hotkey and round_id is not None:
            accessed = await self._get_accessed_ids(miner_hotkey, round_id)
            for aid in accessed:
                results.append({
                    "source_id": aid,
                    "evidence_type": "accessed",
                    "detail": {"round_id": round_id},
                })

        # 2. Frontier context: what was shown in the challenge?
        if round_id is not None:
            frontier_ids = await self._get_round_context_ids(round_id)
            for fid in frontier_ids:
                results.append({
                    "source_id": fid,
                    "evidence_type": "frontier",
                    "detail": {"round_id": round_id},
                })

        # 3. Shared components
        my_comps = await self.get_experiment_components(experiment_id)
        if my_comps:
            for comp in my_comps:
                other_ids = await self.get_component_experiments(comp)
                for oid in other_ids:
                    if oid != experiment_id and oid < experiment_id:
                        results.append({
                            "source_id": oid,
                            "evidence_type": "shared_component",
                            "detail": {"component": comp},
                        })

        return results

    # ── Query: impact ────────────────────────────────────

    async def get_impact(self, experiment_id: int) -> list[dict]:
        """All evidence of what this experiment influenced."""
        results: list[dict] = []

        # Which later experiments' agents accessed this one?
        try:
            rows = await self.pool.fetch(
                "SELECT DISTINCT a.hotkey, a.round_id, e.id as exp_id "
                "FROM miner_access_log a, "
                "jsonb_array_elements(a.experiment_ids) j "
                "JOIN experiments e ON e.miner_hotkey = a.hotkey "
                "  AND e.round_id = a.round_id "
                "WHERE j::text::integer = $1 AND e.id > $1",
                experiment_id,
            )
            for row in rows:
                results.append({
                    "target_id": row["exp_id"],
                    "evidence_type": "accessed_by",
                    "detail": {"round_id": row["round_id"]},
                })
        except asyncpg.PostgresError:
            pass  # miner_access_log not ready

        # Which later experiments were shown this in their frontier?
        ctx_rows = await self.pool.fetch(
            "SELECT round_id FROM round_context WHERE experiment_id = $1",
            experiment_id,
        )
        for row in ctx_rows:
            later = await self.pool.fetch(
                "SELECT id FROM experiments WHERE round_id = $1 AND id > $2",
                row["round_id"], experiment_id,
            )
            for lr in later:
                results.append({
                    "target_id": lr["id"],
                    "evidence_type": "frontier_for",
                    "detail": {"round_id": row["round_id"]},
                })

        return results

    # ── Query: similarity ────────────────────────────────

    async def get_similar(
        self, experiment_id: int,
        pool: Optional[list[int]] = None, top_k: int = 10,
    ) -> list[dict]:
        """Code similarity against a pool (or recent experiments)."""
        exp = await self._get_experiment(experiment_id)
        if exp is None or not exp["code"]:
            return []

        if pool is not None:
            rows = await self.pool.fetch(
                "SELECT id, code FROM experiments WHERE id = ANY($1)",
                pool,
            )
        else:
            rows = await self.pool.fetch(
                "SELECT id, code FROM experiments WHERE id != $1 "
                "ORDER BY id DESC LIMIT 50",
                experiment_id,
            )

        results = []
        for row in rows:
            if row["id"] == experiment_id or not row["code"]:
                continue
            sim = compute_similarity(exp["code"], row["code"])
            results.append({"target_id": row["id"], **sim})

        results.sort(key=lambda x: -x["jaccard"])
        return results[:top_k]

    # ── Query: components ────────────────────────────────

    async def get_component_experiments(self, component: str) -> list[int]:
        rows = await self.pool.fetch(
            "SELECT experiment_id FROM code_components WHERE component = $1",
            component,
        )
        return [r[0] for r in rows]

    async def get_experiment_components(self, experiment_id: int) -> list[str]:
        rows = await self.pool.fetch(
            "SELECT component FROM code_components WHERE experiment_id = $1",
            experiment_id,
        )
        return [r[0] for r in rows]

    async def get_component_stats(self) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT c.component, COUNT(*) as count, "
            "AVG(e.metric) as avg_metric, MIN(e.metric) as best_metric "
            "FROM code_components c "
            "JOIN experiments e ON c.experiment_id = e.id "
            "WHERE e.success = TRUE AND e.metric IS NOT NULL "
            "AND e.metric != 'NaN'::double precision "
            "GROUP BY c.component ORDER BY count DESC",
        )
        return [
            {"component": r["component"], "count": r["count"],
             "avg_metric": _safe_float(r["avg_metric"]),
             "best_metric": _safe_float(r["best_metric"])}
            for r in rows
        ]

    # ── Query: dead ends ─────────────────────────────────

    async def get_dead_ends(self, task: str = "") -> list[int]:
        tc = " AND task = $1" if task else ""
        tp = [task] if task else []
        rows = await self.pool.fetch(
            "SELECT id, code FROM experiments WHERE success = TRUE "
            f"AND metric IS NOT NULL{tc} ORDER BY id", *tp,
        )
        dead = []
        for i, row in enumerate(rows):
            if not row["code"]:
                continue
            has_successor = False
            for later in rows[i + 1:]:
                if later["code"] and code_similarity(row["code"], later["code"]) > 0.3:
                    has_successor = True
                    break
            if not has_successor:
                dead.append(row["id"])
        return dead

    # ── Query: experiment graph ──────────────────────────

    async def get_experiment_graph(
        self, experiment_id: int, depth: int = 3,
    ) -> dict:
        influences = await self.get_influences(experiment_id)
        impact = await self.get_impact(experiment_id)
        components = await self.get_experiment_components(experiment_id)
        return {
            "experiment_id": experiment_id,
            "influences": influences,
            "impact": impact,
            "components": components,
        }

    # ── Export ────────────────────────────────────────────

    async def export(self, task: str = "") -> dict:
        tc = " AND task = $1" if task else ""
        tp = [task] if task else []
        experiments = await self.pool.fetch(
            "SELECT id, name, miner_uid, metric, success FROM experiments "
            f"WHERE TRUE{tc}", *tp,
        )
        components = await self.pool.fetch("SELECT * FROM code_components")
        context = await self.pool.fetch("SELECT * FROM round_context")
        access_count = await self.pool.fetchval(
            "SELECT COUNT(*) FROM miner_access_log"
        )
        return {
            "experiments": [
                {"id": r["id"], "name": r["name"], "miner_uid": r["miner_uid"],
                 "metric": r["metric"], "success": bool(r["success"])}
                for r in experiments
            ],
            "components": [
                {"experiment_id": r["experiment_id"], "component": r["component"]}
                for r in components
            ],
            "round_context": [
                {"round_id": r["round_id"], "experiment_id": r["experiment_id"],
                 "context_type": r["context_type"]}
                for r in context
            ],
            "access_log_count": access_count,
            "component_stats": await self.get_component_stats(),
        }

    # ── Internal helpers ─────────────────────────────────

    async def _get_experiment(self, experiment_id: int) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "SELECT id, code, miner_hotkey, round_id FROM experiments WHERE id = $1",
            experiment_id,
        )
        if row is None:
            return None
        return dict(row)

    async def _get_accessed_ids(self, hotkey: str, round_id: int) -> set[int]:
        import json as _json
        try:
            rows = await self.pool.fetch(
                "SELECT experiment_ids FROM miner_access_log "
                "WHERE hotkey = $1 AND round_id = $2",
                hotkey, round_id,
            )
        except asyncpg.PostgresError:
            return set()
        ids: set[int] = set()
        for row in rows:
            exp_ids = row["experiment_ids"]
            # Pools without a JSONB codec return this as a string.
            if isinstance(exp_ids, str):
                try:
                    exp_ids = _json.loads(exp_ids)
                except (ValueError, TypeError):
                    exp_ids = None
            if isinstance(exp_ids, list):
                ids.update(exp_ids)
        return ids

    async def _get_round_context_ids(self, round_id: int) -> list[int]:
        rows = await self.pool.fetch(
            "SELECT DISTINCT experiment_id FROM round_context WHERE round_id = $1",
            round_id,
        )
        return [r[0] for r in rows]
