"""Provenance: raw observation tables + on-demand query layer.

No pre-computed edges. No confidence scores. No edge types.
Store raw facts, compute relationships on query.

Tables:
- miner_access_log  (managed by AccessLogger)
- round_context     (which frontier experiments were shown)
- code_components   (regex-detected ML components per experiment)

Query class:
- ProvenanceQuery   (all read-only, compute on demand)
"""

import difflib
import json
import logging
import re
import sqlite3
from typing import Optional

from shared.dedup import code_similarity

logger = logging.getLogger(__name__)

# ── Schema for observation tables ────────────────────

PROVENANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS round_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    context_type TEXT NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rc_round ON round_context(round_id);
CREATE INDEX IF NOT EXISTS idx_rc_experiment ON round_context(experiment_id);

CREATE TABLE IF NOT EXISTS code_components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    component TEXT NOT NULL,
    UNIQUE(experiment_id, component)
);
CREATE INDEX IF NOT EXISTS idx_cc_experiment ON code_components(experiment_id);
CREATE INDEX IF NOT EXISTS idx_cc_component ON code_components(component);
"""

# Default ML component patterns (deterministic regex)
DEFAULT_COMPONENT_PATTERNS: dict[str, str] = {
    "RMSNorm": r"\bRMSNorm\b",
    "LayerNorm": r"\bLayerNorm\b",
    "GELU": r"\bGELU\b",
    "SwiGLU": r"\bSwiGLU\b",
    "RotaryEmbedding": r"\b(?:rotary|RoPE|RotaryEmbedding)\b",
    "PatchEmbedding": r"\b(?:PatchEmbed|patch_embed|PatchEmbedding)\b",
    "FlashAttention": r"\b(?:flash_attn|FlashAttention|flash_attention)\b",
    "AdamW": r"\bAdamW\b",
    "CosineSchedule": r"\b(?:CosineAnnealing|cosine_schedule|CosineSchedule)\b",
    "TransformerEncoder": r"\bTransformerEncoder\b",
    "QuantileHead": r"\b(?:QuantileHead|quantile_head|quantile_output)\b",
    "MoE": r"\b(?:MixtureOfExperts|MoE|mixture_of_experts)\b",
}


def detect_components(
    code: str, patterns: Optional[dict[str, str]] = None,
) -> list[str]:
    """Regex scan for known ML components. Returns component names."""
    pats = patterns or DEFAULT_COMPONENT_PATTERNS
    return [name for name, pattern in pats.items() if re.search(pattern, code)]


def compute_similarity(code_a: str, code_b: str) -> dict:
    """Return multiple similarity signals. No thresholds — consumer decides."""
    jaccard = code_similarity(code_a, code_b)

    lines_a = code_a.splitlines(keepends=True)
    lines_b = code_b.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
    diff_ratio = matcher.ratio()

    total_lines = max(len(lines_a), len(lines_b), 1)
    ops = matcher.get_opcodes()
    changed = sum(max(j2 - j1, i2 - i1) for op, i1, i2, j1, j2 in ops if op != "equal")
    diff_size = changed / total_lines

    return {"jaccard": jaccard, "diff_ratio": diff_ratio, "diff_size": diff_size}


class ProvenanceQuery:
    """Query provenance from raw observation tables.

    All methods are read-only. No pre-computed edges, no confidence
    scores. Compute relationships on demand from raw facts.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(PROVENANCE_SCHEMA)
        self.conn.commit()

    # ── Record facts ─────────────────────────────────────

    def record_round_context(
        self, round_id: int, experiment_id: int,
        context_type: str = "frontier", timestamp: Optional[float] = None,
    ):
        """Record that an experiment was shown in a round's challenge."""
        import time
        self.conn.execute(
            "INSERT INTO round_context (round_id, experiment_id, context_type, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (round_id, experiment_id, context_type, timestamp or time.time()),
        )
        self.conn.commit()

    def record_components(self, experiment_id: int, components: list[str]):
        """Store detected components for an experiment."""
        for comp in components:
            try:
                self.conn.execute(
                    "INSERT INTO code_components (experiment_id, component) "
                    "VALUES (?, ?)", (experiment_id, comp),
                )
            except sqlite3.IntegrityError:
                continue
        self.conn.commit()

    # ── Query: influences ────────────────────────────────

    def get_influences(self, experiment_id: int) -> list[dict]:
        """All evidence of what influenced this experiment.

        Returns raw facts: access log entries, frontier context, code
        similarity, shared components. No confidence scores.
        """
        results: list[dict] = []
        exp = self._get_experiment(experiment_id)
        if exp is None:
            return results

        miner_hotkey = exp["miner_hotkey"]
        round_id = exp["round_id"]

        # 1. Access log: what experiments did the miner's agent query?
        if miner_hotkey and round_id is not None:
            accessed = self._get_accessed_ids(miner_hotkey, round_id)
            for aid in accessed:
                results.append({
                    "source_id": aid,
                    "evidence_type": "accessed",
                    "detail": {"round_id": round_id},
                })

        # 2. Frontier context: what was shown in the challenge?
        if round_id is not None:
            frontier_ids = self._get_round_context_ids(round_id)
            for fid in frontier_ids:
                results.append({
                    "source_id": fid,
                    "evidence_type": "frontier",
                    "detail": {"round_id": round_id},
                })

        # 3. Shared components
        my_comps = self.get_experiment_components(experiment_id)
        if my_comps:
            for comp in my_comps:
                other_ids = self.get_component_experiments(comp)
                for oid in other_ids:
                    if oid != experiment_id and oid < experiment_id:
                        results.append({
                            "source_id": oid,
                            "evidence_type": "shared_component",
                            "detail": {"component": comp},
                        })

        return results

    # ── Query: impact ────────────────────────────────────

    def get_impact(self, experiment_id: int) -> list[dict]:
        """All evidence of what this experiment influenced."""
        results: list[dict] = []

        # Which later experiments' agents accessed this one?
        try:
            rows = self.conn.execute(
                "SELECT DISTINCT a.hotkey, a.round_id, e.id as exp_id "
                "FROM miner_access_log a "
                "JOIN experiments e ON e.miner_hotkey = a.hotkey "
                "  AND e.round_id = a.round_id "
                "WHERE a.experiment_ids LIKE ? AND e.id > ?",
                (f'%{experiment_id}%', experiment_id),
            ).fetchall()
            for row in rows:
                results.append({
                    "target_id": row[2],
                    "evidence_type": "accessed_by",
                    "detail": {"round_id": row[1]},
                })
        except sqlite3.OperationalError:
            pass  # miner_access_log not initialized yet

        # Which later experiments were shown this in their frontier?
        ctx_rows = self.conn.execute(
            "SELECT round_id FROM round_context WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for row in ctx_rows:
            later = self.conn.execute(
                "SELECT id FROM experiments WHERE round_id = ? AND id > ?",
                (row[0], experiment_id),
            ).fetchall()
            for lr in later:
                results.append({
                    "target_id": lr[0],
                    "evidence_type": "frontier_for",
                    "detail": {"round_id": row[0]},
                })

        return results

    # ── Query: similarity ────────────────────────────────

    def get_similar(
        self, experiment_id: int,
        pool: Optional[list[int]] = None, top_k: int = 10,
    ) -> list[dict]:
        """Code similarity against a pool (or recent experiments)."""
        exp = self._get_experiment(experiment_id)
        if exp is None or not exp["code"]:
            return []

        if pool is not None:
            placeholders = ",".join("?" * len(pool))
            rows = self.conn.execute(
                f"SELECT id, code FROM experiments WHERE id IN ({placeholders})",
                pool,
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, code FROM experiments WHERE id != ? "
                "ORDER BY id DESC LIMIT 50",
                (experiment_id,),
            ).fetchall()

        results = []
        for row in rows:
            if row[0] == experiment_id or not row[1]:
                continue
            sim = compute_similarity(exp["code"], row[1])
            results.append({"target_id": row[0], **sim})

        results.sort(key=lambda x: -x["jaccard"])
        return results[:top_k]

    # ── Query: components ────────────────────────────────

    def get_component_experiments(self, component: str) -> list[int]:
        """Which experiments use this component?"""
        rows = self.conn.execute(
            "SELECT experiment_id FROM code_components WHERE component = ?",
            (component,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_experiment_components(self, experiment_id: int) -> list[str]:
        """Which components does this experiment use?"""
        rows = self.conn.execute(
            "SELECT component FROM code_components WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_component_stats(self) -> list[dict]:
        """Which components correlate with good metrics?"""
        rows = self.conn.execute(
            "SELECT c.component, COUNT(*) as count, "
            "AVG(e.metric) as avg_metric, MIN(e.metric) as best_metric "
            "FROM code_components c "
            "JOIN experiments e ON c.experiment_id = e.id "
            "WHERE e.success = 1 AND e.metric IS NOT NULL "
            "GROUP BY c.component ORDER BY count DESC",
        ).fetchall()
        return [
            {"component": r[0], "count": r[1],
             "avg_metric": r[2], "best_metric": r[3]}
            for r in rows
        ]

    # ── Query: dead ends ─────────────────────────────────

    def get_dead_ends(self, task: str = "") -> list[int]:
        """Successful experiments with no later similar submissions."""
        tc = " AND task = ?" if task else ""
        tp = (task,) if task else ()
        rows = self.conn.execute(
            "SELECT id, code FROM experiments WHERE success = 1 "
            f"AND metric IS NOT NULL{tc} ORDER BY id", tp,
        ).fetchall()
        dead = []
        for i, row in enumerate(rows):
            if not row[1]:
                continue
            has_successor = False
            for later in rows[i + 1:]:
                if later[1] and code_similarity(row[1], later[1]) > 0.3:
                    has_successor = True
                    break
            if not has_successor:
                dead.append(row[0])
        return dead

    # ── Query: experiment graph ──────────────────────────

    def get_experiment_graph(
        self, experiment_id: int, depth: int = 3,
    ) -> dict:
        """Build a local subgraph around one experiment."""
        influences = self.get_influences(experiment_id)
        impact = self.get_impact(experiment_id)
        components = self.get_experiment_components(experiment_id)
        return {
            "experiment_id": experiment_id,
            "influences": influences,
            "impact": impact,
            "components": components,
        }

    # ── Export ────────────────────────────────────────────

    def export(self, task: str = "") -> dict:
        """Full provenance data for visualization/analysis."""
        tc = " AND task = ?" if task else ""
        tp = (task,) if task else ()

        experiments = self.conn.execute(
            f"SELECT id, name, miner_uid, metric, success FROM experiments WHERE 1=1{tc}",
            tp,
        ).fetchall()

        components = self.conn.execute(
            "SELECT * FROM code_components",
        ).fetchall()

        context = self.conn.execute(
            "SELECT * FROM round_context",
        ).fetchall()

        access_count = self.conn.execute(
            "SELECT COUNT(*) FROM miner_access_log",
        ).fetchone()[0]

        return {
            "experiments": [
                {"id": r[0], "name": r[1], "miner_uid": r[2],
                 "metric": r[3], "success": bool(r[4])}
                for r in experiments
            ],
            "components": [
                {"experiment_id": r[1], "component": r[2]}
                for r in components
            ],
            "round_context": [
                {"round_id": r[1], "experiment_id": r[2],
                 "context_type": r[3]}
                for r in context
            ],
            "access_log_count": access_count,
            "component_stats": self.get_component_stats(),
        }

    # ── Internal helpers ─────────────────────────────────

    def _get_experiment(self, experiment_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id, code, miner_hotkey, round_id FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "code": row[1], "miner_hotkey": row[2], "round_id": row[3]}

    def _get_accessed_ids(self, hotkey: str, round_id: int) -> set[int]:
        try:
            rows = self.conn.execute(
                "SELECT experiment_ids FROM miner_access_log "
                "WHERE hotkey = ? AND round_id = ?",
                (hotkey, round_id),
            ).fetchall()
        except sqlite3.OperationalError:
            return set()  # miner_access_log not initialized yet
        ids: set[int] = set()
        for row in rows:
            ids.update(json.loads(row[0]))
        return ids

    def _get_round_context_ids(self, round_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT experiment_id FROM round_context WHERE round_id = ?",
            (round_id,),
        ).fetchall()
        return [r[0] for r in rows]
