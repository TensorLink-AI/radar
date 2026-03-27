"""SQLite-backed experiment store with FTS and diff API.

Drop-in replacement for ExperimentDB. All existing methods preserved.
New diff methods added for agent context. All query methods accept
an optional task parameter for multi-task isolation.
"""

import logging
import re
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from shared.database import DataElement
from shared.provenance import ProvenanceQuery
from shared.sqlite_schema import (
    SCHEMA_TABLE_DDL, SCHEMA_INDEX_DDL, FTS_DDL, TRIGGERS_DDL, INSERT_SQL,
    row_to_element, element_to_params, compute_diff,
    migrate_add_task_column,
)

logger = logging.getLogger(__name__)


class SQLiteExperimentStore:
    """SQLite-backed experiment store with FTS and diff API."""

    def __init__(self, db_path: str = "./experiments/experiments.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_size_limit=67108864")
        cur.executescript(SCHEMA_TABLE_DDL)
        # Migrate before creating indexes that reference task column
        migrate_add_task_column(self.conn)
        cur.executescript(SCHEMA_INDEX_DDL)
        cur.executescript(FTS_DDL)
        cur.executescript(TRIGGERS_DDL)
        self.conn.commit()
        # Provenance query layer shares the same connection
        self.provenance = ProvenanceQuery(self.conn)

    def _task_clause(self, task: str, prefix: str = "AND") -> tuple[str, tuple]:
        """Return (SQL fragment, params) for optional task filtering."""
        if task:
            return f" {prefix} task = ?", (task,)
        return "", ()

    # ── Core API ─────────────────────────────────────────

    @property
    def size(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    def add(self, element: DataElement) -> int:
        with self._lock:
            next_id = self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            element.index = next_id
            element.timestamp = time.time()
            self.conn.execute(INSERT_SQL, element_to_params(element, next_id))
            self.conn.commit()
            return next_id

    def add_batch(self, elements: list[DataElement]) -> list[int]:
        with self._lock:
            indices = []
            now = time.time()
            next_id = self.conn.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]
            for elem in elements:
                elem.index = next_id
                elem.timestamp = now
                self.conn.execute(INSERT_SQL, element_to_params(elem, next_id))
                indices.append(next_id)
                next_id += 1
            if indices:
                self.conn.commit()
            return indices

    def get(self, index: int) -> Optional[DataElement]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (index,)
            ).fetchone()
            return row_to_element(row) if row else None

    def get_successful(self, task: str = "") -> list[DataElement]:
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE success = 1 AND metric IS NOT NULL"
                f"{tc} ORDER BY metric ASC", tp
            ).fetchall()
            return [row_to_element(r) for r in rows]

    def get_best(self, n: int = 1, task: str = "") -> list[DataElement]:
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE success = 1 AND metric IS NOT NULL"
                f"{tc} ORDER BY metric ASC LIMIT ?", tp + (n,)
            ).fetchall()
            return [row_to_element(r) for r in rows]

    def get_recent(self, n: int = 5, task: str = "") -> list[DataElement]:
        with self._lock:
            tc, tp = self._task_clause(task, prefix="WHERE")
            sql = "SELECT * FROM experiments" + tc + " ORDER BY id DESC LIMIT ?"
            rows = self.conn.execute(sql, tp + (n,)).fetchall()
            return [row_to_element(r) for r in rows]

    def get_failures(self, n: int = 10, task: str = "") -> list[DataElement]:
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE success = 0"
                f"{tc} ORDER BY id DESC LIMIT ?", tp + (n,)
            ).fetchall()
            return [row_to_element(r) for r in rows]

    def get_children(self, parent_index: int, task: str = "") -> list[DataElement]:
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                f"SELECT * FROM experiments WHERE parent_index = ?{tc}",
                (parent_index,) + tp,
            ).fetchall()
            return [row_to_element(r) for r in rows]

    def get_lineage(self, index: int) -> list[DataElement]:
        with self._lock:
            lineage = []
            row = self.conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (index,)
            ).fetchone()
            while row is not None:
                elem = row_to_element(row)
                lineage.append(elem)
                if elem.parent is not None:
                    row = self.conn.execute(
                        "SELECT * FROM experiments WHERE id = ?", (elem.parent,)
                    ).fetchone()
                else:
                    row = None
            return list(reversed(lineage))

    @staticmethod
    def _sanitize_fts(query: str) -> str:
        """Strip FTS5 operators so user input can't break the query."""
        words = query.strip().split()
        if not words:
            return ""
        # Quote each token to neutralise FTS5 syntax (*, NEAR, AND, OR, NOT …)
        return " OR ".join(f'"{w}"' for w in words)

    def search(self, query: str, top_k: int = 10, task: str = "") -> list[DataElement]:
        fts_query = self._sanitize_fts(query)
        if not fts_query:
            return []
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                "SELECT e.* FROM experiments e "
                "JOIN experiments_fts f ON e.id = f.rowid "
                f"WHERE experiments_fts MATCH ?{tc} "
                "ORDER BY f.rank LIMIT ?",
                (fts_query,) + tp + (top_k,),
            ).fetchall()
            return [row_to_element(r) for r in rows]

    def search_failures(self, query: str, top_k: int = 10, task: str = "") -> list[DataElement]:
        fts_query = self._sanitize_fts(query)
        if not fts_query:
            return []
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                "SELECT e.* FROM experiments e "
                "JOIN experiments_fts f ON e.id = f.rowid "
                f"WHERE experiments_fts MATCH ? AND e.success = 0{tc} "
                "ORDER BY f.rank LIMIT ?",
                (fts_query,) + tp + (top_k,),
            ).fetchall()
            return [row_to_element(r) for r in rows]

    def get_component_stats(
        self, patterns: Optional[dict[str, str]] = None, task: str = "",
    ) -> dict:
        if not patterns:
            return {}
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                f"SELECT code FROM experiments WHERE success = 1{tc}", tp
            ).fetchall()
        stats: dict[str, Counter] = {k: Counter() for k in patterns}
        for row in rows:
            for category, pattern in patterns.items():
                for match in re.findall(pattern, row["code"], re.IGNORECASE):
                    stats[category][match] += 1
        return {k: dict(v.most_common(10)) for k, v in stats.items()}

    def get_pareto_elements(self, task: str = "") -> list[DataElement]:
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE success = 1 AND metric IS NOT NULL"
                + tc, tp
            ).fetchall()
            return [row_to_element(r) for r in rows]

    def count_in_flops_range(
        self, min_flops: int, max_flops: int, task: str = "",
    ) -> int:
        with self._lock:
            tc, tp = self._task_clause(task)
            row = self.conn.execute(
                "SELECT COUNT(*) FROM experiments "
                "WHERE success = 1 AND metric IS NOT NULL "
                "AND CAST(json_extract(objectives, '$.flops_equivalent_size') "
                f"AS INTEGER) BETWEEN ? AND ?{tc}",
                (min_flops, max_flops) + tp,
            ).fetchone()
            return row[0]

    def get_in_flops_range(
        self, min_flops: int, max_flops: int, task: str = "",
    ) -> list[DataElement]:
        with self._lock:
            tc, tp = self._task_clause(task)
            rows = self.conn.execute(
                "SELECT * FROM experiments "
                "WHERE success = 1 AND metric IS NOT NULL "
                "AND CAST(json_extract(objectives, '$.flops_equivalent_size') "
                f"AS INTEGER) BETWEEN ? AND ?{tc}",
                (min_flops, max_flops) + tp,
            ).fetchall()
            return [row_to_element(r) for r in rows]

    def stats(self, task: str = "") -> dict:
        with self._lock:
            w = " WHERE task = ?" if task else ""
            p = (task,) if task else ()
            row = self.conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful, "
                f"MAX(generation) as max_gen FROM experiments{w}", p,
            ).fetchone()
            total, successful = row["total"], row["successful"] or 0
            max_gen = row["max_gen"] if row["max_gen"] is not None else 0
            mw = f"{w}{' AND' if task else ' WHERE'} success = 1 AND metric IS NOT NULL"
            mrow = self.conn.execute(
                f"SELECT MIN(metric) as best, MAX(metric) as worst, "
                f"AVG(metric) as mean FROM experiments{mw}", p,
            ).fetchone()
            return {
                "total": total, "successful": successful,
                "failed": total - successful,
                "best_metric": mrow["best"], "worst_metric": mrow["worst"],
                "mean_metric": mrow["mean"], "max_generation": max_gen,
            }

    # ── Task discovery ──────────────────────────────────

    def get_tasks(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT task FROM experiments WHERE task != '' ORDER BY task"
            ).fetchall()
            return [r[0] for r in rows]

    def stats_by_task(self) -> dict[str, dict]:
        return {task: self.stats(task=task) for task in self.get_tasks()}

    # ── Diff API ────────────────────────────────────────

    def get_diff(self, index: int) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (index,)
            ).fetchone()
            if row is None:
                return None
            elem = row_to_element(row)
            parent = None
            if elem.parent is not None:
                prow = self.conn.execute(
                    "SELECT * FROM experiments WHERE id = ?", (elem.parent,)
                ).fetchone()
                parent = row_to_element(prow) if prow else None
            return compute_diff(parent, elem)

    def get_diff_between(self, index_a: int, index_b: int) -> Optional[str]:
        with self._lock:
            a_row = self.conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (index_a,)
            ).fetchone()
            b_row = self.conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (index_b,)
            ).fetchone()
            if not a_row or not b_row:
                return None
            return compute_diff(row_to_element(a_row), row_to_element(b_row))

    def get_lineage_diffs(self, index: int) -> list[dict]:
        lineage = self.get_lineage(index)
        return [
            {"index": e.index, "name": e.name, "metric": e.metric,
             "motivation": e.motivation,
             "diff": compute_diff(lineage[i - 1] if i > 0 else None, e)}
            for i, e in enumerate(lineage)
        ]

    def get_family_summary(self, task: str = "") -> list[dict]:
        with self._lock:
            tc, tp = self._task_clause(task)
            roots = self.conn.execute(
                f"SELECT id, name FROM experiments WHERE parent_index IS NULL{tc}",
                tp,
            ).fetchall()
            result = []
            for root in roots:
                row = self.conn.execute(
                    "WITH RECURSIVE family AS ("
                    "  SELECT id, metric, generation FROM experiments WHERE id = ? "
                    "  UNION ALL "
                    "  SELECT e.id, e.metric, e.generation "
                    "  FROM experiments e JOIN family f ON e.parent_index = f.id"
                    ") "
                    "SELECT COUNT(*) as cnt, MIN(metric) as best, "
                    "MAX(id) as latest, MAX(generation) as max_gen FROM family",
                    (root["id"],),
                ).fetchone()
                result.append({
                    "root_index": root["id"], "root_name": root["name"],
                    "num_descendants": row["cnt"], "best_metric": row["best"],
                    "latest_index": row["latest"],
                    "max_generation": row["max_gen"] or 0,
                })
            return result

    def close(self):
        self.conn.close()
