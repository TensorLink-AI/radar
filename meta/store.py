"""Storage layout for the meta-orchestrator.

One global ``registry.db`` lists experiments + their on-disk paths.
Each experiment has its own ``meta.db`` under
``meta/data/experiments/<name>/meta.db`` with three tables:

* ``units`` — one row per unit (pod_id, ssh, status, instance_id, …)
* ``meta_events`` — append-only audit log of orchestrator actions
* ``frontier`` — denormalized per-unit experiment rows ingested from R2

This mirrors how ``local/`` already works (one SQLite per validator) and
keeps experiments truly isolated — dropping one experiment's DB does
not touch any other.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    name            TEXT PRIMARY KEY,
    path            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    spec_yaml       TEXT NOT NULL DEFAULT '',
    started_at      REAL,
    finished_at     REAL,
    budget_usd      REAL NOT NULL DEFAULT 0,
    spent_usd       REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reg_exp_status ON experiments(status);
"""


META_SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    id                  TEXT PRIMARY KEY,
    instance_id         TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'queued',
    provider            TEXT NOT NULL DEFAULT '',
    pod_id              TEXT NOT NULL DEFAULT '',
    ssh_host            TEXT NOT NULL DEFAULT '',
    ssh_port            INTEGER NOT NULL DEFAULT 22,
    ssh_user            TEXT NOT NULL DEFAULT 'root',
    hourly_usd          REAL NOT NULL DEFAULT 0,
    started_at          REAL,
    finished_at         REAL,
    last_heartbeat_ts   REAL,
    last_round_seen     INTEGER NOT NULL DEFAULT -1,
    pod_hours           REAL NOT NULL DEFAULT 0,
    cost_usd            REAL NOT NULL DEFAULT 0,
    cfg_json            TEXT NOT NULL DEFAULT '{}',
    kill_reason         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS meta_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    unit_id         TEXT NOT NULL DEFAULT '',
    actor           TEXT NOT NULL DEFAULT 'orchestrator',
    kind            TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_unit_kind ON meta_events(unit_id, kind);

CREATE TABLE IF NOT EXISTS frontier (
    -- One row per (unit, ingested experiment from that unit's
    -- radar_local.db). We re-upsert on (unit_id, source_exp_id) so
    -- repeated ingestion is idempotent.
    unit_id             TEXT NOT NULL,
    source_exp_id       INTEGER NOT NULL,
    round_id            INTEGER NOT NULL,
    miner_id            TEXT NOT NULL,
    name                TEXT NOT NULL DEFAULT '',
    metric              REAL,
    success             INTEGER NOT NULL DEFAULT 0,
    objectives_json     TEXT NOT NULL DEFAULT '{}',
    task                TEXT NOT NULL DEFAULT '',
    mode                TEXT NOT NULL DEFAULT 'new',
    cumulative_compute  REAL NOT NULL DEFAULT 0,
    ingested_at         REAL NOT NULL,
    PRIMARY KEY (unit_id, source_exp_id)
);
CREATE INDEX IF NOT EXISTS idx_frontier_unit ON frontier(unit_id);
CREATE INDEX IF NOT EXISTS idx_frontier_metric ON frontier(success, metric);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


class Registry:
    """The tiny global registry of experiments."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn = _connect(self.path)
        self._conn.executescript(REGISTRY_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def upsert_experiment(
        self,
        name: str,
        path: str,
        spec_yaml: str,
        budget_usd: float,
    ) -> None:
        with self.tx() as c:
            c.execute(
                """
                INSERT INTO experiments(name, path, spec_yaml, budget_usd,
                                        status, started_at)
                VALUES (?, ?, ?, ?, 'queued', ?)
                ON CONFLICT(name) DO UPDATE SET
                    path = excluded.path,
                    spec_yaml = excluded.spec_yaml,
                    budget_usd = excluded.budget_usd
                """,
                (name, path, spec_yaml, budget_usd, time.time()),
            )

    def list_experiments(self) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM experiments ORDER BY started_at DESC"
        ))

    def get_experiment(self, name: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM experiments WHERE name = ?", (name,)
        ).fetchone()

    def set_status(self, name: str, status: str) -> None:
        finished = time.time() if status in {"done", "failed"} else None
        with self.tx() as c:
            c.execute(
                "UPDATE experiments SET status = ?, "
                "finished_at = COALESCE(?, finished_at) WHERE name = ?",
                (status, finished, name),
            )

    def add_spend(self, name: str, delta_usd: float) -> float:
        with self.tx() as c:
            c.execute(
                "UPDATE experiments SET spent_usd = spent_usd + ? "
                "WHERE name = ?",
                (delta_usd, name),
            )
        row = self.get_experiment(name)
        return float(row["spent_usd"]) if row else 0.0


class ExperimentStore:
    """Per-experiment ``meta.db``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn = _connect(self.path)
        self._conn.executescript(META_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ── units ────────────────────────────────────────────────────────

    def insert_unit(
        self, unit_id: str, instance_id: str, cfg: dict[str, Any]
    ) -> None:
        with self.tx() as c:
            c.execute(
                """
                INSERT OR IGNORE INTO units(id, instance_id, status, cfg_json)
                VALUES (?, ?, 'queued', ?)
                """,
                (unit_id, instance_id, json.dumps(cfg)),
            )

    def update_unit(self, unit_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        cols = ", ".join(f"{k} = ?" for k in kwargs)
        with self.tx() as c:
            c.execute(
                f"UPDATE units SET {cols} WHERE id = ?",
                (*kwargs.values(), unit_id),
            )

    def get_unit(self, unit_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM units WHERE id = ?", (unit_id,)
        ).fetchone()

    def list_units(self, status: str | None = None) -> list[sqlite3.Row]:
        if status is None:
            return list(self._conn.execute("SELECT * FROM units"))
        return list(self._conn.execute(
            "SELECT * FROM units WHERE status = ?", (status,)
        ))

    # ── events ───────────────────────────────────────────────────────

    def log_event(
        self,
        kind: str,
        *,
        unit_id: str = "",
        actor: str = "orchestrator",
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO meta_events(ts, unit_id, actor, kind, "
                "payload_json) VALUES (?, ?, ?, ?, ?)",
                (time.time(), unit_id, actor, kind,
                 json.dumps(payload or {})),
            )

    def recent_events(self, limit: int = 100) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM meta_events ORDER BY id DESC LIMIT ?", (limit,)
        ))

    # ── frontier ─────────────────────────────────────────────────────

    def upsert_frontier_rows(self, unit_id: str, rows: list[dict[str, Any]]) -> int:
        now = time.time()
        with self.tx() as c:
            for r in rows:
                c.execute(
                    """
                    INSERT INTO frontier(unit_id, source_exp_id, round_id,
                        miner_id, name, metric, success, objectives_json,
                        task, mode, cumulative_compute, ingested_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(unit_id, source_exp_id) DO UPDATE SET
                        metric = excluded.metric,
                        success = excluded.success,
                        objectives_json = excluded.objectives_json,
                        cumulative_compute = excluded.cumulative_compute,
                        ingested_at = excluded.ingested_at
                    """,
                    (
                        unit_id,
                        r["source_exp_id"],
                        r["round_id"],
                        r["miner_id"],
                        r.get("name", ""),
                        r.get("metric"),
                        int(bool(r.get("success", 0))),
                        json.dumps(r.get("objectives", {})),
                        r.get("task", ""),
                        r.get("mode", "new"),
                        float(r.get("cumulative_compute", 0)),
                        now,
                    ),
                )
        return len(rows)

    def best_per_unit(self) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            """
            SELECT unit_id, MIN(metric) AS best_metric,
                   COUNT(*) AS n_rows,
                   SUM(success) AS n_success,
                   MAX(round_id) AS max_round
            FROM frontier
            WHERE metric IS NOT NULL
            GROUP BY unit_id
            """
        ))


def experiment_dir(root: Path, name: str) -> Path:
    return Path(root) / "experiments" / name
