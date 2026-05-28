"""SQLite-backed store for the local stack.

Single file, single process write at a time, used as a broker between
the validator and miner. Tables:

* ``challenges`` — validator-published rounds. One row per challenge_id;
  the miner picks up rows whose ``status='open'``.
* ``proposals`` — miner-submitted designs. Validator drains them when
  it ends Phase A.
* ``experiments`` — Phase C outcomes; this is the "frontier" source.

The store is intentionally smaller than ``shared/pg_store.py`` — we
only need what one round of A → B → C needs.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS challenges (
    challenge_id    TEXT PRIMARY KEY,
    round_id        INTEGER NOT NULL,
    payload_json    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_challenges_status ON challenges(status);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    challenge_id    TEXT NOT NULL,
    round_id        INTEGER NOT NULL,
    miner_id        TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    created_at      REAL NOT NULL,
    UNIQUE(challenge_id, miner_id)
);
CREATE INDEX IF NOT EXISTS idx_proposals_round ON proposals(round_id);

CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id        INTEGER NOT NULL,
    miner_id        TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    code            TEXT NOT NULL DEFAULT '',
    motivation      TEXT NOT NULL DEFAULT '',
    reasoning       TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    metric          REAL,
    success         INTEGER NOT NULL DEFAULT 0,
    objectives_json TEXT NOT NULL DEFAULT '{}',
    score           REAL NOT NULL DEFAULT 0,
    loss_curve_json TEXT NOT NULL DEFAULT '[]',
    analysis        TEXT NOT NULL DEFAULT '',
    parent_index    INTEGER,
    generation      INTEGER NOT NULL DEFAULT 0,
    prompt_id       TEXT NOT NULL DEFAULT '',
    task            TEXT NOT NULL DEFAULT '',
    timestamp       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_round ON experiments(round_id);
CREATE INDEX IF NOT EXISTS idx_exp_success_metric ON experiments(success, metric);

CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id        INTEGER NOT NULL,
    miner_id        TEXT NOT NULL DEFAULT '',
    task            TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL,
    rel_path        TEXT NOT NULL DEFAULT '',
    bucket          TEXT NOT NULL DEFAULT '',
    s3_key          TEXT NOT NULL DEFAULT '',
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    content_text    TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_round_miner ON artifacts(round_id, miner_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);
"""


# Kinds: artifacts.kind values. Text kinds get an inline copy of the
# content in ``artifacts.content_text``; binary kinds only carry the
# ``s3_key`` reference.
ARTIFACT_KINDS = (
    "challenge", "proposal", "submission", "result", "log", "checkpoint",
)
ARTIFACT_TEXT_KINDS = {"challenge", "proposal", "submission", "result", "log"}


class LocalStore:
    """Thin SQLite wrapper. All methods are synchronous — one process at
    a time writes, but readers and writers can run concurrently thanks
    to WAL mode."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets the same connection be used
        # from a background thread in the validator's eval loop. We
        # serialize writes via short ``with self._conn:`` blocks.
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; use explicit BEGIN
            timeout=30,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # ── Challenges ──────────────────────────────────────────

    def post_challenge(self, challenge_id: str, round_id: int, payload: dict) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO challenges "
                "(challenge_id, round_id, payload_json, status, created_at) "
                "VALUES (?, ?, ?, 'open', ?)",
                (challenge_id, round_id, json.dumps(payload), time.time()),
            )

    def open_challenge(self) -> Optional[dict]:
        """Latest open challenge (the miner consumes this)."""
        row = self._conn.execute(
            "SELECT * FROM challenges WHERE status = 'open' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "challenge_id": row["challenge_id"],
            "round_id": row["round_id"],
            "payload": json.loads(row["payload_json"]),
            "status": row["status"],
        }

    def mark_challenge(self, challenge_id: str, status: str) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE challenges SET status = ? WHERE challenge_id = ?",
                (status, challenge_id),
            )

    # ── Proposals ───────────────────────────────────────────

    def post_proposal(
        self, challenge_id: str, round_id: int, miner_id: str, payload: dict
    ) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT OR REPLACE INTO proposals "
                "(challenge_id, round_id, miner_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (challenge_id, round_id, miner_id, json.dumps(payload), time.time()),
            )
            return cur.lastrowid

    def proposals_for(self, challenge_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM proposals WHERE challenge_id = ? "
            "ORDER BY created_at ASC",
            (challenge_id,),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "proposal_id": r["proposal_id"],
                "challenge_id": r["challenge_id"],
                "round_id": r["round_id"],
                "miner_id": r["miner_id"],
                "payload": json.loads(r["payload_json"]),
                "created_at": r["created_at"],
            })
        return out

    # ── Experiments ────────────────────────────────────────

    def add_experiment(self, *, round_id: int, miner_id: str, name: str,
                       code: str, motivation: str, reasoning: str,
                       tool_calls: list, metric: Optional[float],
                       success: bool, objectives: dict, score: float,
                       loss_curve: list, analysis: str = "",
                       parent_index: Optional[int] = None,
                       generation: int = 0, prompt_id: str = "",
                       task: str = "") -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO experiments "
                "(round_id, miner_id, name, code, motivation, reasoning, "
                " tool_calls_json, metric, success, objectives_json, score, "
                " loss_curve_json, analysis, parent_index, generation, "
                " prompt_id, task, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    round_id, miner_id, name, code, motivation, reasoning,
                    json.dumps(tool_calls), metric, 1 if success else 0,
                    json.dumps(objectives), score, json.dumps(loss_curve),
                    analysis, parent_index, generation, prompt_id, task,
                    time.time(),
                ),
            )
            return cur.lastrowid

    def get_experiment(self, exp_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (exp_id,)
        ).fetchone()
        return _row_to_experiment(row) if row else None

    def recent_experiments(self, n: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [_row_to_experiment(r) for r in rows]

    def successful_in_flops_range(
        self, min_flops: int, max_flops: int,
    ) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM experiments WHERE success = 1 AND metric IS NOT NULL"
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            exp = _row_to_experiment(r)
            flops = exp["objectives"].get("flops_equivalent_size", 0)
            if min_flops <= flops <= max_flops:
                out.append(exp)
        return out

    def best_experiment(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE success = 1 AND metric IS NOT NULL "
            "ORDER BY metric ASC LIMIT 1"
        ).fetchone()
        return _row_to_experiment(row) if row else None

    # ── Artifacts ──────────────────────────────────────────

    def add_artifact(
        self, *, round_id: int, miner_id: str, task: str, kind: str,
        rel_path: str = "", bucket: str = "", s3_key: str = "",
        size_bytes: int = 0, content_text: Optional[str] = None,
    ) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO artifacts "
                "(round_id, miner_id, task, kind, rel_path, bucket, s3_key, "
                " size_bytes, content_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(round_id), miner_id, task, kind, rel_path, bucket,
                    s3_key, int(size_bytes), content_text, time.time(),
                ),
            )
            return cur.lastrowid

    def list_artifacts(
        self, *, round_id: Optional[int] = None,
        miner_id: Optional[str] = None, task: Optional[str] = None,
        kind: Optional[str] = None, limit: int = 200,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list = []
        if round_id is not None:
            clauses.append("round_id = ?")
            params.append(int(round_id))
        if miner_id:
            clauses.append("miner_id = ?")
            params.append(miner_id)
        if task:
            clauses.append("task = ?")
            params.append(task)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, int(limit)))
        rows = self._conn.execute(
            "SELECT * FROM artifacts" + where +
            " ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_artifact(r, include_text=False) for r in rows]

    def get_artifact(self, artifact_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (int(artifact_id),)
        ).fetchone()
        return _row_to_artifact(row, include_text=True) if row else None

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful, "
            "MIN(metric) AS best, MAX(metric) AS worst, AVG(metric) AS mean "
            "FROM experiments"
        ).fetchone()
        total = row["total"] or 0
        successful = row["successful"] or 0
        return {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "best_metric": row["best"],
            "worst_metric": row["worst"],
            "mean_metric": row["mean"],
        }


def _row_to_artifact(row: sqlite3.Row, *, include_text: bool) -> dict:
    out = {
        "id": row["id"],
        "round_id": row["round_id"],
        "miner_id": row["miner_id"],
        "task": row["task"],
        "kind": row["kind"],
        "rel_path": row["rel_path"],
        "bucket": row["bucket"],
        "s3_key": row["s3_key"],
        "size_bytes": row["size_bytes"],
        "created_at": row["created_at"],
        "has_inline_text": row["content_text"] is not None,
    }
    if include_text:
        out["content_text"] = row["content_text"]
    return out


def _row_to_experiment(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "round_id": row["round_id"],
        "miner_id": row["miner_id"],
        "name": row["name"],
        "code": row["code"],
        "motivation": row["motivation"],
        "reasoning": row["reasoning"],
        "tool_calls": json.loads(row["tool_calls_json"]),
        "metric": row["metric"],
        "success": bool(row["success"]),
        "objectives": json.loads(row["objectives_json"]),
        "score": row["score"],
        "loss_curve": json.loads(row["loss_curve_json"]),
        "analysis": row["analysis"],
        "parent_index": row["parent_index"],
        "generation": row["generation"],
        "prompt_id": row["prompt_id"],
        "task": row["task"],
        "timestamp": row["timestamp"],
    }
