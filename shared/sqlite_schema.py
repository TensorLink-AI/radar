"""SQLite schema DDL and row conversion helpers for SQLiteExperimentStore."""

import difflib
import json
import sqlite3
from typing import Optional

from shared.database import DataElement

SCHEMA_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    code TEXT NOT NULL DEFAULT '',
    motivation TEXT NOT NULL DEFAULT '',
    trace TEXT NOT NULL DEFAULT '',
    metric REAL,
    success BOOLEAN NOT NULL DEFAULT 0,
    analysis TEXT NOT NULL DEFAULT '',
    parent_index INTEGER,
    generation INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    miner_uid INTEGER NOT NULL DEFAULT -1,
    miner_hotkey TEXT NOT NULL DEFAULT '',
    loss_curve TEXT NOT NULL DEFAULT '[]',
    manifest_sha256 TEXT NOT NULL DEFAULT '',
    generated_samples TEXT NOT NULL DEFAULT '[]',
    objectives TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL DEFAULT 0.0,
    round_id INTEGER,
    task TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (parent_index) REFERENCES experiments(id)
);
"""

SCHEMA_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_success ON experiments(success);
CREATE INDEX IF NOT EXISTS idx_metric ON experiments(metric) WHERE metric IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_flops ON experiments(
    CAST(json_extract(objectives, '$.flops_equivalent_size') AS INTEGER)
) WHERE success = 1;
CREATE INDEX IF NOT EXISTS idx_round ON experiments(round_id);
CREATE INDEX IF NOT EXISTS idx_miner ON experiments(miner_uid);
CREATE INDEX IF NOT EXISTS idx_parent ON experiments(parent_index);
CREATE INDEX IF NOT EXISTS idx_generation ON experiments(generation);
CREATE INDEX IF NOT EXISTS idx_task ON experiments(task);
CREATE INDEX IF NOT EXISTS idx_task_success ON experiments(task, success);
CREATE INDEX IF NOT EXISTS idx_task_metric ON experiments(task, metric) WHERE metric IS NOT NULL;
"""

# Backward compat alias
SCHEMA_DDL = SCHEMA_TABLE_DDL + SCHEMA_INDEX_DDL

FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS experiments_fts USING fts5(
    motivation, analysis, name,
    content='experiments', content_rowid='id'
);
"""

TRIGGERS_DDL = """
CREATE TRIGGER IF NOT EXISTS experiments_ai AFTER INSERT ON experiments BEGIN
    INSERT INTO experiments_fts(rowid, motivation, analysis, name)
    VALUES (new.id, new.motivation, new.analysis, new.name);
END;

CREATE TRIGGER IF NOT EXISTS experiments_ad AFTER DELETE ON experiments BEGIN
    INSERT INTO experiments_fts(experiments_fts, rowid, motivation, analysis, name)
    VALUES ('delete', old.id, old.motivation, old.analysis, old.name);
END;

CREATE TRIGGER IF NOT EXISTS experiments_au AFTER UPDATE ON experiments BEGIN
    INSERT INTO experiments_fts(experiments_fts, rowid, motivation, analysis, name)
    VALUES ('delete', old.id, old.motivation, old.analysis, old.name);
    INSERT INTO experiments_fts(rowid, motivation, analysis, name)
    VALUES (new.id, new.motivation, new.analysis, new.name);
END;
"""


def row_to_element(row: sqlite3.Row) -> DataElement:
    """Convert a SQLite row to a DataElement. Deserialize JSON fields."""
    return DataElement(
        index=row["id"],
        name=row["name"],
        code=row["code"],
        motivation=row["motivation"],
        trace=row["trace"],
        metric=row["metric"],
        success=bool(row["success"]),
        analysis=row["analysis"],
        parent=row["parent_index"],
        generation=row["generation"],
        score=row["score"],
        miner_uid=row["miner_uid"],
        miner_hotkey=row["miner_hotkey"],
        loss_curve=json.loads(row["loss_curve"]),
        manifest_sha256=row["manifest_sha256"],
        generated_samples=json.loads(row["generated_samples"]),
        objectives=json.loads(row["objectives"]),
        timestamp=row["timestamp"],
        task=row["task"],
        round_id=row["round_id"] if row["round_id"] is not None else -1,
    )


def element_to_params(element: DataElement, next_id: int) -> dict:
    """Convert a DataElement to a dict of column values for INSERT."""
    round_id = element.round_id if element.round_id >= 0 else None
    return {
        "id": next_id,
        "name": element.name,
        "code": element.code,
        "motivation": element.motivation,
        "trace": element.trace,
        "metric": element.metric,
        "success": 1 if element.success else 0,
        "analysis": element.analysis,
        "parent_index": element.parent,
        "generation": element.generation,
        "score": element.score,
        "miner_uid": element.miner_uid,
        "miner_hotkey": element.miner_hotkey,
        "loss_curve": json.dumps(element.loss_curve),
        "manifest_sha256": element.manifest_sha256,
        "generated_samples": json.dumps(element.generated_samples),
        "objectives": json.dumps(element.objectives),
        "timestamp": element.timestamp,
        "round_id": round_id,
        "task": element.task,
    }


def compute_diff(
    parent: Optional[DataElement],
    child: DataElement,
) -> str:
    """Compute unified diff between parent and child experiment code."""
    if parent is None:
        from_lines: list[str] = []
        fromfile = "/dev/null"
    else:
        from_lines = parent.code.splitlines(keepends=True)
        fromfile = f"{parent.index:04d}__{parent.name}/architecture.py"
    to_lines = child.code.splitlines(keepends=True)
    tofile = f"{child.index:04d}__{child.name}/architecture.py"
    diff = difflib.unified_diff(
        from_lines, to_lines,
        fromfile=fromfile, tofile=tofile, lineterm="",
    )
    return "\n".join(diff)


def migrate_add_task_column(conn: sqlite3.Connection):
    """Add task column if upgrading from a DB without it."""
    cursor = conn.execute("PRAGMA table_info(experiments)")
    columns = {row[1] for row in cursor.fetchall()}
    if "task" not in columns:
        conn.execute(
            "ALTER TABLE experiments ADD COLUMN task TEXT NOT NULL DEFAULT ''"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task ON experiments(task)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_success "
            "ON experiments(task, success)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_metric "
            "ON experiments(task, metric) WHERE metric IS NOT NULL"
        )
        conn.commit()


INSERT_SQL = """
INSERT INTO experiments (
    id, name, code, motivation, trace, metric, success, analysis,
    parent_index, generation, score, miner_uid, miner_hotkey,
    loss_curve, manifest_sha256, generated_samples, objectives,
    timestamp, round_id, task
) VALUES (
    :id, :name, :code, :motivation, :trace, :metric, :success, :analysis,
    :parent_index, :generation, :score, :miner_uid, :miner_hotkey,
    :loss_curve, :manifest_sha256, :generated_samples, :objectives,
    :timestamp, :round_id, :task
)
"""
