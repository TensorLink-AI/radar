"""Postgres schema DDL and row conversion helpers for PgExperimentStore."""

import difflib
from typing import Optional

from shared.database import DataElement

SCHEMA_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    code TEXT NOT NULL DEFAULT '',
    motivation TEXT NOT NULL DEFAULT '',
    trace TEXT NOT NULL DEFAULT '',
    metric DOUBLE PRECISION,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    analysis TEXT NOT NULL DEFAULT '',
    parent_index INTEGER REFERENCES experiments(id),
    generation INTEGER NOT NULL DEFAULT 0,
    score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    miner_uid INTEGER NOT NULL DEFAULT -1,
    miner_hotkey TEXT NOT NULL DEFAULT '',
    loss_curve JSONB NOT NULL DEFAULT '[]',
    manifest_sha256 TEXT NOT NULL DEFAULT '',
    generated_samples JSONB NOT NULL DEFAULT '[]',
    objectives JSONB NOT NULL DEFAULT '{}',
    timestamp DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    round_id INTEGER,
    task TEXT NOT NULL DEFAULT '',
    search_vector tsvector
);
"""

SCHEMA_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_success ON experiments(success);
CREATE INDEX IF NOT EXISTS idx_metric ON experiments(metric) WHERE metric IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_flops ON experiments(
    CAST((objectives->>'flops_equivalent_size') AS INTEGER)
) WHERE success = TRUE;
CREATE INDEX IF NOT EXISTS idx_round ON experiments(round_id);
CREATE INDEX IF NOT EXISTS idx_miner ON experiments(miner_uid);
CREATE INDEX IF NOT EXISTS idx_parent ON experiments(parent_index);
CREATE INDEX IF NOT EXISTS idx_generation ON experiments(generation);
CREATE INDEX IF NOT EXISTS idx_task ON experiments(task);
CREATE INDEX IF NOT EXISTS idx_task_success ON experiments(task, success);
CREATE INDEX IF NOT EXISTS idx_task_metric ON experiments(task, metric)
    WHERE metric IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_search ON experiments USING GIN(search_vector);
"""

FTS_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION experiments_search_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.motivation, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.analysis, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

FTS_TRIGGER_DDL = """
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'experiments_search_trigger'
    ) THEN
        CREATE TRIGGER experiments_search_trigger
            BEFORE INSERT OR UPDATE ON experiments
            FOR EACH ROW EXECUTE FUNCTION experiments_search_update();
    END IF;
END $$;
"""

PROVENANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS round_context (
    id SERIAL PRIMARY KEY,
    round_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    context_type TEXT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rc_round ON round_context(round_id);
CREATE INDEX IF NOT EXISTS idx_rc_experiment ON round_context(experiment_id);

CREATE TABLE IF NOT EXISTS code_components (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    component TEXT NOT NULL,
    UNIQUE(experiment_id, component)
);
CREATE INDEX IF NOT EXISTS idx_cc_experiment ON code_components(experiment_id);
CREATE INDEX IF NOT EXISTS idx_cc_component ON code_components(component);
"""

ACCESS_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS miner_access_log (
    id SERIAL PRIMARY KEY,
    hotkey TEXT NOT NULL,
    miner_uid INTEGER NOT NULL DEFAULT -1,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT 'GET',
    experiment_ids JSONB NOT NULL DEFAULT '[]',
    timestamp DOUBLE PRECISION NOT NULL,
    round_id INTEGER NOT NULL DEFAULT -1
);
CREATE INDEX IF NOT EXISTS idx_access_hotkey ON miner_access_log(hotkey);
CREATE INDEX IF NOT EXISTS idx_access_round ON miner_access_log(round_id);
CREATE INDEX IF NOT EXISTS idx_access_hotkey_round
    ON miner_access_log(hotkey, round_id);
"""


def row_to_element(row) -> DataElement:
    """Convert an asyncpg Record to a DataElement. JSONB auto-deserialized."""
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
        loss_curve=row["loss_curve"],
        manifest_sha256=row["manifest_sha256"],
        generated_samples=row["generated_samples"],
        objectives=row["objectives"],
        timestamp=row["timestamp"],
        task=row["task"],
        round_id=row["round_id"] if row["round_id"] is not None else -1,
    )


def element_to_params(element: DataElement, next_id: int) -> tuple:
    """Convert a DataElement to a positional tuple for INSERT ($1..$20)."""
    round_id = element.round_id if element.round_id >= 0 else None
    return (
        next_id,
        element.name,
        element.code,
        element.motivation,
        element.trace,
        element.metric,
        element.success,
        element.analysis,
        element.parent,
        element.generation,
        element.score,
        element.miner_uid,
        element.miner_hotkey,
        element.loss_curve,
        element.manifest_sha256,
        element.generated_samples,
        element.objectives,
        element.timestamp,
        round_id,
        element.task,
    )


INSERT_SQL = """
INSERT INTO experiments (
    id, name, code, motivation, trace, metric, success, analysis,
    parent_index, generation, score, miner_uid, miner_hotkey,
    loss_curve, manifest_sha256, generated_samples, objectives,
    timestamp, round_id, task
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8,
    $9, $10, $11, $12, $13,
    $14, $15, $16, $17,
    $18, $19, $20
)
"""


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
