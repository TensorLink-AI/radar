"""Postgres schema DDL and row conversion helpers for PgExperimentStore."""

import difflib
import json
import math
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
    round_id BIGINT,
    task TEXT NOT NULL DEFAULT '',
    search_vector tsvector
);
-- Migration: round_id is derived from ``seed_int % 2**32`` in
-- ``shared.challenge.generate_challenge`` and can exceed INT32's 2.1B max.
-- Widen existing deployments to BIGINT; no-op if already BIGINT.
ALTER TABLE experiments ALTER COLUMN round_id TYPE BIGINT;
"""

SCHEMA_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_success ON experiments(success);
CREATE INDEX IF NOT EXISTS idx_metric ON experiments(metric) WHERE metric IS NOT NULL;
-- Drop legacy INT4-cast version of idx_flops if it exists; the CAST target
-- is widened to BIGINT below so miner-reported flops values > 2^31 don't
-- break INSERTs. The DROP is required because CREATE INDEX IF NOT EXISTS
-- won't update the indexed expression of an existing index.
DROP INDEX IF EXISTS idx_flops;
CREATE INDEX IF NOT EXISTS idx_flops ON experiments(
    CAST((objectives->>'flops_equivalent_size') AS BIGINT)
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
-- The IF NOT EXISTS guard must be scoped to the current schema's
-- experiments table, because tgname is unique only per-relation (NOT
-- globally). Under multi-network deployments (testnet + mainnet schemas in
-- the same database) an unqualified `tgname = 'experiments_search_trigger'`
-- lookup would match the OTHER schema's trigger and skip creation here,
-- leaving this schema's experiments table without the FTS trigger.
-- Casting 'experiments' to regclass resolves it via the current
-- search_path, so the check is schema-local.
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'experiments_search_trigger'
          AND tgrelid = 'experiments'::regclass
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
    round_id BIGINT NOT NULL,
    experiment_id INTEGER NOT NULL,
    context_type TEXT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);
ALTER TABLE round_context ALTER COLUMN round_id TYPE BIGINT;
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

PROXY_QUERY_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS proxy_query_log (
    id SERIAL PRIMARY KEY,
    service TEXT NOT NULL,
    miner_uid INTEGER NOT NULL DEFAULT -1,
    miner_hotkey TEXT NOT NULL DEFAULT '',
    query_text TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    response_summary TEXT NOT NULL DEFAULT '',
    tokens_used INTEGER NOT NULL DEFAULT 0,
    timestamp DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    round_id BIGINT NOT NULL DEFAULT -1
);
ALTER TABLE proxy_query_log ALTER COLUMN round_id TYPE BIGINT;
CREATE INDEX IF NOT EXISTS idx_pql_service ON proxy_query_log(service);
CREATE INDEX IF NOT EXISTS idx_pql_miner ON proxy_query_log(miner_hotkey);
CREATE INDEX IF NOT EXISTS idx_pql_round ON proxy_query_log(round_id);
CREATE INDEX IF NOT EXISTS idx_pql_timestamp ON proxy_query_log(timestamp);
"""

AGENT_CODE_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_submissions (
    id SERIAL PRIMARY KEY,
    hotkey TEXT NOT NULL,
    miner_uid INTEGER NOT NULL DEFAULT -1,
    code_hash TEXT NOT NULL,
    entry_point TEXT NOT NULL DEFAULT 'agent.py',
    r2_key TEXT NOT NULL,
    round_submitted BIGINT NOT NULL DEFAULT -1,
    timestamp DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    UNIQUE(hotkey)
);
ALTER TABLE agent_submissions ALTER COLUMN round_submitted TYPE BIGINT;
CREATE INDEX IF NOT EXISTS idx_agent_hotkey ON agent_submissions(hotkey);
CREATE INDEX IF NOT EXISTS idx_agent_hash ON agent_submissions(code_hash);

-- Append-only history of every agent submission. Unlike agent_submissions
-- (one row per hotkey, overwritten on each upload), this preserves the full
-- timeline so we can answer "which exact bytes was miner X running at
-- round N?" — join miner_access_log entries to the most recent history
-- row where round_submitted <= round_id.
CREATE TABLE IF NOT EXISTS agent_submission_history (
    id SERIAL PRIMARY KEY,
    hotkey TEXT NOT NULL,
    miner_uid INTEGER NOT NULL DEFAULT -1,
    code_hash TEXT NOT NULL,
    entry_point TEXT NOT NULL DEFAULT 'agent.py',
    r2_key TEXT NOT NULL,
    round_submitted BIGINT NOT NULL DEFAULT -1,
    timestamp DOUBLE PRECISION NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_ash_hotkey ON agent_submission_history(hotkey);
CREATE INDEX IF NOT EXISTS idx_ash_hash ON agent_submission_history(code_hash);
CREATE INDEX IF NOT EXISTS idx_ash_round ON agent_submission_history(round_submitted);
CREATE INDEX IF NOT EXISTS idx_ash_hotkey_round
    ON agent_submission_history(hotkey, round_submitted DESC);
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
    round_id BIGINT NOT NULL DEFAULT -1
);
ALTER TABLE miner_access_log ALTER COLUMN round_id TYPE BIGINT;
CREATE INDEX IF NOT EXISTS idx_access_hotkey ON miner_access_log(hotkey);
CREATE INDEX IF NOT EXISTS idx_access_round ON miner_access_log(round_id);
CREATE INDEX IF NOT EXISTS idx_access_hotkey_round
    ON miner_access_log(hotkey, round_id);
"""


def _decode_jsonb(value, default):
    """Coerce a JSONB column value to a Python dict/list.

    asyncpg returns JSONB as a Python object only when a codec is registered
    on the connection (see ``shared.pg_store.create_pg_pool``). In
    environments where codec registration is skipped or silently fails
    (some managed Postgres poolers), JSONB is returned as ``str`` instead.
    This helper parses strings so downstream consumers always see a native
    container, which is critical for ``DataElement.to_api_dict`` and any
    ``.get(...)`` / ``.items()`` calls on ``objectives``.
    """
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return default


def _finite_or(value, default):
    """Replace NaN/Inf floats with ``default``.

    Postgres ``DOUBLE PRECISION`` stores NaN and +/-Inf without complaint,
    but those values poison downstream JSON serialisation (``json.dumps``
    emits the non-standard ``NaN``/``Infinity`` tokens that strict parsers
    reject) and misbehave in ``ORDER BY`` queries (NaN sorts greater than
    every real number).  Applied on both READ (``row_to_element``) and
    WRITE (``element_to_params``) to cover legacy rows already in the DB.
    """
    if value is None:
        return default
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return default
    return value


def row_to_element(row) -> DataElement:
    """Convert an asyncpg Record to a DataElement.

    JSONB columns (objectives, loss_curve, generated_samples) are decoded via
    ``_decode_jsonb`` so callers always get dict/list regardless of whether
    the asyncpg connection has a JSONB codec registered.
    """
    return DataElement(
        index=row["id"],
        name=row["name"],
        code=row["code"],
        motivation=row["motivation"],
        trace=row["trace"],
        metric=_finite_or(row["metric"], None),
        success=bool(row["success"]),
        analysis=row["analysis"],
        parent=row["parent_index"],
        generation=row["generation"],
        score=_finite_or(row["score"], 0.0),
        miner_uid=row["miner_uid"],
        miner_hotkey=row["miner_hotkey"],
        loss_curve=_decode_jsonb(row["loss_curve"], []),
        manifest_sha256=row["manifest_sha256"],
        generated_samples=_decode_jsonb(row["generated_samples"], []),
        objectives=_decode_jsonb(row["objectives"], {}),
        timestamp=row["timestamp"],
        task=row["task"],
        round_id=row["round_id"] if row["round_id"] is not None else -1,
    )


def _sanitize_for_json(obj):
    """Replace float inf/nan with None so json.dumps produces valid JSON.

    PostgreSQL's JSONB parser rejects JavaScript-style ``Infinity`` and
    ``NaN`` tokens that Python's ``json.dumps`` emits by default.
    """
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _jsonb(value) -> str:
    """Serialize a value to a valid JSON string for JSONB columns."""
    return json.dumps(_sanitize_for_json(value))


def element_to_params(element: DataElement, next_id: int) -> tuple:
    """Convert a DataElement to a positional tuple for INSERT ($1..$20).

    JSONB columns (loss_curve, generated_samples, objectives) are explicitly
    serialised with ``json.dumps`` so asyncpg sends a valid JSON string
    regardless of statement-cache or connection-pooler settings.  Float
    ``inf``/``nan`` are replaced with ``null`` to keep PostgreSQL happy.
    """
    round_id = element.round_id if element.round_id >= 0 else None
    return (
        next_id,
        element.name,
        element.code,
        element.motivation,
        element.trace,
        _finite_or(element.metric, None),
        element.success,
        element.analysis,
        element.parent,
        element.generation,
        _finite_or(element.score, 0.0),
        element.miner_uid,
        element.miner_hotkey,
        _jsonb(element.loss_curve),
        element.manifest_sha256,
        _jsonb(element.generated_samples),
        _jsonb(element.objectives),
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
