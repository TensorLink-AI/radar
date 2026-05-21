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
    reasoning TEXT NOT NULL DEFAULT '',
    tool_calls JSONB NOT NULL DEFAULT '[]',
    agent_behavior JSONB NOT NULL DEFAULT '{}',
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
    -- Substrate audit trail (TEN-240). One element per validator that
    -- published a Phase C Hippius bundle covering this experiment.
    substrate_cids JSONB NOT NULL DEFAULT '[]',
    search_vector tsvector
);
-- Migration: round_id is derived from ``seed_int % 2**32`` in
-- ``shared.challenge.generate_challenge`` and can exceed INT32's 2.1B max.
-- Widen existing deployments to BIGINT; no-op if already BIGINT.
ALTER TABLE experiments ALTER COLUMN round_id TYPE BIGINT;
-- Migration (TEN-240): substrate_cids was added to the CREATE TABLE above
-- after deployments already had an experiments table, so the IF NOT EXISTS
-- table-create is a no-op and the column is missing. Add it explicitly.
ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS substrate_cids JSONB NOT NULL DEFAULT '[]';
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
CREATE INDEX IF NOT EXISTS idx_substrate_cids ON experiments USING GIN(substrate_cids);
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

# The IF NOT EXISTS guard must be scoped to the current schema's experiments
# table because tgname is unique per-relation, not globally. Under multi-
# network deployments (testnet + mainnet schemas in the same database) an
# unqualified ``tgname = 'experiments_search_trigger'`` check matches the
# OTHER schema's trigger and silently skips creation here, leaving this
# schema's experiments without the FTS trigger. Casting ``'experiments'`` to
# ``regclass`` resolves it via the session's search_path, so each schema
# gets its own trigger.
FTS_TRIGGER_DDL = """
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

-- Content-addressed cache of agent bundle bytes. Keyed by code_hash so
-- repeated submissions of the same bundle (common when a miner re-submits
-- the same code across rounds) share a single row. Lets the public
-- dashboard JSON API serve bundles without an R2 round-trip — and, more
-- importantly, without R2 credentials at all, so dashboard-mode deploys
-- (Railway) don't need write access to the bucket.
CREATE TABLE IF NOT EXISTS agent_bundles (
    code_hash TEXT PRIMARY KEY,
    bundle JSONB NOT NULL,
    created_at DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now())
);

-- Cache of training_meta.json (loss histories, status, flops, …) keyed by
-- (round_id, hotkey). Validators write this after Phase B so the public
-- dashboard JSON API can serve training-loss curves without R2 access.
CREATE TABLE IF NOT EXISTS training_metas (
    round_id BIGINT NOT NULL,
    hotkey TEXT NOT NULL,
    meta JSONB NOT NULL,
    created_at DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    PRIMARY KEY (round_id, hotkey)
);
CREATE INDEX IF NOT EXISTS idx_tm_round ON training_metas(round_id);
CREATE INDEX IF NOT EXISTS idx_tm_hotkey ON training_metas(hotkey);

-- Post-Phase-C reveal of the opaque submission_id -> miner_hotkey map.
-- During Phase B the trainer-host only sees submission_id; once Phase C
-- closes, validators publish this mapping so:
--   1. miners can verify the submission they queued was the one trained
--   2. the public dashboard can resolve historical bucket paths
--      (``round_{id}/submission_{sid}/...``) back to miner identities
--      for loss-curve / log views.
CREATE TABLE IF NOT EXISTS round_submissions (
    round_id BIGINT NOT NULL,
    submission_id TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    miner_uid INTEGER NOT NULL DEFAULT -1,
    created_at DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    PRIMARY KEY (round_id, submission_id)
);
CREATE INDEX IF NOT EXISTS idx_rs_round_hotkey
    ON round_submissions(round_id, miner_hotkey);
CREATE INDEX IF NOT EXISTS idx_rs_hotkey ON round_submissions(miner_hotkey);
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

# Miner registry + bearer API keys for the non-competitive cutover.
# A ``miner_id`` is an opaque operator-issued identifier (UUID by default)
# that decouples the feedback API from the on-chain hotkey identity.
# During the dual-stack period a miner row carries both: ``hotkey`` is
# the bittensor SS58 (nullable, only set for chain-registered miners) and
# ``miner_id`` is the bearer-auth identifier scoped to /miners/me/*.
MINER_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS miners (
    miner_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    hotkey TEXT NOT NULL DEFAULT '',
    contact TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    revoked_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_miners_hotkey ON miners(hotkey)
    WHERE hotkey <> '';

CREATE TABLE IF NOT EXISTS miner_api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL,
    miner_id TEXT NOT NULL REFERENCES miners(miner_id) ON DELETE CASCADE,
    scope TEXT NOT NULL DEFAULT 'miner',
    label TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    last_used_at DOUBLE PRECISION,
    revoked_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_miner_api_keys_hash
    ON miner_api_keys(key_hash) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_miner_api_keys_miner
    ON miner_api_keys(miner_id);

-- ``prompt_id`` on experiments lets miners correlate Phase C scores
-- with the prompt variant that produced the architecture (drives the
-- GEPA / random_mutate feedback loop on the miner CLI).  Opaque to
-- the operator; only the issuing miner interprets it.
ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS prompt_id TEXT;
CREATE INDEX IF NOT EXISTS idx_experiments_prompt_id
    ON experiments(prompt_id) WHERE prompt_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_experiments_miner_prompt
    ON experiments(miner_hotkey, prompt_id)
    WHERE prompt_id IS NOT NULL;
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
    # New columns may be missing on rows produced by older schemas (e.g. a
    # query against a yet-to-migrate replica). asyncpg Records raise on
    # ``row[name]`` for missing keys, so probe via ``keys()``.
    row_keys = set(row.keys()) if hasattr(row, "keys") else set()
    reasoning = row["reasoning"] if "reasoning" in row_keys else ""
    tool_calls = (
        _decode_jsonb(row["tool_calls"], [])
        if "tool_calls" in row_keys else []
    )
    agent_behavior = (
        _decode_jsonb(row["agent_behavior"], {})
        if "agent_behavior" in row_keys else {}
    )
    substrate_cids = (
        _decode_jsonb(row["substrate_cids"], [])
        if "substrate_cids" in row_keys else []
    )
    prompt_id = (
        (row["prompt_id"] or "") if "prompt_id" in row_keys else ""
    )

    return DataElement(
        index=row["id"],
        name=row["name"],
        code=row["code"],
        motivation=row["motivation"],
        trace=row["trace"],
        reasoning=reasoning,
        tool_calls=tool_calls,
        agent_behavior=agent_behavior,
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
        substrate_cids=substrate_cids,
        prompt_id=prompt_id,
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
    """Convert a DataElement to a positional tuple for INSERT ($1..$25).

    JSONB columns (loss_curve, generated_samples, objectives, tool_calls,
    agent_behavior, substrate_cids) are explicitly serialised with
    ``json.dumps`` so asyncpg sends a valid JSON string regardless of
    statement-cache or connection-pooler settings.  Float ``inf``/``nan``
    are replaced with ``null`` to keep PostgreSQL happy.
    """
    round_id = element.round_id if element.round_id >= 0 else None
    prompt_id = getattr(element, "prompt_id", "") or None
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
        element.reasoning,
        _jsonb(element.tool_calls),
        _jsonb(element.agent_behavior),
        _jsonb(element.substrate_cids),
        prompt_id,
    )


INSERT_SQL = """
INSERT INTO experiments (
    id, name, code, motivation, trace, metric, success, analysis,
    parent_index, generation, score, miner_uid, miner_hotkey,
    loss_curve, manifest_sha256, generated_samples, objectives,
    timestamp, round_id, task,
    reasoning, tool_calls, agent_behavior, substrate_cids,
    prompt_id
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8,
    $9, $10, $11, $12, $13,
    $14, $15, $16, $17,
    $18, $19, $20,
    $21, $22, $23, $24,
    $25
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
