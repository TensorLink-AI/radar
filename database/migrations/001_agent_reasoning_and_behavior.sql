-- Adds the columns that back the new "self-reported" + "validator-observed"
-- provenance tiers on experiments rows:
--
--   reasoning       — agent-authored prose explaining the design (capped
--                     at 256 KB before INSERT in validator/collection.py)
--   tool_calls      — agent-authored list of self-reported tool invocations
--                     ({"tool": "...", ...})
--   agent_behavior  — validator-observed: pod wall-clock, exit status, and
--                     per-category proxy call/error counters keyed by miner
--                     UID
--
-- All three default to empty values so legacy rows remain valid.
--
-- The whole block is guarded on `experiments` existing because the
-- migration runner is exercised against an empty schema in CI before
-- ``init_schema()`` runs (see workflows/schema-migrations.yml ::
-- "Migrations apply to empty DB"). On a fresh deploy the base
-- ``CREATE TABLE`` in ``shared/pg_schema.py`` already declares the new
-- columns, so the conditional makes this migration a no-op there. On
-- existing testnet/mainnet deployments ``init_schema()`` runs first and
-- the table exists, so the ALTERs apply normally.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'CURRENT_SCHEMA'
          AND table_name = 'experiments'
    ) THEN
        ALTER TABLE CURRENT_SCHEMA.experiments
            ADD COLUMN IF NOT EXISTS reasoning TEXT NOT NULL DEFAULT '';
        ALTER TABLE CURRENT_SCHEMA.experiments
            ADD COLUMN IF NOT EXISTS tool_calls JSONB NOT NULL DEFAULT '[]';
        ALTER TABLE CURRENT_SCHEMA.experiments
            ADD COLUMN IF NOT EXISTS agent_behavior JSONB NOT NULL DEFAULT '{}';
    END IF;
END $$;
