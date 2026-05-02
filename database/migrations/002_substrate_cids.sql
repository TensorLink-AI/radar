-- TEN-240: Phase C substrate audit trail.
--
-- Adds a per-experiment list of Hippius CIDs (one per validator that
-- published a Phase C bundle covering this row). The column is JSONB so
-- consumers can query into individual entries without a join, and the
-- GIN index makes "find every experiment a given validator published"
-- and "find every experiment backed by CID X" cheap.
--
-- Each element shape:
--     {"kind": "phase_c_record",
--      "validator_hotkey": "5G...",
--      "cid": "bafy...",
--      "round_id": 123}
--
-- Defaults to '[]' so legacy rows remain valid.
--
-- Guarded on ``experiments`` existing because the migration runner is
-- exercised against an empty schema in CI before init_schema() runs
-- (see workflows/schema-migrations.yml). On a fresh deploy the base
-- CREATE TABLE in shared/pg_schema.py already declares the column, so
-- this migration is a no-op there. On existing deployments init_schema()
-- runs first and the table exists, so the ALTER applies normally.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'CURRENT_SCHEMA'
          AND table_name = 'experiments'
    ) THEN
        ALTER TABLE CURRENT_SCHEMA.experiments
            ADD COLUMN IF NOT EXISTS substrate_cids JSONB NOT NULL DEFAULT '[]';
        CREATE INDEX IF NOT EXISTS idx_substrate_cids
            ON CURRENT_SCHEMA.experiments USING GIN (substrate_cids);
    END IF;
END $$;
