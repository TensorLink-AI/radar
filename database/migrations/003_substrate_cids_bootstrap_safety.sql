-- Bootstrap-order safety net for substrate_cids.
--
-- Problem this records: migration 002_substrate_cids.sql adds the
-- substrate_cids column + GIN index, but the migration runner is
-- invoked AFTER ``PgExperimentStore.init_schema()`` in
-- ``database/neuron.py``. ``init_schema`` runs SCHEMA_INDEX_DDL from
-- ``shared/pg_schema.py``, which contains
--     CREATE INDEX IF NOT EXISTS idx_substrate_cids
--         ON experiments USING GIN(substrate_cids);
-- On legacy deployments that booted before TEN-240 the column does not
-- yet exist when SCHEMA_INDEX_DDL runs, so init_schema crashes with
-- ``UndefinedColumnError: column "substrate_cids" does not exist``
-- before migration 002 ever gets a chance to apply.
--
-- The crash is fixed in ``shared/pg_schema.py`` by an idempotent
-- ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS substrate_cids`` placed in
-- SCHEMA_TABLE_DDL (executed before SCHEMA_INDEX_DDL within the same
-- ``init_schema`` call). This file is the migration counterpart of that
-- fix: an idempotent re-assertion that the column + index exist, so any
-- replica restored from a snapshot taken between TEN-240 and this fix
-- catches up without manual intervention.
--
-- Same table-existence guard as the other migrations: on a fresh deploy
-- the migration runner runs against an empty schema (CI's "Migrations
-- apply to empty DB" job), so the body must be a no-op when the
-- experiments table doesn't exist yet.

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
