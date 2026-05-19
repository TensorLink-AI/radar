-- Miner registry + bearer API keys for the non-competitive cutover,
-- plus the ``experiments.prompt_id`` column that closes the GEPA
-- feedback loop on the miner CLI.
--
-- Non-competitive deployments use operator-issued bearer tokens
-- (rdrk_*) in place of Epistula/metagraph identity.  A ``miner_id``
-- is an opaque operator-issued identifier (UUID by default) that
-- decouples the feedback API from any on-chain hotkey.  During the
-- dual-stack period a miner row carries both: ``hotkey`` is the
-- bittensor SS58 (nullable, only set for chain-registered miners)
-- and ``miner_id`` is the bearer-auth identifier scoped to
-- /miners/me/*.
--
-- ``prompt_id`` on experiments lets miners correlate Phase C scores
-- with the prompt variant that produced the architecture.  Opaque
-- to the operator; only the issuing miner interprets it.  Two
-- partial indexes keep the table footprint sparse for rows without
-- a prompt_id (every existing row + every competitive-mode write).
--
-- All DDL is idempotent — safe under partial re-runs.  The
-- experiments ALTER + its indexes are wrapped in an information_schema
-- guard (same pattern as 002_substrate_cids.sql) because CI exercises
-- the migration runner against an empty schema before init_schema()
-- has bootstrapped the experiments table.  On a fresh deploy the base
-- CREATE TABLE in shared/pg_schema.py already declares prompt_id, so
-- the guarded block is a no-op there.

CREATE TABLE IF NOT EXISTS CURRENT_SCHEMA.miners (
    miner_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    hotkey TEXT NOT NULL DEFAULT '',
    contact TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    revoked_at DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_miners_hotkey
    ON CURRENT_SCHEMA.miners(hotkey)
    WHERE hotkey <> '';

CREATE TABLE IF NOT EXISTS CURRENT_SCHEMA.miner_api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL,
    miner_id TEXT NOT NULL REFERENCES CURRENT_SCHEMA.miners(miner_id)
        ON DELETE CASCADE,
    scope TEXT NOT NULL DEFAULT 'miner',
    label TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    last_used_at DOUBLE PRECISION,
    revoked_at DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_miner_api_keys_hash
    ON CURRENT_SCHEMA.miner_api_keys(key_hash)
    WHERE revoked_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_miner_api_keys_miner
    ON CURRENT_SCHEMA.miner_api_keys(miner_id);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'CURRENT_SCHEMA'
          AND table_name = 'experiments'
    ) THEN
        ALTER TABLE CURRENT_SCHEMA.experiments
            ADD COLUMN IF NOT EXISTS prompt_id TEXT;
        CREATE INDEX IF NOT EXISTS idx_experiments_prompt_id
            ON CURRENT_SCHEMA.experiments(prompt_id)
            WHERE prompt_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_experiments_miner_prompt
            ON CURRENT_SCHEMA.experiments(miner_hotkey, prompt_id)
            WHERE prompt_id IS NOT NULL;
    END IF;
END $$;
