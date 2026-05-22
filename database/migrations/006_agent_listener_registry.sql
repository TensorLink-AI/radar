-- Off-chain miner listener registry.
--
-- The post-bittensor deploy has no on-chain ImageCommitment to carry a
-- miner's listener_url. Miners now self-register their URL via
-- POST /agent_code (carried as a new optional field) and refresh it
-- via POST /miners/me/listener. Validators discover live miners
-- through GET /miners/active using the listener_seen_at timestamp.
--
-- All DDL is idempotent. The block is wrapped in an
-- information_schema guard because the CI migration runner applies
-- this file against an empty schema before init_schema() has
-- bootstrapped the agent_submissions table — same pattern as
-- 005_miner_registry_and_prompt_id.sql. On a fresh deploy the base
-- CREATE TABLE in shared/pg_schema.py already declares the new
-- columns, so the guarded block is a no-op there.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'CURRENT_SCHEMA'
          AND table_name = 'agent_submissions'
    ) THEN
        ALTER TABLE CURRENT_SCHEMA.agent_submissions
            ADD COLUMN IF NOT EXISTS listener_url TEXT NOT NULL DEFAULT '';
        ALTER TABLE CURRENT_SCHEMA.agent_submissions
            ADD COLUMN IF NOT EXISTS listener_seen_at
                DOUBLE PRECISION NOT NULL DEFAULT 0.0;
        CREATE INDEX IF NOT EXISTS idx_agent_listener_seen
            ON CURRENT_SCHEMA.agent_submissions(listener_seen_at)
            WHERE listener_url <> '';
    END IF;
END $$;
