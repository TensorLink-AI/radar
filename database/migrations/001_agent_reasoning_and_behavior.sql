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

ALTER TABLE CURRENT_SCHEMA.experiments
    ADD COLUMN IF NOT EXISTS reasoning TEXT NOT NULL DEFAULT '';

ALTER TABLE CURRENT_SCHEMA.experiments
    ADD COLUMN IF NOT EXISTS tool_calls JSONB NOT NULL DEFAULT '[]';

ALTER TABLE CURRENT_SCHEMA.experiments
    ADD COLUMN IF NOT EXISTS agent_behavior JSONB NOT NULL DEFAULT '{}';
