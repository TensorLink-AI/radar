-- Per-round opaque submission_id -> miner_hotkey reveal map.
--
-- Phase B dispatch now hides the architecture owner's hotkey from the
-- trainer-host (which is itself another miner's pod under cross-eval).
-- The validator mints a per-job ``submission_id`` (32 hex chars) at
-- dispatch time and uses it as the bucket path component:
--     round_{id}/submission_{sid}/checkpoint.safetensors
--     round_{id}/submission_{sid}/architecture.py
--     round_{id}/submission_{sid}/training_meta.json
--     round_{id}/submission_{sid}/stdout.log
-- Trainers never see the architecture owner's hotkey on the wire.
--
-- After Phase C closes, the dispatching validator POSTs the reveal map
-- to ``/round_submissions/reveal``. This table stores it so:
--   * the public dashboard can resolve historical artifact paths back
--     to the (miner_hotkey, miner_uid) pair for log/loss views;
--   * miners can independently confirm their own submission_id was the
--     one trained for any past round.
--
-- Idempotent upsert on (round_id, submission_id): repeated submissions
-- from the same dispatching validator are no-ops, and another
-- dispatcher can append entries for jobs it owned without conflict.
--
-- Bootstrap-order safety: matches the pattern in 003.

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
CREATE INDEX IF NOT EXISTS idx_rs_hotkey
    ON round_submissions(miner_hotkey);
