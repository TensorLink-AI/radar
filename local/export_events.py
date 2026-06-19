"""Export agent_events rows to JSONL for fine-tuning / analysis.

Reads from the local SQLite store and writes one JSON object per line.
Each line is self-describing — kind, round_id, miner_id, request,
response, timing — so the file can be filtered, sharded, or rolled up
into SFT-style ``messages=[...]`` examples by a downstream script.

Optionally uploads the same JSONL blob to Hippius/R2 via
``shared.r2_audit.HippiusStorage`` so it lands in object storage for
later training pipelines. Credentials are picked up from env exactly
the way the rest of the radar stack reads them (``HIPPIUS_*`` first,
then ``R2_*``).

Usage:

    python -m local.export_events --out events.jsonl
    python -m local.export_events --out events.jsonl --round 7
    python -m local.export_events --out events.jsonl \\
        --r2-bucket radar-agent-logs --r2-key runs/2026-05-31.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

from local.store import LocalStore


DEFAULT_R2_PREFIX = "agent-events"


def _round_shard_key(prefix: str, round_id: int) -> str:
    return f"{prefix.rstrip('/')}/round={round_id:06d}.jsonl"


def flush_round_to_r2(
    store: LocalStore, round_id: int, bucket: str,
    *, prefix: str = DEFAULT_R2_PREFIX, delete_after: bool = True,
    retain_rounds: int = 0,
) -> tuple[bool, int]:
    """Upload one round's events to R2 as a single JSONL shard, then
    prune local rows. Returns ``(uploaded, n_events)``.

    Used by the validator's round-end hook. Failure to upload leaves the
    local rows in place so the next round (or a manual export) can retry
    — we never delete without a confirmed upload.

    ``retain_rounds`` keeps the most recent N rounds (including this one)
    in the local SQLite store so the dashboard's service-log tab still has
    something to show; only rounds older than the window are pruned (they
    were already uploaded in their own round, so they stay durable in R2).
    The default of 0 preserves the original behavior — prune this round's
    rows immediately after upload.
    """
    events = list(store.iter_agent_events(round_id=round_id))
    if not events:
        return True, 0

    try:
        from shared.r2_audit import HippiusStorage
    except ImportError as e:
        logger.error("R2 flush requires boto3: %s", e)
        return False, len(events)

    text = "\n".join(json.dumps(ev, default=str) for ev in events) + "\n"
    key = _round_shard_key(prefix, round_id)
    storage = HippiusStorage(bucket=bucket)
    if not storage.upload_text(key, text):
        logger.warning("agent-events upload failed for round=%d", round_id)
        return False, len(events)

    if delete_after:
        if retain_rounds > 0:
            # Keep the trailing window locally; prune anything older.
            cutoff = round_id - retain_rounds
            if cutoff >= 0:
                store.delete_agent_events(max_round=cutoff)
        else:
            store.delete_agent_events(round_id=round_id)
    logger.info(
        "flushed %d agent events for round=%d → s3://%s/%s",
        len(events), round_id, bucket, key,
    )
    return True, len(events)

logger = logging.getLogger("local.export_events")


def write_jsonl(events: Iterable[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, default=str))
            fh.write("\n")
            n += 1
    return n


def upload_jsonl(local_path: Path, bucket: str, key: str) -> bool:
    """Upload via HippiusStorage. Returns True on success."""
    try:
        from shared.r2_audit import HippiusStorage
    except ImportError as e:
        logger.error("R2 upload requires boto3: %s", e)
        return False
    storage = HippiusStorage(bucket=bucket)
    text = local_path.read_text(encoding="utf-8")
    return storage.upload_text(key, text)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="local/radar_local.db",
                        help="SQLite path")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--round", type=int, default=None,
                        help="Filter to a single round_id")
    parser.add_argument("--miner", default=None, help="Filter to one miner_id")
    parser.add_argument("--kind", default=None,
                        help="Filter to one event kind (llm_chat, desearch, "
                             "wiki_read, ...)")
    parser.add_argument("--since-id", type=int, default=0,
                        help="Only export rows with id > since-id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--r2-bucket", default="",
                        help="If set, upload the JSONL to this bucket")
    parser.add_argument("--r2-key", default="",
                        help="Object key for the upload (defaults to "
                             "agent-events/<out filename>)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(message)s")

    store = LocalStore(args.db)
    try:
        events = list(store.iter_agent_events(
            round_id=args.round, miner_id=args.miner, kind=args.kind,
            since_id=args.since_id, limit=args.limit,
        ))
    finally:
        store.close()

    out_path = Path(args.out)
    n = write_jsonl(events, out_path)
    logger.info("wrote %d events → %s", n, out_path)

    if args.r2_bucket:
        key = args.r2_key or f"agent-events/{out_path.name}"
        ok = upload_jsonl(out_path, args.r2_bucket, key)
        if not ok:
            logger.error("upload failed")
            return 1
        logger.info("uploaded → s3://%s/%s", args.r2_bucket, key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
