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
