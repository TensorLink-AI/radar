"""Download the per-task cognition-wiki tarball into a local cache.

Wraps ``shared.cognition_wiki.ensure_wiki_cached`` so operators can
prefetch the markdown corpus before starting the validator (the
validator itself will also do this on startup when ``--wiki_dir`` is
empty). Requires ``boto3`` and ``HIPPIUS_*`` / ``R2_*`` credentials.

Usage::

    python -m local.fetch_cognition_wiki --task ts_forecasting
    python -m local.fetch_cognition_wiki --task ts_forecasting --force
    python -m local.fetch_cognition_wiki --list  # show resolved config
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from shared.cognition_wiki import (
        ensure_wiki_cached, wiki_bucket, wiki_cache_dir, wiki_key, wiki_prefix,
    )

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="ts_forecasting",
                   help="Task name — matches the validator's --task.")
    p.add_argument("--cache_dir", default="",
                   help=f"Override cache root (default: {wiki_cache_dir()}).")
    p.add_argument("--bucket", default="",
                   help=f"Override bucket (default: {wiki_bucket()}).")
    p.add_argument("--prefix", default="",
                   help=f"Override key prefix (default: {wiki_prefix()}).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the cache already has markdown.")
    p.add_argument("--list", action="store_true",
                   help="Print resolved config and exit without downloading.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("local.fetch_cognition_wiki")

    bucket = args.bucket or wiki_bucket()
    prefix = args.prefix or wiki_prefix()
    cache = args.cache_dir or wiki_cache_dir()
    key = wiki_key(args.task, prefix)

    if args.list:
        print(f"bucket    = {bucket}")
        print(f"prefix    = {prefix}")
        print(f"key       = {key}")
        print(f"cache_dir = {cache}/{args.task}")
        return 0

    cached = ensure_wiki_cached(
        args.task,
        bucket=bucket,
        prefix=prefix,
        cache_dir=cache,
        force=args.force,
    )
    if cached is None:
        log.error(
            "cognition-wiki unavailable for task=%s (bucket=%s key=%s). "
            "Check HIPPIUS_*/R2_* creds, bucket name, and that the tarball exists.",
            args.task, bucket, key,
        )
        return 1
    n_md = sum(1 for _ in cached.rglob("*.md"))
    print(f"ready: {cached} ({n_md} markdown files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
