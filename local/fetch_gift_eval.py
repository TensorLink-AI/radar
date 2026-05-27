"""Download GIFT-Eval Arrow datasets from R2/Hippius into a local cache.

Wraps ``shared.gift_eval.ensure_datasets_cached`` so the local stack can
prefetch real time-series data without standing up the full distributed
runner. Requires ``boto3`` + ``pyarrow`` (install via ``pip install -e
.[gift_eval]``) and credentials in the env (``R2_*`` or ``HIPPIUS_*``).

Usage::

    # Download a small subset by leaderboard name
    python -m local.fetch_gift_eval --datasets m4_hourly electricity/H

    # Or by manifest key
    python -m local.fetch_gift_eval --datasets m4_hourly electricity__H

    # Pull the full GIFT-Eval benchmark
    python -m local.fetch_gift_eval --all

    # Custom cache dir (default: $RADAR_GIFT_EVAL_CACHE or /tmp/radar_gift_eval)
    python -m local.fetch_gift_eval --datasets m4_daily --cache_dir ./data
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

from shared.gift_eval import (  # noqa: E402
    GIFT_EVAL_DATASETS,
    SHORT_DATASETS,
    ensure_datasets_cached,
)


def _default_cache_dir() -> str:
    return os.environ.get("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Leaderboard names ('electricity/H') or manifest keys "
             "('electricity__H'). Mutually exclusive with --all.",
    )
    p.add_argument("--all", action="store_true",
                   help="Download every GIFT-Eval dataset (~55 entries).")
    p.add_argument("--short", action="store_true",
                   help="Download the SHORT_DATASETS leaderboard subset.")
    p.add_argument("--cache_dir", default=_default_cache_dir(),
                   help=f"Local cache directory (default: {_default_cache_dir()}).")
    p.add_argument("--list", action="store_true",
                   help="Print the manifest and exit (no downloads).")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        for name in GIFT_EVAL_DATASETS:
            print(name)
        return 0

    if args.all:
        targets = list(GIFT_EVAL_DATASETS)
    elif args.short:
        targets = list(SHORT_DATASETS)
    elif args.datasets:
        targets = list(args.datasets)
    else:
        p.error("specify --datasets, --short, or --all")

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    status = ensure_datasets_cached(targets, cache_dir=args.cache_dir)

    ok = sum(1 for v in status.values() if v)
    fail = sum(1 for v in status.values() if not v)
    print(f"\n{ok} cached, {fail} failed (cache_dir={args.cache_dir})")
    if fail:
        for k, v in status.items():
            if not v:
                print(f"  FAIL  {k}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
