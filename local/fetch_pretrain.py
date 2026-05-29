"""Download GIFT-Eval pretrain shards from R2/Hippius into a local cache.

Wraps ``shared.pretrain_data.PretrainBenchmark`` so the local stack can
prefetch parquet shards before training. Requires ``boto3`` + ``pandas``
+ ``pyarrow`` (install via ``pip install -e .[gift_eval]``) and S3
credentials in the env (``HIPPIUS_*`` / ``R2_*``).

The pretrain bucket is **separate** from the gift-eval benchmark bucket:
it defaults to ``gift-eval-pretrain`` and can be overridden via
``RADAR_PRETRAIN_BUCKET`` (or ``HIPPIUS_PRETRAIN_BUCKET`` /
``R2_PRETRAIN_BUCKET``). S3 credentials and endpoint are shared with the
benchmark client.

Usage::

    python -m local.fetch_pretrain --list
    python -m local.fetch_pretrain --n 8                  # 8 training shards
    python -m local.fetch_pretrain --all
    python -m local.fetch_pretrain --val                  # held-out val shards
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


def _default_cache_dir() -> str:
    return os.environ.get("RADAR_PRETRAIN_CACHE", "/tmp/radar_pretrain")


def _default_val_cache_dir() -> str:
    return os.environ.get("RADAR_PRETRAIN_VAL_CACHE", "/tmp/radar_pretrain_val")


def _pretrain_bucket() -> str:
    """Resolve the pretrain bucket name from env. Independent of the
    benchmark bucket so operators can split read scopes per credential."""
    return (
        os.environ.get("RADAR_PRETRAIN_BUCKET")
        or os.environ.get("HIPPIUS_PRETRAIN_BUCKET")
        or os.environ.get("R2_PRETRAIN_BUCKET")
        or "gift-eval-pretrain"
    )


def make_pretrain_client():
    """Construct an ``R2AuditLog`` pointed at the pretrain bucket.

    Returns ``None`` if boto3 isn't installed — callers should treat
    that as "pretrain unavailable" and fall back to eval-only training.
    """
    try:
        from shared.r2_audit import R2AuditLog
    except ImportError:
        return None
    try:
        return R2AuditLog(bucket=_pretrain_bucket())
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Failed to construct pretrain R2 client: %s", e,
        )
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=0,
                   help="How many training shards to download (0 = all).")
    p.add_argument("--seed", type=int, default=42,
                   help="Selection seed when --n < total shard count.")
    p.add_argument("--all", action="store_true",
                   help="Download every training shard.")
    p.add_argument("--val", action="store_true",
                   help="Also download the val shard split (held out).")
    p.add_argument("--cache_dir", default=_default_cache_dir(),
                   help=f"Training-shard cache directory (default: {_default_cache_dir()}).")
    p.add_argument("--val_cache_dir", default=_default_val_cache_dir(),
                   help="Held-out val-shard cache directory — kept separate from "
                        "the training shards so the in-training val split is fixed "
                        f"across rounds (default: {_default_val_cache_dir()}).")
    p.add_argument("--list", action="store_true",
                   help="Print available shard keys and exit (no downloads).")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("local.fetch_pretrain")

    r2 = make_pretrain_client()
    if r2 is None:
        log.error("No S3 client (install boto3 and set HIPPIUS_*/R2_* env vars)")
        return 2

    try:
        from shared.pretrain_data import PretrainBenchmark
    except ImportError as e:
        log.error("shared.pretrain_data unavailable: %s", e)
        return 2

    bench = PretrainBenchmark(r2=r2)
    all_keys = bench.get_shard_keys()
    val_keys = bench.get_val_shard_keys()

    if not all_keys:
        log.error(
            "Pretrain manifest is empty or unreachable "
            "(bucket=%s prefix=%s). Check creds + bucket name.",
            r2.bucket, bench.r2_prefix,
        )
        return 1

    if args.list:
        print(f"# bucket={r2.bucket} prefix={bench.r2_prefix}")
        print(f"# {len(all_keys)} training shards, {len(val_keys)} val shards")
        for k in all_keys:
            print(k)
        if val_keys:
            print("# val shards:")
            for k in val_keys:
                print(k)
        return 0

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # (key, destination_dir) pairs. Training shards land in the train cache;
    # val shards land in their own dir so the trainer's fixed held-out split
    # never overlaps the round-varying training shards.
    targets: list[tuple[str, Path]] = []
    if args.all or args.n <= 0:
        targets.extend((k, cache) for k in all_keys)
    else:
        targets.extend((k, cache) for k in bench.select_shards(seed=args.seed, n=args.n))
    if args.val:
        if not val_keys:
            log.warning("--val requested but the manifest declares no val shards")
        else:
            val_cache = Path(args.val_cache_dir)
            val_cache.mkdir(parents=True, exist_ok=True)
            targets.extend((k, val_cache) for k in val_keys)

    ok = fail = skipped = 0
    for key, dest in targets:
        local_path = dest / Path(key).name
        if local_path.exists() and local_path.stat().st_size > 0:
            skipped += 1
            continue
        if r2.download_file_to_disk(key, str(local_path)):
            ok += 1
            log.info("downloaded %s (%d bytes)", local_path.name, local_path.stat().st_size)
        else:
            fail += 1
            log.warning("failed to download %s", key)

    print(f"\n{ok} downloaded, {skipped} cached, {fail} failed "
          f"(cache_dir={cache}, val_cache_dir={args.val_cache_dir})")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
