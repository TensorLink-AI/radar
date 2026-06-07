"""R2 → meta.db ingestion.

For each running/done unit, the orchestrator:

1. Downloads ``<backup_prefix>/<instance_id>/latest.db.gz`` from R2 via
   ``shared.r2_audit.HippiusStorage`` (the same client the validator
   uses to push backups).
2. Gunzips into ``meta/data/experiments/<exp>/units/<unit>/local_snapshot.db``.
   This file remains a valid ``radar_local.db`` so ``local.dashboard``
   can be pointed at it for the per-unit drilldown.
3. Opens it read-only and selects relevant ``experiments`` rows, then
   upserts a denormalized projection into the experiment's
   ``meta.db.frontier`` table.

Idempotent: ``frontier`` is keyed on ``(unit_id, source_exp_id)`` so
repeated ingestion never double-counts.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .store import ExperimentStore


logger = logging.getLogger(__name__)


# Default key shape mirrors ``local/backup.py`` (see CLAUDE.md):
#   <prefix>/<instance_id>/latest.db.gz
# Override at call time if the prefix differs.
DEFAULT_PREFIX = "radar-backups"
LATEST_KEY = "latest.db.gz"


def _hippius():
    """Return a ``HippiusStorage`` instance, or raise if creds missing."""
    from shared.r2_audit import HippiusStorage  # type: ignore
    return HippiusStorage()


def backup_key(prefix: str, instance_id: str) -> str:
    p = prefix.strip("/")
    return f"{p}/{instance_id}/{LATEST_KEY}" if p else f"{instance_id}/{LATEST_KEY}"


def fetch_latest_snapshot(
    instance_id: str,
    dest_db: Path,
    *,
    prefix: str = DEFAULT_PREFIX,
    bucket: str = "",
) -> bool:
    """Download + gunzip ``latest.db.gz`` for ``instance_id`` into
    ``dest_db``. Returns False if the key doesn't exist."""
    storage = _hippius()
    if bucket:
        storage.bucket = bucket
    key = backup_key(prefix, instance_id)
    dest_db.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".db.gz", delete=False) as f:
        tmp_gz = Path(f.name)
    try:
        ok = storage.download_file_to_disk(key, str(tmp_gz))
        if not ok or not tmp_gz.exists() or tmp_gz.stat().st_size == 0:
            return False
        # Atomic unpack into dest_db.
        with gzip.open(tmp_gz, "rb") as src, \
                tempfile.NamedTemporaryFile(
                    delete=False, dir=dest_db.parent, suffix=".db.tmp"
                ) as dst:
            shutil.copyfileobj(src, dst)
            tmp_db = Path(dst.name)
        tmp_db.replace(dest_db)
        return True
    finally:
        tmp_gz.unlink(missing_ok=True)


def _open_snapshot_ro(path: Path) -> sqlite3.Connection:
    # URI mode lets us open the file read-only even if WAL files aren't
    # present (the gunzipped checkpoint is a single .db file).
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


# Columns we need from ``experiments``. We keep this list explicit so a
# producer-side schema change (extra columns) doesn't crash the ingester.
_EXP_COLS = (
    "id", "round_id", "miner_id", "name", "metric", "success",
    "objectives_json", "task", "mode", "cumulative_compute",
)


def read_frontier_rows(snapshot_db: Path) -> list[dict[str, Any]]:
    """Pull the per-experiment rows we project into ``meta.db.frontier``."""
    try:
        conn = _open_snapshot_ro(snapshot_db)
    except sqlite3.OperationalError as e:
        logger.warning("ingest: cannot open %s: %s", snapshot_db, e)
        return []
    try:
        # Verify the table+columns exist before SELECTing, so a producer
        # running an older/newer radar release fails gracefully.
        cols_present = {
            r["name"] for r in conn.execute("PRAGMA table_info(experiments)")
        }
        if "id" not in cols_present:
            logger.warning(
                "ingest: %s has no experiments.id column; "
                "schema version unsupported",
                snapshot_db,
            )
            return []
        select = ", ".join(c for c in _EXP_COLS if c in cols_present)
        rows = list(conn.execute(f"SELECT {select} FROM experiments"))
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for r in rows:
        rd = dict(r)
        objectives_raw = rd.get("objectives_json", "{}") or "{}"
        try:
            objectives = json.loads(objectives_raw)
        except json.JSONDecodeError:
            objectives = {}
        out.append({
            "source_exp_id": int(rd["id"]),
            "round_id": int(rd.get("round_id", 0)),
            "miner_id": str(rd.get("miner_id", "")),
            "name": str(rd.get("name", "")),
            "metric": rd.get("metric"),
            "success": int(rd.get("success", 0) or 0),
            "objectives": objectives,
            "task": str(rd.get("task", "")),
            "mode": str(rd.get("mode", "new")),
            "cumulative_compute": float(rd.get("cumulative_compute", 0) or 0),
        })
    return out


def ingest_unit(
    store: ExperimentStore,
    *,
    unit_id: str,
    instance_id: str,
    snapshot_dir: Path,
    prefix: str = DEFAULT_PREFIX,
    bucket: str = "",
) -> int:
    """Fetch + ingest one unit's R2 snapshot. Returns rows upserted."""
    dest = snapshot_dir / "local_snapshot.db"
    ok = fetch_latest_snapshot(
        instance_id, dest, prefix=prefix, bucket=bucket
    )
    if not ok:
        store.log_event(
            "ingest_skip", unit_id=unit_id,
            payload={"reason": "no_snapshot", "instance_id": instance_id},
        )
        return 0
    rows = read_frontier_rows(dest)
    n = store.upsert_frontier_rows(unit_id, rows)
    store.log_event(
        "ingest_ok", unit_id=unit_id,
        payload={"rows": n, "instance_id": instance_id},
    )
    return n
