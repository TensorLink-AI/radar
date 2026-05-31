"""SQLite → R2/Hippius backups for the validator's broker DB.

Two responsibilities:

1. **Restore on start.** If ``RADAR_BACKUP_BUCKET`` is set and the local
   SQLite file does not exist, pull ``<prefix>/latest.db.gz`` so the
   validator picks up where the last run left off (rounds, frontier,
   experiments).
2. **Periodic snapshot uploads.** A background thread takes a consistent
   snapshot via SQLite's online backup API, gzips it, and uploads two
   keys: an overwriting ``<prefix>/latest.db.gz`` (the "main" copy) and
   a timestamped ``<prefix>/snapshots/<UTC>.db.gz`` history entry.

Configuration is env-driven so the numpy-only path stays a no-op:

* ``RADAR_BACKUP_BUCKET``    — required; empty disables backups.
* ``RADAR_BACKUP_PREFIX``    — key prefix (default ``radar-backups``).
* ``RADAR_BACKUP_INTERVAL_SEC`` — seconds between snapshots
  (default ``3600``).

Credentials piggy-back on ``shared.r2_audit.HippiusStorage`` env
resolution (``HIPPIUS_*`` then ``R2_*``).
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

LATEST_KEY = "latest.db.gz"
SNAPSHOT_DIR = "snapshots"
DEFAULT_BUCKET = "radar-backups"
DEFAULT_PREFIX = "radar-backups"


def _has_s3_creds() -> bool:
    """True if HIPPIUS_* or R2_* access creds are set in the env."""
    return bool(
        (os.getenv("HIPPIUS_ACCESS_KEY_ID") and os.getenv("HIPPIUS_SECRET_ACCESS_KEY"))
        or (os.getenv("R2_ACCESS_KEY_ID") and os.getenv("R2_SECRET_ACCESS_KEY"))
    )


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_client(bucket: str):
    """Build a HippiusStorage client pinned to ``bucket``. Returns ``None``
    if boto3 / creds are unavailable so the validator can still run."""
    try:
        from shared.r2_audit import HippiusStorage
    except ImportError as e:
        logger.warning("backup disabled: %s", e)
        return None
    try:
        return HippiusStorage(bucket=bucket)
    except Exception as e:
        logger.warning("backup disabled: cannot init S3 client: %s", e)
        return None


def _snapshot_to_gzip(db_path: Path) -> bytes:
    """Take a consistent snapshot of ``db_path`` and return gzipped bytes.

    Uses ``sqlite3.Connection.backup`` against a fresh temp DB so WAL
    writes from the live validator process don't corrupt the copy.
    """
    with tempfile.TemporaryDirectory() as td:
        dst_path = Path(td) / "snapshot.db"
        src = sqlite3.connect(str(db_path), timeout=30)
        try:
            dst = sqlite3.connect(str(dst_path))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        raw = dst_path.read_bytes()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue()


class R2Backup:
    """Pull/push SQLite snapshots to R2/Hippius."""

    def __init__(
        self,
        db_path: str | Path,
        bucket: str,
        prefix: str = "radar-backups",
        interval_sec: float = 3600.0,
    ):
        self.db_path = Path(db_path)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.interval_sec = float(interval_sec)
        self._client = _make_client(bucket)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    # ── restore ────────────────────────────────────────────────
    def restore_if_missing(self) -> bool:
        """If the local DB is absent, try to download ``latest.db.gz``.

        Returns True if a restore happened, False otherwise. Existing
        local DBs are never overwritten.
        """
        if not self.enabled:
            return False
        if self.db_path.exists() and self.db_path.stat().st_size > 0:
            return False
        key = self._key(LATEST_KEY)
        if not self._client.key_exists(key):
            logger.info("backup: no remote %s; starting fresh DB", key)
            return False
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.db_path.parent, suffix=".db.gz.tmp", delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            ok = self._client.download_file_to_disk(key, str(tmp_path))
            if not ok:
                logger.warning("backup: download of %s failed; fresh DB", key)
                return False
            with gzip.open(tmp_path, "rb") as gz:
                raw = gz.read()
            self.db_path.write_bytes(raw)
            logger.info(
                "backup: restored %s → %s (%d bytes)",
                key, self.db_path, len(raw),
            )
            return True
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    # ── upload ─────────────────────────────────────────────────
    def upload_now(self) -> bool:
        """Take a snapshot and upload both ``latest`` and a timestamped key."""
        if not self.enabled:
            return False
        if not self.db_path.exists():
            logger.debug("backup: %s does not exist yet, skipping", self.db_path)
            return False
        try:
            blob = _snapshot_to_gzip(self.db_path)
        except Exception as e:
            logger.error("backup: snapshot failed: %s", e)
            return False
        stamp = _now_stamp()
        snap_key = self._key(f"{SNAPSHOT_DIR}/{stamp}.db.gz")
        latest_key = self._key(LATEST_KEY)
        ok_snap = self._put_bytes(snap_key, blob)
        ok_latest = self._put_bytes(latest_key, blob)
        if ok_snap and ok_latest:
            logger.info(
                "backup: uploaded %s (%d bytes) + %s",
                snap_key, len(blob), latest_key,
            )
        return ok_snap and ok_latest

    def _put_bytes(self, key: str, body: bytes) -> bool:
        try:
            self._client._s3.put_object(
                Bucket=self.bucket, Key=key, Body=body,
            )
            return True
        except Exception as e:
            logger.error("backup: upload %s failed: %s", key, e)
            return False

    # ── periodic daemon ────────────────────────────────────────
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="r2-backup", daemon=True,
        )
        self._thread.start()
        logger.info(
            "backup: daemon started (bucket=%s prefix=%s interval=%.0fs)",
            self.bucket, self.prefix, self.interval_sec,
        )

    def stop(self, final: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if final and self.enabled:
            self.upload_now()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self.upload_now()
            except Exception as e:
                logger.error("backup: tick raised %s", e)


def from_env(db_path: str | Path) -> Optional[R2Backup]:
    """Construct an ``R2Backup`` from env, or ``None`` if disabled.

    Defaults to bucket ``radar-backups`` / prefix ``radar-backups`` whenever
    HIPPIUS/R2 creds are present, so a properly-configured operator gets
    backups for free. Set ``RADAR_BACKUP_DISABLE=1`` to opt out, or
    override ``RADAR_BACKUP_BUCKET`` / ``RADAR_BACKUP_PREFIX`` to retarget.
    """
    if os.getenv("RADAR_BACKUP_DISABLE", "").strip() in ("1", "true", "yes"):
        return None
    bucket = os.getenv("RADAR_BACKUP_BUCKET", "").strip()
    if not bucket:
        if not _has_s3_creds():
            return None
        bucket = DEFAULT_BUCKET
    prefix = os.getenv("RADAR_BACKUP_PREFIX", DEFAULT_PREFIX).strip()
    try:
        interval = float(os.getenv("RADAR_BACKUP_INTERVAL_SEC", "3600"))
    except ValueError:
        interval = 3600.0
    backup = R2Backup(
        db_path=db_path, bucket=bucket, prefix=prefix, interval_sec=interval,
    )
    if not backup.enabled:
        return None
    return backup
