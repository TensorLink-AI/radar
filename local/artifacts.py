"""Per-round artifact mirror — SQLite (always) + S3/Hippius/R2 (optional).

Every artifact a round produces is recorded in the ``artifacts`` table of
the local SQLite store so miners can ``GET /artifacts...`` against the
validator's services server without needing R2 credentials. Small text
artifacts (challenge / proposal / submission / result / logs) are kept
inline in ``artifacts.content_text``; binary checkpoints only carry the
``s3_key`` reference and are streamed back through ``/artifacts/{id}/
download`` when needed.

Bucket layout (``RADAR_ARTIFACT_BUCKET``, default ``radar-local``)::

    runs/{task}/r{round_id:06d}/
        challenge.json
        miners/{miner_id}/
            proposal.json
            submission.py
            result.json
            checkpoints/...
            logs/...
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from local.store import ARTIFACT_TEXT_KINDS, LocalStore


logger = logging.getLogger(__name__)


_DEFAULT_BUCKET = "radar-local"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cap on the size of any artifact we inline into SQLite as text. Anything
# bigger lives only on R2 (or on the validator's local workdir until
# upload). 1 MiB easily covers code, JSON, and short logs.
_INLINE_TEXT_LIMIT = 1 * 1024 * 1024


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (no python-dotenv dep). Existing env wins."""
    if not path.is_file():
        return
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if not key or key in os.environ:
                continue
            val = val.strip()
            if (len(val) >= 2) and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            os.environ[key] = val
    except OSError as e:
        logger.warning("could not read %s: %s", path, e)


_load_dotenv(_PROJECT_ROOT / ".env")


_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(component: str) -> str:
    """Sanitise a miner_id / task name for use as an S3 key segment."""
    return _SAFE_ID.sub("_", component).strip("_") or "unknown"


def _have_creds() -> bool:
    return bool(
        os.environ.get("HIPPIUS_ACCESS_KEY_ID")
        or os.environ.get("R2_ACCESS_KEY_ID")
    )


@dataclass
class ArtifactSink:
    """Per-round artifact recorder.

    ``store`` is required — every artifact is indexed in SQLite. ``r2_enabled``
    controls whether we also mirror to object storage.
    """

    store: LocalStore
    bucket: str = _DEFAULT_BUCKET
    r2_enabled: bool = False
    _client: object = field(default=None, repr=False)

    @classmethod
    def from_env(cls, store: LocalStore) -> "ArtifactSink":
        bucket = os.environ.get("RADAR_ARTIFACT_BUCKET", _DEFAULT_BUCKET).strip() \
            or _DEFAULT_BUCKET
        if not _have_creds():
            logger.info(
                "artifact sink: SQLite-only (no HIPPIUS_*/R2_* creds in env)"
            )
            return cls(store=store, bucket=bucket, r2_enabled=False)
        try:
            from shared.r2_audit import HippiusStorage
            client = HippiusStorage(bucket=bucket)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "artifact sink: SQLite-only (R2 init failed: %s)", e
            )
            return cls(store=store, bucket=bucket, r2_enabled=False)
        logger.info("artifact sink: SQLite + R2 (bucket=%s)", bucket)
        return cls(store=store, bucket=bucket, r2_enabled=True, _client=client)

    # ── key construction ────────────────────────────────────────────
    def _round_prefix(self, task_name: str, round_id: int) -> str:
        return f"runs/{_safe(task_name)}/r{int(round_id):06d}"

    def _miner_prefix(self, task_name: str, round_id: int, miner_id: str) -> str:
        return f"{self._round_prefix(task_name, round_id)}/miners/{_safe(miner_id)}"

    # ── core recorder ───────────────────────────────────────────────
    def _record_text(
        self,
        *,
        task_name: str,
        round_id: int,
        miner_id: str,
        kind: str,
        rel_path: str,
        s3_key: str,
        text: str,
    ) -> None:
        encoded = text.encode("utf-8")
        size = len(encoded)
        inline = text if size <= _INLINE_TEXT_LIMIT else None
        self.store.add_artifact(
            round_id=round_id,
            miner_id=miner_id,
            task=task_name,
            kind=kind,
            rel_path=rel_path,
            bucket=self.bucket if self.r2_enabled else "",
            s3_key=s3_key if self.r2_enabled else "",
            size_bytes=size,
            content_text=inline,
        )
        if self.r2_enabled:
            self._client.upload_text(s3_key, text)  # type: ignore[union-attr]

    def _record_binary_file(
        self,
        *,
        task_name: str,
        round_id: int,
        miner_id: str,
        kind: str,
        rel_path: str,
        s3_key: str,
        local_path: Path,
    ) -> None:
        try:
            size = local_path.stat().st_size
        except OSError:
            size = 0
        bucket = self.bucket if self.r2_enabled else ""
        key = s3_key if self.r2_enabled else ""
        # Only record binary rows when we have somewhere they live (R2).
        # Without R2 the file is in /tmp and gets cleaned up — there'd
        # be nothing for a miner to fetch.
        if not self.r2_enabled:
            return
        ok = self._client.upload_file_from_disk(str(local_path), s3_key)  # type: ignore[union-attr]
        if not ok:
            return
        self.store.add_artifact(
            round_id=round_id,
            miner_id=miner_id,
            task=task_name,
            kind=kind,
            rel_path=rel_path,
            bucket=bucket,
            s3_key=key,
            size_bytes=size,
            content_text=None,
        )

    # ── public API ──────────────────────────────────────────────────
    def record_challenge(self, task_name: str, round_id: int, challenge: dict) -> None:
        prefix = self._round_prefix(task_name, round_id)
        self._record_text(
            task_name=task_name,
            round_id=round_id,
            miner_id="",
            kind="challenge",
            rel_path="challenge.json",
            s3_key=f"{prefix}/challenge.json",
            text=json.dumps(challenge, indent=2),
        )

    def record_proposal(
        self, task_name: str, round_id: int, miner_id: str, payload: dict,
    ) -> None:
        base = self._miner_prefix(task_name, round_id, miner_id)
        self._record_text(
            task_name=task_name,
            round_id=round_id,
            miner_id=miner_id,
            kind="proposal",
            rel_path="proposal.json",
            s3_key=f"{base}/proposal.json",
            text=json.dumps(payload, indent=2),
        )
        code = payload.get("code", "")
        if isinstance(code, str) and code:
            self._record_text(
                task_name=task_name,
                round_id=round_id,
                miner_id=miner_id,
                kind="submission",
                rel_path="submission.py",
                s3_key=f"{base}/submission.py",
                text=code,
            )

    def record_result(
        self,
        task_name: str,
        round_id: int,
        miner_id: str,
        result: dict,
        workdir: Optional[Path] = None,
    ) -> None:
        base = self._miner_prefix(task_name, round_id, miner_id)
        # Drop the bulky code blob from the JSON — submission.py already has it.
        summary = {k: v for k, v in result.items() if k != "code"}
        self._record_text(
            task_name=task_name,
            round_id=round_id,
            miner_id=miner_id,
            kind="result",
            rel_path="result.json",
            s3_key=f"{base}/result.json",
            text=json.dumps(summary, indent=2),
        )
        if workdir is not None:
            self._upload_tree(
                task_name=task_name,
                round_id=round_id,
                miner_id=miner_id,
                local_dir=workdir / "logs",
                kind="log",
                key_prefix=f"{base}/logs",
                rel_prefix="logs",
                inline_text=True,
            )
            self._upload_tree(
                task_name=task_name,
                round_id=round_id,
                miner_id=miner_id,
                local_dir=workdir / "checkpoints",
                kind="checkpoint",
                key_prefix=f"{base}/checkpoints",
                rel_prefix="checkpoints",
                inline_text=False,
            )

    def _upload_tree(
        self,
        *,
        task_name: str,
        round_id: int,
        miner_id: str,
        local_dir: Path,
        kind: str,
        key_prefix: str,
        rel_prefix: str,
        inline_text: bool,
    ) -> None:
        if not local_dir.is_dir():
            return
        count = 0
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            rel_path = f"{rel_prefix}/{rel}"
            s3_key = f"{key_prefix}/{rel}"
            if inline_text:
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    self._record_binary_file(
                        task_name=task_name, round_id=round_id, miner_id=miner_id,
                        kind=kind, rel_path=rel_path, s3_key=s3_key, local_path=path,
                    )
                else:
                    self._record_text(
                        task_name=task_name, round_id=round_id, miner_id=miner_id,
                        kind=kind, rel_path=rel_path, s3_key=s3_key, text=text,
                    )
            else:
                self._record_binary_file(
                    task_name=task_name, round_id=round_id, miner_id=miner_id,
                    kind=kind, rel_path=rel_path, s3_key=s3_key, local_path=path,
                )
            count += 1
        if count:
            logger.info("recorded %d %s artifact(s) for miner=%s", count, kind, miner_id)

    # ── download helpers (used by the services HTTP layer) ──────────
    def fetch_bytes(self, s3_key: str) -> Optional[bytes]:
        """Stream an artifact body back from R2. ``None`` if unavailable."""
        if not self.r2_enabled or not s3_key:
            return None
        try:
            resp = self._client._s3.get_object(  # type: ignore[union-attr]
                Bucket=self.bucket, Key=s3_key,
            )
            return resp["Body"].read()
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch %s failed: %s", s3_key, e)
            return None


def cleanup_workdir(workdir: Optional[Path]) -> None:
    """Best-effort removal of a trainer workdir."""
    if workdir is None:
        return
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("could not remove workdir %s: %s", workdir, e)


__all__ = ["ArtifactSink", "cleanup_workdir"]
