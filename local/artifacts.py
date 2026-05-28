"""Mirror each round's artifacts to S3/Hippius/R2.

Layout in the bucket (``RADAR_ARTIFACT_BUCKET``, default ``radar-local``)::

    runs/{task}/r{round_id:06d}/
        challenge.json
        miners/{miner_id}/
            proposal.json
            submission.py
            result.json
            checkpoints/...   # ts_forecasting only
            logs/...

The sink is a no-op when credentials are missing or ``boto3`` is not
installed, so the synthetic stack keeps working out of the box.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


_DEFAULT_BUCKET = "radar-local"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    """Round-artifact mirror. ``enabled=False`` means every call is a no-op."""

    bucket: str
    enabled: bool
    _client: object = None

    @classmethod
    def from_env(cls) -> "ArtifactSink":
        bucket = os.environ.get("RADAR_ARTIFACT_BUCKET", _DEFAULT_BUCKET).strip() \
            or _DEFAULT_BUCKET
        if not _have_creds():
            logger.info(
                "artifact sink disabled (no HIPPIUS_*/R2_* creds in env)"
            )
            return cls(bucket=bucket, enabled=False)
        try:
            from shared.r2_audit import HippiusStorage
            client = HippiusStorage(bucket=bucket)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "artifact sink disabled (could not init storage client: %s)", e
            )
            return cls(bucket=bucket, enabled=False)
        logger.info("artifact sink enabled: bucket=%s", bucket)
        return cls(bucket=bucket, enabled=True, _client=client)

    # ── key construction ────────────────────────────────────────────
    def _round_prefix(self, task_name: str, round_id: int) -> str:
        return f"runs/{_safe(task_name)}/r{int(round_id):06d}"

    def _miner_prefix(self, task_name: str, round_id: int, miner_id: str) -> str:
        return f"{self._round_prefix(task_name, round_id)}/miners/{_safe(miner_id)}"

    # ── uploads ─────────────────────────────────────────────────────
    def record_challenge(self, task_name: str, round_id: int, challenge: dict) -> None:
        if not self.enabled:
            return
        key = f"{self._round_prefix(task_name, round_id)}/challenge.json"
        self._client.upload_json(key, challenge)  # type: ignore[union-attr]

    def record_proposal(
        self, task_name: str, round_id: int, miner_id: str, payload: dict,
    ) -> None:
        if not self.enabled:
            return
        base = self._miner_prefix(task_name, round_id, miner_id)
        self._client.upload_json(f"{base}/proposal.json", payload)  # type: ignore[union-attr]
        code = payload.get("code", "")
        if isinstance(code, str) and code:
            self._client.upload_text(f"{base}/submission.py", code)  # type: ignore[union-attr]

    def record_result(
        self,
        task_name: str,
        round_id: int,
        miner_id: str,
        result: dict,
        workdir: Optional[Path] = None,
    ) -> None:
        if not self.enabled:
            return
        base = self._miner_prefix(task_name, round_id, miner_id)
        # Keep result.json self-contained but drop the redundant code blob —
        # submission.py already has it.
        summary = {k: v for k, v in result.items() if k != "code"}
        self._client.upload_json(f"{base}/result.json", summary)  # type: ignore[union-attr]
        if workdir is not None:
            self._upload_tree(workdir / "checkpoints", f"{base}/checkpoints")
            self._upload_tree(workdir / "logs", f"{base}/logs")

    def _upload_tree(self, local_dir: Path, key_prefix: str) -> None:
        if not local_dir.is_dir():
            return
        count = 0
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            key = f"{key_prefix}/{rel}"
            if self._client.upload_file_from_disk(str(path), key):  # type: ignore[union-attr]
                count += 1
        if count:
            logger.info("uploaded %d file(s) under %s/", count, key_prefix)


def cleanup_workdir(workdir: Optional[Path]) -> None:
    """Best-effort removal of a trainer workdir."""
    if workdir is None:
        return
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("could not remove workdir %s: %s", workdir, e)
