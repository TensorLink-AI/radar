"""Training log access via R2 (stdout + training_meta.json + architecture.py).

Wraps ``shared.r2_audit.R2AuditLog`` with a byte cap so the dashboard
never streams pathologically large stdout blobs over HTTP. Callers that
want the raw file use ``?direct=1`` to get a short-lived presigned URL.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


def _stdout_key(round_id: int, hotkey: str) -> str:
    return f"round_{round_id}/miner_{hotkey}/stdout.log"


def _meta_key(round_id: int, hotkey: str) -> str:
    return f"round_{round_id}/miner_{hotkey}/training_meta.json"


def _arch_key(round_id: int, hotkey: str) -> str:
    return f"round_{round_id}/miner_{hotkey}/architecture.py"


def fetch_meta(r2, round_id: int, hotkey: str) -> Optional[dict]:
    """Return ``training_meta.json`` for a (round, miner) or None if missing."""
    if r2 is None:
        return None
    try:
        return r2.download_json(_meta_key(round_id, hotkey))
    except Exception:
        logger.exception("fetch_meta failed: round=%s hotkey=%s", round_id, hotkey[:16])
        return None


async def cached_meta(pool, round_id: int, hotkey: str) -> Optional[dict]:
    """Return the Postgres-cached training_meta blob, or None if absent.

    Validators write here after Phase B so dashboard-mode deploys without
    R2 credentials can still serve loss curves.
    """
    if pool is None:
        return None
    try:
        row = await pool.fetchrow(
            "SELECT meta FROM training_metas WHERE round_id = $1 AND hotkey = $2",
            int(round_id), hotkey,
        )
    except Exception:
        logger.exception(
            "cached_meta query failed: round=%s hotkey=%s", round_id, hotkey[:16],
        )
        return None
    if row is None:
        return None
    from shared.pg_schema import _decode_jsonb
    meta = _decode_jsonb(row["meta"], None)
    return meta if isinstance(meta, dict) else None


async def fetch_meta_cached_or_r2(
    pool, r2, round_id: int, hotkey: str,
) -> Optional[dict]:
    """Postgres cache first, R2 fallback. None if neither has it."""
    meta = await cached_meta(pool, round_id, hotkey)
    if meta is not None:
        return meta
    return fetch_meta(r2, round_id, hotkey)


def fetch_stdout(
    r2,
    round_id: int,
    hotkey: str,
    max_bytes: Optional[int] = None,
) -> Optional[dict]:
    """Return ``{text, truncated, size}`` for the stdout file, or None.

    ``text`` is capped at ``max_bytes`` (defaults to
    ``Config.DASHBOARD_MAX_LOG_BYTES``). If the file is larger we take the
    tail so operators see the most recent lines.
    """
    if r2 is None:
        return None
    cap = max_bytes if max_bytes is not None else Config.DASHBOARD_MAX_LOG_BYTES
    try:
        raw = r2.download_text(_stdout_key(round_id, hotkey))
    except Exception:
        logger.exception("fetch_stdout failed: round=%s hotkey=%s", round_id, hotkey[:16])
        return None
    if raw is None:
        return None
    data = raw.encode("utf-8", errors="replace")
    size = len(data)
    if size > cap:
        # Keep the tail — most recent lines are the useful ones
        tail = data[-cap:].decode("utf-8", errors="replace")
        return {"text": tail, "truncated": True, "size": size}
    return {"text": raw, "truncated": False, "size": size}


def presigned_stdout_url(r2, round_id: int, hotkey: str, ttl: int = 900) -> str:
    """Return a 15-minute presigned GET URL for the stdout file."""
    if r2 is None:
        return ""
    try:
        return r2.generate_presigned_get_url(_stdout_key(round_id, hotkey), ttl=ttl)
    except Exception:
        logger.exception(
            "presigned_stdout_url failed: round=%s hotkey=%s", round_id, hotkey[:16],
        )
        return ""


def fetch_architecture(
    r2,
    round_id: int,
    hotkey: str,
    max_bytes: Optional[int] = None,
) -> Optional[dict]:
    """Return ``{text, truncated, size}`` for architecture.py, or None."""
    if r2 is None:
        return None
    cap = max_bytes if max_bytes is not None else Config.DASHBOARD_MAX_LOG_BYTES
    try:
        raw = r2.download_text(_arch_key(round_id, hotkey))
    except Exception:
        logger.exception(
            "fetch_architecture failed: round=%s hotkey=%s", round_id, hotkey[:16],
        )
        return None
    if raw is None:
        return None
    data = raw.encode("utf-8", errors="replace")
    size = len(data)
    if size > cap:
        head = data[:cap].decode("utf-8", errors="replace")
        return {"text": head, "truncated": True, "size": size}
    return {"text": raw, "truncated": False, "size": size}


def presigned_architecture_url(r2, round_id: int, hotkey: str, ttl: int = 900) -> str:
    """Return a 15-minute presigned GET URL for architecture.py."""
    if r2 is None:
        return ""
    try:
        return r2.generate_presigned_get_url(_arch_key(round_id, hotkey), ttl=ttl)
    except Exception:
        logger.exception(
            "presigned_architecture_url failed: round=%s hotkey=%s",
            round_id, hotkey[:16],
        )
        return ""


__all__ = [
    "cached_meta",
    "fetch_architecture",
    "fetch_meta",
    "fetch_meta_cached_or_r2",
    "fetch_stdout",
    "presigned_architecture_url",
    "presigned_stdout_url",
]
