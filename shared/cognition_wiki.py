"""Cognition wiki: per-task markdown corpus for miner agents.

Each task ships a markdown reference corpus (architectural notes, prior-art
summaries, recipe hints) bundled as a single ``wiki.tar.gz`` on R2 under
``<COGNITION_WIKI_R2_PREFIX>/<task_name>/wiki.tar.gz``.

The validator presigns one GET URL per round and attaches it to
``challenge.cognition_wiki_url``. The agent fetches the tarball via the
GatedClient, untars it, and reads markdown files locally. Keeping it to a
single URL avoids the per-file-presigned-URL TTL drift problem and gives
the agent atomic access to the whole corpus.

Tarballs are pre-built and uploaded out-of-band — this module only signs
URLs, it does not assemble or upload corpora.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from shared.r2_audit import R2AuditLog

logger = logging.getLogger(__name__)


# Tasks pick this name themselves via task.name; restrict to filesystem-safe
# characters so the R2 key can't be steered by a poisoned task spec.
_SAFE_TASK = re.compile(r"^[A-Za-z0-9_.-]+$")


def wiki_key(task_name: str, prefix: str) -> str:
    """Build the R2 key for a task's wiki tarball.

    Raises ValueError if task_name contains unsafe characters.
    """
    if not task_name or not _SAFE_TASK.match(task_name):
        raise ValueError(f"Unsafe task_name for wiki key: {task_name!r}")
    base = prefix.rstrip("/")
    if base:
        return f"{base}/{task_name}/wiki.tar.gz"
    return f"{task_name}/wiki.tar.gz"


def build_wiki_r2(
    *,
    bucket: str,
    account_id: str = "",
    access_key_id: str = "",
    secret_access_key: str = "",
) -> Optional["R2AuditLog"]:
    """Build an R2 client pinned to the cognition-wiki bucket.

    Returns None if the bucket is empty (feature disabled). Credentials
    fall back to the standard R2_* env vars used elsewhere.
    """
    if not bucket:
        return None
    try:
        from shared.r2_audit import R2AuditLog
        return R2AuditLog(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket=bucket,
        )
    except Exception as e:
        logger.warning("Failed to build cognition-wiki R2 client: %s", e)
        return None


def presign_wiki_url(
    r2: Optional["R2AuditLog"],
    task_name: str,
    *,
    prefix: str,
    ttl: int = 5400,
) -> str:
    """Return a presigned GET URL for a task's wiki tarball.

    Returns an empty string if r2 is None, the task name is unsafe, the
    tarball doesn't exist for this task, or signing fails. Empty string
    is the canonical "feature not active for this task" sentinel that
    Challenge serialisation already handles for sibling URL fields.
    """
    if r2 is None:
        return ""
    try:
        key = wiki_key(task_name, prefix)
    except ValueError as e:
        logger.warning("Skipping cognition-wiki URL: %s", e)
        return ""

    if not r2.key_exists(key):
        logger.debug(
            "No cognition-wiki tarball at bucket=%s key=%s — skipping",
            r2.bucket, key,
        )
        return ""

    url = r2.generate_presigned_get_url(key, ttl=ttl)
    if not url:
        logger.warning(
            "Failed to presign cognition-wiki URL for bucket=%s key=%s",
            r2.bucket, key,
        )
        return ""
    logger.info(
        "Presigned cognition-wiki tarball: task=%s key=%s ttl=%ds",
        task_name, key, ttl,
    )
    return url
