"""Cognition wiki: per-task markdown corpus for miner agents.

Each task ships a markdown reference corpus (architectural notes, prior-art
summaries, recipe hints) bundled as a single ``wiki.tar.gz`` on R2/Hippius
under ``<prefix>/<task_name>/wiki.tar.gz`` (default prefix
``cognition_wiki/v2``).

In the distributed stack the validator presigns one GET URL per round and
attaches it to ``challenge.cognition_wiki_url`` so the agent fetches the
tarball through the ``GatedClient``. The local single-laptop stack instead
downloads the tarball once on validator startup, untars it into a cache
dir, and points the in-process ``WikiStore`` at the result — the agent
still sees ``cognition_wiki_url = {services_url}/wiki``, so the miner side
is identical in both deployments.

Tarballs are pre-built and uploaded out-of-band — this module only reads.
"""

from __future__ import annotations

import logging
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from shared.r2_audit import HippiusStorage

logger = logging.getLogger(__name__)


DEFAULT_PREFIX = "cognition_wiki/v2"
DEFAULT_BUCKET = "radar-cognition-wiki"
DEFAULT_CACHE_DIR = "/tmp/radar_cognition_wiki"

# Tasks pick this name themselves via task.name; restrict to filesystem-safe
# characters so the R2 key can't be steered by a poisoned task spec.
_SAFE_TASK = re.compile(r"^[A-Za-z0-9_.-]+$")


def wiki_key(task_name: str, prefix: str = DEFAULT_PREFIX) -> str:
    """Build the R2 key for a task's wiki tarball.

    Raises ``ValueError`` if ``task_name`` contains unsafe characters.
    """
    if not task_name or not _SAFE_TASK.match(task_name):
        raise ValueError(f"Unsafe task_name for wiki key: {task_name!r}")
    base = prefix.rstrip("/")
    if base:
        return f"{base}/{task_name}/wiki.tar.gz"
    return f"{task_name}/wiki.tar.gz"


def wiki_bucket() -> str:
    """Resolve the cognition-wiki bucket from env, with a sensible default."""
    return (
        os.environ.get("RADAR_COGNITION_WIKI_BUCKET")
        or os.environ.get("HIPPIUS_COGNITION_WIKI_BUCKET")
        or os.environ.get("R2_COGNITION_WIKI_BUCKET")
        or DEFAULT_BUCKET
    )


def wiki_prefix() -> str:
    return os.environ.get("RADAR_COGNITION_WIKI_PREFIX", DEFAULT_PREFIX)


def wiki_cache_dir() -> str:
    return os.environ.get("RADAR_COGNITION_WIKI_CACHE", DEFAULT_CACHE_DIR)


def make_wiki_client(bucket: str = "") -> Optional["HippiusStorage"]:
    """Construct an S3 client pinned to the cognition-wiki bucket.

    Returns ``None`` if boto3 isn't installed or construction fails —
    callers should treat that as "wiki unavailable" and continue without
    a corpus.
    """
    bucket = bucket or wiki_bucket()
    if not bucket:
        return None
    try:
        from shared.r2_audit import HippiusStorage
    except ImportError as e:
        logger.warning("cognition-wiki: boto3 unavailable (%s)", e)
        return None
    try:
        return HippiusStorage(bucket=bucket)
    except Exception as e:
        logger.warning("cognition-wiki: failed to build S3 client: %s", e)
        return None


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tar`` into ``dest`` rejecting path traversal / abs paths.

    Python 3.12 added a ``data`` filter that does exactly this; we use it
    when present and fall back to a manual check otherwise so 3.11 still
    works.
    """
    dest_resolved = dest.resolve()
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
        return
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError:
            raise RuntimeError(f"unsafe tar member: {member.name!r}")
    tar.extractall(dest)


def ensure_wiki_cached(
    task_name: str,
    *,
    bucket: str = "",
    prefix: str = "",
    cache_dir: str = "",
    force: bool = False,
) -> Optional[Path]:
    """Download + extract the per-task wiki tarball into a local cache.

    Returns the directory containing the extracted markdown (suitable for
    ``WikiStore(root=...)``), or ``None`` when the feature is disabled /
    the tarball is unreachable. Idempotent: re-uses an existing cache dir
    that already contains ``*.md`` unless ``force=True``.
    """
    resolved_prefix = prefix or wiki_prefix()
    try:
        key = wiki_key(task_name, resolved_prefix)
    except ValueError as e:
        logger.warning("cognition-wiki: %s", e)
        return None

    # Namespace the cache by the prefix's tail (e.g. "v2") so bumping
    # RADAR_COGNITION_WIKI_PREFIX from v1 → v2 doesn't serve stale markdown
    # from an earlier extract.
    version = Path(resolved_prefix.rstrip("/")).name or "default"
    root = Path(cache_dir or wiki_cache_dir()) / version / task_name
    if not force and root.is_dir() and any(root.rglob("*.md")):
        logger.debug("cognition-wiki: cache hit at %s", root)
        return root

    r2 = make_wiki_client(bucket)
    if r2 is None:
        return None

    if not r2.key_exists(key):
        logger.info(
            "cognition-wiki: no tarball at bucket=%s key=%s — skipping",
            r2.bucket, key,
        )
        return None

    # Download into a temp file first so a partial transfer can't poison
    # the cache, then extract atomically into ``root``.
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        if not r2.download_file_to_disk(key, tmp_path):
            logger.warning(
                "cognition-wiki: download failed bucket=%s key=%s",
                r2.bucket, key,
            )
            return None
        size = os.path.getsize(tmp_path)
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp_path, "r:gz") as tar:
            _safe_extract(tar, root)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    n_md = sum(1 for _ in root.rglob("*.md"))
    logger.info(
        "cognition-wiki: extracted %s (%d bytes, %d md files) → %s",
        key, size, n_md, root,
    )
    return root
