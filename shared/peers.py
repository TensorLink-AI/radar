"""Static peer registry — replacement for the removed on-chain peer set.

Peers are loaded from a JSON file referenced by ``MINERS_CONFIG_PATH``
(default ``./miners.json``). The file schema is:

    {
      "miners": [
        {"uid": 0, "hotkey": "miner0", "endpoint": "http://miner0:8000", "stake": 1.0},
        ...
      ]
    }

If the file is missing or malformed we log a warning and return an empty
peer list (so callers degrade gracefully in dev / tests).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MINERS_CONFIG_PATH = "./miners.json"


@dataclass
class Peer:
    """A single peer entry from miners.json."""
    uid: int
    hotkey: str
    endpoint: str = ""
    stake: float = 0.0


_cache_lock = threading.Lock()
_cache: list[Peer] = []
_cache_path: str = ""
_cache_mtime: float = 0.0


def _config_path() -> str:
    return os.getenv("MINERS_CONFIG_PATH", DEFAULT_MINERS_CONFIG_PATH)


def _read_peers_from_disk(path: str) -> list[Peer]:
    """Read peers from a JSON file. Returns [] on any error."""
    if not os.path.isfile(path):
        logger.warning(
            "MINERS_CONFIG_PATH=%s does not exist — using empty peer list",
            path,
        )
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load miners config %s: %s", path, e)
        return []

    raw = data.get("miners") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        logger.warning(
            "miners.json missing 'miners' list (got %s); using empty peer list",
            type(raw).__name__,
        )
        return []

    peers: list[Peer] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            peers.append(Peer(
                uid=int(entry["uid"]),
                hotkey=str(entry["hotkey"]),
                endpoint=str(entry.get("endpoint", "")),
                stake=float(entry.get("stake", 0.0)),
            ))
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Skipping malformed peer entry %r: %s", entry, e)
    return peers


def load_peers(force_reload: bool = False) -> list[Peer]:
    """Load peers from disk, caching the result by file mtime.

    Args:
        force_reload: Bypass the cache and re-read the file.

    Returns:
        List of ``Peer`` records (possibly empty).
    """
    global _cache, _cache_path, _cache_mtime
    path = _config_path()

    with _cache_lock:
        try:
            mtime = os.path.getmtime(path) if os.path.isfile(path) else 0.0
        except OSError:
            mtime = 0.0
        if (
            not force_reload
            and _cache_path == path
            and _cache_mtime == mtime
            and (_cache or not os.path.isfile(path))
        ):
            return list(_cache)

        peers = _read_peers_from_disk(path)
        _cache = peers
        _cache_path = path
        _cache_mtime = mtime
        return list(peers)


def get_peer_by_hotkey(hotkey: str) -> Optional[Peer]:
    """Return the peer record matching ``hotkey`` or ``None``."""
    if not hotkey:
        return None
    for p in load_peers():
        if p.hotkey == hotkey:
            return p
    return None


def get_peer_by_uid(uid: int) -> Optional[Peer]:
    """Return the peer record matching ``uid`` or ``None``."""
    for p in load_peers():
        if p.uid == uid:
            return p
    return None


def get_hotkey_for_uid(uid: int, default: Optional[str] = None) -> str:
    """Convenience helper used in place of the old hotkey lookup."""
    peer = get_peer_by_uid(uid)
    if peer:
        return peer.hotkey
    if default is None:
        return f"uid_{uid}"
    return default


def reset_cache() -> None:
    """Clear the in-memory peer cache (mainly for tests)."""
    global _cache, _cache_path, _cache_mtime
    with _cache_lock:
        _cache = []
        _cache_path = ""
        _cache_mtime = 0.0
