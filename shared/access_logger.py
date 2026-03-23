"""Append-only log of every DB API call miners make.

Stores raw access facts in the same SQLite DB as experiments.
No analytics, no inference — just facts. Query-time interpretation
belongs in ProvenanceQuery.
"""

import json
import logging
import sqlite3
import time
from typing import Optional

logger = logging.getLogger(__name__)

ACCESS_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS miner_access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hotkey TEXT NOT NULL,
    miner_uid INTEGER NOT NULL DEFAULT -1,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT 'GET',
    experiment_ids TEXT NOT NULL DEFAULT '[]',
    timestamp REAL NOT NULL,
    round_id INTEGER NOT NULL DEFAULT -1
);
CREATE INDEX IF NOT EXISTS idx_access_hotkey ON miner_access_log(hotkey);
CREATE INDEX IF NOT EXISTS idx_access_round ON miner_access_log(round_id);
CREATE INDEX IF NOT EXISTS idx_access_hotkey_round
    ON miner_access_log(hotkey, round_id);
"""


def _extract_experiment_ids(response_data) -> list[int]:
    """Pull experiment IDs from any db_server response shape."""
    ids: set[int] = set()

    if isinstance(response_data, dict):
        for key in ("index", "root_index", "latest_index"):
            val = response_data.get(key)
            if isinstance(val, int):
                ids.add(val)

    elif isinstance(response_data, list):
        for item in response_data:
            if isinstance(item, dict):
                for key in ("index", "root_index", "latest_index"):
                    val = item.get(key)
                    if isinstance(val, int):
                        ids.add(val)

    return sorted(ids)


class AccessLogger:
    """Append-only log of miner API access. Just stores facts."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._round_id: int = -1
        # In-memory fast path: hotkey -> set of experiment IDs for current round
        self._current_round: dict[str, set[int]] = {}
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(ACCESS_LOG_SCHEMA)
        self.conn.commit()

    def set_round(self, round_id: int):
        """Call at round start. Resets in-memory per-round index."""
        self._round_id = round_id
        self._current_round = {}

    def log_access(
        self,
        hotkey: str,
        miner_uid: int,
        endpoint: str,
        experiment_ids: list[int],
        method: str = "GET",
    ):
        """Append one access record."""
        now = time.time()
        self.conn.execute(
            "INSERT INTO miner_access_log "
            "(hotkey, miner_uid, endpoint, method, experiment_ids, "
            " timestamp, round_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (hotkey, miner_uid, endpoint, method,
             json.dumps(experiment_ids), now, self._round_id),
        )
        self.conn.commit()

        # Update in-memory index for current round
        self._current_round.setdefault(hotkey, set()).update(experiment_ids)

    def log_request(
        self,
        hotkey: str,
        endpoint: str,
        method: str = "GET",
        response_data=None,
        miner_uid: int = -1,
    ):
        """Convenience: log a request and auto-extract experiment IDs."""
        exp_ids = _extract_experiment_ids(response_data) if response_data else []
        self.log_access(hotkey, miner_uid, endpoint, exp_ids, method)

    def get_accessed(self, hotkey: str, round_id: Optional[int] = None) -> set[int]:
        """Which experiments did this miner's agent see?

        Fast path: in-memory dict for current round.
        Falls back to SQL for historical rounds.
        """
        rid = round_id if round_id is not None else self._round_id
        if rid == self._round_id and hotkey in self._current_round:
            return set(self._current_round[hotkey])

        rows = self.conn.execute(
            "SELECT experiment_ids FROM miner_access_log "
            "WHERE hotkey = ? AND round_id = ?",
            (hotkey, rid),
        ).fetchall()
        ids: set[int] = set()
        for row in rows:
            ids.update(json.loads(row[0]))
        return ids

    def get_round_access(self, round_id: int) -> dict[str, set[int]]:
        """All miners' access for a round."""
        rows = self.conn.execute(
            "SELECT hotkey, experiment_ids FROM miner_access_log "
            "WHERE round_id = ?",
            (round_id,),
        ).fetchall()
        result: dict[str, set[int]] = {}
        for row in rows:
            result.setdefault(row[0], set()).update(json.loads(row[1]))
        return result
