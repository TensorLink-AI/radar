"""Async Postgres access logger.

Same public API as AccessLogger but async, backed by asyncpg pool.
"""

import logging
import time
from typing import Optional

import asyncpg

from shared.pg_schema import ACCESS_LOG_SCHEMA

logger = logging.getLogger(__name__)


class PgAccessLogger:
    """Append-only async log of miner API access."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self._round_id: int = -1
        self._current_round: dict[str, set[int]] = {}

    async def init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute(ACCESS_LOG_SCHEMA)

    def set_round(self, round_id: int):
        """Call at round start. Resets in-memory per-round index."""
        self._round_id = round_id
        self._current_round = {}

    async def log_access(
        self,
        hotkey: str,
        miner_uid: int,
        endpoint: str,
        experiment_ids: list[int],
        method: str = "GET",
    ):
        """Append one access record."""
        import json
        # Coerce to list so the JSONB column always stores an array — scalars
        # would break `jsonb_array_elements_text` on the read side.
        ids_list = list(experiment_ids) if experiment_ids else []
        now = time.time()
        await self.pool.execute(
            "INSERT INTO miner_access_log "
            "(hotkey, miner_uid, endpoint, method, experiment_ids, "
            " timestamp, round_id) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7)",
            hotkey, miner_uid, endpoint, method,
            json.dumps(ids_list), now, self._round_id,
        )
        self._current_round.setdefault(hotkey, set()).update(ids_list)

    async def log_request(
        self,
        hotkey: str,
        endpoint: str,
        method: str = "GET",
        response_data=None,
        miner_uid: int = -1,
    ):
        """Convenience: log a request and auto-extract experiment IDs."""
        from shared.access_logger import _extract_experiment_ids
        exp_ids = _extract_experiment_ids(response_data) if response_data else []
        await self.log_access(hotkey, miner_uid, endpoint, exp_ids, method)

    async def get_accessed(self, hotkey: str, round_id: Optional[int] = None) -> set[int]:
        """Which experiments did this miner's agent see?"""
        rid = round_id if round_id is not None else self._round_id
        if rid == self._round_id and hotkey in self._current_round:
            return set(self._current_round[hotkey])

        rows = await self.pool.fetch(
            "SELECT experiment_ids FROM miner_access_log "
            "WHERE hotkey = $1 AND round_id = $2",
            hotkey, rid,
        )
        ids: set[int] = set()
        for row in rows:
            exp_ids = row["experiment_ids"]
            if isinstance(exp_ids, list):
                ids.update(exp_ids)
        return ids

    async def get_round_access(self, round_id: int) -> dict[str, set[int]]:
        """All miners' access for a round."""
        rows = await self.pool.fetch(
            "SELECT hotkey, experiment_ids FROM miner_access_log "
            "WHERE round_id = $1",
            round_id,
        )
        result: dict[str, set[int]] = {}
        for row in rows:
            exp_ids = row["experiment_ids"]
            if isinstance(exp_ids, list):
                result.setdefault(row["hotkey"], set()).update(exp_ids)
        return result
