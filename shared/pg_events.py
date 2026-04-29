"""Async Postgres store for validator log/metric events.

Append-only stream backing the wandb-style live tail at
``/dashboard/api/validators/{hotkey}/events``. Validators batch events
in-memory and POST them to ``/events`` every few seconds; the dashboard
SPA polls the tail endpoint with a ``since_id`` cursor.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import asyncpg

from shared.pg_schema import VALIDATOR_EVENTS_SCHEMA, _decode_jsonb, _sanitize_for_json

logger = logging.getLogger(__name__)

# Cap an individual event's serialized payload. Anything larger gets
# truncated to a placeholder so a runaway log line can't blow out the
# JSONB column or the public response.
_MAX_PAYLOAD_BYTES = 16 * 1024

# Recognised event kinds. Anything else is rejected at insert time so the
# tail endpoint never has to defend against arbitrary type strings.
VALID_KINDS = frozenset({"log", "metric", "phase"})

# Recognised log levels (matches stdlib ``logging`` plus ``""`` for
# non-log kinds).
VALID_LEVELS = frozenset({"", "debug", "info", "warning", "error", "critical"})


def _normalise_payload(payload) -> dict:
    """Coerce payload to a JSON-safe dict, truncating if oversized."""
    if not isinstance(payload, dict):
        payload = {"value": payload}
    payload = _sanitize_for_json(payload)
    body = json.dumps(payload)
    if len(body.encode()) > _MAX_PAYLOAD_BYTES:
        return {
            "_truncated": True,
            "_size": len(body.encode()),
            "preview": body[:1024],
        }
    return payload


class PgEventStore:
    """Append-only async log of validator events (logs + metrics)."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def init_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(VALIDATOR_EVENTS_SCHEMA)

    async def insert_batch(
        self,
        hotkey: str,
        events: list[dict],
    ) -> int:
        """Insert a batch of events for ``hotkey``. Returns rows written.

        Each event is a dict with keys: ``ts`` (float, optional — defaults
        to now), ``round_id`` (int, optional), ``kind`` (str, required —
        one of :data:`VALID_KINDS`), ``level`` (str, optional), ``payload``
        (dict, required). Events with an unknown ``kind`` are skipped.
        """
        if not events:
            return 0

        rows: list[tuple] = []
        now = time.time()
        for ev in events:
            kind = str(ev.get("kind") or "").lower()
            if kind not in VALID_KINDS:
                continue
            level = str(ev.get("level") or "").lower()
            if level not in VALID_LEVELS:
                level = ""
            ts = ev.get("ts")
            try:
                ts = float(ts) if ts is not None else now
            except (TypeError, ValueError):
                ts = now
            try:
                round_id = int(ev.get("round_id", -1))
            except (TypeError, ValueError):
                round_id = -1
            payload = _normalise_payload(ev.get("payload") or {})
            rows.append((
                hotkey, ts, round_id, kind, level, json.dumps(payload),
            ))

        if not rows:
            return 0

        await self.pool.executemany(
            "INSERT INTO validator_events "
            "(hotkey, ts, round_id, kind, level, payload) "
            "VALUES ($1, $2, $3, $4, $5, $6::jsonb)",
            rows,
        )
        return len(rows)

    async def tail(
        self,
        hotkey: str,
        since_id: int = 0,
        limit: int = 200,
        kind: Optional[str] = None,
    ) -> list[dict]:
        """Return events for ``hotkey`` with ``id > since_id``.

        Results are returned newest first if ``since_id == 0`` (initial
        load, "give me the most recent N"), otherwise oldest first so the
        SPA can append in chronological order.
        """
        limit = max(1, min(int(limit), 1000))
        params: list = [hotkey]
        sql = "SELECT id, hotkey, ts, round_id, kind, level, payload " \
              "FROM validator_events WHERE hotkey = $1"
        if kind:
            params.append(str(kind).lower())
            sql += f" AND kind = ${len(params)}"
        if since_id > 0:
            params.append(int(since_id))
            sql += f" AND id > ${len(params)} ORDER BY id ASC"
        else:
            sql += " ORDER BY id DESC"
        params.append(limit)
        sql += f" LIMIT ${len(params)}"

        records = await self.pool.fetch(sql, *params)
        out = [
            {
                "id": int(r["id"]),
                "hotkey": r["hotkey"],
                "ts": float(r["ts"]),
                "round_id": int(r["round_id"]),
                "kind": r["kind"],
                "level": r["level"] or "",
                "payload": _decode_jsonb(r["payload"], {}),
            }
            for r in records
        ]
        # When seeded (since_id == 0) we fetched DESC for "newest N" but
        # the SPA wants chronological order — flip back here so the
        # response shape is consistent regardless of cursor.
        if since_id <= 0:
            out.reverse()
        return out

    async def metrics(
        self,
        hotkey: str,
        metric: str,
        round_id: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Return ``{ts, round_id, value}`` points for a named metric.

        Looks for events with ``kind='metric'`` whose payload carries the
        ``name`` and ``value`` fields the validator-side ``EventBuffer``
        emits.
        """
        limit = max(1, min(int(limit), 5000))
        params: list = [hotkey, metric]
        sql = (
            "SELECT id, ts, round_id, payload "
            "FROM validator_events "
            "WHERE hotkey = $1 AND kind = 'metric' "
            "AND payload->>'name' = $2"
        )
        if round_id is not None:
            params.append(int(round_id))
            sql += f" AND round_id = ${len(params)}"
        params.append(limit)
        sql += f" ORDER BY id ASC LIMIT ${len(params)}"

        records = await self.pool.fetch(sql, *params)
        out = []
        for r in records:
            payload = _decode_jsonb(r["payload"], {}) or {}
            value = payload.get("value")
            if value is None:
                continue
            out.append({
                "id": int(r["id"]),
                "ts": float(r["ts"]),
                "round_id": int(r["round_id"]),
                "value": value,
            })
        return out

    async def hotkeys_with_recent_events(
        self, since_seconds: float = 86400, limit: int = 200,
    ) -> list[dict]:
        """List validator hotkeys that have logged events recently.

        Powers the dashboard "validators" landing page so users can pick a
        run to inspect. Returns ``{hotkey, last_ts, last_id, n}``.
        """
        cutoff = time.time() - max(0.0, float(since_seconds))
        rows = await self.pool.fetch(
            """
            SELECT hotkey,
                   MAX(ts) AS last_ts,
                   MAX(id) AS last_id,
                   COUNT(*) AS n
            FROM validator_events
            WHERE ts >= $1
            GROUP BY hotkey
            ORDER BY last_ts DESC
            LIMIT $2
            """,
            cutoff, max(1, min(int(limit), 500)),
        )
        return [
            {
                "hotkey": r["hotkey"],
                "last_ts": float(r["last_ts"]),
                "last_id": int(r["last_id"]),
                "n": int(r["n"]),
            }
            for r in rows
        ]

    async def prune(self, retention_days: int) -> int:
        """Delete events older than ``retention_days``. Returns rows deleted."""
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        result = await self.pool.execute(
            "DELETE FROM validator_events WHERE ts < $1", cutoff,
        )
        # asyncpg returns "DELETE <n>" — parse the count for the caller.
        try:
            return int(str(result).split()[-1])
        except (ValueError, IndexError):
            return 0
