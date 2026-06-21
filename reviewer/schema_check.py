"""Fail-fast schema introspection for the reviewer.

The reviewer queries radar's SQLite store with hardcoded column names.
When ``local/store.py`` evolves, we want to break loudly at session
start rather than silently feed the model stale assumptions.

``ensure_schema()`` accepts a sqlite3 connection and a dict of expected
columns per table, and raises ``SchemaDriftError`` if any expected
column is missing.
"""

from __future__ import annotations

import sqlite3


class SchemaDriftError(RuntimeError):
    """Raised when the live DB schema is missing columns we depend on."""


EXPECTED: dict[str, set[str]] = {
    "experiments": {
        "id", "round_id", "miner_id", "name", "code", "motivation",
        "reasoning", "metric", "success", "objectives_json", "score",
        "prompt_id", "task", "parent_index", "generation", "timestamp",
    },
    "agent_events": {
        "id", "ts", "round_id", "miner_id", "kind", "endpoint",
        "status", "latency_ms", "request_json", "response_json", "error",
    },
    "proposals": {
        "proposal_id", "challenge_id", "round_id", "miner_id",
        "payload_json", "created_at",
    },
    "challenges": {
        "challenge_id", "round_id", "payload_json", "status", "created_at",
    },
}


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Raises ``SchemaDriftError`` listing every missing column."""
    missing: list[str] = []
    for table, expected_cols in EXPECTED.items():
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.OperationalError as e:
            missing.append(f"{table}: table not present ({e})")
            continue
        present = {r[1] for r in rows}
        gaps = expected_cols - present
        if gaps:
            missing.append(f"{table}: missing columns {sorted(gaps)}")
    if missing:
        raise SchemaDriftError(
            "schema drift detected — reviewer queries will not work:\n  "
            + "\n  ".join(missing)
        )
