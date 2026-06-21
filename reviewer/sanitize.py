"""Treat ``agent_events`` payloads as untrusted before they reach the
reviewer's prompt.

``agent_events.request_json`` and ``response_json`` carry raw LLM
provider output, web fetch results, and other content that originated
outside our trust boundary. A determined attacker who can place text
into any of those payloads (via a compromised provider, a crafted web
result, or anything else) could instruct the reviewer to write
malicious prompt variants. We mitigate that by:

* Wrapping every untrusted blob in an ``<untrusted_external_data>`` tag
  the reviewer prompt is taught to treat as data, not instructions.
* Stripping NUL bytes (which can break downstream JSON parsers).
* Capping each row hard at 64KB regardless of the DB's 4MB cap — the
  reviewer rarely needs more than the head of a transcript, and a 4MB
  payload can drown the prompt's token budget by itself.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


HARD_CAP_BYTES = 64 * 1024
UNTRUSTED_OPEN = "<untrusted_external_data>"
UNTRUSTED_CLOSE = "</untrusted_external_data>"


def _clip(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.replace("\x00", "")
    if len(cleaned) > HARD_CAP_BYTES:
        cleaned = cleaned[:HARD_CAP_BYTES] + "...[clipped]"
    return f"{UNTRUSTED_OPEN}\n{cleaned}\n{UNTRUSTED_CLOSE}"


def sanitize_event_row(row: sqlite3.Row | dict) -> dict[str, Any]:
    """Return a JSON-safe dict for one ``agent_events`` row with the
    free-text payloads wrapped + clipped."""
    if isinstance(row, sqlite3.Row):
        row = dict(row)
    return {
        "id": row.get("id"),
        "ts": row.get("ts"),
        "round_id": row.get("round_id"),
        "miner_id": row.get("miner_id"),
        "kind": row.get("kind"),
        "endpoint": row.get("endpoint"),
        "status": row.get("status"),
        "latency_ms": row.get("latency_ms"),
        "error": row.get("error"),
        "request_clipped": _clip(row.get("request_json")),
        "response_clipped": _clip(row.get("response_json")),
    }


def sanitize_experiment_text(row: dict) -> dict[str, Any]:
    """Strip the dangerous-by-context text fields out of an experiment
    row and wrap them. ``code`` is left raw — it's executed downstream
    and the validator already gates it; if the reviewer's reading code
    we want the literal source."""
    safe = dict(row)
    for field in ("motivation", "reasoning", "analysis"):
        if field in safe and isinstance(safe[field], str):
            safe[field] = _clip(safe[field])
    return safe


def sample_events(
    conn: sqlite3.Connection, *, lo: int, hi: int,
    failures_cap: int = 200, winners_cap: int = 40,
    random_pct: float = 0.05,
) -> list[dict]:
    """Pull a defensible sample of ``agent_events`` for the window.

    Strategy: every row carrying a non-null ``error`` (up to ``failures_cap``)
    + a deterministic uniform sample of the rest. We avoid ``ORDER BY
    RANDOM()`` because it scans the table; ``id % stride = 0`` is good
    enough and stable across reruns.
    """
    out: list[dict] = []
    cur = conn.execute(
        "SELECT * FROM agent_events "
        "WHERE round_id BETWEEN ? AND ? AND error IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (lo, hi, failures_cap),
    )
    for row in cur:
        out.append(sanitize_event_row(row))

    stride = max(1, int(1.0 / max(random_pct, 0.001)))
    cur = conn.execute(
        "SELECT * FROM agent_events "
        "WHERE round_id BETWEEN ? AND ? AND error IS NULL "
        "  AND id % ? = 0 ORDER BY id DESC LIMIT ?",
        (lo, hi, stride, winners_cap),
    )
    for row in cur:
        out.append(sanitize_event_row(row))
    return out


def to_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
