"""Access logger helper — pure-Python experiment ID extraction.

The AccessLogger class has been moved to shared/pg_access_logger.py
(async Postgres-backed PgAccessLogger).

This module retains the stateless helper function with no DB dependency.
"""

import json


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


def extract_ids_from_body(body: bytes, content_type: str) -> list[int]:
    """Best-effort JSON parse of a response body, then extract experiment IDs.

    Returns ``[]`` for non-JSON content, parse errors, or empty bodies so the
    caller can unconditionally pass the result to ``log_access``.
    """
    if not body or "application/json" not in (content_type or "").lower():
        return []
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return []
    return _extract_experiment_ids(data)

