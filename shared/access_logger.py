"""Access logger helper — pure-Python experiment ID extraction.

The AccessLogger class has been moved to shared/pg_access_logger.py
(async Postgres-backed PgAccessLogger).

This module retains the stateless helper functions with no DB dependency.
"""

import json
import re

# Scalar keys whose value is a single experiment ID across the API surface:
# - /experiments/{id} → "index"
# - /experiments/families → "root_index" / "latest_index"
# - /experiments/diff/{a}/{b} → "index_a" / "index_b"
# - /experiments/{id}/verify, /provenance/components → "experiment_id"
# - /provenance/{id}/influences → list of {"source_id": ...}
# - /provenance/{id}/impact → list of {"target_id": ...}
_SCALAR_ID_KEYS = (
    "index", "root_index", "latest_index",
    "index_a", "index_b",
    "experiment_id", "source_id", "target_id",
)

# List-valued keys returning a flat array of experiment IDs:
# - /provenance/components → {"experiment_ids": [...]}
# - /provenance/dead_ends → {"dead_ends": [...]}
_LIST_ID_KEYS = ("experiment_ids", "dead_ends")

# Path-encoded experiment IDs. The agent's intent is captured by the URL
# even when the response body has no extractable IDs (404, errors, terse
# graph shapes). Match the integer segment(s) directly.
_PATH_ID_PATTERNS = (
    re.compile(r"^/experiments/(\d+)(?:/|$)"),
    re.compile(r"^/experiments/lineage/(\d+)(?:/|$)"),
    re.compile(r"^/experiments/diff/(\d+)/(\d+)(?:/|$)"),
    re.compile(r"^/provenance/(\d+)(?:/|$)"),
)


def _collect_from_dict(d: dict, ids: set[int]) -> None:
    for key in _SCALAR_ID_KEYS:
        val = d.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            ids.add(val)
    for key in _LIST_ID_KEYS:
        val = d.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, int) and not isinstance(item, bool):
                    ids.add(item)


def _extract_experiment_ids(response_data) -> list[int]:
    """Pull experiment IDs from any db_server response shape."""
    ids: set[int] = set()

    if isinstance(response_data, dict):
        _collect_from_dict(response_data, ids)
    elif isinstance(response_data, list):
        for item in response_data:
            if isinstance(item, dict):
                _collect_from_dict(item, ids)
            elif isinstance(item, int) and not isinstance(item, bool):
                # Bare-int lists: /provenance/components → experiment_ids
                # gets unwrapped this way in some responses.
                ids.add(item)

    return sorted(ids)


def extract_ids_from_path(path: str) -> list[int]:
    """Pull experiment IDs the agent named directly in the URL path.

    Captures the queried experiment even when the response body is empty,
    a terse graph wrapper, or a 4xx — the URL itself proves intent.
    """
    if not path:
        return []
    ids: set[int] = set()
    for pattern in _PATH_ID_PATTERNS:
        m = pattern.match(path)
        if m:
            for group in m.groups():
                try:
                    ids.add(int(group))
                except (TypeError, ValueError):
                    continue
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

