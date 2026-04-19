"""Access logger helper — pure-Python experiment ID extraction.

The AccessLogger class has been moved to shared/pg_access_logger.py
(async Postgres-backed PgAccessLogger).

This module retains the stateless helper function with no DB dependency.
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
