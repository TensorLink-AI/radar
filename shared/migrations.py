"""Lightweight async schema migration runner.

Applies forward-only SQL migrations from ``database/migrations/`` to a
single Postgres schema and records each applied file in
``<schema>.schema_migrations``. Designed to run at database server
startup after ``ensure_schema_exists`` and ``init_schema``.

Design notes:

* No third-party migration framework (Alembic / SQLAlchemy). Just
  asyncpg, hashlib, and pathlib.
* Each migration file is applied inside its own transaction. The row
  in ``schema_migrations`` is inserted in the same transaction, so a
  failure partway through the SQL reverts everything — including the
  bookkeeping row — keeping the runner safe to re-enter.
* The schema name is passed through the same allowlist used by
  ``shared.pg_store.create_pg_pool``. Postgres identifiers cannot be
  parameterised, so validation is the only defence against a malicious
  schema argument being spliced into DDL.
* ``CURRENT_SCHEMA`` is a plain-text placeholder that the runner
  substitutes with the validated schema name. This means a single
  migration file works against every managed schema
  (``testnet``/``mainnet``).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

# Mirrors the allowlist in ``shared.pg_store._validate_schema_name``.
# Intentionally duplicated rather than imported to keep this module
# loadable with a trivial import graph (no asyncpg pool helpers).
_SCHEMA_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")

# ``NNN_description.sql`` where NNN is a zero-padded 3-digit prefix and
# the description is lowercase ASCII + digits + underscores.
_MIGRATION_FILENAME_PATTERN = re.compile(r"^\d{3}_[a-z0-9_]+\.sql$")

# Placeholder inside migration files, replaced with the validated
# schema name at apply time.
_SCHEMA_PLACEHOLDER = "CURRENT_SCHEMA"


def _default_migrations_dir() -> Path:
    """Return the canonical on-disk migrations directory.

    Resolved relative to this file so the runner works regardless of
    the process's current working directory.
    """
    return Path(__file__).resolve().parent.parent / "database" / "migrations"


def _validate_schema_name(schema: str) -> str:
    """Validate ``schema`` against the Postgres-identifier allowlist.

    Re-implemented here (mirror of ``pg_store._validate_schema_name``)
    so ``apply_migrations`` doesn't depend on the pool module.
    """
    if not isinstance(schema, str) or not _SCHEMA_NAME_PATTERN.match(schema):
        raise ValueError(
            f"Invalid Postgres schema name {schema!r}: must match "
            f"{_SCHEMA_NAME_PATTERN.pattern} (lowercase start, "
            "lowercase/digit/underscore only, <=63 chars)"
        )
    return schema


def _checksum(contents: str) -> str:
    """SHA-256 of migration file contents, encoded as hex."""
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


async def _ensure_migrations_table(
    conn: asyncpg.Connection, schema: str,
) -> None:
    """Create ``<schema>.schema_migrations`` if it doesn't exist."""
    await conn.execute(
        f'''
        CREATE TABLE IF NOT EXISTS "{schema}".schema_migrations (
            filename   TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            checksum   TEXT NOT NULL
        )
        '''
    )


async def apply_migrations(
    conn: asyncpg.Connection,
    schema: str,
    migrations_dir: Optional[Path] = None,
) -> list[str]:
    """Apply pending migrations to ``schema``.

    Ensures ``<schema>.schema_migrations`` exists, then walks every
    ``.sql`` file in ``migrations_dir`` in lexicographic order. Files
    not already recorded are applied inside a transaction and
    registered on success. Files already recorded have their checksums
    compared against the stored value; mismatches log a warning but do
    not fail (committed migrations are expected to be immutable, so a
    drift is a human error worth surfacing but not crashing on).

    Returns the list of filenames newly applied during this call (empty
    if everything was already up to date).
    """
    schema = _validate_schema_name(schema)
    if migrations_dir is None:
        migrations_dir = _default_migrations_dir()

    # List + validate filenames BEFORE touching the database. A bad
    # filename is a packaging bug and we want it to raise cleanly
    # without side effects (in particular, without creating an empty
    # ``schema_migrations`` table on a fresh schema).
    if migrations_dir.exists():
        sql_files = sorted(
            p for p in migrations_dir.iterdir()
            if p.is_file() and p.suffix == ".sql"
        )
    else:
        sql_files = []

    for path in sql_files:
        if not _MIGRATION_FILENAME_PATTERN.match(path.name):
            raise ValueError(
                f"Invalid migration filename {path.name!r} in "
                f"{migrations_dir}: must match "
                f"{_MIGRATION_FILENAME_PATTERN.pattern} "
                "(zero-padded 3-digit prefix, lowercase/digit/underscore "
                "description, .sql suffix)"
            )

    await _ensure_migrations_table(conn, schema)

    applied_rows = await conn.fetch(
        f'SELECT filename, checksum FROM "{schema}".schema_migrations'
    )
    already_applied: dict[str, str] = {
        r["filename"]: r["checksum"] for r in applied_rows
    }

    newly_applied: list[str] = []

    for path in sql_files:
        contents = path.read_text(encoding="utf-8")
        checksum = _checksum(contents)

        if path.name in already_applied:
            if already_applied[path.name] != checksum:
                logger.warning(
                    "Migration %s on schema %r has drifted: stored "
                    "checksum %s, current %s. Committed migrations "
                    "should be immutable — editing a file that has "
                    "already been applied is almost always a mistake.",
                    path.name, schema,
                    already_applied[path.name], checksum,
                )
            continue

        sql = contents.replace(_SCHEMA_PLACEHOLDER, schema)

        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                f'''
                INSERT INTO "{schema}".schema_migrations
                    (filename, checksum)
                VALUES ($1, $2)
                ''',
                path.name, checksum,
            )

        logger.info(
            "Applied migration %s to schema %r", path.name, schema,
        )
        newly_applied.append(path.name)

    return newly_applied
