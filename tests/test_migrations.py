"""Tests for the async schema migration runner.

Most cases require a real Postgres to exercise transaction rollback,
schema isolation, and the bookkeeping table. Skip if ``TEST_PG_DSN`` is
unset, matching the convention used by ``test_pg_store.py`` and
``test_pg_schema_isolation.py``.

A handful of purely structural cases (filename validation) run without
Postgres.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import pytest_asyncio

PG_DSN = os.getenv("TEST_PG_DSN", "")
skip_no_pg = pytest.mark.skipif(not PG_DSN, reason="TEST_PG_DSN not set")


async def _drop_schema(dsn: str, schema: str) -> None:
    """Drop a schema CASCADE on a one-shot connection."""
    import asyncpg
    from shared.pg_store import _validate_schema_name

    safe = _validate_schema_name(schema)
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{safe}" CASCADE')
    finally:
        await conn.close()


@pytest.fixture
def empty_migrations_dir(tmp_path: Path) -> Path:
    """A fresh migrations directory with no .sql files in it."""
    d = tmp_path / "migrations"
    d.mkdir()
    return d


@pytest_asyncio.fixture
async def pg_schema():
    """Set up an empty, isolated schema for a single test and drop it
    after. Yields (conn, schema_name)."""
    import asyncpg

    schema = "radar_test_migrations"
    await _drop_schema(PG_DSN, schema)

    conn = await asyncpg.connect(PG_DSN)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        yield conn, schema
    finally:
        await conn.close()
        await _drop_schema(PG_DSN, schema)


# ── Filename validation (no Postgres needed) ───────────────────────────


@pytest.mark.parametrize(
    "bad_name",
    [
        "abc.sql",               # missing numeric prefix
        "01_foo.sql",            # 2-digit prefix, must be 3
        "001_BadCase.sql",       # uppercase in description
        "001-foo.sql",           # hyphen separator
        "0001_foo.sql",          # 4-digit prefix
        "001_.sql",              # empty description
        "001_foo bar.sql",       # whitespace in description
    ],
)
@skip_no_pg
@pytest.mark.asyncio
async def test_invalid_filename_rejected(
    pg_schema, tmp_path: Path, bad_name: str,
):
    """Any filename outside the strict pattern must raise ValueError."""
    from shared.migrations import apply_migrations

    conn, schema = pg_schema
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / bad_name).write_text("SELECT 1;")

    with pytest.raises(ValueError):
        await apply_migrations(conn, schema, migrations_dir=migrations_dir)


# ── Behaviour (Postgres-backed) ────────────────────────────────────────


@skip_no_pg
@pytest.mark.asyncio
async def test_empty_migrations_dir_creates_table(
    pg_schema, empty_migrations_dir: Path,
):
    """An empty migrations dir is a no-op but still creates the
    bookkeeping table — so the second deploy after a migration is added
    can look it up without a missing-relation error."""
    from shared.migrations import apply_migrations

    conn, schema = pg_schema
    applied = await apply_migrations(
        conn, schema, migrations_dir=empty_migrations_dir,
    )

    assert applied == []
    exists = await conn.fetchval(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = $1 AND table_name = 'schema_migrations'
        """,
        schema,
    )
    assert exists == 1


@skip_no_pg
@pytest.mark.asyncio
async def test_single_migration_applied_once(
    pg_schema, tmp_path: Path,
):
    """A migration applied twice should run exactly once and leave a
    single row in schema_migrations."""
    from shared.migrations import apply_migrations

    conn, schema = pg_schema
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_create_widget.sql").write_text(
        "CREATE TABLE CURRENT_SCHEMA.widget (id INT PRIMARY KEY);\n"
    )

    first = await apply_migrations(
        conn, schema, migrations_dir=migrations_dir,
    )
    second = await apply_migrations(
        conn, schema, migrations_dir=migrations_dir,
    )

    assert first == ["001_create_widget.sql"]
    assert second == []
    count = await conn.fetchval(
        f'SELECT COUNT(*) FROM "{schema}".schema_migrations'
    )
    assert count == 1


@skip_no_pg
@pytest.mark.asyncio
async def test_current_schema_substitution(
    pg_schema, tmp_path: Path,
):
    """CURRENT_SCHEMA must be replaced with the validated schema name
    so the same migration file targets whichever schema it's run
    against."""
    from shared.migrations import apply_migrations

    conn, schema = pg_schema

    # Seed the base table the migration expects to alter, without using
    # CURRENT_SCHEMA so there's no interference with the placeholder.
    await conn.execute(
        f'CREATE TABLE "{schema}".experiments (id INT PRIMARY KEY)'
    )

    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_add_foo.sql").write_text(
        "ALTER TABLE CURRENT_SCHEMA.experiments "
        "ADD COLUMN foo INT DEFAULT 0;\n"
    )

    applied = await apply_migrations(
        conn, schema, migrations_dir=migrations_dir,
    )
    assert applied == ["001_add_foo.sql"]

    # Confirm the column landed in the intended schema, not (say)
    # public.experiments or a literal schema called CURRENT_SCHEMA.
    col = await conn.fetchval(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = 'experiments'
          AND column_name = 'foo'
        """,
        schema,
    )
    assert col == "foo"


@skip_no_pg
@pytest.mark.asyncio
async def test_transaction_rollback_on_failure(
    pg_schema, tmp_path: Path,
):
    """A failing migration must not leave a row in schema_migrations.

    The runner wraps each file in a transaction; when the SQL raises,
    the bookkeeping insert rolls back with it. Re-running later (after
    the file is fixed) must apply it as if it had never been tried.
    """
    from shared.migrations import apply_migrations

    conn, schema = pg_schema
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    # Syntactically valid SQL that fails at execution time (references
    # a nonexistent table). Still a single statement so asyncpg's
    # ``execute`` path runs it as one command.
    (migrations_dir / "001_broken.sql").write_text(
        "ALTER TABLE CURRENT_SCHEMA.does_not_exist "
        "ADD COLUMN foo INT;\n"
    )

    with pytest.raises(Exception):
        await apply_migrations(
            conn, schema, migrations_dir=migrations_dir,
        )

    count = await conn.fetchval(
        f'SELECT COUNT(*) FROM "{schema}".schema_migrations'
    )
    assert count == 0


@skip_no_pg
@pytest.mark.asyncio
async def test_checksum_drift_warns_not_fails(
    pg_schema, tmp_path: Path, caplog,
):
    """Editing an already-applied migration triggers a warning but does
    not fail the runner — a committed migration is expected to be
    immutable, but we don't want the DB server to refuse to start when
    an operator hotfixes one in place."""
    from shared.migrations import apply_migrations

    conn, schema = pg_schema
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    path = migrations_dir / "001_create_drift.sql"
    path.write_text("CREATE TABLE CURRENT_SCHEMA.drift (id INT);\n")

    await apply_migrations(conn, schema, migrations_dir=migrations_dir)

    # Mutate the file after it's been applied. The runner should detect
    # the checksum mismatch and warn, but not re-run the SQL or raise.
    path.write_text("-- edited after applying\n" + path.read_text())

    with caplog.at_level(logging.WARNING, logger="shared.migrations"):
        applied = await apply_migrations(
            conn, schema, migrations_dir=migrations_dir,
        )

    assert applied == []
    assert any(
        "drifted" in rec.message and "001_create_drift.sql" in rec.message
        for rec in caplog.records
    ), [r.message for r in caplog.records]


@skip_no_pg
@pytest.mark.asyncio
async def test_schema_isolation(tmp_path: Path):
    """Each schema tracks its own migration history independently.

    Applying a migration to one schema must not leave any artefact —
    neither the table the migration creates, nor a row in
    ``schema_migrations`` — in the other schema. Re-running then
    applies the same file fresh against the second schema.
    """
    import asyncpg
    from shared.migrations import apply_migrations

    schema_a = "radar_test_iso_a"
    schema_b = "radar_test_iso_b"
    for s in (schema_a, schema_b):
        await _drop_schema(PG_DSN, s)

    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_create_row.sql").write_text(
        "CREATE TABLE CURRENT_SCHEMA.only_here (id INT);\n"
    )

    conn = await asyncpg.connect(PG_DSN)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema_a}"')
        await conn.execute(f'CREATE SCHEMA "{schema_b}"')

        # Apply only to schema_a.
        applied_a = await apply_migrations(
            conn, schema_a, migrations_dir=migrations_dir,
        )
        assert applied_a == ["001_create_row.sql"]

        # The migration must NOT have touched schema_b. Cheap check:
        # the table ``only_here`` lives in schema_a, and schema_b has
        # no ``schema_migrations`` table yet (apply_migrations has
        # never been called for it).
        exists_a = await conn.fetchval(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = $1 AND table_name = 'only_here'
            """,
            schema_a,
        )
        exists_b = await conn.fetchval(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = $1 AND table_name = 'only_here'
            """,
            schema_b,
        )
        bookkeeping_b = await conn.fetchval(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = $1 AND table_name = 'schema_migrations'
            """,
            schema_b,
        )
        assert exists_a == 1
        assert exists_b is None
        assert bookkeeping_b is None

        # Re-running against schema_a is a no-op; running against
        # schema_b applies the file fresh. Both schemas end up with
        # exactly one row in their own schema_migrations.
        reapplied_a = await apply_migrations(
            conn, schema_a, migrations_dir=migrations_dir,
        )
        applied_b = await apply_migrations(
            conn, schema_b, migrations_dir=migrations_dir,
        )
        assert reapplied_a == []
        assert applied_b == ["001_create_row.sql"]

        count_a = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema_a}".schema_migrations'
        )
        count_b = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema_b}".schema_migrations'
        )
        assert count_a == 1
        assert count_b == 1
    finally:
        await conn.close()
        for s in (schema_a, schema_b):
            await _drop_schema(PG_DSN, s)


@skip_no_pg
@pytest.mark.asyncio
async def test_invalid_schema_name_rejected(tmp_path: Path):
    """Schema names outside the allowlist must raise before any DDL
    is attempted — defence in depth against a caller accidentally
    splicing user-controlled input into the schema argument."""
    import asyncpg
    from shared.migrations import apply_migrations

    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    conn = await asyncpg.connect(PG_DSN)
    try:
        for bad in ["", "Public", 'evil"; DROP SCHEMA x', "1name"]:
            with pytest.raises(ValueError):
                await apply_migrations(
                    conn, bad, migrations_dir=migrations_dir,
                )
    finally:
        await conn.close()
