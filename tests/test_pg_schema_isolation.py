"""Tests for per-network Postgres schema isolation.

Covers the Crunchy Bridge refactor:
  - ``create_pg_pool(dsn, schema=...)`` sets ``search_path`` on every
    pooled connection so unqualified identifiers resolve in the target
    schema.
  - ``ensure_schema_exists`` is idempotent.
  - ``PgExperimentStore.init_schema()`` creates the FTS trigger on *each*
    schema it's invoked against — the regression guard for the old
    globally-scoped ``pg_trigger`` existence check that silently skipped
    every schema after the first.

Requires a running Postgres instance. Skip if TEST_PG_DSN not set.
"""

from __future__ import annotations

import os

import pytest

from shared.database import DataElement

PG_DSN = os.getenv("TEST_PG_DSN", "")
skip_no_pg = pytest.mark.skipif(not PG_DSN, reason="TEST_PG_DSN not set")


async def _drop_schema(dsn: str, schema: str) -> None:
    """Drop a schema CASCADE on a one-shot connection."""
    import asyncpg
    from shared.pg_store import _quote_ident
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
    finally:
        await conn.close()


@skip_no_pg
@pytest.mark.asyncio
async def test_ensure_schema_exists_is_idempotent():
    """Calling ensure_schema_exists twice must not error."""
    import asyncpg
    from shared.pg_store import ensure_schema_exists

    schema = "radar_test_idem"
    await _drop_schema(PG_DSN, schema)

    conn = await asyncpg.connect(PG_DSN)
    try:
        await ensure_schema_exists(conn, schema)
        await ensure_schema_exists(conn, schema)  # second call: no-op
        row = await conn.fetchval(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = $1",
            schema,
        )
        assert row == 1
    finally:
        await conn.close()
        await _drop_schema(PG_DSN, schema)


@skip_no_pg
@pytest.mark.asyncio
async def test_search_path_set_on_every_pool_connection():
    """Every connection checked out of the pool must see the right search_path.

    Regression guard: a per-call ``SET search_path`` inside a single
    fetch() would work for that call but silently revert on the next
    checkout.  Confirm the init hook wires it for the lifetime of the
    connection.
    """
    from shared.pg_store import create_pg_pool, ensure_schema_exists

    schema = "radar_test_sp"
    import asyncpg
    conn0 = await asyncpg.connect(PG_DSN)
    try:
        await ensure_schema_exists(conn0, schema)
    finally:
        await conn0.close()

    pool = await create_pg_pool(PG_DSN, schema=schema, min_size=2, max_size=2)
    try:
        # Round-robin through both connections, verify each has the
        # right search_path.
        seen: set[str] = set()
        for _ in range(4):
            async with pool.acquire() as conn:
                path = await conn.fetchval("SHOW search_path")
                seen.add(path)
        assert all("radar_test_sp" in p for p in seen), seen
    finally:
        await pool.close()
        await _drop_schema(PG_DSN, schema)


@skip_no_pg
@pytest.mark.asyncio
async def test_fts_trigger_exists_on_both_schemas():
    """init_schema() must create the FTS trigger on every schema it runs in.

    Regression guard for the Crunchy launch: the old FTS_TRIGGER_DDL
    checked ``pg_trigger WHERE tgname = 'experiments_search_trigger'``
    without filtering by ``tgrelid``.  With two schemas sharing a
    cluster (testnet + mainnet), the first init_schema() call created
    the trigger, and every subsequent schema's init_schema() saw a
    matching row in pg_trigger and silently skipped creation.  Result:
    experiments in the second schema never had their search_vector
    populated, and FTS queries returned zero rows.

    This test boots two independent schemas, calls init_schema() on
    both, then asserts the trigger exists on BOTH schemas.
    """
    from shared.pg_store import (
        PgExperimentStore, create_pg_pool, ensure_schema_exists,
    )

    schema_a = "radar_test_net_a"
    schema_b = "radar_test_net_b"

    # Clean slate
    for s in (schema_a, schema_b):
        await _drop_schema(PG_DSN, s)

    import asyncpg
    conn0 = await asyncpg.connect(PG_DSN)
    try:
        await ensure_schema_exists(conn0, schema_a)
        await ensure_schema_exists(conn0, schema_b)
    finally:
        await conn0.close()

    pool_a = await create_pg_pool(PG_DSN, schema=schema_a, min_size=1, max_size=2)
    pool_b = await create_pg_pool(PG_DSN, schema=schema_b, min_size=1, max_size=2)
    try:
        await PgExperimentStore(pool_a).init_schema()
        await PgExperimentStore(pool_b).init_schema()

        # Query pg_trigger via the admin connection so we see the
        # unfiltered global view. Both schemas' experiments table must
        # have the trigger.
        conn = await asyncpg.connect(PG_DSN)
        try:
            rows = await conn.fetch(
                """
                SELECT n.nspname AS schema, c.relname AS tbl, t.tgname AS trig
                FROM pg_trigger t
                JOIN pg_class c ON t.tgrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE t.tgname = 'experiments_search_trigger'
                  AND NOT t.tgisinternal
                  AND n.nspname = ANY($1::text[])
                ORDER BY n.nspname
                """,
                [schema_a, schema_b],
            )
        finally:
            await conn.close()

        schemas_with_trigger = {r["schema"] for r in rows}
        assert schemas_with_trigger == {schema_a, schema_b}, (
            f"FTS trigger missing on one of the schemas: {schemas_with_trigger}. "
            "The FTS_TRIGGER_DDL existence check is likely back to globally "
            "scoped instead of tgrelid-scoped to the current schema's "
            "experiments table."
        )
    finally:
        await pool_a.close()
        await pool_b.close()
        for s in (schema_a, schema_b):
            await _drop_schema(PG_DSN, s)


@skip_no_pg
@pytest.mark.asyncio
async def test_writes_isolated_between_schemas():
    """A row written through one schema's pool must not be visible through
    another schema's pool.  End-to-end check of the isolation contract."""
    from shared.pg_store import (
        PgExperimentStore, create_pg_pool, ensure_schema_exists,
    )

    schema_a = "radar_test_iso_a"
    schema_b = "radar_test_iso_b"

    for s in (schema_a, schema_b):
        await _drop_schema(PG_DSN, s)

    import asyncpg
    conn0 = await asyncpg.connect(PG_DSN)
    try:
        await ensure_schema_exists(conn0, schema_a)
        await ensure_schema_exists(conn0, schema_b)
    finally:
        await conn0.close()

    pool_a = await create_pg_pool(PG_DSN, schema=schema_a, min_size=1, max_size=2)
    pool_b = await create_pg_pool(PG_DSN, schema=schema_b, min_size=1, max_size=2)
    try:
        store_a = PgExperimentStore(pool_a)
        store_b = PgExperimentStore(pool_b)
        await store_a.init_schema()
        await store_b.init_schema()

        await store_a.add(DataElement(
            name="only_in_a", code="x", success=True, metric=0.5,
        ))
        assert await store_a.get_size() == 1
        assert await store_b.get_size() == 0
    finally:
        await pool_a.close()
        await pool_b.close()
        for s in (schema_a, schema_b):
            await _drop_schema(PG_DSN, s)


def test_quote_ident_rejects_bad_input():
    """Unit-only guard on the identifier quoter used for SET search_path
    and CREATE SCHEMA. No Postgres required."""
    from shared.pg_store import _quote_ident

    assert _quote_ident("mainnet") == '"mainnet"'
    assert _quote_ident('foo"bar') == '"foo""bar"'

    for bad in ("", None, 123):
        with pytest.raises(ValueError):
            _quote_ident(bad)  # type: ignore[arg-type]
