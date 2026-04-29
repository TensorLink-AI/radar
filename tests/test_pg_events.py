"""Tests for shared.pg_events — PgEventStore.

Skip if TEST_PG_DSN not set. Requires a running Postgres instance.
"""

import os
import time

import pytest

PG_DSN = os.getenv("TEST_PG_DSN", "")
skip_no_pg = pytest.mark.skipif(not PG_DSN, reason="TEST_PG_DSN not set")


@pytest.fixture
async def store():
    """Fresh PgEventStore on a clean validator_events table."""
    from shared.pg_events import PgEventStore
    from shared.pg_store import create_pg_pool

    pool = await create_pg_pool(PG_DSN, min_size=1, max_size=3)
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS validator_events CASCADE")
    s = PgEventStore(pool)
    await s.init_schema()
    yield s
    await pool.close()


@skip_no_pg
@pytest.mark.asyncio
async def test_insert_batch_roundtrip(store):
    n = await store.insert_batch("hk1", [
        {"kind": "log", "level": "info", "ts": 100.0, "round_id": 5,
         "payload": {"message": "first"}},
        {"kind": "metric", "ts": 101.0, "round_id": 5,
         "payload": {"name": "loss", "value": 0.42}},
    ])
    assert n == 2
    rows = await store.tail("hk1", limit=10)
    assert len(rows) == 2
    assert [r["kind"] for r in rows] == ["log", "metric"]
    assert rows[0]["payload"]["message"] == "first"
    assert rows[1]["payload"]["value"] == 0.42


@skip_no_pg
@pytest.mark.asyncio
async def test_invalid_kind_skipped(store):
    n = await store.insert_batch("hk1", [
        {"kind": "garbage", "payload": {}},
        {"kind": "log", "payload": {"message": "ok"}},
    ])
    assert n == 1


@skip_no_pg
@pytest.mark.asyncio
async def test_oversized_payload_truncated(store):
    big = "x" * 50_000
    await store.insert_batch("hk1", [
        {"kind": "log", "payload": {"message": big}},
    ])
    rows = await store.tail("hk1", limit=10)
    assert rows[0]["payload"].get("_truncated") is True
    assert rows[0]["payload"]["_size"] > 16 * 1024


@skip_no_pg
@pytest.mark.asyncio
async def test_tail_with_cursor(store):
    for i in range(5):
        await store.insert_batch("hk2", [
            {"kind": "log", "payload": {"message": f"m{i}"}},
        ])
    initial = await store.tail("hk2", limit=10)
    assert [r["payload"]["message"] for r in initial] == [
        "m0", "m1", "m2", "m3", "m4",
    ]
    cursor = initial[2]["id"]
    after = await store.tail("hk2", since_id=cursor, limit=10)
    assert [r["payload"]["message"] for r in after] == ["m3", "m4"]


@skip_no_pg
@pytest.mark.asyncio
async def test_tail_filter_by_kind(store):
    await store.insert_batch("hk3", [
        {"kind": "log", "payload": {"message": "L"}},
        {"kind": "metric", "payload": {"name": "x", "value": 1}},
        {"kind": "log", "payload": {"message": "L2"}},
    ])
    only_metrics = await store.tail("hk3", limit=10, kind="metric")
    assert len(only_metrics) == 1
    assert only_metrics[0]["kind"] == "metric"


@skip_no_pg
@pytest.mark.asyncio
async def test_metrics_series(store):
    for i in range(3):
        await store.insert_batch("hk4", [
            {"kind": "metric", "round_id": 9,
             "payload": {"name": "loss", "value": 1.0 - i * 0.1}},
        ])
    pts = await store.metrics("hk4", "loss")
    assert [p["value"] for p in pts] == [1.0, 0.9, 0.8]


@skip_no_pg
@pytest.mark.asyncio
async def test_hotkeys_with_recent_events(store):
    await store.insert_batch("hkA", [
        {"kind": "log", "ts": time.time(), "payload": {"message": "x"}},
    ])
    await store.insert_batch("hkB", [
        {"kind": "log", "ts": time.time() - 10, "payload": {"message": "x"}},
    ])
    rows = await store.hotkeys_with_recent_events(since_seconds=3600)
    hotkeys = {r["hotkey"] for r in rows}
    assert {"hkA", "hkB"} <= hotkeys


@skip_no_pg
@pytest.mark.asyncio
async def test_prune_old(store):
    very_old = time.time() - 100 * 86400
    await store.insert_batch("hk5", [
        {"kind": "log", "ts": very_old, "payload": {"message": "ancient"}},
        {"kind": "log", "payload": {"message": "fresh"}},
    ])
    deleted = await store.prune(retention_days=30)
    assert deleted == 1
    rows = await store.tail("hk5", limit=10)
    assert [r["payload"]["message"] for r in rows] == ["fresh"]
