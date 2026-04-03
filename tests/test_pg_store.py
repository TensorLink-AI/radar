"""Tests for shared.pg_store — async Postgres experiment store.

Skip if TEST_PG_DSN not set. Requires a running Postgres instance.
"""

import os
import pytest
import asyncio

from shared.database import DataElement

PG_DSN = os.getenv("TEST_PG_DSN", "")
skip_no_pg = pytest.mark.skipif(not PG_DSN, reason="TEST_PG_DSN not set")


@pytest.fixture
async def store():
    """Create a PgExperimentStore with a clean schema for each test."""
    import asyncpg
    from shared.pg_store import PgExperimentStore

    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=3)
    # Clean tables before each test
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS code_components CASCADE")
        await conn.execute("DROP TABLE IF EXISTS round_context CASCADE")
        await conn.execute("DROP TABLE IF EXISTS miner_access_log CASCADE")
        await conn.execute("DROP TABLE IF EXISTS experiments CASCADE")
        await conn.execute("DROP FUNCTION IF EXISTS experiments_search_update() CASCADE")

    s = PgExperimentStore(pool)
    await s.init_schema()
    yield s
    await pool.close()


@skip_no_pg
@pytest.mark.asyncio
async def test_add_and_get(store):
    elem = DataElement(name="test1", code="print('hello')", success=True, metric=0.9)
    idx = await store.add(elem)
    assert idx == 0
    size = await store.get_size()
    assert size == 1
    got = await store.get(0)
    assert got.name == "test1"


@skip_no_pg
@pytest.mark.asyncio
async def test_lineage(store):
    await store.add(DataElement(name="root", code="v0", success=True, metric=1.0, parent=None, generation=0))
    await store.add(DataElement(name="child", code="v1", success=True, metric=0.9, parent=0, generation=1))
    await store.add(DataElement(name="grandchild", code="v2", success=True, metric=0.8, parent=1, generation=2))
    lineage = await store.get_lineage(2)
    assert len(lineage) == 3
    assert lineage[0].name == "root"
    assert lineage[2].name == "grandchild"


@skip_no_pg
@pytest.mark.asyncio
async def test_search(store):
    await store.add(DataElement(name="e1", code="x", motivation="gated linear attention", analysis="improved throughput"))
    await store.add(DataElement(name="e2", code="y", motivation="cosine schedule", analysis="better convergence"))
    results = await store.search("attention")
    assert len(results) >= 1
    assert results[0].name == "e1"


@skip_no_pg
@pytest.mark.asyncio
async def test_failures(store):
    await store.add(DataElement(name="ok", code="x", success=True, metric=0.9))
    await store.add(DataElement(name="fail1", code="y", success=False))
    await store.add(DataElement(name="fail2", code="z", success=False))
    failures = await store.get_failures(10)
    assert len(failures) == 2
    assert failures[0].name == "fail2"


@skip_no_pg
@pytest.mark.asyncio
async def test_stats(store):
    await store.add(DataElement(name="a", success=True, metric=0.9))
    await store.add(DataElement(name="b", success=True, metric=0.8))
    await store.add(DataElement(name="c", success=False))
    s = await store.stats()
    assert s["total"] == 3
    assert s["successful"] == 2
    assert s["failed"] == 1
    assert s["best_metric"] == 0.8


@skip_no_pg
@pytest.mark.asyncio
async def test_count_in_flops_range(store):
    await store.add(DataElement(name="a", success=True, metric=0.9, objectives={"flops_equivalent_size": 200000}))
    await store.add(DataElement(name="b", success=True, metric=0.8, objectives={"flops_equivalent_size": 500000}))
    await store.add(DataElement(name="c", success=True, metric=0.7, objectives={"flops_equivalent_size": 2000000}))
    assert await store.count_in_flops_range(100000, 600000) == 2


@skip_no_pg
@pytest.mark.asyncio
async def test_get_in_flops_range(store):
    await store.add(DataElement(name="a", success=True, metric=0.9, objectives={"flops_equivalent_size": 200000}))
    await store.add(DataElement(name="b", success=True, metric=0.8, objectives={"flops_equivalent_size": 2000000}))
    results = await store.get_in_flops_range(100000, 500000)
    assert len(results) == 1
    assert results[0].name == "a"


@skip_no_pg
@pytest.mark.asyncio
async def test_get_family_summary(store):
    await store.add(DataElement(name="root_a", code="a", success=True, metric=1.0))
    await store.add(DataElement(name="child_a", code="a1", success=True, metric=0.8, parent=0, generation=1))
    await store.add(DataElement(name="grand_a", code="a2", success=True, metric=0.6, parent=1, generation=2))
    await store.add(DataElement(name="root_b", code="b", success=True, metric=0.9))
    families = await store.get_family_summary()
    assert len(families) == 2
    fam_a = next(f for f in families if f["root_name"] == "root_a")
    assert fam_a["num_descendants"] == 3
    assert fam_a["best_metric"] == 0.6


@skip_no_pg
@pytest.mark.asyncio
async def test_get_diff(store):
    await store.add(DataElement(
        name="v1", code="def build_model():\n    return Linear(10, 10)\n",
        success=True, metric=0.5,
    ))
    await store.add(DataElement(
        name="v2", code="def build_model():\n    return Linear(10, 20)\n",
        success=True, metric=0.4, parent=0, generation=1,
    ))
    diff = await store.get_diff(1)
    assert diff is not None
    assert "Linear(10, 10)" in diff
    assert "Linear(10, 20)" in diff


@skip_no_pg
@pytest.mark.asyncio
async def test_get_pareto_elements(store):
    await store.add(DataElement(name="a", success=True, metric=0.5))
    await store.add(DataElement(name="b", success=False))
    await store.add(DataElement(name="c", success=True, metric=None))
    await store.add(DataElement(name="d", success=True, metric=0.3))
    elems = await store.get_pareto_elements()
    assert len(elems) == 2


@skip_no_pg
@pytest.mark.asyncio
async def test_add_batch(store):
    elems = [
        DataElement(name=f"batch_{i}", code=f"v{i}", success=True, metric=float(i))
        for i in range(5)
    ]
    indices = await store.add_batch(elems)
    assert indices == [0, 1, 2, 3, 4]
    assert await store.get_size() == 5


@skip_no_pg
@pytest.mark.asyncio
async def test_task_filtering(store):
    await store.add(DataElement(name="ts1", code="a", task="ts_forecasting", success=True, metric=0.5))
    await store.add(DataElement(name="lm1", code="b", task="nanogpt", success=True, metric=0.3))
    ts = await store.get_pareto_elements(task="ts_forecasting")
    assert len(ts) == 1
    assert ts[0].task == "ts_forecasting"
    all_el = await store.get_pareto_elements()
    assert len(all_el) == 2
