"""Tests for shared.pg_provenance — async Postgres provenance queries.

Skip if TEST_PG_DSN not set.
"""

import os
import pytest

from shared.database import DataElement
from shared.provenance import detect_components, compute_similarity

PG_DSN = os.getenv("TEST_PG_DSN", "")
skip_no_pg = pytest.mark.skipif(not PG_DSN, reason="TEST_PG_DSN not set")


@pytest.fixture
async def prov_store():
    """Create store + provenance with clean schema."""
    import asyncpg
    from shared.pg_store import PgExperimentStore
    from shared.pg_access_logger import PgAccessLogger

    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=3)
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS code_components CASCADE")
        await conn.execute("DROP TABLE IF EXISTS round_context CASCADE")
        await conn.execute("DROP TABLE IF EXISTS miner_access_log CASCADE")
        await conn.execute("DROP TABLE IF EXISTS experiments CASCADE")
        await conn.execute("DROP FUNCTION IF EXISTS experiments_search_update() CASCADE")

    store = PgExperimentStore(pool)
    await store.init_schema()
    al = PgAccessLogger(pool)
    await al.init_schema()
    yield store, store.provenance, al, pool
    await pool.close()


async def _add_exp(pool, id, code="", hotkey="", round_id=None, metric=None,
                   success=True, task="ts", miner_uid=-1):
    from shared.pg_schema import INSERT_SQL
    import time
    await pool.execute(
        INSERT_SQL,
        id, "", code, "", "", metric, success, "",
        None, 0, 0.0, miner_uid, hotkey,
        [], "", [], {}, time.time(), round_id, task,
    )


# ── Pure-Python helpers (no DB needed) ──────────────────

def test_detect_rmsnorm_and_gelu():
    comps = detect_components("self.norm = RMSNorm(dim)\nself.act = GELU()")
    assert "RMSNorm" in comps
    assert "GELU" in comps


def test_detect_no_components():
    assert detect_components("x = 1 + 2") == []


def test_compute_similarity_identical():
    code = "class Model:\n    def forward(self): return self.linear()"
    sim = compute_similarity(code, code)
    assert sim["jaccard"] == 1.0


# ── Async provenance tests ──────────────────────────────

@skip_no_pg
@pytest.mark.asyncio
async def test_record_and_query_components(prov_store):
    store, prov, al, pool = prov_store
    await _add_exp(pool, 0, code="v0")
    await prov.record_components(0, ["RMSNorm", "GELU"])
    comps = await prov.get_experiment_components(0)
    assert set(comps) == {"RMSNorm", "GELU"}


@skip_no_pg
@pytest.mark.asyncio
async def test_record_components_dedup(prov_store):
    store, prov, al, pool = prov_store
    await _add_exp(pool, 0, code="v0")
    await prov.record_components(0, ["RMSNorm", "RMSNorm"])
    comps = await prov.get_experiment_components(0)
    assert comps == ["RMSNorm"]


@skip_no_pg
@pytest.mark.asyncio
async def test_get_component_experiments(prov_store):
    store, prov, al, pool = prov_store
    await _add_exp(pool, 0)
    await _add_exp(pool, 1)
    await _add_exp(pool, 2)
    await prov.record_components(0, ["RMSNorm"])
    await prov.record_components(1, ["RMSNorm", "GELU"])
    await prov.record_components(2, ["GELU"])
    result = await prov.get_component_experiments("RMSNorm")
    assert set(result) == {0, 1}


@skip_no_pg
@pytest.mark.asyncio
async def test_get_influences_from_shared_components(prov_store):
    store, prov, al, pool = prov_store
    await _add_exp(pool, 0, code="v0")
    await _add_exp(pool, 1, code="v1", hotkey="hk1", round_id=1)
    await prov.record_components(0, ["RMSNorm"])
    await prov.record_components(1, ["RMSNorm"])
    influences = await prov.get_influences(1)
    shared = [i for i in influences if i["evidence_type"] == "shared_component"]
    assert any(i["source_id"] == 0 for i in shared)


@skip_no_pg
@pytest.mark.asyncio
async def test_get_influences_from_frontier(prov_store):
    store, prov, al, pool = prov_store
    await _add_exp(pool, 0, code="v0")
    await _add_exp(pool, 1, code="v1", hotkey="hk1", round_id=1)
    await prov.record_round_context(1, 0, "frontier")
    influences = await prov.get_influences(1)
    frontier = [i for i in influences if i["evidence_type"] == "frontier"]
    assert any(i["source_id"] == 0 for i in frontier)


@skip_no_pg
@pytest.mark.asyncio
async def test_get_component_stats(prov_store):
    store, prov, al, pool = prov_store
    await _add_exp(pool, 0, code="v0", metric=0.5, success=True)
    await _add_exp(pool, 1, code="v1", metric=0.3, success=True)
    await prov.record_components(0, ["RMSNorm"])
    await prov.record_components(1, ["RMSNorm", "GELU"])
    stats = await prov.get_component_stats()
    rmsnorm = [s for s in stats if s["component"] == "RMSNorm"]
    assert len(rmsnorm) == 1
    assert rmsnorm[0]["count"] == 2


@skip_no_pg
@pytest.mark.asyncio
async def test_get_similar(prov_store):
    store, prov, al, pool = prov_store
    base = "class Model:\n    def forward(self, x): return self.linear(x)"
    similar = "class Model:\n    def forward(self, x): return self.linear(x) + self.bias"
    different = "import numpy\ndef compute(): return numpy.zeros(100)"
    await _add_exp(pool, 0, code=base)
    await _add_exp(pool, 1, code=similar)
    await _add_exp(pool, 2, code=different)
    results = await prov.get_similar(0)
    assert len(results) == 2
    assert results[0]["target_id"] == 1
