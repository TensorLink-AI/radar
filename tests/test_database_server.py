"""Tests for database/server.py — centralized DB API endpoints.

Uses a mock PgExperimentStore (async methods returning test data) so
no actual Postgres is needed.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from shared.database import DataElement
from database.server import (
    app, set_db, set_challenge, set_frontier,
    _ip_rate_window, _ip_rate_lock, _IP_RATE_LIMIT,
    _check_ip_rate_limit, _check_nonce, _nonce_cache, _nonce_lock,
    _MAX_AGENT_FILES, _MAX_AGENT_FILE_BYTES,
)


class MockStore:
    """Minimal async mock of PgExperimentStore for route testing."""

    def __init__(self):
        self._elements = []
        self.provenance = MockProvenance()

    async def get_size(self):
        return len(self._elements)

    async def get(self, index: int) -> Optional[DataElement]:
        for e in self._elements:
            if e.index == index:
                return e
        return None

    async def get_recent(self, n: int = 5, **kw) -> list[DataElement]:
        return list(reversed(self._elements[-n:]))

    async def get_failures(self, n: int = 10, **kw) -> list[DataElement]:
        return [e for e in reversed(self._elements) if not e.success][:n]

    async def get_pareto_elements(self, **kw) -> list[DataElement]:
        return [e for e in self._elements if e.success and e.metric is not None]

    async def stats(self, **kw) -> dict:
        total = len(self._elements)
        successful = sum(1 for e in self._elements if e.success)
        return {"total": total, "successful": successful, "failed": total - successful,
                "best_metric": None, "worst_metric": None, "mean_metric": None, "max_generation": 0}

    async def get_tasks(self) -> list[str]:
        return sorted(set(e.task for e in self._elements if e.task))

    async def stats_by_task(self) -> dict:
        return {}

    async def get_family_summary(self, **kw) -> list[dict]:
        return []

    async def get_diff(self, index: int) -> Optional[str]:
        return "diff output" if any(e.index == index for e in self._elements) else None

    async def get_diff_between(self, a: int, b: int) -> Optional[str]:
        return "diff"

    async def get_lineage(self, index: int) -> list[DataElement]:
        elem = await self.get(index)
        return [elem] if elem else []

    async def get_lineage_diffs(self, index: int) -> list[dict]:
        return []

    async def search(self, query: str, **kw) -> list[DataElement]:
        return [e for e in self._elements if query.lower() in e.motivation.lower()]

    async def add(self, element: DataElement) -> int:
        element.index = len(self._elements)
        self._elements.append(element)
        return element.index


class MockProvenance:
    async def get_influences(self, eid):
        return []

    async def get_impact(self, eid):
        return []

    async def get_similar(self, eid, top_k=10):
        return []

    async def get_component_experiments(self, comp):
        return []

    async def get_component_stats(self):
        return []

    async def get_dead_ends(self, task=""):
        return []

    async def get_experiment_graph(self, eid, depth=3):
        return {"experiment_id": eid, "influences": [], "impact": [], "components": []}

    async def record_components(self, eid, comps):
        pass

    async def record_round_context(self, rid, eid, ct="frontier"):
        pass


def _setup():
    store = MockStore()
    store._elements = [
        DataElement(index=0, name="exp_0", code="v0", motivation="baseline",
                    success=True, metric=1.0, task="ts"),
        DataElement(index=1, name="exp_1", code="v1", motivation="gated attention",
                    success=True, metric=0.9, parent=0, generation=1, task="ts"),
        DataElement(index=2, name="exp_2", code="v2", motivation="failure",
                    success=False, task="ts"),
    ]
    set_db(store)
    return store


def test_health():
    _setup()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_get_experiment():
    _setup()
    client = TestClient(app)
    r = client.get("/experiments/0")
    assert r.status_code == 200
    assert r.json()["name"] == "exp_0"


def test_get_experiment_not_found():
    _setup()
    client = TestClient(app)
    r = client.get("/experiments/999")
    assert r.status_code == 404


def test_get_recent():
    _setup()
    client = TestClient(app)
    r = client.get("/experiments/recent?n=2")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_failures():
    _setup()
    client = TestClient(app)
    r = client.get("/experiments/failures")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_get_stats():
    _setup()
    client = TestClient(app)
    r = client.get("/experiments/stats")
    assert r.status_code == 200
    assert r.json()["total"] == 3


def test_get_pareto():
    _setup()
    client = TestClient(app)
    r = client.get("/experiments/pareto")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_search():
    _setup()
    client = TestClient(app)
    r = client.post("/experiments/search", json={"query": "attention"})
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_challenge_no_active():
    _setup()
    set_challenge(None)
    client = TestClient(app)
    r = client.get("/challenge")
    assert r.status_code == 404


def test_challenge_active():
    _setup()
    set_challenge({"round_id": 1, "seed": 42})
    client = TestClient(app)
    r = client.get("/challenge")
    assert r.status_code == 200
    assert r.json()["round_id"] == 1


def test_frontier():
    _setup()
    set_frontier([{"code": "x"}])
    client = TestClient(app)
    r = client.get("/frontier")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_add_experiment():
    _setup()
    client = TestClient(app)
    r = client.post("/experiments/add", json={"data": {
        "name": "new_exp", "code": "test", "success": True, "metric": 0.5,
    }})
    assert r.status_code == 200
    assert "index" in r.json()


def test_provenance_influences():
    _setup()
    client = TestClient(app)
    r = client.get("/provenance/0/influences")
    assert r.status_code == 200


def test_provenance_component_stats():
    _setup()
    client = TestClient(app)
    r = client.get("/provenance/component_stats")
    assert r.status_code == 200


def test_record_components():
    _setup()
    client = TestClient(app)
    r = client.post("/provenance/record_components", json={
        "experiment_id": 0, "components": ["RMSNorm"],
    })
    assert r.status_code == 200


def test_record_context():
    _setup()
    client = TestClient(app)
    r = client.post("/provenance/record_context", json={
        "round_id": 1, "experiment_id": 0, "context_type": "frontier",
    })
    assert r.status_code == 200


def test_ip_rate_limit():
    """IP-based rate limiter blocks after threshold."""
    test_ip = "10.99.99.99"
    # Clear any state
    with _ip_rate_lock:
        _ip_rate_window.pop(test_ip, None)

    for _ in range(_IP_RATE_LIMIT):
        assert _check_ip_rate_limit(test_ip) is True
    # Next request should be blocked
    assert _check_ip_rate_limit(test_ip) is False

    # Clean up
    with _ip_rate_lock:
        _ip_rate_window.pop(test_ip, None)


def test_nonce_replay_protection():
    """Same nonce rejected within the tolerance window."""
    with _nonce_lock:
        _nonce_cache.clear()

    assert _check_nonce("nonce-aaa") is True
    assert _check_nonce("nonce-bbb") is True
    # Replay should fail
    assert _check_nonce("nonce-aaa") is False
    # New nonce still works
    assert _check_nonce("nonce-ccc") is True


def test_agent_code_file_limits():
    """Agent code rejects too many files or oversized files."""
    assert _MAX_AGENT_FILES == 10
    assert _MAX_AGENT_FILE_BYTES == 50_000
