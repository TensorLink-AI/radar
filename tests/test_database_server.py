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
    _ip_rate_window, _ip_rate_lock, _ip_rate_limit,
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

    for _ in range(_ip_rate_limit):
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
    assert _MAX_AGENT_FILES == 25
    assert _MAX_AGENT_FILE_BYTES == 100_000


# ── Access log experiment-ID capture ───────────────────────────


class _RecordingAccessLogger:
    def __init__(self):
        self.calls: list[dict] = []

    async def log_access(self, **kw):
        self.calls.append(kw)


def _install_access_logger():
    from database import server as srv
    logger = _RecordingAccessLogger()
    srv._access_logger = logger
    return logger


def _uninstall_access_logger():
    from database import server as srv
    srv._access_logger = None


def test_access_log_captures_ids_from_experiment_detail():
    _setup()
    logger = _install_access_logger()
    try:
        client = TestClient(app)
        r = client.get("/experiments/0")
        assert r.status_code == 200
        # Body should still be delivered intact
        assert r.json()["name"] == "exp_0"
        # asyncio.create_task was scheduled — give it a tick to run
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        loop.run_until_complete(_asyncio.sleep(0.05))
        loop.close()
        assert logger.calls, "middleware did not log the request"
        call = logger.calls[-1]
        assert 0 in call["experiment_ids"]
        assert call["endpoint"] == "/experiments/0"
    finally:
        _uninstall_access_logger()


def test_access_log_captures_ids_from_list_response():
    _setup()
    logger = _install_access_logger()
    try:
        client = TestClient(app)
        r = client.get("/experiments/recent?n=2")
        assert r.status_code == 200
        assert len(r.json()) == 2
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        loop.run_until_complete(_asyncio.sleep(0.05))
        loop.close()
        # Captured IDs match the response payload
        payload_ids = {item["index"] for item in r.json()}
        captured = set(logger.calls[-1]["experiment_ids"])
        assert captured == payload_ids
    finally:
        _uninstall_access_logger()


def test_access_log_skips_non_experiment_paths():
    _setup()
    logger = _install_access_logger()
    # Other tests in the suite exercise the same global ``app`` from the
    # TestClient ('testclient' IP), so the per-IP rate window can be near
    # the threshold by the time we get here.
    with _ip_rate_lock:
        _ip_rate_window.pop("testclient", None)
    try:
        client = TestClient(app)
        r = client.get("/provenance/component_stats")
        assert r.status_code == 200
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        loop.run_until_complete(_asyncio.sleep(0.05))
        loop.close()
        # Still logs the access, but with no experiment IDs extracted
        assert logger.calls
        assert logger.calls[-1]["experiment_ids"] == []
    finally:
        _uninstall_access_logger()


def test_access_log_runs_on_proxy_api_key_path():
    """Agent traffic arrives via the validator proxy with X-Radar-API-Key.
    The trusted-proxy branch must still fall through to the access-logger
    so miner_access_log.experiment_ids gets populated (otherwise the
    provenance dashboard stays empty).
    """
    from config import Config
    from database import server as srv

    _setup()
    logger = _install_access_logger()
    prev_key = Config.DB_API_KEY
    Config.DB_API_KEY = "test-key"
    # Avoid both per-hotkey and per-IP rate-limit windows from other tests
    srv._rate_window.clear()
    with _ip_rate_lock:
        _ip_rate_window.pop("testclient", None)
    try:
        client = TestClient(app)
        r = client.get(
            "/experiments/0",
            headers={
                "X-Radar-API-Key": "test-key",
                "X-Miner-Hotkey": "miner_hk_abc",
                "X-Miner-UID": "7",
            },
        )
        assert r.status_code == 200
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        loop.run_until_complete(_asyncio.sleep(0.05))
        loop.close()
        assert logger.calls, "proxy path did not reach access logger"
        call = logger.calls[-1]
        assert call["hotkey"] == "miner_hk_abc"
        assert call["miner_uid"] == 7
        assert 0 in call["experiment_ids"]
    finally:
        Config.DB_API_KEY = prev_key
        _uninstall_access_logger()


# ── Agent code submission history ──────────────────────────────


class _FakeR2:
    """In-memory R2 stand-in for agent-code submission tests."""

    def __init__(self):
        self._blobs: dict[str, dict] = {}
        self.uploads: list[str] = []

    def upload_json(self, key, data):
        self._blobs[key] = dict(data)
        self.uploads.append(key)
        return True

    def download_json(self, key):
        blob = self._blobs.get(key)
        return dict(blob) if blob is not None else None


class _FakePool:
    """In-memory stand-in for the asyncpg pool used by agent_code routes.

    Only implements the three statements the submission + read endpoints
    exercise (upsert into agent_submissions, append to history, fetch).
    """

    def __init__(self):
        self.registry: dict[str, dict] = {}
        self.history: list[dict] = []

    class _Conn:
        def __init__(self, parent):
            self.parent = parent

        async def execute(self, sql, *params):
            self.parent._execute(sql, *params)

        def transaction(self):
            parent = self.parent

            class _Tx:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

            return _Tx()

    class _Acquire:
        def __init__(self, parent):
            self.parent = parent

        async def __aenter__(self):
            return _FakePool._Conn(self.parent)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def acquire(self):
        return _FakePool._Acquire(self)

    def _execute(self, sql: str, *params):
        sql_l = sql.lower()
        import time as _time
        if "insert into agent_submissions" in sql_l:
            hk, uid, h, ep, r2, rs = params
            self.registry[hk] = {
                "hotkey": hk, "miner_uid": uid, "code_hash": h,
                "entry_point": ep, "r2_key": r2, "round_submitted": rs,
                "timestamp": _time.time(),
            }
        elif "insert into agent_submission_history" in sql_l:
            hk, uid, h, ep, r2, rs = params
            self.history.append({
                "hotkey": hk, "miner_uid": uid, "code_hash": h,
                "entry_point": ep, "r2_key": r2, "round_submitted": rs,
                "timestamp": _time.time(),
            })

    async def execute(self, sql, *params):
        self._execute(sql, *params)

    async def fetch(self, sql: str, *params):
        sql_l = sql.lower()
        if "from agent_submission_history" in sql_l and "where hotkey" in sql_l:
            hk = params[0]
            limit = params[1] if len(params) > 1 else 100
            rows = [r for r in self.history if r["hotkey"] == hk]
            rows.sort(key=lambda r: r["timestamp"], reverse=True)
            return rows[:limit]
        return []

    async def fetchrow(self, sql: str, *params):
        sql_l = sql.lower()
        if "from agent_submission_history" in sql_l and "code_hash = $1" in sql_l:
            target = params[0]
            matches = [r for r in self.history if r["code_hash"] == target]
            if not matches:
                return None
            matches.sort(key=lambda r: r["timestamp"], reverse=True)
            return matches[0]
        return None


def _call_async(coro):
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_request(hotkey: str = "hk_test"):
    """Construct a minimal fake Request with caller_hotkey in state."""

    class _State:
        pass

    class _Request:
        def __init__(self):
            self.state = _State()
            self.state.caller_hotkey = hotkey

    return _Request()


def test_agent_submission_writes_history_and_content_addressed_blob():
    from database import server as srv
    from database.server import SubmitAgentCodeRequest, submit_agent_code

    fake_r2 = _FakeR2()
    fake_pool = _FakePool()
    srv._r2 = fake_r2
    srv._pool = fake_pool
    srv._current_challenge = {"round_id": 42}
    try:
        files = {"agent.py": "def design_architecture():\n    return 'v1'\n"}
        req = SubmitAgentCodeRequest(files=files, entry_point="agent.py")
        result = _call_async(submit_agent_code(_make_request("hk_alice"), req))
        assert result["status"] == "ok"
        code_hash = result["code_hash"]
        assert result["round_submitted"] == 42
        # Both R2 keys were written
        assert f"agents/hk_alice/{code_hash}.json" in fake_r2.uploads
        assert "agents/hk_alice/latest.json" in fake_r2.uploads
        # Registry + history got populated
        assert fake_pool.registry["hk_alice"]["code_hash"] == code_hash
        assert fake_pool.registry["hk_alice"]["round_submitted"] == 42
        assert len(fake_pool.history) == 1
        assert fake_pool.history[0]["code_hash"] == code_hash
        assert fake_pool.history[0]["r2_key"] == f"agents/hk_alice/{code_hash}.json"
    finally:
        srv._r2 = None
        srv._pool = None
        srv._current_challenge = None


def test_agent_submission_preserves_previous_via_content_addressing():
    from database import server as srv
    from database.server import (
        SubmitAgentCodeRequest, submit_agent_code,
        get_agent_code_by_hash, get_agent_code_history,
    )

    fake_r2 = _FakeR2()
    fake_pool = _FakePool()
    srv._r2 = fake_r2
    srv._pool = fake_pool
    srv._current_challenge = {"round_id": 10}
    try:
        # First submission
        req1 = SubmitAgentCodeRequest(
            files={"agent.py": "def design_architecture():\n    return 'v1'\n"},
            entry_point="agent.py",
        )
        r1 = _call_async(submit_agent_code(_make_request("hk_bob"), req1))
        hash_v1 = r1["code_hash"]

        # Second submission (different content → different hash)
        srv._current_challenge = {"round_id": 11}
        req2 = SubmitAgentCodeRequest(
            files={"agent.py": "def design_architecture():\n    return 'v2'\n"},
            entry_point="agent.py",
        )
        r2 = _call_async(submit_agent_code(_make_request("hk_bob"), req2))
        hash_v2 = r2["code_hash"]
        assert hash_v1 != hash_v2

        # by_hash can still retrieve the first submission's exact bytes
        bundle_v1 = _call_async(get_agent_code_by_hash(hash_v1))
        assert "return 'v1'" in bundle_v1["files"]["agent.py"]
        assert bundle_v1["code_hash"] == hash_v1

        # History lists both submissions, most recent first, with round IDs
        hist = _call_async(get_agent_code_history("hk_bob", limit=100))
        rounds = [s["round_submitted"] for s in hist["submissions"]]
        assert rounds == [11, 10]
        hashes = [s["code_hash"] for s in hist["submissions"]]
        assert hashes == [hash_v2, hash_v1]
    finally:
        srv._r2 = None
        srv._pool = None
        srv._current_challenge = None


def test_agent_code_by_hash_unknown_returns_404():
    from database import server as srv
    from database.server import get_agent_code_by_hash
    from fastapi import HTTPException

    srv._r2 = _FakeR2()
    srv._pool = _FakePool()
    try:
        with pytest.raises(HTTPException) as exc:
            _call_async(get_agent_code_by_hash("nonexistent_hash"))
        assert exc.value.status_code == 404
    finally:
        srv._r2 = None
        srv._pool = None
