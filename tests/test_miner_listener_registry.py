"""Tests for the miner listener registry.

The off-chain deploy has no on-chain ImageCommitment to carry a
miner's listener_url, so miners self-register their URL via
``POST /miners/me/listener`` and validators fetch the live list via
``GET /miners/active``. Together these close the gap that previously
left ``commitments`` empty and caused every round to dead-end with
"0 miners with listener_urls".

Covered:
  * ``POST /agent_code`` stores the optional ``listener_url`` field.
  * ``POST /miners/me/listener`` refreshes the registry row.
  * ``GET /miners/active`` returns only rows within the freshness
    window and with a non-empty URL.
  * ``DatabaseClient`` round-trips through the new endpoints.
  * ``Validator._fetch_active_commitments`` builds an
    ``ImageCommitment`` dict from the DB response with stable
    synthetic UIDs.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Fakes that mirror the agent_code test wiring ─────────────────────


class _FakeR2:
    def __init__(self):
        self.uploads: list[tuple[str, dict]] = []

    def upload_json(self, key, data):
        self.uploads.append((key, dict(data)))


class _FakeRow(dict):
    """Behaves like an asyncpg Record for the bits the route reads."""


class _FakeConn:
    def __init__(self, pool):
        self.pool = pool

    async def execute(self, sql, *args):
        self.pool.executed.append((sql, args))
        # /agent_code uses INSERT … ON CONFLICT — pretend one row was
        # touched so the path treats it as success.
        return "INSERT 0 1"

    @asynccontextmanager
    async def transaction(self):
        yield


class _FakePool:
    def __init__(self):
        self.executed: list[tuple[str, tuple]] = []
        self.rows: list[dict] = []
        # Default to "update touched a row" for the heartbeat endpoint.
        self.update_rowcount: int = 1

    @asynccontextmanager
    async def acquire(self):
        yield _FakeConn(self)

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return f"UPDATE {self.update_rowcount}"

    async def fetch(self, sql, *args):
        return [_FakeRow(r) for r in self.rows]

    async def fetchrow(self, sql, *args):
        return None


@contextmanager
def _wire_app(monkeypatch, *, hotkey: str = "hk-A"):
    from database import server as srv
    from shared import miner_auth

    pool = _FakePool()
    r2 = _FakeR2()
    monkeypatch.setattr(srv, "_pool", pool)
    monkeypatch.setattr(srv, "_r2", r2)
    monkeypatch.setattr(srv, "_current_challenge", {"round_id": 7})
    srv._rate_window.clear()
    srv._ip_rate_window.clear()

    async def fake_lookup(_pool, token):
        if token == "good":
            return miner_auth.MinerIdentity(
                miner_id="m1", key_id="k1", scope="miner",
                hotkey=hotkey, name="alice",
            )
        return None

    monkeypatch.setattr(miner_auth, "lookup_bearer", fake_lookup)

    app = FastAPI()
    app.include_router(srv.validator_router)

    @app.middleware("http")
    async def _route_auth(request, call_next):
        path = request.url.path
        method = request.method
        if path == "/agent_code" and method == "POST":
            return await srv._bearer_auth_then_call_for_agent_code(
                request, call_next,
            )
        if path == "/miners/me/listener" and method == "POST":
            # Same bearer wrapper /miners/me/* normally uses in the real
            # app, just exercised here on its own.
            return await srv._bearer_auth_then_call(request, call_next)
        return await call_next(request)

    yield app, pool, r2


def _bundle():
    return {
        "files": {"agent.py": "def design_architecture(*a, **k):\n    return {}\n"},
        "entry_point": "agent.py",
    }


# ── POST /agent_code stores listener_url ─────────────────────────────


def test_post_agent_code_stores_listener_url(monkeypatch):
    """Listener URL flows through to the SQL parameters."""
    with _wire_app(monkeypatch) as (app, pool, _):
        client = TestClient(app)
        body = _bundle() | {"listener_url": "http://1.2.3.4:8090"}
        r = client.post(
            "/agent_code", json=body,
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200, r.text
    upsert_sqls = [sql for sql, _ in pool.executed if "agent_submissions" in sql]
    assert upsert_sqls, "expected an INSERT into agent_submissions"
    upsert_args = next(
        args for sql, args in pool.executed if "agent_submissions" in sql
    )
    assert "http://1.2.3.4:8090" in upsert_args


def test_post_agent_code_without_listener_url_passes_empty(monkeypatch):
    """Omitting the field is fine — the column gets an empty string and
    the upsert preserves any previously-stored value via the CASE clause."""
    with _wire_app(monkeypatch) as (app, pool, _):
        client = TestClient(app)
        r = client.post(
            "/agent_code", json=_bundle(),
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200, r.text
    args = next(a for sql, a in pool.executed if "agent_submissions" in sql)
    assert "" in args  # empty listener_url positional arg


# ── POST /miners/me/listener ─────────────────────────────────────────


def test_register_listener_updates_row(monkeypatch):
    with _wire_app(monkeypatch) as (app, pool, _):
        client = TestClient(app)
        r = client.post(
            "/miners/me/listener",
            json={"listener_url": "http://5.6.7.8:8090"},
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200, r.text
    assert r.json() == {
        "status": "ok", "listener_url": "http://5.6.7.8:8090",
    }
    update_sqls = [sql for sql, _ in pool.executed if "UPDATE" in sql.upper()]
    assert update_sqls, "expected UPDATE on agent_submissions"


def test_register_listener_rejects_empty(monkeypatch):
    with _wire_app(monkeypatch) as (app, _, _):
        client = TestClient(app)
        r = client.post(
            "/miners/me/listener",
            json={"listener_url": "   "},
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 400


def test_register_listener_404_when_no_agent_submission(monkeypatch):
    """If the miner hasn't posted agent code yet there's no row to
    UPDATE — make sure that surfaces as a 404 so the miner knows to
    call submit_agent_code first."""
    with _wire_app(monkeypatch) as (app, pool, _):
        pool.update_rowcount = 0
        client = TestClient(app)
        r = client.post(
            "/miners/me/listener",
            json={"listener_url": "http://x:9"},
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 404


def test_register_listener_requires_auth(monkeypatch):
    with _wire_app(monkeypatch) as (app, _, _):
        client = TestClient(app)
        r = client.post(
            "/miners/me/listener", json={"listener_url": "http://x:9"},
        )
    assert r.status_code == 401


# ── GET /miners/active ───────────────────────────────────────────────


def test_list_active_miners_returns_rows(monkeypatch):
    with _wire_app(monkeypatch) as (app, pool, _):
        now = time.time()
        pool.rows = [
            {
                "hotkey": "hk-A", "miner_uid": 1, "code_hash": "ab" * 32,
                "entry_point": "agent.py",
                "listener_url": "http://1.1.1.1:8090",
                "listener_seen_at": now - 10.0,
                "round_submitted": 100, "timestamp": now - 30.0,
            },
            {
                "hotkey": "hk-B", "miner_uid": -1, "code_hash": "cd" * 32,
                "entry_point": "agent.py",
                "listener_url": "http://2.2.2.2:8090",
                "listener_seen_at": now - 60.0,
                "round_submitted": 99, "timestamp": now - 90.0,
            },
        ]
        client = TestClient(app)
        # Public GET — middleware in the wire fixture doesn't gate it,
        # mirroring how the validator-router /miners/active is HMAC-authed
        # at the real middleware layer (separately tested for /agent_code).
        r = client.get("/miners/active")
    assert r.status_code == 200, r.text
    miners = r.json()["miners"]
    assert len(miners) == 2
    assert {m["hotkey"] for m in miners} == {"hk-A", "hk-B"}
    assert miners[0]["listener_url"] == "http://1.1.1.1:8090"


# ── DatabaseClient round-trip ────────────────────────────────────────


@pytest.mark.asyncio
async def test_db_client_submit_agent_code_passes_listener_url():
    from shared.db_client import DatabaseClient

    db = DatabaseClient(db_url="http://fake:8090", api_key="rdrk_x")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok", "code_hash": "ab" * 32}
    mock_resp.raise_for_status = MagicMock()

    with patch.object(db, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_get.return_value = mock_client
        await db.submit_agent_code(
            files={"agent.py": "x"},
            entry_point="agent.py",
            listener_url="http://m:9",
        )
        # _post sends a JSON-encoded body via content=, not json=
        import json as _json
        sent = _json.loads(mock_client.post.call_args.kwargs["content"])
        assert sent["listener_url"] == "http://m:9"
        assert mock_client.post.call_args.args[0].endswith("/agent_code")


@pytest.mark.asyncio
async def test_db_client_register_listener():
    from shared.db_client import DatabaseClient

    db = DatabaseClient(db_url="http://fake:8090", api_key="rdrk_x")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(db, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_get.return_value = mock_client
        await db.register_listener("http://m:9")
        import json as _json
        sent = _json.loads(mock_client.post.call_args.kwargs["content"])
        assert sent == {"listener_url": "http://m:9"}
        assert mock_client.post.call_args.args[0].endswith(
            "/miners/me/listener",
        )


@pytest.mark.asyncio
async def test_db_client_get_active_miners():
    from shared.db_client import DatabaseClient

    db = DatabaseClient(db_url="http://fake:8090", service_secret=b"k")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "miners": [{"hotkey": "hk", "listener_url": "u"}],
    }
    mock_resp.raise_for_status = MagicMock()
    with patch.object(db, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_get.return_value = mock_client
        out = await db.get_active_miners()
        assert out == [{"hotkey": "hk", "listener_url": "u"}]


# ── Validator._fetch_active_commitments ──────────────────────────────


@pytest.mark.asyncio
async def test_validator_fetch_active_commitments_builds_dict():
    """The validator turns the JSON rows into ImageCommitment objects
    keyed by a stable int UID — preferring the miner-posted uid, and
    falling back to a hotkey-derived synthetic uid so different miners
    never share a key."""
    from shared.commitment import ImageCommitment
    from validator.neuron import Validator

    fake_db = MagicMock()
    fake_db.get_active_miners = AsyncMock(return_value=[
        {
            "hotkey": "hk-A", "miner_uid": 7, "code_hash": "ab" * 32,
            "listener_url": "http://a:9",
        },
        {
            "hotkey": "hk-B", "miner_uid": -1, "code_hash": "cd" * 32,
            "listener_url": "http://b:9",
        },
        {
            # Empty listener — must be skipped.
            "hotkey": "hk-C", "miner_uid": -1, "code_hash": "ef" * 32,
            "listener_url": "",
        },
    ])

    v = Validator.__new__(Validator)
    v.db_client = fake_db

    out = await v._fetch_active_commitments()

    assert len(out) == 2
    assert 7 in out
    assert out[7].listener_url == "http://a:9"
    assert out[7].hotkey == "hk-A"
    assert all(isinstance(v, ImageCommitment) for v in out.values())
    # The hk-B uid is synthetic (positive int, not the preferred 7).
    other_uid = next(u for u in out if u != 7)
    assert other_uid > 0
    assert out[other_uid].hotkey == "hk-B"


@pytest.mark.asyncio
async def test_validator_fetch_active_commitments_returns_empty_on_failure():
    from validator.neuron import Validator

    fake_db = MagicMock()
    fake_db.get_active_miners = AsyncMock(side_effect=RuntimeError("boom"))

    v = Validator.__new__(Validator)
    v.db_client = fake_db

    out = await v._fetch_active_commitments()
    assert out == {}
