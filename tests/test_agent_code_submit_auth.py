"""POST /agent_code auth tests.

Miners submit agent code with operator-issued bearer tokens, not the
shared HMAC service key — the validator proxy intentionally does not
forward this endpoint, so the DB server must accept Authorization:
Bearer here while still requiring HMAC on the GET reads used by
validators.

We build a fresh FastAPI app that wires the real
``_bearer_auth_then_call_for_agent_code`` helper into a minimal
middleware and mounts the real ``submit_agent_code`` route — same
pattern as ``test_miner_feedback_routes.py`` so we don't tangle
test state with the module-level ``database.server.app`` singleton.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Fakes ───────────────────────────────────────────────────────────


class _FakeR2:
    def __init__(self):
        self.uploads: list[tuple[str, dict]] = []

    def upload_json(self, key, data):
        self.uploads.append((key, dict(data)))


class _FakeConn:
    def __init__(self):
        self.executed: list[tuple[str, tuple]] = []

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "INSERT 0 1"

    @asynccontextmanager
    async def transaction(self):
        yield


class _FakePool:
    def __init__(self):
        self.conn = _FakeConn()

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


@contextmanager
def _wire_app(monkeypatch, *, with_hotkey: str = "hk-1"):
    """Build a minimal FastAPI app with the real bearer-auth helper and
    the real /agent_code POST handler, but no global app reuse."""
    from database import server as srv
    from shared import miner_auth

    pool = _FakePool()
    r2 = _FakeR2()
    monkeypatch.setattr(srv, "_pool", pool)
    monkeypatch.setattr(srv, "_r2", r2)
    monkeypatch.setattr(srv, "_current_challenge", {"round_id": 5})
    # Reset rate-limit windows so a stale window from another test doesn't
    # poison this run.
    srv._rate_window.clear()
    srv._ip_rate_window.clear()

    async def fake_lookup(_pool, token):
        if token == "good":
            return miner_auth.MinerIdentity(
                miner_id="m1", key_id="k1", scope="miner",
                hotkey=with_hotkey, name="alice",
            )
        return None

    monkeypatch.setattr(miner_auth, "lookup_bearer", fake_lookup)

    app = FastAPI()
    app.include_router(srv.validator_router)

    @app.middleware("http")
    async def _route_auth(request, call_next):
        path = request.url.path
        if path == "/agent_code" and request.method == "POST":
            return await srv._bearer_auth_then_call_for_agent_code(
                request, call_next,
            )
        return await call_next(request)

    yield app, pool, r2


# ── Tests ───────────────────────────────────────────────────────────


def _bundle():
    return {
        "files": {"agent.py": "def design_architecture(*a, **k):\n    return {}\n"},
        "entry_point": "agent.py",
    }


def test_post_agent_code_with_bearer_succeeds(monkeypatch):
    """Miner bearer token → 200, hotkey from identity is stored."""
    with _wire_app(monkeypatch, with_hotkey="hk-A") as (app, pool, r2):
        client = TestClient(app)
        r = client.post(
            "/agent_code", json=_bundle(),
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["code_hash"]
    # Both immutable + latest blobs uploaded under the resolved hotkey.
    keys = [k for k, _ in r2.uploads]
    assert any(k.startswith("agents/hk-A/") for k in keys)
    assert "agents/hk-A/latest.json" in keys


def test_post_agent_code_missing_auth_returns_401(monkeypatch):
    with _wire_app(monkeypatch) as (app, _, _):
        client = TestClient(app)
        r = client.post("/agent_code", json=_bundle())
    assert r.status_code == 401
    assert "Bearer" in r.json()["error"]


def test_post_agent_code_unknown_token_returns_403(monkeypatch):
    with _wire_app(monkeypatch) as (app, _, _):
        client = TestClient(app)
        r = client.post(
            "/agent_code", json=_bundle(),
            headers={"Authorization": "Bearer wrong"},
        )
    assert r.status_code == 403
    err = r.json()["error"]
    assert "Unknown" in err or "revoked" in err


def test_post_agent_code_rejects_identity_without_hotkey(monkeypatch):
    """A miner registered without a hotkey can't write — agent_submissions
    is keyed by hotkey, so a blank one would silently overwrite another
    row in the worst case."""
    with _wire_app(monkeypatch, with_hotkey="") as (app, _, _):
        client = TestClient(app)
        r = client.post(
            "/agent_code", json=_bundle(),
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 400
    assert "hotkey" in r.json()["error"]
