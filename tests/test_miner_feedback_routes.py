"""End-to-end-ish tests for the /miners/me/* + /tasks/{task}/frontier
routes.

We stub the asyncpg pool with a small fake so the test runs without
Postgres.  The middleware's bearer-lookup path is exercised, plus the
SQL parameterisation and JSON shaping.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.auth import hash_api_key


# ── Fake pool ────────────────────────────────────────────────────────


class _Row(dict):
    """asyncpg.Record-like dict — supports both ``row[k]`` and ``row.keys()``."""


class _FakePool:
    """Records calls and answers fetch* with scripted rows."""

    def __init__(self):
        self.fetch_rows: list[list[_Row]] = []
        self.fetchrow_rows: list[_Row | None] = []
        self.fetchval_rows: list = []
        self.executed: list[tuple[str, tuple]] = []

    async def fetch(self, sql, *args):
        self.executed.append((sql, args))
        if self.fetch_rows:
            return self.fetch_rows.pop(0)
        return []

    async def fetchrow(self, sql, *args):
        self.executed.append((sql, args))
        if self.fetchrow_rows:
            return self.fetchrow_rows.pop(0)
        return None

    async def fetchval(self, sql, *args):
        self.executed.append((sql, args))
        if self.fetchval_rows:
            return self.fetchval_rows.pop(0)
        return None

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "UPDATE 1"


@contextmanager
def _wire_app(monkeypatch):
    """Build a fresh FastAPI app with our routes + fake pool."""
    # Force a clean import so previous tests' state doesn't leak.
    from database import miner_feedback as mf, server as srv

    pool = _FakePool()
    monkeypatch.setattr(srv, "_pool", pool)
    app = FastAPI()
    mf.include_miner_feedback_routes(app)

    # Replicate the bearer middleware locally so we don't pull the full
    # validator surface in.
    @app.middleware("http")
    async def _bearer(request, call_next):
        if request.url.path.startswith("/miners/me/"):
            return await srv._bearer_auth_then_call(request, call_next)
        return await call_next(request)

    yield app, pool


# ── lookup_bearer test scaffolding ──────────────────────────────────


def _install_bearer_identity(monkeypatch, ident_dict):
    """Patch shared.miner_auth.lookup_bearer to return ``ident_dict`` (or
    ``None`` for unknown)."""
    from shared import miner_auth

    async def fake_lookup(pool, token):
        if token == "good":
            return miner_auth.MinerIdentity(**ident_dict)
        return None

    monkeypatch.setattr(miner_auth, "lookup_bearer", fake_lookup)


def _ident(hotkey="hk-1", miner_id="m1"):
    return {
        "miner_id": miner_id, "key_id": "k1",
        "scope": "miner", "hotkey": hotkey, "name": "alice",
    }


# ── /miners/me/submissions ──────────────────────────────────────────


def test_submissions_requires_bearer(monkeypatch):
    _install_bearer_identity(monkeypatch, _ident())
    with _wire_app(monkeypatch) as (app, _):
        client = TestClient(app)
        r = client.get("/miners/me/submissions")
    assert r.status_code == 401


def test_submissions_rejects_unknown_token(monkeypatch):
    _install_bearer_identity(monkeypatch, _ident())
    with _wire_app(monkeypatch) as (app, _):
        client = TestClient(app)
        r = client.get(
            "/miners/me/submissions",
            headers={"Authorization": "Bearer wrong"},
        )
    assert r.status_code == 403


def test_submissions_returns_rows_scoped_to_hotkey(monkeypatch):
    _install_bearer_identity(monkeypatch, _ident(hotkey="hk-A"))
    with _wire_app(monkeypatch) as (app, pool):
        pool.fetch_rows = [[
            _Row(
                submission_id=1, round_id=42, task_name="ts_forecasting",
                prompt_id="p-1", architecture_code="class M: pass",
                motivation="m", reasoning="r", tool_calls=[],
                created_at=1700000000.0,
            ),
        ]]
        client = TestClient(app)
        r = client.get(
            "/miners/me/submissions?limit=10&task=ts_forecasting",
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200
    payload = r.json()
    assert len(payload["submissions"]) == 1
    sub = payload["submissions"][0]
    assert sub["prompt_id"] == "p-1"
    assert sub["task_name"] == "ts_forecasting"
    # First arg of the query is the hotkey filter — confirm scoping.
    sql, args = pool.executed[0]
    assert args[0] == "hk-A"


def test_submissions_empty_when_no_hotkey_on_identity(monkeypatch):
    _install_bearer_identity(monkeypatch, _ident(hotkey=""))
    with _wire_app(monkeypatch) as (app, _):
        client = TestClient(app)
        r = client.get(
            "/miners/me/submissions",
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200
    assert r.json() == {"submissions": []}


# ── /miners/me/results ──────────────────────────────────────────────


def test_results_join_includes_scores(monkeypatch):
    _install_bearer_identity(monkeypatch, _ident(hotkey="hk-A"))
    with _wire_app(monkeypatch) as (app, pool):
        pool.fetch_rows = [[
            _Row(
                submission_id=7, round_id=12, task_name="ts_forecasting",
                prompt_id="p-7", architecture_code="x", motivation="m",
                metric=0.123, score=0.45, success=True,
                objectives={"flops_equivalent_size": 1_000_000, "crps": 0.123},
                created_at=1700000000.0,
            ),
        ]]
        client = TestClient(app)
        r = client.get(
            "/miners/me/results",
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200
    row = r.json()["results"][0]
    assert row["prompt_id"] == "p-7"
    assert row["scores"]["raw_score"] == 0.45
    assert row["scores"]["metric"] == 0.123
    assert row["scores"]["flops_equivalent_size"] == 1_000_000
    assert row["scores"]["success"] is True


def test_results_handles_objectives_as_json_string(monkeypatch):
    _install_bearer_identity(monkeypatch, _ident(hotkey="hk-A"))
    with _wire_app(monkeypatch) as (app, pool):
        pool.fetch_rows = [[
            _Row(
                submission_id=1, round_id=1, task_name="t",
                prompt_id=None, architecture_code="", motivation="",
                metric=None, score=0.0, success=False,
                objectives='{"flops_equivalent_size": 5}',
                created_at=0.0,
            ),
        ]]
        client = TestClient(app)
        r = client.get(
            "/miners/me/results",
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200
    assert r.json()["results"][0]["scores"]["flops_equivalent_size"] == 5


# ── /miners/me/summary ──────────────────────────────────────────────


def test_summary_aggregates(monkeypatch):
    _install_bearer_identity(monkeypatch, _ident(hotkey="hk-A"))
    with _wire_app(monkeypatch) as (app, pool):
        pool.fetchrow_rows = [
            _Row(total=10, last_round=42),
            _Row(mean_score=0.55, successes=3, n=8),
        ]
        client = TestClient(app)
        r = client.get(
            "/miners/me/summary",
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200
    s = r.json()
    assert s["total_submissions"] == 10
    assert s["last_round_id"] == 42
    assert s["mean_score_recent"] == 0.55
    assert s["successes_recent"] == 3
    assert s["recent_window"] == 8


def test_summary_zero_when_no_rows(monkeypatch):
    _install_bearer_identity(monkeypatch, _ident(hotkey="hk-A"))
    with _wire_app(monkeypatch) as (app, pool):
        pool.fetchrow_rows = [
            _Row(total=0, last_round=None),
            _Row(mean_score=None, successes=0, n=0),
        ]
        client = TestClient(app)
        r = client.get(
            "/miners/me/summary",
            headers={"Authorization": "Bearer good"},
        )
    assert r.status_code == 200
    s = r.json()
    assert s["total_submissions"] == 0
    assert s["last_round_id"] is None
    assert s["mean_score_recent"] is None


# ── /tasks/{task}/frontier (public) ─────────────────────────────────


def test_task_frontier_public_no_auth(monkeypatch):
    with _wire_app(monkeypatch) as (app, pool):
        pool.fetch_rows = [[
            _Row(metric=0.1, objectives={"flops_equivalent_size": 1_000_000}, score=0.9),
            _Row(metric=0.2, objectives='{"flops_equivalent_size": 5_000_000}', score=0.7),
        ]]
        client = TestClient(app)
        r = client.get("/tasks/ts_forecasting/frontier?limit=2")
    assert r.status_code == 200
    payload = r.json()
    assert payload["task"] == "ts_forecasting"
    assert len(payload["points"]) == 2
    assert payload["points"][0]["flops"] == 1_000_000
    assert payload["points"][0]["metric"] == 0.1


def test_task_frontier_caps_limit(monkeypatch):
    with _wire_app(monkeypatch) as (app, pool):
        pool.fetch_rows = [[]]
        client = TestClient(app)
        client.get("/tasks/foo/frontier?limit=99999")
        # Last arg of the SQL call is the LIMIT — confirm it's capped at 500.
        _, args = pool.executed[-1]
        assert args[-1] == 500
