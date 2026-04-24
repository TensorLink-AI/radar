"""End-to-end-ish tests for the validator-events HTTP surface.

Stubs out the PgEventStore so no Postgres is required. Exercises:
  * POST /events (auth + insert)
  * GET /dashboard/api/validators/{hotkey}/events.json (public, redacted)
  * GET /dashboard/api/validators/{hotkey}/metrics.json
  * GET /dashboard/api/validators.json
"""

from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from database.server import set_event_store


@pytest.fixture
def server_router_app():
    """A fresh FastAPI app with the validator router mounted.

    Avoids triggering the shared ``database.server.app``'s middleware
    stack from being frozen by the TestClient (which would prevent
    other test modules — notably ``test_dashboard.py`` — from
    registering exception handlers later).
    """
    from database.server import include_validator_routes
    app = FastAPI()
    include_validator_routes(app)
    return app


@pytest.fixture
def public_api_app():
    """A fresh FastAPI app with the dashboard public API router attached.

    Mirrors what ``DatabaseNeuron._init_db`` does for dashboard / all modes,
    minus the real Postgres pool. The public router reads the event store
    via ``database.server.get_event_store`` so we don't need to inject it
    into the dashboard state.
    """
    from database.dashboard import app as dash_app_mod
    from database.dashboard.app import DashboardState, mount_public_api

    prev_state = dash_app_mod._state
    state = DashboardState(
        store=None, pool=None, r2=None,
        get_challenge=lambda: None, get_frontier=lambda: None,
    )
    dash_app_mod._state = state
    app = FastAPI()
    mount_public_api(
        app,
        store=state.store, pool=state.pool, r2=state.r2,
        get_challenge=state.get_challenge, get_frontier=state.get_frontier,
    )
    yield app
    dash_app_mod._state = prev_state


class FakeEventStore:
    def __init__(self):
        self.inserts: list[tuple[str, list[dict]]] = []
        self.events_by_hotkey: dict[str, list[dict]] = {}
        self._next_id = 1

    async def init_schema(self):
        pass

    async def insert_batch(self, hotkey: str, events: list[dict]) -> int:
        self.inserts.append((hotkey, events))
        rows = self.events_by_hotkey.setdefault(hotkey, [])
        for ev in events:
            row = {
                "id": self._next_id,
                "hotkey": hotkey,
                "ts": ev.get("ts", 0.0),
                "round_id": ev.get("round_id", -1),
                "kind": ev["kind"],
                "level": ev.get("level", ""),
                "payload": ev.get("payload", {}),
            }
            rows.append(row)
            self._next_id += 1
        return len(events)

    async def tail(self, hotkey, since_id=0, limit=200, kind=None):
        rows = self.events_by_hotkey.get(hotkey, [])
        out = [r for r in rows if r["id"] > since_id]
        if kind:
            out = [r for r in out if r["kind"] == kind]
        return out[-limit:]

    async def metrics(self, hotkey, metric, round_id=None, limit=1000):
        rows = self.events_by_hotkey.get(hotkey, [])
        out = []
        for r in rows:
            if r["kind"] != "metric":
                continue
            if r["payload"].get("name") != metric:
                continue
            if round_id is not None and r["round_id"] != round_id:
                continue
            out.append({
                "id": r["id"], "ts": r["ts"], "round_id": r["round_id"],
                "value": r["payload"].get("value"),
            })
        return out[:limit]

    async def hotkeys_with_recent_events(self, since_seconds=86400, limit=200):
        return [
            {"hotkey": hk, "last_ts": rows[-1]["ts"],
             "last_id": rows[-1]["id"], "n": len(rows)}
            for hk, rows in self.events_by_hotkey.items() if rows
        ]


@pytest.fixture
def fake_store():
    s = FakeEventStore()
    set_event_store(s)
    yield s
    set_event_store(None)


def test_dashboard_event_tail_redacts_and_paginates(fake_store, public_api_app):
    fake_store.events_by_hotkey["5HK"] = [
        {"id": 1, "hotkey": "5HK", "ts": 1.0, "round_id": 7, "kind": "log",
         "level": "info", "payload": {
            "message": "uploaded https://x.com/?X-Amz-Signature=secret"
        }},
        {"id": 2, "hotkey": "5HK", "ts": 2.0, "round_id": 7, "kind": "log",
         "level": "warning", "payload": {
            "message": 'File "/home/user/radar/x.py", line 1',
            "env": {"AWS_SECRET": "leak"},
         }},
    ]
    fake_store._next_id = 3

    client = TestClient(public_api_app)
    r = client.get("/dashboard/api/validators/5HK/events.json?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["hotkey"] == "5HK"
    assert body["next_cursor"] == 2
    events = body["events"]
    assert len(events) == 2
    # Presigned URL redacted
    assert "secret" not in events[0]["payload"]["message"]
    # Path collapsed
    assert "/home/user" not in events[1]["payload"]["message"]
    # Sensitive key dropped
    assert "env" not in events[1]["payload"]


def test_dashboard_event_tail_cursor(fake_store, public_api_app):
    fake_store.events_by_hotkey["hk"] = [
        {"id": i, "hotkey": "hk", "ts": float(i), "round_id": -1,
         "kind": "log", "level": "info", "payload": {"message": f"m{i}"}}
        for i in range(1, 6)
    ]
    fake_store._next_id = 6

    client = TestClient(public_api_app)
    r = client.get("/dashboard/api/validators/hk/events.json?since_id=2&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert [e["id"] for e in body["events"]] == [3, 4, 5]
    assert body["next_cursor"] == 5


def test_dashboard_metric_series(fake_store, public_api_app):
    fake_store.events_by_hotkey["hk"] = [
        {"id": 1, "hotkey": "hk", "ts": 1.0, "round_id": 1, "kind": "metric",
         "level": "", "payload": {"name": "loss", "value": 0.5}},
        {"id": 2, "hotkey": "hk", "ts": 2.0, "round_id": 1, "kind": "metric",
         "level": "", "payload": {"name": "loss", "value": 0.3}},
        {"id": 3, "hotkey": "hk", "ts": 3.0, "round_id": 1, "kind": "log",
         "level": "info", "payload": {"message": "ignored"}},
    ]

    client = TestClient(public_api_app)
    r = client.get("/dashboard/api/validators/hk/metrics.json?metric=loss")
    assert r.status_code == 200
    body = r.json()
    assert body["metric"] == "loss"
    assert [p["value"] for p in body["points"]] == [0.5, 0.3]


def test_dashboard_metrics_requires_metric_param(fake_store, public_api_app):
    client = TestClient(public_api_app)
    r = client.get("/dashboard/api/validators/hk/metrics.json")
    assert r.status_code == 422  # pydantic / fastapi missing required query


def test_dashboard_validators_index(fake_store, public_api_app):
    fake_store.events_by_hotkey["A"] = [
        {"id": 1, "hotkey": "A", "ts": 1.0, "round_id": -1, "kind": "log",
         "level": "info", "payload": {"message": "x"}},
    ]
    fake_store.events_by_hotkey["B"] = [
        {"id": 2, "hotkey": "B", "ts": 2.0, "round_id": -1, "kind": "log",
         "level": "info", "payload": {"message": "y"}},
    ]

    client = TestClient(public_api_app)
    r = client.get("/dashboard/api/validators.json")
    assert r.status_code == 200
    hks = {v["hotkey"] for v in r.json()["validators"]}
    assert hks == {"A", "B"}


def test_post_events_requires_hotkey(fake_store, server_router_app):
    client = TestClient(server_router_app)
    # The fresh app has no auth middleware, so caller_hotkey is unset and
    # we don't pass X-Validator-Hotkey — must be rejected.
    r = client.post("/events", json={"events": [
        {"kind": "log", "payload": {"message": "hi"}},
    ]})
    assert r.status_code == 403


def test_post_events_with_proxy_hotkey_header(fake_store, server_router_app):
    client = TestClient(server_router_app)
    r = client.post(
        "/events",
        json={"events": [
            {"kind": "log", "payload": {"message": "first"}},
            {"kind": "metric", "payload": {"name": "loss", "value": 0.5}},
            {"kind": "phase", "payload": {"name": "round_start"}},
        ]},
        headers={"X-Validator-Hotkey": "5HK"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["inserted"] == 3
    assert fake_store.inserts[0][0] == "5HK"


def test_post_events_unavailable_when_store_unset(server_router_app):
    set_event_store(None)
    client = TestClient(server_router_app)
    r = client.post(
        "/events",
        json={"events": [{"kind": "log", "payload": {"message": "x"}}]},
        headers={"X-Validator-Hotkey": "hk"},
    )
    assert r.status_code == 503
