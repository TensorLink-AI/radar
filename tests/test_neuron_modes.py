"""Tests for the RADAR_NEURON_MODE split.

Verifies:
  * dashboard mode: JSON API is mounted without cookie auth; validator
    routes return 404; Jinja routes return 404; no bittensor wallet /
    subtensor / metagraph are ever instantiated; no chain sync/round
    loops run.
  * validator mode: validator routes are mounted; JSON API is only
    mounted when RADAR_DASHBOARD_ENABLED=true; the old cookie-session
    flow still works for operators.
  * all mode: everything on one process, default behaviour preserved.
  * auth: hitting /dashboard/api/*.json with no cookie returns 200 in
    modes that mount it.

Every test constructs its own throwaway FastAPI app and wires routes
directly via ``include_validator_routes`` / ``mount_public_api`` /
``mount_dashboard``. We deliberately do NOT reload ``database.server`` or
mutate ``sys.modules["database.server"]`` — that leaks module state into
the rest of the test suite (notably ``tests/test_dashboard.py``, which
imports ``database.server.app`` at module scope).
"""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_dashboard_state():
    """Drop any DashboardState + app mutations this test installs so the
    shared ``database.server.app`` / ``database.dashboard.app`` globals
    don't leak into the next test module (notably ``test_dashboard.py``).
    """
    from database.dashboard import app as dash_app_mod
    prev_state = dash_app_mod._state
    yield
    dash_app_mod._state = prev_state


def _fresh_app_with_validator_routes() -> FastAPI:
    """A new FastAPI app with only the validator router mounted.

    Mirrors what ``DatabaseNeuron._init_db`` does in validator / all
    modes, minus the real Postgres pool.
    """
    from database.server import include_validator_routes
    app = FastAPI()
    include_validator_routes(app)
    # Health route exists on the shared server app; provide an equivalent
    # so tests can check it alongside the validator routes.
    @app.get("/health")
    def _health():
        return {"status": "ok"}
    return app


def _install_fake_state() -> "DashboardState":  # noqa: F821
    """Install a minimal DashboardState with in-memory fakes."""
    from database.dashboard import app as dash_app_mod
    from database.dashboard.app import DashboardState
    from tests.test_dashboard import FakePool, FakeR2, FakeStore, _sample_elements

    elements = _sample_elements()
    state = DashboardState(
        store=FakeStore(elements),
        pool=FakePool(elements),
        r2=FakeR2(),
        get_challenge=lambda: {"round_id": 42, "task": "ts", "size_bucket": "small"},
        get_frontier=lambda: [{"id": 1}],
    )
    dash_app_mod._state = state
    return state


def _mount_public_api_on(app: FastAPI) -> None:
    from database.dashboard.app import mount_public_api
    state = _install_fake_state()
    mount_public_api(
        app,
        store=state.store, pool=state.pool, r2=state.r2,
        get_challenge=state.get_challenge, get_frontier=state.get_frontier,
    )


def _mount_jinja_on(app: FastAPI) -> None:
    from database.dashboard.app import mount_dashboard
    state = _install_fake_state()
    mount_dashboard(
        app,
        store=state.store, pool=state.pool, r2=state.r2,
        get_challenge=state.get_challenge, get_frontier=state.get_frontier,
    )


# ── Config validation ─────────────────────────────────────────


def test_invalid_neuron_mode_rejected():
    from config import Config, validate_neuron_mode
    prev = Config.NEURON_MODE
    Config.NEURON_MODE = "garbage"
    try:
        with pytest.raises(ValueError):
            validate_neuron_mode()
    finally:
        Config.NEURON_MODE = prev


def test_valid_modes_all_accepted():
    from config import Config, validate_neuron_mode
    prev = Config.NEURON_MODE
    try:
        for mode in ("validator", "dashboard", "all"):
            Config.NEURON_MODE = mode
            validate_neuron_mode()  # no raise
    finally:
        Config.NEURON_MODE = prev


# ── dashboard mode ────────────────────────────────────────────


def test_dashboard_mode_no_validator_routes():
    """Dashboard app only has /health + /dashboard/api/* — no /experiments etc."""
    app = FastAPI()
    # Dashboard mode: only /health + public JSON API. No include_validator_routes.
    @app.get("/health")
    def _health():
        return {"status": "ok"}
    _mount_public_api_on(app)

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/experiments/recent").status_code == 404
    assert client.get("/frontier").status_code == 404
    assert client.get("/challenge").status_code == 404
    assert client.get("/provenance/component_stats").status_code == 404
    assert client.get("/agent_code/some_hotkey").status_code == 404


def test_dashboard_mode_json_api_open():
    """JSON API is reachable WITHOUT any auth headers in dashboard mode."""
    app = FastAPI()
    _mount_public_api_on(app)

    client = TestClient(app)
    r = client.get("/dashboard/api/stats.json")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data

    assert client.get("/dashboard/api/tasks.json").status_code == 200
    assert client.get("/dashboard/api/miners.json").status_code == 200
    assert client.get("/dashboard/api/recent.json?n=5").status_code == 200


def test_dashboard_mode_jinja_not_mounted():
    """The cookie-gated Jinja index returns 404 when only the public API is mounted."""
    app = FastAPI()
    _mount_public_api_on(app)

    client = TestClient(app)
    assert client.get("/dashboard/", follow_redirects=False).status_code == 404
    assert client.get("/dashboard/login").status_code == 404


def test_dashboard_mode_does_not_import_bittensor():
    """DatabaseNeuron.__init__ in dashboard mode must not touch bittensor.

    Install a sentinel ``bittensor`` module; any call into it raises.
    """
    from config import Config
    prev = Config.NEURON_MODE
    Config.NEURON_MODE = "dashboard"

    sentinel = types.ModuleType("bittensor")

    def _boom(*a, **kw):
        raise AssertionError("bittensor must not be invoked in dashboard mode")

    sentinel.Wallet = _boom
    sentinel.Subtensor = _boom
    sentinel.Config = _boom

    saved_bt = sys.modules.get("bittensor")
    saved_neuron = sys.modules.get("database.neuron")
    sys.modules["bittensor"] = sentinel
    # Force a fresh import of database.neuron against the sentinel bittensor
    sys.modules.pop("database.neuron", None)
    try:
        import database.neuron as neuron_mod
        import argparse
        cfg = argparse.Namespace(netuid=1, port=8091, pg_dsn="", task="")
        n = neuron_mod.DatabaseNeuron(cfg)
        assert n.wallet is None
        assert n.subtensor is None
        assert n.metagraph is None
    finally:
        Config.NEURON_MODE = prev
        if saved_bt is not None:
            sys.modules["bittensor"] = saved_bt
        else:
            sys.modules.pop("bittensor", None)
        if saved_neuron is not None:
            sys.modules["database.neuron"] = saved_neuron
        else:
            sys.modules.pop("database.neuron", None)


def test_dashboard_mode_skips_sync_loops():
    """Metagraph sync / round refresh are no-ops in dashboard mode."""
    from config import Config
    prev = Config.NEURON_MODE
    Config.NEURON_MODE = "dashboard"
    saved_neuron = sys.modules.get("database.neuron")
    sys.modules.pop("database.neuron", None)
    try:
        import database.neuron as neuron_mod
        import argparse
        cfg = argparse.Namespace(netuid=1, port=8091, pg_dsn="", task="")
        n = neuron_mod.DatabaseNeuron(cfg)
        # Both of these must short-circuit without touching subtensor
        assert n._refresh_round_id() == -1
        n._sync_metagraph()  # no-op, no AttributeError
    finally:
        Config.NEURON_MODE = prev
        if saved_neuron is not None:
            sys.modules["database.neuron"] = saved_neuron
        else:
            sys.modules.pop("database.neuron", None)


# ── validator mode ────────────────────────────────────────────


def test_validator_mode_has_validator_routes_no_json_api():
    """Validator routes mounted; JSON API 404 when dashboard is not mounted."""
    app = _fresh_app_with_validator_routes()
    client = TestClient(app)

    # Validator routes present (return 503 since no DB wired; that's fine)
    assert client.get("/experiments/recent").status_code in (200, 503, 429)
    # JSON API not mounted
    assert client.get("/dashboard/api/stats.json").status_code == 404


def test_validator_mode_jinja_login_still_works_when_enabled():
    """Operators can still log in to the Jinja UI under validator mode."""
    from config import Config
    prev_enabled = Config.DASHBOARD_ENABLED
    prev_key = Config.DASHBOARD_KEY
    Config.DASHBOARD_ENABLED = True
    Config.DASHBOARD_KEY = "testkey"

    # Reset dashboard state so mount_dashboard does its work on a fresh app
    from database.dashboard import app as dash_app_mod
    prev_state = dash_app_mod._state
    dash_app_mod._state = None

    try:
        app = _fresh_app_with_validator_routes()
        _mount_jinja_on(app)

        client = TestClient(app)
        assert client.get("/dashboard/login").status_code == 200

        r = client.post(
            "/dashboard/login",
            data={"key": "testkey", "next": "/dashboard/"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "radar_dashboard_session=" in r.headers.get("set-cookie", "")

        r2 = client.get("/dashboard/", follow_redirects=False)
        assert r2.status_code == 200
    finally:
        Config.DASHBOARD_ENABLED = prev_enabled
        Config.DASHBOARD_KEY = prev_key
        dash_app_mod._state = prev_state


# ── all mode ──────────────────────────────────────────────────


def test_all_mode_serves_both_surfaces():
    """All mode: validator routes AND public JSON API on the same app."""
    app = _fresh_app_with_validator_routes()
    _mount_public_api_on(app)

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    # Validator routes present
    assert client.get("/experiments/recent").status_code in (200, 503)
    # JSON API public
    assert client.get("/dashboard/api/stats.json").status_code == 200


# ── CORS ──────────────────────────────────────────────────────


def test_cors_mounted_when_origins_configured():
    """DASHBOARD_CORS_ORIGINS triggers CORS on the public API."""
    from config import Config
    prev_mode = Config.NEURON_MODE
    prev_origins = Config.DASHBOARD_CORS_ORIGINS
    Config.NEURON_MODE = "dashboard"
    Config.DASHBOARD_CORS_ORIGINS = "https://radarnet.io"
    saved_neuron = sys.modules.get("database.neuron")
    sys.modules.pop("database.neuron", None)
    try:
        import database.neuron as neuron_mod
        import argparse
        cfg = argparse.Namespace(netuid=1, port=8091, pg_dsn="", task="")
        n = neuron_mod.DatabaseNeuron(cfg)

        # Build a fresh throwaway app, mount the public JSON, and let the
        # neuron stamp CORS onto it. We monkeypatch the module-level ``app``
        # that _maybe_mount_cors reads.
        fresh = FastAPI()
        _mount_public_api_on(fresh)
        original_app = neuron_mod.app
        neuron_mod.app = fresh
        try:
            n._maybe_mount_cors()
        finally:
            neuron_mod.app = original_app

        client = TestClient(fresh)
        r = client.options(
            "/dashboard/api/stats.json",
            headers={
                "Origin": "https://radarnet.io",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.status_code in (200, 204)
        assert r.headers.get("access-control-allow-origin") == "https://radarnet.io"
    finally:
        Config.NEURON_MODE = prev_mode
        Config.DASHBOARD_CORS_ORIGINS = prev_origins
        if saved_neuron is not None:
            sys.modules["database.neuron"] = saved_neuron
        else:
            sys.modules.pop("database.neuron", None)


def test_cors_not_mounted_without_origins():
    """An empty RADAR_DASHBOARD_CORS_ORIGINS means no CORS middleware."""
    from config import Config
    prev_mode = Config.NEURON_MODE
    prev_origins = Config.DASHBOARD_CORS_ORIGINS
    Config.NEURON_MODE = "dashboard"
    Config.DASHBOARD_CORS_ORIGINS = ""
    saved_neuron = sys.modules.get("database.neuron")
    sys.modules.pop("database.neuron", None)
    try:
        import database.neuron as neuron_mod
        import argparse
        cfg = argparse.Namespace(netuid=1, port=8091, pg_dsn="", task="")
        n = neuron_mod.DatabaseNeuron(cfg)

        fresh = FastAPI()
        original_app = neuron_mod.app
        neuron_mod.app = fresh
        try:
            n._maybe_mount_cors()
        finally:
            neuron_mod.app = original_app

        from fastapi.middleware.cors import CORSMiddleware
        for m in getattr(fresh, "user_middleware", []):
            assert m.cls is not CORSMiddleware
    finally:
        Config.NEURON_MODE = prev_mode
        Config.DASHBOARD_CORS_ORIGINS = prev_origins
        if saved_neuron is not None:
            sys.modules["database.neuron"] = saved_neuron
        else:
            sys.modules.pop("database.neuron", None)
