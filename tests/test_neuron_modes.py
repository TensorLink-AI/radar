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
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Optional
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# These tests poke Config.NEURON_MODE at runtime but also need a clean
# database/server import so routes pick up the per-mode auto-mount.


def _reload_server_app(mode: str) -> FastAPI:
    """Reload database.server with the given NEURON_MODE so auto-mount fires."""
    from config import Config
    Config.NEURON_MODE = mode
    # Drop any previously imported version so the @app auto-mount block runs
    # again against a fresh FastAPI app.
    for mod in ("database.server", "database.neuron"):
        sys.modules.pop(mod, None)
    # Also reset dashboard mount state — the public-API mount is idempotent
    # but we want it deterministic per test.
    import database.dashboard.app as dash_app_mod
    dash_app_mod._state = None
    importlib.import_module("database.server")
    return sys.modules["database.server"].app


def _install_fake_state_and_mount(app: FastAPI, with_jinja: bool):
    """Install a minimal DashboardState + mount the JSON API (and Jinja if asked).

    Mirrors what DatabaseNeuron._init_db() does in dashboard / all modes,
    minus the real Postgres pool and bittensor imports.
    """
    from database.dashboard import app as dash_app_mod
    from database.dashboard.app import DashboardState, mount_public_api, mount_dashboard

    # Reuse the same in-memory fakes used by test_dashboard.py
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

    # Mount the public JSON router (no cookie gate)
    mount_public_api(
        app,
        store=state.store, pool=state.pool, r2=state.r2,
        get_challenge=state.get_challenge, get_frontier=state.get_frontier,
    )

    if with_jinja:
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
    """/experiments/* and friends are not reachable in dashboard mode."""
    app = _reload_server_app("dashboard")
    client = TestClient(app)

    # Health still works everywhere
    assert client.get("/health").status_code == 200

    # Validator surface: unmounted
    assert client.get("/experiments/recent").status_code == 404
    assert client.get("/frontier").status_code == 404
    assert client.get("/challenge").status_code == 404
    assert client.get("/provenance/component_stats").status_code == 404
    assert client.get("/agent_code/some_hotkey").status_code == 404


def test_dashboard_mode_json_api_open():
    """JSON API is reachable WITHOUT any auth headers in dashboard mode."""
    app = _reload_server_app("dashboard")
    _install_fake_state_and_mount(app, with_jinja=False)

    client = TestClient(app)
    # No cookies, no Authorization header — should still get 200 JSON
    r = client.get("/dashboard/api/stats.json")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data

    # Other SPA endpoints also open
    assert client.get("/dashboard/api/tasks.json").status_code == 200
    assert client.get("/dashboard/api/miners.json").status_code == 200
    assert client.get("/dashboard/api/recent.json?n=5").status_code == 200


def test_dashboard_mode_jinja_not_mounted():
    """The cookie-gated Jinja index returns 404 in dashboard mode."""
    app = _reload_server_app("dashboard")
    _install_fake_state_and_mount(app, with_jinja=False)

    client = TestClient(app)
    r = client.get("/dashboard/", follow_redirects=False)
    assert r.status_code == 404
    # The login form should also be missing
    assert client.get("/dashboard/login").status_code == 404


def test_dashboard_mode_does_not_import_bittensor():
    """Dashboard mode must not touch bt.Wallet / bt.Subtensor / metagraph.

    Install a sentinel ``bittensor`` module into sys.modules so that if
    DatabaseNeuron.__init__ ever tries to instantiate Wallet/Subtensor the
    test fails loudly. The real bittensor package is never imported.
    """
    import types
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
    sys.modules["bittensor"] = sentinel
    try:
        sys.modules.pop("database.neuron", None)
        sys.modules.pop("database.server", None)
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
        sys.modules.pop("database.neuron", None)
        sys.modules.pop("database.server", None)


def test_dashboard_mode_skips_sync_loops():
    """Metagraph sync / round refresh are no-ops in dashboard mode."""
    from config import Config
    prev = Config.NEURON_MODE
    Config.NEURON_MODE = "dashboard"
    try:
        sys.modules.pop("database.neuron", None)
        sys.modules.pop("database.server", None)
        import database.neuron as neuron_mod
        import argparse
        cfg = argparse.Namespace(netuid=1, port=8091, pg_dsn="", task="")
        n = neuron_mod.DatabaseNeuron(cfg)

        # Both of these must short-circuit without touching subtensor
        # (which is None in dashboard mode)
        assert n._refresh_round_id() == -1
        # No-op, no AttributeError for self.metagraph.sync()
        n._sync_metagraph()
    finally:
        Config.NEURON_MODE = prev
        sys.modules.pop("database.neuron", None)
        sys.modules.pop("database.server", None)


# ── validator mode ────────────────────────────────────────────


def test_validator_mode_json_api_404_when_dashboard_disabled():
    """With RADAR_DASHBOARD_ENABLED=false, the JSON API is not mounted."""
    from config import Config
    prev_mode, prev_en = Config.NEURON_MODE, Config.DASHBOARD_ENABLED
    Config.NEURON_MODE = "validator"
    Config.DASHBOARD_ENABLED = False
    try:
        app = _reload_server_app("validator")
        # Deliberately DO NOT mount the public API — that's what validator
        # mode with dashboard disabled does in production.
        client = TestClient(app)

        # Validator routes ARE mounted
        assert client.get("/experiments/recent").status_code in (200, 503, 429)

        # JSON API is NOT mounted
        assert client.get("/dashboard/api/stats.json").status_code == 404
    finally:
        Config.NEURON_MODE = prev_mode
        Config.DASHBOARD_ENABLED = prev_en


def test_validator_mode_jinja_login_still_works_when_enabled():
    """Operators can still log in to the Jinja UI under validator mode."""
    from config import Config
    prev_mode = Config.NEURON_MODE
    prev_enabled = Config.DASHBOARD_ENABLED
    prev_key = Config.DASHBOARD_KEY
    Config.NEURON_MODE = "validator"
    Config.DASHBOARD_ENABLED = True
    Config.DASHBOARD_KEY = "testkey"
    try:
        app = _reload_server_app("validator")
        _install_fake_state_and_mount(app, with_jinja=True)

        client = TestClient(app)

        # /dashboard/login renders
        assert client.get("/dashboard/login").status_code == 200

        # Logging in with the right key sets the cookie and redirects
        r = client.post(
            "/dashboard/login",
            data={"key": "testkey", "next": "/dashboard/"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "radar_dashboard_session=" in r.headers.get("set-cookie", "")

        # Authenticated browse hits the Jinja index
        r2 = client.get("/dashboard/", follow_redirects=False)
        assert r2.status_code == 200
    finally:
        Config.NEURON_MODE = prev_mode
        Config.DASHBOARD_ENABLED = prev_enabled
        Config.DASHBOARD_KEY = prev_key


# ── all mode ──────────────────────────────────────────────────


def test_all_mode_serves_both_surfaces():
    """Default (all) mode serves validator and dashboard-json on one app."""
    app = _reload_server_app("all")
    _install_fake_state_and_mount(app, with_jinja=False)

    client = TestClient(app)
    # Validator routes mounted
    assert client.get("/health").status_code == 200
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
    try:
        app = _reload_server_app("dashboard")
        _install_fake_state_and_mount(app, with_jinja=False)

        # Apply CORS via the neuron helper without constructing the full
        # neuron (avoids needing an argparse cfg)
        sys.modules.pop("database.neuron", None)
        import database.neuron as neuron_mod
        import argparse
        cfg = argparse.Namespace(netuid=1, port=8091, pg_dsn="", task="")
        n = neuron_mod.DatabaseNeuron(cfg)
        # Point at our already-mounted app (the reload changed sys.modules)
        import database.server as srv
        n_app = srv.app
        # Install the fake state on the reimported app too
        _install_fake_state_and_mount(n_app, with_jinja=False)
        n._maybe_mount_cors()

        client = TestClient(n_app)
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
        sys.modules.pop("database.neuron", None)
        sys.modules.pop("database.server", None)


def test_cors_not_mounted_without_origins():
    """An empty RADAR_DASHBOARD_CORS_ORIGINS means no CORS middleware."""
    from config import Config
    prev_mode = Config.NEURON_MODE
    prev_origins = Config.DASHBOARD_CORS_ORIGINS
    Config.NEURON_MODE = "dashboard"
    Config.DASHBOARD_CORS_ORIGINS = ""
    try:
        sys.modules.pop("database.neuron", None)
        sys.modules.pop("database.server", None)
        import database.neuron as neuron_mod
        import argparse
        cfg = argparse.Namespace(netuid=1, port=8091, pg_dsn="", task="")
        n = neuron_mod.DatabaseNeuron(cfg)
        n._maybe_mount_cors()

        import database.server as srv
        from fastapi.middleware.cors import CORSMiddleware
        for m in srv.app.user_middleware:
            assert m.cls is not CORSMiddleware
    finally:
        Config.NEURON_MODE = prev_mode
        Config.DASHBOARD_CORS_ORIGINS = prev_origins
        sys.modules.pop("database.neuron", None)
        sys.modules.pop("database.server", None)
