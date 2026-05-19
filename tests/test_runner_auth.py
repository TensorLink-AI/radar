"""Auth tests for runner/server.py — confirms metagraph is gone and HMAC
service-key verification is in place."""

from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

from shared.auth import hmac_sign_request


SECRET = b"a" * 32
KEY_ID = "operator"


def _client(monkeypatch):
    monkeypatch.setenv("RADAR_SERVICE_KEY", SECRET.decode())
    monkeypatch.setenv("RADAR_SERVICE_KEY_ID", KEY_ID)
    monkeypatch.delenv("RADAR_LOCALNET", raising=False)
    # Reload to pick up env.
    import importlib

    import runner.server as srv
    importlib.reload(srv)
    return TestClient(srv.app)


def test_train_rejects_without_hmac_headers(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/train", content=b"{}", headers={"Content-Type": "application/json"})
    assert r.status_code == 403
    assert "missing" in r.json()["error"].lower()


def test_train_rejects_bad_signature(monkeypatch):
    client = _client(monkeypatch)
    body = b'{"architecture": "x", "task_name": "ts_forecasting"}'
    headers = hmac_sign_request(b"WRONG_SECRET", body, key_id=KEY_ID)
    headers["Content-Type"] = "application/json"
    r = client.post("/train", content=body, headers=headers)
    assert r.status_code == 403


def test_train_rejects_unknown_key_id(monkeypatch):
    client = _client(monkeypatch)
    body = b'{"architecture": "x", "task_name": "ts_forecasting"}'
    headers = hmac_sign_request(SECRET, body, key_id="not-operator")
    headers["Content-Type"] = "application/json"
    r = client.post("/train", content=body, headers=headers)
    assert r.status_code == 403


def test_train_503_when_service_key_unset(monkeypatch):
    monkeypatch.delenv("RADAR_SERVICE_KEY", raising=False)
    monkeypatch.delenv("RADAR_LOCALNET", raising=False)
    import importlib

    import runner.server as srv
    importlib.reload(srv)
    client = TestClient(srv.app)
    body = b'{"architecture": "x", "task_name": "ts_forecasting"}'
    headers = hmac_sign_request(SECRET, body, key_id=KEY_ID)
    headers["Content-Type"] = "application/json"
    r = client.post("/train", content=body, headers=headers)
    assert r.status_code == 503


def test_health_no_auth(monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_localnet_skips_auth(monkeypatch):
    monkeypatch.setenv("RADAR_LOCALNET", "1")
    monkeypatch.delenv("RADAR_SERVICE_KEY", raising=False)
    import importlib

    import runner.server as srv
    importlib.reload(srv)
    client = TestClient(srv.app)
    body = json.dumps({
        "architecture": "class M: pass",
        "task_name": "ts_forecasting",
    }).encode()
    r = client.post(
        "/train", content=body,
        headers={"Content-Type": "application/json"},
    )
    # 202 (accepted) — auth was skipped.  We don't actually train
    # anything because run_sandbox would need a real sandbox; the test
    # just confirms auth doesn't reject.
    assert r.status_code == 202


def test_no_bittensor_or_metagraph_symbols():
    """Ensure the metagraph machinery is gone from the module."""
    import runner.server as srv
    assert not hasattr(srv, "_load_metagraph")
    assert not hasattr(srv, "_metagraph_cache")
    src = open(srv.__file__).read()
    assert "import bittensor" not in src
