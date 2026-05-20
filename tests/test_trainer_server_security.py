"""Tests for trainer server security hardening.

Covers: fail-closed auth, rate limiting, concurrency gate.

Tests the generalist runner/server.py (the deployed server).
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Must set RADAR_LOCALNET before importing the server module in some tests
os.environ.setdefault("RADAR_LOCALNET", "")


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset module-level state between tests."""
    import runner.server as srv
    srv._hotkey_last_request.clear()
    # Reset semaphore to unlocked
    srv._train_semaphore = asyncio.Semaphore(1)
    srv._RUNNERS.clear()
    yield


def _make_body(**overrides):
    data = {
        "architecture": "def build_model(c, p, n, q): pass\ndef build_optimizer(m): pass",
        "seed": 42,
        "round_id": 1,
        "miner_hotkey": "miner_abc",
        "time_budget": 10,
        "task_name": "ts_forecasting",
    }
    data.update(overrides)
    return json.dumps(data).encode()


class TestFailClosed:
    """Auth must reject requests without a valid signature."""

    def test_localnet_skips_auth(self):
        """RADAR_LOCALNET=true should skip auth (for local dev)."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        mock_runner = MagicMock(return_value={
            "status": "success", "round_id": 1, "miner_hotkey": "miner_abc",
            "checkpoint_path": "/tmp/ckpt",
        })
        srv._RUNNERS["ts_forecasting"] = mock_runner

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}), \
             patch("runner.server._upload_artifacts", return_value={"status": "success"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        assert resp.status_code == 202

    def test_rejects_missing_headers(self):
        """No Epistula headers → 403 (signature-only auth, fail closed)."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._RUNNERS["ts_forecasting"] = MagicMock()
        with patch.dict(os.environ, {"RADAR_LOCALNET": ""}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        assert resp.status_code == 403

    def test_rejects_invalid_auth(self):
        """verify_request returning ok=False → 403."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._RUNNERS["ts_forecasting"] = MagicMock()
        mock_verify = MagicMock(return_value=(False, "Invalid signature", ""))
        with patch.dict(os.environ, {"RADAR_LOCALNET": ""}):
            with patch.dict("sys.modules", {"shared.auth": MagicMock(verify_request=mock_verify)}):
                client = TestClient(srv.app)
                resp = client.post("/train", content=_make_body())

        assert resp.status_code == 403

    def test_allowlist_rejects_unknown_hotkey(self):
        """When RADAR_TRAINER_ALLOWED_HOTKEYS is set, unknown hotkey → 403."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._RUNNERS["ts_forecasting"] = MagicMock()
        # verify_request will receive allowed_hotkeys={"hk_good"} and the
        # default mock returns the value we wire here.
        captured = {}

        def fake_verify(headers, body, metagraph=None, require_stake=False, allowed_hotkeys=None):
            captured["allowed"] = set(allowed_hotkeys or [])
            return False, "Hotkey not in allowlist", ""

        with patch.dict(os.environ, {
            "RADAR_LOCALNET": "",
            "RADAR_TRAINER_ALLOWED_HOTKEYS": "hk_good",
        }):
            with patch.dict("sys.modules", {"shared.auth": MagicMock(verify_request=fake_verify)}):
                client = TestClient(srv.app)
                resp = client.post("/train", content=_make_body())

        assert resp.status_code == 403
        assert captured["allowed"] == {"hk_good"}


class TestRateLimiting:
    """Per-hotkey cooldown must throttle repeated requests."""

    def test_second_request_rate_limited(self):
        """Same hotkey within cooldown should get 429."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        mock_runner = MagicMock(return_value={
            "status": "success", "round_id": 1, "miner_hotkey": "miner_abc",
            "checkpoint_path": "/tmp/ckpt",
        })
        srv._RUNNERS["ts_forecasting"] = mock_runner

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}), \
             patch("runner.server._upload_artifacts", return_value={"status": "success"}):
            client = TestClient(srv.app)
            # First request — will get past rate limit
            resp1 = client.post("/train", content=_make_body())
            # Second request — should be rate limited
            resp2 = client.post("/train", content=_make_body())

        assert resp2.status_code == 429
        assert "Rate limited" in resp2.json()["error"]


class TestConcurrencyGate:
    """Only 1 concurrent training job allowed."""

    def test_locked_semaphore_returns_429(self):
        """If semaphore is locked, should immediately return 429."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._RUNNERS["ts_forecasting"] = MagicMock()
        # Pre-lock the semaphore
        loop = asyncio.new_event_loop()
        loop.run_until_complete(srv._train_semaphore.acquire())

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        assert resp.status_code == 429
        assert "already in progress" in resp.json()["error"]
        loop.close()


class TestAllowlistLoader:
    """RADAR_TRAINER_ALLOWED_HOTKEYS parsing."""

    def test_unset_returns_none(self):
        import runner.server as srv
        with patch.dict(os.environ, {"RADAR_TRAINER_ALLOWED_HOTKEYS": ""}):
            assert srv._load_allowed_hotkeys() is None

    def test_csv_parsed_to_set(self):
        import runner.server as srv
        with patch.dict(os.environ, {"RADAR_TRAINER_ALLOWED_HOTKEYS": "hk_a, hk_b ,hk_c"}):
            assert srv._load_allowed_hotkeys() == {"hk_a", "hk_b", "hk_c"}


class TestHealthEndpoint:
    """Health endpoint should remain accessible."""

    def test_health_ok(self):
        import runner.server as srv
        from fastapi.testclient import TestClient
        client = TestClient(srv.app)
        assert client.get("/health").status_code == 200
