"""Tests for the generalist runner/server.py — one server, all tasks.

Covers: task routing, unknown task rejection, runner registration,
health endpoint, auth/rate-limit (inherited from shared logic).
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset module-level state between tests."""
    import runner.server as srv
    srv._auth_cache = None
    srv._auth_last_refresh = 0.0
    srv._hotkey_last_request.clear()
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


class TestTaskRouting:
    """Generalist server routes to the correct runner by task_name."""

    def test_unknown_task_returns_400(self):
        """Requesting an unknown task returns 400."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body(task_name="nonexistent_task"))

        assert resp.status_code == 400
        assert "Unknown task" in resp.json()["error"]

    def test_routes_to_registered_runner(self):
        """Valid task_name dispatches to the registered runner."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        mock_runner = MagicMock(return_value={
            "status": "success",
            "round_id": 1,
            "miner_hotkey": "miner_abc",
            "flops_equivalent_size": 100000,
            "training_time_seconds": 5.0,
            "num_steps": 10,
            "num_params_M": 0.5,
            "peak_vram_mb": 100.0,
            "checkpoint_path": "/workspace/checkpoints/model.safetensors",
        })
        srv._RUNNERS["ts_forecasting"] = mock_runner
        srv._RUNNERS["ml_training"] = mock_runner

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}), \
             patch("runner.server._upload_artifacts", return_value={"status": "success"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body(task_name="ts_forecasting"))

        assert resp.status_code in (200, 202)
        # Background dispatch — give the event loop a chance to run.
        for _ in range(20):
            if mock_runner.called:
                break
            asyncio.run(asyncio.sleep(0.05))
        mock_runner.assert_called_once()
        call_args = mock_runner.call_args
        assert call_args[0][0] == "def build_model(c, p, n, q): pass\ndef build_optimizer(m): pass"
        assert call_args[0][1]["seed"] == 42

    def test_defaults_to_ts_forecasting(self):
        """Missing task_name defaults to ts_forecasting."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        mock_runner = MagicMock(return_value={
            "status": "success", "round_id": 1, "miner_hotkey": "miner_abc",
            "checkpoint_path": "/tmp/ckpt",
        })
        srv._RUNNERS["ts_forecasting"] = mock_runner

        body = _make_body()
        data = json.loads(body)
        del data["task_name"]
        body = json.dumps(data).encode()

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}), \
             patch("runner.server._upload_artifacts", return_value={"status": "success"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=body)

        assert resp.status_code in (200, 202)
        for _ in range(20):
            if mock_runner.called:
                break
            asyncio.run(asyncio.sleep(0.05))
        mock_runner.assert_called_once()

    def test_failed_training_returns_without_upload(self):
        """If runner returns build_failed, no R2 upload attempted."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        mock_runner = MagicMock(return_value={
            "status": "build_failed", "error": "Missing build_model()",
            "round_id": 1, "miner_hotkey": "miner_abc",
        })
        srv._RUNNERS["ts_forecasting"] = mock_runner

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}), \
             patch("runner.server._upload_artifacts") as mock_upload:
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        # /train is async-accept: 202 immediately, training runs in background.
        assert resp.status_code in (200, 202)
        # Let the background task finish so we can assert no upload occurred.
        for _ in range(40):
            if mock_runner.called:
                asyncio.run(asyncio.sleep(0.05))
                break
            asyncio.run(asyncio.sleep(0.05))
        mock_upload.assert_not_called()


class TestSharedSecretAuth:
    """HMAC shared-secret auth on /train."""

    def test_missing_signature_returns_401_when_secret_set(self, monkeypatch):
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._RUNNERS["ts_forecasting"] = MagicMock()
        monkeypatch.delenv("RADAR_LOCALNET", raising=False)
        monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
        srv._dev_mode_warned = False

        client = TestClient(srv.app)
        resp = client.post("/train", content=_make_body())
        assert resp.status_code == 401

    def test_valid_signature_accepted_when_secret_set(self, monkeypatch):
        import runner.server as srv
        from fastapi.testclient import TestClient
        from shared.auth import sign_request_hmac

        srv._RUNNERS["ts_forecasting"] = MagicMock(return_value={
            "status": "success", "round_id": 1, "miner_hotkey": "m",
            "checkpoint_path": "/tmp/ckpt",
        })
        monkeypatch.delenv("RADAR_LOCALNET", raising=False)
        monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
        srv._dev_mode_warned = False

        body = _make_body()
        sig = sign_request_hmac(body, "test-secret")

        with patch("runner.server._upload_artifacts", return_value={"status": "success"}):
            client = TestClient(srv.app)
            resp = client.post(
                "/train", content=body,
                headers={"X-Radar-Signature": sig},
            )
        assert resp.status_code in (200, 202)

    def test_unset_secret_runs_in_dev_mode(self, monkeypatch):
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._RUNNERS["ts_forecasting"] = MagicMock(return_value={
            "status": "success", "round_id": 1, "miner_hotkey": "m",
            "checkpoint_path": "/tmp/ckpt",
        })
        monkeypatch.delenv("RADAR_LOCALNET", raising=False)
        monkeypatch.delenv("RADAR_SHARED_SECRET", raising=False)
        srv._dev_mode_warned = False

        with patch("runner.server._upload_artifacts", return_value={"status": "success"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())
        # Unsigned but no secret configured → dev mode accept.
        assert resp.status_code in (200, 202)


class TestRunnerRegistry:
    """Runner registration and lazy loading."""

    def test_register_runners_populates_registry(self):
        """_register_runners() loads ts_forecasting and ml_training."""
        import runner.server as srv
        srv._register_runners()
        assert "ts_forecasting" in srv._RUNNERS
        assert "ml_training" in srv._RUNNERS

    def test_register_runners_idempotent(self):
        """Calling _register_runners() twice doesn't double-register."""
        import runner.server as srv
        srv._register_runners()
        first = dict(srv._RUNNERS)
        srv._register_runners()
        assert srv._RUNNERS == first


class TestGeneralistHealth:
    """Health endpoint on generalist server."""

    def test_health_ok(self):
        import runner.server as srv
        from fastapi.testclient import TestClient
        client = TestClient(srv.app)
        assert client.get("/health").status_code == 200


class TestGeneralistAuth:
    """Concurrency / fail-closed behavior on the generalist server."""

    def test_concurrency_gate(self):
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._RUNNERS["ts_forecasting"] = MagicMock()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(srv._train_semaphore.acquire())

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        assert resp.status_code == 429
        assert "already in progress" in resp.json()["error"]
        loop.close()
