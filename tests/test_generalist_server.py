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

        assert resp.status_code == 202
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

        assert resp.status_code == 202
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
             patch("runner.server._upload_artifacts") as mock_upload, \
             patch("runner.server._upload_failure_meta"):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        # Server returns 202 immediately; training runs in background.
        assert resp.status_code == 202
        assert resp.json()["status"] == "accepted"
        mock_upload.assert_not_called()


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
    """Auth on generalist server mirrors per-task server."""

    def test_rejects_unsigned_requests(self):
        """No Epistula headers and not localnet → 403 (signature-only auth)."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._RUNNERS["ts_forecasting"] = MagicMock()

        with patch.dict(os.environ, {"RADAR_LOCALNET": ""}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        assert resp.status_code == 403

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
