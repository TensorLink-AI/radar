"""Tests for the generalist runner/server.py — one server, all tasks.

Covers: task routing, unknown task rejection, sandbox dispatch,
health endpoint, auth/rate-limit (inherited from shared logic).
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset module-level state between tests."""
    import runner.server as srv
    srv._metagraph_cache = None
    srv._metagraph_last_refresh = 0.0
    srv._hotkey_last_request.clear()
    srv._train_semaphore = asyncio.Semaphore(1)
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


def _success_result():
    return ({
        "status": "success",
        "round_id": 1,
        "miner_hotkey": "miner_abc",
        "flops_equivalent_size": 100000,
        "training_time_seconds": 5.0,
        "num_steps": 10,
        "num_params_M": 0.5,
        "peak_vram_mb": 100.0,
        "checkpoint_path": "/workspace/sandbox/checkpoints/model.safetensors",
    }, "")


class TestTaskRouting:
    """Generalist server routes through the sandbox by task_name."""

    def test_unknown_task_returns_400(self):
        import runner.server as srv
        from fastapi.testclient import TestClient

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body(task_name="nonexistent_task"))

        assert resp.status_code == 400
        assert "Unknown task" in resp.json()["error"]

    def test_accepts_known_task(self):
        """Valid task_name returns 202 Accepted (background task spawned)."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        mock_sandbox = AsyncMock(return_value=_success_result())
        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}), \
             patch("runner.server.run_sandbox", mock_sandbox), \
             patch("runner.server.upload_artifacts", return_value={"status": "success"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body(task_name="ts_forecasting"))

        assert resp.status_code == 202
        assert resp.json() == {"status": "accepted", "round_id": 1}

    def test_defaults_to_ts_forecasting(self):
        """Missing task_name defaults to ts_forecasting."""
        import runner.server as srv
        from fastapi.testclient import TestClient

        body = _make_body()
        data = json.loads(body)
        del data["task_name"]
        body = json.dumps(data).encode()

        mock_sandbox = AsyncMock(return_value=_success_result())
        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}), \
             patch("runner.server.run_sandbox", mock_sandbox), \
             patch("runner.server.upload_artifacts", return_value={"status": "success"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=body)

        assert resp.status_code == 202

    def test_train_and_upload_dispatches_sandbox(self):
        """The background coroutine forwards architecture + config into run_sandbox."""
        import runner.server as srv

        mock_sandbox = AsyncMock(return_value=_success_result())
        mock_upload = MagicMock(return_value={"status": "success"})
        config = {
            "seed": 42, "round_id": 1, "miner_hotkey": "miner_abc",
            "min_flops": 0, "max_flops": 0, "time_budget": 5,
            "task_name": "ts_forecasting",
        }
        with patch("runner.server.run_sandbox", mock_sandbox), \
             patch("runner.server.upload_artifacts", mock_upload), \
             patch("runner.server.prefetch_shards", AsyncMock(return_value=[])), \
             patch("runner.server.prefetch_gift_eval", AsyncMock(return_value=None)):
            asyncio.run(srv._train_and_upload(
                "def build_model(c, p, n, q): pass\ndef build_optimizer(m): pass",
                config, upload_urls={}, gift_eval_urls={},
                pretrain_shard_urls=[], pretrain_val_shard_urls=[],
            ))

        mock_sandbox.assert_awaited_once()
        sandbox_config = mock_sandbox.call_args[0][0]
        assert sandbox_config["task_name"] == "ts_forecasting"
        assert sandbox_config["architecture_code"].startswith("def build_model")
        assert sandbox_config["seed"] == 42
        mock_upload.assert_called_once()

    def test_train_and_upload_failed_sandbox_skips_upload(self):
        """If the sandbox reports build_failed, no artifact upload runs."""
        import runner.server as srv

        mock_sandbox = AsyncMock(return_value=({
            "status": "build_failed", "error": "Missing build_model()",
            "round_id": 1, "miner_hotkey": "miner_abc",
        }, ""))
        mock_upload = MagicMock()
        mock_failure = MagicMock()
        with patch("runner.server.run_sandbox", mock_sandbox), \
             patch("runner.server.upload_artifacts", mock_upload), \
             patch("runner.server.upload_failure_meta", mock_failure), \
             patch("runner.server.prefetch_shards", AsyncMock(return_value=[])), \
             patch("runner.server.prefetch_gift_eval", AsyncMock(return_value=None)):
            asyncio.run(srv._train_and_upload(
                "code", {
                    "seed": 0, "round_id": 1, "miner_hotkey": "miner_abc",
                    "min_flops": 0, "max_flops": 0, "time_budget": 5,
                    "task_name": "ts_forecasting",
                },
                upload_urls={"meta": "https://x"}, gift_eval_urls={},
            ))

        mock_upload.assert_not_called()
        mock_failure.assert_called_once()


class TestGeneralistHealth:
    """Health endpoint on generalist server."""

    def test_health_ok(self):
        import runner.server as srv
        from fastapi.testclient import TestClient
        client = TestClient(srv.app)
        assert client.get("/health").status_code == 200


class TestGeneralistAuth:
    """Auth on generalist server mirrors per-task server."""

    def test_rejects_when_metagraph_unavailable(self):
        import runner.server as srv
        from fastapi.testclient import TestClient

        srv._metagraph_cache = None

        with patch.dict(os.environ, {"RADAR_LOCALNET": ""}), \
             patch.object(srv, "_load_metagraph", return_value=None):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        assert resp.status_code == 503

    def test_concurrency_gate(self):
        import runner.server as srv
        from fastapi.testclient import TestClient

        loop = asyncio.new_event_loop()
        loop.run_until_complete(srv._train_semaphore.acquire())

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        assert resp.status_code == 429
        assert "already in progress" in resp.json()["error"]
        loop.close()
