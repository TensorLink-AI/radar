"""Tests for trainer server security hardening.

Covers: fail-closed auth, rate limiting, concurrency gate, metagraph caching.
"""

import asyncio
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

# Must set RADAR_LOCALNET before importing the server module in some tests
os.environ.setdefault("RADAR_LOCALNET", "")


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset module-level state between tests."""
    import runner.timeseries_forecast.server as srv
    srv._metagraph_cache = None
    srv._metagraph_last_refresh = 0.0
    srv._hotkey_last_request.clear()
    # Reset semaphore to unlocked
    srv._train_semaphore = asyncio.Semaphore(1)
    yield


def _make_body(**overrides):
    data = {
        "architecture": "def build_model(c, p, n, q): pass\ndef build_optimizer(m): pass",
        "seed": 42,
        "round_id": 1,
        "miner_hotkey": "miner_abc",
        "time_budget": 10,
    }
    data.update(overrides)
    return json.dumps(data).encode()


class TestFailClosed:
    """Auth must reject when metagraph is unavailable."""

    def test_rejects_when_metagraph_unavailable(self):
        """With no metagraph and not localnet, should return 503."""
        with patch.dict(os.environ, {"RADAR_LOCALNET": ""}):
            import runner.timeseries_forecast.server as srv
            from fastapi.testclient import TestClient

            srv._metagraph_cache = None

            with patch.object(srv, "_load_metagraph", return_value=None):
                client = TestClient(srv.app)
                resp = client.post("/train", content=_make_body())

        assert resp.status_code == 503
        assert "Auth unavailable" in resp.json()["error"]

    def test_localnet_skips_auth(self):
        """RADAR_LOCALNET=true should skip auth (for local dev)."""
        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}):
            import runner.timeseries_forecast.server as srv
            from fastapi.testclient import TestClient

            with patch.object(srv, "_execute_training", return_value={"status": "ok"}):
                client = TestClient(srv.app)
                resp = client.post("/train", content=_make_body())

        assert resp.status_code == 200

    def test_rejects_invalid_auth(self):
        """Bad Epistula headers should return 403."""
        import runner.timeseries_forecast.server as srv
        from fastapi.testclient import TestClient

        fake_mg = MagicMock()
        mock_verify = MagicMock(return_value=(False, "Invalid signature", ""))
        with patch.dict(os.environ, {"RADAR_LOCALNET": ""}):
            with patch.object(srv, "_load_metagraph", return_value=fake_mg):
                with patch.dict("sys.modules", {"shared.auth": MagicMock(verify_request=mock_verify)}):
                    client = TestClient(srv.app)
                    resp = client.post("/train", content=_make_body())

        assert resp.status_code == 403


class TestRateLimiting:
    """Per-hotkey cooldown must throttle repeated requests."""

    def test_second_request_rate_limited(self):
        """Same hotkey within cooldown should get 429."""
        import runner.timeseries_forecast.server as srv
        from fastapi.testclient import TestClient

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}):
            with patch.object(srv, "_execute_training", return_value={"status": "ok"}):
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
        import runner.timeseries_forecast.server as srv
        from fastapi.testclient import TestClient

        # Pre-lock the semaphore
        loop = asyncio.new_event_loop()
        loop.run_until_complete(srv._train_semaphore.acquire())

        with patch.dict(os.environ, {"RADAR_LOCALNET": "true"}):
            client = TestClient(srv.app)
            resp = client.post("/train", content=_make_body())

        assert resp.status_code == 429
        assert "already in progress" in resp.json()["error"]
        loop.close()


class TestMetagraphCache:
    """Metagraph caching behavior."""

    def test_returns_cached_within_interval(self):
        """Should not re-fetch if cache is fresh."""
        import runner.timeseries_forecast.server as srv

        fake_mg = MagicMock()
        fake_mg.n = 10
        srv._metagraph_cache = fake_mg
        srv._metagraph_last_refresh = time.time()

        result = srv._load_metagraph()
        assert result is fake_mg

    def test_refreshes_after_interval(self):
        """Should re-fetch if cache is stale."""
        import runner.timeseries_forecast.server as srv

        old_mg = MagicMock()
        old_mg.n = 10
        srv._metagraph_cache = old_mg
        srv._metagraph_last_refresh = time.time() - srv.METAGRAPH_REFRESH_INTERVAL - 1

        new_mg = MagicMock()
        new_mg.n = 20

        with patch("bittensor.Subtensor") as mock_sub:
            mock_sub.return_value.metagraph.return_value = new_mg
            result = srv._load_metagraph()

        assert result is new_mg
        assert srv._metagraph_cache is new_mg

    def test_returns_stale_on_refresh_failure(self):
        """If refresh fails but stale cache exists, should return stale."""
        import runner.timeseries_forecast.server as srv

        old_mg = MagicMock()
        old_mg.n = 10
        srv._metagraph_cache = old_mg
        srv._metagraph_last_refresh = time.time() - srv.METAGRAPH_REFRESH_INTERVAL - 1

        with patch("bittensor.Subtensor", side_effect=Exception("chain down")):
            result = srv._load_metagraph()

        assert result is old_mg

    def test_returns_none_on_first_failure(self):
        """If no cache and refresh fails, should return None."""
        import runner.timeseries_forecast.server as srv

        with patch("bittensor.Subtensor", side_effect=Exception("chain down")):
            result = srv._load_metagraph()

        assert result is None


class TestHealthEndpoint:
    """Health endpoint should remain accessible."""

    def test_health_ok(self):
        import runner.timeseries_forecast.server as srv
        from fastapi.testclient import TestClient
        client = TestClient(srv.app)
        assert client.get("/health").status_code == 200
