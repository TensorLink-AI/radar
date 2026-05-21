"""Tests for shared/db_client.py — HTTP client for validators."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.db_client import DatabaseClient


class MockWallet:
    class hotkey:
        ss58_address = "5FakeHotkey"
        @staticmethod
        def sign(msg):
            return b"\x00" * 64


@pytest.fixture
def client():
    return DatabaseClient(db_url="http://fake-db:8090", wallet=MockWallet())


@pytest.mark.asyncio
async def test_health_success(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(client, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_get.return_value = mock_client
        assert await client.health() is True


@pytest.mark.asyncio
async def test_health_failure(client):
    with patch.object(client, "_get_client") as mock_get:
        mock_get.side_effect = Exception("connection refused")
        assert await client.health() is False


@pytest.mark.asyncio
async def test_add_experiment(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"index": 42}
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_get.return_value = mock_client
        idx = await client.add_experiment({"name": "test", "code": "x"})
        assert idx == 42


@pytest.mark.asyncio
async def test_get_frontier(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"code": "x", "metric": 0.5}]
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_get.return_value = mock_client
        frontier = await client.get_frontier()
        assert len(frontier) == 1


@pytest.mark.asyncio
async def test_record_components(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_get.return_value = mock_client
        result = await client.record_components(0, ["RMSNorm"])
        assert result is True


@pytest.mark.asyncio
async def test_post_failure_returns_none(client):
    with patch.object(client, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))
        mock_get.return_value = mock_client
        result = await client.add_experiment({"name": "test"})
        assert result is None


def test_signs_with_hotkey_header(monkeypatch):
    """When ``hotkey`` is set, _sign attaches X-Miner-Hotkey for the server."""
    monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
    c = DatabaseClient(db_url="http://db", wallet=None, hotkey="hk1")
    headers = c._sign(b'{"foo": 1}')
    assert headers.get("X-Miner-Hotkey") == "hk1"
    assert headers.get("X-Radar-Signature")  # non-empty signature


def test_warns_when_no_shared_secret(monkeypatch, caplog):
    """No RADAR_SHARED_SECRET → loud, actionable error at construction time."""
    monkeypatch.delenv("RADAR_SHARED_SECRET", raising=False)
    with caplog.at_level("ERROR"):
        DatabaseClient(db_url="http://db", wallet=None)
    assert any("RADAR_SHARED_SECRET" in rec.message for rec in caplog.records)
