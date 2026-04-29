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
async def test_post_events_success(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok", "inserted": 2}
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_get.return_value = mock_client
        ok = await client.post_events([
            {"kind": "log", "payload": {"message": "hi"}},
            {"kind": "metric", "payload": {"name": "x", "value": 1}},
        ])
        assert ok is True
        # Body should be {"events": [...]}, signed and posted to /events.
        call_kwargs = mock_client.post.call_args.kwargs
        sent = json.loads(call_kwargs["content"].decode())
        assert "events" in sent and len(sent["events"]) == 2
        assert mock_client.post.call_args.args[0].endswith("/events")


@pytest.mark.asyncio
async def test_post_events_empty_short_circuits(client):
    # Empty batch returns True without an HTTP call.
    with patch.object(client, "_get_client") as mock_get:
        ok = await client.post_events([])
        assert ok is True
        mock_get.assert_not_called()


@pytest.mark.asyncio
async def test_post_events_failure_returns_false(client):
    with patch.object(client, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("net down"))
        mock_get.return_value = mock_client
        ok = await client.post_events([{"kind": "log", "payload": {}}])
        assert ok is False


@pytest.mark.asyncio
async def test_post_failure_returns_none(client):
    with patch.object(client, "_get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))
        mock_get.return_value = mock_client
        result = await client.add_experiment({"name": "test"})
        assert result is None
