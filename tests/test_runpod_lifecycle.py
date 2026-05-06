"""Tests for miner/runpod_lifecycle.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from miner.runpod_lifecycle import (
    cancel_jobs_for_round, make_runpod_client, validate_credentials,
)


def test_make_runpod_client_uses_config_defaults(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_test")
    client = make_runpod_client()
    assert client.api_key == "rp_test"
    assert client.base_url
    assert client.timeout > 0


@pytest.mark.asyncio
async def test_validate_credentials_ok():
    client = MagicMock()
    client.validate_credentials = AsyncMock()
    await validate_credentials(client)
    client.validate_credentials.assert_awaited_once()


@pytest.mark.asyncio
async def test_validate_credentials_raises_with_helpful_msg():
    client = MagicMock()
    client.validate_credentials = AsyncMock(side_effect=RuntimeError("401 Unauthorized"))
    with pytest.raises(RuntimeError, match="RunPod credentials"):
        await validate_credentials(client)


@pytest.mark.asyncio
async def test_cancel_jobs_for_round_no_op_when_empty():
    client = MagicMock()
    client.cancel_job = AsyncMock()
    await cancel_jobs_for_round(client, "ep", [])
    client.cancel_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_jobs_for_round_filters_empty_strings():
    client = MagicMock()
    client.cancel_job = AsyncMock()
    await cancel_jobs_for_round(client, "ep", ["", "j1", ""])
    client.cancel_job.assert_awaited_once_with("ep", "j1")
