"""Tests for miner/health_monitor.py.

Drives the monitor with patched httpx + tight intervals so we can
verify the failure-grace transition without sleeping for real.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miner.health_monitor import HealthMonitor


def _resp(status):
    r = MagicMock()
    r.status_code = status
    return r


@pytest.mark.asyncio
async def test_healthy_probe_does_not_compromise():
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=_resp(200))

    monitor = HealthMonitor(
        round_id=1, trainer_url="https://t",
        workload_uid="wl",
        poll_interval_s=0.005, fail_grace_s=0.05,
    )
    with patch("miner.health_monitor.httpx.AsyncClient", return_value=client):
        monitor.start()
        await asyncio.sleep(0.1)
        await monitor.stop()
    assert monitor.locally_compromised is False


@pytest.mark.asyncio
async def test_unhealthy_long_enough_marks_compromised():
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=_resp(503))

    monitor = HealthMonitor(
        round_id=2, trainer_url="https://t",
        workload_uid="wl",
        poll_interval_s=0.005, fail_grace_s=0.05,
    )
    with patch("miner.health_monitor.httpx.AsyncClient", return_value=client):
        monitor.start()
        await asyncio.sleep(0.2)
        await monitor.stop()
    assert monitor.locally_compromised is True


@pytest.mark.asyncio
async def test_brief_failure_does_not_compromise():
    """Single missed probe doesn't trip — must persist past fail_grace_s."""
    responses = [_resp(503), _resp(200), _resp(200), _resp(200)]
    iter_resp = iter(responses)

    async def _get(url):
        try:
            return next(iter_resp)
        except StopIteration:
            return _resp(200)

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(side_effect=_get)

    monitor = HealthMonitor(
        round_id=3, trainer_url="https://t",
        workload_uid="wl",
        poll_interval_s=0.005, fail_grace_s=0.5,  # long grace
    )
    with patch("miner.health_monitor.httpx.AsyncClient", return_value=client):
        monitor.start()
        await asyncio.sleep(0.1)
        await monitor.stop()
    assert monitor.locally_compromised is False


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=_resp(200))

    monitor = HealthMonitor(
        round_id=4, trainer_url="https://t",
        workload_uid="wl",
        poll_interval_s=0.01, fail_grace_s=1.0,
    )
    with patch("miner.health_monitor.httpx.AsyncClient", return_value=client):
        monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()
        await monitor.stop()  # second stop must not raise


@pytest.mark.asyncio
async def test_exception_counts_as_failure():
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(side_effect=ConnectionError("boom"))

    monitor = HealthMonitor(
        round_id=5, trainer_url="https://t",
        workload_uid="wl",
        poll_interval_s=0.005, fail_grace_s=0.05,
    )
    with patch("miner.health_monitor.httpx.AsyncClient", return_value=client):
        monitor.start()
        await asyncio.sleep(0.2)
        await monitor.stop()
    assert monitor.locally_compromised is True
