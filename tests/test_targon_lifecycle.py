"""Tests for miner/targon_lifecycle.py extracted helpers.

The helpers themselves are small; we drive them in isolation here so a
future refactor of Miner doesn't lose coverage of the Targon-specific
orchestration paths.
"""

from __future__ import annotations

import asyncio
import signal as signal_mod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miner.hosting import Deployment
from miner.targon_lifecycle import (
    HealthMonitorRegistry,
    install_shutdown_handlers,
    make_targon_client,
    validate_and_reap_orphans,
)


# ── make_targon_client ──────────────────────────────────────────────


def test_make_targon_client_uses_config_defaults(monkeypatch):
    monkeypatch.setenv("TARGON_API_KEY", "k")
    client = make_targon_client()
    assert client.base_url
    assert client.tower_url
    assert client.timeout > 0


# ── validate_and_reap_orphans ───────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_and_reap_no_orphans():
    client = MagicMock()
    client.validate_credentials = AsyncMock()
    client.list_active_workloads = AsyncMock(return_value=[])
    client.teardown_workload = AsyncMock()
    await validate_and_reap_orphans(client)
    client.validate_credentials.assert_awaited_once()
    client.list_active_workloads.assert_awaited_once()
    client.teardown_workload.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_and_reap_with_orphans():
    client = MagicMock()
    client.validate_credentials = AsyncMock()
    orphan = MagicMock(uid="wl_orphan")
    client.list_active_workloads = AsyncMock(return_value=[orphan])
    client.teardown_workload = AsyncMock()
    await validate_and_reap_orphans(client)
    client.teardown_workload.assert_awaited_with("wl_orphan")


@pytest.mark.asyncio
async def test_validate_and_reap_raises_on_bad_creds():
    client = MagicMock()
    client.validate_credentials = AsyncMock(side_effect=RuntimeError("401 Unauthorized"))
    with pytest.raises(RuntimeError, match="Targon credentials"):
        await validate_and_reap_orphans(client)


@pytest.mark.asyncio
async def test_validate_and_reap_continues_when_list_fails():
    """list_active_workloads failing must NOT crash startup — log and move on."""
    client = MagicMock()
    client.validate_credentials = AsyncMock()
    client.list_active_workloads = AsyncMock(side_effect=RuntimeError("transient"))
    client.teardown_workload = AsyncMock()
    await validate_and_reap_orphans(client)  # does not raise
    client.teardown_workload.assert_not_awaited()


# ── HealthMonitorRegistry ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_registry_start_stop_one():
    reg = HealthMonitorRegistry()
    dep = Deployment(name="t", url="https://t", targon_workload_uid="wl_1")
    with patch("miner.targon_lifecycle.HealthMonitor") as MonitorCls:
        instance = MagicMock()
        instance.stop = AsyncMock()
        MonitorCls.return_value = instance

        reg.start(round_id=1, deployment=dep)
        instance.start.assert_called_once()
        await reg.stop(round_id=1)
        instance.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_registry_start_replaces_existing():
    reg = HealthMonitorRegistry()
    dep = Deployment(name="t", url="https://t", targon_workload_uid="wl_1")
    with patch("miner.targon_lifecycle.HealthMonitor") as MonitorCls:
        first = MagicMock()
        first.stop = AsyncMock()
        second = MagicMock()
        second.stop = AsyncMock()
        MonitorCls.side_effect = [first, second]

        reg.start(1, dep)
        reg.start(1, dep)  # replaces — schedules first.stop()
        await asyncio.sleep(0.01)  # let the scheduled task run
        first.stop.assert_awaited_once()
        await reg.stop(1)
        second.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_registry_stop_all():
    reg = HealthMonitorRegistry()
    dep = Deployment(name="t", url="https://t", targon_workload_uid="wl_1")
    with patch("miner.targon_lifecycle.HealthMonitor") as MonitorCls:
        m1 = MagicMock(); m1.stop = AsyncMock()
        m2 = MagicMock(); m2.stop = AsyncMock()
        MonitorCls.side_effect = [m1, m2]
        reg.start(1, dep)
        reg.start(2, dep)
        await reg.stop_all()
    m1.stop.assert_awaited_once()
    m2.stop.assert_awaited_once()


# ── install_shutdown_handlers ───────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown_handler_sets_event_and_runs_teardown():
    loop = asyncio.get_event_loop()
    event = asyncio.Event()
    teardown_called = MagicMock()

    async def _teardown():
        teardown_called()

    install_shutdown_handlers(loop, event, _teardown, teardown_timeout_s=2.0)

    # The signal-handler API is platform-specific; test the inner
    # coroutine directly by extracting it.
    # Easier: send a signal via os.kill and check the event flips.
    import os
    os.kill(os.getpid(), signal_mod.SIGTERM)
    # Wait briefly for the loop to run the handler.
    try:
        await asyncio.wait_for(event.wait(), timeout=2.0)
    finally:
        # Reset the SIGTERM handler so subsequent tests aren't affected.
        loop.remove_signal_handler(signal_mod.SIGTERM)
        loop.remove_signal_handler(signal_mod.SIGINT)
    assert event.is_set()
    teardown_called.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_handler_timeout_still_sets_event():
    loop = asyncio.get_event_loop()
    event = asyncio.Event()

    async def _slow_teardown():
        await asyncio.sleep(60)

    install_shutdown_handlers(loop, event, _slow_teardown, teardown_timeout_s=0.05)

    import os
    os.kill(os.getpid(), signal_mod.SIGTERM)
    try:
        await asyncio.wait_for(event.wait(), timeout=2.0)
    finally:
        loop.remove_signal_handler(signal_mod.SIGTERM)
        loop.remove_signal_handler(signal_mod.SIGINT)
    assert event.is_set()
