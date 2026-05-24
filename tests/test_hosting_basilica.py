"""Tests for miner/hosting.py — Basilica deploy path.

Focuses on the failure-diagnostics path added in this change: on a
``wait_until_ready`` timeout we must (a) dump k8s events + container
log tail, (b) delete the dead deployment, and (c) re-raise.
"""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


# ── Stub `basilica` so we can run without the real SDK installed ────

def _install_fake_basilica():
    """Install a minimal `basilica` / `basilica._deployment` stand-in
    so ``from basilica import BasilicaClient`` and
    ``from basilica._deployment import Deployment`` inside
    ``deploy_basilica`` resolve to MagicMocks we control per-test."""
    pkg = types.ModuleType("basilica")
    pkg.BasilicaClient = MagicMock()
    sub = types.ModuleType("basilica._deployment")
    sub.Deployment = MagicMock()
    pkg._deployment = sub
    sys.modules["basilica"] = pkg
    sys.modules["basilica._deployment"] = sub
    return pkg, sub


@pytest.fixture
def fake_basilica():
    pkg, sub = _install_fake_basilica()
    yield pkg, sub
    sys.modules.pop("basilica", None)
    sys.modules.pop("basilica._deployment", None)


@dataclass
class FakeRequest:
    round_id: int = 42
    gpu_count: int = 1
    min_gpu_memory_gb: int = 16
    memory: str = "16Gi"
    time_budget: int = 300


# ── _dump_basilica_failure ─────────────────────────────────────────


def test_dump_failure_logs_events_and_log_tail(caplog):
    from miner.hosting import _dump_basilica_failure
    client = MagicMock()
    client.get_deployment_events.return_value = {
        "events": [
            {
                "event_type": "Warning",
                "reason": "FailedScheduling",
                "count": 12,
                "last_timestamp": "2026-05-22T21:40:00Z",
                "message": "0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
            },
        ],
    }
    client.get_deployment_logs.return_value = "boot line one\nboot line two\n"

    with caplog.at_level(logging.ERROR, logger="miner.hosting"):
        _dump_basilica_failure(client, "radar-trainer-abcd1234-42")

    msgs = [r.getMessage() for r in caplog.records]
    assert any("BASILICA_EVENT" in m and "FailedScheduling" in m for m in msgs)
    assert any("BASILICA_LOG" in m and "boot line one" in m for m in msgs)
    assert any("BASILICA_LOG" in m and "boot line two" in m for m in msgs)
    client.get_deployment_events.assert_called_once_with(
        "radar-trainer-abcd1234-42", limit=50,
    )
    client.get_deployment_logs.assert_called_once_with(
        "radar-trainer-abcd1234-42", tail=200,
    )


def test_dump_failure_tolerates_events_endpoint_error(caplog):
    """If events lookup raises, logs still get dumped — and vice-versa."""
    from miner.hosting import _dump_basilica_failure
    client = MagicMock()
    client.get_deployment_events.side_effect = RuntimeError("events 500")
    client.get_deployment_logs.return_value = "tail line\n"

    with caplog.at_level(logging.WARNING, logger="miner.hosting"):
        _dump_basilica_failure(client, "dep-1")

    msgs = [r.getMessage() for r in caplog.records]
    assert any("BASILICA_EVENTS_UNAVAILABLE" in m for m in msgs)
    assert any("BASILICA_LOG" in m and "tail line" in m for m in msgs)


# ── deploy_basilica success path ───────────────────────────────────


@pytest.mark.asyncio
async def test_deploy_basilica_success_returns_handle(fake_basilica):
    pkg, sub = fake_basilica

    client = MagicMock()
    pkg.BasilicaClient.return_value = client
    client.create_deployment.return_value = MagicMock(name="raw-response")

    bdep = MagicMock()
    bdep.name = "radar-trainer-abcd1234-42"
    bdep.url = "https://example.run/trainer"
    bdep.wait_until_ready = MagicMock()
    bdep.refresh = MagicMock()
    sub.Deployment._from_response.return_value = bdep

    from miner.hosting import deploy_basilica
    out = await deploy_basilica(
        request=FakeRequest(),
        image="ghcr.io/x/y:tag",
        hotkey="abcd1234deadbeef",
        netuid=0,
        subtensor_network="",
        ttl=600,
    )

    assert out.name == "radar-trainer-abcd1234-42"
    assert out.url == "https://example.run/trainer"
    # create called with public_metadata=True (replaces enroll_metadata round-trip)
    kwargs = client.create_deployment.call_args.kwargs
    assert kwargs["public_metadata"] is True
    assert kwargs["instance_name"] == "radar-trainer-abcd1234-42"
    assert kwargs["image"] == "ghcr.io/x/y:tag"
    assert kwargs["port"] == 8081
    # Wait + refresh both invoked exactly once on success.
    bdep.wait_until_ready.assert_called_once()
    bdep.refresh.assert_called_once()
    # No teardown on success.
    client.delete_deployment.assert_not_called()


# ── deploy_basilica failure path ───────────────────────────────────


@pytest.mark.asyncio
async def test_deploy_basilica_timeout_dumps_and_tears_down(fake_basilica, caplog):
    pkg, sub = fake_basilica

    client = MagicMock()
    pkg.BasilicaClient.return_value = client
    client.create_deployment.return_value = MagicMock()
    client.get_deployment_events.return_value = {
        "events": [{
            "event_type": "Warning",
            "reason": "ImagePullBackOff",
            "count": 5,
            "last_timestamp": "t",
            "message": "Back-off pulling image",
        }],
    }
    client.get_deployment_logs.return_value = "no logs yet\n"

    bdep = MagicMock()
    bdep.name = "radar-trainer-abcd1234-42"
    bdep.wait_until_ready.side_effect = RuntimeError("simulated DeploymentTimeout")
    sub.Deployment._from_response.return_value = bdep

    from miner.hosting import deploy_basilica
    with caplog.at_level(logging.INFO, logger="miner.hosting"):
        with pytest.raises(RuntimeError, match="simulated DeploymentTimeout"):
            await deploy_basilica(
                request=FakeRequest(),
                image="ghcr.io/x/y:tag",
                hotkey="abcd1234deadbeef",
                netuid=0,
                subtensor_network="",
                ttl=600,
            )

    msgs = [r.getMessage() for r in caplog.records]
    # Diagnostics surfaced
    assert any("BASILICA_EVENT" in m and "ImagePullBackOff" in m for m in msgs)
    assert any("BASILICA_LOG" in m for m in msgs)
    # Dead pod torn down
    client.delete_deployment.assert_called_once_with("radar-trainer-abcd1234-42")
    assert any("BASILICA_TEARDOWN" in m for m in msgs)
    # Refresh never reached (wait raised first)
    bdep.refresh.assert_not_called()


@pytest.mark.asyncio
async def test_deploy_basilica_failure_swallows_teardown_error(fake_basilica, caplog):
    pkg, sub = fake_basilica

    client = MagicMock()
    pkg.BasilicaClient.return_value = client
    client.create_deployment.return_value = MagicMock()
    client.get_deployment_events.return_value = {"events": []}
    client.get_deployment_logs.return_value = ""
    client.delete_deployment.side_effect = RuntimeError("delete 503")

    bdep = MagicMock()
    bdep.name = "radar-trainer-abcd1234-42"
    bdep.wait_until_ready.side_effect = RuntimeError("timeout")
    sub.Deployment._from_response.return_value = bdep

    from miner.hosting import deploy_basilica
    with caplog.at_level(logging.WARNING, logger="miner.hosting"):
        # Original deploy error must propagate, not the teardown error.
        with pytest.raises(RuntimeError, match="timeout"):
            await deploy_basilica(
                request=FakeRequest(),
                image="ghcr.io/x/y:tag",
                hotkey="abcd1234deadbeef",
                netuid=0,
                subtensor_network="",
                ttl=600,
            )

    msgs = [r.getMessage() for r in caplog.records]
    assert any("BASILICA_TEARDOWN_FAILED" in m and "delete 503" in m for m in msgs)
