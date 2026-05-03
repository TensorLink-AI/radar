"""Miner-side Targon deploy path tests.

The full Miner class touches bittensor, so we exercise the targon
branch via ``miner/hosting.py``'s ``deploy_targon`` directly plus a
narrow set of Miner-init checks.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miner.hosting import Deployment, deploy_targon, get_targon_registry_creds
from shared.protocol import TrainerRequest


@pytest.fixture
def request_obj():
    return TrainerRequest(
        round_id=42,
        gpu_count=1,
        min_gpu_memory_gb=80,
        memory="80Gi",
        time_budget=300,
    )


# ── deploy_targon ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deploy_targon_passes_empty_command_args(request_obj):
    """The whole point of the hardening: command/args must be empty."""
    targon = MagicMock()
    handle = MagicMock(uid="wl_xyz", url="https://wl_xyz.targon.network",
                      cvm_ip="wl_xyz.targon.network", name="trainer-1", status="running")
    targon.deploy_workload = AsyncMock(return_value=handle)

    deployment = await deploy_targon(
        targon_client=targon,
        request=request_obj,
        image="ghcr.io/x:y",
        deployed_image_digest="sha256:abc",
        hotkey="hk1234567890abcdef",
        netuid=42,
        subtensor_network="finney",
        gpu_class="H200",
    )

    targon.deploy_workload.assert_awaited_once()
    kwargs = targon.deploy_workload.await_args.kwargs
    assert kwargs["image"] == "ghcr.io/x:y"
    assert kwargs["gpu_class"] == "H200"
    assert kwargs["env"]["NETUID"] == "42"
    assert kwargs["env"]["SUBTENSOR_NETWORK"] == "finney"
    # name encodes hotkey prefix + round.
    assert "hk123456" in kwargs["name"]
    assert "42" in kwargs["name"]

    assert deployment.targon_workload_uid == "wl_xyz"
    assert deployment.deployed_image_digest == "sha256:abc"
    assert deployment.gpu_class == "H200"
    assert deployment.cvm_ip == "wl_xyz.targon.network"


@pytest.mark.asyncio
async def test_deploy_targon_propagates_failure(request_obj):
    targon = MagicMock()
    targon.deploy_workload = AsyncMock(side_effect=RuntimeError("targon API 500"))

    with pytest.raises(RuntimeError, match="targon API 500"):
        await deploy_targon(
            targon_client=targon,
            request=request_obj,
            image="img",
            deployed_image_digest="sha256:def",
            hotkey="hk",
            netuid=1,
            subtensor_network="finney",
            gpu_class="H100",
        )


# ── get_targon_registry_creds ───────────────────────────────────────


def test_registry_creds_none_when_unset(monkeypatch):
    monkeypatch.delenv("RADAR_TARGON_REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("RADAR_TARGON_REGISTRY_PASSWORD", raising=False)
    assert get_targon_registry_creds() is None


def test_registry_creds_built_when_both_set(monkeypatch):
    monkeypatch.setenv("RADAR_TARGON_REGISTRY_USERNAME", "u")
    monkeypatch.setenv("RADAR_TARGON_REGISTRY_PASSWORD", "p")
    monkeypatch.setenv("RADAR_TARGON_REGISTRY_SERVER", "ghcr.io")
    creds = get_targon_registry_creds()
    assert creds is not None
    assert creds.username == "u"
    assert creds.password == "p"
    assert creds.server == "ghcr.io"


# ── Miner init refuses without TARGON_API_KEY ───────────────────────


def test_miner_refuses_without_api_key_when_targon_backend(monkeypatch):
    """RADAR_HOSTING_BACKEND=targon + missing TARGON_API_KEY → fail fast at __init__."""
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    from miner import neuron as mn
    monkeypatch.setattr(mn.Config, "HOSTING_BACKEND", "targon")

    stub_cfg = MagicMock(netuid=1)
    with patch.object(mn, "bt") as bt_mock:
        with pytest.raises(RuntimeError, match="TARGON_API_KEY"):
            mn.Miner(stub_cfg)


def test_miner_does_not_require_api_key_for_basilica(monkeypatch):
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    from miner import neuron as mn
    monkeypatch.setattr(mn.Config, "HOSTING_BACKEND", "basilica")

    stub_cfg = MagicMock(netuid=1)
    with patch.object(mn, "bt") as bt_mock:
        bt_mock.Subtensor.return_value.metagraph.return_value = MagicMock(n=1)
        m = mn.Miner(stub_cfg)
    assert m.netuid == 1
