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
    # cvm_ip empty in handle is the production reality (Targon routing
    # subdomain isn't a usable CVM endpoint).
    handle = MagicMock(uid="wl_xyz", url="https://wl_xyz.targon.network",
                      cvm_ip="", name="trainer-1", status="running")
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
    assert deployment.cvm_ip == ""


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


# ── wait_for_trainer_ready ──────────────────────────────────────────


from miner.hosting import (
    TargonReadinessTimeout, teardown_targon_with_retry, wait_for_trainer_ready,
)


@pytest.mark.asyncio
async def test_readiness_returns_when_both_endpoints_up(monkeypatch):
    """Both /health and /api/v1/evidence reachable → returns cleanly."""
    health_resp = MagicMock(status_code=200)
    evidence_resp = MagicMock(status_code=405)  # method not allowed but server is up
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=health_resp)
    client.post = AsyncMock(return_value=evidence_resp)

    with patch("miner.hosting.httpx.AsyncClient", return_value=client):
        await wait_for_trainer_ready(
            trainer_url="https://t",
            cvm_ip="1.2.3.4",
            timeout_s=5.0,
            poll_interval_s=0.01,
        )

    client.get.assert_awaited()
    client.post.assert_awaited()


@pytest.mark.asyncio
async def test_readiness_times_out_when_health_never_ok(monkeypatch):
    health_resp = MagicMock(status_code=503)
    evidence_resp = MagicMock(status_code=405)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=health_resp)
    client.post = AsyncMock(return_value=evidence_resp)

    with patch("miner.hosting.httpx.AsyncClient", return_value=client):
        with pytest.raises(TargonReadinessTimeout) as excinfo:
            await wait_for_trainer_ready(
                trainer_url="https://t",
                cvm_ip="1.2.3.4",
                timeout_s=0.1,
                poll_interval_s=0.01,
            )
    assert "healthy=False" in str(excinfo.value)


@pytest.mark.asyncio
async def test_readiness_no_cvm_ip_skips_evidence_check():
    health_resp = MagicMock(status_code=200)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=health_resp)
    client.post = AsyncMock()  # should never be called

    with patch("miner.hosting.httpx.AsyncClient", return_value=client):
        await wait_for_trainer_ready(
            trainer_url="https://t", cvm_ip="",
            timeout_s=2.0, poll_interval_s=0.01,
        )
    client.post.assert_not_awaited()


# ── teardown_targon_with_retry ──────────────────────────────────────


@pytest.mark.asyncio
async def test_teardown_succeeds_first_attempt():
    targon = MagicMock()
    targon.teardown_workload = AsyncMock()
    ok = await teardown_targon_with_retry(targon, "wl_x", attempts=3)
    assert ok is True
    targon.teardown_workload.assert_awaited_once_with("wl_x")


@pytest.mark.asyncio
async def test_teardown_retries_then_succeeds():
    targon = MagicMock()
    calls = [RuntimeError("fail"), None]
    async def _td(uid):
        x = calls.pop(0)
        if isinstance(x, Exception):
            raise x
    targon.teardown_workload = AsyncMock(side_effect=[RuntimeError("fail"), None])

    with patch("miner.hosting.asyncio.sleep", AsyncMock()):
        ok = await teardown_targon_with_retry(targon, "wl_x", attempts=3)
    assert ok is True
    assert targon.teardown_workload.await_count == 2


@pytest.mark.asyncio
async def test_teardown_gives_up_after_attempts():
    targon = MagicMock()
    targon.teardown_workload = AsyncMock(side_effect=RuntimeError("nope"))

    with patch("miner.hosting.asyncio.sleep", AsyncMock()):
        ok = await teardown_targon_with_retry(targon, "wl_x", attempts=3)
    assert ok is False
    assert targon.teardown_workload.await_count == 3


# ── Targon startup orphan cleanup ───────────────────────────────────


@pytest.mark.asyncio
async def test_startup_check_validates_then_reaps_orphans(monkeypatch):
    monkeypatch.setenv("TARGON_API_KEY", "k")
    from miner import neuron as mn
    monkeypatch.setattr(mn.Config, "HOSTING_BACKEND", "targon")
    monkeypatch.setattr(mn.Config, "OFFICIAL_TRAINING_IMAGE_DIGEST", "sha256:abc")

    stub_cfg = MagicMock(netuid=1)
    with patch.object(mn, "bt"):
        m = mn.Miner(stub_cfg)

    fake_client = MagicMock()
    fake_client.validate_credentials = AsyncMock()
    orphan = MagicMock(uid="orphan_1")
    fake_client.list_active_workloads = AsyncMock(return_value=[orphan])
    fake_client.teardown_workload = AsyncMock()
    m._targon_client = fake_client

    await m._targon_startup_check()
    fake_client.validate_credentials.assert_awaited_once()
    fake_client.list_active_workloads.assert_awaited_once()
    fake_client.teardown_workload.assert_awaited_with("orphan_1")


@pytest.mark.asyncio
async def test_startup_check_raises_on_invalid_credentials(monkeypatch):
    monkeypatch.setenv("TARGON_API_KEY", "k")
    from miner import neuron as mn
    monkeypatch.setattr(mn.Config, "HOSTING_BACKEND", "targon")
    monkeypatch.setattr(mn.Config, "OFFICIAL_TRAINING_IMAGE_DIGEST", "sha256:abc")

    stub_cfg = MagicMock(netuid=1)
    with patch.object(mn, "bt"):
        m = mn.Miner(stub_cfg)

    fake_client = MagicMock()
    fake_client.validate_credentials = AsyncMock(side_effect=RuntimeError("401"))
    m._targon_client = fake_client

    with pytest.raises(RuntimeError, match="Targon credentials"):
        await m._targon_startup_check()


# ── Shutdown teardown ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown_tears_down_all_active(monkeypatch):
    monkeypatch.setenv("TARGON_API_KEY", "k")
    from miner import neuron as mn
    monkeypatch.setattr(mn.Config, "HOSTING_BACKEND", "targon")
    monkeypatch.setattr(mn.Config, "OFFICIAL_TRAINING_IMAGE_DIGEST", "sha256:abc")

    stub_cfg = MagicMock(netuid=1)
    with patch.object(mn, "bt"):
        m = mn.Miner(stub_cfg)

    dep1 = Deployment(name="t1", url="https://t1", targon_workload_uid="wl_1")
    dep2 = Deployment(name="t2", url="https://t2", targon_workload_uid="wl_2")
    m.active_deployments[1] = (dep1, 0.0, 1800)
    m.active_deployments[2] = (dep2, 0.0, 1800)
    m.active_deployments[3] = "pending"

    fake = MagicMock()
    fake.teardown_workload = AsyncMock()
    m._targon_client = fake

    with patch("miner.hosting.asyncio.sleep", AsyncMock()):
        await m._teardown_all_active()

    assert m.active_deployments == {}
    assert fake.teardown_workload.await_count == 2
    called = {c.args[0] for c in fake.teardown_workload.await_args_list}
    assert called == {"wl_1", "wl_2"}


# ── heartbeat listener self-check ───────────────────────────────────


def _build_miner(monkeypatch):
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    from miner import neuron as mn
    monkeypatch.setattr(mn.Config, "HOSTING_BACKEND", "basilica")
    stub_cfg = MagicMock(netuid=1, listener_port=8066)
    with patch.object(mn, "bt") as bt_mock:
        bt_mock.Subtensor.return_value.metagraph.return_value = MagicMock(n=1)
        m = mn.Miner(stub_cfg)
    return mn, m


@pytest.mark.asyncio
async def test_probe_listener_returns_true_on_200(monkeypatch):
    mn, m = _build_miner(monkeypatch)
    resp = MagicMock(status_code=200)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=resp)
    with patch.object(mn.httpx, "AsyncClient", return_value=client):
        assert await m._probe_listener() is True
    client.get.assert_awaited_once()
    assert "127.0.0.1:8066/health" in client.get.await_args.args[0]


@pytest.mark.asyncio
async def test_probe_listener_returns_false_on_non_200(monkeypatch):
    mn, m = _build_miner(monkeypatch)
    resp = MagicMock(status_code=503)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=resp)
    with patch.object(mn.httpx, "AsyncClient", return_value=client):
        assert await m._probe_listener() is False


@pytest.mark.asyncio
async def test_probe_listener_returns_false_on_connection_error(monkeypatch):
    mn, m = _build_miner(monkeypatch)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(side_effect=ConnectionError("refused"))
    with patch.object(mn.httpx, "AsyncClient", return_value=client):
        assert await m._probe_listener() is False
