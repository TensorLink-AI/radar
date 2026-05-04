"""Tests for miner/runpod_lifecycle.py — deploy, teardown, orphan reap."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from miner.hosting import Deployment, TargonReadinessTimeout
from miner.runpod_lifecycle import (
    deploy_runpod,
    parse_gpu_types,
    teardown_runpod_with_retry,
    validate_and_reap_orphans_runpod,
    wait_for_runpod_ready,
)
from shared.runpod_client import PodHandle


def _request(round_id=7, gpu_count=1):
    r = MagicMock()
    r.round_id = round_id
    r.gpu_count = gpu_count
    return r


# ── parse_gpu_types ────────────────────────────────────────────────


def test_parse_gpu_types_strips_and_filters():
    out = parse_gpu_types(" NVIDIA A100 80GB PCIe , , NVIDIA RTX 4090 ,")
    assert out == ["NVIDIA A100 80GB PCIe", "NVIDIA RTX 4090"]


def test_parse_gpu_types_empty_string():
    assert parse_gpu_types("") == []
    assert parse_gpu_types("   ") == []


# ── deploy_runpod ──────────────────────────────────────────────────


class TestDeployRunpod:
    @pytest.mark.asyncio
    async def test_pins_digest_when_image_lacks_one(self):
        client = MagicMock()
        client.deploy_pod = AsyncMock(return_value=PodHandle(
            pod_id="p1", name="radar-trainer-x-7", status="PROVISIONING",
            gpu_type_id="NVIDIA A100 80GB PCIe",
        ))
        deployment = await deploy_runpod(
            runpod_client=client,
            request=_request(),
            image="ghcr.io/x/y:latest",  # no digest
            deployed_image_digest="sha256:abc",
            hotkey="hk1234567890abcdef",
            netuid=1,
            subtensor_network="finney",
            gpu_type_ids=["NVIDIA A100 80GB PCIe"],
            cloud_type="SECURE",
            container_disk_gb=50,
        )
        # Deploy was called with the digest-pinned ref.
        kwargs = client.deploy_pod.call_args.kwargs
        assert kwargs["image"] == "ghcr.io/x/y:latest@sha256:abc"
        # Deployment carries backend identifier + pod id for teardown.
        assert deployment.backend == "runpod"
        assert deployment.runpod_pod_id == "p1"
        assert deployment.deployed_image_digest == "sha256:abc"

    @pytest.mark.asyncio
    async def test_keeps_image_unchanged_if_already_pinned(self):
        client = MagicMock()
        client.deploy_pod = AsyncMock(return_value=PodHandle(pod_id="p1"))
        await deploy_runpod(
            runpod_client=client,
            request=_request(),
            image="ghcr.io/x/y@sha256:abc",  # already pinned
            deployed_image_digest="sha256:abc",
            hotkey="hk1234567890abcdef",
            netuid=1,
            subtensor_network="finney",
            gpu_type_ids=["NVIDIA H100 80GB HBM3"],
            cloud_type="COMMUNITY",
            container_disk_gb=80,
        )
        assert client.deploy_pod.call_args.kwargs["image"] == "ghcr.io/x/y@sha256:abc"
        assert client.deploy_pod.call_args.kwargs["cloud_type"] == "COMMUNITY"

    @pytest.mark.asyncio
    async def test_propagates_subtensor_env(self):
        client = MagicMock()
        client.deploy_pod = AsyncMock(return_value=PodHandle(pod_id="p1"))
        await deploy_runpod(
            runpod_client=client,
            request=_request(round_id=99),
            image="x@sha256:y",
            deployed_image_digest="sha256:y",
            hotkey="hkABCDEFGHIJKL",
            netuid=42,
            subtensor_network="test",
            gpu_type_ids=["NVIDIA A100 80GB PCIe"],
            cloud_type="SECURE",
            container_disk_gb=50,
        )
        env = client.deploy_pod.call_args.kwargs["env"]
        assert env["SUBTENSOR_NETWORK"] == "test"
        assert env["NETUID"] == "42"


# ── teardown_runpod_with_retry ─────────────────────────────────────


class TestTeardownRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        client = MagicMock()
        client.teardown_pod = AsyncMock(return_value=None)
        ok = await teardown_runpod_with_retry(client, "p1", attempts=3)
        assert ok
        assert client.teardown_pod.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch):
        # Skip real sleeps so the test runs fast.
        async def _no_sleep(_):
            return None
        monkeypatch.setattr("miner.runpod_lifecycle.asyncio.sleep", _no_sleep)

        attempts = []
        async def _flaky_teardown(uid):
            attempts.append(uid)
            if len(attempts) < 3:
                raise RuntimeError("transient")
        client = MagicMock()
        client.teardown_pod = _flaky_teardown
        ok = await teardown_runpod_with_retry(client, "p2", attempts=3)
        assert ok
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_all_attempts(self, monkeypatch):
        async def _no_sleep(_):
            return None
        monkeypatch.setattr("miner.runpod_lifecycle.asyncio.sleep", _no_sleep)

        client = MagicMock()
        client.teardown_pod = AsyncMock(side_effect=RuntimeError("boom"))
        ok = await teardown_runpod_with_retry(client, "p3", attempts=2)
        assert not ok
        assert client.teardown_pod.await_count == 2


# ── validate_and_reap_orphans_runpod ───────────────────────────────


class TestOrphanReaper:
    @pytest.mark.asyncio
    async def test_credentials_failure_raises_runtime_error(self):
        client = MagicMock()
        client.validate_credentials = AsyncMock(side_effect=RuntimeError("401"))
        with pytest.raises(RuntimeError, match="invalid or API unreachable"):
            await validate_and_reap_orphans_runpod(client)

    @pytest.mark.asyncio
    async def test_no_orphans_short_circuits(self):
        client = MagicMock()
        client.validate_credentials = AsyncMock()
        client.list_active_pods = AsyncMock(return_value=[])
        client.teardown_pod = AsyncMock()
        await validate_and_reap_orphans_runpod(client)
        client.teardown_pod.assert_not_called()

    @pytest.mark.asyncio
    async def test_tears_down_orphans(self, monkeypatch):
        async def _no_sleep(_):
            return None
        monkeypatch.setattr("miner.runpod_lifecycle.asyncio.sleep", _no_sleep)

        client = MagicMock()
        client.validate_credentials = AsyncMock()
        client.list_active_pods = AsyncMock(return_value=[
            PodHandle(pod_id="p1"),
            PodHandle(pod_id="p2"),
        ])
        client.teardown_pod = AsyncMock(return_value=None)
        await validate_and_reap_orphans_runpod(client)
        assert client.teardown_pod.await_count == 2

    @pytest.mark.asyncio
    async def test_list_failure_does_not_crash_startup(self):
        client = MagicMock()
        client.validate_credentials = AsyncMock()
        client.list_active_pods = AsyncMock(side_effect=RuntimeError("network"))
        # Must not raise — the next round will reap.
        await validate_and_reap_orphans_runpod(client)


# ── wait_for_runpod_ready ──────────────────────────────────────────


class TestWaitForReady:
    @pytest.mark.asyncio
    async def test_timeout_raises_targon_readiness_timeout(self, monkeypatch):
        async def _no_sleep(_):
            return None
        monkeypatch.setattr("miner.hosting.asyncio.sleep", _no_sleep)
        # Pod never returns a usable URL.
        client = MagicMock()
        client.get_pod = AsyncMock(return_value=PodHandle(
            pod_id="p1", status="PROVISIONING",
        ))
        with pytest.raises(TargonReadinessTimeout):
            await wait_for_runpod_ready(
                runpod_client=client,
                pod_id="p1",
                trainer_url="",
                timeout_s=0.05,
                poll_interval_s=0.01,
            )
