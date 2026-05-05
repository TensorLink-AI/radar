"""Tests for miner/hosting_runpod.py — deploy, cancel, submit relay."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from miner.hosting import Deployment
from miner.hosting_runpod import (
    RunpodReadinessTimeout, deploy_runpod, cancel_active_jobs,
    submit_dispatch_to_runpod,
)
from shared.runpod_breaker import RunpodUnavailable
from shared.runpod_client import EndpointInfo, JobHandle


def _request(round_id=42):
    r = MagicMock()
    r.round_id = round_id
    return r


# ── deploy_runpod ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deploy_returns_listener_url_not_runpod_url():
    """Validators dispatch to the listener, not RunPod's API URL."""
    info = EndpointInfo(
        endpoint_id="ep_1",
        template_id="tpl_1",
        image_name="ghcr.io/r/r@sha256:abc",
        image_digest="sha256:abc",
        workers_running=0,
        workers_max=1,
    )
    client = MagicMock()
    client.get_endpoint = AsyncMock(return_value=info)

    deployment = await deploy_runpod(
        runpod_client=client,
        endpoint_id="ep_1",
        listener_url="http://miner-host:8091",
        deployed_image_digest="sha256:abc",
        gpu_class="H100",
        request=_request(),
        readiness_timeout_s=2.0,
        poll_interval_s=0.01,
    )

    assert deployment.url == "http://miner-host:8091"
    assert deployment.runpod_endpoint_id == "ep_1"
    assert deployment.runpod_template_id == "tpl_1"
    assert deployment.deployed_image_digest == "sha256:abc"
    assert deployment.gpu_class == "H100"
    # Targon-only fields stay empty.
    assert deployment.targon_workload_uid == ""
    assert deployment.cvm_ip == ""


@pytest.mark.asyncio
async def test_deploy_falls_back_to_endpoint_digest_when_explicit_empty():
    info = EndpointInfo(endpoint_id="ep", template_id="t", workers_max=1, image_digest="sha256:fromendpoint")
    client = MagicMock()
    client.get_endpoint = AsyncMock(return_value=info)

    deployment = await deploy_runpod(
        runpod_client=client,
        endpoint_id="ep",
        listener_url="http://x",
        deployed_image_digest="",
        gpu_class="",
        request=_request(),
        readiness_timeout_s=1.0,
        poll_interval_s=0.01,
    )
    assert deployment.deployed_image_digest == "sha256:fromendpoint"


@pytest.mark.asyncio
async def test_deploy_raises_on_workers_max_zero():
    info = EndpointInfo(endpoint_id="ep", workers_max=0, workers_running=0)
    client = MagicMock()
    client.get_endpoint = AsyncMock(return_value=info)

    with pytest.raises(RunpodReadinessTimeout):
        await deploy_runpod(
            runpod_client=client,
            endpoint_id="ep",
            listener_url="http://x",
            deployed_image_digest="sha256:abc",
            gpu_class="H100",
            request=_request(),
            readiness_timeout_s=0.1,
            poll_interval_s=0.05,
        )


@pytest.mark.asyncio
async def test_deploy_validates_required_inputs():
    client = MagicMock()
    with pytest.raises(ValueError, match="endpoint_id"):
        await deploy_runpod(
            runpod_client=client, endpoint_id="", listener_url="x",
            deployed_image_digest="", gpu_class="", request=_request(),
        )
    with pytest.raises(ValueError, match="listener_url"):
        await deploy_runpod(
            runpod_client=client, endpoint_id="ep", listener_url="",
            deployed_image_digest="", gpu_class="", request=_request(),
        )


# ── cancel_active_jobs ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_returns_true_when_no_jobs():
    client = MagicMock()
    client.cancel_job = AsyncMock()
    assert await cancel_active_jobs(client, "ep", []) is True
    client.cancel_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_iterates_all_job_ids():
    client = MagicMock()
    client.cancel_job = AsyncMock()
    ok = await cancel_active_jobs(client, "ep", ["j1", "j2", "j3"], attempts=1)
    assert ok is True
    assert client.cancel_job.await_count == 3


@pytest.mark.asyncio
async def test_cancel_returns_false_when_attempts_exhausted():
    client = MagicMock()
    client.cancel_job = AsyncMock(side_effect=RuntimeError("boom"))
    ok = await cancel_active_jobs(client, "ep", ["j1"], attempts=2)
    assert ok is False
    assert client.cancel_job.await_count == 2


# ── submit_dispatch_to_runpod ─────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_returns_job_id_on_success():
    client = MagicMock()
    client.submit_job = AsyncMock(return_value=JobHandle(job_id="j_x", endpoint_id="ep", status="IN_QUEUE"))
    job_id = await submit_dispatch_to_runpod(
        runpod_client=client, endpoint_id="ep", payload={"a": 1},
    )
    assert job_id == "j_x"
    client.submit_job.assert_awaited_with("ep", {"a": 1})


@pytest.mark.asyncio
async def test_submit_returns_none_on_runpod_unavailable():
    client = MagicMock()
    client.submit_job = AsyncMock(side_effect=RunpodUnavailable("breaker open"))
    job_id = await submit_dispatch_to_runpod(
        runpod_client=client, endpoint_id="ep", payload={"a": 1},
    )
    assert job_id is None


@pytest.mark.asyncio
async def test_submit_returns_none_on_empty_job_id():
    client = MagicMock()
    client.submit_job = AsyncMock(return_value=JobHandle(job_id="", endpoint_id="ep", status="FAILED"))
    job_id = await submit_dispatch_to_runpod(
        runpod_client=client, endpoint_id="ep", payload={},
    )
    assert job_id is None


# ── Deployment dataclass parity ───────────────────────────────────


def test_deployment_runpod_fields_default_empty():
    """Adding RunPod fields must not break the Targon / Basilica defaults."""
    d = Deployment(name="t", url="https://t", targon_workload_uid="wl_1")
    assert d.runpod_endpoint_id == ""
    assert d.runpod_template_id == ""
    assert d.targon_workload_uid == "wl_1"
