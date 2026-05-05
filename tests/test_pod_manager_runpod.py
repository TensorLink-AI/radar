"""Tests for validator-side RunPod endpoint verification."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from validator.pod_manager import verify_runpod_endpoint
from shared.runpod_breaker import RunpodUnavailable
from shared.runpod_client import EndpointInfo


def _client_with_endpoint(info):
    c = MagicMock()
    c.get_endpoint = AsyncMock(return_value=info)
    return c


@pytest.mark.asyncio
async def test_verify_ok_when_digest_and_workers_match():
    info = EndpointInfo(
        endpoint_id="ep",
        template_id="tpl",
        image_name="ghcr.io/x/y@sha256:abc",
        image_digest="sha256:abc",
        workers_max=2,
    )
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="ep",
        expected_image_digest="sha256:abc",
        declared_image_digest="sha256:abc",
        runpod_client=_client_with_endpoint(info),
    )
    assert ok, reason


@pytest.mark.asyncio
async def test_verify_rejects_missing_endpoint_id():
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="",
        runpod_client=MagicMock(),
    )
    assert not ok
    assert "endpoint_id" in reason


@pytest.mark.asyncio
async def test_verify_rejects_endpoint_outside_allowlist(monkeypatch):
    monkeypatch.setattr(
        "config.Config.OFFICIAL_RUNPOD_ENDPOINTS", "ep_blessed,ep_other",
    )
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="ep_random",
        runpod_client=MagicMock(),
    )
    assert not ok
    assert "allowlist" in reason


@pytest.mark.asyncio
async def test_verify_rejects_declared_digest_mismatch():
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="ep",
        expected_image_digest="sha256:abc",
        declared_image_digest="sha256:other",
        runpod_client=MagicMock(),
    )
    assert not ok
    assert "declared" in reason


@pytest.mark.asyncio
async def test_verify_rejects_tag_form_template():
    info = EndpointInfo(
        endpoint_id="ep",
        template_id="tpl",
        image_name="ghcr.io/x/y:v1",
        image_digest="",
        workers_max=2,
    )
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="ep",
        expected_image_digest="sha256:abc",
        runpod_client=_client_with_endpoint(info),
    )
    assert not ok
    assert "tag-form" in reason


@pytest.mark.asyncio
async def test_verify_rejects_digest_mismatch_on_endpoint():
    info = EndpointInfo(
        endpoint_id="ep",
        template_id="tpl",
        image_name="ghcr.io/x/y@sha256:other",
        image_digest="sha256:other",
        workers_max=2,
    )
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="ep",
        expected_image_digest="sha256:abc",
        runpod_client=_client_with_endpoint(info),
    )
    assert not ok
    assert "digest" in reason


@pytest.mark.asyncio
async def test_verify_rejects_zero_capacity():
    info = EndpointInfo(
        endpoint_id="ep",
        template_id="tpl",
        image_name="ghcr.io/x/y@sha256:abc",
        image_digest="sha256:abc",
        workers_max=0,
    )
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="ep",
        expected_image_digest="sha256:abc",
        runpod_client=_client_with_endpoint(info),
    )
    assert not ok
    assert "workers_max" in reason


@pytest.mark.asyncio
async def test_verify_returns_unavailable_on_breaker():
    c = MagicMock()
    c.get_endpoint = AsyncMock(side_effect=RunpodUnavailable("breaker open"))
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="ep",
        runpod_client=c,
    )
    assert not ok
    assert "runpod unavailable" in reason


@pytest.mark.asyncio
async def test_verify_rejects_endpoint_not_visible():
    """Empty template_id + empty image_name = endpoint not in this account."""
    info = EndpointInfo(endpoint_id="ep")  # all defaults empty
    ok, reason = await verify_runpod_endpoint(
        endpoint_id="ep",
        runpod_client=_client_with_endpoint(info),
    )
    assert not ok
    assert "not visible" in reason
