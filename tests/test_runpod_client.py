"""Tests for shared/runpod_client.py — parallel structure to test_targon_client.

Mocks the underlying ``httpx.AsyncClient`` so no live network is
required. Covers the four RunPod control-plane operations the rest
of the codebase relies on: submit / status / cancel / get_endpoint.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import httpx
import pytest

from shared.runpod_breaker import RunpodCircuitBreaker, RunpodUnavailable
from shared.runpod_client import (
    JobHandle, EndpointInfo, RunpodClient, RunpodError, _extract_digest,
)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_test")


# ── _extract_digest ────────────────────────────────────────────────


def test_extract_digest_pulls_from_at_form():
    assert (
        _extract_digest("ghcr.io/foo/bar@sha256:abc")
        == "sha256:abc"
    )


def test_extract_digest_empty_when_tag_form():
    assert _extract_digest("ghcr.io/foo/bar:v1") == ""
    assert _extract_digest("") == ""


# ── construction ───────────────────────────────────────────────────


def test_constructor_requires_api_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RunpodError, match="RUNPOD_API_KEY"):
        RunpodClient()


def test_constructor_uses_env_key(api_key):
    c = RunpodClient()
    assert c.api_key == "rp_test"


# ── HTTP plumbing ──────────────────────────────────────────────────


class _FakeAsyncClient:
    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kwargs):
        return self._handler("POST", url, kwargs)

    async def get(self, url, **kwargs):
        return self._handler("GET", url, kwargs)


def _ok(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _err(status, payload=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "err",
        request=MagicMock(),
        response=MagicMock(status_code=status),
    )
    return resp


# ── submit_job ─────────────────────────────────────────────────────


class TestSubmitJob:
    @pytest.mark.asyncio
    async def test_returns_job_handle(self, api_key):
        c = RunpodClient()

        def handler(method, url, kwargs):
            assert method == "POST"
            assert url.endswith("/v2/ep_test/run")
            body = kwargs["json"]
            assert body["input"] == {"foo": "bar"}
            return _ok({"id": "job_abc", "status": "IN_QUEUE"})

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            handle = await c.submit_job("ep_test", {"foo": "bar"})

        assert handle.job_id == "job_abc"
        assert handle.endpoint_id == "ep_test"
        assert handle.status == "IN_QUEUE"
        assert not handle.is_terminal

    @pytest.mark.asyncio
    async def test_requires_endpoint_id(self, api_key):
        c = RunpodClient()
        with pytest.raises(RunpodError):
            await c.submit_job("", {})

    @pytest.mark.asyncio
    async def test_breaker_opens_on_repeated_5xx(self, api_key):
        c = RunpodClient(
            breaker=RunpodCircuitBreaker(threshold=1, reset_after=10),
            max_retries=2,
        )

        def handler(method, url, kwargs):
            return _err(500, {"err": "boom"})

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            with pytest.raises(RunpodUnavailable):
                await c.submit_job("ep", {"x": 1})


# ── get_status ─────────────────────────────────────────────────────


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_returns_status(self, api_key):
        c = RunpodClient()

        def handler(method, url, kwargs):
            assert method == "GET"
            assert url.endswith("/v2/ep/status/job_1")
            return _ok({"status": "IN_PROGRESS"})

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            handle = await c.get_status("ep", "job_1")
        assert handle.status == "IN_PROGRESS"
        assert handle.is_terminal is False

    @pytest.mark.asyncio
    async def test_terminal_states(self, api_key):
        c = RunpodClient()
        for s in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            def handler(method, url, kwargs, _s=s):
                return _ok({"status": _s})

            with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
                handle = await c.get_status("ep", "j")
            assert handle.is_terminal, f"{s} should be terminal"


# ── cancel_job ─────────────────────────────────────────────────────


class TestCancelJob:
    @pytest.mark.asyncio
    async def test_swallows_4xx_already_terminal(self, api_key):
        c = RunpodClient()

        # 4xx on cancel is "job already done" — should NOT raise.
        def handler(method, url, kwargs):
            return _err(404)

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            await c.cancel_job("ep", "j")  # no exception

    @pytest.mark.asyncio
    async def test_does_nothing_when_inputs_missing(self, api_key):
        c = RunpodClient()
        # Should silently no-op; no http calls.
        await c.cancel_job("", "j")
        await c.cancel_job("ep", "")

    @pytest.mark.asyncio
    async def test_swallows_unavailable(self, api_key):
        c = RunpodClient(breaker=RunpodCircuitBreaker(threshold=1))
        await c.breaker.on_failure()
        # Breaker open — call should NOT raise (cancel is best-effort).

        def handler(method, url, kwargs):
            return _ok({})

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            await c.cancel_job("ep", "j")


# ── get_endpoint ──────────────────────────────────────────────────


class TestGetEndpoint:
    @pytest.mark.asyncio
    async def test_parses_template_and_digest(self, api_key):
        c = RunpodClient()

        def handler(method, url, kwargs):
            assert method == "POST"
            assert url.endswith("/graphql")
            return _ok({"data": {"myself": {"endpoints": [
                {
                    "id": "ep_xyz",
                    "templateId": "tpl_abc",
                    "workersRunning": 1,
                    "workersMax": 3,
                    "template": {"imageName": "ghcr.io/r/r@sha256:deadbeef"},
                },
                {"id": "other"},
            ]}}})

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            info = await c.get_endpoint("ep_xyz")

        assert info.endpoint_id == "ep_xyz"
        assert info.template_id == "tpl_abc"
        assert info.image_name == "ghcr.io/r/r@sha256:deadbeef"
        assert info.image_digest == "sha256:deadbeef"
        assert info.workers_running == 1
        assert info.workers_max == 3

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_empty_info(self, api_key):
        c = RunpodClient()

        def handler(method, url, kwargs):
            return _ok({"data": {"myself": {"endpoints": []}}})

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            info = await c.get_endpoint("ep_missing")
        assert info.endpoint_id == "ep_missing"
        assert info.template_id == ""
        assert info.image_name == ""

    @pytest.mark.asyncio
    async def test_tag_form_image_yields_empty_digest(self, api_key):
        """RunPod templates that aren't pinned by digest must be detected."""
        c = RunpodClient()

        def handler(method, url, kwargs):
            return _ok({"data": {"myself": {"endpoints": [
                {
                    "id": "ep",
                    "templateId": "tpl",
                    "workersRunning": 0,
                    "workersMax": 1,
                    "template": {"imageName": "ghcr.io/r/r:v1"},
                },
            ]}}})

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            info = await c.get_endpoint("ep")
        assert info.image_digest == ""


# ── validate_credentials ──────────────────────────────────────────


class TestValidateCredentials:
    @pytest.mark.asyncio
    async def test_passes_on_2xx(self, api_key):
        c = RunpodClient()

        def handler(method, url, kwargs):
            return _ok({"data": {"myself": {"id": "u_1"}}})

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            await c.validate_credentials()

    @pytest.mark.asyncio
    async def test_raises_on_401(self, api_key):
        c = RunpodClient()

        def handler(method, url, kwargs):
            return _err(401)

        with patch("httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            with pytest.raises(RunpodError, match="invalid"):
                await c.validate_credentials()
