"""Tests for shared/targon_client.py + targon_breaker + targon_attest.

We mock the SDK and httpx via monkeypatching — no respx dependency.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.targon_attest import (
    AttestationResult,
    fresh_nonce,
    parse_tower_response,
)
from shared.targon_breaker import CircuitBreaker, TargonUnavailable
from shared.targon_client import (
    DEFAULT_BASE_URL,
    RegistryCreds,
    TargonClient,
    TargonError,
    WorkloadHandle,
    _extract_cvm_ip,
)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("TARGON_API_KEY", "test-key")
    return "test-key"


@pytest.fixture
def fast_breaker():
    return CircuitBreaker(threshold=2, reset_after=0.05)


# ── Circuit breaker ─────────────────────────────────────────────────


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_starts_closed(self):
        b = CircuitBreaker()
        assert b.state == "closed"
        await b.before_call()  # does not raise

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        b = CircuitBreaker(threshold=3)
        for _ in range(3):
            await b.on_failure()
        assert b.state == "open"
        with pytest.raises(TargonUnavailable):
            await b.before_call()

    @pytest.mark.asyncio
    async def test_success_resets_counter(self):
        b = CircuitBreaker(threshold=3)
        await b.on_failure()
        await b.on_failure()
        await b.on_success()
        assert b._consecutive_failures == 0
        assert b.state == "closed"

    @pytest.mark.asyncio
    async def test_half_opens_after_reset(self):
        clock = FakeClock()
        b = CircuitBreaker(threshold=2, reset_after=10.0, clock=clock)
        await b.on_failure()
        await b.on_failure()
        assert b.state == "open"
        clock.t = 11.0
        assert b.state == "half_open"
        # Half-open lets one trial through.
        await b.before_call()
        # A second concurrent half-open trial is rejected.
        with pytest.raises(TargonUnavailable):
            await b.before_call()
        # On success, breaker fully closes.
        await b.on_success()
        assert b.state == "closed"


# ── Targon client constructor ───────────────────────────────────────


class TestConstructor:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TARGON_API_KEY", raising=False)
        with pytest.raises(TargonError):
            TargonClient()

    def test_uses_env_var(self, api_key):
        c = TargonClient()
        assert c.api_key == api_key

    def test_explicit_key_overrides_env(self, api_key):
        c = TargonClient(api_key="explicit")
        assert c.api_key == "explicit"


# ── Workload management ─────────────────────────────────────────────


class _FakeServerlessResp:
    def __init__(self, uid="wl_abc", url="https://wl_abc.targon.network", name="trainer-1", status="running"):
        self.uid = uid
        self.url = url
        self.name = name
        self.status = status


class _FakeListItem:
    def __init__(self, uid, urls=None, status="running"):
        self.uid = uid
        self.urls = urls or []
        self.status = status
        self.name = f"item-{uid}"


class TestDeployWorkload:
    @pytest.mark.asyncio
    async def test_passes_empty_command_args(self, api_key):
        c = TargonClient()

        captured = {}

        class FakeSL:
            def __init__(self, sdk):
                pass

            async def deploy_container(self, **kwargs):
                captured.update(kwargs)
                return _FakeServerlessResp()

        with patch("targon.client.serverless.AsyncServerlessClient", FakeSL), \
             patch.object(c, "_get_sdk_client", return_value=MagicMock()):
            handle = await c.deploy_workload(image="ghcr.io/x:y", gpu_class="H200")

        assert captured["command"] == []
        assert captured["args"] == []
        assert captured["image"] == "ghcr.io/x:y"
        assert captured["resource"] == "h200"
        assert handle.uid == "wl_abc"
        # Targon routing-edge subdomain → empty cvm_ip until the SDK
        # exposes the raw CVM IP. See _extract_cvm_ip docstring.
        assert handle.cvm_ip == ""
        assert handle.gpu_class == "H200"

    @pytest.mark.asyncio
    async def test_with_registry_creds(self, api_key):
        c = TargonClient()
        captured = {}

        class FakeSL:
            def __init__(self, sdk):
                pass

            async def deploy_container(self, **kwargs):
                captured.update(kwargs)
                return _FakeServerlessResp()

        creds = RegistryCreds(server="ghcr.io", username="u", password="p")
        with patch("targon.client.serverless.AsyncServerlessClient", FakeSL), \
             patch.object(c, "_get_sdk_client", return_value=MagicMock()):
            await c.deploy_workload(image="img", gpu_class="H100", registry=creds)
        assert captured["registry"] is not None


class TestValidateCredentials:
    @pytest.mark.asyncio
    async def test_passes_on_successful_list(self, api_key):
        c = TargonClient()

        class FakeSL:
            def __init__(self, sdk): pass
            async def list_container(self): return []

        with patch("targon.client.serverless.AsyncServerlessClient", FakeSL), \
             patch.object(c, "_get_sdk_client", return_value=MagicMock()):
            await c.validate_credentials()  # does not raise

    @pytest.mark.asyncio
    async def test_raises_unavailable_on_repeated_5xx(self, api_key, fast_breaker):
        c = TargonClient(breaker=fast_breaker, max_retries=2)

        class FakeSL:
            def __init__(self, sdk): pass
            async def list_container(self):
                raise httpx.HTTPStatusError(
                    "boom", request=MagicMock(),
                    response=MagicMock(status_code=503),
                )

        with patch("targon.client.serverless.AsyncServerlessClient", FakeSL), \
             patch.object(c, "_get_sdk_client", return_value=MagicMock()), \
             patch("shared.targon_client.asyncio.sleep", AsyncMock()):
            with pytest.raises(TargonUnavailable):
                await c.validate_credentials()


class TestTeardown:
    @pytest.mark.asyncio
    async def test_calls_delete(self, api_key):
        c = TargonClient()
        called = {}

        class FakeSL:
            def __init__(self, sdk): pass
            async def delete_container(self, uid):
                called["uid"] = uid
                return {}

        with patch("targon.client.serverless.AsyncServerlessClient", FakeSL), \
             patch.object(c, "_get_sdk_client", return_value=MagicMock()):
            await c.teardown_workload("wl_abc")
        assert called["uid"] == "wl_abc"

    @pytest.mark.asyncio
    async def test_swallows_targon_unavailable(self, api_key):
        c = TargonClient(breaker=CircuitBreaker(threshold=1))
        await c.breaker.on_failure()  # open the breaker

        class FakeSL:
            def __init__(self, sdk): pass
            async def delete_container(self, uid): return {}

        # Should not raise — TargonUnavailable swallowed in teardown.
        with patch("targon.client.serverless.AsyncServerlessClient", FakeSL), \
             patch.object(c, "_get_sdk_client", return_value=MagicMock()):
            await c.teardown_workload("wl_abc")


# ── verify_image_digest ─────────────────────────────────────────────


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in.

    Exposes ``.post`` returning a mock response; ``.get`` similarly.
    Use ``handler`` to vary behaviour per (method, url).
    """

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


def _ok_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _err_response(status, payload=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "err", request=MagicMock(), response=MagicMock(status_code=status),
    )
    return resp


class TestVerifyImageDigest:
    @pytest.mark.asyncio
    async def test_returns_true_when_verified(self, api_key):
        c = TargonClient()

        def handler(method, url, kwargs):
            assert method == "POST"
            assert "/tha/v2/workloads/verify" in url
            assert kwargs["json"]["workload_uid"] == "wl_x"
            assert kwargs["json"]["expected_digest"] == "sha256:abc"
            return _ok_response({"verified": True})

        with patch("shared.targon_client.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            ok = await c.verify_image_digest("wl_x", "sha256:abc")
        assert ok is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_verified(self, api_key):
        c = TargonClient()

        def handler(method, url, kwargs):
            return _ok_response({"verified": False, "reason": "digest mismatch"})

        with patch("shared.targon_client.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            ok = await c.verify_image_digest("wl_x", "sha256:abc")
        assert ok is False

    @pytest.mark.asyncio
    async def test_5xx_raises_targon_unavailable_after_retries(self, api_key, fast_breaker):
        c = TargonClient(breaker=fast_breaker, max_retries=2)

        def handler(method, url, kwargs):
            return _err_response(500)

        with patch("shared.targon_client.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)), \
             patch("shared.targon_client.asyncio.sleep", AsyncMock()):
            with pytest.raises(TargonUnavailable):
                await c.verify_image_digest("wl_x", "sha256:abc")


# ── verify_attestation ──────────────────────────────────────────────


class TestVerifyAttestation:
    @pytest.mark.asyncio
    async def test_happy_path(self, api_key):
        c = TargonClient()

        evidence = {"quote": "AAA", "user_data": {"nvcc_response": "BBB"}}
        verdict = {
            "verified": True,
            "gpu": {"class": "H200", "count": 1},
            "cpu": {"model": "Intel Xeon"},
        }

        def handler(method, url, kwargs):
            if "/api/v1/evidence" in url:
                return _ok_response(evidence)
            if "verify-attestation" in url:
                # Tower received correct payload shape.
                p = kwargs["json"]
                assert p["attestation"] == evidence
                assert p["ip_address"] == "1.2.3.4"
                assert p["nonce"]
                return _ok_response(verdict)
            raise AssertionError(f"unexpected url {url}")

        # Both the cvm-evidence call (in targon_attest) and the tower call
        # (in targon_attest) use httpx.AsyncClient.
        with patch("shared.targon_attest.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await c.verify_attestation(
                cvm_ip="1.2.3.4",
                miner_hotkey="mh",
                validator_hotkey="vh",
            )

        assert result.verified is True
        assert result.gpu_class == "H200"
        assert result.gpu_count == 1

    @pytest.mark.asyncio
    async def test_evidence_failure_returns_unverified(self, api_key):
        c = TargonClient()

        def handler(method, url, kwargs):
            if "/api/v1/evidence" in url:
                raise httpx.ConnectError("CVM unreachable", request=MagicMock())
            raise AssertionError("tower should not be called")

        with patch("shared.targon_attest.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)), \
             patch("shared.targon_client.asyncio.sleep", AsyncMock()):
            with pytest.raises(TargonUnavailable):
                await c.verify_attestation(
                    cvm_ip="1.2.3.4", miner_hotkey="mh", validator_hotkey="vh",
                )

    @pytest.mark.asyncio
    async def test_no_cvm_ip_returns_unverified_without_calling_endpoints(self, api_key):
        """Empty cvm_ip → AttestationResult(verified=False) with a clear error,
        no CVM or tower calls made."""
        c = TargonClient()
        called = {"any": False}

        def handler(method, url, kwargs):
            called["any"] = True
            return _ok_response({"verified": True})

        with patch("shared.targon_attest.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await c.verify_attestation(
                cvm_ip="", miner_hotkey="mh", validator_hotkey="vh",
            )
        assert result.verified is False
        assert "no CVM IP" in result.error
        assert not called["any"]

    @pytest.mark.asyncio
    async def test_tower_rejects_returns_unverified(self, api_key):
        c = TargonClient()
        evidence = {"quote": "AAA"}
        verdict = {"verified": False, "error": "TDX quote signature invalid"}

        def handler(method, url, kwargs):
            if "/api/v1/evidence" in url:
                return _ok_response(evidence)
            return _ok_response(verdict)

        with patch("shared.targon_attest.httpx.AsyncClient", lambda **kw: _FakeAsyncClient(handler)):
            result = await c.verify_attestation(
                cvm_ip="1.2.3.4", miner_hotkey="mh", validator_hotkey="vh",
            )

        assert result.verified is False
        assert "TDX quote signature invalid" in result.error


# ── Helpers ─────────────────────────────────────────────────────────


class TestExtractCvmIp:
    @pytest.mark.parametrize("url,expected", [
        # Routing-edge subdomains return empty — see docstring.
        ("https://wl_abc.targon.network", ""),
        ("https://wl_xyz.targon.com", ""),
        # Raw IP is returned as-is.
        ("https://1.2.3.4:8081", "1.2.3.4"),
        # Unrelated host (e.g. private deployment) is passed through.
        ("https://my-cvm.example.com", "my-cvm.example.com"),
        ("", ""),
        ("not-a-url", ""),
    ])
    def test_extract(self, url, expected):
        assert _extract_cvm_ip(url) == expected


class TestParseTowerResponse:
    def test_unverified_keeps_error(self):
        result = parse_tower_response({"verified": False, "error": "bad quote"})
        assert result.verified is False
        assert result.error == "bad quote"

    def test_verified_extracts_gpu_cpu(self):
        result = parse_tower_response({
            "verified": True,
            "gpu": {"class": "H100", "count": 2},
            "cpu": {"model": "Xeon Platinum"},
        })
        assert result.verified is True
        assert result.gpu_class == "H100"
        assert result.gpu_count == 2
        assert result.cpu_model == "Xeon Platinum"

    def test_alternate_schema(self):
        # Tower may flatten the response — accept that too.
        result = parse_tower_response({
            "verified": True, "gpu_class": "H200", "gpu_count": 1, "cpu_model": "X",
        })
        assert result.gpu_class == "H200"

    def test_non_dict(self):
        result = parse_tower_response("not a dict")
        assert result.verified is False
