"""Tests for shared/runpod_client.py — the RunPod REST API wrapper.

We stub httpx via monkeypatching the AsyncClient so no network calls
escape and we can assert exact request/response shapes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.runpod_client import (
    DEFAULT_BASE_URL,
    PodHandle,
    RegistryAuth,
    RunPodClient,
    RunPodError,
    _parse_pod,
    _proxy_url,
)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    return "test-key"


# ── Constructor ────────────────────────────────────────────────────


class TestConstructor:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with pytest.raises(RunPodError, match="RUNPOD_API_KEY"):
            RunPodClient()

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        client = RunPodClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_env_key_is_picked_up(self, api_key):
        client = RunPodClient()
        assert client.api_key == "test-key"

    def test_default_base_url(self, api_key):
        client = RunPodClient()
        assert client.base_url == DEFAULT_BASE_URL.rstrip("/")

    def test_auth_header(self, api_key):
        client = RunPodClient()
        assert client._headers()["Authorization"] == "Bearer test-key"
        assert client._headers()["Content-Type"] == "application/json"


# ── _parse_pod / _proxy_url helpers ────────────────────────────────


class TestParsePod:
    def test_proxy_url_format(self):
        assert _proxy_url("abc123", 8081) == "https://abc123-8081.proxy.runpod.net"
        assert _proxy_url("", 8081) == ""
        assert _proxy_url("abc123", 0) == ""

    def test_parse_minimal(self):
        pod = _parse_pod({"id": "p1", "imageName": "foo:bar", "desiredStatus": "RUNNING"})
        assert pod.pod_id == "p1"
        assert pod.image_name == "foo:bar"
        assert pod.status == "RUNNING"
        assert pod.is_running

    def test_parse_with_runtime_ports(self):
        pod = _parse_pod({
            "id": "p1",
            "imageName": "foo:bar@sha256:abc",
            "desiredStatus": "RUNNING",
            "runtime": {
                "ports": [
                    {"isIpPublic": True, "ip": "1.2.3.4", "publicPort": 12345, "privatePort": 8081},
                ],
            },
        })
        assert pod.public_ip == "1.2.3.4"
        assert pod.public_port == 12345
        assert pod.proxy_url == "https://p1-8081.proxy.runpod.net"
        assert pod.url == pod.proxy_url  # proxy URL preferred

    def test_url_falls_back_to_raw_ip(self):
        pod = PodHandle(pod_id="", public_ip="1.2.3.4", public_port=8080)
        assert pod.url == "http://1.2.3.4:8080"

    def test_status_aliases(self):
        for s in ("RUNNING", "running", "Ready"):
            assert PodHandle(pod_id="x", status=s).is_running

    def test_gpu_type_extracted(self):
        pod = _parse_pod({
            "id": "p1",
            "imageName": "foo:bar",
            "machine": {"gpuTypeId": "NVIDIA A100 80GB PCIe"},
        })
        assert pod.gpu_type_id == "NVIDIA A100 80GB PCIe"


# ── HTTP request handling ──────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, content=b"{}"):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.content = content
        self.request = MagicMock()
        self.response = self

    def json(self):
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=self.request, response=self,
            )


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient that returns canned responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def request(self, method, url, *, headers=None, json=None, params=None):
        self.calls.append({"method": method, "url": url, "json": json, "params": params})
        if not self._responses:
            raise AssertionError("Unexpected extra request")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _patch_httpx(responses):
    fake = _FakeAsyncClient(responses)
    return patch("shared.runpod_client.httpx.AsyncClient", return_value=fake), fake


class TestDeployPod:
    @pytest.mark.asyncio
    async def test_deploy_pod_builds_request_body(self, api_key):
        canned = _FakeResponse(json_body={
            "id": "pod-123", "imageName": "foo@sha256:abc",
            "desiredStatus": "PROVISIONING",
        })
        patcher, fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            handle = await client.deploy_pod(
                image="foo@sha256:abc",
                gpu_type_ids=["NVIDIA A100 80GB PCIe"],
                gpu_count=1,
                name="radar-trainer-x-7",
                env={"FOO": "bar"},
                cloud_type="SECURE",
            )
        assert handle.pod_id == "pod-123"
        body = fake.calls[0]["json"]
        assert body["imageName"] == "foo@sha256:abc"
        assert body["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"]
        assert body["cloudType"] == "SECURE"
        assert body["env"] == {"FOO": "bar"}
        assert body["ports"] == "8081/http"

    @pytest.mark.asyncio
    async def test_deploy_pod_passes_registry_auth(self, api_key):
        canned = _FakeResponse(json_body={"id": "p1", "imageName": "x"})
        patcher, fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            await client.deploy_pod(
                image="ghcr.io/private/x@sha256:y",
                gpu_type_ids=["NVIDIA H100 80GB HBM3"],
                registry=RegistryAuth(username="u", password="p", server="ghcr.io"),
            )
        body = fake.calls[0]["json"]
        assert body["containerRegistryAuth"]["username"] == "u"
        assert body["containerRegistryAuth"]["registry"] == "ghcr.io"


class TestGetTeardownList:
    @pytest.mark.asyncio
    async def test_get_pod_returns_handle(self, api_key):
        canned = _FakeResponse(json_body={"id": "p1", "imageName": "x", "desiredStatus": "RUNNING"})
        patcher, fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            pod = await client.get_pod("p1")
        assert pod.pod_id == "p1"
        assert fake.calls[0]["method"] == "GET"
        assert fake.calls[0]["url"].endswith("/pods/p1")

    @pytest.mark.asyncio
    async def test_get_pod_404_returns_none(self, api_key):
        canned = _FakeResponse(status_code=404, json_body={"error": "not found"})
        patcher, _fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            assert await client.get_pod("missing") is None

    @pytest.mark.asyncio
    async def test_teardown_pod_404_swallowed(self, api_key):
        canned = _FakeResponse(status_code=404, json_body={})
        patcher, _fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            await client.teardown_pod("already-gone")  # does not raise

    @pytest.mark.asyncio
    async def test_list_active_pods_handles_top_level_list(self, api_key):
        canned = _FakeResponse(json_body=[{"id": "a"}, {"id": "b"}])
        patcher, _fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            pods = await client.list_active_pods(name_prefix="radar-")
        assert [p.pod_id for p in pods] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_validate_credentials_raises_on_401(self, api_key):
        canned = _FakeResponse(status_code=401, json_body={})
        patcher, _fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            with pytest.raises(RunPodError, match="rejected"):
                await client.validate_credentials()


# ── verify_pod_image — the load-bearing security check ─────────────


class TestVerifyPodImage:
    @pytest.mark.asyncio
    async def test_pinned_digest_match_passes(self, api_key):
        canned = _FakeResponse(json_body={
            "id": "p1",
            "imageName": "ghcr.io/foo/bar:v1@sha256:abc123",
            "desiredStatus": "RUNNING",
        })
        patcher, _fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            ok, why = await client.verify_pod_image("p1", "sha256:abc123")
        assert ok, why
        assert why == ""

    @pytest.mark.asyncio
    async def test_unpinned_image_rejected(self, api_key):
        canned = _FakeResponse(json_body={
            "id": "p1",
            "imageName": "ghcr.io/foo/bar:v1",  # tag-only, no @sha256
            "desiredStatus": "RUNNING",
        })
        patcher, _fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            ok, why = await client.verify_pod_image("p1", "sha256:abc123")
        assert not ok
        assert "not digest-pinned" in why

    @pytest.mark.asyncio
    async def test_digest_mismatch_rejected(self, api_key):
        canned = _FakeResponse(json_body={
            "id": "p1",
            "imageName": "ghcr.io/foo/bar:v1@sha256:zzz",
            "desiredStatus": "RUNNING",
        })
        patcher, _fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            ok, why = await client.verify_pod_image("p1", "sha256:abc123")
        assert not ok
        assert "mismatch" in why

    @pytest.mark.asyncio
    async def test_missing_pod_rejected(self, api_key):
        canned = _FakeResponse(status_code=404, json_body={})
        patcher, _fake = _patch_httpx([canned])
        with patcher:
            client = RunPodClient(max_retries=1)
            ok, why = await client.verify_pod_image("ghost", "sha256:abc")
        assert not ok
        assert "not found" in why

    @pytest.mark.asyncio
    async def test_empty_expected_digest_rejected(self, api_key):
        # Defense against the misconfig where OFFICIAL_TRAINING_IMAGE_DIGEST is unset.
        client = RunPodClient(api_key="x", max_retries=1)
        ok, why = await client.verify_pod_image("p1", "")
        assert not ok
        assert "no expected digest" in why
