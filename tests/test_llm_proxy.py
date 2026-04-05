"""Tests for validator.llm_proxy — rate-limited LLM chat proxy."""

import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from validator.llm_proxy import (
    ChatRequest,
    ChatMessage,
    LLMProxy,
    register_routes,
    set_proxy,
)


# ── Unit tests for LLMProxy ────────────────────────────────────────────


class TestLLMProxy:
    def test_initial_quota(self):
        proxy = LLMProxy(max_requests=50)
        assert proxy.remaining_requests(0) == 50
        assert proxy.remaining_requests(1) == 50

    def test_rate_limiting(self):
        proxy = LLMProxy(max_requests=3)
        proxy._record_request(0)
        assert proxy.remaining_requests(0) == 2
        proxy._record_request(0)
        assert proxy.remaining_requests(0) == 1
        proxy._record_request(0)
        assert proxy.remaining_requests(0) == 0

    def test_rate_limit_per_miner(self):
        proxy = LLMProxy(max_requests=2)
        proxy._record_request(0)
        proxy._record_request(0)
        assert proxy.remaining_requests(0) == 0
        assert proxy.remaining_requests(1) == 2

    def test_reset_limits(self):
        proxy = LLMProxy(max_requests=2)
        proxy._record_request(0)
        proxy._record_request(0)
        assert proxy.remaining_requests(0) == 0
        proxy.reset_limits()
        assert proxy.remaining_requests(0) == 2

    def test_old_requests_pruned(self):
        proxy = LLMProxy(max_requests=2, tempo_seconds=10)
        proxy._request_counts[0].append(time.time() - 20)
        assert proxy.remaining_requests(0) == 2

    @pytest.mark.asyncio
    async def test_chat_rate_limit_exceeded(self):
        proxy = LLMProxy(max_requests=1)
        proxy._record_request(0)
        req = ChatRequest(messages=[ChatMessage(role="user", content="hello")])
        with pytest.raises(HTTPException) as exc_info:
            await proxy.chat(0, req)
        assert exc_info.value.status_code == 429

    def test_default_api_key_openai(self):
        proxy = LLMProxy(provider="openai")
        # Falls back to OPENAI_API_KEY env var (may be empty)
        assert isinstance(proxy.api_key, str)

    def test_default_base_url_openai(self):
        proxy = LLMProxy(provider="openai")
        assert "openai" in proxy.base_url

    def test_default_base_url_anthropic(self):
        proxy = LLMProxy(provider="anthropic")
        assert "anthropic" in proxy.base_url


# ── Integration tests with FastAPI ──────────────────────────────────────────


def _make_app() -> tuple[FastAPI, LLMProxy]:
    app = FastAPI()
    proxy = LLMProxy(max_requests=5)
    set_proxy(proxy)
    register_routes(app)
    return app, proxy


class TestLLMRoutes:
    def test_health(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/llm/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_quota(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/llm/quota", headers={"X-Miner-UID": "0"})
        assert r.status_code == 200
        assert r.json()["remaining_requests"] == 5
        assert r.json()["miner_uid"] == 0

    def test_missing_miner_uid(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/llm/quota")
        assert r.status_code == 400

    def test_invalid_miner_uid(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/llm/quota", headers={"X-Miner-UID": "abc"})
        assert r.status_code == 400

    def test_negative_miner_uid(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/llm/quota", headers={"X-Miner-UID": "-1"})
        assert r.status_code == 400

    def test_chat_validation_empty_messages(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.post(
            "/llm/chat",
            json={"messages": []},
            headers={"X-Miner-UID": "0"},
        )
        assert r.status_code == 422
