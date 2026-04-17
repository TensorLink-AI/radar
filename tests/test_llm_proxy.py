"""Tests for validator/llm_proxy.py — Chutes AI LLM proxy."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from validator.llm_proxy import LLMProxy


def _make_payload(model="gpt-4", content="Hello"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.7,
        "max_tokens": 256,
    }


class TestLLMProxy:
    def test_remaining_queries_initial(self):
        proxy = LLMProxy(max_queries=10)
        assert proxy.remaining_queries(0) == 10

    def test_remaining_queries_decreases(self):
        proxy = LLMProxy(max_queries=3)
        proxy._record_query(0)
        proxy._record_query(0)
        assert proxy.remaining_queries(0) == 1

    def test_allowed_models_empty_allows_all(self):
        proxy = LLMProxy(allowed_models=[])
        # Should not raise
        proxy._validate_model("any-model")

    def test_allowed_models_rejects_unlisted(self):
        proxy = LLMProxy(allowed_models=["gpt-4", "claude-3"])
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            proxy._validate_model("bad-model")
        assert exc_info.value.status_code == 400
        assert "bad-model" in str(exc_info.value.detail)

    def test_allowed_models_accepts_listed(self):
        proxy = LLMProxy(allowed_models=["gpt-4", "claude-3"])
        proxy._validate_model("gpt-4")  # should not raise

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        proxy = LLMProxy(max_queries=1)
        proxy._record_query(5)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("chat/completions", 5, _make_payload())
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_model_rejected(self):
        proxy = LLMProxy(allowed_models=["gpt-4"], max_queries=10)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("chat/completions", 0, _make_payload(model="bad"))
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_model_rejected(self):
        proxy = LLMProxy(max_queries=10)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("chat/completions", 0,
                                {"messages": [{"role": "user", "content": "hi"}]})
        assert exc_info.value.status_code == 400

    def test_reset_limits(self):
        proxy = LLMProxy(max_queries=5)
        proxy._record_query(0)
        proxy._record_query(0)
        proxy.reset_limits()
        assert proxy.remaining_queries(0) == 5

    @pytest.mark.asyncio
    async def test_blocked_fields_stripped(self):
        """Blocked fields like 'api_key' and 'user' should be removed."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = _make_payload()
        payload["api_key"] = "evil-key"
        payload["user"] = "impersonate"

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await proxy.forward("chat/completions", 0, payload)

        assert "api_key" not in payload
        assert "user" not in payload

    @pytest.mark.asyncio
    async def test_stream_field_preserved(self):
        """stream: true should NOT be stripped — it's passed to upstream."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = _make_payload()
        payload["stream"] = True

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await proxy.forward("chat/completions", 0, payload)

        # stream should still be in payload (not blocked)
        assert payload.get("stream") is True

    @pytest.mark.asyncio
    async def test_max_tokens_capped(self):
        """max_tokens above 16384 should be capped."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = _make_payload()
        payload["max_tokens"] = 100000

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await proxy.forward("chat/completions", 0, payload)

        assert payload["max_tokens"] == 16384

    @pytest.mark.asyncio
    async def test_tool_calls_in_payload_accepted(self):
        """Payload with tools should pass validation (fail at HTTP, not parse)."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        }

        from fastapi import HTTPException
        # Should fail at HTTP call, not at validation — tools pass through
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("chat/completions", 0, payload)
        # Should be a connection error (502), not a validation error (400)
        assert exc_info.value.status_code in (502, 503)

    @pytest.mark.asyncio
    async def test_openai_params_preserved(self):
        """Extra OpenAI params (top_p, seed, response_format, etc.) pass through."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = _make_payload()
        payload["top_p"] = 0.9
        payload["frequency_penalty"] = 0.5
        payload["presence_penalty"] = 0.3
        payload["seed"] = 42
        payload["response_format"] = {"type": "json_object"}
        payload["stop"] = ["\n"]
        payload["n"] = 2

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await proxy.forward("chat/completions", 0, payload)

        # All params should survive preparation
        assert payload["top_p"] == 0.9
        assert payload["frequency_penalty"] == 0.5
        assert payload["seed"] == 42
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["stop"] == ["\n"]
        assert payload["n"] == 2

    @pytest.mark.asyncio
    async def test_timeout_opens_circuit_immediately(self):
        """A single timeout should trip the circuit breaker."""
        import httpx
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")

        # Mock the client to raise TimeoutException
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
        proxy._client = mock_client

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("chat/completions", 0, _make_payload())
        assert exc_info.value.status_code == 502

        # Circuit should now be open — next call fails fast with 503
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("chat/completions", 0, _make_payload())
        assert exc_info.value.status_code == 503
        assert "unavailable" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_circuit_open_returns_retry_after(self):
        """When the circuit is open, 503 responses should carry Retry-After."""
        import httpx
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
        proxy._client = mock_client

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await proxy.forward("chat/completions", 0, _make_payload())

        # Circuit is now open; next call should return 503 with Retry-After.
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("chat/completions", 0, _make_payload())
        assert exc_info.value.status_code == 503
        headers = exc_info.value.headers or {}
        assert "Retry-After" in headers
        assert int(headers["Retry-After"]) > 0

    @pytest.mark.asyncio
    async def test_connect_timeout_is_short(self):
        """Ensure proxy fails fast when upstream is unreachable."""
        import time
        proxy = LLMProxy(
            chutes_url="http://192.0.2.1:1",  # TEST-NET-1, blackholed
            chutes_api_key="test-key",
            max_queries=10,
        )
        from fastapi import HTTPException
        t0 = time.monotonic()
        with pytest.raises(HTTPException):
            await proxy.forward("chat/completions", 0, _make_payload())
        elapsed = time.monotonic() - t0
        # Should fail within ~connect timeout (5s) + 1 retry + small overhead,
        # definitely not 70+ seconds
        assert elapsed < 20, f"connect not capped: {elapsed:.1f}s"

    def test_timeout_values(self):
        """Connect timeout is short; read timeout accommodates slow models."""
        proxy = LLMProxy()
        assert proxy.timeout.connect == 5.0
        assert proxy.timeout.read == 120.0
        assert proxy.timeout.pool == 10.0

    def test_recent_latencies_deque_bounded(self):
        """Latency deque should be capped at 50 entries."""
        proxy = LLMProxy()
        assert proxy._recent_latencies.maxlen == 50
        for i in range(100):
            proxy._recent_latencies.append((float(i), 1.0))
        assert len(proxy._recent_latencies) == 50

    @pytest.mark.asyncio
    async def test_embeddings_path(self):
        """Embeddings endpoint should accept payload and fail at HTTP, not validation."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = {"model": "text-embedding-ada-002", "input": "hello world"}

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("embeddings", 0, payload)
        assert exc_info.value.status_code in (502, 503)

    @pytest.mark.asyncio
    async def test_completions_path(self):
        """Text completions endpoint should work."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = {"model": "gpt-3.5-turbo-instruct", "prompt": "Hello", "max_tokens": 50}

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.forward("completions", 0, payload)
        assert exc_info.value.status_code in (502, 503)
