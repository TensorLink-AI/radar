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
            await proxy.chat(5, _make_payload())
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_model_rejected(self):
        proxy = LLMProxy(allowed_models=["gpt-4"], max_queries=10)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.chat(0, _make_payload(model="bad"))
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_model_rejected(self):
        proxy = LLMProxy(max_queries=10)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.chat(0, {"messages": [{"role": "user", "content": "hi"}]})
        assert exc_info.value.status_code == 400

    def test_reset_limits(self):
        proxy = LLMProxy(max_queries=5)
        proxy._record_query(0)
        proxy._record_query(0)
        proxy.reset_limits()
        assert proxy.remaining_queries(0) == 5

    @pytest.mark.asyncio
    async def test_blocked_fields_stripped(self):
        """Blocked fields like 'stream' and 'api_key' should be removed."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = _make_payload()
        payload["stream"] = True
        payload["api_key"] = "evil-key"

        # Will fail at the HTTP call, but we can verify fields were stripped
        # by checking the payload dict is mutated
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await proxy.chat(0, payload)

        assert "stream" not in payload
        assert "api_key" not in payload

    @pytest.mark.asyncio
    async def test_max_tokens_capped(self):
        """max_tokens above 16384 should be capped."""
        proxy = LLMProxy(max_queries=10, chutes_api_key="test-key")
        payload = _make_payload()
        payload["max_tokens"] = 100000

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await proxy.chat(0, payload)

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
            await proxy.chat(0, payload)
        # Should be a connection error (502), not a validation error (400)
        assert exc_info.value.status_code in (502, 503)
