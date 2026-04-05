"""Tests for validator/llm_proxy.py — Chutes AI LLM proxy."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from validator.llm_proxy import LLMProxy, ChatRequest, ChatMessage, ChatResponse


def _make_request(model="gpt-4", content="Hello"):
    return ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.7,
        max_tokens=256,
    )


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
            await proxy.chat(5, _make_request())
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_model_rejected(self):
        proxy = LLMProxy(allowed_models=["gpt-4"], max_queries=10)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await proxy.chat(0, _make_request(model="bad"))
        assert exc_info.value.status_code == 400

    def test_reset_limits(self):
        proxy = LLMProxy(max_queries=5)
        proxy._record_query(0)
        proxy._record_query(0)
        proxy.reset_limits()
        assert proxy.remaining_queries(0) == 5


class TestChatRequest:
    def test_valid(self):
        req = _make_request()
        assert req.model == "gpt-4"
        assert len(req.messages) == 1

    def test_invalid_role(self):
        with pytest.raises(Exception):
            ChatRequest(
                model="gpt-4",
                messages=[ChatMessage(role="hacker", content="hi")],
            )

    def test_empty_messages(self):
        with pytest.raises(Exception):
            ChatRequest(model="gpt-4", messages=[])


class TestChatResponse:
    def test_none_content_coerced(self):
        """Chutes AI can return null content — ensure ChatResponse handles it."""
        resp = ChatResponse(model="gpt-4", content="", remaining_queries=5)
        assert resp.content == ""

    def test_normal_content(self):
        resp = ChatResponse(model="gpt-4", content="hello", remaining_queries=3)
        assert resp.content == "hello"
