"""Tests for per-round agent token auth in db_proxy."""

import secrets

from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from validator.db_proxy import (
    app, rotate_agent_token, get_agent_token, _verify_agent_token,
)


class TestAgentToken:
    def test_rotate_returns_token(self):
        token = rotate_agent_token()
        assert len(token) > 20
        assert get_agent_token() == token

    def test_rotate_changes_token(self):
        t1 = rotate_agent_token()
        t2 = rotate_agent_token()
        assert t1 != t2

    def test_verify_valid_token(self):
        token = rotate_agent_token()
        request = MagicMock()
        request.headers = {"X-Agent-Token": token}
        assert _verify_agent_token(request) is True

    def test_verify_invalid_token(self):
        rotate_agent_token()
        request = MagicMock()
        request.headers = {"X-Agent-Token": "wrong-token"}
        assert _verify_agent_token(request) is False

    def test_verify_missing_token(self):
        rotate_agent_token()
        request = MagicMock()
        request.headers = {}
        assert _verify_agent_token(request) is False


class TestGatedClientDefaultHeaders:
    """Test that GatedClient sends default headers on all requests."""

    def test_default_headers_applied(self):
        from shared.url_gate import GatedClient
        import urllib.request

        client = GatedClient(
            allowed_prefixes=["http://example.com/"],
            default_headers={"X-Agent-Token": "secret123"},
        )
        # We can't easily test actual HTTP calls without a server,
        # but we can verify the headers dict is stored
        assert client._default_headers == {"X-Agent-Token": "secret123"}

    def test_no_default_headers(self):
        from shared.url_gate import GatedClient
        client = GatedClient(allowed_prefixes=[])
        assert client._default_headers == {}
