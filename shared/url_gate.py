"""URL allowlist gate for miner agent sandboxes.

Provides a GatedClient that only allows HTTP requests to pre-approved
URL patterns.  The harness injects this client into the miner's agent
module so it never gets raw ``requests`` / ``urllib`` access.

Allowed URL patterns are simple prefix matches:
  - "https://r2.example.com/"  matches any path under that host
  - "http://localhost:8080/"    matches the validator proxy

The gate is enforced at the application layer *inside* a locked-down
official Docker image that has no ``requests`` / ``urllib`` / ``httpx``
installed.  Even if a miner finds another HTTP library, the container's
iptables rules (set in the Dockerfile) block all egress except to the
addresses behind the allowed URLs.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class URLNotAllowedError(Exception):
    """Raised when a request targets a URL outside the allowlist."""


def check_url(url: str, allowed_prefixes: list[str]) -> bool:
    """Return True if *url* starts with any allowed prefix."""
    for prefix in allowed_prefixes:
        if url.startswith(prefix):
            return True
    return False


def parse_allowed_urls(raw: str) -> list[str]:
    """Parse a comma-separated or JSON list of URL prefixes.

    Accepts either ``"http://a,http://b"`` or ``'["http://a","http://b"]'``.
    Each entry is stripped and trailing slashes are normalised.
    """
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = raw.split(",")
    else:
        items = raw.split(",")
    result: list[str] = []
    for item in items:
        item = str(item).strip().rstrip("/") + "/"
        if item != "/":
            result.append(item)
    return result


def extract_hosts(allowed_prefixes: list[str]) -> list[str]:
    """Extract unique ``host:port`` pairs from allowed URL prefixes.

    Used by the Dockerfile / iptables setup to build the egress allowlist.
    """
    hosts: list[str] = []
    seen: set[str] = set()
    for prefix in allowed_prefixes:
        parsed = urlparse(prefix)
        host = parsed.hostname or ""
        port = parsed.port
        if not host:
            continue
        key = f"{host}:{port}" if port else host
        if key not in seen:
            seen.add(key)
            hosts.append(key)
    return hosts


# ── Gated HTTP helpers (stdlib only, no third-party deps) ────────────

class GatedClient:
    """Minimal HTTP client that enforces a URL allowlist.

    Uses only ``urllib.request`` (stdlib) so it works in a stripped-down
    Docker image with no ``requests`` / ``httpx``.

    The miner agent receives an instance of this as its *only* way to
    make HTTP calls.
    """

    def __init__(
        self,
        allowed_prefixes: list[str],
        default_headers: dict[str, str] | None = None,
        timeout: int = 10,
        llm_timeout: int = 30,
        max_retries: int = 1,
    ):
        self._allowed = allowed_prefixes
        self._default_headers = default_headers or {}
        self._timeout = timeout
        self._llm_timeout = llm_timeout
        self._max_retries = max_retries

    def _apply_headers(self, req) -> None:
        """Apply default headers to a urllib Request."""
        for k, v in self._default_headers.items():
            req.add_header(k, v)

    def _check(self, url: str) -> None:
        if not check_url(url, self._allowed):
            raise URLNotAllowedError(
                f"URL not in allowlist: {url!r}  "
                f"(allowed prefixes: {self._allowed})"
            )

    def _effective_timeout(self, url: str, explicit: int | None) -> int:
        """Return the timeout for a request — longer for LLM endpoints."""
        if explicit is not None:
            return explicit
        if "/llm/" in url or url.rstrip("/").endswith("/llm"):
            return self._llm_timeout
        return self._timeout

    def _retries_for_url(self, url: str) -> int:
        """LLM requests are too expensive to retry — fail fast."""
        if "/llm/" in url or url.rstrip("/").endswith("/llm"):
            return 0
        return self._max_retries

    def _do_request(self, req, url: str, timeout: int) -> tuple[bytes, int]:
        """Execute a urllib request with retries for transient failures.

        Returns (body_bytes, http_status_code).
        Connection errors and timeouts are retried with exponential backoff.
        HTTP 5xx errors are retried. HTTP 4xx errors are raised immediately.

        Note on timeouts: urllib.request.urlopen's `timeout=` parameter only
        bounds the READ phase once the TCP connection is established — it
        does NOT cap the connect phase. Without socket.setdefaulttimeout,
        an unreachable or slow-to-accept server costs ~75-127s per attempt
        (Linux tcp_syn_retries default), which wrecks agent time budgets.
        We set the socket default for the duration of each attempt and
        restore the previous value in `finally` so we don't clobber any
        caller that set their own default.
        """
        import urllib.request

        max_retries = self._retries_for_url(url)
        last_err: BaseException | None = None

        # Note: max_retries=N means N+1 total attempts (1 initial + N retries).
        for attempt in range(1 + max_retries):
            prev_default = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read(), resp.status
            except urllib.error.HTTPError as e:
                if e.code in (502, 503, 504) and attempt < max_retries:
                    wait = min(2 ** attempt, 4)
                    logger.warning(
                        "HTTP %s %s returned %d (attempt %d/%d) — retry in %ds",
                        req.method, url[:80], e.code,
                        attempt + 1, 1 + max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, socket.timeout, OSError) as e:
                last_err = e
                if attempt < max_retries:
                    wait = min(2 ** attempt, 4)
                    logger.warning(
                        "HTTP %s %s failed (attempt %d/%d): %s — retry in %ds",
                        req.method, url[:80],
                        attempt + 1, 1 + max_retries, e, wait,
                    )
                    time.sleep(wait)
                    continue
            finally:
                socket.setdefaulttimeout(prev_default)

        raise last_err  # type: ignore[misc]

    # ── Public API (what miner agents call) ──────────────────────────

    def get(self, url: str, timeout: int | None = None) -> bytes:
        """HTTP GET, returns response body bytes."""
        self._check(url)
        import urllib.request
        req = urllib.request.Request(url, method="GET")
        self._apply_headers(req)
        body, _ = self._do_request(req, url, self._effective_timeout(url, timeout))
        return body

    def get_json(self, url: str, timeout: int | None = None) -> dict:
        """HTTP GET, returns parsed JSON."""
        return json.loads(self.get(url, timeout))

    def post(self, url: str, data: bytes | str, timeout: int | None = None) -> bytes:
        """HTTP POST, returns response body bytes."""
        self._check(url)
        import urllib.request
        if isinstance(data, str):
            data = data.encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        self._apply_headers(req)
        body, _ = self._do_request(req, url, self._effective_timeout(url, timeout))
        return body

    def post_json(self, url: str, payload: dict, timeout: int | None = None) -> dict:
        """HTTP POST with JSON body, returns parsed JSON."""
        return json.loads(self.post(url, json.dumps(payload), timeout))

    def put(self, url: str, data: bytes, content_type: str = "application/octet-stream",
            timeout: int | None = None) -> int:
        """HTTP PUT, returns status code."""
        self._check(url)
        import urllib.request
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Content-Type", content_type)
        self._apply_headers(req)
        _, status = self._do_request(req, url, timeout or self._timeout)
        return status
