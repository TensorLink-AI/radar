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

    def __init__(self, allowed_prefixes: list[str], timeout: int = 30):
        self._allowed = allowed_prefixes
        self._timeout = timeout

    def _check(self, url: str) -> None:
        if not check_url(url, self._allowed):
            raise URLNotAllowedError(
                f"URL not in allowlist: {url!r}  "
                f"(allowed prefixes: {self._allowed})"
            )

    # ── Public API (what miner agents call) ──────────────────────────

    def get(self, url: str, timeout: int | None = None) -> bytes:
        """HTTP GET, returns response body bytes."""
        self._check(url)
        import urllib.request
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
            return resp.read()

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
        with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
            return resp.read()

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
        with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
            return resp.status
