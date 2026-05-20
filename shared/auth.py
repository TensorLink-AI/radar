"""HMAC-based shared-secret authentication.

Replaces the previous chain-based Epistula signing. Callers sign request
bodies with a shared secret (``RADAR_SHARED_SECRET``) and the receiving
side validates the ``X-Radar-Signature`` header with a constant-time
compare.

This module keeps a few thin compatibility shims (``set_auth``, the
``Epistula`` header names returned by :func:`sign_request`) so that the
rest of the codebase — which used to call into chain-based signing —
keeps working without changes. New code should prefer the explicit
``sign_request_hmac`` / ``verify_request_hmac`` helpers.
"""

from __future__ import annotations

import hmac
import hashlib
import logging
import os
import time
import uuid
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Header used by the new HMAC scheme. Kept in sync with runner/server.py
# and validator/db_proxy.py.
SIGNATURE_HEADER = "X-Radar-Signature"

# Kept for backward compatibility with code that imports the constant.
EPISTULA_TIMESTAMP_TOLERANCE = int(os.getenv("RADAR_EPISTULA_TOLERANCE", "30"))

_ENV_SECRET = "RADAR_SHARED_SECRET"


def _get_secret(explicit: Optional[str] = None) -> str:
    """Return the configured shared secret (explicit override > env)."""
    if explicit is not None:
        return explicit
    return os.getenv(_ENV_SECRET, "") or ""


# ── Primary HMAC primitives ─────────────────────────────────────────


def sign_request_hmac(body: bytes, secret: Optional[str] = None) -> str:
    """Return the hex HMAC-SHA256 of ``body`` keyed by ``secret``.

    If ``secret`` is None we read ``RADAR_SHARED_SECRET`` from the env.
    Returns an empty string if no secret is configured (dev mode).
    """
    s = _get_secret(secret)
    if not s:
        return ""
    mac = hmac.new(s.encode("utf-8"), body or b"", hashlib.sha256)
    return mac.hexdigest()


def verify_request_hmac(
    body: bytes, signature: str, secret: Optional[str] = None,
) -> bool:
    """Constant-time HMAC-SHA256 verification.

    Returns ``True`` when the signature matches. If the secret is unset,
    returns ``False`` — callers may decide to fall back to "dev mode" in
    that case (see ``runner/server.py``).
    """
    s = _get_secret(secret)
    if not s:
        return False
    expected = sign_request_hmac(body, s)
    if not expected or not signature:
        return False
    return hmac.compare_digest(expected, signature)


# ── Compatibility shims used across the codebase ────────────────────


def sign_request(wallet=None, body: Union[bytes, str] = b"") -> dict[str, str]:
    """Return signed headers for an outbound HTTP request.

    The ``wallet`` argument is ignored (it used to carry a Bittensor
    Keypair). We always sign with the process-wide shared secret. The
    returned dict carries both the new ``X-Radar-Signature`` header and a
    minimal set of the legacy Epistula-style headers so old callers keep
    working unmodified.
    """
    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    else:
        body_bytes = body or b""

    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    signature = sign_request_hmac(body_bytes)

    headers = {
        # New HMAC header — the one runner/server.py actually checks.
        SIGNATURE_HEADER: signature,
        # Legacy header names retained so unrelated code paths that read
        # them (logs, proxies forwarding identity) don't break.
        "X-Epistula-Signed-By": "",
        "X-Epistula-Timestamp": timestamp,
        "X-Epistula-Nonce": nonce,
        "X-Epistula-Signature": signature,
    }
    return headers


def verify_request(
    headers: dict,
    body: bytes,
    *args,
    **kwargs,
) -> tuple[bool, str, str]:
    """Verify the HMAC signature on an inbound request.

    Returns ``(ok, signer, error)``. ``signer`` is left empty under the
    shared-secret scheme. Header lookup is case-insensitive against both
    the new and the legacy header names.

    If no secret is configured the request is allowed through (dev
    fallback) and ``signer == "dev-mode"``.
    """
    if not headers:
        headers = {}
    # Case-insensitive lookup
    def _h(name: str) -> str:
        for k, v in headers.items():
            if k.lower() == name.lower():
                return v or ""
        return ""

    if not _get_secret():
        return True, "dev-mode", ""

    sig = _h(SIGNATURE_HEADER) or _h("X-Epistula-Signature")
    if not sig:
        return False, "", "Missing signature header"

    if verify_request_hmac(body, sig):
        return True, "", ""
    return False, "", "Bad signature"


def set_auth(*args, **kwargs):
    """Compatibility entry point.

    Old code wired the on-chain peer set in here to teach the auth layer
    about registered hotkeys. With shared-secret HMAC that bookkeeping
    isn't needed, but we still load the peer registry so callers that
    depend on a populated peer cache (rate limiting, logging) get a warm
    cache.
    """
    try:
        from shared.peers import load_peers
        peers = load_peers()
        logger.debug("set_auth: loaded %d peers", len(peers))
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("set_auth: peer load failed: %s", e)
    return None


def register_peers_for_auth(*args, **kwargs):
    """Alias for :func:`set_auth` used by some callers."""
    return set_auth(*args, **kwargs)


def get_uid_for_hotkey(hotkey: str, *args, **kwargs) -> Optional[int]:
    """Lookup helper backed by the static peer registry."""
    from shared.peers import get_peer_by_hotkey
    p = get_peer_by_hotkey(hotkey)
    return p.uid if p else None
