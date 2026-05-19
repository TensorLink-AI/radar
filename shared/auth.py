"""Auth primitives — both legacy Epistula and new HMAC/bearer surfaces.

Two coexisting auth schemes during the non-competitive cutover:

  Legacy (Epistula / SR25519, metagraph-backed)
    sign_request(wallet, body)             -> headers
    verify_request(headers, body, mg, …)   -> (ok, error, hotkey)

  New (HMAC service-key + bearer API tokens)
    hmac_sign_request(secret, body, kid)   -> headers
    hmac_verify_request(headers, body, lk) -> (ok, error, key_id)
    extract_bearer(headers)                -> token | None
    hash_api_key(key)                      -> sha256 hex (for DB storage)

The HMAC scheme signs ``f"{timestamp}.{sha256(body).hex()}"`` with
HMAC-SHA256.  ``key_id`` is sent in cleartext so the verifier can look
up the right secret; secrets themselves are never on the wire.

Once every caller is on the new surface we'll drop the Epistula
functions and the ``bittensor`` import below.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Legacy Epistula ──────────────────────────────────────────────────

# Epistula timestamp tolerance (seconds). Cross-machine clock skew between
# validator hosts and miner trainer pods is the most common cause of 403s
# here, so the default is intentionally generous.
EPISTULA_TIMESTAMP_TOLERANCE = int(os.getenv("RADAR_EPISTULA_TOLERANCE", "120"))


def sign_request(wallet, body: bytes) -> dict[str, str]:
    """Sign an outgoing HTTP request with Epistula headers (legacy)."""
    import bittensor as bt  # local import keeps non-chain processes light

    _ = bt  # silence linter
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    message = body + timestamp.encode() + nonce.encode()
    signature = wallet.hotkey.sign(message).hex()
    return {
        "X-Epistula-Signed-By": wallet.hotkey.ss58_address,
        "X-Epistula-Timestamp": timestamp,
        "X-Epistula-Nonce": nonce,
        "X-Epistula-Signature": signature,
    }


def verify_request(
    headers: dict[str, str],
    body: bytes,
    metagraph,
    require_stake: bool = False,
) -> tuple[bool, str, str]:
    """Verify Epistula headers on an inbound request (legacy)."""
    import bittensor as bt

    hotkey = headers.get("x-epistula-signed-by", "")
    timestamp_str = headers.get("x-epistula-timestamp", "")
    nonce = headers.get("x-epistula-nonce", "")
    signature = headers.get("x-epistula-signature", "")

    if not all([hotkey, timestamp_str, nonce, signature]):
        return False, "Missing Epistula headers", ""

    try:
        ts = int(timestamp_str)
    except ValueError:
        return False, "Invalid timestamp", ""
    skew = int(time.time() - ts)
    if abs(skew) > EPISTULA_TIMESTAMP_TOLERANCE:
        err = (
            f"Timestamp too old or too far in future "
            f"(skew={skew}s, tolerance=±{EPISTULA_TIMESTAMP_TOLERANCE}s)"
        )
        return False, err, ""

    uid = get_uid_for_hotkey(metagraph, hotkey)
    if uid is None:
        return False, "Unknown hotkey", ""

    if require_stake and float(metagraph.S[uid]) <= 0:
        return False, "Not a validator (no stake)", ""

    message = body + timestamp_str.encode() + nonce.encode()
    try:
        keypair = bt.Keypair(ss58_address=hotkey)
        if not keypair.verify(message, bytes.fromhex(signature)):
            return False, "Invalid signature", ""
    except Exception:
        return False, "Signature verification failed", ""

    return True, "", hotkey


def get_uid_for_hotkey(metagraph, hotkey: str) -> Optional[int]:
    hotkeys = metagraph.hotkeys
    if hotkeys is None:
        return None
    for i in range(metagraph.n):
        if i < len(hotkeys) and hotkeys[i] == hotkey:
            return i
    return None


# ── New HMAC primitives ──────────────────────────────────────────────

# How far the wire timestamp may drift from the verifier's clock.
HMAC_TIMESTAMP_TOLERANCE = int(os.getenv("RADAR_HMAC_TOLERANCE", "300"))

DEFAULT_KEY_ID = "operator"

H_SIG = "x-radar-signature"
H_TS = "x-radar-timestamp"
H_KID = "x-radar-key-id"
H_AUTH = "authorization"

KeyLookup = Callable[[str], Optional[bytes]]


def _canonical(body: bytes, timestamp: str) -> bytes:
    body_digest = hashlib.sha256(body).hexdigest()
    return f"{timestamp}.{body_digest}".encode()


def hmac_sign_request(
    secret: bytes,
    body: bytes,
    key_id: str = DEFAULT_KEY_ID,
) -> dict[str, str]:
    """HMAC-SHA256 sign a request. Caller attaches the returned headers
    to its outbound HTTP request."""
    if not isinstance(secret, (bytes, bytearray)):
        raise TypeError("secret must be bytes")
    timestamp = str(int(time.time()))
    sig = hmac.new(secret, _canonical(body, timestamp), hashlib.sha256).hexdigest()
    return {
        "X-Radar-Signature": sig,
        "X-Radar-Timestamp": timestamp,
        "X-Radar-Key-Id": key_id,
    }


def hmac_verify_request(
    headers: dict[str, str],
    body: bytes,
    key_lookup: KeyLookup,
    tolerance_s: int = HMAC_TIMESTAMP_TOLERANCE,
) -> tuple[bool, str, str]:
    """Verify HMAC headers on an inbound request.

    ``key_lookup`` resolves ``key_id -> secret``.  Returning ``None``
    fails verification closed.  Returns ``(ok, error, key_id)`` with
    ``key_id`` empty on failure.
    """
    h = {k.lower(): v for k, v in headers.items()}
    sig = h.get(H_SIG, "")
    ts = h.get(H_TS, "")
    kid = h.get(H_KID, "")

    if not sig or not ts or not kid:
        return False, "missing HMAC headers", ""

    try:
        ts_int = int(ts)
    except ValueError:
        return False, "invalid timestamp", ""
    skew = int(time.time()) - ts_int
    if abs(skew) > tolerance_s:
        return False, (
            f"timestamp out of tolerance (skew={skew}s, "
            f"tolerance=±{tolerance_s}s)"
        ), ""

    secret = key_lookup(kid)
    if secret is None:
        return False, "unknown key_id", ""

    expected = hmac.new(secret, _canonical(body, ts), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return False, "bad signature", ""

    return True, "", kid


def static_key_lookup(key_id: str, secret: bytes) -> KeyLookup:
    """Build a ``key_lookup`` that accepts exactly one ``(key_id, secret)``
    — the common case for trainer-side service auth."""
    expected = key_id

    def lookup(kid: str) -> Optional[bytes]:
        if kid == expected:
            return secret
        return None

    return lookup


# ── Bearer tokens (per-agent / per-miner API keys) ───────────────────


def extract_bearer(headers: dict[str, str]) -> Optional[str]:
    """Return the bearer token from an ``Authorization`` header, or
    ``None`` if missing/malformed."""
    h = {k.lower(): v for k, v in headers.items()}
    raw = h.get(H_AUTH, "")
    if not raw:
        return None
    parts = raw.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def hash_api_key(key: str) -> str:
    """SHA-256 hex digest. The DB stores this — never the raw key."""
    return hashlib.sha256(key.encode()).hexdigest()


def load_service_secret(env: str = "RADAR_SERVICE_KEY") -> bytes:
    """Read the shared HMAC service key from the environment.  Raises
    ``RuntimeError`` if unset — callers that need fail-closed behavior
    should let this propagate at startup."""
    raw = os.getenv(env, "").strip()
    if not raw:
        raise RuntimeError(
            f"{env} is unset — set it to a long random string shared "
            f"between operator, validator, and trainer processes."
        )
    return raw.encode()
