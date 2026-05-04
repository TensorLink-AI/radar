"""Epistula authentication — sign and verify HTTP requests.

Extracted from validator/gossip.py. Used by validators, trainers, and miners
for authenticating HTTP requests with SR25519 hotkey signatures.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional

import bittensor as bt

logger = logging.getLogger(__name__)

# Epistula timestamp tolerance (seconds). Cross-machine clock skew between
# validator hosts and miner trainer pods is the most common cause of 403s
# here, so the default is intentionally generous. Replay protection still
# relies on the SR25519 signature + uuid4 nonce; the timestamp window only
# bounds how long a captured signed message can be replayed.
EPISTULA_TIMESTAMP_TOLERANCE = int(os.getenv("RADAR_EPISTULA_TOLERANCE", "120"))


def sign_request(wallet: bt.Wallet, body: bytes) -> dict[str, str]:
    """Sign an outgoing HTTP request with Epistula headers.

    Headers: X-Epistula-Signed-By, X-Epistula-Timestamp,
             X-Epistula-Nonce, X-Epistula-Signature
    """
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
    """Verify an incoming request's Epistula headers.

    Returns (ok, error_message, sender_hotkey).
    """
    hotkey = headers.get("x-epistula-signed-by", "")
    timestamp_str = headers.get("x-epistula-timestamp", "")
    nonce = headers.get("x-epistula-nonce", "")
    signature = headers.get("x-epistula-signature", "")

    if not all([hotkey, timestamp_str, nonce, signature]):
        return False, "Missing Epistula headers", ""

    # Check timestamp freshness
    try:
        ts = int(timestamp_str)
    except ValueError:
        return False, "Invalid timestamp", ""
    skew = int(time.time() - ts)
    if abs(skew) > EPISTULA_TIMESTAMP_TOLERANCE:
        # Surface the actual skew + tolerance so miner operators can see at
        # a glance that this is a clock-sync problem (NTP, container drift)
        # rather than a signing bug, and tune RADAR_EPISTULA_TOLERANCE if
        # needed.
        err = (
            f"Timestamp too old or too far in future "
            f"(skew={skew}s, tolerance=±{EPISTULA_TIMESTAMP_TOLERANCE}s)"
        )
        return False, err, ""

    # Verify sender is registered
    uid = get_uid_for_hotkey(metagraph, hotkey)
    if uid is None:
        return False, "Unknown hotkey", ""

    if require_stake and float(metagraph.S[uid]) <= 0:
        return False, "Not a validator (no stake)", ""

    # Verify signature
    message = body + timestamp_str.encode() + nonce.encode()
    try:
        keypair = bt.Keypair(ss58_address=hotkey)
        if not keypair.verify(message, bytes.fromhex(signature)):
            return False, "Invalid signature", ""
    except Exception:
        return False, "Signature verification failed", ""

    return True, "", hotkey


def get_uid_for_hotkey(metagraph, hotkey: str) -> Optional[int]:
    """Look up UID for a hotkey in the metagraph."""
    hotkeys = metagraph.hotkeys
    if hotkeys is None:
        return None
    for i in range(metagraph.n):
        if i < len(hotkeys) and hotkeys[i] == hotkey:
            return i
    return None
