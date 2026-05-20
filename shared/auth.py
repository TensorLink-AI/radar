"""Epistula authentication — sign and verify HTTP requests.

Used by validators, trainers, and miners for authenticating HTTP requests
with SR25519 hotkey signatures.

Signature crypto uses `substrate-interface` directly so trainer/miner
containers don't need the full `bittensor` SDK (no chain RPC, no metagraph).
Validators may still pass a live metagraph for registration + stake checks;
other components can use an `allowed_hotkeys` allowlist or signature-only.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Iterable, Optional

from substrateinterface import Keypair

logger = logging.getLogger(__name__)

# Epistula timestamp tolerance (seconds) — increase for cross-machine clock skew
EPISTULA_TIMESTAMP_TOLERANCE = int(os.getenv("RADAR_EPISTULA_TOLERANCE", "30"))


def sign_request(wallet, body: bytes) -> dict[str, str]:
    """Sign an outgoing HTTP request with Epistula headers.

    `wallet` is duck-typed: anything with `.hotkey.ss58_address` and
    `.hotkey.sign(bytes) -> bytes` works (bittensor wallets, raw keypairs).

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
    metagraph=None,
    require_stake: bool = False,
    allowed_hotkeys: Optional[Iterable[str]] = None,
) -> tuple[bool, str, str]:
    """Verify an incoming request's Epistula headers.

    Authorization layers (applied if provided):
      - `metagraph`: caller is registered (and staked if `require_stake`).
      - `allowed_hotkeys`: caller hotkey is in an explicit allowlist.

    If neither is provided, only signature freshness + validity are checked.
    Returns (ok, error_message, sender_hotkey).
    """
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
    if abs(time.time() - ts) > EPISTULA_TIMESTAMP_TOLERANCE:
        return False, "Timestamp too old or too far in future", ""

    if metagraph is not None:
        uid = get_uid_for_hotkey(metagraph, hotkey)
        if uid is None:
            return False, "Unknown hotkey", ""
        if require_stake and float(metagraph.S[uid]) <= 0:
            return False, "Not a validator (no stake)", ""

    if allowed_hotkeys is not None and hotkey not in set(allowed_hotkeys):
        return False, "Hotkey not in allowlist", ""

    message = body + timestamp_str.encode() + nonce.encode()
    try:
        keypair = Keypair(ss58_address=hotkey)
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
