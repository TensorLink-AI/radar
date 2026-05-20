"""Stub authentication module.

The chain-based Epistula signature / hotkey verification has been removed.
The functions below preserve their signatures so existing callers keep
importing successfully, but they no longer perform any real verification.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# Kept for backward compatibility with code that imports the constant.
EPISTULA_TIMESTAMP_TOLERANCE = int(os.getenv("RADAR_EPISTULA_TOLERANCE", "30"))


def sign_request(wallet, body: bytes) -> dict[str, str]:
    """No-op stub. Returns a minimal set of headers without a real signature."""
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    return {
        "X-Epistula-Signed-By": "",
        "X-Epistula-Timestamp": timestamp,
        "X-Epistula-Nonce": nonce,
        "X-Epistula-Signature": "",
    }


def verify_request(
    headers: dict[str, str],
    body: bytes,
    *args,
    **kwargs,
) -> tuple[bool, str, str]:
    """No-op stub — auth has been removed. Returns (True, "", "")."""
    return True, "", ""


def get_uid_for_hotkey(*args, **kwargs) -> Optional[int]:
    """No-op stub — returns None."""
    return None
