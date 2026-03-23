"""Tests for shared.auth — Epistula signing and verification.

(Renamed from test_gossip.py — auth logic extracted to shared/auth.py)
"""

import time

import pytest

from shared.auth import sign_request, verify_request, get_uid_for_hotkey


# ── Mock Bittensor objects ────────────────────────────────────────────────


class MockMetagraph:
    """Minimal metagraph mock for auth tests."""

    def __init__(self, n: int = 4, hotkeys=None, stakes=None):
        self.n = n
        self.hotkeys = hotkeys or [f"hk_{i}" for i in range(n)]
        self.S = stakes or [100.0] * n


class MockWallet:
    class HK:
        ss58_address = "hk_0"

        def sign(self, message: bytes) -> bytes:
            import hashlib
            return hashlib.sha256(b"fake_key" + message).digest()

    hotkey = HK()


# ── Epistula signing tests ────────────────────────────────────────────────


class TestEpistulaSigning:
    def test_sign_request_returns_headers(self):
        wallet = MockWallet()
        body = b'{"test": true}'
        headers = sign_request(wallet, body)
        assert "X-Epistula-Signed-By" in headers
        assert "X-Epistula-Timestamp" in headers
        assert "X-Epistula-Nonce" in headers
        assert "X-Epistula-Signature" in headers
        assert headers["X-Epistula-Signed-By"] == "hk_0"

    def test_sign_request_unique_nonces(self):
        wallet = MockWallet()
        body = b'{"test": true}'
        h1 = sign_request(wallet, body)
        h2 = sign_request(wallet, body)
        assert h1["X-Epistula-Nonce"] != h2["X-Epistula-Nonce"]


class TestEpistulaVerification:
    def test_missing_headers_rejected(self):
        metagraph = MockMetagraph()
        ok, err, hotkey = verify_request({}, b"body", metagraph)
        assert not ok
        assert "Missing" in err

    def test_unknown_hotkey_rejected(self):
        metagraph = MockMetagraph()
        headers = {
            "x-epistula-signed-by": "unknown_hk",
            "x-epistula-timestamp": str(int(time.time())),
            "x-epistula-nonce": "abc123",
            "x-epistula-signature": "deadbeef",
        }
        ok, err, hotkey = verify_request(headers, b"body", metagraph)
        assert not ok
        assert "Unknown hotkey" in err

    def test_stale_timestamp_rejected(self):
        metagraph = MockMetagraph()
        headers = {
            "x-epistula-signed-by": "hk_0",
            "x-epistula-timestamp": str(int(time.time()) - 9999),
            "x-epistula-nonce": "abc123",
            "x-epistula-signature": "deadbeef",
        }
        ok, err, hotkey = verify_request(headers, b"body", metagraph)
        assert not ok
        assert "Timestamp" in err

    def test_require_stake_rejects_zero_stake(self):
        metagraph = MockMetagraph(
            n=2,
            hotkeys=["hk_0", "hk_1"],
            stakes=[0.0, 100.0],
        )
        headers = {
            "x-epistula-signed-by": "hk_0",
            "x-epistula-timestamp": str(int(time.time())),
            "x-epistula-nonce": "abc123",
            "x-epistula-signature": "deadbeef",
        }
        ok, err, hotkey = verify_request(headers, b"body", metagraph, require_stake=True)
        assert not ok
        assert "no stake" in err


# ── get_uid_for_hotkey tests ─────────────────────────────────────────────


class TestGetUidForHotkey:
    def test_found(self):
        metagraph = MockMetagraph(n=3, hotkeys=["a", "b", "c"])
        assert get_uid_for_hotkey(metagraph, "b") == 1

    def test_not_found(self):
        metagraph = MockMetagraph(n=3, hotkeys=["a", "b", "c"])
        assert get_uid_for_hotkey(metagraph, "d") is None
