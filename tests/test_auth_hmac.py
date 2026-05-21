"""Tests for shared.auth — HMAC shared-secret signing."""

import os

import pytest

from shared import auth


def test_sign_request_hmac_deterministic(monkeypatch):
    monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
    sig1 = auth.sign_request_hmac(b"hello world")
    sig2 = auth.sign_request_hmac(b"hello world")
    assert sig1 and sig1 == sig2
    # Length of hex SHA-256 digest.
    assert len(sig1) == 64


def test_sign_request_hmac_differs_by_body(monkeypatch):
    monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
    assert auth.sign_request_hmac(b"a") != auth.sign_request_hmac(b"b")


def test_sign_request_hmac_no_secret_returns_empty(monkeypatch):
    monkeypatch.delenv("RADAR_SHARED_SECRET", raising=False)
    assert auth.sign_request_hmac(b"anything") == ""


def test_verify_request_hmac_roundtrip(monkeypatch):
    monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
    body = b"payload"
    sig = auth.sign_request_hmac(body)
    assert auth.verify_request_hmac(body, sig) is True
    # Tampered body fails.
    assert auth.verify_request_hmac(b"payload-tampered", sig) is False
    # Tampered signature fails.
    assert auth.verify_request_hmac(body, sig[:-1] + ("0" if sig[-1] != "0" else "1")) is False


def test_verify_request_hmac_no_secret_fails_closed(monkeypatch):
    monkeypatch.delenv("RADAR_SHARED_SECRET", raising=False)
    assert auth.verify_request_hmac(b"x", "deadbeef") is False


def test_sign_request_returns_headers(monkeypatch):
    monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
    headers = auth.sign_request(None, b"body")
    assert headers["X-Radar-Signature"] == auth.sign_request_hmac(b"body")
    # Legacy header still populated for backwards-compat.
    assert "X-Epistula-Timestamp" in headers


def test_verify_request_dev_mode_allows_when_secret_missing(monkeypatch):
    monkeypatch.delenv("RADAR_SHARED_SECRET", raising=False)
    ok, signer, err = auth.verify_request({}, b"")
    assert ok is True
    assert signer == "dev-mode"
    assert err == ""


def test_verify_request_requires_signature_when_secret_set(monkeypatch):
    monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
    ok, signer, err = auth.verify_request({}, b"body")
    assert ok is False
    assert "signature" in err.lower()


def test_verify_request_accepts_valid_hmac(monkeypatch):
    monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
    body = b"hello"
    sig = auth.sign_request_hmac(body)
    headers = {"X-Radar-Signature": sig}
    ok, _, err = auth.verify_request(headers, body)
    assert ok is True
    assert err == ""


def test_verify_request_legacy_header_name(monkeypatch):
    monkeypatch.setenv("RADAR_SHARED_SECRET", "test-secret")
    body = b"hi"
    sig = auth.sign_request_hmac(body)
    ok, _, _ = auth.verify_request({"x-epistula-signature": sig}, body)
    assert ok is True


def test_get_uid_for_hotkey_uses_peers(monkeypatch, tmp_path):
    import json
    path = tmp_path / "miners.json"
    path.write_text(json.dumps({
        "miners": [{"uid": 5, "hotkey": "hk5"}, {"uid": 6, "hotkey": "hk6"}],
    }))
    monkeypatch.setenv("MINERS_CONFIG_PATH", str(path))
    from shared import peers
    peers.reset_cache()
    assert auth.get_uid_for_hotkey("hk6") == 6
    assert auth.get_uid_for_hotkey("missing") is None
