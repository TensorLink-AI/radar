"""Tests for the HMAC + bearer-token primitives in shared/auth.py."""

from __future__ import annotations

import time

import pytest

from shared import auth


SECRET = b"super-secret-shared-key-do-not-leak"
OTHER = b"another-secret"


def _headers_as_wire(d: dict[str, str]) -> dict[str, str]:
    """Simulate the casing servers usually present headers in."""
    return {k.lower(): v for k, v in d.items()}


def test_sign_verify_roundtrip():
    body = b'{"hello": "world"}'
    headers = auth.hmac_sign_request(SECRET, body, key_id="op")
    ok, err, kid = auth.hmac_verify_request(
        _headers_as_wire(headers), body, auth.static_key_lookup("op", SECRET)
    )
    assert ok, err
    assert kid == "op"


def test_verify_rejects_tampered_body():
    body = b'{"a": 1}'
    headers = auth.hmac_sign_request(SECRET, body, key_id="op")
    tampered = b'{"a": 2}'
    ok, err, _ = auth.hmac_verify_request(
        _headers_as_wire(headers), tampered, auth.static_key_lookup("op", SECRET)
    )
    assert not ok
    assert "signature" in err


def test_verify_rejects_wrong_key():
    body = b"x"
    headers = auth.hmac_sign_request(SECRET, body, key_id="op")
    ok, err, _ = auth.hmac_verify_request(
        _headers_as_wire(headers), body, auth.static_key_lookup("op", OTHER)
    )
    assert not ok
    assert "signature" in err


def test_verify_rejects_unknown_key_id():
    body = b"x"
    headers = auth.hmac_sign_request(SECRET, body, key_id="ghost")
    ok, err, _ = auth.hmac_verify_request(
        _headers_as_wire(headers), body, auth.static_key_lookup("op", SECRET)
    )
    assert not ok
    assert "unknown key_id" in err


def test_verify_rejects_stale_timestamp():
    body = b"x"
    # Sign with a timestamp far outside tolerance.
    old_ts = str(int(time.time()) - 10_000)
    sig = auth.hmac_sign_request(SECRET, body, key_id="op")
    sig["X-Radar-Timestamp"] = old_ts
    # Recompute the signature against the old ts so we exercise the
    # timestamp-window check rather than the signature check.
    import hashlib
    import hmac as _hmac

    digest = hashlib.sha256(body).hexdigest()
    sig["X-Radar-Signature"] = _hmac.new(
        SECRET, f"{old_ts}.{digest}".encode(), hashlib.sha256
    ).hexdigest()

    ok, err, _ = auth.hmac_verify_request(
        _headers_as_wire(sig), body, auth.static_key_lookup("op", SECRET)
    )
    assert not ok
    assert "timestamp" in err


def test_verify_requires_all_headers():
    body = b"x"
    headers = auth.hmac_sign_request(SECRET, body, key_id="op")
    for missing in ("X-Radar-Signature", "X-Radar-Timestamp", "X-Radar-Key-Id"):
        partial = {k: v for k, v in headers.items() if k != missing}
        ok, err, _ = auth.hmac_verify_request(
            _headers_as_wire(partial), body, auth.static_key_lookup("op", SECRET)
        )
        assert not ok, f"missing {missing} should fail"
        assert "missing" in err.lower()


def test_hmac_sign_request_rejects_string_secret():
    with pytest.raises(TypeError):
        auth.hmac_sign_request("not-bytes", b"x")  # type: ignore[arg-type]


def test_extract_bearer():
    assert auth.extract_bearer({"authorization": "Bearer abc123"}) == "abc123"
    assert auth.extract_bearer({"Authorization": "bearer abc123"}) == "abc123"
    assert auth.extract_bearer({"authorization": "Basic xx"}) is None
    assert auth.extract_bearer({"authorization": "Bearer  "}) is None
    assert auth.extract_bearer({}) is None
    assert auth.extract_bearer({"authorization": "Bearer"}) is None


def test_hash_api_key_stable():
    h1 = auth.hash_api_key("hunter2")
    h2 = auth.hash_api_key("hunter2")
    assert h1 == h2
    assert h1 != auth.hash_api_key("hunter3")
    assert len(h1) == 64  # sha256 hex


def test_load_service_secret_unset(monkeypatch):
    monkeypatch.delenv("RADAR_SERVICE_KEY", raising=False)
    with pytest.raises(RuntimeError):
        auth.load_service_secret()


def test_load_service_secret_set(monkeypatch):
    monkeypatch.setenv("RADAR_SERVICE_KEY", "abc")
    assert auth.load_service_secret() == b"abc"
