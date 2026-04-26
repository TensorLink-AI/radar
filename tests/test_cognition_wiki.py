"""Tests for the cognition-wiki signed-URL manifest helper.

Covers:
  - R2 key construction + task-name validation
  - presign_wiki_url short-circuits on missing R2 / missing object
  - presign_wiki_url returns a signed URL when the tarball exists
  - Challenge.cognition_wiki_url survives JSON round-trip
  - validator.collection._build_allowed_urls picks up the wiki URL
"""

from __future__ import annotations

import json

import pytest

from shared import cognition_wiki
from shared.cognition_wiki import (
    build_wiki_r2,
    presign_wiki_url,
    wiki_key,
)
from shared.protocol import Challenge


class _StubR2:
    """Minimal R2 stub with controllable key_exists + presign behaviour."""

    bucket = "cognition-wiki-test"

    _UNSET = object()

    def __init__(self, exists: bool = True, presigned=_UNSET):
        self._exists = exists
        self._presigned = (
            "https://r2.example.com/signed?sig=abc"
            if presigned is _StubR2._UNSET
            else presigned
        )
        self.presign_calls: list[tuple[str, int]] = []
        self.head_calls: list[str] = []

    def key_exists(self, key: str) -> bool:
        self.head_calls.append(key)
        return self._exists

    def generate_presigned_get_url(self, key: str, ttl: int = 900) -> str:
        self.presign_calls.append((key, ttl))
        return self._presigned


# ── wiki_key ────────────────────────────────────────────────────────


def test_wiki_key_basic():
    assert wiki_key("nanogpt", "cognition_wiki/v1") == \
        "cognition_wiki/v1/nanogpt/wiki.tar.gz"


def test_wiki_key_strips_trailing_slash():
    assert wiki_key("nanogpt", "cognition_wiki/v1/") == \
        "cognition_wiki/v1/nanogpt/wiki.tar.gz"


def test_wiki_key_empty_prefix():
    assert wiki_key("nanogpt", "") == "nanogpt/wiki.tar.gz"


@pytest.mark.parametrize("bad", [
    "",
    "../etc/passwd",
    "task with spaces",
    "task/with/slash",
    "task;rm",
])
def test_wiki_key_rejects_unsafe_task(bad):
    with pytest.raises(ValueError):
        wiki_key(bad, "cognition_wiki/v1")


# ── build_wiki_r2 ───────────────────────────────────────────────────


def test_build_wiki_r2_returns_none_when_disabled():
    assert build_wiki_r2(bucket="") is None


# ── presign_wiki_url ────────────────────────────────────────────────


def test_presign_wiki_url_no_r2_returns_empty():
    assert presign_wiki_url(None, "nanogpt", prefix="x") == ""


def test_presign_wiki_url_unsafe_task_returns_empty(caplog):
    r2 = _StubR2()
    out = presign_wiki_url(r2, "../oops", prefix="cognition_wiki/v1")
    assert out == ""
    assert r2.presign_calls == []  # never reached signing


def test_presign_wiki_url_skips_when_object_missing():
    r2 = _StubR2(exists=False)
    out = presign_wiki_url(r2, "nanogpt", prefix="cognition_wiki/v1", ttl=300)
    assert out == ""
    assert r2.head_calls == ["cognition_wiki/v1/nanogpt/wiki.tar.gz"]
    assert r2.presign_calls == []  # don't sign nonexistent keys


def test_presign_wiki_url_returns_signed_url_when_present():
    r2 = _StubR2(exists=True, presigned="https://r2.example.com/wiki.tar.gz?sig=zzz")
    out = presign_wiki_url(r2, "nanogpt", prefix="cognition_wiki/v1", ttl=600)
    assert out == "https://r2.example.com/wiki.tar.gz?sig=zzz"
    assert r2.presign_calls == [("cognition_wiki/v1/nanogpt/wiki.tar.gz", 600)]


def test_presign_wiki_url_propagates_signing_failure():
    """If R2 returns an empty URL we surface that as no-wiki rather than crash."""
    r2 = _StubR2(exists=True, presigned="")
    out = presign_wiki_url(r2, "nanogpt", prefix="cognition_wiki/v1")
    assert out == ""


# ── Challenge wire format ───────────────────────────────────────────


def test_challenge_round_trips_cognition_wiki_url():
    c = Challenge(
        challenge_id="round_42",
        round_id=42,
        cognition_wiki_url="https://r2.example.com/nanogpt/wiki.tar.gz?sig=xyz",
    )
    blob = c.to_json()
    parsed = json.loads(blob)
    assert parsed["cognition_wiki_url"] == \
        "https://r2.example.com/nanogpt/wiki.tar.gz?sig=xyz"
    restored = Challenge.from_json(blob)
    assert restored.cognition_wiki_url == c.cognition_wiki_url


def test_challenge_default_cognition_wiki_url_is_empty():
    c = Challenge()
    assert c.cognition_wiki_url == ""
    parsed = json.loads(c.to_json())
    assert parsed["cognition_wiki_url"] == ""


# ── validator/collection._build_allowed_urls ────────────────────────


def test_build_allowed_urls_includes_cognition_wiki_url():
    from validator.collection import _build_allowed_urls

    challenge = Challenge(
        db_url="https://proxy.example.com",
        cognition_wiki_url=(
            "https://r2.example.com/cognition_wiki/v1/nanogpt/wiki.tar.gz?sig=zzz"
        ),
    )
    allowed = _build_allowed_urls(challenge.to_json())
    parts = allowed.split(",") if allowed else []
    assert any(
        p.startswith(
            "https://r2.example.com/cognition_wiki/v1/nanogpt/wiki.tar.gz"
        )
        for p in parts
    ), allowed


def test_build_allowed_urls_skips_empty_cognition_wiki_url():
    from validator.collection import _build_allowed_urls

    challenge = Challenge(db_url="https://proxy.example.com")
    allowed = _build_allowed_urls(challenge.to_json())
    # Empty URL should not introduce an empty allowlist entry.
    assert "" not in allowed.split(",")


def test_build_allowed_urls_dedups_when_wiki_already_covered():
    """If the wiki URL is already covered by an existing allowed prefix,
    don't duplicate it."""
    from validator.collection import _build_allowed_urls

    wiki_url = "https://shared.example.com/wiki.tar.gz?sig=zzz"
    challenge = Challenge(
        db_url="https://shared.example.com",  # base prefix already in allowlist
        cognition_wiki_url=wiki_url,
    )
    allowed = _build_allowed_urls(challenge.to_json())
    parts = [p for p in allowed.split(",") if p]
    assert wiki_url not in parts


# ── public-shape sanity check ───────────────────────────────────────


def test_module_exports():
    """Loosely guard the public surface so callers don't break silently."""
    assert callable(cognition_wiki.wiki_key)
    assert callable(cognition_wiki.build_wiki_r2)
    assert callable(cognition_wiki.presign_wiki_url)
