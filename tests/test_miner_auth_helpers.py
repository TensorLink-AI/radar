"""Unit tests for shared.miner_auth without a live Postgres.

The async pool methods are exercised against a tiny in-memory fake so
the helpers' SQL shape, key minting, hashing, and dataclass mapping
are all covered.  Real-DB integration is covered by the
``test_pg_*`` suite when ``TEST_PG_DSN`` is set.
"""

from __future__ import annotations

import re

import pytest

from shared import miner_auth
from shared.auth import hash_api_key


class _FakePool:
    """Records execute() calls, answers fetch*/execute with scripted rows."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.rows: dict[str, list[dict]] = {}
        self.next_execute_result = "UPDATE 1"

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))
        return self.next_execute_result

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args))
        rows = self.rows.get("fetchrow", [])
        return rows.pop(0) if rows else None

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return self.rows.get("fetch", [])


@pytest.mark.asyncio
async def test_register_miner_assigns_id_when_omitted():
    pool = _FakePool()
    mid = await miner_auth.register_miner(pool, name="alice")
    assert isinstance(mid, str) and len(mid) == 32
    assert pool.calls[0][0] == "execute"


@pytest.mark.asyncio
async def test_register_miner_respects_supplied_id():
    pool = _FakePool()
    mid = await miner_auth.register_miner(
        pool, miner_id="custom", name="bob",
    )
    assert mid == "custom"


@pytest.mark.asyncio
async def test_issue_api_key_returns_prefixed_plaintext_and_handle():
    pool = _FakePool()
    plaintext, key_id = await miner_auth.issue_api_key(
        pool, "m1", label="prod",
    )
    assert plaintext.startswith(miner_auth.KEY_PREFIX)
    assert re.match(r"^rdrk_[0-9a-f]{32}$", plaintext)
    assert len(key_id) == 32
    # The INSERT stores the *hash*, never the plaintext.
    _, sql, args = pool.calls[-1]
    assert "INSERT INTO miner_api_keys" in sql
    assert args[1] == hash_api_key(plaintext)
    assert plaintext not in str(args)


@pytest.mark.asyncio
async def test_revoke_api_key_returns_true_when_row_updated():
    pool = _FakePool()
    pool.next_execute_result = "UPDATE 1"
    assert await miner_auth.revoke_api_key(pool, "k1") is True


@pytest.mark.asyncio
async def test_revoke_api_key_returns_false_when_no_row():
    pool = _FakePool()
    pool.next_execute_result = "UPDATE 0"
    assert await miner_auth.revoke_api_key(pool, "missing") is False


@pytest.mark.asyncio
async def test_lookup_bearer_returns_none_for_empty_token():
    pool = _FakePool()
    assert await miner_auth.lookup_bearer(pool, "") is None


@pytest.mark.asyncio
async def test_lookup_bearer_returns_none_when_db_has_no_match():
    pool = _FakePool()
    assert await miner_auth.lookup_bearer(pool, "any-token") is None


@pytest.mark.asyncio
async def test_lookup_bearer_maps_row_to_identity():
    pool = _FakePool()
    pool.rows["fetchrow"] = [{
        "key_id": "k1",
        "miner_id": "m1",
        "scope": "miner",
        "hotkey": "5xxx",
        "name": "alice",
    }]
    ident = await miner_auth.lookup_bearer(pool, "secret-token")
    assert ident is not None
    assert ident.miner_id == "m1"
    assert ident.key_id == "k1"
    assert ident.scope == "miner"
    assert ident.hotkey == "5xxx"
    assert ident.name == "alice"
    # The lookup hashes the token before hitting the DB.
    _, sql, args = pool.calls[-1]
    assert args[0] == hash_api_key("secret-token")


@pytest.mark.asyncio
async def test_lookup_bearer_handles_null_optional_fields():
    pool = _FakePool()
    pool.rows["fetchrow"] = [{
        "key_id": "k", "miner_id": "m", "scope": "miner",
        "hotkey": None, "name": None,
    }]
    ident = await miner_auth.lookup_bearer(pool, "x")
    assert ident.hotkey == ""
    assert ident.name == ""


@pytest.mark.asyncio
async def test_touch_key_usage_swallows_errors():
    class Bad(_FakePool):
        async def execute(self, *args, **kwargs):
            raise RuntimeError("db down")

    await miner_auth.touch_key_usage(Bad(), "k1")  # must not raise


@pytest.mark.asyncio
async def test_list_api_keys_returns_dicts():
    pool = _FakePool()
    pool.rows["fetch"] = [{"key_id": "k1"}, {"key_id": "k2"}]
    rows = await miner_auth.list_api_keys(pool, "m1")
    assert [r["key_id"] for r in rows] == ["k1", "k2"]


def test_hotkey_fallback():
    assert miner_auth.hotkey_to_miner_id_fallback("hk1") == "hotkey:hk1"
    assert miner_auth.hotkey_to_miner_id_fallback("") == ""
