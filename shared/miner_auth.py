"""Bearer-token miner auth — registry helpers backed by Postgres.

The non-competitive Radar feedback API (``/miners/me/*`` and a couple
of CLI helpers) authenticates miners with operator-issued bearer
tokens stored hashed in ``miner_api_keys``.  This module is the choke
point for issuing, looking up, and revoking those keys; the FastAPI
middleware delegates here so the SQL stays in one place.

The hash is SHA-256 of the raw key (see ``shared.auth.hash_api_key``)
so a database leak doesn't recover the plaintext bearer.

Key format on the wire: ``rdrk_<32-hex-chars>`` — the prefix lets
operators grep keys out of logs, but isn't required by the verifier.
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from shared.auth import hash_api_key

logger = logging.getLogger(__name__)

KEY_PREFIX = "rdrk_"


@dataclass(slots=True)
class MinerIdentity:
    """Resolved bearer-auth identity returned to the request handler."""

    miner_id: str
    key_id: str
    scope: str
    hotkey: str = ""
    name: str = ""


def _new_key_pair() -> tuple[str, str]:
    """Return ``(plaintext_key, key_id)``.  ``key_id`` is the
    user-facing identifier (not a secret) so operators can list +
    revoke keys without ever holding the plaintext again."""
    token = secrets.token_hex(16)
    return f"{KEY_PREFIX}{token}", uuid.uuid4().hex


async def register_miner(
    pool,
    *,
    miner_id: Optional[str] = None,
    name: str = "",
    hotkey: str = "",
    contact: str = "",
) -> str:
    """Insert a row in ``miners`` and return the (possibly generated) id."""
    mid = miner_id or uuid.uuid4().hex
    await pool.execute(
        "INSERT INTO miners (miner_id, name, hotkey, contact) "
        "VALUES ($1, $2, $3, $4) "
        "ON CONFLICT (miner_id) DO UPDATE SET "
        "  name = EXCLUDED.name, hotkey = EXCLUDED.hotkey, "
        "  contact = EXCLUDED.contact",
        mid, name, hotkey, contact,
    )
    return mid


async def issue_api_key(
    pool,
    miner_id: str,
    *,
    scope: str = "miner",
    label: str = "",
) -> tuple[str, str]:
    """Mint a new API key.  Returns ``(plaintext, key_id)``.

    The plaintext is shown to the operator ONCE; the DB only stores
    the hash.  ``key_id`` is the stable handle for revocation.
    """
    plaintext, key_id = _new_key_pair()
    await pool.execute(
        "INSERT INTO miner_api_keys (key_id, key_hash, miner_id, scope, label) "
        "VALUES ($1, $2, $3, $4, $5)",
        key_id, hash_api_key(plaintext), miner_id, scope, label,
    )
    return plaintext, key_id


async def revoke_api_key(pool, key_id: str) -> bool:
    """Mark a key revoked.  Returns True iff a row was updated."""
    result = await pool.execute(
        "UPDATE miner_api_keys SET revoked_at = $1 "
        "WHERE key_id = $2 AND revoked_at IS NULL",
        time.time(), key_id,
    )
    # asyncpg returns 'UPDATE n' — split off the count.
    try:
        return int(result.split()[-1]) > 0
    except (ValueError, IndexError):
        return False


async def lookup_bearer(pool, token: str) -> Optional[MinerIdentity]:
    """Resolve a bearer token to its miner identity, or ``None`` if
    the token is unknown / revoked."""
    if not token:
        return None
    digest = hash_api_key(token)
    row = await pool.fetchrow(
        "SELECT k.key_id, k.miner_id, k.scope, m.hotkey, m.name "
        "FROM miner_api_keys k "
        "JOIN miners m ON m.miner_id = k.miner_id "
        "WHERE k.key_hash = $1 AND k.revoked_at IS NULL "
        "  AND m.revoked_at IS NULL "
        "LIMIT 1",
        digest,
    )
    if row is None:
        return None
    return MinerIdentity(
        miner_id=row["miner_id"],
        key_id=row["key_id"],
        scope=row["scope"],
        hotkey=row["hotkey"] or "",
        name=row["name"] or "",
    )


async def touch_key_usage(pool, key_id: str) -> None:
    """Update ``last_used_at`` (best-effort — swallow errors)."""
    try:
        await pool.execute(
            "UPDATE miner_api_keys SET last_used_at = $1 WHERE key_id = $2",
            time.time(), key_id,
        )
    except Exception as e:
        logger.warning("touch_key_usage failed for %s: %s", key_id, e)


async def list_api_keys(pool, miner_id: str) -> list[dict]:
    """Return non-secret metadata for every key of a miner."""
    rows = await pool.fetch(
        "SELECT key_id, scope, label, created_at, last_used_at, revoked_at "
        "FROM miner_api_keys WHERE miner_id = $1 ORDER BY created_at DESC",
        miner_id,
    )
    return [dict(r) for r in rows]


def hotkey_to_miner_id_fallback(hotkey: str) -> str:
    """Deterministic placeholder ``miner_id`` for a hotkey when the
    operator hasn't issued a real one yet — lets the dual-stack period
    write ``experiments.prompt_id`` against a stable key derived from
    the bittensor identity."""
    return f"hotkey:{hotkey}" if hotkey else ""
