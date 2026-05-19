"""Operator CLI — register miners and issue / revoke their API keys.

Run against the DB Postgres directly:

  python -m database.operator_cli register --name alice [--hotkey 5xxx] [--contact ...]
  python -m database.operator_cli issue-key --miner-id <id> [--label prod]
  python -m database.operator_cli revoke-key --key-id <id>
  python -m database.operator_cli list-keys --miner-id <id>
  python -m database.operator_cli list-miners
  python -m database.operator_cli rotate-service-key

DSN is read from ``RADAR_PG_DSN`` (same as the rest of the stack).

Plaintext API keys are printed exactly once at issuance — copy them
out of the terminal immediately; the DB only stores the SHA-256.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from typing import Optional

from shared import miner_auth


# ── DB connection ──────────────────────────────────────────────────


async def _connect():
    import asyncpg
    dsn = os.getenv("RADAR_PG_DSN", "").strip()
    if not dsn:
        print(
            "operator: RADAR_PG_DSN not set.  Example: "
            "postgresql://radar:radar@localhost:5432/radar",
            file=sys.stderr,
        )
        sys.exit(2)
    return await asyncpg.connect(dsn)


async def _ensure_schema(conn) -> None:
    """Create the miners + miner_api_keys tables if they don't exist
    yet.  Mirrors PgExperimentStore.init_schema so the operator CLI
    works against a stock Postgres on day one."""
    from shared.pg_schema import MINER_REGISTRY_SCHEMA

    # The DDL references experiments.prompt_id; if the table doesn't
    # exist yet we still want miners/miner_api_keys created.  Run each
    # statement separately and tolerate the ALTER failing.
    for stmt in [s.strip() for s in MINER_REGISTRY_SCHEMA.split(";") if s.strip()]:
        try:
            await conn.execute(stmt + ";")
        except Exception as e:
            # ALTER on missing experiments table is fine to ignore;
            # other failures surface so the operator notices.
            msg = str(e).lower()
            if "experiments" in msg and "does not exist" in msg:
                continue
            print(f"warn: DDL stmt failed: {e}", file=sys.stderr)


# ── Command handlers ───────────────────────────────────────────────


async def cmd_register(args) -> int:
    conn = await _connect()
    try:
        await _ensure_schema(conn)
        mid = await miner_auth.register_miner(
            conn,
            miner_id=args.miner_id,
            name=args.name,
            hotkey=args.hotkey,
            contact=args.contact,
        )
        print(f"miner_id: {mid}")
        return 0
    finally:
        await conn.close()


async def cmd_issue_key(args) -> int:
    conn = await _connect()
    try:
        await _ensure_schema(conn)
        plaintext, key_id = await miner_auth.issue_api_key(
            conn, args.miner_id, scope=args.scope, label=args.label,
        )
        print("=" * 60)
        print("API KEY (shown ONCE — copy now, will not be stored):")
        print(plaintext)
        print()
        print(f"key_id   {key_id}")
        print(f"miner_id {args.miner_id}")
        print(f"scope    {args.scope}")
        print(f"label    {args.label or '(none)'}")
        print("=" * 60)
        return 0
    finally:
        await conn.close()


async def cmd_revoke_key(args) -> int:
    conn = await _connect()
    try:
        ok = await miner_auth.revoke_api_key(conn, args.key_id)
        if ok:
            print(f"revoked {args.key_id}")
            return 0
        print(f"no active key with id {args.key_id}", file=sys.stderr)
        return 1
    finally:
        await conn.close()


async def cmd_list_keys(args) -> int:
    conn = await _connect()
    try:
        rows = await miner_auth.list_api_keys(conn, args.miner_id)
        if not rows:
            print("(no keys)")
            return 0
        print(f"{'key_id':<34}  {'scope':<8}  {'label':<18}  "
              f"{'created':<22}  {'revoked':<22}  last_used")
        for r in rows:
            print(
                f"{r['key_id']:<34}  {r['scope']:<8}  "
                f"{(r.get('label') or '')[:18]:<18}  "
                f"{_fmt_ts(r.get('created_at')):<22}  "
                f"{_fmt_ts(r.get('revoked_at')):<22}  "
                f"{_fmt_ts(r.get('last_used_at'))}"
            )
        return 0
    finally:
        await conn.close()


async def cmd_list_miners(args) -> int:
    conn = await _connect()
    try:
        rows = await conn.fetch(
            "SELECT miner_id, name, hotkey, contact, created_at, revoked_at "
            "FROM miners ORDER BY created_at DESC LIMIT $1",
            int(args.limit),
        )
        if not rows:
            print("(no miners)")
            return 0
        print(f"{'miner_id':<34}  {'name':<20}  {'hotkey':<20}  "
              f"{'created':<22}  revoked")
        for r in rows:
            print(
                f"{r['miner_id']:<34}  "
                f"{(r['name'] or '')[:20]:<20}  "
                f"{(r['hotkey'] or '')[:20]:<20}  "
                f"{_fmt_ts(r.get('created_at')):<22}  "
                f"{_fmt_ts(r.get('revoked_at'))}"
            )
        return 0
    finally:
        await conn.close()


def cmd_rotate_service_key(args) -> int:
    """Print a fresh service key — operator pastes into env."""
    key = secrets.token_hex(32)
    print("New RADAR_SERVICE_KEY value (set on operator + trainers + validators):")
    print(key)
    print()
    print("Rotation procedure:")
    print("  1. Roll out to every trainer first (they're the verifiers).")
    print("  2. Then update validators / operator dispatchers.")
    print("  3. Confirm /train accepts the new key before retiring the old one.")
    return 0


# ── argparse plumbing ──────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="radar-operator",
        description="Manage miners + API keys for non-competitive Radar.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("register", help="Register a new miner.")
    r.add_argument("--name", default="")
    r.add_argument("--hotkey", default="",
                   help="Bittensor SS58 (optional; for dual-stack period).")
    r.add_argument("--contact", default="")
    r.add_argument("--miner-id", default=None,
                   help="Explicit id (default: random UUID).")

    i = sub.add_parser("issue-key", help="Mint an API key for a miner.")
    i.add_argument("--miner-id", required=True)
    i.add_argument("--scope", default="miner",
                   help="Scope tag stored on the key (default: 'miner').")
    i.add_argument("--label", default="",
                   help="Operator note shown in list-keys output.")

    rk = sub.add_parser("revoke-key", help="Revoke an API key by key_id.")
    rk.add_argument("--key-id", required=True)

    lk = sub.add_parser("list-keys", help="List keys for a miner.")
    lk.add_argument("--miner-id", required=True)

    lm = sub.add_parser("list-miners", help="List registered miners.")
    lm.add_argument("--limit", type=int, default=100)

    sub.add_parser("rotate-service-key",
                   help="Print a fresh RADAR_SERVICE_KEY.")
    return p


def _fmt_ts(value) -> str:
    if value in (None, 0, 0.0):
        return "-"
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(
            timespec="seconds",
        )
    except (TypeError, ValueError):
        return str(value)


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "rotate-service-key":
        return cmd_rotate_service_key(args)
    handler = {
        "register": cmd_register,
        "issue-key": cmd_issue_key,
        "revoke-key": cmd_revoke_key,
        "list-keys": cmd_list_keys,
        "list-miners": cmd_list_miners,
    }[args.command]
    return asyncio.run(handler(args))


if __name__ == "__main__":
    sys.exit(main())
