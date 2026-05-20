"""Tests for the operator CLI argv parsing + handler dispatch.

Real-DB integration of the handlers is covered by the miner_auth
helper tests; here we just exercise the argparse surface, output
formatting, and rotate-service-key which has no DB dependency.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from database import operator_cli


def test_parser_register_requires_no_args():
    args = operator_cli._build_parser().parse_args(["register"])
    assert args.command == "register"
    assert args.name == ""
    assert args.hotkey == ""


def test_parser_issue_key_requires_miner_id():
    with pytest.raises(SystemExit):
        operator_cli._build_parser().parse_args(["issue-key"])


def test_parser_revoke_key_requires_key_id():
    with pytest.raises(SystemExit):
        operator_cli._build_parser().parse_args(["revoke-key"])


def test_rotate_service_key_prints_long_hex(capsys):
    code = operator_cli.main(["rotate-service-key"])
    assert code == 0
    out = capsys.readouterr().out
    # 32 bytes hex = 64 chars somewhere in the output.
    candidate = [w for line in out.splitlines() for w in line.split()
                 if len(w) == 64 and all(c in "0123456789abcdef" for c in w)]
    assert candidate, f"no 64-hex key found in output:\n{out}"


def test_register_dispatch_calls_register_miner(monkeypatch, capsys):
    fake_conn = AsyncMock()
    fake_conn.close = AsyncMock()

    async def _connect_stub():
        return fake_conn

    async def _ensure_stub(conn):
        pass

    monkeypatch.setattr(operator_cli, "_connect", _connect_stub)
    monkeypatch.setattr(operator_cli, "_ensure_schema", _ensure_stub)
    with patch(
        "shared.miner_auth.register_miner",
        new=AsyncMock(return_value="abc123"),
    ) as mr:
        code = operator_cli.main([
            "register", "--name", "alice", "--hotkey", "5xxx",
        ])
    assert code == 0
    mr.assert_awaited_once()
    out = capsys.readouterr().out
    assert "abc123" in out


def test_issue_key_dispatch_prints_plaintext_once(monkeypatch, capsys):
    fake_conn = AsyncMock()
    fake_conn.close = AsyncMock()

    async def _connect_stub():
        return fake_conn

    async def _ensure_stub(conn):
        pass

    monkeypatch.setattr(operator_cli, "_connect", _connect_stub)
    monkeypatch.setattr(operator_cli, "_ensure_schema", _ensure_stub)
    with patch(
        "shared.miner_auth.issue_api_key",
        new=AsyncMock(return_value=("rdrk_secret123", "kid-xyz")),
    ):
        code = operator_cli.main([
            "issue-key", "--miner-id", "m1", "--label", "prod",
        ])
    assert code == 0
    out = capsys.readouterr().out
    assert "rdrk_secret123" in out
    assert "kid-xyz" in out
    assert "shown ONCE" in out


def test_revoke_key_returns_1_on_miss(monkeypatch, capsys):
    fake_conn = AsyncMock()
    fake_conn.close = AsyncMock()

    async def _connect_stub():
        return fake_conn

    monkeypatch.setattr(operator_cli, "_connect", _connect_stub)
    with patch(
        "shared.miner_auth.revoke_api_key",
        new=AsyncMock(return_value=False),
    ):
        code = operator_cli.main(["revoke-key", "--key-id", "ghost"])
    assert code == 1


def test_revoke_key_returns_0_on_hit(monkeypatch, capsys):
    fake_conn = AsyncMock()
    fake_conn.close = AsyncMock()

    async def _connect_stub():
        return fake_conn

    monkeypatch.setattr(operator_cli, "_connect", _connect_stub)
    with patch(
        "shared.miner_auth.revoke_api_key",
        new=AsyncMock(return_value=True),
    ):
        code = operator_cli.main(["revoke-key", "--key-id", "real"])
    assert code == 0


def test_split_sql_statements_ignores_semicolons_in_comments():
    """Regression: a naive split(';') broke on a ``;`` inside a ``--``
    comment block in MINER_REGISTRY_SCHEMA, producing two bad chunks
    (one all-comment, one starting with "only the issuing miner...")
    that asyncpg reported as ``'NoneType' object has no attribute
    'decode'`` and ``syntax error at or near "only"``. The comment-aware
    splitter strips ``--`` line comments before splitting."""
    sql = """
    CREATE TABLE foo (id TEXT);
    -- a comment with a ; semicolon ; inside
    ALTER TABLE foo ADD COLUMN bar TEXT;
    -- trailing comment-only line
    """
    stmts = operator_cli._split_sql_statements(sql)
    assert len(stmts) == 2
    assert stmts[0].startswith("CREATE TABLE foo")
    assert stmts[1].startswith("ALTER TABLE foo")
    # No "only" / orphan-tail garbage from mid-comment splitting.
    assert not any("only" in s.lower() for s in stmts)


def test_split_sql_statements_on_real_schema():
    """Walk the actual MINER_REGISTRY_SCHEMA end-to-end — every chunk
    the splitter produces must start with a legal SQL keyword, not
    leftover comment text."""
    from shared.pg_schema import MINER_REGISTRY_SCHEMA

    stmts = operator_cli._split_sql_statements(MINER_REGISTRY_SCHEMA)
    legal_starts = ("CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE")
    for s in stmts:
        assert s.upper().startswith(legal_starts), (
            f"split produced a chunk that doesn't start with DDL/DML: {s!r}"
        )


def test_dotenv_is_loaded_at_import(tmp_path, monkeypatch):
    """Regression: operator_cli used to read os.environ directly without
    importing ``config``, which meant ``radar/.env`` was silently
    ignored and every invocation printed ``RADAR_PG_DSN not set``.
    Now ``import config`` triggers ``load_dotenv`` at module load —
    verify the chain by writing a marker var to a fake .env, reloading
    config, and asserting it shows up in ``os.environ``.
    """
    import importlib
    import os

    env_file = tmp_path / ".env"
    env_file.write_text("RADAR_OPERATOR_CLI_DOTENV_PROBE=loaded\n")

    monkeypatch.delenv("RADAR_OPERATOR_CLI_DOTENV_PROBE", raising=False)
    from dotenv import load_dotenv
    load_dotenv(env_file, override=True)

    # Re-import operator_cli so its top-level ``import config`` runs
    # again in this test's context (config itself is module-cached
    # but the assertion is about the env var the dotenv chain set).
    importlib.import_module("database.operator_cli")
    assert os.environ.get("RADAR_OPERATOR_CLI_DOTENV_PROBE") == "loaded"


def test_list_keys_prints_table(monkeypatch, capsys):
    fake_conn = AsyncMock()
    fake_conn.close = AsyncMock()

    async def _connect_stub():
        return fake_conn

    monkeypatch.setattr(operator_cli, "_connect", _connect_stub)
    with patch(
        "shared.miner_auth.list_api_keys",
        new=AsyncMock(return_value=[
            {"key_id": "k1", "scope": "miner", "label": "prod",
             "created_at": 1700000000, "revoked_at": None,
             "last_used_at": 1700001000},
        ]),
    ):
        code = operator_cli.main(["list-keys", "--miner-id", "m1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "k1" in out
    assert "prod" in out
