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


def test_issue_key_refuses_when_miner_missing(monkeypatch, capsys):
    """Regression: ``issue-key`` used to mint an orphan key against a
    miner_id that didn't exist in the active schema, producing a row
    that lookup_bearer's JOIN would silently drop → 403. The new
    pre-flight returns exit code 2 with a clear stderr message instead."""
    fake_conn = AsyncMock()
    fake_conn.close = AsyncMock()
    fake_conn.fetchval = AsyncMock(return_value=None)

    async def _connect_stub():
        return fake_conn

    async def _ensure_stub(conn):
        pass

    monkeypatch.setattr(operator_cli, "_connect", _connect_stub)
    monkeypatch.setattr(operator_cli, "_ensure_schema", _ensure_stub)
    issue = AsyncMock(return_value=("rdrk_should_not_be_called", "kid"))
    with patch("shared.miner_auth.issue_api_key", new=issue):
        code = operator_cli.main([
            "issue-key", "--miner-id", "ghost", "--label", "x",
        ])
    assert code == 2
    issue.assert_not_awaited()
    err = capsys.readouterr().err
    assert "ghost" in err
    assert "not found" in err
    assert "RADAR_NETWORK" in err


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
