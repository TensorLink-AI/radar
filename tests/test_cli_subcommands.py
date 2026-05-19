"""Tests for the miner CLI subcommands.

We exercise the CLI by importing ``miner.cli`` directly and calling
``dispatch(argv)`` with controlled argv.  HTTP calls are intercepted via
``httpx.MockTransport`` on a fake ``MinerResultsClient``.

GEPA itself isn't exercised here — we route to ``random_mutate`` to keep
the test fast and dependency-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from miner import cli
from miner_template import prompts as prompts_mod
from miner_template.prompts import Prompt
from miner_template.results_client import MinerResultsClient


@pytest.fixture
def fake_results_client(monkeypatch):
    """Install a MockTransport-backed MinerResultsClient.

    Returns a dict the test populates with ``submissions``, ``results``,
    ``summary`` to control the fake server's responses.
    """
    state: dict = {"submissions": [], "results": [], "summary": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/miners/me/submissions":
            return httpx.Response(200, json={"submissions": state["submissions"]})
        if path == "/miners/me/results":
            return httpx.Response(200, json={"results": state["results"]})
        if path == "/miners/me/summary":
            return httpx.Response(200, json=state["summary"])
        return httpx.Response(404, json={"error": "no route"})

    original_init = MinerResultsClient.__init__

    def patched_init(self, db_url, api_key, *, timeout=30.0, client=None):
        if client is None:
            transport = httpx.MockTransport(handler)
            client = httpx.Client(transport=transport)
        original_init(self, db_url, api_key, timeout=timeout, client=client)

    monkeypatch.setattr(MinerResultsClient, "__init__", patched_init)
    return state


# ── results ─────────────────────────────────────────────────────────


def test_results_requires_db_url_and_key(capsys):
    code = cli.dispatch([
        "miner", "results", "--db-url", "", "--api-key", "",
    ])
    assert code == 2
    assert "missing --db-url or --api-key" in capsys.readouterr().err


def test_results_prints_table(fake_results_client, capsys):
    fake_results_client["results"] = [{
        "round_id": 12, "submission_id": "s-1", "task_name": "ts",
        "prompt_id": "p-abc", "architecture_code": "code",
        "scores": {"raw_score": 0.42},
    }]
    code = cli.dispatch([
        "miner", "results",
        "--db-url", "http://db", "--api-key", "k",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "round" in out
    assert "s-1" in out
    assert "0.4200" in out


def test_results_json_mode(fake_results_client, capsys):
    fake_results_client["results"] = [{
        "round_id": 1, "submission_id": "s-1", "task_name": "t",
        "prompt_id": "p", "scores": {"raw_score": 0.5},
    }]
    code = cli.dispatch([
        "miner", "results",
        "--db-url", "http://db", "--api-key", "k", "--json",
    ])
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    rec = json.loads(out[0])
    assert rec["submission_id"] == "s-1"
    assert rec["scores"]["raw_score"] == 0.5


def test_results_empty(fake_results_client, capsys):
    code = cli.dispatch([
        "miner", "results",
        "--db-url", "http://db", "--api-key", "k",
    ])
    assert code == 0
    assert "(no results)" in capsys.readouterr().out


# ── optimize ─────────────────────────────────────────────────────────


def test_optimize_requires_db_url_and_key(capsys):
    code = cli.dispatch([
        "miner", "optimize", "--db-url", "", "--api-key", "",
    ])
    assert code == 2
    assert "missing --db-url" in capsys.readouterr().err


def test_optimize_without_seed_or_active_pop_fails(
    fake_results_client, tmp_path, capsys,
):
    code = cli.dispatch([
        "miner", "optimize",
        "--db-url", "http://db", "--api-key", "k",
        "--prompts-dir", str(tmp_path),
        "--optimizer", "random_mutate",
    ])
    assert code == 2
    assert "no active prompt population" in capsys.readouterr().err


def test_optimize_with_seed_creates_pop_and_writes_generation(
    fake_results_client, tmp_path, capsys,
):
    code = cli.dispatch([
        "miner", "optimize",
        "--db-url", "http://db", "--api-key", "k",
        "--prompts-dir", str(tmp_path),
        "--optimizer", "random_mutate",
        "--population", "4",
        "--seed",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Seeded default prompt" in out
    pop = prompts_mod.load_active(tmp_path)
    assert len(pop) == 4
    assert prompts_mod.list_history(tmp_path) == [1]


def test_optimize_dry_run_does_not_write(
    fake_results_client, tmp_path, capsys,
):
    prompts_mod.save_active(
        [Prompt.new(template="A"), Prompt.new(template="B")], tmp_path,
    )
    fake_results_client["results"] = [{
        "round_id": 1, "submission_id": "s", "task_name": "t",
        "prompt_id": "doesnt-matter", "scores": {"raw_score": 0.1},
    }]
    before = prompts_mod.load_active(tmp_path)
    code = cli.dispatch([
        "miner", "optimize",
        "--db-url", "http://db", "--api-key", "k",
        "--prompts-dir", str(tmp_path),
        "--optimizer", "random_mutate",
        "--population", "3", "--dry-run",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "would write population" in out
    after = prompts_mod.load_active(tmp_path)
    assert [p.id for p in before] == [p.id for p in after]
    assert prompts_mod.list_history(tmp_path) == []


def test_optimize_unknown_optimizer(fake_results_client, tmp_path, capsys):
    prompts_mod.save_active([Prompt.new(template="A")], tmp_path)
    code = cli.dispatch([
        "miner", "optimize",
        "--db-url", "http://db", "--api-key", "k",
        "--prompts-dir", str(tmp_path),
        "--optimizer", "no-such-optimizer",
    ])
    assert code == 2
    assert "cannot resolve optimizer" in capsys.readouterr().err


# ── prompts ──────────────────────────────────────────────────────────


def test_prompts_list_empty(tmp_path, capsys):
    code = cli.dispatch([
        "miner", "prompts", "list", "--prompts-dir", str(tmp_path),
    ])
    assert code == 0
    assert "(no active prompt population)" in capsys.readouterr().out


def test_prompts_list_shows_population(tmp_path, capsys):
    prompts_mod.save_active([
        Prompt(id="aaaaaaaa", template="hello world", generation=2),
    ], tmp_path)
    code = cli.dispatch([
        "miner", "prompts", "list", "--prompts-dir", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "gen 2" in out
    assert "aaaaaaaa" in out


def test_prompts_history_empty(tmp_path, capsys):
    code = cli.dispatch([
        "miner", "prompts", "history", "--prompts-dir", str(tmp_path),
    ])
    assert code == 0
    assert "(no archived generations)" in capsys.readouterr().out


def test_prompts_history_lists_archives(tmp_path, capsys):
    prompts_mod.save_active([Prompt.new(template="a")], tmp_path)
    prompts_mod.archive_current(1, tmp_path)
    prompts_mod.archive_current(7, tmp_path)
    code = cli.dispatch([
        "miner", "prompts", "history", "--prompts-dir", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert "gen_001" in out
    assert "gen_007" in out


def test_prompts_rollback_missing_generation(tmp_path, capsys):
    code = cli.dispatch([
        "miner", "prompts", "rollback", "99", "--prompts-dir", str(tmp_path),
    ])
    assert code == 2
    assert "no archived generation" in capsys.readouterr().err


def test_prompts_rollback_restores_archived_pop(tmp_path):
    prompts_mod.save_active([Prompt(id="x", template="A")], tmp_path)
    prompts_mod.archive_current(1, tmp_path)
    prompts_mod.save_active([Prompt(id="y", template="B")], tmp_path)
    code = cli.dispatch([
        "miner", "prompts", "rollback", "1", "--prompts-dir", str(tmp_path),
    ])
    assert code == 0
    cur = prompts_mod.load_active(tmp_path)
    assert [p.id for p in cur] == ["x"]


# ── routing ──────────────────────────────────────────────────────────


def test_is_subcommand_matches_known_tokens():
    assert cli.is_subcommand(["miner", "results"])
    assert cli.is_subcommand(["miner", "optimize"])
    assert cli.is_subcommand(["miner", "prompts"])
    assert not cli.is_subcommand(["miner"])
    assert not cli.is_subcommand(["miner", "run"])
    assert not cli.is_subcommand(["miner", "--netuid", "1"])


def test_dispatch_unknown_subcommand(capsys):
    code = cli.dispatch(["miner", "wat"])
    assert code == 2
    assert "Unknown subcommand" in capsys.readouterr().err
