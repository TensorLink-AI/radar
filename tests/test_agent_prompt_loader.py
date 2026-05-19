"""Tests for the in-agent prompt loader (miner_template/agent.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miner_template import agent


def _write_active(tmp_path: Path, prompts: list[dict]) -> Path:
    (tmp_path / "prompts").mkdir()
    p = tmp_path / "prompts" / "active.json"
    p.write_text(json.dumps({"prompts": prompts}))
    return p


def test_loader_returns_empty_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "missing"))
    got = agent._load_active_prompt(round_id=0)
    assert got == {"id": "", "template": ""}


def test_loader_returns_empty_when_population_empty(monkeypatch, tmp_path):
    _write_active(tmp_path, [])
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "prompts"))
    got = agent._load_active_prompt(round_id=0)
    assert got == {"id": "", "template": ""}


def test_loader_returns_round_robin_pick(monkeypatch, tmp_path):
    _write_active(tmp_path, [
        {"id": "a", "template": "A"},
        {"id": "b", "template": "B"},
        {"id": "c", "template": "C"},
    ])
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "prompts"))
    picks = [agent._load_active_prompt(i)["id"] for i in range(5)]
    assert picks == ["a", "b", "c", "a", "b"]


def test_loader_handles_corrupt_json(monkeypatch, tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "active.json").write_text("{not json")
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "prompts"))
    assert agent._load_active_prompt(0) == {"id": "", "template": ""}


def test_loader_handles_legacy_list_payload(monkeypatch, tmp_path):
    """Older optimizer versions wrote a bare list — be permissive."""
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "active.json").write_text(json.dumps([
        {"id": "x", "template": "T"},
    ]))
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "prompts"))
    got = agent._load_active_prompt(0)
    assert got["id"] == "x"


def test_design_architecture_emits_prompt_id(monkeypatch, tmp_path):
    """End-to-end: design_architecture round-trips the prompt id."""
    _write_active(tmp_path, [{"id": "the-active-one", "template": "T"}])
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "prompts"))

    # design_architecture also calls load_scratchpad/save_scratchpad;
    # those names are injected by the validator's sandbox at runtime,
    # so we inject stubs into the module globals here.
    agent.__dict__["load_scratchpad"] = lambda c: tmp_path
    agent.__dict__["save_scratchpad"] = lambda c, d: None

    class _FakeClient:
        def post_json(self, *a, **k): return {}
        def get_json(self, *a, **k): return {}

    out = agent.design_architecture(
        challenge={
            "round_id": 0, "min_flops_equivalent": 1_000_000,
            "max_flops_equivalent": 5_000_000, "feasible_frontier": [],
        },
        client=_FakeClient(),
    )
    assert out["prompt_id"] == "the-active-one"


def test_design_architecture_empty_prompt_id_when_no_population(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(agent, "load_scratchpad", lambda c: tmp_path)
    monkeypatch.setattr(agent, "save_scratchpad", lambda c, d: None)

    class _FakeClient:
        def post_json(self, *a, **k): return {}
        def get_json(self, *a, **k): return {}

    out = agent.design_architecture(
        challenge={
            "round_id": 0, "min_flops_equivalent": 1_000_000,
            "max_flops_equivalent": 5_000_000, "feasible_frontier": [],
        },
        client=_FakeClient(),
    )
    assert out["prompt_id"] == ""
