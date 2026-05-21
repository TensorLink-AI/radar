"""Tests for the miner-side prompt-population store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miner_template import prompts as ps
from miner_template.prompts import Prompt


def test_load_active_empty(tmp_path: Path):
    assert ps.load_active(tmp_path) == []


def test_save_then_load_roundtrip(tmp_path: Path):
    pop = [
        Prompt.new(template="p1", generation=1),
        Prompt.new(template="p2", generation=1),
    ]
    ps.save_active(pop, tmp_path)
    got = ps.load_active(tmp_path)
    assert len(got) == 2
    assert {p.template for p in got} == {"p1", "p2"}
    assert all(p.generation == 1 for p in got)


def test_save_atomic_no_partial_on_failure(tmp_path: Path, monkeypatch):
    """If os.fsync raises, ``active.json`` must not exist."""
    pop = [Prompt.new(template="ok")]
    ps.save_active(pop, tmp_path)  # establish baseline
    original = ps.load_active(tmp_path)

    # Now break the write halfway through and confirm the original
    # active.json is untouched.
    real_replace = ps.os.replace

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ps.os, "replace", boom)
    with pytest.raises(OSError):
        ps.save_active([Prompt.new(template="will-fail")], tmp_path)

    monkeypatch.setattr(ps.os, "replace", real_replace)
    assert ps.load_active(tmp_path) == original


def test_archive_and_list_history(tmp_path: Path):
    ps.save_active([Prompt.new(template="gen0")], tmp_path)
    p = ps.archive_current(1, tmp_path)
    assert p is not None and p.exists()
    ps.save_active([Prompt.new(template="gen1")], tmp_path)
    ps.archive_current(2, tmp_path)
    assert ps.list_history(tmp_path) == [1, 2]


def test_archive_noop_when_no_active(tmp_path: Path):
    assert ps.archive_current(1, tmp_path) is None


def test_load_generation_roundtrip(tmp_path: Path):
    seed = [Prompt.new(template="archived")]
    ps.save_active(seed, tmp_path)
    ps.archive_current(5, tmp_path)
    loaded = ps.load_generation(5, tmp_path)
    assert len(loaded) == 1
    assert loaded[0].template == "archived"


def test_load_generation_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ps.load_generation(99, tmp_path)


def test_rollback_archives_current_then_restores(tmp_path: Path):
    a = [Prompt.new(template="A")]
    ps.save_active(a, tmp_path)
    ps.archive_current(1, tmp_path)
    b = [Prompt.new(template="B")]
    ps.save_active(b, tmp_path)
    # Rollback to gen 1 ('A'); the current 'B' should be archived too.
    ps.rollback_to(1, tmp_path)
    active = ps.load_active(tmp_path)
    assert [p.template for p in active] == ["A"]
    # Both 1 and 2 (the just-archived B) should now be in history.
    assert ps.list_history(tmp_path) == [1, 2]
    assert [p.template for p in ps.load_generation(2, tmp_path)] == ["B"]


def test_pick_for_round_deterministic_roundrobin(tmp_path: Path):
    pop = [
        Prompt(id="a", template="A"),
        Prompt(id="b", template="B"),
        Prompt(id="c", template="C"),
    ]
    picks = [ps.pick_for_round(pop, i).id for i in range(7)]
    assert picks == ["a", "b", "c", "a", "b", "c", "a"]


def test_pick_for_round_empty_raises():
    with pytest.raises(ValueError):
        ps.pick_for_round([], 0)


def test_seed_default_creates_active(tmp_path: Path):
    pop = ps.seed_default(tmp_path)
    assert len(pop) == 1
    on_disk = ps.load_active(tmp_path)
    assert len(on_disk) == 1
    assert on_disk[0].generation == 0


def test_seed_default_is_noop_when_active_exists(tmp_path: Path):
    existing = [Prompt.new(template="kept")]
    ps.save_active(existing, tmp_path)
    pop = ps.seed_default(tmp_path)
    assert len(pop) == 1
    assert pop[0].template == "kept"


def test_prompt_from_dict_resilient_to_missing_fields():
    p = Prompt.from_dict({"template": "t"})
    assert p.template == "t"
    assert p.id  # auto-generated
    assert p.generation == 0
    assert p.parent_id is None
    assert p.metadata == {}


def test_save_uses_atomic_replace(tmp_path: Path):
    """Sanity: the final file lands at ``active.json`` and no ``.tmp``
    file is left behind."""
    ps.save_active([Prompt.new(template="x")], tmp_path)
    assert (tmp_path / "active.json").exists()
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_archived_file_is_readable_json(tmp_path: Path):
    """Make sure archived generations are still valid JSON, not just bytes."""
    ps.save_active([Prompt.new(template="t1"), Prompt.new(template="t2")], tmp_path)
    dst = ps.archive_current(3, tmp_path)
    assert dst is not None
    data = json.loads(dst.read_text())
    assert isinstance(data, dict)
    assert len(data["prompts"]) == 2
