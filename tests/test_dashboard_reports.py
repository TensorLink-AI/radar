"""Tests for local/dashboard_reports.py — round-kind classification,
lab-report listing, noise floors over a read-only connection."""

from __future__ import annotations

import sqlite3

import pytest

from local.dashboard_reports import (
    get_lab_report,
    list_lab_reports,
    noise_floors,
    round_kind,
    special_counts,
)
from local.store import LocalStore


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.db"
    s = LocalStore(path)
    yield s, path
    s.close()


def _ro(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _add(s, *, metric=0.5, mode="new", objs=None, parent=None):
    return s.add_experiment(
        round_id=1, miner_id="m", name="e", code="x", motivation="",
        reasoning="", tool_calls=[], metric=metric, success=True,
        objectives=objs or {}, score=0.0, loss_curve=[],
        parent_index=parent, task="ts_forecasting", mode=mode,
    )


def test_round_kind_precedence():
    assert round_kind("new", {}) == "new"
    assert round_kind("continue", {}) == "continuation"
    assert round_kind("continue", {"continuation_kind": "extend"}) \
        == "continuation:extend"
    assert round_kind("replicate", {}) == "replicate"
    assert round_kind("new", {"ablation_of": 3}) == "ablation"
    assert round_kind("new", {"recipe_only_of": 3}) == "recipe_only"
    assert round_kind("new", {"transfer_of": 3}) == "transfer"
    # Stamp beats mode.
    assert round_kind("replicate", {"replicate_of": 1}) == "replicate"


def test_lab_reports_roundtrip_and_missing_table(db, tmp_path):
    s, path = db
    eid = _add(s)
    s.add_lab_report(experiment_id=eid, round_id=1, task="ts_forecasting",
                     report={"experiment_id": eid, "verdict": "ok"})
    conn = _ro(path)
    rows = list_lab_reports(conn, n=10)
    assert len(rows) == 1 and rows[0]["verdict"] == "ok"
    assert "_created_at" in rows[0]
    assert get_lab_report(conn, eid)["verdict"] == "ok"
    assert get_lab_report(conn, 999) is None
    conn.close()
    # A DB without the table degrades to empty, never raises.
    bare = sqlite3.connect(":memory:")
    bare.row_factory = sqlite3.Row
    bare.execute("CREATE TABLE experiments (id INTEGER)")
    assert list_lab_reports(bare) == []
    assert get_lab_report(bare, 1) is None


def test_noise_floors_and_special_counts(db):
    s, path = db
    src = _add(s, metric=0.50)
    _add(s, metric=0.54, mode="replicate", objs={"replicate_of": src})
    _add(s, metric=0.45, objs={"ablation_of": src})
    conn = _ro(path)
    floors = noise_floors(conn)
    assert floors["overall"]["n_pairs"] == 1
    assert "ts_forecasting" in floors["by_task"]
    counts = special_counts(conn)
    assert counts["n_replicate"] == 1
    assert counts["n_ablation"] == 1
    assert counts["n_transfer"] == 0
    conn.close()
