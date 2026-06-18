"""Tests for local/dashboard_reports.py — round-kind classification,
lab-report listing, noise floors over a read-only connection."""

from __future__ import annotations

import sqlite3

import pytest

from local.dashboard_reports import (
    get_lab_report,
    lineage_curve,
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


def test_lineage_curve_stitches_parent_chain(db, tmp_path):
    s, path = db
    root = s.add_experiment(
        round_id=1, miner_id="m", name="v1", code="x", motivation="",
        reasoning="", tool_calls=[], metric=0.5, success=True, objectives={},
        score=0.0, loss_curve=[3.0, 2.0, 1.5], val_curve=[[0, 2.5]],
        task="synthetic_data_generator", mode="new",
    )
    child = s.add_experiment(
        round_id=2, miner_id="m", name="v1+", code="y", motivation="",
        reasoning="", tool_calls=[], metric=0.45, success=True,
        objectives={"continuation_kind": "extend"}, score=0.0,
        loss_curve=[1.1, 0.9], val_curve=[[0, 1.0]],
        task="synthetic_data_generator", mode="continue", n_rounds=2,
        parent_index=root,
    )
    conn = _ro(path)
    lc = lineage_curve(conn, child)
    assert lc["n_runs"] == 2
    assert lc["task"] == "synthetic_data_generator"
    # Ordered root → leaf.
    assert [r["round_id"] for r in lc["runs"]] == [1, 2]
    assert [r["continuation_kind"] for r in lc["runs"]] == [None, "extend"]
    assert [len(r["loss_curve"]) for r in lc["runs"]] == [3, 2]
    assert [len(r["val_curve"]) for r in lc["runs"]] == [1, 1]
    # A novel run is its own single-element lineage.
    assert lineage_curve(conn, root)["n_runs"] == 1
    conn.close()


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
