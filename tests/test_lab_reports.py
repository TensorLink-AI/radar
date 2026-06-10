"""Tests for local/lab_reports.py — deterministic report content, store
roundtrip, HTTP handler shapes. LLM narrative is exercised only as a
no-op (no provider key in tests)."""

from __future__ import annotations

import pytest

from local.lab_reports import (
    build_report,
    generate_round_reports,
    recent_reports,
    report_for,
)
from local.store import LocalStore


@pytest.fixture()
def store(tmp_path):
    s = LocalStore(tmp_path / "t.db")
    yield s
    s.close()


def _pt(name, m):
    return {"name": name, "ncrps": m, "nmase": m}


def _add(store, *, metric, round_id=1, mode="new", parent=None,
         objectives=None, success=True, code="x = 1",
         motivation="try wider patches", task="ts_forecasting",
         analysis=""):
    return store.add_experiment(
        round_id=round_id, miner_id="m", name="exp", code=code,
        motivation=motivation, reasoning="", tool_calls=[], metric=metric,
        success=success, objectives=objectives or {}, score=0.0,
        loss_curve=[], parent_index=parent, task=task, mode=mode,
        analysis=analysis,
    )


def test_report_new_design_no_baseline(store):
    eid = _add(store, metric=0.8)
    report = build_report(store, store.get_experiment(eid))
    assert report["experiment_id"] == eid
    assert report["round_kind"] == "new"
    assert report["hypothesis"] == "try wider patches"
    assert "baseline" not in report
    assert "no baseline" in report["verdict"]


def test_report_failed_experiment(store):
    eid = _add(store, metric=None, success=False,
               analysis="task=x failed: boom")
    report = build_report(store, store.get_experiment(eid))
    assert report["outcome"]["success"] is False
    assert report["verdict"].startswith("failed:")


def test_report_continuation_with_paired_outcome(store):
    parent_pt = [_pt(f"d{i}", 1.0) for i in range(30)]
    child_pt = [_pt(f"d{i}", 0.8) for i in range(30)]
    pid = _add(store, metric=1.0,
               objectives={"per_task": parent_pt}, code="a = 1")
    cid = _add(store, metric=0.8, mode="continue", parent=pid,
               objectives={"per_task": child_pt}, code="a = 2")
    report = build_report(store, store.get_experiment(cid))
    assert report["baseline"]["kind"] == "parent"
    assert report["delta"] == pytest.approx(0.2)
    assert report["paired"]["significant"] is True
    assert "paired-significant" in report["verdict"]
    assert report["diff_stats"]["lines_added"] == 1
    assert report["datasets"]["top_improved"]


def test_report_unsignificant_improvement_called_out(store):
    parent_pt = [_pt(f"d{i}", 1.0) for i in range(30)]
    child_pt = [
        _pt(f"d{i}", 0.9 if i % 2 == 0 else 1.1) for i in range(30)
    ]
    pid = _add(store, metric=1.0, objectives={"per_task": parent_pt})
    cid = _add(store, metric=0.99, mode="continue", parent=pid,
               objectives={"per_task": child_pt})
    report = build_report(store, store.get_experiment(cid))
    assert "NOT paired-significant" in report["verdict"]


def test_report_replicate_verdict(store):
    oid = _add(store, metric=0.50)
    rid = _add(store, metric=0.52, mode="replicate",
               objectives={"replicate_of": oid})
    report = build_report(store, store.get_experiment(rid))
    assert report["round_kind"] == "replicate"
    assert report["verdict"].startswith("noise probe")


def test_report_special_baseline_keys(store):
    base = _add(store, metric=0.5)
    abl = _add(store, metric=0.45, objectives={"ablation_of": base})
    report = build_report(store, store.get_experiment(abl))
    assert report["round_kind"] == "ablation"
    assert report["baseline"]["kind"] == "ablation_target"
    assert report["delta"] == pytest.approx(0.05)


def test_generate_round_reports_and_handlers(store):
    eid = _add(store, metric=0.7, round_id=42)
    n = generate_round_reports(store, 42, "ts_forecasting")
    assert n == 1
    assert store.get_lab_report(eid)["experiment_id"] == eid
    listed = recent_reports(store, n=10, task="ts_forecasting")
    assert len(listed["reports"]) == 1
    # report_for serves the stored copy.
    assert report_for(store, eid)["report"]["experiment_id"] == eid


def test_report_for_builds_on_demand(store):
    eid = _add(store, metric=0.7)
    out = report_for(store, eid)  # no stored report yet — backfill
    assert out["report"]["experiment_id"] == eid
    assert store.get_lab_report(eid) is not None
    assert report_for(store, 9999) == {"error": "not found"}
