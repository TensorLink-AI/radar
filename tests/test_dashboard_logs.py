"""Tests for local/dashboard_logs.py — task-scoped service-log queries.

``agent_events`` has no task column, so a per-task service log is built by
mapping events back to the rounds that ran that task. These cover that the
scoping is correct and degrades gracefully."""

from __future__ import annotations

import sqlite3

import pytest

from local.dashboard_logs import list_events
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


def _exp(s, *, round_id, task):
    return s.add_experiment(
        round_id=round_id, miner_id="m", name="e", code="x", motivation="",
        reasoning="", tool_calls=[], metric=0.5, success=True, objectives={},
        score=0.0, loss_curve=[], task=task,
    )


def test_events_scoped_to_task_rounds(db):
    s, path = db
    _exp(s, round_id=1, task="synthetic_data_generator")
    _exp(s, round_id=2, task="ts_forecasting")
    s.record_agent_event(kind="llm", miner_id="m", round_id=1,
                         endpoint="/llm/chat", status=200)
    s.record_agent_event(kind="frontier", miner_id="m", round_id=1,
                         endpoint="/frontier", status=200)
    s.record_agent_event(kind="llm", miner_id="m", round_id=2,
                         endpoint="/llm/chat", status=200)

    conn = _ro(path)
    # Datagen scope sees only its own round's events.
    sdg = list_events(conn, task="synthetic_data_generator")
    assert sorted(e["round_id"] for e in sdg) == [1, 1]
    # ts scope sees only its round.
    ts = list_events(conn, task="ts_forecasting")
    assert [e["round_id"] for e in ts] == [2]
    # A task with no rounds yields nothing (not the whole unscoped log).
    assert list_events(conn, task="nope") == []
    # No task filter returns everything.
    assert len(list_events(conn)) == 3
    # Task scope still composes with other filters.
    assert len(list_events(conn, task="synthetic_data_generator",
                           kind="frontier")) == 1
    conn.close()


def test_events_task_scope_missing_table():
    bare = sqlite3.connect(":memory:")
    bare.row_factory = sqlite3.Row
    bare.execute("CREATE TABLE agent_events (id INTEGER)")
    # No experiments table → task scope finds no rounds → empty, never raises.
    assert list_events(bare, task="synthetic_data_generator") == []
