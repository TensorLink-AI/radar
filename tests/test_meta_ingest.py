"""Tests for meta.ingest — read frontier rows out of a synthetic
``radar_local.db`` snapshot, exercising the schema-tolerance logic."""

from __future__ import annotations

import json
import sqlite3

import pytest

from meta.ingest import read_frontier_rows
from meta.store import ExperimentStore


def _make_local_db(path, rows):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER NOT NULL,
            miner_id TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            metric REAL,
            success INTEGER NOT NULL DEFAULT 0,
            objectives_json TEXT NOT NULL DEFAULT '{}',
            task TEXT NOT NULL DEFAULT '',
            mode TEXT NOT NULL DEFAULT 'new',
            cumulative_compute REAL NOT NULL DEFAULT 0
        );
        """
    )
    for r in rows:
        conn.execute(
            """INSERT INTO experiments(round_id, miner_id, name, metric,
                success, objectives_json, task, mode, cumulative_compute)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                r["round_id"], r["miner_id"], r.get("name", ""),
                r.get("metric"), int(r.get("success", 0)),
                json.dumps(r.get("objectives", {})),
                r.get("task", ""), r.get("mode", "new"),
                float(r.get("cumulative_compute", 0)),
            ),
        )
    conn.commit()
    conn.close()


def test_read_frontier_rows_returns_expected_shape(tmp_path):
    db = tmp_path / "local.db"
    _make_local_db(db, [
        {"round_id": 1, "miner_id": "m-00", "name": "first",
         "metric": 0.5, "success": 1,
         "objectives": {"crps": 0.4}, "task": "ts_forecasting"},
        {"round_id": 2, "miner_id": "m-00", "name": "second",
         "metric": 0.4, "success": 0, "task": "ts_forecasting"},
    ])
    rows = read_frontier_rows(db)
    assert len(rows) == 2
    assert rows[0]["source_exp_id"] == 1
    assert rows[0]["metric"] == 0.5
    assert rows[0]["objectives"] == {"crps": 0.4}
    assert rows[1]["success"] == 0


def test_read_frontier_handles_missing_columns(tmp_path):
    """A producer running an older radar release with fewer columns
    must not crash the ingester."""
    db = tmp_path / "local.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE experiments(id INTEGER PRIMARY KEY, round_id INT, "
        "miner_id TEXT, metric REAL);"
    )
    conn.execute("INSERT INTO experiments VALUES (1, 1, 'm', 0.5)")
    conn.commit()
    conn.close()
    rows = read_frontier_rows(db)
    assert len(rows) == 1
    assert rows[0]["source_exp_id"] == 1
    assert rows[0]["metric"] == 0.5


def test_read_frontier_skips_bad_schema(tmp_path):
    """A wholly unrecognized schema (no experiments.id) returns []
    rather than raising — so a corrupted snapshot doesn't kill the
    ingester."""
    db = tmp_path / "local.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE experiments(round_id INT, miner_id TEXT)")
    conn.commit()
    conn.close()
    assert read_frontier_rows(db) == []


def test_read_then_upsert_roundtrip(tmp_path):
    db = tmp_path / "local.db"
    _make_local_db(db, [
        {"round_id": 1, "miner_id": "m", "metric": 0.5, "success": 1},
        {"round_id": 2, "miner_id": "m", "metric": 0.4, "success": 1},
    ])
    rows = read_frontier_rows(db)
    store = ExperimentStore(tmp_path / "meta.db")
    try:
        n = store.upsert_frontier_rows("unit-1", rows)
        assert n == 2
        best = store.best_per_unit()
        assert best[0]["best_metric"] == pytest.approx(0.4)
    finally:
        store.close()
