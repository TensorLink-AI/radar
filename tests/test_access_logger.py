"""Tests for shared.access_logger — append-only miner API access log."""

import sqlite3

from shared.access_logger import AccessLogger, _extract_experiment_ids


def _make_logger():
    conn = sqlite3.connect(":memory:")
    return AccessLogger(conn)


# ── Schema ───────────────────────────────────────────


def test_schema_creates_table():
    al = _make_logger()
    tables = al.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    assert ("miner_access_log",) in tables


# ── log_access + get_accessed ────────────────────────


def test_log_and_retrieve():
    al = _make_logger()
    al.set_round(1)
    al.log_access("hk1", miner_uid=0, endpoint="/experiments/0", experiment_ids=[0])
    assert al.get_accessed("hk1", 1) == {0}


def test_log_multiple_experiments():
    al = _make_logger()
    al.set_round(1)
    al.log_access("hk1", 0, "/experiments/0", [0, 1, 2])
    al.log_access("hk1", 0, "/experiments/5", [5])
    assert al.get_accessed("hk1", 1) == {0, 1, 2, 5}


def test_separate_miners():
    al = _make_logger()
    al.set_round(1)
    al.log_access("hk1", 0, "/e/0", [0])
    al.log_access("hk2", 1, "/e/1", [1])
    assert al.get_accessed("hk1", 1) == {0}
    assert al.get_accessed("hk2", 1) == {1}


def test_separate_rounds():
    al = _make_logger()
    al.set_round(1)
    al.log_access("hk1", 0, "/e/0", [0])
    al.set_round(2)
    al.log_access("hk1", 0, "/e/1", [1])
    assert al.get_accessed("hk1", 1) == {0}
    assert al.get_accessed("hk1", 2) == {1}


# ── log_request convenience ──────────────────────────


def test_log_request_single_experiment():
    al = _make_logger()
    al.set_round(1)
    al.log_request("hk1", "/experiments/42", response_data={"index": 42, "name": "exp"})
    assert 42 in al.get_accessed("hk1", 1)


def test_log_request_list_response():
    al = _make_logger()
    al.set_round(1)
    al.log_request("hk1", "/experiments/recent", response_data=[
        {"index": 0, "name": "a"}, {"index": 1, "name": "b"},
    ])
    assert al.get_accessed("hk1", 1) == {0, 1}


def test_log_request_no_response():
    al = _make_logger()
    al.set_round(1)
    al.log_request("hk1", "/experiments/stats")
    assert al.get_accessed("hk1", 1) == set()


# ── _extract_experiment_ids ──────────────────────────


def test_extract_from_dict():
    ids = _extract_experiment_ids({"index": 5, "root_index": 0, "latest_index": 10})
    assert set(ids) == {0, 5, 10}


def test_extract_from_list():
    ids = _extract_experiment_ids([{"index": 1}, {"index": 2}, {"index": 3}])
    assert set(ids) == {1, 2, 3}


def test_extract_empty():
    assert _extract_experiment_ids(None) == []
    assert _extract_experiment_ids({}) == []
    assert _extract_experiment_ids([]) == []


def test_extract_family_summary():
    ids = _extract_experiment_ids([
        {"root_index": 0, "latest_index": 5},
        {"root_index": 10, "latest_index": 15},
    ])
    assert set(ids) == {0, 5, 10, 15}


# ── get_round_access ─────────────────────────────────


def test_get_round_access():
    al = _make_logger()
    al.set_round(1)
    al.log_access("hk1", 0, "/e", [0, 1])
    al.log_access("hk2", 1, "/e", [2])
    access = al.get_round_access(1)
    assert access["hk1"] == {0, 1}
    assert access["hk2"] == {2}


# ── In-memory fast path ──────────────────────────────


def test_in_memory_fast_path():
    al = _make_logger()
    al.set_round(5)
    al.log_access("hk1", 0, "/e", [10, 20])
    assert al.get_accessed("hk1") == {10, 20}


def test_historical_round_falls_back_to_sql():
    al = _make_logger()
    al.set_round(1)
    al.log_access("hk1", 0, "/e", [5])
    al.set_round(2)
    assert al.get_accessed("hk1", 1) == {5}


def test_miner_uid_stored():
    al = _make_logger()
    al.set_round(1)
    al.log_access("hk1", miner_uid=42, endpoint="/e", experiment_ids=[0])
    row = al.conn.execute("SELECT miner_uid FROM miner_access_log").fetchone()
    assert row[0] == 42
