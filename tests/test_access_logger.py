"""Tests for shared.access_logger — pure-Python _extract_experiment_ids helper.

DB-backed access logger tests are in tests/test_pg_provenance.py
(PgAccessLogger is tested alongside PgProvenanceQuery).
"""

from shared.access_logger import _extract_experiment_ids


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
