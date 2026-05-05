"""Tests for shared.access_logger — pure-Python _extract_experiment_ids helper.

DB-backed access logger tests are in tests/test_pg_provenance.py
(PgAccessLogger is tested alongside PgProvenanceQuery).
"""

from shared.access_logger import (
    _extract_experiment_ids,
    extract_ids_from_body,
    extract_ids_from_path,
)


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


def test_extract_ids_from_body_json_dict():
    body = b'{"index": 42, "name": "foo"}'
    assert extract_ids_from_body(body, "application/json") == [42]


def test_extract_ids_from_body_json_list():
    body = b'[{"index": 1}, {"index": 2}, {"index": 3}]'
    assert extract_ids_from_body(body, "application/json; charset=utf-8") == [1, 2, 3]


def test_extract_ids_from_body_non_json():
    assert extract_ids_from_body(b"binary\x00data", "application/octet-stream") == []


def test_extract_ids_from_body_empty():
    assert extract_ids_from_body(b"", "application/json") == []
    assert extract_ids_from_body(b"null", "application/json") == []


def test_extract_ids_from_body_parse_error():
    assert extract_ids_from_body(b"{not json", "application/json") == []


# ── New keys: provenance + diff + verify response shapes ──────────


def test_extract_diff_endpoint_shape():
    # /experiments/diff/{a}/{b} → {"index_a": A, "index_b": B, "diff": "..."}
    ids = _extract_experiment_ids({"index_a": 7, "index_b": 12, "diff": "..."})
    assert set(ids) == {7, 12}


def test_extract_verify_endpoint_shape():
    # /experiments/{id}/verify → {"experiment_id": N, ...}
    ids = _extract_experiment_ids({"experiment_id": 99, "substrate_cids": []})
    assert ids == [99]


def test_extract_influences_response():
    # /provenance/{id}/influences → list of {"source_id": ...}
    ids = _extract_experiment_ids([
        {"source_id": 3, "evidence_type": "accessed"},
        {"source_id": 7, "evidence_type": "frontier"},
    ])
    assert set(ids) == {3, 7}


def test_extract_impact_response():
    # /provenance/{id}/impact → list of {"target_id": ...}
    ids = _extract_experiment_ids([
        {"target_id": 4, "evidence_type": "accessed_by"},
        {"target_id": 9, "evidence_type": "frontier_for"},
    ])
    assert set(ids) == {4, 9}


def test_extract_components_response():
    # /provenance/components → {"experiment_ids": [...], "component": "..."}
    ids = _extract_experiment_ids({
        "component": "MultiHeadAttention", "experiment_ids": [1, 2, 3],
    })
    assert set(ids) == {1, 2, 3}


def test_extract_dead_ends_response():
    # /provenance/dead_ends → {"dead_ends": [...]}
    ids = _extract_experiment_ids({"dead_ends": [10, 20, 30]})
    assert set(ids) == {10, 20, 30}


def test_extract_bools_not_treated_as_ids():
    # ``isinstance(True, int)`` is True in Python — guard against it.
    ids = _extract_experiment_ids({"index": True, "experiment_id": False})
    assert ids == []


# ── Path-encoded IDs ──────────────────────────────────────────────


def test_path_extract_experiment_detail():
    assert extract_ids_from_path("/experiments/42") == [42]


def test_path_extract_experiment_subroute():
    assert extract_ids_from_path("/experiments/42/diff") == [42]
    assert extract_ids_from_path("/experiments/42/lineage_diffs") == [42]
    assert extract_ids_from_path("/experiments/42/verify") == [42]


def test_path_extract_lineage():
    assert extract_ids_from_path("/experiments/lineage/7") == [7]


def test_path_extract_diff_pair():
    assert extract_ids_from_path("/experiments/diff/3/8") == [3, 8]


def test_path_extract_provenance():
    assert extract_ids_from_path("/provenance/15/influences") == [15]
    assert extract_ids_from_path("/provenance/15/impact") == [15]
    assert extract_ids_from_path("/provenance/15/similar") == [15]
    assert extract_ids_from_path("/provenance/15/graph") == [15]


def test_path_extract_skips_non_id_routes():
    # Aggregate routes like /experiments/recent must not match.
    assert extract_ids_from_path("/experiments/recent") == []
    assert extract_ids_from_path("/experiments/pareto") == []
    assert extract_ids_from_path("/experiments/families") == []
    assert extract_ids_from_path("/provenance/components") == []
    assert extract_ids_from_path("/provenance/dead_ends") == []
    assert extract_ids_from_path("/frontier") == []
    assert extract_ids_from_path("/challenge") == []
    assert extract_ids_from_path("") == []
