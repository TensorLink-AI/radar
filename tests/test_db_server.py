"""Tests for validator.db_server — FastAPI experiment DB endpoints."""

import tempfile
from fastapi.testclient import TestClient

from shared.database import DataElement, ExperimentDB
from shared.sqlite_store import SQLiteExperimentStore
from validator.db_server import (
    app, set_db, set_access_logger, set_hotkey_map,
    _extract_experiment_ids_from_path,
)


def _seed_db():
    d = tempfile.mkdtemp()
    db = ExperimentDB(db_dir=d)
    db.add(DataElement(
        name="exp_0", code="v0", motivation="baseline", analysis="initial run",
        success=True, metric=1.0, parent=None, generation=0,
        objectives={"val_bpb": 1.0, "exec_time": 200}, score=0.5,
    ))
    db.add(DataElement(
        name="exp_1", code="v1", motivation="gated attention",
        analysis="improved metric", success=True, metric=0.95,
        parent=0, generation=1,
        objectives={"val_bpb": 0.95, "exec_time": 210}, score=0.7,
    ))
    db.add(DataElement(
        name="exp_2", code="v2", motivation="cosine schedule",
        analysis="better convergence", success=True, metric=0.92,
        parent=1, generation=2,
        objectives={"val_bpb": 0.92, "exec_time": 190}, score=0.75,
    ))
    db.add(DataElement(
        name="exp_3", code="v3", motivation="larger batch",
        analysis="OOM crash", success=False, metric=None,
        parent=2, generation=3,
    ))
    db.add(DataElement(
        name="exp_4", code="v4", motivation="reduced depth",
        analysis="faster but worse", success=True, metric=0.98,
        parent=2, generation=3,
        objectives={"val_bpb": 0.98, "exec_time": 150}, score=0.55,
    ))
    set_db(db)
    return db


def test_health():
    _seed_db()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_get_experiment():
    _seed_db()
    client = TestClient(app)
    r = client.get("/experiments/0")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "exp_0"
    assert data["results"]["success"] is True


def test_get_experiment_not_found():
    _seed_db()
    client = TestClient(app)
    r = client.get("/experiments/999")
    assert r.status_code == 404


def test_get_recent():
    _seed_db()
    client = TestClient(app)
    r = client.get("/experiments/recent?n=3")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3
    assert data[0]["name"] == "exp_4"  # most recent first


def test_get_failures():
    _seed_db()
    client = TestClient(app)
    r = client.get("/experiments/failures?n=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["name"] == "exp_3"


def test_get_lineage():
    _seed_db()
    client = TestClient(app)
    r = client.get("/experiments/lineage/2")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3
    assert data[0]["name"] == "exp_0"
    assert data[2]["name"] == "exp_2"


def test_search():
    _seed_db()
    client = TestClient(app)
    r = client.post("/experiments/search", json={"query": "attention"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1
    assert data[0]["name"] == "exp_1"


def test_get_stats():
    _seed_db()
    client = TestClient(app)
    r = client.get("/experiments/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 5
    assert data["successful"] == 4
    assert data["failed"] == 1


def test_pareto():
    _seed_db()
    client = TestClient(app)
    r = client.get("/experiments/pareto")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1


def test_challenge_no_active():
    _seed_db()
    from validator.db_server import set_challenge
    set_challenge(None)
    client = TestClient(app)
    r = client.get("/challenge")
    assert r.status_code == 404


# ── Access logging helper tests ──────────────────────


def test_extract_experiment_ids_from_path():
    assert _extract_experiment_ids_from_path("/experiments/42") == [42]
    assert _extract_experiment_ids_from_path("/experiments/lineage/7") == [7]
    assert sorted(_extract_experiment_ids_from_path("/experiments/diff/3/7")) == [3, 7]
    assert _extract_experiment_ids_from_path("/experiments/stats") == []


# ── Provenance endpoints ─────────────────────────────


def _seed_sqlite_db(tmp_path):
    """Seed a SQLiteExperimentStore with provenance support."""
    store = SQLiteExperimentStore(db_path=str(tmp_path / "test.db"))
    store.add(DataElement(
        name="exp_0", code="class Model: pass", motivation="baseline",
        analysis="initial", success=True, metric=1.0, parent=None, generation=0,
        objectives={"val_bpb": 1.0}, score=0.5, task="ts",
    ))
    store.add(DataElement(
        name="exp_1", code="class Model:\n    def forward(self): pass",
        motivation="improved", analysis="better",
        success=True, metric=0.9, parent=0, generation=1,
        objectives={"val_bpb": 0.9}, score=0.7, task="ts",
    ))
    # Record components
    store.provenance.record_components(0, ["LayerNorm"])
    store.provenance.record_components(1, ["LayerNorm", "GELU"])
    set_db(store)
    return store


def test_provenance_influences(tmp_path):
    store = _seed_sqlite_db(tmp_path)
    client = TestClient(app)
    r = client.get("/provenance/1/influences")
    assert r.status_code == 200
    data = r.json()
    # exp_1 shares LayerNorm with exp_0
    shared = [i for i in data if i["evidence_type"] == "shared_component"]
    assert any(i["source_id"] == 0 for i in shared)


def test_provenance_impact(tmp_path):
    store = _seed_sqlite_db(tmp_path)
    client = TestClient(app)
    r = client.get("/provenance/0/impact")
    assert r.status_code == 200


def test_provenance_similar(tmp_path):
    store = _seed_sqlite_db(tmp_path)
    client = TestClient(app)
    r = client.get("/provenance/0/similar?top_k=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1
    assert "jaccard" in data[0]
    assert "diff_ratio" in data[0]


def test_provenance_component_stats(tmp_path):
    store = _seed_sqlite_db(tmp_path)
    client = TestClient(app)
    r = client.get("/provenance/component_stats")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1
    layer_norm = [s for s in data if s["component"] == "LayerNorm"]
    assert layer_norm[0]["count"] == 2


def test_provenance_components(tmp_path):
    store = _seed_sqlite_db(tmp_path)
    client = TestClient(app)
    r = client.get("/provenance/components?component=LayerNorm")
    assert r.status_code == 200
    data = r.json()
    assert set(data["experiment_ids"]) == {0, 1}


def test_provenance_dead_ends(tmp_path):
    store = _seed_sqlite_db(tmp_path)
    client = TestClient(app)
    r = client.get("/provenance/dead_ends")
    assert r.status_code == 200
    data = r.json()
    assert "dead_ends" in data


def test_provenance_experiment_graph(tmp_path):
    store = _seed_sqlite_db(tmp_path)
    client = TestClient(app)
    r = client.get("/provenance/1/graph")
    assert r.status_code == 200
    data = r.json()
    assert data["experiment_id"] == 1
    assert "influences" in data
    assert "impact" in data
    assert "components" in data
