"""Tests for shared.provenance — raw facts + on-demand query layer."""

import sqlite3
import time

from shared.access_logger import AccessLogger
from shared.provenance import (
    ProvenanceQuery, detect_components, compute_similarity,
)


def _make_db():
    """Create in-memory DB with experiments table, access logger, and provenance."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            code TEXT NOT NULL DEFAULT '',
            miner_uid INTEGER NOT NULL DEFAULT -1,
            miner_hotkey TEXT NOT NULL DEFAULT '',
            metric REAL,
            success BOOLEAN NOT NULL DEFAULT 0,
            round_id INTEGER,
            task TEXT NOT NULL DEFAULT '',
            parent_index INTEGER,
            generation INTEGER NOT NULL DEFAULT 0,
            motivation TEXT NOT NULL DEFAULT '',
            trace TEXT NOT NULL DEFAULT '',
            analysis TEXT NOT NULL DEFAULT '',
            score REAL NOT NULL DEFAULT 0.0,
            loss_curve TEXT NOT NULL DEFAULT '[]',
            manifest_sha256 TEXT NOT NULL DEFAULT '',
            generated_samples TEXT NOT NULL DEFAULT '[]',
            objectives TEXT NOT NULL DEFAULT '{}',
            timestamp REAL NOT NULL DEFAULT 0.0
        )
    """)
    conn.commit()
    al = AccessLogger(conn)
    prov = ProvenanceQuery(conn)
    return conn, al, prov


def _add_exp(conn, id, code="", hotkey="", round_id=None, metric=None,
             success=1, task="ts", miner_uid=-1):
    conn.execute(
        "INSERT INTO experiments (id, code, miner_hotkey, round_id, metric, "
        "success, task, miner_uid) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (id, code, hotkey, round_id, metric, success, task, miner_uid),
    )
    conn.commit()


# ── Schema ───────────────────────────────────────────


def test_schema_creates_tables():
    conn, _, prov = _make_db()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "round_context" in tables
    assert "code_components" in tables


# ── detect_components ────────────────────────────────


def test_detect_rmsnorm_and_gelu():
    comps = detect_components("self.norm = RMSNorm(dim)\nself.act = GELU()")
    assert "RMSNorm" in comps
    assert "GELU" in comps


def test_detect_no_components():
    assert detect_components("x = 1 + 2") == []


def test_detect_adamw():
    comps = detect_components("optimizer = AdamW(model.parameters())")
    assert "AdamW" in comps


def test_detect_custom_patterns():
    comps = detect_components("MyCustomLayer()", {"MyCustomLayer": r"\bMyCustomLayer\b"})
    assert comps == ["MyCustomLayer"]


# ── compute_similarity ───────────────────────────────


def test_compute_similarity_identical():
    code = "class Model:\n    def forward(self, input_tensor): return self.linear(input_tensor)"
    sim = compute_similarity(code, code)
    assert sim["jaccard"] == 1.0
    assert sim["diff_ratio"] == 1.0
    assert sim["diff_size"] == 0.0


def test_compute_similarity_different():
    sim = compute_similarity(
        "class EncoderModel:\n    def forward(self): return self.encoder()",
        "class DecoderModel:\n    def backward(self): return self.decoder()",
    )
    assert sim["jaccard"] < 1.0
    assert sim["diff_ratio"] < 1.0
    assert sim["diff_size"] > 0.0


def test_compute_similarity_returns_all_signals():
    sim = compute_similarity("class Foo: pass", "class Bar: pass")
    assert "jaccard" in sim
    assert "diff_ratio" in sim
    assert "diff_size" in sim


# ── record_round_context ─────────────────────────────


def test_record_and_query_round_context():
    conn, _, prov = _make_db()
    prov.record_round_context(1, 5, "frontier")
    prov.record_round_context(1, 10, "frontier")
    ids = prov._get_round_context_ids(1)
    assert set(ids) == {5, 10}


# ── record_components ────────────────────────────────


def test_record_and_query_components():
    conn, _, prov = _make_db()
    prov.record_components(0, ["RMSNorm", "GELU"])
    assert set(prov.get_experiment_components(0)) == {"RMSNorm", "GELU"}


def test_record_components_dedup():
    conn, _, prov = _make_db()
    prov.record_components(0, ["RMSNorm", "RMSNorm"])
    assert prov.get_experiment_components(0) == ["RMSNorm"]


def test_get_component_experiments():
    conn, _, prov = _make_db()
    prov.record_components(0, ["RMSNorm"])
    prov.record_components(1, ["RMSNorm", "GELU"])
    prov.record_components(2, ["GELU"])
    assert set(prov.get_component_experiments("RMSNorm")) == {0, 1}


# ── get_influences ───────────────────────────────────


def test_get_influences_from_access_log():
    conn, al, prov = _make_db()
    _add_exp(conn, 0, code="v0")
    _add_exp(conn, 1, code="v1", hotkey="hk1", round_id=1)
    al.set_round(1)
    al.log_access("hk1", 0, "/experiments/0", [0])
    influences = prov.get_influences(1)
    accessed = [i for i in influences if i["evidence_type"] == "accessed"]
    assert any(i["source_id"] == 0 for i in accessed)


def test_get_influences_from_frontier():
    conn, al, prov = _make_db()
    _add_exp(conn, 0, code="v0")
    _add_exp(conn, 1, code="v1", hotkey="hk1", round_id=1)
    prov.record_round_context(1, 0, "frontier")
    influences = prov.get_influences(1)
    frontier = [i for i in influences if i["evidence_type"] == "frontier"]
    assert any(i["source_id"] == 0 for i in frontier)


def test_get_influences_from_shared_components():
    conn, al, prov = _make_db()
    _add_exp(conn, 0, code="v0")
    _add_exp(conn, 1, code="v1", hotkey="hk1", round_id=1)
    prov.record_components(0, ["RMSNorm"])
    prov.record_components(1, ["RMSNorm"])
    influences = prov.get_influences(1)
    shared = [i for i in influences if i["evidence_type"] == "shared_component"]
    assert any(i["source_id"] == 0 for i in shared)


def test_get_influences_empty_for_unknown():
    conn, al, prov = _make_db()
    assert prov.get_influences(999) == []


# ── get_impact ───────────────────────────────────────


def test_get_impact_from_frontier():
    conn, al, prov = _make_db()
    _add_exp(conn, 0, code="v0", round_id=1)
    _add_exp(conn, 1, code="v1", round_id=1)
    prov.record_round_context(1, 0, "frontier")
    impact = prov.get_impact(0)
    frontier_for = [i for i in impact if i["evidence_type"] == "frontier_for"]
    assert any(i["target_id"] == 1 for i in frontier_for)


# ── get_similar ──────────────────────────────────────


def test_get_similar():
    conn, al, prov = _make_db()
    base_code = "class Model:\n    def forward(self, x): return self.linear(x)"
    similar_code = "class Model:\n    def forward(self, x): return self.linear(x) + self.bias"
    different_code = "import numpy\ndef compute(): return numpy.zeros(100)"
    _add_exp(conn, 0, code=base_code)
    _add_exp(conn, 1, code=similar_code)
    _add_exp(conn, 2, code=different_code)
    results = prov.get_similar(0)
    assert len(results) == 2
    # Similar code should rank higher
    assert results[0]["target_id"] == 1
    assert results[0]["jaccard"] > results[1]["jaccard"]


def test_get_similar_with_pool():
    conn, al, prov = _make_db()
    _add_exp(conn, 0, code="class A: pass")
    _add_exp(conn, 1, code="class A: pass")
    _add_exp(conn, 2, code="class B: pass")
    results = prov.get_similar(0, pool=[1])
    assert len(results) == 1
    assert results[0]["target_id"] == 1


# ── get_component_stats ──────────────────────────────


def test_get_component_stats():
    conn, al, prov = _make_db()
    _add_exp(conn, 0, code="v0", metric=0.5, success=1)
    _add_exp(conn, 1, code="v1", metric=0.3, success=1)
    prov.record_components(0, ["RMSNorm"])
    prov.record_components(1, ["RMSNorm", "GELU"])
    stats = prov.get_component_stats()
    rmsnorm = [s for s in stats if s["component"] == "RMSNorm"]
    assert len(rmsnorm) == 1
    assert rmsnorm[0]["count"] == 2


# ── get_dead_ends ────────────────────────────────────


def test_get_dead_ends():
    conn, al, prov = _make_db()
    base = "class Model:\n    def forward(self, x): return self.linear(x)"
    variant = "class Model:\n    def forward(self, x): return self.linear(x) + 1"
    unrelated = "import numpy\ndef run(): return numpy.zeros(100)"
    _add_exp(conn, 0, code=base, metric=0.5, success=1)
    _add_exp(conn, 1, code=variant, metric=0.4, success=1)
    _add_exp(conn, 2, code=unrelated, metric=0.6, success=1)
    dead = prov.get_dead_ends()
    # exp 1 and 2 have no later similar experiments
    assert 1 in dead
    assert 2 in dead


# ── get_experiment_graph ─────────────────────────────


def test_get_experiment_graph():
    conn, al, prov = _make_db()
    _add_exp(conn, 0, code="v0")
    _add_exp(conn, 1, code="v1", hotkey="hk1", round_id=1)
    prov.record_components(1, ["RMSNorm"])
    graph = prov.get_experiment_graph(1)
    assert graph["experiment_id"] == 1
    assert "influences" in graph
    assert "impact" in graph
    assert "components" in graph
    assert "RMSNorm" in graph["components"]


# ── export ───────────────────────────────────────────


def test_export():
    conn, al, prov = _make_db()
    _add_exp(conn, 0, code="v0", metric=0.5, success=1)
    prov.record_components(0, ["RMSNorm"])
    prov.record_round_context(1, 0, "frontier")
    export = prov.export()
    assert len(export["experiments"]) == 1
    assert len(export["components"]) == 1
    assert len(export["round_context"]) == 1
    assert "component_stats" in export
    assert "access_log_count" in export
