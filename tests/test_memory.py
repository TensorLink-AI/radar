"""Tests for the optional ``local.memory`` layer."""

from __future__ import annotations

import os

import pytest

from local.memory import Memory, _hash_embed, _regex_extract


# ── toggle ────────────────────────────────────────────────────


def test_disabled_is_full_noop(tmp_path, monkeypatch):
    """RADAR_MEMORY_ENABLED unset → every method is a silent no-op."""
    monkeypatch.delenv("RADAR_MEMORY_ENABLED", raising=False)
    monkeypatch.setenv("RADAR_MEMORY_DB", str(tmp_path / "m.db"))
    mem = Memory.from_env()
    assert mem.enabled is False
    assert mem.remember("hello world") is None
    assert mem.add_fact("a", "knows", "b") is None
    assert mem.recall("anything") == []
    # No DB file written when disabled.
    assert not (tmp_path / "m.db").exists()


def test_enabled_writes_db_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_MEMORY_ENABLED", "1")
    monkeypatch.setenv("RADAR_MEMORY_DB", str(tmp_path / "m.db"))
    mem = Memory.from_env()
    assert mem.enabled is True
    mem.remember("alice met bob")
    assert (tmp_path / "m.db").exists()
    mem.close()


# ── remember / recall ─────────────────────────────────────────


@pytest.fixture
def mem(tmp_path):
    m = Memory(db_path=str(tmp_path / "m.db"), enabled=True)
    m._open()
    yield m
    m.close()


def test_remember_returns_node_id(mem):
    nid = mem.remember("the cat sat on the mat", ts=100.0)
    assert isinstance(nid, int) and nid > 0


def test_recall_finds_relevant_memory(mem):
    mem.remember("the cat sat on the mat", ts=100.0)
    mem.remember("quantum chromodynamics is a gauge theory", ts=101.0)
    hits = mem.recall("where did the cat sit", k=2)
    assert hits, "expected at least one hit"
    assert "cat" in hits[0].text


def test_recall_empty_when_no_data(mem):
    assert mem.recall("anything") == []


def test_recall_respects_as_of(mem):
    mem.remember("event A", ts=100.0)
    mem.remember("event B", ts=200.0)
    early = mem.recall("event", as_of=150.0, k=5)
    texts = {h.text for h in early}
    assert "event A" in texts
    assert "event B" not in texts


# ── facts + bitemporal supersession ───────────────────────────


def test_add_fact_creates_edges(mem):
    mem.add_fact("alice", "knows", "bob", ts=100.0)
    hits = mem.recall("alice", k=5, as_of=200.0, hops=1)
    edges = [e for h in hits for e in h.edges]
    rels = {e["rel"] for e in edges}
    assert "knows" in rels


def test_supersession_closes_prior_edge(mem):
    mem.add_fact("alice", "lives_in", "paris", ts=100.0)
    mem.add_fact("alice", "lives_in", "paris", ts=200.0)  # re-assertion
    # Live edges at as_of=300 should be 1 (the latest); earlier ones closed.
    rows = list(mem._conn.execute(
        "SELECT COUNT(*) FROM mem_edges WHERE valid_to IS NULL"
    ))
    assert rows[0][0] == 1


def test_extract_triples_from_text(mem):
    mem.remember("recorded: (alice, met, bob)", ts=100.0)
    hits = mem.recall("alice", k=5, hops=1)
    edges = [e for h in hits for e in h.edges]
    assert any(e["rel"] == "met" for e in edges)


# ── extractors / embeds ───────────────────────────────────────


def test_regex_extract_basic():
    out = _regex_extract("noise (a, knows, b) noise (c, eats, d)")
    rels = [t["pred"] for t in out]
    assert rels == ["knows", "eats"]


def test_hash_embed_is_unit_norm():
    v = _hash_embed("hello world")
    import numpy as np
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_hash_embed_is_deterministic():
    assert (_hash_embed("hello world") == _hash_embed("hello world")).all()


# ── services wiring (toggle visible at the HTTP layer) ────────


def test_services_404s_memory_when_disabled(tmp_path, monkeypatch):
    """With memory off, POST /memory/* returns 404 — current behavior."""
    monkeypatch.delenv("RADAR_MEMORY_ENABLED", raising=False)
    from local.services import ServicesServer
    from local.store import LocalStore

    db = tmp_path / "radar.db"
    store = LocalStore(str(db))
    server = ServicesServer(store=store, wiki_dir=None, port=0)
    server.start()
    try:
        import urllib.request, urllib.error, json
        req = urllib.request.Request(
            f"{server.url}/memory/recall",
            data=json.dumps({"query": "x"}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=2)
        assert exc.value.code == 404
    finally:
        server.stop()
        store.close()


def test_services_serves_memory_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_MEMORY_ENABLED", "1")
    monkeypatch.setenv("RADAR_MEMORY_DB", str(tmp_path / "m.db"))
    from local.services import ServicesServer
    from local.store import LocalStore

    db = tmp_path / "radar.db"
    store = LocalStore(str(db))
    server = ServicesServer(store=store, wiki_dir=None, port=0)
    server.start()
    try:
        import urllib.request, json
        for path, body in [
            ("/memory/remember", {"text": "alice met bob"}),
            ("/memory/recall", {"query": "alice", "k": 3}),
        ]:
            req = urllib.request.Request(
                f"{server.url}{path}",
                data=json.dumps(body).encode(),
                headers={"content-type": "application/json",
                         "x-miner-id": "test-miner"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                assert resp.status == 200
                data = json.loads(resp.read())
            if path.endswith("recall"):
                assert "hits" in data
                assert any("alice" in h["text"] for h in data["hits"])
    finally:
        server.stop()
        store.close()
