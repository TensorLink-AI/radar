"""Tests for shared.database — add, search, lineage, failures."""

import tempfile
import os

from shared.database import DataElement, ExperimentDB


def _make_db():
    d = tempfile.mkdtemp()
    return ExperimentDB(db_dir=d)


def test_add_and_get():
    db = _make_db()
    elem = DataElement(name="test1", code="print('hello')", success=True, metric=0.9)
    idx = db.add(elem)
    assert idx == 0
    assert db.size == 1
    got = db.get(0)
    assert got.name == "test1"


def test_lineage():
    db = _make_db()
    db.add(DataElement(name="root", code="v0", success=True, metric=1.0, parent=None, generation=0))
    db.add(DataElement(name="child", code="v1", success=True, metric=0.9, parent=0, generation=1))
    db.add(DataElement(name="grandchild", code="v2", success=True, metric=0.8, parent=1, generation=2))
    lineage = db.get_lineage(2)
    assert len(lineage) == 3
    assert lineage[0].name == "root"
    assert lineage[2].name == "grandchild"


def test_search():
    db = _make_db()
    db.add(DataElement(name="e1", code="x", motivation="gated linear attention", analysis="improved throughput"))
    db.add(DataElement(name="e2", code="y", motivation="cosine schedule", analysis="better convergence"))
    db.add(DataElement(name="e3", code="z", motivation="batch size increase", analysis="OOM crash"))
    results = db.search("attention")
    assert len(results) >= 1
    assert results[0].name == "e1"


def test_failures():
    db = _make_db()
    db.add(DataElement(name="ok", code="x", success=True, metric=0.9))
    db.add(DataElement(name="fail1", code="y", success=False))
    db.add(DataElement(name="fail2", code="z", success=False))
    failures = db.get_failures(10)
    assert len(failures) == 2
    assert failures[0].name == "fail2"  # most recent first


def test_search_failures():
    db = _make_db()
    db.add(DataElement(name="ok", code="x", success=True, motivation="attention fix"))
    db.add(DataElement(name="fail", code="y", success=False, motivation="attention crash", analysis="OOM"))
    results = db.search_failures("attention")
    assert len(results) == 1
    assert results[0].name == "fail"


def test_to_api_dict():
    elem = DataElement(
        index=5, name="test", code="x", success=True, metric=0.9,
        objectives={"val_bpb": 0.9, "exec_time": 200},
        loss_curve=[1.0, 0.95, 0.9], analysis="good", score=0.75,
        parent=3, generation=2, miner_uid=42, miner_hotkey="5Grw..abc",
    )
    d = elem.to_api_dict()
    assert d["index"] == 5
    assert d["results"]["success"] is True
    assert d["results"]["metric"] == 0.9
    assert d["results"]["loss_curve"] == [1.0, 0.95, 0.9]
    assert d["analysis"] == "good"
    assert d["parent_index"] == 3
    assert d["miner_uid"] == 42
    assert d["miner_hotkey"] == "5Grw..abc"


def test_miner_hotkey_roundtrip():
    """miner_hotkey survives to_dict / from_dict serialization."""
    elem = DataElement(name="x", code="y", miner_uid=7, miner_hotkey="5Foo")
    d = elem.to_dict()
    assert d["miner_hotkey"] == "5Foo"
    restored = DataElement.from_dict(d)
    assert restored.miner_hotkey == "5Foo"
    assert restored.miner_uid == 7


def test_miner_hotkey_defaults_empty():
    elem = DataElement(name="x", code="y")
    assert elem.miner_hotkey == ""


def test_stats():
    db = _make_db()
    db.add(DataElement(name="a", success=True, metric=0.9))
    db.add(DataElement(name="b", success=True, metric=0.8))
    db.add(DataElement(name="c", success=False))
    s = db.stats()
    assert s["total"] == 3
    assert s["successful"] == 2
    assert s["failed"] == 1
    assert s["best_metric"] == 0.8


def test_component_stats():
    db = _make_db()
    db.add(DataElement(name="a", code="optimizer = AdamW(...)", success=True, metric=0.9))
    db.add(DataElement(name="b", code="norm = RMSNorm(d)", success=True, metric=0.8))

    # Without patterns, returns empty
    assert db.get_component_stats() == {}

    # With task-specific patterns
    ml_patterns = {
        "optimizer": r"(Adam|AdamW|SGD|RMSprop|LAMB|Lion)\b",
        "norm": r"(LayerNorm|RMSNorm|BatchNorm|GroupNorm)\b",
    }
    stats = db.get_component_stats(patterns=ml_patterns)
    assert "optimizer" in stats
    assert "norm" in stats
    assert "AdamW" in stats["optimizer"]
    assert "RMSNorm" in stats["norm"]


# ── New methods for Phase C ──


def test_count_in_flops_range():
    db = _make_db()
    db.add(DataElement(name="a", success=True, metric=0.9, objectives={"flops_equivalent_size": 200_000}))
    db.add(DataElement(name="b", success=True, metric=0.8, objectives={"flops_equivalent_size": 500_000}))
    db.add(DataElement(name="c", success=True, metric=0.7, objectives={"flops_equivalent_size": 2_000_000}))
    assert db.count_in_flops_range(100_000, 600_000) == 2
    assert db.count_in_flops_range(100_000, 100_000_000) == 3
    assert db.count_in_flops_range(1_000_000, 5_000_000) == 1


def test_get_in_flops_range():
    db = _make_db()
    db.add(DataElement(name="a", success=True, metric=0.9, objectives={"flops_equivalent_size": 200_000}))
    db.add(DataElement(name="b", success=True, metric=0.8, objectives={"flops_equivalent_size": 2_000_000}))
    results = db.get_in_flops_range(100_000, 500_000)
    assert len(results) == 1
    assert results[0].name == "a"


# ── Task field tests ──


def test_task_field_roundtrip():
    """task field survives to_dict/from_dict."""
    elem = DataElement(name="x", code="y", task="ts_forecasting")
    d = elem.to_dict()
    assert d["task"] == "ts_forecasting"
    restored = DataElement.from_dict(d)
    assert restored.task == "ts_forecasting"


def test_task_in_api_dict():
    """task appears in to_api_dict output."""
    elem = DataElement(name="x", code="y", task="nanogpt", index=0)
    d = elem.to_api_dict()
    assert d["task"] == "nanogpt"


def test_task_defaults_empty():
    """Backward compat — task defaults to empty string."""
    elem = DataElement(name="x", code="y")
    assert elem.task == ""


