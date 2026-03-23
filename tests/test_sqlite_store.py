"""Tests for shared.sqlite_store — ported from test_database.py + new diff/FTS tests."""

import sqlite3
import threading

from shared.database import DataElement
from shared.sqlite_store import SQLiteExperimentStore


def _make_store(tmp_path):
    return SQLiteExperimentStore(db_path=str(tmp_path / "test.db"))


# ── Ported from test_database.py (identical assertions) ──────


def test_add_and_get(tmp_path):
    store = _make_store(tmp_path)
    elem = DataElement(name="test1", code="print('hello')", success=True, metric=0.9)
    idx = store.add(elem)
    assert idx == 0
    assert store.size == 1
    got = store.get(0)
    assert got.name == "test1"


def test_lineage(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="root", code="v0", success=True, metric=1.0, parent=None, generation=0))
    store.add(DataElement(name="child", code="v1", success=True, metric=0.9, parent=0, generation=1))
    store.add(DataElement(name="grandchild", code="v2", success=True, metric=0.8, parent=1, generation=2))
    lineage = store.get_lineage(2)
    assert len(lineage) == 3
    assert lineage[0].name == "root"
    assert lineage[2].name == "grandchild"


def test_search(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="e1", code="x", motivation="gated linear attention", analysis="improved throughput"))
    store.add(DataElement(name="e2", code="y", motivation="cosine schedule", analysis="better convergence"))
    store.add(DataElement(name="e3", code="z", motivation="batch size increase", analysis="OOM crash"))
    results = store.search("attention")
    assert len(results) >= 1
    assert results[0].name == "e1"


def test_failures(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ok", code="x", success=True, metric=0.9))
    store.add(DataElement(name="fail1", code="y", success=False))
    store.add(DataElement(name="fail2", code="z", success=False))
    failures = store.get_failures(10)
    assert len(failures) == 2
    assert failures[0].name == "fail2"  # most recent first


def test_search_failures(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ok", code="x", success=True, motivation="attention fix"))
    store.add(DataElement(name="fail", code="y", success=False, motivation="attention crash", analysis="OOM"))
    results = store.search_failures("attention")
    assert len(results) == 1
    assert results[0].name == "fail"


def test_to_api_dict(tmp_path):
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


def test_miner_hotkey_roundtrip(tmp_path):
    store = _make_store(tmp_path)
    elem = DataElement(name="x", code="y", miner_uid=7, miner_hotkey="5Foo")
    store.add(elem)
    got = store.get(0)
    assert got.miner_hotkey == "5Foo"
    assert got.miner_uid == 7


def test_miner_hotkey_defaults_empty():
    elem = DataElement(name="x", code="y")
    assert elem.miner_hotkey == ""


def test_stats(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", success=True, metric=0.9))
    store.add(DataElement(name="b", success=True, metric=0.8))
    store.add(DataElement(name="c", success=False))
    s = store.stats()
    assert s["total"] == 3
    assert s["successful"] == 2
    assert s["failed"] == 1
    assert s["best_metric"] == 0.8


def test_stats_empty(tmp_path):
    store = _make_store(tmp_path)
    s = store.stats()
    assert s["total"] == 0
    assert s["best_metric"] is None
    assert s["mean_metric"] is None
    assert s["max_generation"] == 0


def test_component_stats(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", code="optimizer = AdamW(...)", success=True, metric=0.9))
    store.add(DataElement(name="b", code="norm = RMSNorm(d)", success=True, metric=0.8))

    assert store.get_component_stats() == {}

    ml_patterns = {
        "optimizer": r"(Adam|AdamW|SGD|RMSprop|LAMB|Lion)\b",
        "norm": r"(LayerNorm|RMSNorm|BatchNorm|GroupNorm)\b",
    }
    stats = store.get_component_stats(patterns=ml_patterns)
    assert "optimizer" in stats
    assert "norm" in stats
    assert "AdamW" in stats["optimizer"]
    assert "RMSNorm" in stats["norm"]


def test_count_in_flops_range(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", success=True, metric=0.9, objectives={"flops_equivalent_size": 200_000}))
    store.add(DataElement(name="b", success=True, metric=0.8, objectives={"flops_equivalent_size": 500_000}))
    store.add(DataElement(name="c", success=True, metric=0.7, objectives={"flops_equivalent_size": 2_000_000}))
    assert store.count_in_flops_range(100_000, 600_000) == 2
    assert store.count_in_flops_range(100_000, 100_000_000) == 3
    assert store.count_in_flops_range(1_000_000, 5_000_000) == 1


def test_get_in_flops_range(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", success=True, metric=0.9, objectives={"flops_equivalent_size": 200_000}))
    store.add(DataElement(name="b", success=True, metric=0.8, objectives={"flops_equivalent_size": 2_000_000}))
    results = store.get_in_flops_range(100_000, 500_000)
    assert len(results) == 1
    assert results[0].name == "a"


def test_get_successful(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="bad", code="x", success=False))
    store.add(DataElement(name="ok1", code="y", success=True, metric=0.9))
    store.add(DataElement(name="ok2", code="z", success=True, metric=0.5))
    results = store.get_successful()
    assert len(results) == 2
    assert results[0].metric == 0.5  # sorted ascending


def test_get_best(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", success=True, metric=0.9))
    store.add(DataElement(name="b", success=True, metric=0.5))
    store.add(DataElement(name="c", success=True, metric=0.7))
    best = store.get_best(2)
    assert len(best) == 2
    assert best[0].metric == 0.5
    assert best[1].metric == 0.7


def test_get_recent(tmp_path):
    store = _make_store(tmp_path)
    for i in range(5):
        store.add(DataElement(name=f"exp_{i}", code=f"v{i}"))
    recent = store.get_recent(3)
    assert len(recent) == 3
    assert recent[0].name == "exp_4"  # newest first
    assert recent[2].name == "exp_2"


def test_get_children(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="root", code="v0"))
    store.add(DataElement(name="child1", code="v1", parent=0))
    store.add(DataElement(name="child2", code="v2", parent=0))
    store.add(DataElement(name="other", code="v3", parent=1))
    children = store.get_children(0)
    assert len(children) == 2
    assert {c.name for c in children} == {"child1", "child2"}


def test_get_pareto_elements(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", success=True, metric=0.5))
    store.add(DataElement(name="b", success=False))
    store.add(DataElement(name="c", success=True, metric=None))
    store.add(DataElement(name="d", success=True, metric=0.3))
    elems = store.get_pareto_elements()
    assert len(elems) == 2


def test_add_batch(tmp_path):
    store = _make_store(tmp_path)
    elems = [
        DataElement(name=f"batch_{i}", code=f"v{i}", success=True, metric=float(i))
        for i in range(5)
    ]
    indices = store.add_batch(elems)
    assert indices == [0, 1, 2, 3, 4]
    assert store.size == 5
    # All should share the same timestamp
    timestamps = {store.get(i).timestamp for i in indices}
    assert len(timestamps) == 1


def test_get_nonexistent(tmp_path):
    store = _make_store(tmp_path)
    assert store.get(0) is None
    assert store.get(-1) is None
    assert store.get(999) is None


# ── New diff API tests ──────────────────────────────────


def test_get_diff_with_parent(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(
        name="v1", code="def build_model():\n    return Linear(10, 10)\n",
        success=True, metric=0.5,
    ))
    store.add(DataElement(
        name="v2", code="def build_model():\n    return Linear(10, 20)\n",
        success=True, metric=0.4, parent=0, generation=1,
    ))
    diff = store.get_diff(1)
    assert diff is not None
    assert "Linear(10, 10)" in diff
    assert "Linear(10, 20)" in diff


def test_get_diff_no_parent(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="root", code="x = 1\n", success=True, metric=0.5))
    diff = store.get_diff(0)
    assert diff is not None
    assert "+x = 1" in diff


def test_get_diff_not_found(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_diff(999) is None


def test_get_diff_between(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", code="x = 1\n"))
    store.add(DataElement(name="b", code="x = 2\n"))
    diff = store.get_diff_between(0, 1)
    assert "-x = 1" in diff
    assert "+x = 2" in diff


def test_get_diff_between_not_found(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", code="x = 1\n"))
    assert store.get_diff_between(0, 999) is None
    assert store.get_diff_between(999, 0) is None


def test_get_lineage_diffs(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="gen0", code="v0", success=True, metric=1.0, motivation="baseline"))
    store.add(DataElement(name="gen1", code="v1", success=True, metric=0.9, parent=0, generation=1, motivation="first change"))
    store.add(DataElement(name="gen2", code="v2", success=True, metric=0.8, parent=1, generation=2, motivation="second change"))
    store.add(DataElement(name="gen3", code="v3", success=True, metric=0.7, parent=2, generation=3, motivation="third change"))
    diffs = store.get_lineage_diffs(3)
    assert len(diffs) == 4
    assert diffs[0]["index"] == 0
    assert diffs[0]["motivation"] == "baseline"
    assert diffs[3]["index"] == 3
    # Root: full code as addition
    assert "+v0" in diffs[0]["diff"]
    # Each subsequent step shows the change
    assert "-v0" in diffs[1]["diff"] and "+v1" in diffs[1]["diff"]


def test_get_family_summary(tmp_path):
    store = _make_store(tmp_path)
    # Family 1: root -> child -> grandchild
    store.add(DataElement(name="root_a", code="a", success=True, metric=1.0))
    store.add(DataElement(name="child_a", code="a1", success=True, metric=0.8, parent=0, generation=1))
    store.add(DataElement(name="grand_a", code="a2", success=True, metric=0.6, parent=1, generation=2))
    # Family 2: single root
    store.add(DataElement(name="root_b", code="b", success=True, metric=0.9))
    families = store.get_family_summary()
    assert len(families) == 2
    fam_a = next(f for f in families if f["root_name"] == "root_a")
    assert fam_a["num_descendants"] == 3
    assert fam_a["best_metric"] == 0.6


# ── FTS tests ──────────────────────────────────────────


def test_search_fts(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="e1", code="x", motivation="gated linear attention", analysis="improved throughput"))
    store.add(DataElement(name="e2", code="y", motivation="cosine schedule", analysis="better convergence"))
    results = store.search("attention")
    assert len(results) >= 1
    assert results[0].name == "e1"


def test_search_fts_partial_words(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="e1", code="x", motivation="multi-head self-attention mechanism"))
    # FTS5 tokenizes on punctuation, "self-attention" -> "self", "attention"
    results = store.search("attention")
    assert len(results) >= 1


def test_search_empty_query(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="e1", code="x", motivation="test"))
    results = store.search("")
    assert results == []


def test_search_name_field(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="transformer_v1", code="x", motivation="baseline"))
    store.add(DataElement(name="lstm_v1", code="y", motivation="baseline"))
    results = store.search("transformer")
    assert len(results) >= 1
    assert results[0].name == "transformer_v1"


# ── Concurrency test ──────────────────────────────────


def test_concurrent_reads_and_writes(tmp_path):
    store = _make_store(tmp_path)
    for i in range(10):
        store.add(DataElement(name=f"exp_{i}", code=f"v{i}", success=True, metric=float(i)))

    errors = []

    def reader():
        try:
            for _ in range(50):
                store.get_successful()
                store.stats()
        except Exception as e:
            errors.append(e)

    def writer():
        try:
            for i in range(10, 20):
                store.add(DataElement(name=f"exp_{i}", code=f"v{i}", success=True, metric=float(i)))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(3)] + [threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Concurrent access errors: {errors}"


# ── Round-trip and persistence test ──────────────────────


def test_persistence(tmp_path):
    db_path = str(tmp_path / "persist.db")
    store = SQLiteExperimentStore(db_path=db_path)
    store.add(DataElement(name="persist_test", code="x = 1", success=True, metric=0.5))
    store.close()

    store2 = SQLiteExperimentStore(db_path=db_path)
    assert store2.size == 1
    assert store2.get(0).name == "persist_test"
    store2.close()


def test_objectives_roundtrip(tmp_path):
    store = _make_store(tmp_path)
    obj = {"flops_equivalent_size": 200_000, "crps": 0.42, "passed_size_gate": True}
    store.add(DataElement(name="a", code="x", success=True, metric=0.42, objectives=obj))
    got = store.get(0)
    assert got.objectives == obj
    assert got.objectives["flops_equivalent_size"] == 200_000


def test_loss_curve_roundtrip(tmp_path):
    store = _make_store(tmp_path)
    curve = [1.0, 0.95, 0.9, 0.85]
    store.add(DataElement(name="a", code="x", loss_curve=curve))
    got = store.get(0)
    assert got.loss_curve == curve


def test_generated_samples_roundtrip(tmp_path):
    store = _make_store(tmp_path)
    samples = [{"x": 1}, {"x": 2}]
    store.add(DataElement(name="a", code="x", generated_samples=samples))
    got = store.get(0)
    assert got.generated_samples == samples


# ── Multi-task isolation tests ──────────────────────────


def test_task_field_stored_and_retrieved(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="exp1", code="x", task="ts_forecasting", success=True, metric=0.5))
    elem = store.get(0)
    assert elem.task == "ts_forecasting"


def test_empty_task_default(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="old", code="x", success=True, metric=0.5))
    elem = store.get(0)
    assert elem.task == ""


def test_get_successful_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting", success=True, metric=0.5))
    store.add(DataElement(name="lm1", code="b", task="nanogpt", success=True, metric=1.2))
    store.add(DataElement(name="ts2", code="c", task="ts_forecasting", success=True, metric=0.3))

    ts_results = store.get_successful(task="ts_forecasting")
    assert len(ts_results) == 2
    assert all(r.task == "ts_forecasting" for r in ts_results)

    all_results = store.get_successful()
    assert len(all_results) == 3


def test_get_best_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting", success=True, metric=0.5))
    store.add(DataElement(name="lm1", code="b", task="nanogpt", success=True, metric=0.1))
    store.add(DataElement(name="ts2", code="c", task="ts_forecasting", success=True, metric=0.3))

    best_ts = store.get_best(n=1, task="ts_forecasting")
    assert len(best_ts) == 1
    assert best_ts[0].name == "ts2"
    assert best_ts[0].task == "ts_forecasting"


def test_get_recent_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting"))
    store.add(DataElement(name="lm1", code="b", task="nanogpt"))
    store.add(DataElement(name="ts2", code="c", task="ts_forecasting"))

    recent_ts = store.get_recent(n=5, task="ts_forecasting")
    assert len(recent_ts) == 2
    assert recent_ts[0].name == "ts2"


def test_get_failures_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts_ok", code="a", task="ts_forecasting", success=True, metric=0.5))
    store.add(DataElement(name="ts_fail", code="b", task="ts_forecasting", success=False))
    store.add(DataElement(name="lm_fail", code="c", task="nanogpt", success=False))

    ts_failures = store.get_failures(n=10, task="ts_forecasting")
    assert len(ts_failures) == 1
    assert ts_failures[0].name == "ts_fail"


def test_search_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting",
                          motivation="attention mechanism for time series"))
    store.add(DataElement(name="lm1", code="b", task="nanogpt",
                          motivation="attention mechanism for language"))

    ts_results = store.search("attention", task="ts_forecasting")
    assert len(ts_results) == 1
    assert ts_results[0].task == "ts_forecasting"

    all_results = store.search("attention")
    assert len(all_results) == 2


def test_get_pareto_elements_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting", success=True, metric=0.5))
    store.add(DataElement(name="lm1", code="b", task="nanogpt", success=True, metric=0.3))

    ts_pareto = store.get_pareto_elements(task="ts_forecasting")
    assert len(ts_pareto) == 1
    assert ts_pareto[0].task == "ts_forecasting"


def test_count_in_flops_range_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting", success=True, metric=0.5,
                          objectives={"flops_equivalent_size": 200_000}))
    store.add(DataElement(name="lm1", code="b", task="nanogpt", success=True, metric=0.3,
                          objectives={"flops_equivalent_size": 300_000}))

    assert store.count_in_flops_range(100_000, 500_000, task="ts_forecasting") == 1
    assert store.count_in_flops_range(100_000, 500_000) == 2


def test_stats_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting", success=True, metric=0.5))
    store.add(DataElement(name="ts2", code="b", task="ts_forecasting", success=False))
    store.add(DataElement(name="lm1", code="c", task="nanogpt", success=True, metric=0.3))

    ts_stats = store.stats(task="ts_forecasting")
    assert ts_stats["total"] == 2
    assert ts_stats["successful"] == 1
    assert ts_stats["best_metric"] == 0.5

    all_stats = store.stats()
    assert all_stats["total"] == 3


def test_get_tasks(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", code="x", task="ts_forecasting"))
    store.add(DataElement(name="b", code="y", task="nanogpt"))
    store.add(DataElement(name="c", code="z", task="ts_forecasting"))

    tasks = store.get_tasks()
    assert tasks == ["nanogpt", "ts_forecasting"]


def test_stats_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting", success=True, metric=0.5))
    store.add(DataElement(name="lm1", code="b", task="nanogpt", success=True, metric=0.3))

    by_task = store.stats_by_task()
    assert "ts_forecasting" in by_task
    assert "nanogpt" in by_task
    assert by_task["ts_forecasting"]["total"] == 1
    assert by_task["nanogpt"]["best_metric"] == 0.3


def test_get_family_summary_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts_root", code="a", task="ts_forecasting", success=True, metric=1.0))
    store.add(DataElement(name="ts_child", code="b", task="ts_forecasting", success=True, metric=0.8,
                          parent=0, generation=1))
    store.add(DataElement(name="lm_root", code="c", task="nanogpt", success=True, metric=2.0))

    ts_families = store.get_family_summary(task="ts_forecasting")
    assert len(ts_families) == 1
    assert ts_families[0]["root_name"] == "ts_root"
    assert ts_families[0]["num_descendants"] == 2

    all_families = store.get_family_summary()
    assert len(all_families) == 2


def test_diff_works_within_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="v1", code="def f():\n    return 1\n",
                          task="ts_forecasting", success=True, metric=0.5))
    store.add(DataElement(name="v2", code="def f():\n    return 2\n",
                          task="ts_forecasting", success=True, metric=0.4, parent=0, generation=1))
    diff = store.get_diff(1)
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_cross_task_parent_lineage(tmp_path):
    """Lineage follows parent chain regardless of task."""
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts_root", code="a", task="ts_forecasting", success=True, metric=0.5))
    store.add(DataElement(name="ts_child", code="b", task="ts_forecasting", success=True, metric=0.4,
                          parent=0, generation=1))
    lineage = store.get_lineage(1)
    assert len(lineage) == 2
    assert all(e.task == "ts_forecasting" for e in lineage)


def test_search_failures_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts_fail", code="a", task="ts_forecasting", success=False,
                          motivation="attention bug"))
    store.add(DataElement(name="lm_fail", code="b", task="nanogpt", success=False,
                          motivation="attention overflow"))

    ts_results = store.search_failures("attention", task="ts_forecasting")
    assert len(ts_results) == 1
    assert ts_results[0].name == "ts_fail"


def test_component_stats_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="a", code="optimizer = AdamW(...)",
                          task="ts_forecasting", success=True, metric=0.9))
    store.add(DataElement(name="b", code="optimizer = SGD(...)",
                          task="nanogpt", success=True, metric=0.8))

    patterns = {"optimizer": r"(AdamW|SGD)\b"}
    ts_stats = store.get_component_stats(patterns=patterns, task="ts_forecasting")
    assert "AdamW" in ts_stats["optimizer"]
    assert "SGD" not in ts_stats.get("optimizer", {})


def test_get_in_flops_range_filters_by_task(tmp_path):
    store = _make_store(tmp_path)
    store.add(DataElement(name="ts1", code="a", task="ts_forecasting", success=True, metric=0.5,
                          objectives={"flops_equivalent_size": 200_000}))
    store.add(DataElement(name="lm1", code="b", task="nanogpt", success=True, metric=0.3,
                          objectives={"flops_equivalent_size": 300_000}))

    ts_results = store.get_in_flops_range(100_000, 500_000, task="ts_forecasting")
    assert len(ts_results) == 1
    assert ts_results[0].name == "ts1"


# ── Migration test ────────────────────────────────────


def test_migrate_add_task_column(tmp_path):
    """Create a DB without the task column, reopen with new code, verify migration."""
    db_path = str(tmp_path / "legacy.db")
    # Build a legacy schema without the task column
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            code TEXT NOT NULL DEFAULT '',
            motivation TEXT NOT NULL DEFAULT '',
            trace TEXT NOT NULL DEFAULT '',
            metric REAL,
            success BOOLEAN NOT NULL DEFAULT 0,
            analysis TEXT NOT NULL DEFAULT '',
            parent_index INTEGER,
            generation INTEGER NOT NULL DEFAULT 0,
            score REAL NOT NULL DEFAULT 0.0,
            miner_uid INTEGER NOT NULL DEFAULT -1,
            miner_hotkey TEXT NOT NULL DEFAULT '',
            loss_curve TEXT NOT NULL DEFAULT '[]',
            manifest_sha256 TEXT NOT NULL DEFAULT '',
            generated_samples TEXT NOT NULL DEFAULT '[]',
            objectives TEXT NOT NULL DEFAULT '{}',
            timestamp REAL NOT NULL DEFAULT 0.0,
            round_id INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO experiments (id, name, code, success, metric, timestamp) "
        "VALUES (0, 'old_exp', 'x = 1', 1, 0.5, 100.0)"
    )
    conn.commit()
    conn.close()

    # Reopen with SQLiteExperimentStore — should migrate
    store = SQLiteExperimentStore(db_path=db_path)
    assert store.size == 1
    elem = store.get(0)
    assert elem.name == "old_exp"
    assert elem.task == ""  # default from migration

    # New inserts should work with task
    store.add(DataElement(name="new_exp", code="y", task="ts_forecasting", success=True, metric=0.3))
    assert store.size == 2
    assert store.get(1).task == "ts_forecasting"
    store.close()


# ── FTS5 injection safety test ────────────────────────


def test_search_fts5_special_syntax(tmp_path):
    """FTS5 operator characters in queries should not cause errors."""
    store = _make_store(tmp_path)
    store.add(DataElement(name="e1", code="x", motivation="test model"))

    # These would crash with unquoted FTS5 input
    for query in ["test AND", "test*", "NOT test", 'NEAR(test model)', '"phrase"']:
        results = store.search(query)
        # Should not raise — results may or may not match, that's fine
        assert isinstance(results, list)
