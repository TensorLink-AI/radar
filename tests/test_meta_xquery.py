"""Cross-experiment xquery — verify the digest fold over multiple
``meta.db`` files and that the registry-driven open_all() works."""

from __future__ import annotations

import pytest

from meta import xquery
from meta.store import ExperimentStore, Registry, experiment_dir


def _seed(root, exp_name, units_metrics):
    edir = experiment_dir(root, exp_name)
    edir.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(edir / "meta.db")
    try:
        for uid, metric in units_metrics:
            store.insert_unit(uid, f"orch-{exp_name}-{uid}", {})
            store.upsert_frontier_rows(uid, [{
                "source_exp_id": 1,
                "round_id": 1,
                "miner_id": "m",
                "metric": metric,
                "success": 1 if metric is not None else 0,
                "objectives": {},
                "task": "ts_forecasting",
                "mode": "new",
                "cumulative_compute": 100.0,
            }])
    finally:
        store.close()


def test_xcompare_digest_aggregates_across_experiments(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    try:
        reg.upsert_experiment("a", str(experiment_dir(tmp_path, "a")),
                              "", 10.0)
        reg.upsert_experiment("b", str(experiment_dir(tmp_path, "b")),
                              "", 10.0)
    finally:
        reg.close()
    _seed(tmp_path, "a", [("u1", 0.5), ("u2", 0.3)])
    _seed(tmp_path, "b", [("u1", 0.4), ("u2", None)])

    digest = xquery.xcompare_digest(tmp_path)
    assert digest["n_experiments"] == 2
    assert {e["name"] for e in digest["experiments"]} == {"a", "b"}
    # u1+u2 from each experiment = 4 unit-summaries.
    assert digest["n_units"] == 4
    assert digest["n_units_with_results"] == 3  # 0.5, 0.3, 0.4
    # Top is the smallest metric — 0.3 from experiment a.
    top = digest["top_5"][0]
    assert top["experiment"] == "a"
    assert top["unit_id"] == "u2"
    assert top["metric"] == pytest.approx(0.3)


def test_xcompare_handles_no_experiments(tmp_path):
    digest = xquery.xcompare_digest(tmp_path)
    assert digest["n_experiments"] == 0
    assert digest["n_units"] == 0
    assert digest["top_5"] == []
