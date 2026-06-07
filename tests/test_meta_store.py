"""Tests for meta.store — registry + per-experiment SQLite isolation."""

from __future__ import annotations

import json

import pytest

from meta.store import ExperimentStore, Registry, experiment_dir


def test_registry_upsert_and_status(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    try:
        reg.upsert_experiment(
            "exp-1", path=str(tmp_path / "exp-1"),
            spec_yaml="name: exp-1\n", budget_usd=10.0,
        )
        rows = reg.list_experiments()
        assert len(rows) == 1
        assert rows[0]["name"] == "exp-1"
        assert rows[0]["status"] == "queued"

        reg.set_status("exp-1", "running")
        assert reg.get_experiment("exp-1")["status"] == "running"

        spent = reg.add_spend("exp-1", 2.5)
        assert spent == pytest.approx(2.5)
        spent = reg.add_spend("exp-1", 1.5)
        assert spent == pytest.approx(4.0)
    finally:
        reg.close()


def test_experiment_store_units_and_events(tmp_path):
    store = ExperimentStore(tmp_path / "meta.db")
    try:
        store.insert_unit("a", "orch-default-exp-a", {"task": "x"})
        store.insert_unit("b", "orch-default-exp-b", {"task": "y"})
        # Idempotent insert
        store.insert_unit("a", "orch-default-exp-a", {"task": "x"})

        queued = store.list_units(status="queued")
        assert {u["id"] for u in queued} == {"a", "b"}

        store.update_unit("a", status="running", pod_id="p-1",
                          ssh_host="1.2.3.4", hourly_usd=0.25)
        a = store.get_unit("a")
        assert a["status"] == "running"
        assert a["pod_id"] == "p-1"
        assert a["hourly_usd"] == 0.25

        store.log_event(
            "pod_created", unit_id="a", payload={"pod_id": "p-1"},
        )
        events = store.recent_events()
        assert events[0]["kind"] == "pod_created"
        assert json.loads(events[0]["payload_json"])["pod_id"] == "p-1"
    finally:
        store.close()


def test_frontier_upsert_idempotent(tmp_path):
    store = ExperimentStore(tmp_path / "meta.db")
    try:
        rows = [
            {
                "source_exp_id": 1, "round_id": 1, "miner_id": "miner-00",
                "name": "first", "metric": 0.5, "success": 1,
                "objectives": {"crps": 0.4, "mase": 0.6},
                "task": "ts_forecasting", "mode": "new",
                "cumulative_compute": 100.0,
            },
            {
                "source_exp_id": 2, "round_id": 2, "miner_id": "miner-00",
                "name": "second", "metric": 0.4, "success": 1,
                "objectives": {}, "task": "ts_forecasting", "mode": "new",
                "cumulative_compute": 200.0,
            },
        ]
        n = store.upsert_frontier_rows("a", rows)
        assert n == 2

        # Re-ingest with one row updated → row count unchanged, metric updated.
        rows[0]["metric"] = 0.3
        store.upsert_frontier_rows("a", rows[:1])
        best = list(store.best_per_unit())
        assert len(best) == 1
        assert best[0]["unit_id"] == "a"
        # min(0.3, 0.4) = 0.3
        assert best[0]["best_metric"] == pytest.approx(0.3)
        assert best[0]["n_rows"] == 2
        assert best[0]["max_round"] == 2
    finally:
        store.close()


def test_per_experiment_isolation(tmp_path):
    """Dropping one experiment's meta.db must not touch the other."""
    a = ExperimentStore(experiment_dir(tmp_path, "exp-a") / "meta.db")
    b = ExperimentStore(experiment_dir(tmp_path, "exp-b") / "meta.db")
    try:
        a.insert_unit("u1", "iid-a", {})
        b.insert_unit("u1", "iid-b", {})
        assert a.get_unit("u1")["instance_id"] == "iid-a"
        assert b.get_unit("u1")["instance_id"] == "iid-b"
        # Distinct on-disk paths.
        assert a.path != b.path
    finally:
        a.close()
        b.close()
