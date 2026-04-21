"""Tests for the read-only dashboard under database/dashboard.

Uses a mock PgExperimentStore, a fake pool, and MockR2 so we exercise the
full HTTP path without a real Postgres or R2.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

# Flip dashboard env before Config is imported. Because other tests in the
# suite may have already imported Config, we also overwrite the class
# attributes directly below.
os.environ["RADAR_DASHBOARD_ENABLED"] = "true"
os.environ["RADAR_DASHBOARD_KEY"] = "testkey"

from config import Config  # noqa: E402
Config.DASHBOARD_ENABLED = True
Config.DASHBOARD_KEY = "testkey"
Config.DASHBOARD_PAGE_SIZE = 10
Config.DASHBOARD_SESSION_TTL = 60

from shared.database import DataElement  # noqa: E402
from database.server import app, set_db  # noqa: E402

# ── Fakes ──────────────────────────────────────────────────────


class FakeStore:
    def __init__(self, elements: list[DataElement]):
        self._elements = elements
        self.provenance = None

    async def stats(self, **kw):
        total = len(self._elements)
        successful = sum(1 for e in self._elements if e.success)
        metrics = [e.metric for e in self._elements if e.success and e.metric is not None]
        return {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "best_metric": min(metrics) if metrics else None,
            "worst_metric": max(metrics) if metrics else None,
            "mean_metric": sum(metrics) / len(metrics) if metrics else None,
            "max_generation": max((e.generation for e in self._elements), default=0),
        }

    async def stats_by_task(self):
        tasks = sorted({e.task for e in self._elements if e.task})
        return {t: await self.stats() for t in tasks}

    async def get_recent(self, n=20, **kw):
        return list(reversed(self._elements[-n:]))

    async def get_tasks(self):
        return sorted({e.task for e in self._elements if e.task})

    async def get(self, index):
        for e in self._elements:
            if e.index == index:
                return e
        return None

    async def get_diff(self, index):
        elem = await self.get(index)
        if elem is None:
            return None
        return f"--- /dev/null\n+++ #{index}\n+{elem.code}"

    async def get_lineage(self, index):
        lineage = []
        current = await self.get(index)
        while current is not None:
            lineage.append(current)
            current = await self.get(current.parent) if current.parent is not None else None
        return list(reversed(lineage))

    async def get_lineage_diffs(self, index):
        lineage = await self.get_lineage(index)
        return [
            {"index": e.index, "name": e.name, "metric": e.metric,
             "motivation": e.motivation,
             "diff": f"diff@{e.index}"}
            for e in lineage
        ]

    async def get_pareto_elements(self, **kw):
        return [e for e in self._elements if e.success and e.metric is not None]


class FakePool:
    """Minimal asyncpg-like pool with pre-canned query responses."""

    def __init__(
        self,
        elements: list[DataElement],
        access_log: Optional[list[dict]] = None,
    ):
        self.elements = elements
        self.access_log = access_log or []

    def _rows(self):
        return [self._row(e) for e in self.elements]

    @staticmethod
    def _row(e: DataElement) -> dict:
        return {
            "id": e.index,
            "name": e.name,
            "code": e.code,
            "motivation": e.motivation,
            "trace": e.trace,
            "metric": e.metric,
            "success": e.success,
            "analysis": e.analysis,
            "parent_index": e.parent,
            "generation": e.generation,
            "score": e.score,
            "miner_uid": e.miner_uid,
            "miner_hotkey": e.miner_hotkey,
            "loss_curve": json.dumps(e.loss_curve),
            "manifest_sha256": e.manifest_sha256,
            "generated_samples": json.dumps(e.generated_samples),
            "objectives": json.dumps(e.objectives),
            "timestamp": e.timestamp,
            "round_id": e.round_id if e.round_id >= 0 else None,
            "task": e.task,
        }

    async def fetchval(self, sql: str, *params) -> Any:
        sql_l = sql.lower()
        if "count(*)" in sql_l and "experiments" in sql_l:
            # Apply any trailing WHERE by ignoring filters — tests that care
            # about filtered counts inspect the items list.
            return len(self.elements)
        if "loss_curve" in sql_l:
            return None
        return 0

    async def fetch(self, sql: str, *params):
        sql_l = sql.lower()
        # Provenance: miner × round activity
        if "count(distinct ref)" in sql_l and "group by hotkey, round_id" in sql_l:
            rounds_limit = params[0] if params else 30
            distinct = sorted(
                {a["round_id"] for a in self.access_log if a["round_id"] >= 0},
                reverse=True,
            )[:rounds_limit]
            round_set = set(distinct)
            agg: dict = {}
            for a in self.access_log:
                if a["round_id"] not in round_set:
                    continue
                key = (a["hotkey"], a["round_id"])
                agg.setdefault(key, set()).update(a["experiment_ids"])
            return [
                {"hotkey": hk, "round_id": rid, "unique_queried": len(eids)}
                for (hk, rid), eids in agg.items()
            ]
        # Provenance: top experiment counts
        if "group by exp_id" in sql_l and "order by cnt desc" in sql_l:
            top_k = params[0] if params else 20
            counts: dict = {}
            for a in self.access_log:
                for eid in a["experiment_ids"]:
                    counts[eid] = counts.get(eid, 0) + 1
            sorted_items = sorted(counts.items(), key=lambda x: -x[1])[:top_k]
            return [{"exp_id": eid, "cnt": cnt} for eid, cnt in sorted_items]
        # Provenance: names lookup for top experiment ids
        if (
            "select id, name from experiments" in sql_l
            and "= any" in sql_l
        ):
            ids = set(params[0]) if params else set()
            return [
                {"id": e.index, "name": e.name}
                for e in self.elements if e.index in ids
            ]
        # Provenance: miner × experiment matrix
        if "group by hotkey, exp_id" in sql_l:
            ids = set(params[0]) if params else set()
            agg2: dict = {}
            for a in self.access_log:
                for eid in a["experiment_ids"]:
                    if eid not in ids:
                        continue
                    key = (a["hotkey"], eid)
                    agg2[key] = agg2.get(key, 0) + 1
            return [
                {"hotkey": hk, "exp_id": eid, "cnt": cnt}
                for (hk, eid), cnt in agg2.items()
            ]
        if "distinct round_id" in sql_l:
            rows = sorted({e.round_id for e in self.elements if e.round_id >= 0}, reverse=True)
            return [{"round_id": r} for r in rows]
        if "distinct miner_hotkey" in sql_l:
            keys = sorted({e.miner_hotkey for e in self.elements if e.miner_hotkey})
            return [{"miner_hotkey": k} for k in keys]
        if "group by miner_hotkey" in sql_l:
            agg: dict[str, dict] = {}
            for e in self.elements:
                if not e.miner_hotkey:
                    continue
                row = agg.setdefault(e.miner_hotkey, {
                    "miner_hotkey": e.miner_hotkey,
                    "total": 0, "successes": 0,
                    "best_metric": None, "last_seen": 0.0, "last_uid": e.miner_uid,
                })
                row["total"] += 1
                if e.success:
                    row["successes"] += 1
                    if e.metric is not None:
                        row["best_metric"] = (
                            e.metric if row["best_metric"] is None
                            else min(row["best_metric"], e.metric)
                        )
                row["last_seen"] = max(row["last_seen"], e.timestamp or 0.0)
            return list(agg.values())
        if "where miner_hotkey = $1" in sql_l:
            hotkey = params[0]
            rows = [self._row(e) for e in self.elements if e.miner_hotkey == hotkey]
            rows.sort(key=lambda r: r["id"], reverse=True)
            return rows
        if "from experiments" in sql_l and "order by id desc" in sql_l:
            # Browse query
            rows = sorted(self._rows(), key=lambda r: r["id"], reverse=True)
            # Tail params are (limit, offset)
            if len(params) >= 2:
                limit, offset = params[-2], params[-1]
                return rows[offset : offset + limit]
            return rows
        return []

    async def fetchrow(self, sql: str, *params):
        sql_l = sql.lower()
        if "loss_curve" in sql_l and "where id" in sql_l:
            for e in self.elements:
                if e.index == params[0]:
                    return {"loss_curve": json.dumps(e.loss_curve)}
            return None
        return None


class FakeR2:
    """In-memory stand-in for R2AuditLog used by logs tests."""

    def __init__(self):
        self._blobs: dict[str, bytes] = {}

    def upload_json(self, key, data):
        self._blobs[key] = json.dumps(data).encode()
        return True

    def upload_text(self, key, text):
        self._blobs[key] = text.encode()
        return True

    def download_json(self, key):
        if key not in self._blobs:
            return None
        return json.loads(self._blobs[key])

    def download_text(self, key):
        if key not in self._blobs:
            return None
        return self._blobs[key].decode()

    def generate_presigned_get_url(self, key, ttl=900):
        return f"https://fake-r2.example/{key}?ttl={ttl}"


# ── Fixtures ───────────────────────────────────────────────────


def _sample_elements() -> list[DataElement]:
    return [
        DataElement(
            index=0, name="root", code="print(0)", success=True, metric=1.2,
            task="ts", miner_hotkey="hk_a", miner_uid=1, generation=0,
            round_id=7, timestamp=1000.0,
            objectives={"flops_equivalent_size": 500_000},
            loss_curve=[2.0, 1.5, 1.2],
        ),
        DataElement(
            index=1, name="child", code="print(1)", success=True, metric=0.9,
            parent=0, task="ts", miner_hotkey="hk_a", miner_uid=1, generation=1,
            round_id=8, timestamp=2000.0,
            objectives={"flops_equivalent_size": 800_000},
            loss_curve=[1.5, 1.1, 0.9],
        ),
        DataElement(
            index=2, name="other", code="print(2)", success=False,
            task="ts", miner_hotkey="hk_b", miner_uid=2, generation=0,
            round_id=8, timestamp=3000.0,
            objectives={"flops_equivalent_size": 1_000_000},
        ),
    ]


@pytest.fixture(scope="module")
def dashboard_client():
    """Mount the dashboard once and reuse the client across tests."""
    elements = _sample_elements()
    store = FakeStore(elements)
    access_log = [
        {"hotkey": "hk_a", "round_id": 8, "experiment_ids": [0, 1]},
        {"hotkey": "hk_a", "round_id": 8, "experiment_ids": [1]},
        {"hotkey": "hk_b", "round_id": 8, "experiment_ids": [0]},
        {"hotkey": "hk_a", "round_id": 7, "experiment_ids": [0]},
    ]
    pool = FakePool(elements, access_log=access_log)
    r2 = FakeR2()
    # Seed a training log file so the log route has something to read.
    r2.upload_text("round_7/miner_hk_a/stdout.log", "epoch 0 loss=2.0\nepoch 1 loss=1.5\n")
    r2.upload_json("round_7/miner_hk_a/training_meta.json", {
        "flops": 1234, "ok": True,
        "train_loss_history": [
            {"step": 10, "loss": 22.19}, {"step": 20, "loss": 14.71},
        ],
        "val_loss_history": [
            {"step": 10, "loss": 27.20}, {"step": 20, "loss": 21.28},
        ],
    })
    r2.upload_text(
        "round_7/miner_hk_a/architecture.py",
        "class Model:\n    def __init__(self):\n        self.name = 'unit-test-model'\n",
    )

    set_db(store)

    # Mount dashboard — idempotent guard in case a previous module already did it.
    from database.dashboard import mount_dashboard
    from database.dashboard import app as dash_app
    if dash_app._state is None:
        mount_dashboard(
            app, store=store, pool=pool, r2=r2,
            get_challenge=lambda: {"round_id": 42, "task": "ts", "size_bucket": "small"},
            get_frontier=lambda: [{"id": 1}, {"id": 0}],
        )
    else:
        # Refresh state for this test module
        dash_app._state.store = store
        dash_app._state.pool = pool
        dash_app._state.r2 = r2
        dash_app._state.get_challenge = lambda: {
            "round_id": 42, "task": "ts", "size_bucket": "small",
        }
        dash_app._state.get_frontier = lambda: [{"id": 1}, {"id": 0}]

    return TestClient(app)


@pytest.fixture
def logged_in(dashboard_client):
    r = dashboard_client.post(
        "/dashboard/login",
        data={"key": "testkey", "next": "/dashboard/"},
        follow_redirects=False,
    )
    assert r.status_code == 302
    return dashboard_client


# ── Auth tests ────────────────────────────────────────────────


def test_login_form_renders(dashboard_client):
    r = dashboard_client.get("/dashboard/login")
    assert r.status_code == 200
    assert "Radar Dashboard" in r.text
    assert "Access key" in r.text


def test_login_bad_key(dashboard_client):
    r = dashboard_client.post(
        "/dashboard/login",
        data={"key": "wrong", "next": "/dashboard/"},
        follow_redirects=False,
    )
    assert r.status_code == 401
    assert "Invalid key" in r.text


def test_login_good_key_sets_cookie(dashboard_client):
    r = dashboard_client.post(
        "/dashboard/login",
        data={"key": "testkey", "next": "/dashboard/"},
        follow_redirects=False,
    )
    assert r.status_code == 302
    assert "radar_dashboard_session=" in r.headers.get("set-cookie", "")


def test_overview_requires_auth(dashboard_client):
    # Use a fresh client so we don't inherit cookies from another test
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/", follow_redirects=False)
    assert r.status_code == 302
    assert "/dashboard/login" in r.headers["location"]


def test_login_next_open_redirect_guard(dashboard_client):
    """Login must only accept /dashboard-prefixed next URLs."""
    r = dashboard_client.post(
        "/dashboard/login",
        data={"key": "testkey", "next": "https://evil.example/steal"},
        follow_redirects=False,
    )
    assert r.status_code == 302
    assert r.headers["location"].startswith("/dashboard")


# ── View tests ────────────────────────────────────────────────


def test_overview(logged_in):
    r = logged_in.get("/dashboard/")
    assert r.status_code == 200
    assert "Overview" in r.text
    assert "Active round" in r.text
    assert "root" in r.text  # recent list includes our first element


def test_browse_no_filter(logged_in):
    r = logged_in.get("/dashboard/experiments")
    assert r.status_code == 200
    assert "matching" in r.text
    # Items include both successful and failed entries
    assert "root" in r.text and "other" in r.text


def test_experiment_detail(logged_in):
    r = logged_in.get("/dashboard/experiments/1")
    assert r.status_code == 200
    assert "#1 child" in r.text
    assert "Architecture code" in r.text
    assert "print(1)" in r.text


def test_experiment_detail_diff_partial(logged_in):
    r = logged_in.get("/dashboard/experiments/1/diff")
    assert r.status_code == 200
    # Partial template renders the diff, not the full page
    assert "language-diff" in r.text


def test_experiment_detail_missing(logged_in):
    r = logged_in.get("/dashboard/experiments/9999")
    assert r.status_code == 404


def test_pareto_view_renders(logged_in):
    r = logged_in.get("/dashboard/pareto")
    assert r.status_code == 200
    assert "pareto-scatter" in r.text


def test_pareto_json_flags_dominated(logged_in):
    r = logged_in.get("/dashboard/api/pareto.json")
    assert r.status_code == 200
    data = r.json()
    assert {p["id"] for p in data["points"]} == {0, 1}
    # Index 1 dominates index 0 (lower metric, higher flops doesn't dominate
    # — actually index 0 is smaller in FLOPs but has worse metric, so both
    # should be on-frontier). Index 1 has better metric but higher flops,
    # so neither dominates the other. Both should be on_frontier.
    frontier_ids = {p["id"] for p in data["points"] if p["on_frontier"]}
    assert frontier_ids == {0, 1}


def test_loss_curve_api(logged_in):
    r = logged_in.get("/dashboard/api/loss_curve/0.json")
    assert r.status_code == 200
    data = r.json()
    assert data["index"] == 0
    assert data["points"] == [2.0, 1.5, 1.2]


def test_miners_list(logged_in):
    r = logged_in.get("/dashboard/miners")
    assert r.status_code == 200
    assert "hk_a" in r.text and "hk_b" in r.text


def test_miner_detail(logged_in):
    r = logged_in.get("/dashboard/miners/hk_a")
    assert r.status_code == 200
    assert "hk_a" in r.text
    assert "root" in r.text and "child" in r.text


def test_miner_detail_missing(logged_in):
    r = logged_in.get("/dashboard/miners/nonexistent_hotkey")
    assert r.status_code == 404


# ── Provenance heatmap tests ──────────────────────────────────


def test_provenance_view_renders(logged_in):
    r = logged_in.get("/dashboard/provenance")
    assert r.status_code == 200
    assert "Provenance activity" in r.text
    assert "heatmap-miner-rounds" in r.text
    assert "heatmap-top-experiments" in r.text


def test_provenance_miner_rounds_json(logged_in):
    r = logged_in.get("/dashboard/api/provenance/miner_rounds.json")
    assert r.status_code == 200
    data = r.json()
    assert set(data["rounds"]) == {7, 8}
    # hk_a outranks hk_b in activity
    assert data["miners"][0] == "hk_a"
    assert "hk_b" in data["miners"]
    # Unique-experiment count: hk_a round 8 accessed {0, 1} = 2
    r8 = data["rounds"].index(8)
    a = data["miners"].index("hk_a")
    assert data["matrix"][a][r8] == 2


def test_provenance_top_experiments_json(logged_in):
    r = logged_in.get("/dashboard/api/provenance/top_experiments.json")
    assert r.status_code == 200
    data = r.json()
    exp_ids = [e["id"] for e in data["experiments"]]
    # Experiment 0 was referenced 3 times, experiment 1 twice → both in top-K
    assert 0 in exp_ids and 1 in exp_ids
    # Top-referenced first
    assert exp_ids[0] == 0
    # Matrix contains per-miner query counts
    a = data["miners"].index("hk_a")
    assert data["matrix"][a][exp_ids.index(0)] == 2  # hk_a queried exp 0 twice
    assert data["matrix"][a][exp_ids.index(1)] == 2  # hk_a queried exp 1 twice


def test_provenance_top_experiments_includes_names(logged_in):
    r = logged_in.get("/dashboard/api/provenance/top_experiments.json")
    data = r.json()
    names = {e["id"]: e["name"] for e in data["experiments"]}
    assert names[0] == "root"
    assert names[1] == "child"


# ── Logs (R2) tests ───────────────────────────────────────────


def test_logs_view_renders(logged_in):
    r = logged_in.get("/dashboard/logs/7/hk_a")
    assert r.status_code == 200
    assert "training_meta.json" in r.text
    assert "epoch 0 loss=2.0" in r.text


def test_logs_meta_json(logged_in):
    r = logged_in.get("/dashboard/logs/7/hk_a/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["flops"] == 1234
    assert body["ok"] is True


def test_logs_stdout_cap(logged_in):
    r = logged_in.get("/dashboard/logs/7/hk_a/stdout")
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is False
    assert "epoch 0" in body["text"]


def test_logs_stdout_tail_when_large(logged_in, monkeypatch):
    """When stdout exceeds the byte cap, only the tail is returned."""
    monkeypatch.setattr(Config, "DASHBOARD_MAX_LOG_BYTES", 32)
    r = logged_in.get("/dashboard/logs/7/hk_a/stdout")
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is True
    assert len(body["text"]) <= 32


def test_logs_stdout_direct_redirect(logged_in):
    r = logged_in.get(
        "/dashboard/logs/7/hk_a/stdout?direct=1", follow_redirects=False,
    )
    assert r.status_code == 302
    assert r.headers["location"].startswith("https://fake-r2.example/")


def test_logs_view_renders_architecture_and_loss_canvas(logged_in):
    r = logged_in.get("/dashboard/logs/7/hk_a")
    assert r.status_code == 200
    # Architecture code is inlined into the page
    assert "unit-test-model" in r.text
    assert "architecture.py" in r.text
    # Loss-history canvas is rendered when meta contains loss arrays
    assert 'id="loss-history"' in r.text
    assert 'data-round="7"' in r.text


def test_logs_architecture_json(logged_in):
    r = logged_in.get("/dashboard/logs/7/hk_a/architecture")
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is False
    assert "unit-test-model" in body["text"]


def test_logs_architecture_direct_redirect(logged_in):
    r = logged_in.get(
        "/dashboard/logs/7/hk_a/architecture?direct=1", follow_redirects=False,
    )
    assert r.status_code == 302
    assert r.headers["location"].startswith("https://fake-r2.example/")


def test_logs_architecture_missing(logged_in):
    r = logged_in.get("/dashboard/logs/99/hk_missing/architecture")
    assert r.status_code == 404


def test_logs_meta_includes_loss_history(logged_in):
    r = logged_in.get("/dashboard/logs/7/hk_a/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["train_loss_history"][0] == {"step": 10, "loss": 22.19}
    assert body["val_loss_history"][1] == {"step": 20, "loss": 21.28}
