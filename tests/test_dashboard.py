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
        agent_history: Optional[list[dict]] = None,
        agent_bundles: Optional[dict[str, dict]] = None,
        training_metas: Optional[dict[tuple[int, str], dict]] = None,
    ):
        self.elements = elements
        self.access_log = access_log or []
        self.agent_history = agent_history or []
        # code_hash -> bundle dict (mirrors the agent_bundles Postgres table)
        self.agent_bundles = agent_bundles or {}
        # (round_id, hotkey) -> meta dict (mirrors training_metas table)
        self.training_metas = training_metas or {}

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
        # Benchmark: per-round aggregates for a task (with best_hotkey join)
        if "group by round_id" in sql_l and "lateral" in sql_l:
            task = params[0] if params else ""
            limit = params[1] if len(params) > 1 else 100
            matching = [e for e in self.elements if e.task == task and e.round_id >= 0]
            per_round: dict[int, list] = {}
            for e in matching:
                per_round.setdefault(e.round_id, []).append(e)
            out = []
            for rid in sorted(per_round, reverse=True)[:limit]:
                bucket = per_round[rid]
                succ = [e for e in bucket if e.success and e.metric is not None]
                best = min(succ, key=lambda e: e.metric) if succ else None
                out.append({
                    "round_id": rid,
                    "total": len(bucket),
                    "successes": len(succ),
                    "best_metric": best.metric if best else None,
                    "mean_metric": (
                        sum(e.metric for e in succ) / len(succ) if succ else None
                    ),
                    "started_at": min(e.timestamp or 0.0 for e in bucket),
                    "best_hotkey": best.miner_hotkey if best else None,
                })
            return out
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
        # Agent history for a hotkey
        if "from agent_submission_history" in sql_l and "where hotkey = $1" in sql_l:
            hk = params[0]
            limit = params[1] if len(params) > 1 else 100
            rows = [h for h in self.agent_history if h["hotkey"] == hk]
            rows.sort(key=lambda h: h["timestamp"], reverse=True)
            return rows[:limit]
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
        if "max(round_id)" in sql_l and "max(timestamp)" in sql_l:
            rounds = [e.round_id for e in self.elements if e.round_id >= 0]
            stamps = [e.timestamp for e in self.elements if e.timestamp]
            return {
                "last_round": max(rounds) if rounds else None,
                "last_at": max(stamps) if stamps else None,
            }
        if "from agent_submission_history" in sql_l and "code_hash = $1" in sql_l:
            target = params[0]
            matches = [h for h in self.agent_history if h["code_hash"] == target]
            if not matches:
                return None
            matches.sort(key=lambda h: h["timestamp"], reverse=True)
            return matches[0]
        if "from agent_bundles" in sql_l and "code_hash = $1" in sql_l:
            target = params[0]
            bundle = self.agent_bundles.get(target)
            if bundle is None:
                return None
            return {"bundle": json.dumps(bundle)}
        if "from training_metas" in sql_l and "round_id = $1" in sql_l:
            key = (int(params[0]), params[1])
            meta = self.training_metas.get(key)
            if meta is None:
                return None
            return {"meta": json.dumps(meta)}
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
    agent_history = [
        {
            "hotkey": "hk_a", "miner_uid": 1,
            "code_hash": "hash_v1" + ("0" * 50),
            "entry_point": "agent.py",
            "r2_key": "agents/hk_a/hash_v1.json",
            "round_submitted": 6, "timestamp": 1500.0,
        },
        {
            "hotkey": "hk_a", "miner_uid": 1,
            "code_hash": "hash_v2" + ("0" * 50),
            "entry_point": "agent.py",
            "r2_key": "agents/hk_a/hash_v2.json",
            "round_submitted": 8, "timestamp": 2500.0,
        },
    ]
    # Bundle cache (Postgres ``agent_bundles`` table) — v1 lives here so the
    # dashboard serves it from Postgres alone. v2 is intentionally absent so
    # the R2 fallback path stays covered.
    bundle_v1 = {
        "files": {"agent.py": "def design_architecture():\n    return 'v1'\n"},
        "entry_point": "agent.py",
        "code_hash": "hash_v1" + ("0" * 50),
    }
    bundle_v2 = {
        "files": {"agent.py": "def design_architecture():\n    return 'v2'\n"},
        "entry_point": "agent.py",
        "code_hash": "hash_v2" + ("0" * 50),
    }
    # Round 7's meta lives in the Postgres cache; round 8's only in R2 — this
    # keeps the R2 fallback path covered while round 7 exercises the cache.
    cached_meta_round7 = {
        "round_id": 7, "miner_hotkey": "hk_a",
        "flops": 1234, "ok": True,
        "train_loss_history": [
            {"step": 10, "loss": 22.19}, {"step": 20, "loss": 14.71},
        ],
        "val_loss_history": [
            {"step": 10, "loss": 27.20}, {"step": 20, "loss": 21.28},
        ],
    }
    r2_meta_round8 = {
        "round_id": 8, "miner_hotkey": "hk_a",
        "flops": 5678, "ok": True,
    }
    pool = FakePool(
        elements,
        access_log=access_log,
        agent_history=agent_history,
        agent_bundles={bundle_v1["code_hash"]: bundle_v1},
        training_metas={(7, "hk_a"): cached_meta_round7},
    )
    r2 = FakeR2()
    r2.upload_json("agents/hk_a/hash_v1.json", bundle_v1)
    r2.upload_json("agents/hk_a/hash_v2.json", bundle_v2)
    # Seed a training log file so the log route has something to read.
    r2.upload_text("round_7/miner_hk_a/stdout.log", "epoch 0 loss=2.0\nepoch 1 loss=1.5\n")
    # Round 7 meta also goes to R2 so existing tests keep working — the cache
    # is preferred when both exist.
    r2.upload_json("round_7/miner_hk_a/training_meta.json", cached_meta_round7)
    r2.upload_json("round_8/miner_hk_a/training_meta.json", r2_meta_round8)
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


def test_json_api_reachable_without_cookie(dashboard_client):
    """The JSON API is public — a fresh client with no cookie gets 200."""
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    for path in (
        "/dashboard/api/stats.json",
        "/dashboard/api/tasks.json",
        "/dashboard/api/miners.json",
        "/dashboard/api/recent.json?n=5",
        "/dashboard/api/challenge.json",
        "/dashboard/api/rounds.json",
        "/dashboard/api/tasks_stats.json",
    ):
        r = fresh.get(path)
        assert r.status_code == 200, f"expected 200 for {path}, got {r.status_code}"


def test_json_api_reachable_with_cookie(logged_in):
    """Cookie-holding operators hit the same JSON API without being blocked."""
    r = logged_in.get("/dashboard/api/stats.json")
    assert r.status_code == 200


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


def test_benchmark_view_renders(logged_in):
    r = logged_in.get("/dashboard/benchmark")
    assert r.status_code == 200
    assert "Benchmark scores over time" in r.text
    # Task auto-picks the first known task ("ts") so the table renders.
    assert "0.9000" in r.text  # best metric from round 8
    assert "1.2000" in r.text  # best metric from round 7
    # Round IDs in the first column link into the browse view.
    assert "round_id=8" in r.text and "round_id=7" in r.text


def test_benchmark_view_unknown_task(logged_in):
    r = logged_in.get("/dashboard/benchmark?task=nope")
    assert r.status_code == 200
    assert "No completed rounds yet" in r.text


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


# ── Agent bundle viewer + diff ────────────────────────────────


def test_miner_detail_shows_agent_history(logged_in):
    r = logged_in.get("/dashboard/miners/hk_a")
    assert r.status_code == 200
    assert "Agent code timeline" in r.text
    # Both hash prefixes render, most recent first
    v1_prefix = "hash_v1" + ("0" * 9)
    v2_prefix = "hash_v2" + ("0" * 9)
    assert v2_prefix in r.text and v1_prefix in r.text
    # Diff link from v2 → v1 wired up
    assert "/dashboard/agent_code/" in r.text
    assert "/diff/" in r.text


def test_agent_bundle_view_renders_files(logged_in):
    hash_v1 = "hash_v1" + ("0" * 50)
    r = logged_in.get(f"/dashboard/agent_code/{hash_v1}")
    assert r.status_code == 200
    # Bundle metadata + source code surfaced (quotes get HTML-escaped)
    assert "hk_a" in r.text
    assert "agent.py" in r.text
    assert "return &#39;v1&#39;" in r.text
    # The other hash appears in the compare picker
    assert "hash_v2" in r.text


def test_agent_bundle_view_unknown_hash_404(logged_in):
    r = logged_in.get("/dashboard/agent_code/" + ("f" * 64))
    assert r.status_code == 404


def test_agent_bundle_diff_renders_unified_diff(logged_in):
    hash_v1 = "hash_v1" + ("0" * 50)
    hash_v2 = "hash_v2" + ("0" * 50)
    r = logged_in.get(f"/dashboard/agent_code/{hash_v2}/diff/{hash_v1}")
    assert r.status_code == 200
    # Unified diff markers present (quotes are HTML-escaped in rendered text)
    assert "language-diff" in r.text
    assert "return &#39;v1&#39;" in r.text
    assert "return &#39;v2&#39;" in r.text
    # Unified-diff +/- prefixes on the changed lines
    assert "-    return &#39;v1&#39;" in r.text
    assert "+    return &#39;v2&#39;" in r.text


def test_agent_bundle_diff_rejects_self(logged_in):
    hash_v1 = "hash_v1" + ("0" * 50)
    r = logged_in.get(f"/dashboard/agent_code/{hash_v1}/diff/{hash_v1}")
    assert r.status_code == 400


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


# ── Public JSON endpoints for the SPA ─────────────────────────


def test_public_tasks_stats_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/tasks_stats.json")
    assert r.status_code == 200
    assert "ts" in r.json()


def test_public_tasks_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/tasks.json")
    assert r.status_code == 200
    assert "ts" in r.json()


def test_public_miners_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/miners.json")
    assert r.status_code == 200
    hks = {m["miner_hotkey"] for m in r.json()}
    assert {"hk_a", "hk_b"} <= hks


def test_public_recent_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/recent.json?n=10")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    assert len(payload) == 3  # 3 sample elements


def test_public_challenge_json_nests_frontier_size(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/challenge.json")
    assert r.status_code == 200
    body = r.json()
    assert body["round_id"] == 42
    assert body["task"] == "ts"
    assert body["frontier_size"] == 2


def test_public_rounds_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/rounds.json")
    assert r.status_code == 200
    assert set(r.json()) == {7, 8}


def test_public_heartbeat_json(dashboard_client):
    """Heartbeat exposes ``now``, ``last_submission_at``, and ``last_round_id``."""
    from fastapi.testclient import TestClient
    from database.dashboard import api as dash_api

    # Reset the in-process cache so a previous test doesn't shadow this one.
    dash_api._heartbeat_cache = None
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/heartbeat.json")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"now", "last_submission_at", "last_round_id"}
    assert body["last_round_id"] == 8
    assert body["last_submission_at"] == 3000.0
    assert body["now"] > 0


def test_public_heartbeat_caches_in_process(dashboard_client):
    """Repeated polls within the TTL share one cached payload."""
    from fastapi.testclient import TestClient
    from database.dashboard import api as dash_api

    dash_api._heartbeat_cache = None
    fresh = TestClient(app)
    a = fresh.get("/dashboard/api/heartbeat.json").json()
    b = fresh.get("/dashboard/api/heartbeat.json").json()
    # ``now`` advances on every request, but the cached fields stay stable
    # (and pinned to whatever was in the DB at the first call).
    assert a["last_round_id"] == b["last_round_id"]
    assert a["last_submission_at"] == b["last_submission_at"]
    assert b["now"] >= a["now"]


def test_public_benchmark_json(dashboard_client):
    """Public benchmark endpoint mirrors the Jinja /dashboard/benchmark view."""
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/benchmark.json?task=ts&limit=100")
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "ts"
    assert "ts" in body["tasks"]
    by_round = {row["round_id"]: row for row in body["rows"]}
    assert set(by_round) == {7, 8}
    assert by_round[8]["best_metric"] == 0.9
    assert by_round[7]["best_metric"] == 1.2


def test_public_benchmark_json_defaults_to_first_task(dashboard_client):
    """Omitting ``task`` falls back to the first known task (matches Jinja)."""
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/benchmark.json")
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "ts"
    assert body["rows"]


def test_public_benchmark_json_unknown_task(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/benchmark.json?task=nope")
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "nope"
    assert body["rows"] == []


def test_public_browse_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/browse.json?page=0&page_size=10")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body and "total" in body
    assert "page" in body and "page_size" in body


def test_public_experiment_by_index_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/experiments/1.json")
    assert r.status_code == 200
    assert r.json()["name"] == "child"

    r2 = fresh.get("/dashboard/api/experiments/9999.json")
    assert r2.status_code == 404


def test_public_miner_submissions_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/miners/hk_a/submissions.json?limit=20")
    assert r.status_code == 200
    body = r.json()
    assert body["hotkey"] == "hk_a"
    assert len(body["submissions"]) == 2
    assert len(body["agent_history"]) == 2


def test_public_experiment_lineage_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/experiments/1/lineage.json")
    assert r.status_code == 200
    body = r.json()
    assert body["root"]["index"] == 1
    assert isinstance(body["diffs"], list)


def test_public_logs_endpoints(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    meta = fresh.get("/dashboard/api/logs/7/hk_a/meta.json")
    assert meta.status_code == 200
    assert meta.json()["flops"] == 1234

    stdout = fresh.get("/dashboard/api/logs/7/hk_a/stdout.json")
    assert stdout.status_code == 200
    assert "epoch 0" in stdout.json()["text"]

    arch = fresh.get("/dashboard/api/logs/7/hk_a/architecture.json")
    assert arch.status_code == 200
    assert "unit-test-model" in arch.json()["text"]


def test_public_agent_code_json(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    hash_v1 = "hash_v1" + ("0" * 50)
    r = fresh.get(f"/dashboard/api/agent_code/{hash_v1}.json")
    assert r.status_code == 200
    body = r.json()
    assert body["entry_point"] == "agent.py"
    assert "agent.py" in body["files"]
    assert "v1" in body["files"]["agent.py"]
    # history excludes the requested hash
    assert all(h["code_hash"] != hash_v1 for h in body["history"])


def test_public_agent_code_unknown_returns_404(dashboard_client):
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/agent_code/" + ("f" * 64) + ".json")
    assert r.status_code == 404


def test_public_logs_meta_serves_from_postgres_cache_without_r2(dashboard_client):
    """training_meta cached in Postgres is served even when R2 is unavailable.

    Reproduces "training loss missing on the new dashboard": dashboard-mode
    deploys (no R2) used to 404 the meta endpoint, hiding the loss curves.
    """
    from fastapi.testclient import TestClient
    from database.dashboard import app as dash_app

    saved_r2 = dash_app._state.r2
    dash_app._state.r2 = None
    try:
        fresh = TestClient(app)
        r = fresh.get("/dashboard/api/logs/7/hk_a/meta.json")
        assert r.status_code == 200
        meta = r.json()
        assert meta["train_loss_history"][0]["loss"] == 22.19
        assert meta["val_loss_history"][-1]["loss"] == 21.28

        # Round 8 only lives in R2, so without R2 it's a clean 404.
        miss = fresh.get("/dashboard/api/logs/8/hk_a/meta.json")
        assert miss.status_code == 404
    finally:
        dash_app._state.r2 = saved_r2


def test_public_logs_meta_falls_back_to_r2(dashboard_client):
    """Rows that predate the cache are still served from R2."""
    from fastapi.testclient import TestClient
    fresh = TestClient(app)
    r = fresh.get("/dashboard/api/logs/8/hk_a/meta.json")
    assert r.status_code == 200
    assert r.json()["flops"] == 5678


def test_public_agent_code_json_works_without_r2(dashboard_client):
    """Dashboard-mode deploys without R2 still serve bundles cached in Postgres.

    Reproduces the original 503: drops the R2 client and confirms the cached
    v1 bundle is served from the ``agent_bundles`` Postgres table alone.
    """
    from fastapi.testclient import TestClient
    from database.dashboard import app as dash_app

    saved_r2 = dash_app._state.r2
    dash_app._state.r2 = None
    try:
        fresh = TestClient(app)
        hash_v1 = "hash_v1" + ("0" * 50)
        r = fresh.get(f"/dashboard/api/agent_code/{hash_v1}.json")
        assert r.status_code == 200
        body = r.json()
        assert body["files"]["agent.py"].endswith("'v1'\n")

        # v2 isn't in the Postgres cache and there's no R2 fallback either —
        # this should 404 rather than the legacy 503.
        hash_v2 = "hash_v2" + ("0" * 50)
        r2 = fresh.get(f"/dashboard/api/agent_code/{hash_v2}.json")
        assert r2.status_code == 404
    finally:
        dash_app._state.r2 = saved_r2
