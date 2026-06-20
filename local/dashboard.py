"""Read-only HTTP dashboard over ``local/radar_local.db``.

Standalone — does not need the validator to be running. Opens the
SQLite file in ``mode=ro`` so it's safe to point at a db that another
process is writing. Stdlib only, no JS deps. The UI shell lives in
``local/dashboard.html`` and is served at ``/``.

  python -m local.dashboard --db local/radar_local.db --port 8765
  # then open http://127.0.0.1:8765/

Endpoints:
  GET /                            HTML page
  GET /api/stats                   {total, successful, best_metric, ...}
  GET /api/leaderboard?n=N         top-N by metric ASC (lower=better)
  GET /api/recent?n=N              latest N by id
  GET /api/frontier                Pareto front on (metric, flops)
  GET /api/frontier_crps_mase      Pareto front on (crps, mase) — GIFT-scored
                                   tasks (ts_forecasting, synthetic_data_generator)
  GET /api/continuation_frontier   Pareto front on (cumulative_compute, Δ) for
                                   warm-started runs only
  GET /api/data_pipeline_frontier  Pareto front on (aulc, gift_metric) —
                                   ts_data_pipeline runs only
  GET /api/synth_frontier          Pareto front on (crps, mase) —
                                   synthetic_data_generator runs only
  GET /api/lineage                 Lineage forest (nodes/edges) + frozen-arch
                                   cross-task interaction links
  GET /api/frozen_archs            Frozen-arch version list (ts_data_pipeline)
  GET /api/experiment/<id>         full row (incl. code, loss_curve)
  GET /api/experiment/<id>/lineage_curve
                                   parent→child loss/val curves for a
                                   continuation lineage (stitched curve)
  GET /api/lab_reports?n=&task=    structured per-experiment post-mortems
  GET /api/lab_report/<exp_id>     one experiment's lab report
  GET /api/noise                   replicate-derived eval noise floors
                                   (overall + per task)
  GET /api/events                  agent_events list (round_id, miner_id,
                                   kind, endpoint, task, only_errors,
                                   before_id, since_id, limit filters)
  GET /api/event/<id>              full event (request/response bodies)
  GET /api/event_stats             aggregated counts for filter chips
  GET /api/checkpoints             local safetensors checkpoints + meta
  GET /api/checkpoint/<exp_id>/signature
                                   tensor name → shape from header
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3

math_isfinite = math.isfinite
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from local import dashboard_logs
from local import dashboard_reports

logger = logging.getLogger(__name__)

_HTML_PATH = Path(__file__).with_name("dashboard.html")
_CSS_PATH = Path(__file__).with_name("dashboard.css")
_JS_PATH = Path(__file__).with_name("dashboard.js")


def _qstr(q: dict[str, list[str]], key: str) -> str | None:
    v = q.get(key, [""])[0]
    return v or None


def _qint(q: dict[str, list[str]], key: str) -> int | None:
    v = q.get(key, [""])[0]
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Open the db in read-only mode so concurrent writers aren't blocked."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _row(r: sqlite3.Row, *, with_code: bool = False) -> dict[str, Any]:
    # ``mode``/``n_rounds``/``parent_index``/``cumulative_compute`` were
    # added when continuation training landed; older rows may pre-date
    # the schema migration so fall back to the "fresh run" defaults.
    def _opt(key: str, default: Any) -> Any:
        try:
            v = r[key]
        except (IndexError, KeyError):
            return default
        return default if v is None else v

    mode = _opt("mode", "new")
    n_rounds = int(_opt("n_rounds", 1))
    parent_index = r["parent_index"] if "parent_index" in r.keys() else None
    cumulative_compute = float(_opt("cumulative_compute", 0.0))
    is_cont = mode == "continue" or (n_rounds >= 2 and parent_index is not None)
    objectives = json.loads(r["objectives_json"] or "{}")
    out = {
        "id": r["id"],
        "round_id": r["round_id"],
        "miner_id": r["miner_id"],
        "name": r["name"],
        "metric": r["metric"],
        "score": r["score"],
        "success": bool(r["success"]),
        "objectives": objectives,
        "analysis": r["analysis"],
        "task": r["task"],
        "generation": r["generation"],
        "prompt_id": r["prompt_id"],
        "timestamp": r["timestamp"],
        "mode": mode,
        "n_rounds": n_rounds,
        "parent_index": parent_index,
        "cumulative_compute": cumulative_compute,
        "is_continuation": is_cont,
        # Validator-owned round type (replicate / ablation / recipe_only /
        # transfer / continuation[:kind] / new) derived from the
        # objectives stamps — drives the kind badges in the UI.
        "round_kind": dashboard_reports.round_kind(mode, objectives),
    }
    if with_code:
        out["code"] = r["code"]
        out["motivation"] = r["motivation"]
        out["reasoning"] = r["reasoning"]
        out["loss_curve"] = json.loads(r["loss_curve_json"] or "[]")
        try:
            out["val_curve"] = json.loads(r["val_curve_json"] or "[]")
        except (IndexError, KeyError):
            # Older db schemas without the val_curve_json column.
            out["val_curve"] = []
        out["tool_calls"] = json.loads(r["tool_calls_json"] or "[]")
    return out


def _stats(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) AS successful, "
        "MIN(metric) AS best, MAX(metric) AS worst, AVG(metric) AS mean, "
        "MAX(round_id) AS last_round "
        "FROM experiments"
    ).fetchone()
    total = row["total"] or 0
    successful = row["successful"] or 0
    n_miners = conn.execute(
        "SELECT COUNT(DISTINCT miner_id) AS n FROM experiments"
    ).fetchone()["n"] or 0
    # Pre-continuation DBs lack mode/n_rounds; the validator's
    # schema-migration usually fills them in, but tolerate absence so a
    # standalone dashboard against an old snapshot doesn't 500.
    try:
        cont_row = conn.execute(
            "SELECT "
            " SUM(CASE WHEN mode='continue' OR (n_rounds>=2 AND parent_index IS NOT NULL) "
            "          THEN 1 ELSE 0 END) AS n_cont, "
            " SUM(CASE WHEN success=1 AND "
            "          (mode='continue' OR (n_rounds>=2 AND parent_index IS NOT NULL)) "
            "          THEN 1 ELSE 0 END) AS n_cont_ok "
            "FROM experiments"
        ).fetchone()
        n_continuation = cont_row["n_cont"] or 0
        n_continuation_ok = cont_row["n_cont_ok"] or 0
    except sqlite3.OperationalError:
        n_continuation = 0
        n_continuation_ok = 0
    # Scheduled-vs-actual continuation counts come from the challenges table:
    # the validator stamps scheduled_round_type + downgrade_reason on the
    # payload before persisting. n_scheduled distinguishes "always-novel"
    # rounds from "scheduled continuation, downgraded because no parents",
    # which the experiments table alone can't tell apart.
    n_cont_scheduled = 0
    n_cont_downgraded = 0
    try:
        sched_row = conn.execute(
            "SELECT "
            " SUM(CASE WHEN json_extract(payload_json,'$.scheduled_round_type')"
            "          = 'continuation' THEN 1 ELSE 0 END) AS n_sched, "
            " SUM(CASE WHEN json_extract(payload_json,'$.scheduled_round_type')"
            "          = 'continuation' AND "
            "          json_extract(payload_json,'$.round_type') = 'new' "
            "          THEN 1 ELSE 0 END) AS n_downgraded "
            "FROM challenges"
        ).fetchone()
        n_cont_scheduled = sched_row["n_sched"] or 0
        n_cont_downgraded = sched_row["n_downgraded"] or 0
    except sqlite3.OperationalError:
        pass
    out = {
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "best_metric": row["best"],
        "worst_metric": row["worst"],
        "mean_metric": row["mean"],
        "last_round": row["last_round"],
        "n_miners": n_miners,
        "n_continuation": n_continuation,
        "n_continuation_successful": n_continuation_ok,
        "n_continuation_scheduled": n_cont_scheduled,
        "n_continuation_downgraded": n_cont_downgraded,
        "n_novel": total - n_continuation,
        "n_novel_successful": successful - n_continuation_ok,
    }
    # Special-round counts + the replicate-derived eval noise floor —
    # the context that makes frontier movements interpretable.
    try:
        out.update(dashboard_reports.special_counts(conn))
        out["noise"] = dashboard_reports.noise_floors(conn)["overall"]
    except Exception as e:  # noqa: BLE001
        logger.debug("special-round stats failed: %s", e)
    return out


def _leaderboard(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM experiments "
        "WHERE success=1 AND metric IS NOT NULL "
        "ORDER BY metric ASC LIMIT ?",
        (n,),
    ).fetchall()
    return [_row(r) for r in rows]


def _recent(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (n,),
    ).fetchall()
    return [_row(r) for r in rows]


def _frontier(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Non-dominated set on (metric, flops_equivalent_size). Both lower=better."""
    rows = conn.execute(
        "SELECT * FROM experiments WHERE success=1 AND metric IS NOT NULL"
    ).fetchall()
    points = [_row(r) for r in rows]
    front: list[dict[str, Any]] = []
    for p in points:
        pm = p["metric"]
        pf = p["objectives"].get("flops_equivalent_size", 0)
        dominated = False
        for o in points:
            if o is p:
                continue
            om = o["metric"]
            of = o["objectives"].get("flops_equivalent_size", 0)
            if om <= pm and of <= pf and (om < pm or of < pf):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda e: e["objectives"].get("flops_equivalent_size", 0))
    return front


def _frontier_crps_mase(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Non-dominated set on (crps, mase). Both lower=better. Only experiments
    that have both values in their objectives (i.e. GIFT-Eval succeeded)
    contribute — that spans every GIFT-scored task (ts_forecasting and
    synthetic_data_generator; ts_data_pipeline stamps them too)."""
    rows = conn.execute(
        "SELECT * FROM experiments WHERE success=1 AND metric IS NOT NULL"
    ).fetchall()
    points = [_row(r) for r in rows]
    points = [
        p for p in points
        if p["objectives"].get("crps") is not None
        and p["objectives"].get("mase") is not None
    ]
    front: list[dict[str, Any]] = []
    for p in points:
        pc = p["objectives"]["crps"]
        pm = p["objectives"]["mase"]
        dominated = False
        for o in points:
            if o is p:
                continue
            oc = o["objectives"]["crps"]
            om = o["objectives"]["mase"]
            if oc <= pc and om <= pm and (oc < pc or om < pm):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda e: e["objectives"]["crps"])
    return front


def _continuation_frontier(conn: sqlite3.Connection) -> dict[str, Any]:
    """Continuation-only Pareto front on (cumulative_compute ↓, Δ ↑).

    Δ is computed as ``parent.metric − this.metric`` so a positive Δ means
    the warm-start outperformed its parent. Points with non-positive or
    non-finite Δ are excluded from the frontier but still returned in
    ``all`` so the chart can render them as off-frontier dots.
    """
    rows = conn.execute(
        "SELECT * FROM experiments WHERE success=1 AND metric IS NOT NULL"
    ).fetchall()
    by_id: dict[int, dict[str, Any]] = {}
    points: list[dict[str, Any]] = []
    for r in rows:
        p = _row(r)
        by_id[p["id"]] = p
        if p["is_continuation"]:
            points.append(p)
    enriched: list[dict[str, Any]] = []
    for p in points:
        parent = by_id.get(p.get("parent_index"))
        pm = parent["metric"] if parent else None
        if pm is None or p["metric"] is None:
            continue
        delta = float(pm) - float(p["metric"])
        compute = float(p.get("cumulative_compute") or 0.0)
        if not (math_isfinite(delta) and math_isfinite(compute)):
            continue
        p = dict(p)
        p["delta"] = delta
        p["parent_metric"] = float(pm)
        enriched.append(p)
    front: list[dict[str, Any]] = []
    for p in enriched:
        if p["delta"] <= 0:
            continue
        pd_, pc_ = p["delta"], p["cumulative_compute"]
        dominated = False
        for o in enriched:
            if o is p or o["delta"] <= 0:
                continue
            od_, oc_ = o["delta"], o["cumulative_compute"]
            if oc_ <= pc_ and od_ >= pd_ and (oc_ < pc_ or od_ > pd_):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda e: e["cumulative_compute"])
    return {"frontier": front, "all": enriched}


def _synth_frontier(conn: sqlite3.Connection) -> dict[str, Any]:
    """Non-dominated set on (crps, mase) restricted to synthetic_data_generator.

    The synth task shares ts_forecasting's GIFT-only scoring but trains a
    *fixed* reference arch, so it earns its own crps×mase frontier tab rather
    than sharing the architecture tab. Both axes lower=better. Returns
    ``{frontier, all, versions}`` — ``versions`` lists the distinct
    ``synth_arch_version`` values present so the UI can flag a model swap
    (continuation lineages pin to one version).
    """
    rows = conn.execute(
        "SELECT * FROM experiments WHERE success=1 AND metric IS NOT NULL "
        "AND task='synthetic_data_generator'"
    ).fetchall()
    points = [_row(r) for r in rows]
    points = [
        p for p in points
        if p["objectives"].get("crps") is not None
        and p["objectives"].get("mase") is not None
    ]
    front: list[dict[str, Any]] = []
    for p in points:
        pc = p["objectives"]["crps"]
        pm = p["objectives"]["mase"]
        dominated = False
        for o in points:
            if o is p:
                continue
            oc = o["objectives"]["crps"]
            om = o["objectives"]["mase"]
            if oc <= pc and om <= pm and (oc < pc or om < pm):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda e: e["objectives"]["crps"])
    versions = sorted({
        int(p["objectives"].get("synth_arch_version") or 0)
        for p in points
    })
    return {"frontier": front, "all": points, "versions": versions}


def _data_pipeline_frontier(conn: sqlite3.Connection) -> dict[str, Any]:
    """Non-dominated set on (aulc_axis, gift_metric) for ts_data_pipeline runs.

    Both lower=better. Returns ``{frontier, all, versions, aulc_axis}`` so the
    chart can render off-frontier dots and segment per frozen-arch version
    (since each version anchors its own comparison). ``aulc_axis`` is
    ``"aulc_ratio"`` whenever every plotted point carries the
    baseline-normalised ratio (the quantity the metric uses) and falls
    back to ``"aulc"`` for legacy rows logged before the baseline anchor.
    """
    rows = conn.execute(
        "SELECT * FROM experiments WHERE success=1 AND metric IS NOT NULL "
        "AND task='ts_data_pipeline'"
    ).fetchall()
    points = [_row(r) for r in rows]
    points = [
        p for p in points
        if p["objectives"].get("aulc") is not None
        and p["objectives"].get("gift_metric") is not None
    ]
    aulc_axis = (
        "aulc_ratio"
        if points and all(
            p["objectives"].get("aulc_ratio") is not None for p in points
        )
        else "aulc"
    )
    front: list[dict[str, Any]] = []
    for p in points:
        pa = p["objectives"][aulc_axis]
        pg = p["objectives"]["gift_metric"]
        dominated = False
        for o in points:
            if o is p:
                continue
            oa = o["objectives"][aulc_axis]
            og = o["objectives"]["gift_metric"]
            if oa <= pa and og <= pg and (oa < pa or og < pg):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda e: e["objectives"][aulc_axis])
    versions = sorted({
        int(p["objectives"].get("frozen_arch_version") or 0)
        for p in points
    })
    return {
        "frontier": front, "all": points, "versions": versions,
        "aulc_axis": aulc_axis,
    }


def _frozen_archs(base_dir: str) -> list[dict[str, Any]]:
    """List persisted frozen-arch snapshots (metadata only, no code)."""
    try:
        from local.frozen_arch import FrozenArchStore
        return FrozenArchStore(base_dir or None).all_versions()
    except Exception as e:  # noqa: BLE001
        logger.warning("frozen_archs read failed: %s", e)
        return []


def _experiment(conn: sqlite3.Connection, exp_id: int) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT * FROM experiments WHERE id=?", (exp_id,)
    ).fetchone()
    return _row(r, with_code=True) if r else None


def _lineage(conn: sqlite3.Connection, frozen_arch_dir: str) -> dict[str, Any]:
    """Lineage forest + cross-task interactions for the tree viewer.

    Nodes are experiments; ``parent_index`` chains form within-task
    lineages (novel root → continuation children). The cross-task
    interaction is the frozen-arch promotion: a ``ts_forecasting``
    experiment is snapshotted as a frozen-arch version which
    ``ts_data_pipeline`` runs then train against. Each version becomes an
    ``arch:<v>`` node with a ``promote`` edge in from its source forecasting
    experiment and ``train`` edges out to the pipeline runs that consumed it.

    Returns ``{nodes, archs, edges}``. ``edges`` reference experiments by
    integer id and arch versions by the string id ``"arch:<v>"``. Each node
    carries ``depth`` (lineage distance from its root) and ``in_lineage``
    (participates in a chain or interaction) so the client can hide isolated
    single-run nodes by default.
    """
    rows = conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    by_id: dict[int, dict[str, Any]] = {}
    nodes: list[dict[str, Any]] = []
    for r in rows:
        p = _row(r)
        by_id[p["id"]] = p
        nodes.append(p)

    children: dict[int, list[int]] = {}
    for p in nodes:
        pid = p.get("parent_index")
        if pid is not None and pid in by_id:
            children.setdefault(pid, []).append(p["id"])

    def _depth(start: int) -> int:
        seen: set[int] = set()
        d = 0
        cur = by_id.get(start)
        while cur is not None and cur["id"] not in seen:
            seen.add(cur["id"])
            par = cur.get("parent_index")
            if par is None or par not in by_id:
                break
            d += 1
            cur = by_id.get(par)
        return d

    # Pipeline runs grouped by the frozen-arch version they trained against.
    consumers: dict[int, list[int]] = {}
    for p in nodes:
        if p["task"] == "ts_data_pipeline":
            v = p["objectives"].get("frozen_arch_version")
            if v:
                consumers.setdefault(int(v), []).append(p["id"])

    edges: list[dict[str, Any]] = []
    for p in nodes:
        pid = p.get("parent_index")
        if pid is not None and pid in by_id:
            edges.append({"kind": "lineage", "from": pid, "to": p["id"]})

    archs_out: list[dict[str, Any]] = []
    for a in _frozen_archs(frozen_arch_dir):
        v = int(a.get("version") or 0)
        src = a.get("source_experiment_id")
        cons = consumers.get(v, [])
        archs_out.append({
            "version": v,
            "source_experiment_id": src,
            "source_metric": a.get("source_metric"),
            "source_task": "ts_forecasting",
            "created_at": a.get("created_at"),
            "n_consumers": len(cons),
        })
        if src is not None and src in by_id:
            edges.append({"kind": "promote", "from": src, "to": f"arch:{v}"})
        for cid in cons:
            edges.append({"kind": "train", "from": f"arch:{v}", "to": cid})

    # A node "participates" if it sits on a parent/child chain or touches an
    # interaction edge — used to hide noise-floor singleton novel runs.
    touched: set[int] = set()
    for e in edges:
        for end in (e["from"], e["to"]):
            if isinstance(end, int):
                touched.add(end)
    for p in nodes:
        p["depth"] = _depth(p["id"])
        p["child_ids"] = children.get(p["id"], [])
        p["in_lineage"] = (
            p["id"] in touched
            or p.get("parent_index") is not None
            or bool(children.get(p["id"]))
        )

    return {"nodes": nodes, "archs": archs_out, "edges": edges}


class _Handler(BaseHTTPRequestHandler):
    db_path: str = ""
    frozen_arch_dir: str = ""
    checkpoint_dir: str = ""
    html: bytes = b""
    css: bytes = b""
    js: bytes = b""

    def log_message(self, format, *args):  # noqa: A002
        logger.debug("dash %s - %s", self.address_string(), format % args)

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        q = parse_qs(parsed.query)

        if path == "/":
            return self._send(200, self.html, "text/html; charset=utf-8")
        if path == "/dashboard.css":
            return self._send(200, self.css, "text/css; charset=utf-8")
        if path == "/dashboard.js":
            return self._send(200, self.js, "application/javascript; charset=utf-8")

        if not path.startswith("/api/"):
            return self._json(404, {"error": "not found"})

        conn = None
        try:
            conn = _connect_ro(self.db_path)
            if path == "/api/stats":
                return self._json(200, _stats(conn))
            if path == "/api/leaderboard":
                n = int(q.get("n", ["20"])[0])
                return self._json(200, _leaderboard(conn, n))
            if path == "/api/recent":
                n = int(q.get("n", ["30"])[0])
                return self._json(200, _recent(conn, n))
            if path == "/api/frontier":
                return self._json(200, _frontier(conn))
            if path == "/api/frontier_crps_mase":
                return self._json(200, _frontier_crps_mase(conn))
            if path == "/api/continuation_frontier":
                return self._json(200, _continuation_frontier(conn))
            if path == "/api/data_pipeline_frontier":
                return self._json(200, _data_pipeline_frontier(conn))
            if path == "/api/synth_frontier":
                return self._json(200, _synth_frontier(conn))
            if path == "/api/lineage":
                return self._json(200, _lineage(conn, self.frozen_arch_dir))
            if path == "/api/frozen_archs":
                return self._json(200, _frozen_archs(self.frozen_arch_dir))
            if (path.startswith("/api/experiment/")
                    and path.endswith("/lineage_curve")):
                try:
                    exp_id = int(path.split("/")[3])
                except (ValueError, IndexError):
                    return self._json(400, {"error": "bad id"})
                return self._json(200, dashboard_reports.lineage_curve(
                    conn, exp_id,
                ))
            if path.startswith("/api/experiment/"):
                try:
                    exp_id = int(path.rsplit("/", 1)[1])
                except ValueError:
                    return self._json(400, {"error": "bad id"})
                exp = _experiment(conn, exp_id)
                if exp is None:
                    return self._json(404, {"error": "not found"})
                return self._json(200, exp)
            if path == "/api/lab_reports":
                n = int(q.get("n", ["50"])[0])
                task = _qstr(q, "task")
                return self._json(200, dashboard_reports.list_lab_reports(
                    conn, n=n, task=task,
                ))
            if path.startswith("/api/lab_report/"):
                try:
                    exp_id = int(path.rsplit("/", 1)[1])
                except ValueError:
                    return self._json(400, {"error": "bad id"})
                report = dashboard_reports.get_lab_report(conn, exp_id)
                if report is None:
                    return self._json(404, {"error": "not found"})
                return self._json(200, report)
            if path == "/api/noise":
                return self._json(200, dashboard_reports.noise_floors(conn))
            if path == "/api/events":
                return self._json(200, dashboard_logs.list_events(
                    conn,
                    round_id=_qint(q, "round_id"),
                    miner_id=_qstr(q, "miner_id"),
                    kind=_qstr(q, "kind"),
                    endpoint_q=_qstr(q, "endpoint"),
                    task=_qstr(q, "task"),
                    only_errors=_qstr(q, "errors") in {"1", "true"},
                    before_id=_qint(q, "before_id"),
                    since_id=_qint(q, "since_id"),
                    limit=_qint(q, "limit") or 100,
                ))
            if path == "/api/event_stats":
                return self._json(200, dashboard_logs.event_stats(conn))
            if path.startswith("/api/event/"):
                try:
                    eid = int(path.rsplit("/", 1)[1])
                except ValueError:
                    return self._json(400, {"error": "bad id"})
                ev = dashboard_logs.get_event(conn, eid)
                if ev is None:
                    return self._json(404, {"error": "not found"})
                return self._json(200, ev)
            if path == "/api/checkpoints":
                return self._json(200, dashboard_logs.list_checkpoints(
                    conn, self.checkpoint_dir,
                ))
            if path.startswith("/api/checkpoint/") and path.endswith("/signature"):
                try:
                    exp_id = int(path.split("/")[3])
                except (ValueError, IndexError):
                    return self._json(400, {"error": "bad id"})
                sig = dashboard_logs.checkpoint_signature(
                    exp_id, self.checkpoint_dir,
                )
                if sig is None:
                    return self._json(404, {"error": "no checkpoint"})
                return self._json(200, sig)
            self._json(404, {"error": f"unknown path {path}"})
        except Exception as e:  # noqa: BLE001
            # Without this a raising handler closes the socket with no
            # response and the browser reports a bare "Failed to fetch";
            # answer with a JSON 500 so the failing endpoint is visible.
            logger.exception("dashboard %s failed", path)
            try:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:  # noqa: BLE001
                pass  # headers already sent — nothing more we can do
        finally:
            if conn is not None:
                conn.close()


def serve(db_path: str, host: str, port: int,
          frozen_arch_dir: str = "",
          checkpoint_dir: str = "") -> None:
    _Handler.db_path = db_path
    _Handler.frozen_arch_dir = frozen_arch_dir
    _Handler.checkpoint_dir = checkpoint_dir
    _Handler.html = _HTML_PATH.read_bytes()
    _Handler.css = _CSS_PATH.read_bytes() if _CSS_PATH.exists() else b""
    _Handler.js = _JS_PATH.read_bytes() if _JS_PATH.exists() else b""
    server = ThreadingHTTPServer((host, port), _Handler)
    logger.info("dashboard listening on http://%s:%d (db=%s)", host, port, db_path)
    print(f"dashboard → http://{host}:{port}/  (db={db_path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="local/radar_local.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--frozen_arch_dir", default="",
        help="Frozen-arch snapshot dir for the ts_data_pipeline task. "
             "Empty = $RADAR_FROZEN_ARCH_DIR or local/frozen_archs.",
    )
    parser.add_argument(
        "--checkpoint_dir", default="",
        help="Checkpoint dir for the Checkpoints tab. "
             "Empty = $RADAR_CHECKPOINT_DIR or local/checkpoints.",
    )
    args = parser.parse_args()
    if not Path(args.db).exists():
        raise SystemExit(f"db not found: {args.db}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(
        args.db, args.host, args.port,
        frozen_arch_dir=args.frozen_arch_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
