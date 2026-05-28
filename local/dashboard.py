"""Read-only HTTP dashboard over ``local/radar_local.db``.

Standalone — does not need the validator to be running. Opens the
SQLite file in ``mode=ro`` so it's safe to point at a db that another
process is writing. Stdlib only, no JS deps. The UI shell lives in
``local/dashboard.html`` and is served at ``/``.

  python -m local.dashboard --db local/radar_local.db --port 8765
  # then open http://127.0.0.1:8765/

Endpoints:
  GET /                       HTML page
  GET /api/stats              {total, successful, best_metric, ...}
  GET /api/leaderboard?n=N    top-N by metric ASC (lower=better)
  GET /api/recent?n=N         latest N by id
  GET /api/frontier           Pareto front on (metric, flops)
  GET /api/frontier_crps_mase Pareto front on (crps, mase) — ts_forecasting only
  GET /api/experiment/<id>    full row (incl. code, loss_curve)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

_HTML_PATH = Path(__file__).with_name("dashboard.html")


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Open the db in read-only mode so concurrent writers aren't blocked."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _row(r: sqlite3.Row, *, with_code: bool = False) -> dict[str, Any]:
    out = {
        "id": r["id"],
        "round_id": r["round_id"],
        "miner_id": r["miner_id"],
        "name": r["name"],
        "metric": r["metric"],
        "score": r["score"],
        "success": bool(r["success"]),
        "objectives": json.loads(r["objectives_json"] or "{}"),
        "analysis": r["analysis"],
        "task": r["task"],
        "generation": r["generation"],
        "prompt_id": r["prompt_id"],
        "timestamp": r["timestamp"],
    }
    if with_code:
        out["code"] = r["code"]
        out["motivation"] = r["motivation"]
        out["reasoning"] = r["reasoning"]
        out["loss_curve"] = json.loads(r["loss_curve_json"] or "[]")
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
    return {
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "best_metric": row["best"],
        "worst_metric": row["worst"],
        "mean_metric": row["mean"],
        "last_round": row["last_round"],
        "n_miners": n_miners,
    }


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
    contribute."""
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


def _experiment(conn: sqlite3.Connection, exp_id: int) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT * FROM experiments WHERE id=?", (exp_id,)
    ).fetchone()
    return _row(r, with_code=True) if r else None


class _Handler(BaseHTTPRequestHandler):
    db_path: str = ""
    html: bytes = b""

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

        if not path.startswith("/api/"):
            return self._json(404, {"error": "not found"})

        conn = _connect_ro(self.db_path)
        try:
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
            if path.startswith("/api/experiment/"):
                try:
                    exp_id = int(path.rsplit("/", 1)[1])
                except ValueError:
                    return self._json(400, {"error": "bad id"})
                exp = _experiment(conn, exp_id)
                if exp is None:
                    return self._json(404, {"error": "not found"})
                return self._json(200, exp)
            self._json(404, {"error": f"unknown path {path}"})
        finally:
            conn.close()


def serve(db_path: str, host: str, port: int) -> None:
    _Handler.db_path = db_path
    _Handler.html = _HTML_PATH.read_bytes()
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
    args = parser.parse_args()
    if not Path(args.db).exists():
        raise SystemExit(f"db not found: {args.db}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(args.db, args.host, args.port)


if __name__ == "__main__":
    main()
