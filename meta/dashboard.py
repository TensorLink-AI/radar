"""Read-only HTTP dashboard over the meta-orchestrator state.

Mirrors ``local/dashboard.py``'s pattern: stdlib HTTP server, JSON API
+ a static HTML page that fetches the JSON and renders it.

Endpoints:
    /                       — HTML
    /api/experiments        — registry rows
    /api/experiment/<name>  — registry row + units
    /api/xcompare           — cross-experiment digest
    /api/events?experiment=…&limit=…  — recent orchestrator events

Bound to 127.0.0.1 by default — same posture as ``local/dashboard``.
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import xquery
from .store import ExperimentStore, Registry, experiment_dir


logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    server_version = "RadarMeta/0.1"
    data_root: Path  # set by build_server

    # Silence the default per-request stderr noise.
    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        logger.debug(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = url.path
        qs = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if path == "/":
                self._serve_html()
            elif path == "/api/experiments":
                self._json(self._experiments())
            elif path.startswith("/api/experiment/"):
                name = path.rsplit("/", 1)[-1]
                self._json(self._experiment(name))
            elif path == "/api/xcompare":
                self._json(xquery.xcompare_digest(self.data_root))
            elif path == "/api/events":
                self._json(self._events(
                    experiment=qs.get("experiment", ""),
                    limit=int(qs.get("limit", "100")),
                ))
            else:
                self._not_found()
        except FileNotFoundError as e:
            self._not_found(str(e))
        except Exception:
            logger.exception("dashboard handler error")
            self._error()

    # ── handlers ────────────────────────────────────────────────────────

    def _experiments(self) -> dict:
        reg = Registry(self.data_root / "registry.db")
        try:
            return {"experiments": [dict(r) for r in reg.list_experiments()]}
        finally:
            reg.close()

    def _experiment(self, name: str) -> dict:
        reg = Registry(self.data_root / "registry.db")
        try:
            row = reg.get_experiment(name)
        finally:
            reg.close()
        if row is None:
            raise FileNotFoundError(name)
        store = ExperimentStore(Path(row["path"]) / "meta.db")
        try:
            units = [dict(u) for u in store.list_units()]
            best = [dict(b) for b in store.best_per_unit()]
        finally:
            store.close()
        return {"experiment": dict(row), "units": units, "best": best}

    def _events(self, *, experiment: str, limit: int) -> dict:
        if not experiment:
            return {"events": []}
        path = experiment_dir(self.data_root, experiment) / "meta.db"
        if not path.exists():
            raise FileNotFoundError(experiment)
        store = ExperimentStore(path)
        try:
            rows = [dict(r) for r in store.recent_events(limit)]
        finally:
            store.close()
        return {"events": rows}

    # ── serialization ──────────────────────────────────────────────────

    def _json(self, data: dict) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self) -> None:
        html = (HERE / "dashboard.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _not_found(self, msg: str = "not found") -> None:
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(msg.encode("utf-8"))

    def _error(self) -> None:
        self.send_response(500)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"internal error")


def build_server(
    data_root: Path, host: str = "127.0.0.1", port: int = 8766
) -> ThreadingHTTPServer:
    handler_cls = type("BoundHandler", (Handler,), {"data_root": data_root})
    return ThreadingHTTPServer((host, port), handler_cls)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Meta-orchestrator dashboard")
    p.add_argument("--data-root", type=Path, default=HERE / "data")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="[meta-dashboard] %(message)s")
    srv = build_server(args.data_root, args.host, args.port)
    logger.info("serving on http://%s:%d/", args.host, args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
