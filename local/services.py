"""Localhost HTTP services the validator stands up.

Miner agents get a real ``shared.url_gate.GatedClient`` pointed at this
server, so the local stack matches the distributed agent-harness API
without a real network. Endpoints (handlers in ``experiments_api``):

  GET  /health
  GET  /frontier?task=
  GET  /challenge
  GET  /experiments/recent?n=         (``limit=`` accepted as alias)
  GET  /experiments/pareto?task=
  GET  /experiments/failures?n=
  GET  /experiments/families?task=
  GET  /experiments/stats?task=
  GET  /experiments/tasks
  GET  /experiments/{idx}
  GET  /experiments/{idx}/artifacts
  GET  /experiments/{idx}/diff
  GET  /experiments/{idx}/lineage_diffs
  GET  /experiments/lineage/{idx}
  GET  /experiments/diff/{a}/{b}
  GET  /artifacts?round_id=&miner_id=&task=&kind=&limit=
  GET  /artifacts/{id}                metadata + inline text if available
  GET  /artifacts/{id}/download       raw bytes (text inline or proxied
                                      from object storage)
  POST /experiments/search            body ``{"query": "..."}``
  POST /llm/chat                      legacy ``{content, model}`` shape
  POST /llm/v1/chat/completions       OpenAI-compatible ChatCompletion
  GET  /llm/models
  GET  /llm/v1/models                 OpenAI-compatible model list
  POST /desearch/search
  GET  /wiki                          JSON listing of available files
  GET  /wiki/<path>                   raw markdown content

Provider behavior lives in ``local/providers.py``; experiment-query
logic in ``local/experiments_api.py``.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from local import experiments_api as exp_api
from local.providers import (
    WikiStore, desearch, llm_available_models, llm_dispatch,
    llm_openai_dispatch,
)
from local.store import LocalStore

logger = logging.getLogger("local.services")


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 keep-alive: every response below sets Content-Length, so the
    # client can reuse one warm connection for the whole round instead of
    # opening a fresh TCP socket per call. That matters because this server
    # runs as a daemon thread inside the validator process — while the
    # validator is GIL-bound (torch pretrain + the 97-task GIFT-Eval pass)
    # the accept loop is starved, and the default 5-slot listen backlog
    # overflows, refusing brand-new connections (the miner sees an instant
    # APIConnectionError). Reusing a connection sidesteps that window
    # entirely; ``_Services`` widens the backlog for the unavoidable ones.
    protocol_version = "HTTP/1.1"
    # Bound idle keep-alive connections so a parked socket doesn't pin its
    # handler thread forever (BaseHTTPRequestHandler closes on read timeout).
    timeout = 120

    # Injected by ServicesServer.start.
    store: LocalStore = None  # type: ignore[assignment]
    wiki: WikiStore = None    # type: ignore[assignment]
    sink: object = None       # ArtifactSink | None — set by ServicesServer.start

    def log_message(self, format, *args):  # noqa: A002
        logger.debug("svc %s - %s", self.address_string(), format % args)

    def _json(self, status: int, body) -> None:
        data = json.dumps(body).encode()
        try:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # Client (typically a miner that hit its own timeout) closed
            # the socket before we could reply. Nothing to do.
            logger.debug("svc client disconnected before response sent")

    def _serve_artifact_body(self, artifact_id: int) -> None:
        art = self.store.get_artifact(artifact_id)
        if art is None:
            return self._json(404, {"error": "not found"})
        # Inline text path — return immediately, no R2 round-trip needed.
        if art.get("content_text") is not None:
            data = art["content_text"].encode("utf-8")
            rel = art.get("rel_path") or ""
            ctype = "application/json" if rel.endswith(".json") else (
                "text/x-python" if rel.endswith(".py") else "text/plain; charset=utf-8"
            )
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        # Binary path — proxy from object storage via the sink.
        if self.sink is None or not getattr(self.sink, "r2_enabled", False):
            return self._json(503, {"error": "binary artifact requires R2 backend"})
        body = self.sink.fetch_bytes(art["s3_key"])  # type: ignore[attr-defined]
        if body is None:
            return self._json(502, {"error": "object storage fetch failed"})
        self.send_response(200)
        self.send_header("content-type", "application/octet-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        n = int(self.headers.get("content-length", 0) or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except json.JSONDecodeError:
            return {}

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        q = urllib.parse.parse_qs(parsed.query)

        if path == "/health":
            return self._json(200, {"status": "ok"})

        if path == "/frontier":
            task = q.get("task", [None])[0]
            return self._json(200, exp_api.frontier(self.store, task=task))

        if path == "/challenge":
            return self._json(200, exp_api.active_challenge(self.store))

        # /experiments/* — list/aggregate endpoints first, then the
        # multi-segment paths, then the bare /experiments/{idx}.
        if path == "/experiments/recent":
            n = int(q.get("n", q.get("limit", ["10"]))[0])
            task = q.get("task", [None])[0]
            return self._json(200, exp_api.recent(self.store, n=n, task=task))

        if path == "/experiments/pareto":
            task = q.get("task", [None])[0]
            return self._json(200, exp_api.pareto(self.store, task=task))

        if path == "/experiments/failures":
            n = int(q.get("n", ["5"])[0])
            task = q.get("task", [None])[0]
            return self._json(200, exp_api.failures(self.store, n=n, task=task))

        if path == "/experiments/families":
            task = q.get("task", [None])[0]
            return self._json(200, exp_api.families(self.store, task=task))

        if path == "/experiments/stats":
            task = q.get("task", [None])[0]
            return self._json(200, exp_api.stats(self.store, task=task))

        if path == "/experiments/tasks":
            return self._json(200, exp_api.tasks(self.store))

        if path == "/artifacts":
            def _opt_int(name: str) -> Optional[int]:
                v = q.get(name, [None])[0]
                try:
                    return int(v) if v is not None else None
                except ValueError:
                    return None
            return self._json(200, exp_api.list_artifacts(
                self.store,
                round_id=_opt_int("round_id"),
                miner_id=q.get("miner_id", [None])[0],
                task=q.get("task", [None])[0],
                kind=q.get("kind", [None])[0],
                limit=int(q.get("limit", ["200"])[0] or "200"),
            ))

        if path.startswith("/artifacts/") and path.endswith("/download"):
            try:
                aid = int(path.split("/")[2])
            except (IndexError, ValueError):
                return self._json(400, {"error": "bad id"})
            return self._serve_artifact_body(aid)

        if path.startswith("/artifacts/"):
            try:
                aid = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json(400, {"error": "bad id"})
            art = self.store.get_artifact(aid)
            if art is None:
                return self._json(404, {"error": "not found"})
            return self._json(200, art)

        if path.startswith("/experiments/") and path.endswith("/artifacts"):
            try:
                idx = int(path.split("/")[2])
            except (IndexError, ValueError):
                return self._json(400, {"error": "bad id"})
            return self._json(200, exp_api.artifacts_for_experiment(self.store, idx))

        if path.startswith("/experiments/diff/"):
            parts = path.split("/")
            try:
                a, b = int(parts[3]), int(parts[4])
            except (IndexError, ValueError):
                return self._json(400, {"error": "bad ids"})
            return self._json(200, exp_api.pair_diff(self.store, a, b))

        if path.startswith("/experiments/lineage/"):
            try:
                idx = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json(400, {"error": "bad id"})
            return self._json(200, exp_api.lineage(self.store, idx))

        if path.startswith("/experiments/") and path.endswith("/diff"):
            try:
                idx = int(path.split("/")[2])
            except (IndexError, ValueError):
                return self._json(400, {"error": "bad id"})
            return self._json(200, exp_api.parent_diff(self.store, idx))

        if path.startswith("/experiments/") and path.endswith("/lineage_diffs"):
            try:
                idx = int(path.split("/")[2])
            except (IndexError, ValueError):
                return self._json(400, {"error": "bad id"})
            return self._json(200, exp_api.lineage_diffs(self.store, idx))

        if path.startswith("/experiments/"):
            try:
                idx = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json(400, {"error": "bad id"})
            exp = self.store.get_experiment(idx)
            if exp is None:
                return self._json(404, {"error": "not found"})
            return self._json(200, exp)

        if path == "/llm/models":
            return self._json(200, {"models": llm_available_models()})

        if path == "/llm/v1/models":
            now = 0
            return self._json(200, {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": now,
                     "owned_by": "local"}
                    for m in llm_available_models()
                ],
            })

        if path == "/wiki":
            return self._json(200, {"files": self.wiki.list()})

        if path.startswith("/wiki/"):
            data = self.wiki.read(path[len("/wiki/"):])
            if data is None:
                return self._json(404, {"error": "not found"})
            try:
                self.send_response(200)
                self.send_header("content-type", "text/markdown; charset=utf-8")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("svc client disconnected before wiki body sent")
            return

        self._json(404, {"error": f"unknown path {path}"})

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        payload = self._read_body()

        if path == "/experiments/search":
            return self._json(
                200, exp_api.search(self.store, payload.get("query", "")),
            )

        if path == "/llm/chat":
            try:
                return self._json(200, llm_dispatch(payload))
            except Exception as e:  # noqa: BLE001
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})

        if path == "/llm/v1/chat/completions":
            try:
                return self._json(200, llm_openai_dispatch(payload))
            except Exception as e:  # noqa: BLE001
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})

        if path == "/desearch/search":
            try:
                return self._json(200, desearch(payload))
            except Exception as e:  # noqa: BLE001
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})

        self._json(404, {"error": f"unknown path {path}"})


class _Services(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` tuned for a starved accept loop.

    The server thread shares a process (and the GIL) with the validator's
    training/eval work, so the accept loop can stall for long stretches.
    A 5-slot listen backlog (the stdlib default) overflows in that window
    and the kernel refuses new connections; widening it lets bursts queue
    instead of bouncing. Daemon threads keep ``stop()`` from blocking on
    in-flight keep-alive handlers.
    """

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


class ServicesServer:
    """Threaded HTTP server bound to 127.0.0.1.

    Used by the validator as a context manager — ``start()`` returns
    the bound URL, ``stop()`` shuts it down.
    """

    def __init__(self, store: LocalStore, wiki_dir: Optional[str],
                 port: int = 0, sink: object = None):
        self.store = store
        self.wiki = WikiStore(wiki_dir)
        self.sink = sink
        self._port = port
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._url: str = ""

    @property
    def url(self) -> str:
        return self._url

    def start(self) -> str:
        port = self._port or _pick_free_port()
        store = self.store
        wiki = self.wiki
        sink = self.sink

        class BoundHandler(_Handler):
            pass

        BoundHandler.store = store
        BoundHandler.wiki = wiki
        BoundHandler.sink = sink

        self._httpd = _Services(("127.0.0.1", port), BoundHandler)
        self._url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True,
            name="local-services",
        )
        self._thread.start()
        logger.info("services listening on %s", self._url)
        return self._url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._httpd = None
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
