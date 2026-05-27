"""Localhost HTTP services the validator stands up.

Miner agents get a real ``shared.url_gate.GatedClient`` pointed at this
server, so the local stack matches the distributed agent-harness API
without a real network. Endpoints:

  GET  /health
  GET  /experiments/recent?limit=N
  GET  /experiments/{id}
  GET  /frontier
  POST /llm/chat                      — native shape ``{content, model}``
  GET  /llm/models                    — native list
  POST /llm/v1/chat/completions       — OpenAI-compatible alias
  GET  /llm/v1/models                 — OpenAI-compatible alias
  POST /desearch/search
  GET  /wiki                — JSON listing of available files
  GET  /wiki/<path>         — raw markdown content

Provider behavior lives in ``local/providers.py``.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from local.providers import (
    WikiStore, desearch, llm_available_models, llm_dispatch,
)
from local.scoring import compute_pareto
from local.store import LocalStore

logger = logging.getLogger("local.services")


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Handler(BaseHTTPRequestHandler):
    # Injected by ServicesServer.start.
    store: LocalStore = None  # type: ignore[assignment]
    wiki: WikiStore = None    # type: ignore[assignment]

    def log_message(self, format, *args):  # noqa: A002
        logger.debug("svc %s - %s", self.address_string(), format % args)

    def _json(self, status: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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

        if path == "/experiments/recent":
            limit = int(q.get("limit", ["10"])[0])
            return self._json(200, self.store.recent_experiments(n=limit))

        if path.startswith("/experiments/"):
            try:
                idx = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json(400, {"error": "bad id"})
            exp = self.store.get_experiment(idx)
            if exp is None:
                return self._json(404, {"error": "not found"})
            return self._json(200, exp)

        if path == "/frontier":
            all_exps = self.store.recent_experiments(n=10_000)
            return self._json(200, {"frontier": compute_pareto(all_exps)})

        if path == "/llm/models":
            return self._json(200, {"models": llm_available_models()})

        if path == "/llm/v1/models":
            data = [{"id": m, "object": "model"}
                    for m in llm_available_models()]
            return self._json(200, {"object": "list", "data": data})

        if path == "/wiki":
            return self._json(200, {"files": self.wiki.list()})

        if path.startswith("/wiki/"):
            data = self.wiki.read(path[len("/wiki/"):])
            if data is None:
                return self._json(404, {"error": "not found"})
            self.send_response(200)
            self.send_header("content-type", "text/markdown; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self._json(404, {"error": f"unknown path {path}"})

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        payload = self._read_body()

        if path == "/llm/chat":
            try:
                return self._json(200, llm_dispatch(payload))
            except Exception as e:  # noqa: BLE001
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})

        if path == "/llm/v1/chat/completions":
            try:
                native = llm_dispatch(payload)
            except Exception as e:  # noqa: BLE001
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})
            envelope = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": native.get("model", payload.get("model", "stub")),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": native.get("content", ""),
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            return self._json(200, envelope)

        if path == "/desearch/search":
            try:
                return self._json(200, desearch(payload))
            except Exception as e:  # noqa: BLE001
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})

        self._json(404, {"error": f"unknown path {path}"})


class ServicesServer:
    """Threaded HTTP server bound to 127.0.0.1.

    Used by the validator as a context manager — ``start()`` returns
    the bound URL, ``stop()`` shuts it down.
    """

    def __init__(self, store: LocalStore, wiki_dir: Optional[str],
                 port: int = 0):
        self.store = store
        self.wiki = WikiStore(wiki_dir)
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

        class BoundHandler(_Handler):
            pass

        BoundHandler.store = store
        BoundHandler.wiki = wiki

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), BoundHandler)
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
