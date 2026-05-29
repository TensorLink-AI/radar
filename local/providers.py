"""LLM, arxiv, and wiki providers for the local services server.

Kept in a separate module so ``local/services.py`` stays focused on
HTTP plumbing. Provider rules:

* ``CHUTES_API_KEY`` set → proxy to Chutes AI (OpenAI-compatible).
* ``OPENAI_API_KEY`` set → proxy to OpenAI Chat Completions.
* Otherwise → deterministic stub so agents can still test the call path.

Arxiv calls the public Atom API at ``export.arxiv.org`` (no key).
Wiki reads markdown from a local directory.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


# ── LLM ────────────────────────────────────────────────────────

# Matches the real validator's llm_proxy default — Chutes serves the
# OpenAI-compatible API at /v1/chat/completions.
CHUTES_BASE_URL = "https://llm.chutes.ai/v1"
CHUTES_DEFAULT_MODEL = "moonshotai/Kimi-K2.6-TEE"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _openai_compat_chat(base_url: str, api_key: str, payload: dict,
                        default_model: str) -> dict:
    """Hit any OpenAI-compatible /chat/completions endpoint with Bearer
    auth and return the upstream JSON verbatim. Used for both Chutes and
    OpenAI."""
    model = payload.get("model") or default_model
    forwarded = {
        "model": model,
        "messages": payload.get("messages") or [],
        "temperature": float(payload.get("temperature", 0.7)),
        "max_tokens": int(payload.get("max_tokens", 1024)),
    }
    # Pass tool-calling fields through when present — the OpenAI SDK
    # miners rely on this to drive multi-round tool loops.
    for key in ("tools", "tool_choice", "response_format"):
        if key in payload and payload[key] is not None:
            forwarded[key] = payload[key]
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(forwarded).encode(), method="POST",
    )
    req.add_header("authorization", f"Bearer {api_key}")
    req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())


def _llm_chutes(payload: dict) -> dict:
    return _openai_compat_chat(
        CHUTES_BASE_URL, os.environ["CHUTES_API_KEY"],
        payload, CHUTES_DEFAULT_MODEL,
    )


def _llm_openai(payload: dict) -> dict:
    return _openai_compat_chat(
        OPENAI_BASE_URL, os.environ["OPENAI_API_KEY"],
        payload, "gpt-4o-mini",
    )


def _stub_text(payload: dict) -> str:
    msgs = payload.get("messages") or []
    last = msgs[-1].get("content", "") if msgs else ""
    return (
        "[local stub LLM — no API key set] "
        f"received {len(msgs)} message(s); last_len={len(last)}. "
        "Set CHUTES_API_KEY (or OPENAI_API_KEY) for real responses."
    )


def _llm_stub_openai(payload: dict) -> dict:
    """OpenAI ChatCompletion-shaped stub response."""
    return {
        "id": f"chatcmpl-stub-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or "stub",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": _stub_text(payload)},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def llm_openai_dispatch(payload: dict) -> dict:
    """Return an OpenAI ChatCompletion-shaped response.

    This is what the OpenAI-SDK miner clients (and the raw httpx
    autonomous miner) expect when they POST to ``/llm/v1/chat/completions``.
    """
    if os.environ.get("CHUTES_API_KEY"):
        return _llm_chutes(payload)
    if os.environ.get("OPENAI_API_KEY"):
        return _llm_openai(payload)
    return _llm_stub_openai(payload)


def llm_dispatch(payload: dict) -> dict:
    """Legacy ``/llm/chat`` shape: ``{content, model}``.

    Wraps the OpenAI dispatcher and extracts the assistant text so older
    callers keep working.
    """
    data = llm_openai_dispatch(payload)
    text = (
        data.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    return {"content": text, "model": data.get("model", "")}


def llm_available_models() -> list[str]:
    if os.environ.get("CHUTES_API_KEY"):
        return [
            CHUTES_DEFAULT_MODEL,
            "deepseek-ai/DeepSeek-R1",
            "moonshotai/Kimi-K2-Instruct",
        ]
    if os.environ.get("OPENAI_API_KEY"):
        return ["gpt-4o-mini", "gpt-4o"]
    return ["stub"]


# ── Desearch (real SN22 API + arxiv fallback) ──────────────────

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

DESEARCH_DEFAULT_URL = "https://api.desearch.ai"
DESEARCH_AI_PATH = "/desearch/ai/search"
DESEARCH_DATE_FILTERS = {
    "PAST_24_HOURS", "PAST_2_DAYS", "PAST_WEEK", "PAST_2_WEEKS",
    "PAST_MONTH", "PAST_2_MONTHS", "PAST_YEAR", "PAST_2_YEARS",
}

# Miner tool wrappers POST to /desearch/search with a ~15s client deadline
# (``TOOL_HTTP_TIMEOUT`` in miners/*/tools.py) and give up + retry once it
# elapses. The server's upstream call therefore has to come back *under*
# that budget, otherwise the miner times out before we can answer and
# desearch "never works". We bound the upstream call below the client
# deadline (env-overridable via ``RADAR_DESEARCH_TIMEOUT``).
DESEARCH_UPSTREAM_TIMEOUT = float(
    os.environ.get("RADAR_DESEARCH_TIMEOUT", "12")
)


def _urlopen_bounded(req, timeout: float) -> bytes:
    """``urlopen`` with the connect phase bounded too.

    ``urllib``'s ``timeout=`` only caps the read phase once the TCP
    connection is established — it does NOT cap the connect phase, so an
    unreachable or slow-to-accept host (the norm behind a locked-down
    egress policy) can hang for ~75-127s regardless. We pin
    ``socket.setdefaulttimeout`` for the duration of the call so the
    connect phase is bounded too, then restore the previous default.
    """
    prev = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    finally:
        socket.setdefaulttimeout(prev)


def _desearch_remote(payload: dict, base_url: str, api_key: str) -> dict:
    """Forward to the production Desearch SN22 AI-search endpoint.

    Maps the local payload shape (``{query, count, max_results, tool,
    date_filter}``) onto the upstream ``/desearch/ai/search`` schema and
    normalises the response into ``{"results": [{title, abstract, url}]}``
    so the miner's tool wrapper can consume it unchanged.
    """
    prompt = (
        payload.get("prompt")
        or payload.get("query")
        or ""
    ).strip()
    if not prompt:
        return {"results": []}

    raw_n = payload.get("count", payload.get("max_results", 10))
    try:
        count = max(1, min(20, int(raw_n)))
    except (TypeError, ValueError):
        count = 10

    # The miner sends `tool` (singular string: "arxiv" | "web"). Upstream
    # expects a `tools` array of source identifiers.
    tools = payload.get("tools")
    if not tools:
        tool = (payload.get("tool") or "").lower()
        if tool == "arxiv":
            tools = ["arxiv"]
        elif tool == "web":
            tools = ["web"]
        else:
            tools = ["web"]

    date_filter = payload.get("date_filter")
    if date_filter not in DESEARCH_DATE_FILTERS:
        date_filter = "PAST_2_YEARS"

    body = {
        "prompt": prompt,
        "tools": tools,
        "count": count,
        "date_filter": date_filter,
        "streaming": False,
    }
    if payload.get("model"):
        body["model"] = payload["model"]

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{DESEARCH_AI_PATH}",
        data=json.dumps(body).encode(),
        method="POST",
    )
    # Desearch expects the raw key in the Authorization header — no
    # "Bearer " prefix (per the official desearch-py SDK).
    req.add_header("Authorization", api_key)
    req.add_header("Content-Type", "application/json")
    data = json.loads(_urlopen_bounded(req, DESEARCH_UPSTREAM_TIMEOUT))

    return {"results": _normalise_desearch_results(data, count)}


def _normalise_desearch_results(data, limit: int) -> list[dict]:
    """Coerce a desearch response into ``[{title, abstract, url}, ...]``.

    The upstream schema bundles results under various keys depending on
    which tools ran (``organic_results``, ``arxiv_search``,
    ``web_search``, ``results``, ``sources`` …). We look at the obvious
    candidates and pull whatever we can — anything we can't map shows
    up as untitled with the raw URL.
    """
    if not isinstance(data, dict):
        return []

    candidates: list = []
    for key in (
        "results", "organic_results", "arxiv_search", "web_search",
        "sources", "links", "data",
    ):
        v = data.get(key)
        if isinstance(v, list):
            candidates.extend(v)

    out: list[dict] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        out.append({
            "title": (
                item.get("title")
                or item.get("name")
                or "untitled"
            ),
            "abstract": (
                item.get("abstract")
                or item.get("summary")
                or item.get("snippet")
                or item.get("description")
                or item.get("text")
                or ""
            ),
            "arxiv_id": item.get("arxiv_id", ""),
            "url": (
                item.get("url")
                or item.get("link")
                or item.get("href")
                or ""
            ),
        })
        if len(out) >= limit:
            break
    return out


def desearch(payload: dict) -> dict:
    """Local desearch endpoint.

    Resolution order:

    1. If ``RADAR_DESEARCH_SN22_URL`` and ``DESEARCH_API_KEY`` are set,
       forward to the production Desearch SN22 ``/desearch/ai/search``
       endpoint.
    2. Otherwise fall back to the public arxiv Atom API (works for
       prompts about ML papers, requires egress to export.arxiv.org).
    3. On any network/HTTP error, return ``{"results": [], "error":
       "..."}`` so the miner sees "no papers found" instead of a 502.
    """
    sn22_url = os.environ.get("RADAR_DESEARCH_SN22_URL", "").strip()
    api_key = os.environ.get("DESEARCH_API_KEY", "").strip()
    if sn22_url and api_key:
        try:
            return _desearch_remote(payload, sn22_url, api_key)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"results": [], "error": f"{type(exc).__name__}: {exc}"}
        except (json.JSONDecodeError, ValueError) as exc:
            return {"results": [], "error": f"{type(exc).__name__}: {exc}"}

    # Accept the upstream `count` alias for `max_results` so callers
    # written against the production desearch API work unchanged.
    query = (payload.get("query") or payload.get("prompt") or "").strip()
    if not query:
        return {"results": []}
    raw_n = payload.get("max_results", payload.get("count", 5))
    try:
        max_results = max(1, min(20, int(raw_n)))
    except (TypeError, ValueError):
        max_results = 5
    qs = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    })
    try:
        req = urllib.request.Request(f"{ARXIV_API}?{qs}", method="GET")
        body = _urlopen_bounded(req, DESEARCH_UPSTREAM_TIMEOUT)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Sandbox / offline environments routinely block egress to
        # export.arxiv.org. Degrade to an empty result set instead of
        # raising — the miner already handles "no papers found" cleanly,
        # whereas a 502 just triggers a retry loop and a fallback round.
        return {"results": [], "error": f"{type(exc).__name__}: {exc}"}
    root = ET.fromstring(body)
    results = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        title = (entry.findtext("atom:title", "", ARXIV_NS) or "").strip()
        summary = (entry.findtext("atom:summary", "", ARXIV_NS) or "").strip()
        arxiv_id = (entry.findtext("atom:id", "", ARXIV_NS) or "").strip()
        link = ""
        for ln in entry.findall("atom:link", ARXIV_NS):
            if ln.get("type") == "text/html":
                link = ln.get("href", "")
                break
        results.append({
            "title": title,
            "abstract": summary,
            "arxiv_id": arxiv_id,
            "url": link or arxiv_id,
        })
    return {"results": results}


# ── Wiki ───────────────────────────────────────────────────────

class WikiStore:
    """File-backed markdown wiki. Empty/missing dir → empty listing."""

    def __init__(self, root: Optional[str]):
        self.root: Optional[Path] = Path(root).resolve() if root else None

    def list(self) -> list[dict]:
        if not self.root or not self.root.is_dir():
            return []
        out: list[dict] = []
        for p in sorted(self.root.rglob("*.md")):
            rel = p.relative_to(self.root)
            out.append({"path": str(rel), "size": p.stat().st_size})
        return out

    def read(self, rel_path: str) -> Optional[bytes]:
        if not self.root or not self.root.is_dir():
            return None
        target = (self.root / rel_path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            # path-traversal attempt
            return None
        if not target.is_file():
            return None
        return target.read_bytes()
