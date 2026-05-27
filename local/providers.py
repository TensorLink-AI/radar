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
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


# ── LLM ────────────────────────────────────────────────────────

# Matches the real validator's llm_proxy default — Chutes serves the
# OpenAI-compatible API at /v1/chat/completions.
CHUTES_BASE_URL = "https://llm.chutes.ai/v1"
CHUTES_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3-0324"
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


# ── Arxiv (Desearch shim) ──────────────────────────────────────

def desearch(payload: dict) -> dict:
    query = (payload.get("query") or "").strip()
    if not query:
        return {"results": []}
    max_results = max(1, min(20, int(payload.get("max_results", 5))))
    qs = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    })
    with urllib.request.urlopen(f"{ARXIV_API}?{qs}", timeout=20) as resp:
        body = resp.read()
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
