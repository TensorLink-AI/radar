"""OpenAI-SDK LLM client for the miner agent.

Points at the validator's OpenAI-compatible LLM proxy. The proxy handles
auth (X-Agent-Token), rate limiting, and forwarding to the upstream
inference provider.

Patterns borrowed from a proven agent codebase:
  - Cached singleton client (avoids per-call TCP handshake cost)
  - Transient-error classification (retries 429/5xx/timeout, fails fast
    on 4xx)
  - Model pool with rotation — if one model is slow, the next attempt
    uses a different one
  - Exponential backoff on transient errors
"""
from __future__ import annotations

import os
import sys
import time
from typing import Iterable

import httpx
from openai import (
    OpenAI,
    APIError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
)

# Transient error substrings — covers SDK errors and passthrough HTTP
# errors that have already been stringified.
_TRANSIENT_TOKENS = (
    "rate limit", "429", "500", "502", "503", "504", "529",
    "timeout", "timed out", "connection", "overloaded",
    "eof", "reset by peer", "broken pipe", "unreachable",
)


def _is_transient(exc: BaseException) -> bool:
    """True when an exception looks worth retrying.

    Retryable: SDK timeout/connection/rate-limit subclasses, plus any
    APIError whose ``status_code`` is a known transient HTTP status.
    Anything else (4xx other than 408/409/429, validation errors,
    bad payloads) is non-transient — burning retry budget on it just
    delays the inevitable failure.
    """
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if status in (408, 409, 429, 500, 502, 503, 504, 529):
            return True
    s = str(exc).lower()
    return any(t in s for t in _TRANSIENT_TOKENS)


def _log(msg: str) -> None:
    # stdout is reserved for the final proposal JSON the harness parses —
    # all diagnostic output goes to stderr.
    print(msg, file=sys.stderr, flush=True)


_cached_client: OpenAI | None = None
_cached_config: tuple | None = None


def get_client(
    llm_url: str = "",
    agent_token: str = "",
    miner_uid: str = "",
) -> OpenAI:
    """Return a cached OpenAI client pointed at the validator proxy.

    The caller (agent) should pass ``challenge["llm_url"]`` and
    ``challenge["agent_token"]`` — those are the authoritative sources.
    The harness injects the token into the challenge dict, not into the
    pod environment, so reading from env alone yields an unauthenticated
    client and every request returns 403.

    The ``LLM_URL`` / ``AGENT_TOKEN`` env vars are kept as fallbacks so
    scripts and manual invocations still work. The cache key includes the
    resolved URL, token, uid, and timeout, so changing any of them builds
    a fresh client.
    """
    global _cached_client, _cached_config

    url = llm_url or os.environ.get("LLM_URL", "")
    token = agent_token or os.environ.get("AGENT_TOKEN", "")
    if not url and not token:
        raise RuntimeError(
            "No LLM URL or agent token: challenge['llm_url'] and "
            "challenge['agent_token'] both empty, and LLM_URL / AGENT_TOKEN "
            "env vars unset"
        )
    if not url:
        raise RuntimeError(
            "No LLM URL: challenge['llm_url'] empty and LLM_URL env var unset"
        )
    if not token:
        raise RuntimeError(
            "No agent token: challenge['agent_token'] empty and AGENT_TOKEN "
            "env var unset"
        )

    base_url = url.rstrip("/") + "/v1"
    uid = miner_uid or os.environ.get("MINER_UID", "0")
    read_timeout = int(os.environ.get("LLM_READ_TIMEOUT", "180"))

    config = (base_url, token, uid, read_timeout)
    if _cached_client is not None and _cached_config == config:
        return _cached_client

    http_client = httpx.Client(
        timeout=httpx.Timeout(
            connect=30.0,
            read=float(read_timeout),
            write=60.0,
            pool=60.0,
        ),
    )

    _cached_client = OpenAI(
        base_url=base_url,
        api_key="unused-but-required",  # SDK demands non-empty; proxy ignores
        default_headers={
            "X-Agent-Token": token,
            "X-Miner-UID": str(uid),
        },
        http_client=http_client,
        max_retries=1,  # proxy already retries; don't double up
    )
    _cached_config = config
    _log(
        f"[llm_client] initialised base_url={base_url} "
        f"read_timeout={read_timeout}s"
    )
    return _cached_client


def chat(
    messages: list[dict],
    *,
    llm_url: str = "",
    agent_token: str = "",
    miner_uid: str = "",
    model: str | Iterable[str] = "moonshotai/Kimi-K2.5-TEE",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    max_retries: int = 3,
    base_delay: float = 2.0,
    deadline: float | None = None,
):
    """Chat completion with retry, backoff, and optional model pool rotation.

    Args:
        messages: OpenAI-format message list.
        model: single model string OR an iterable of model strings to rotate
            through across retries (next attempt picks the next model).
        deadline: absolute monotonic-clock value; if set, abort before
            sleeping past it instead of blowing the round budget.

    Returns the raw ``ChatCompletion`` object — the caller pulls out
    ``.choices[0].message`` (and ``.tool_calls`` if any).
    """
    client = get_client(llm_url, agent_token, miner_uid)
    pool = [model] if isinstance(model, str) else list(model)
    if not pool:
        raise ValueError("model pool is empty")

    last_exc: BaseException | None = None

    for attempt in range(max_retries):
        current = pool[attempt % len(pool)]
        kwargs = {
            "model": current,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kwargs)
            dt = time.monotonic() - t0
            finish = "?"
            try:
                finish = resp.choices[0].finish_reason
            except (AttributeError, IndexError):
                pass
            _log(
                f"[llm_client] ok model={current} dt={dt:.1f}s "
                f"finish={finish}"
            )
            return resp
        except Exception as exc:
            last_exc = exc
            dt = time.monotonic() - t0
            _log(
                f"[llm_client] attempt {attempt + 1}/{max_retries} "
                f"model={current} failed after {dt:.1f}s: "
                f"{type(exc).__name__}: {exc}"
            )
            if not _is_transient(exc) or attempt >= max_retries - 1:
                break
            delay = base_delay * (2 ** attempt)
            if deadline is not None and time.monotonic() + delay > deadline:
                _log(f"[llm_client] no time left for backoff ({delay}s)")
                break
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc
