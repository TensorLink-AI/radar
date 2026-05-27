"""LLM reasoning engine — uses GatedClient to call the validator-provided LLM.

Key design decisions:
  - Up to 15 LLM calls per round (uses half the 30-request rate limit,
    leaving headroom for future enhancements)
  - Multi-turn conversation: each retry feeds back validation errors so the
    LLM can self-correct across turns
  - No internal retries within chat() — caller manages retry logic
  - Graceful degradation: returns None on failure, never raises
"""

import json
import sys
import time

from core import call_with_timeout
from core.validation import validate_code
from core.prompt_builder import build_system_prompt, build_user_prompt

DEFAULT_MODEL = "moonshotai/Kimi-K2.5-TEE"
MAX_LLM_ATTEMPTS = 15  # up to 15 turns — half the 30-request rate limit
# Must exceed the validator proxy's 60s upstream-read timeout, plus buffer
# for reasoning models that take a few extra seconds to produce their first
# token. The old 45s was shorter than the proxy window, so the miner gave up
# on calls that would have succeeded.
LLM_REQUEST_TIMEOUT = 180

# Exponential-backoff base delay between retry attempts.
RETRY_BASE_DELAY = 2.0

# Substrings (lowercased) that mark an exception as a retryable transient
# error — network/overload/timeout. Everything else (4xx, validation, bad
# payload, etc.) is raised immediately because retrying won't help.
TRANSIENT_TOKENS = (
    "rate limit", "429", "500", "502", "503", "504", "529",
    "timeout", "timed out", "connection", "overloaded",
    "eof", "reset by peer", "broken pipe", "unreachable",
)


def _is_transient(exc: BaseException) -> bool:
    """True when the exception string looks like a retryable transient.

    Covers OS-level socket failures, HTTP 5xx/429, and common overload
    markers. A 4xx (auth, bad-request, validation) is intentionally NOT
    transient — burning retry budget on them just delays the real failure.
    """
    s = str(exc).lower()
    return any(tok in s for tok in TRANSIENT_TOKENS)


def chat(client, llm_url: str, messages: list[dict], *,
         temperature: float = 0.7, max_tokens: int = 4096,
         model: str = DEFAULT_MODEL) -> str:
    """Single LLM call. No internal retries — caller manages retry logic."""
    if not llm_url:
        raise RuntimeError("No llm_url provided")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print(f"[llm] calling {llm_url}/v1/chat/completions model={model} "
          f"msgs={len(messages)} temp={temperature}", file=sys.stderr)

    resp = call_with_timeout(
        client.post_json,
        args=(f"{llm_url}/v1/chat/completions", payload),
        timeout=LLM_REQUEST_TIMEOUT,
    )
    content = resp["choices"][0]["message"]["content"]
    print(f"[llm] response received: {len(content)} chars", file=sys.stderr)
    return content


def chat_with_tools(
    client,
    llm_url: str,
    messages: list[dict],
    tools: list[dict],
    tool_handlers: dict,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_rounds: int = 8,
    model: str = DEFAULT_MODEL,
    deadline: float | None = None,
) -> str:
    """Multi-round tool-calling loop using the OpenAI chat completions format.

    Sends ``tools`` definitions alongside messages. When the assistant responds
    with ``tool_calls``, each call is dispatched to the matching handler in
    *tool_handlers*, and the results are appended as ``role: tool`` messages
    before the next request.  The loop continues until the model returns a plain
    text response (``finish_reason == "stop"``) or *max_rounds* is exhausted.

    Unlike ``chat()``, this function retries transient HTTP failures internally
    (up to 2 attempts per round) so callers don't need their own retry wrapper.
    Pass ``deadline`` (absolute time.time() value) to skip backoff sleeps that
    would blow the round's time budget.
    """
    if not llm_url:
        raise RuntimeError("No llm_url provided")

    url = f"{llm_url}/v1/chat/completions"
    # Retry only on transient network/overload errors. Non-transient failures
    # (4xx, bad payload) raise on first attempt — no point burning budget.
    max_retries = 2

    for round_idx in range(max_rounds):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # On the final round, omit tools to force a text response.
        if round_idx < max_rounds - 1 and tools:
            payload["tools"] = tools

        print(
            f"[llm] tool-call round {round_idx + 1}/{max_rounds} "
            f"msgs={len(messages)} temp={temperature}",
            file=sys.stderr,
        )

        resp = None
        for attempt in range(max_retries):
            try:
                resp = call_with_timeout(
                    client.post_json, args=(url, payload),
                    timeout=LLM_REQUEST_TIMEOUT,
                )
                break
            except Exception as exc:
                transient = _is_transient(exc)
                is_last = attempt >= max_retries - 1
                if transient and not is_last:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    remaining = (
                        (deadline - time.time()) if deadline is not None
                        else float("inf")
                    )
                    if remaining > delay + 10:
                        print(
                            f"[llm] round {round_idx + 1} attempt "
                            f"{attempt + 1} transient: {exc} — "
                            f"sleeping {delay:.0f}s",
                            file=sys.stderr,
                        )
                        time.sleep(delay)
                        continue
                    print(
                        f"[llm] round {round_idx + 1} attempt "
                        f"{attempt + 1} transient but no time to retry "
                        f"({remaining:.0f}s left): {exc}",
                        file=sys.stderr,
                    )
                    raise
                if transient:
                    print(
                        f"[llm] round {round_idx + 1} attempt "
                        f"{attempt + 1} transient (final): {exc}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[llm] round {round_idx + 1} attempt "
                        f"{attempt + 1} non-transient: {exc}",
                        file=sys.stderr,
                    )
                raise

        if resp is None:
            raise RuntimeError(
                f"LLM call returned no response after {max_retries} attempts"
            )

        choice = resp["choices"][0]
        assistant_msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        tool_calls = assistant_msg.get("tool_calls")

        # --- No tool calls → return text content ---
        # NOTE: we intentionally do NOT short-circuit on
        # ``finish_reason == "stop"``. Some OpenAI-compatible servers (notably
        # certain Kimi deployments) return ``finish_reason="stop"`` together
        # with a populated ``tool_calls`` list; branching on finish_reason
        # would silently drop those tool calls.
        if not tool_calls:
            content = assistant_msg.get("content") or ""
            print(
                f"[llm] final response: {len(content)} chars "
                f"(finish_reason={finish_reason})",
                file=sys.stderr,
            )
            return content

        # --- Process tool calls ---
        messages.append(assistant_msg)

        for tc in tool_calls:
            func = tc["function"]
            fn_name = func["name"]
            call_id = tc["id"]

            try:
                kwargs = json.loads(func.get("arguments") or "{}")
            except json.JSONDecodeError:
                kwargs = {}

            handler = tool_handlers.get(fn_name)
            if handler is None:
                result_str = f"Tool error: unknown tool '{fn_name}'"
                print(f"[llm] unknown tool: {fn_name}", file=sys.stderr)
            else:
                try:
                    result = handler(**kwargs)
                    result_str = str(result)
                except Exception as exc:
                    result_str = f"Tool error: {exc}"
                    print(
                        f"[llm] tool '{fn_name}' raised: {exc}", file=sys.stderr
                    )

            print(
                f"[llm] tool {fn_name} → {len(result_str)} chars",
                file=sys.stderr,
            )
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": result_str}
            )

    # All rounds exhausted.
    return assistant_msg.get("content") or ""


def extract_code(text: str) -> str:
    """Extract the first fenced python code block from LLM output."""
    markers = ["```python", "```Python", "```py"]
    for marker in markers:
        if marker in text:
            start = text.index(marker) + len(marker)
            closing = text.find("```", start)
            if closing == -1:
                return text[start:].strip()
            return text[start:closing].strip()
    # Fallback: bare triple-backtick block
    if "```" in text:
        start = text.index("```") + 3
        nl = text.find("\n", start)
        if nl == -1:
            return text[start:].strip()
        start = nl + 1
        closing = text.find("```", start)
        if closing == -1:
            return text[start:].strip()
        return text[start:closing].strip()
    # Fallback: raw Python that looks like valid code
    if "def build_model" in text and "import" in text:
        return text.strip()
    return ""


def reason_and_generate(client, challenge: dict,
                        context: dict) -> tuple[str, str, str] | None:
    """Use the LLM to reason about architecture design and generate code.

    Returns (code, name, motivation) if LLM produced valid code, else None.
    Uses up to 15 LLM calls with multi-turn error correction. Each failed
    attempt feeds validation errors back to the LLM so it can self-correct.
    Returns as soon as valid code is produced.
    """
    llm_url = challenge.get("llm_url", "")
    if not llm_url:
        return None

    system_prompt = build_system_prompt(challenge)
    user_prompt = build_user_prompt(challenge, context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    code = ""
    name = "reliable_baseline_submission"
    motivation = "LLM-designed architecture"

    for attempt in range(MAX_LLM_ATTEMPTS):
        print(f"[llm] attempt {attempt + 1}/{MAX_LLM_ATTEMPTS}",
              file=sys.stderr)

        try:
            response = chat(client, llm_url, messages, temperature=0.7)
            code = extract_code(response)

            # Try to extract name/motivation from response
            for line in response.split("\n"):
                stripped = line.strip()
                if stripped.startswith("# Name:"):
                    name = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("# Motivation:"):
                    motivation = stripped.split(":", 1)[1].strip()

            ok, errors = validate_code(code, challenge)
            if ok:
                print(f"[llm] validation passed on attempt {attempt + 1}",
                      file=sys.stderr)
                return code, name, motivation

            # Validation failed — add error feedback for retry
            print(f"[llm] validation errors: {errors}", file=sys.stderr)
            if attempt < MAX_LLM_ATTEMPTS - 1:
                if code:
                    messages.append(
                        {"role": "assistant", "content": f"```python\n{code}\n```"})
                    messages.append({"role": "user", "content": (
                        "The code has validation errors:\n"
                        + "\n".join(f"- {e}" for e in errors)
                        + "\n\nFix these errors and return the corrected code. "
                        "Remember: build_model and build_optimizer MUST be "
                        "top-level def statements (not inside a class)."
                    )})
                else:
                    tp = challenge.get("task", {}).get("task_params", {})
                    param_str = ", ".join(tp.keys()) if tp else "**task_params"
                    messages.append({"role": "user", "content": (
                        "Previous attempt failed: no Python code block found. "
                        "You MUST respond with a single ```python code block "
                        f"containing def build_model({param_str}) and def build_optimizer(model)."
                    )})

        except Exception as exc:
            transient = _is_transient(exc)
            print(
                f"[llm] attempt {attempt + 1} error "
                f"({'transient' if transient else 'non-transient'}): {exc}",
                file=sys.stderr,
            )
            # Non-transient errors (4xx, bad payload, etc.) won't improve by
            # retrying — stop immediately so the caller can fall back.
            if not transient:
                print("[llm] non-transient error — aborting retry loop",
                      file=sys.stderr)
                break
            if attempt < MAX_LLM_ATTEMPTS - 1:
                messages.append({"role": "user", "content": (
                    "Previous request failed. Please try again with a single "
                    "```python code block containing build_model and "
                    "build_optimizer functions."
                )})

    print("[llm] all attempts exhausted, returning None", file=sys.stderr)
    return None
