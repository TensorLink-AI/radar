"""Autonomous agent tool definitions — a rich set of tools the LLM can call
to research, generate, validate, and submit model architectures.

The agent decides what to do and when. Tools cover:
  - Research: arxiv search, experiment DB queries
  - Analysis: frontier details, task analysis, FLOPs estimation, layer cost
  - Design: sketch_architecture for cheap iterative prototyping
  - Validation: full AST + FLOPs validation
  - State: scratchpad read/write
  - Control: submit (ends loop), time remaining

None of the tools restrict what the LLM can design — ``estimate_layer_flops``
and ``sketch_architecture`` accept arbitrary PyTorch code and measure what it
actually does. The LLM is free to use any PyTorch operation, standard or
custom.
"""

import copy
import json
import sys
import time

import torch
import torch.nn as nn

from core import validation
from core.arch_knowledge import scan_frontier_ops
from core.flops_estimator import estimate_flops, suggest_resize
from core.history import (
    extract_flops_budget, identify_bucket, load_state, save_state,
    get_history, format_history,
)
from core.input_shape import infer_input
from core.output_shape import infer_output_shape, verify_output_shape
from core import call_with_timeout
from core.trace import trace_architecture, format_trace

TOOL_HTTP_TIMEOUT = 15   # seconds per research/DB request
FRONTIER_HTTP_TIMEOUT = 10  # seconds for frontier fetch

# Circuit breaker for tools that keep returning the same error. The agent
# loop can fixate on a broken tool (e.g. calling get_frontier_member 3-4
# times in a row with invalid indices) without learning. After 2 identical
# error responses, further calls short-circuit to a hard-stop message.
#
# An "error" is either a short response (≤200 chars, typical for compact
# handler errors) OR a longer response that contains one of the error
# markers below. The content check catches verbose error payloads (e.g.
# a multi-line traceback) that would otherwise slip past the length check.
CIRCUIT_BREAKER_ERROR_MAX_LEN = 200
CIRCUIT_BREAKER_THRESHOLD = 2
CIRCUIT_BREAKER_ERROR_MARKERS = (
    "error", "failed", "not found", "timeout", "rejected", "unavailable",
)

# Keyed by tool name. Cleared at the start of each round in ``build_handlers``.
_tool_error_counters: dict = {}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _looks_like_error(result: str) -> bool:
    """True when the tool result looks like an error response.

    Short results are treated as errors (most handler error paths return
    <200 chars), and longer results are scanned for common error markers.
    """
    if len(result) <= CIRCUIT_BREAKER_ERROR_MAX_LEN:
        return True
    low = result.lower()
    return any(m in low for m in CIRCUIT_BREAKER_ERROR_MARKERS)


def _apply_circuit_breaker(tool_name: str, result: str) -> str:
    """Update the circuit-breaker counter for a tool and return either the
    original result or a hard-stop message when the breaker trips.

    Semantics:
    - A result that doesn't look like an error resets the counter.
    - A repeated identical error-looking result increments the counter;
      a different error restarts it at 1.
    - Once the same error has been returned more than
      CIRCUIT_BREAKER_THRESHOLD times in a row, the breaker trips and the
      tool's result is replaced with a hard-stop message telling the LLM
      to try a different approach.
    """
    if not _looks_like_error(result):
        _tool_error_counters.pop(tool_name, None)
        return result

    state = _tool_error_counters.get(tool_name)
    if state is None or state["last"] != result:
        _tool_error_counters[tool_name] = {"last": result, "count": 1}
        return result

    state["count"] += 1
    if state["count"] > CIRCUIT_BREAKER_THRESHOLD:
        n = state["count"]
        _log(f"[tools] circuit breaker tripped on {tool_name} after {n} identical errors")
        return (
            f"[CIRCUIT OPEN] Tool {tool_name} returned identical errors "
            f"{n} times. Do NOT call this tool again this round — try a "
            "different approach."
        )
    return result

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    # ── Research tools ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search arxiv for papers relevant to your architecture design. "
                "Returns titles, abstracts, and key findings. Use this to find "
                "state-of-the-art techniques for the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                        "description": "Search query (e.g. 'efficient <task_name> neural network')",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 20,
                        "default": 10,
                        "description": "Maximum results to return (10-20, default 10)",
                    },
                    "tool": {
                        "type": "string",
                        "enum": ["arxiv", "web"],
                        "default": "arxiv",
                        "description": "Search backend (default 'arxiv')",
                    },
                    "date_filter": {
                        "type": "string",
                        "enum": [
                            "PAST_24_HOURS", "PAST_2_DAYS", "PAST_WEEK",
                            "PAST_2_WEEKS", "PAST_MONTH", "PAST_2_MONTHS",
                            "PAST_YEAR", "PAST_2_YEARS",
                        ],
                        "description": (
                            "Optional recency filter. Omit to default to "
                            "PAST_2_YEARS upstream."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": (
                "Query the experiment database. Read-only HTTP proxy with a "
                "~60 calls/min budget per round. Calls are logged under your "
                "miner ID and rendered on the public dashboard, so plan queries "
                "rather than bulk-dumping.\n\n"
                "Frontier + listing:\n"
                "  GET  /frontier?task={task}             current Pareto frontier (cheapest signal)\n"
                "  GET  /experiments/pareto?task={task}   frontier + near-frontier\n"
                "  GET  /experiments/recent?n=15          most recent N\n"
                "  GET  /experiments/failures?n=5         recent failed runs\n"
                "  GET  /experiments/families?task={task} similar-arch clusters\n"
                "  GET  /experiments/stats?task={task}    aggregate counts\n"
                "  GET  /experiments/tasks                known task names\n"
                "  GET  /challenge                        active round metadata\n\n"
                "Per-experiment:\n"
                "  GET  /experiments/{idx}                full record incl. code\n"
                "  GET  /experiments/{idx}/diff           unified diff vs. parent (credits read)\n"
                "  GET  /experiments/lineage/{idx}        root->idx chain\n"
                "  GET  /experiments/{idx}/lineage_diffs  per-step diffs along the chain\n"
                "  GET  /experiments/diff/{a}/{b}         pair diff — does NOT credit reads\n"
                "  POST /experiments/search   body {\"query\": \"...\"}   substring search\n\n"
                "Provenance:\n"
                "  GET  /provenance/{idx}/influences      likely ancestors\n"
                "  GET  /provenance/{idx}/impact          descendants (fertile designs)\n"
                "  GET  /provenance/{idx}/similar?top_k=5 nearest neighbors — use before submit\n"
                "  GET  /provenance/{idx}/graph           full lineage graph\n"
                "  GET  /provenance/components            inventory + usage counts\n"
                "  GET  /provenance/component_stats       per-component success/score stats\n"
                "  GET  /provenance/dead_ends             lineages that stopped improving\n\n"
                "Notes: responses >256 KB are returned intact but skipped by the "
                "provenance ID extractor (bulk dumps don't credit reads). 4xx/5xx "
                "aren't logged as accesses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method (default GET)",
                    },
                    "path": {
                        "type": "string",
                        "description": "API path, e.g. '/experiments/recent?n=10' or '/provenance/component_stats'",
                    },
                    "body": {
                        "type": "object",
                        "description": "JSON body for POST requests (optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    # ── Analysis tools ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analyze_task",
            "description": (
                "Analyze the current task and return a structured summary: "
                "parsed input shape, parsed output shape, the transformation "
                "needed between them, FLOPs budget per element, and what "
                "operations the frontier uses vs. what hasn't been tried. "
                "Call this FIRST before designing anything — it saves you "
                "from manually parsing the challenge."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_layer_flops",
            "description": (
                "Estimate the FLOPs of a single layer or small module by running "
                "it. Provide a short code snippet that defines the layer and an "
                "input shape — the tool instantiates it, runs a forward pass "
                "with FlopCounterMode, and returns the measured FLOPs plus what "
                "percentage of your budget it consumes. Use this to cost-check "
                "individual building blocks while planning your architecture. "
                "Works with ANY PyTorch module — not limited to standard layers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code that defines and instantiates a module. "
                            "Must assign the module to a variable called `layer`. "
                            "Example: 'import torch.nn as nn\\nlayer = nn.Conv1d(7, 32, 5)' "
                            "or a custom module class."
                        ),
                    },
                    "input_shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Input tensor shape including batch dim, e.g. [1, 512, 7]",
                    },
                },
                "required": ["code", "input_shape"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sketch_architecture",
            "description": (
                "Test a complete architecture sketch by providing a minimal "
                "build_model function. The tool runs it, measures total FLOPs, "
                "traces per-layer shapes and costs, and returns a budget "
                "breakdown table. Use this to validate your design BEFORE "
                "writing full production code with optimizer, hooks, etc. "
                "The code only needs build_model — build_optimizer is auto-generated. "
                "Works with ANY architecture — no restrictions on layer types."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code containing a `def build_model(...)` that "
                            "returns an nn.Module. Does NOT need build_optimizer — "
                            "a default one is added automatically. Write the minimum "
                            "code needed to express your architecture idea."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_frontier",
            "description": (
                "List all frontier members with their metrics and architecture "
                "type. Returns a compact summary — no code. Use "
                "get_frontier_member(index) to retrieve a specific member's "
                "code when you need implementation details."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_frontier_member",
            "description": (
                "Get a specific frontier member's code and metrics by index. "
                "Call list_frontier first to see available members."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Member index from list_frontier output",
                    },
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_model_flops",
            "description": (
                "Estimate the FLOPs of a model code snippet WITHOUT full validation. "
                "Use this for quick checks while iterating on your design. "
                "Returns the estimated FLOPs count and whether it fits the budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code with build_model and build_optimizer",
                    },
                },
                "required": ["code"],
            },
        },
    },
    # ── Validation tools ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "trace_architecture",
            "description": (
                "Run a dummy forward pass through build_model and return an "
                "op-by-op trace: each leaf module's name, class, input shape, "
                "output shape, and parameter count. Memory-efficient — only "
                "shape tuples and ints are captured, never tensor data. Use "
                "this to debug where a shape goes wrong inside the model, or "
                "to verify an architecture behaves as you expect before "
                "submission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code with build_model",
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": (
                            "Maximum trace rows to include in the rendered "
                            "output (default 60). Deeper traces are still "
                            "recorded internally but truncated for display."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_output_shape",
            "description": (
                "Run build_model and a dummy forward pass, then compare the "
                "output tensor's shape against the expected shape parsed from "
                "the task constraints. Catches 'tensor a (X) must match tensor "
                "b (Y)' training failures before submission. Works for any "
                "output rank (2D, 3D, 4D, ...) — the expected shape is "
                "derived from the challenge, never hardcoded."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code with build_model",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_code",
            "description": (
                "Run full validation on your code: syntax check, required functions, "
                "parameter signatures, forbidden imports, and FLOPs bounds. "
                "Returns pass/fail with detailed error messages. "
                "ALWAYS validate before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code to validate",
                    },
                },
                "required": ["code"],
            },
        },
    },
    # ── State tools ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_scratchpad",
            "description": (
                "Load persistent state from scratchpad. Contains history of "
                "your previous submissions across rounds — what you tried, "
                "what worked, what didn't. Check this first each round."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_scratchpad",
            "description": (
                "Save data to persistent scratchpad for future rounds. "
                "Use this to record what you learned, what worked, and "
                "what to try next time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": "Free-form notes to save for future rounds",
                    },
                },
                "required": [],
            },
        },
    },
    # ── Control tools ─────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit your final model code. This ends the loop. "
                "The code MUST have passed validate_code first. "
                "Include a descriptive name and motivation explaining "
                "your design choices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The validated Python code to submit",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short descriptive name for this submission",
                    },
                    "motivation": {
                        "type": "string",
                        "description": "Why you designed the model this way — what you expect to improve",
                    },
                },
                "required": ["code", "name", "motivation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time_remaining",
            "description": (
                "Check how many seconds remain in the time budget. "
                "Plan your work accordingly — if time is low, submit "
                "what you have rather than starting new research."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------

class SubmitSignal(Exception):
    """Raised when the agent calls submit — carries the result out of the loop."""
    def __init__(self, code: str, name: str, motivation: str):
        self.code = code
        self.name = name
        self.motivation = motivation


def build_handlers(client, challenge: dict, scratch_dir, deadline: float) -> dict:
    """Build tool handler callables that close over agent state.

    Each handler returns a string for the LLM to see as tool output.
    The ``submit`` handler raises ``SubmitSignal`` to break the loop.
    """
    db_url = challenge.get("db_url", "")
    desearch_url = challenge.get("desearch_url", "")
    flops_min, flops_max = extract_flops_budget(challenge)
    target_flops = int(flops_max * 0.6) if flops_max else 0

    # Mutable state holder for scratchpad notes
    state_holder = {"state": load_state(scratch_dir) if scratch_dir else {}}

    def _fmt(data) -> str:
        try:
            return json.dumps(data, default=str, indent=2)
        except (TypeError, ValueError):
            return str(data)

    # ── Frontier caching ─────────────────────────────────────────
    # Frontier data is immutable within a round — fetch once and reuse.
    _frontier_cache: dict = {"data": None, "fetched": False}
    # Track which frontier members the LLM has already pulled code for.
    _fetched_members: set = set()
    _MAX_MEMBER_FETCHES = 3  # Enough to study; more just bloats context.

    def _fetch_frontier() -> list:
        """Fetch frontier from DB API, fall back to challenge dict.

        Cached per round — subsequent calls return the stored result
        without making another HTTP request.
        """
        if _frontier_cache["fetched"]:
            return _frontier_cache["data"]

        result: list = []
        if db_url:
            try:
                result = call_with_timeout(
                    client.get_json,
                    args=(f"{db_url.rstrip('/')}/frontier",),
                    timeout=FRONTIER_HTTP_TIMEOUT,
                )
                if not (isinstance(result, list) and result):
                    result = []
            except TimeoutError:
                _log(f"[tools] GET /frontier timed out after {FRONTIER_HTTP_TIMEOUT}s")
                result = []
            except Exception as exc:
                _log(f"[tools] GET /frontier failed: {exc}")
                result = []

        if not result:
            frontier = challenge.get("feasible_frontier", [])
            if not frontier:
                frontier = challenge.get("pareto_frontier", [])
            result = frontier if isinstance(frontier, list) else []

        _frontier_cache["data"] = result
        _frontier_cache["fetched"] = True
        return result

    def _detect_arch_type(code: str) -> str:
        """Infer architecture type by scanning code for operation classes."""
        if not code:
            return "unknown"
        types = []
        if "MultiheadAttention" in code or "TransformerEncoder" in code:
            types.append("attention")
        if "Conv1d" in code or "Conv2d" in code:
            types.append("conv")
        if "GRU" in code or "LSTM" in code:
            types.append("recurrence")
        if not types and "Linear" in code:
            types.append("MLP")
        return "+".join(types) if types else "custom"

    # ── Research handlers ─────────────────────────────────────────

    def search_papers(query: str, max_results: int = 10,
                      tool: str | None = None,
                      date_filter: str | None = None) -> str:
        if not desearch_url:
            return "Paper search unavailable (no desearch_url)."
        try:
            max_results = max(10, min(int(max_results), 20))
        except (TypeError, ValueError):
            max_results = 10
        # Upstream API renamed `max_results` → `count` (min 10) and
        # rejects requests where `date_filter` is missing or "NONE",
        # so always send a valid enum value.
        payload: dict = {
            "query": query,
            "count": max_results,
            "max_results": max_results,
        }
        if tool in ("arxiv", "web"):
            payload["tool"] = tool
        if date_filter not in {
            "PAST_24_HOURS", "PAST_2_DAYS", "PAST_WEEK", "PAST_2_WEEKS",
            "PAST_MONTH", "PAST_2_MONTHS", "PAST_YEAR", "PAST_2_YEARS",
        }:
            date_filter = "PAST_2_YEARS"
        payload["date_filter"] = date_filter
        try:
            resp = call_with_timeout(
                client.post_json,
                args=(f"{desearch_url}/search", payload),
                timeout=TOOL_HTTP_TIMEOUT,
            )
            results = resp.get("results", [])
            if not results:
                return "No papers found."
            lines = []
            for r in results[:max_results]:
                lines.append(f"**{r.get('title', 'untitled')}**")
                abstract = r.get("abstract", "")
                if abstract:
                    lines.append(abstract[:500])
                lines.append("")
            return "\n".join(lines)
        except TimeoutError:
            return f"Paper search timed out after {TOOL_HTTP_TIMEOUT}s."
        except Exception as exc:
            return f"Paper search failed: {exc}"

    def query_db(path: str, method: str = "GET", body: dict | None = None) -> str:
        if not db_url:
            return "Experiment DB unavailable (no db_url in challenge)."
        # Build the full URL: db_url base + path
        base = db_url.rstrip("/")
        # Ensure path starts with /
        if not path.startswith("/"):
            path = "/" + path
        url = f"{base}{path}"
        try:
            if method.upper() == "POST":
                result = call_with_timeout(
                    client.post_json, args=(url, body or {}),
                    timeout=TOOL_HTTP_TIMEOUT,
                )
            else:
                result = call_with_timeout(
                    client.get_json, args=(url,),
                    timeout=TOOL_HTTP_TIMEOUT,
                )
            if isinstance(result, dict) and "error" in result:
                return f"DB returned error: {result['error']}"
            formatted = _fmt(result)
            # Truncate very large responses to avoid flooding context
            if len(formatted) > 8000:
                formatted = formatted[:8000] + "\n... (truncated, try a more specific query)"
            return formatted
        except TimeoutError:
            print(f"[tools] query_db {method} {path} timed out", file=sys.stderr)
            return f"DB query timed out after {TOOL_HTTP_TIMEOUT}s — try a simpler query."
        except Exception as exc:
            print(f"[tools] query_db {method} {path} failed: {exc}",
                  file=sys.stderr)
            return f"DB query failed: {exc}"

    # ── Analysis handlers ─────────────────────────────────────────

    def analyze_task_handler() -> str:
        task = challenge.get("task", {}) or {}
        tp = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        task_name = task.get("name", "unknown")

        lines: list[str] = []
        lines.append(f"## Task: {task_name}")
        if tp:
            lines.append(
                "Task params: "
                + ", ".join(f"{k}={v}" for k, v in tp.items())
            )

        # Input shape
        try:
            input_shape, input_dtype = infer_input(tp, constraints)
            in_elem = 1
            for d in input_shape[1:]:
                in_elem *= d
            dt = "long (token IDs)" if input_dtype == torch.long else "float32"
            lines.append(
                f"\n## Input\n"
                f"Shape (incl. batch): {tuple(input_shape)}\n"
                f"Dtype: {dt}\n"
                f"Elements per sample: {in_elem:,}"
            )
        except Exception as exc:
            in_elem = 0
            lines.append(f"\n## Input\nCould not infer input shape: {exc}")

        # Output shape
        expected = infer_output_shape(tp, constraints)
        if expected is not None:
            out_elem = 1
            for d in expected:
                if d > 0:
                    out_elem *= d
            pretty = "(B, " + ", ".join(
                str(e) if e >= 0 else "?" for e in expected
            ) + ")"
            lines.append(
                f"\n## Output\n"
                f"Expected shape: {pretty}\n"
                f"Resolved elements (non-batch, excluding wildcards): {out_elem:,}"
            )
        else:
            out_elem = 0
            lines.append(
                "\n## Output\nNo parseable output-shape constraint found in "
                "task.constraints. Design the output from the task "
                "description / objectives."
            )

        # Transformation
        if in_elem and out_elem:
            lines.append(
                f"\n## Transformation\n"
                f"Input elements: {in_elem:,}  →  Output elements: {out_elem:,}  "
                f"(ratio {out_elem / max(1, in_elem):.3f}x)"
            )

        # Budget per element
        if flops_max:
            denom = max(1, in_elem) * max(1, out_elem) if (in_elem and out_elem) else max(1, in_elem or out_elem or 1)
            per_elem = target_flops / denom if denom else 0
            lines.append(
                f"\n## FLOPs Budget\n"
                f"Range: [{flops_min:,}, {flops_max:,}]\n"
                f"Target (60% of max): {target_flops:,}\n"
                f"Per input×output element: ~{per_elem:.2f} ops"
            )

        # Frontier gap analysis — lean summary only, no code
        frontier = _fetch_frontier()
        if frontier:
            gap = scan_frontier_ops(frontier)
            lines.append(f"\n## Frontier Ops\n{gap}")
        else:
            lines.append(
                "\n## Frontier Ops\nNo frontier exists — you are "
                "bootstrapping the first model for this task."
            )

        return "\n".join(lines)

    def estimate_layer_flops_handler(code: str, input_shape: list) -> str:
        if not code or not code.strip():
            return "No code provided."
        if not input_shape:
            return "input_shape is required."

        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            return "FlopCounterMode not available in this torch version."

        namespace: dict = {"torch": torch, "nn": nn}
        try:
            exec(compile(code, "<layer_snippet>", "exec"), namespace)
        except Exception as exc:
            return f"Layer code execution failed: {exc}"

        layer = namespace.get("layer")
        if layer is None:
            return (
                "Snippet did not assign a variable named `layer`. "
                "Example: `layer = nn.Conv1d(7, 32, 5)`."
            )
        if not isinstance(layer, nn.Module):
            return f"`layer` is not an nn.Module (got {type(layer).__name__})."

        shape = [int(d) for d in input_shape]
        try:
            dummy = torch.randn(*shape)
        except Exception as exc:
            return f"Failed to build dummy input of shape {shape}: {exc}"

        try:
            layer.eval()
            with torch.no_grad():
                counter = FlopCounterMode(display=False)
                with counter:
                    out = layer(dummy)
                total = int(counter.get_total_flops())
        except Exception as exc:
            return f"Forward pass failed: {exc}"

        out_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else "(non-tensor)"
        param_count = sum(p.numel() for p in layer.parameters())

        pct = (100.0 * total / target_flops) if target_flops else 0.0
        result = [
            f"Measured FLOPs: {total:,}",
            f"Output shape: {out_shape}",
            f"Params: {param_count:,}",
        ]
        if target_flops:
            result.append(
                f"Budget usage: {pct:.1f}% of target {target_flops:,}"
            )
        return "\n".join(result)

    def sketch_architecture_handler(code: str) -> str:
        if not code or not code.strip():
            return "No code provided."

        # Inject a default build_optimizer if the sketch omitted one.
        if "def build_optimizer" not in code:
            code = code.rstrip() + (
                "\n\ndef build_optimizer(model):\n"
                "    import torch\n"
                "    return torch.optim.Adam(model.parameters(), lr=1e-3)\n"
            )

        # Measure total FLOPs + capture output shape
        sink: list = []
        estimated, err = estimate_flops(code, challenge, sink)
        if err:
            return f"Sketch forward pass failed: {err}"

        lines: list[str] = []
        entries, trace_err = trace_architecture(code, challenge)
        if trace_err:
            lines.append(f"(per-layer trace unavailable: {trace_err})")
        elif entries:
            lines.append(format_trace(entries, max_rows=40))
        else:
            lines.append("(no leaf-op entries captured)")

        lines.append("")
        if estimated is not None:
            pct = (100.0 * estimated / target_flops) if target_flops else 0.0
            lines.append(
                f"TOTAL FLOPs: {estimated:,}"
                + (f" ({pct:.0f}% of target {target_flops:,})" if target_flops else "")
            )
            if flops_min and flops_max:
                gate_min = int(flops_min * 0.9)
                gate_max = int(flops_max * 1.1)
                if estimated < gate_min:
                    lines.append(
                        f"STATUS: below hard gate [{gate_min:,}, {gate_max:,}] "
                        "— increase capacity"
                    )
                elif estimated > gate_max:
                    lines.append(
                        f"STATUS: above hard gate [{gate_min:,}, {gate_max:,}] "
                        "— reduce capacity"
                    )
                else:
                    lines.append(
                        f"STATUS: within hard gate [{gate_min:,}, {gate_max:,}]"
                    )

        # Output shape coherence
        task = challenge.get("task", {}) or {}
        tp = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        expected = infer_output_shape(tp, constraints)
        if expected is not None and sink:
            shape_err = verify_output_shape(sink[0], expected)
            actual = tuple(sink[0])
            if shape_err:
                lines.append(f"OUTPUT SHAPE: {actual} — MISMATCH: {shape_err}")
            else:
                lines.append(f"OUTPUT SHAPE: {actual} — matches expected ✓")
        elif sink:
            lines.append(f"OUTPUT SHAPE: {tuple(sink[0])} (no declared expectation)")

        lines.append(
            "\nNote: sketch_architecture is lenient — it does NOT reject on "
            "FLOPs bounds. Use validate_code for the final pre-submission check."
        )
        return "\n".join(lines)

    def list_frontier_handler() -> str:
        frontier = _fetch_frontier()
        if not frontier:
            return "No frontier exists — you are bootstrapping the first model."

        lines = [f"Frontier has {len(frontier)} member(s):\n"]
        for i, member in enumerate(frontier):
            metrics = member.get("objectives", {})
            metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            code = member.get("code", "")
            arch = _detect_arch_type(code)
            lines.append(
                f"  [{i}] {metric_str} | arch={arch} | "
                f"code_len={len(code)}"
            )
        lines.append(
            "\nCall get_frontier_member(index=N) to read a member's code."
        )
        return "\n".join(lines)

    def get_frontier_member_handler(index: int) -> str:
        # ── Dedup: don't re-fetch a member already in context ────
        if index in _fetched_members:
            return (
                f"Frontier member [{index}] was already retrieved this round. "
                "Refer to the earlier tool result instead of re-fetching."
            )

        # ── Cap: limit total member fetches to control context size ──
        if len(_fetched_members) >= _MAX_MEMBER_FETCHES:
            return (
                f"Already fetched {_MAX_MEMBER_FETCHES} frontier members "
                f"(indices {sorted(_fetched_members)}). Use what you have — "
                "additional fetches add context bloat without new insight."
            )

        frontier = _fetch_frontier()
        if not frontier:
            return "No frontier available."
        if index < 0 or index >= len(frontier):
            return (f"Invalid index {index}. Frontier has "
                    f"{len(frontier)} members (0-{len(frontier)-1}).")

        _fetched_members.add(index)

        member = frontier[index]
        metrics = member.get("objectives", {})
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        code = member.get("code", "")
        motivation = member.get("motivation", "")

        if len(code) > 2000:
            code = (code[:2000] +
                    f"\n# ... (truncated — full code is {len(code)} chars)")

        parts = [f"## Frontier Member [{index}]",
                 f"Metrics: {metric_str}"]
        if motivation:
            parts.append(f"Motivation: {motivation}")
        if code:
            parts.append(f"```python\n{code}\n```")
        else:
            parts.append("(no code available)")
        return "\n".join(parts)

    def estimate_model_flops(code: str) -> str:
        estimated, err = estimate_flops(code, challenge)
        if err:
            return f"FLOPs estimation failed: {err}"
        gate_min = int(flops_min * 0.9)
        gate_max = int(flops_max * 1.1)
        status = "WITHIN BUDGET"
        hint = ""
        if estimated < gate_min:
            status = f"TOO LOW (below gate minimum {gate_min:,})"
            hint = suggest_resize(estimated, gate_min, gate_max, target_flops)
        elif estimated > gate_max:
            status = f"TOO HIGH (above gate maximum {gate_max:,})"
            hint = suggest_resize(estimated, gate_min, gate_max, target_flops)
        result = (
            f"Estimated FLOPs: {estimated:,}\n"
            f"Target: {target_flops:,} (60% of max)\n"
            f"Budget: [{flops_min:,}, {flops_max:,}]\n"
            f"Hard gate: [{gate_min:,}, {gate_max:,}]\n"
            f"Status: {status}"
        )
        if hint:
            result += f"\n{hint}"
        return result

    # ── Validation handler ────────────────────────────────────────

    def trace_architecture_handler(code: str, max_rows: int = 60) -> str:
        entries, err = trace_architecture(code, challenge)
        if err:
            return f"Trace failed: {err}"
        if not entries:
            return (
                "Trace produced no entries — the model has no leaf modules "
                "with parameters, or the forward pass produced no tensor "
                "outputs through hooked modules."
            )
        return format_trace(entries, max_rows=max_rows)

    def check_output_shape_handler(code: str) -> str:
        task = challenge.get("task", {}) or {}
        tp = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        expected = infer_output_shape(tp, constraints)
        if expected is None:
            return (
                "No parseable output-shape constraint in task.constraints; "
                "skipping shape check. (This is not an error — the task "
                "simply doesn't declare an expected output shape.)"
            )
        sink: list = []
        _, err = estimate_flops(code, challenge, sink)
        if err:
            return f"Could not run forward pass: {err}"
        if not sink:
            return (
                "Forward pass completed but no tensor output was captured "
                "(possibly via a fallback path). Add a concrete forward() "
                "that returns a torch.Tensor."
            )
        shape_err = verify_output_shape(sink[0], expected)
        if shape_err:
            return f"FAILED: {shape_err}"
        return (
            f"PASSED: Output shape {tuple(sink[0])} matches expected "
            f"(B, {', '.join(str(e) if e >= 0 else '?' for e in expected)})."
        )

    def validate_code_handler(code: str) -> str:
        ok, errors = validation.validate_code(code, challenge)
        if ok:
            # Run a shape check to report the actual output shape
            task = challenge.get("task", {}) or {}
            tp = task.get("task_params", {}) or {}
            constraints = task.get("constraints", []) or []
            expected = infer_output_shape(tp, constraints)

            if expected is not None:
                sink = []
                estimate_flops(code, challenge, sink)
                if sink:
                    shape_str = f"Output shape {tuple(sink[0])} matches expected."
                else:
                    shape_str = "Output shape could not be captured but no mismatch detected."
            else:
                shape_str = "No output shape constraint to check."

            return f"PASSED: Code is valid, FLOPs within budget, {shape_str}"
        return "FAILED:\n" + "\n".join(f"- {e}" for e in errors)

    # ── State handlers ────────────────────────────────────────────

    def read_scratchpad() -> str:
        state = state_holder["state"]
        history_entries = get_history(state)
        notes = state.get("agent_notes", "")
        parts = []
        if notes:
            parts.append(f"## Saved Notes\n{notes}")
        if history_entries:
            parts.append(f"## Submission History\n{format_history(history_entries, max_entries=10)}")
        if not parts:
            return "Scratchpad is empty — this is your first round."
        return "\n\n".join(parts)

    def write_scratchpad(notes: str = "") -> str:
        state = state_holder["state"]
        if notes:
            state["agent_notes"] = notes
        state_holder["state"] = state
        return "Scratchpad updated."

    # ── Control handlers ──────────────────────────────────────────

    def submit_handler(code: str, name: str, motivation: str) -> str:
        # Validate one last time before accepting
        ok, errors = validation.validate_code(code, challenge)
        if not ok:
            return (
                "SUBMIT REJECTED — code failed validation:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nFix the errors and try again."
            )
        raise SubmitSignal(code, name, motivation)

    def time_remaining_handler() -> str:
        remaining = max(0, deadline - time.time())
        if remaining > 60:
            return f"{remaining:.0f} seconds remaining ({remaining / 60:.1f} minutes)."
        elif remaining > 0:
            return f"{remaining:.0f} seconds remaining — wrap up soon!"
        return "Time is up! Submit now or the round will end without a submission."

    # Expose state_holder so the agent loop can persist state after the loop
    submit_handler._state_holder = state_holder

    raw_handlers = {
        "search_papers": search_papers,
        "query_db": query_db,
        "analyze_task": analyze_task_handler,
        "estimate_layer_flops": estimate_layer_flops_handler,
        "sketch_architecture": sketch_architecture_handler,
        "list_frontier": list_frontier_handler,
        "get_frontier_member": get_frontier_member_handler,
        "estimate_model_flops": estimate_model_flops,
        "trace_architecture": trace_architecture_handler,
        "check_output_shape": check_output_shape_handler,
        "validate_code": validate_code_handler,
        "read_scratchpad": read_scratchpad,
        "write_scratchpad": write_scratchpad,
        "submit": submit_handler,
        "time_remaining": time_remaining_handler,
    }

    # ── Circuit breaker wrapping ──────────────────────────────────
    # Reset per-round counters so state from the previous round doesn't
    # bleed into this one. Wrap each handler to short-circuit after 2
    # identical short errors, and to update counters on every result.
    # ``submit`` is exempt — it raises SubmitSignal on success, which
    # must not be swallowed or counted.
    _tool_error_counters.clear()

    def _wrap(tool_name: str, handler):
        if tool_name == "submit":
            return handler

        def wrapped(*args, **kwargs):
            result = str(handler(*args, **kwargs))
            return _apply_circuit_breaker(tool_name, result)
        # Preserve submit_handler-style attributes on any wrapped handler
        for attr in ("_state_holder",):
            if hasattr(handler, attr):
                setattr(wrapped, attr, getattr(handler, attr))
        return wrapped

    return {name: _wrap(name, h) for name, h in raw_handlers.items()}


def build_tools(challenge: dict | None = None) -> list[dict]:
    """Return a tools list customized for this challenge.

    The only customization right now is swapping the placeholder in the
    ``search_papers`` description for the task's actual name, so the LLM
    sees a relevant example query instead of a generic one.
    """
    if not challenge:
        return TOOLS
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "") or "neural network"
    tools = copy.deepcopy(TOOLS)
    for t in tools:
        fn = t.get("function", {})
        if fn.get("name") == "search_papers":
            props = fn.get("parameters", {}).get("properties", {})
            q = props.get("query", {})
            q["description"] = (
                f"Search query (e.g. 'efficient {task_name} neural network')"
            )
    return tools
