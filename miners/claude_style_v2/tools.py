"""Tool definitions and handlers for the OpenAI-SDK agent.

Matches the autonomous agent's tool surface: research (arxiv + DB),
analysis (task + FLOPs + layer probes), design (sketch), validation
(validate / trace / check output), state (scratchpad read/write),
control (submit, time_remaining).

Each tool entry follows the OpenAI function-calling schema. Handlers are
built per-round via :func:`build_handlers` so they close over the round's
challenge dict, HTTP client, scratch directory, and deadline. When a
dependency is missing (e.g. the client is ``None`` during unit tests),
the affected handler returns a short "unavailable" string rather than
raising.

A simple per-tool circuit breaker stops the LLM from calling the same
tool with the same failing args over and over.
"""
from __future__ import annotations

import copy
import io
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from core import call_with_timeout
from core.arch_knowledge import scan_frontier_ops
from core.flops_estimator import estimate_flops, suggest_resize
from core.history import (
    HYPOTHESIS_VERDICTS, MAX_MACRO_STEPS, MAX_MACROS, add_hypothesis,
    add_macro, add_note, add_submission, extract_flops_budget,
    find_candidate, find_macro, format_history, format_notes,
    format_scratchpad_summary, get_history, get_macros, get_submissions,
    link_hypothesis, load_state, mark_candidate_submitted,
    mark_candidate_validated, save_state, upsert_candidate,
)
from core.input_shape import infer_input
from core.output_shape import infer_output_shape, verify_output_shape
from core.sizing import MAX_PROBES, sweep_sizes
from core.trace import format_trace, trace_architecture
from core.validation import validate_code

TOOL_HTTP_TIMEOUT = 15       # seconds per research/DB request
FRONTIER_HTTP_TIMEOUT = 10   # seconds for frontier fetch

# Wall-clock window (seconds) before deadline within which a `submit` call
# actually ships. Earlier submits stash the candidate as the running "best
# so far" and the LLM is told to keep iterating. This stops the agent from
# ending at minute 4 of a 30-minute budget.
EARLY_SUBMIT_GATE_SECONDS = 300

# Scratchpad file tool limits. The scratchpad upload itself is capped by
# ``challenge["scratchpad_max_mb"]``; these caps keep a single round from
# burning the whole budget on one file.
FILE_MAX_BYTES = 256 * 1024           # 256 KiB per file
FILE_MAX_COUNT = 50                   # total user-written files
FILE_SEARCH_MAX_MATCHES_PER_FILE = 5  # keep hits-per-file output bounded
FILE_SEARCH_MAX_TOTAL_MATCHES = 30    # global cap across all files
# ``state.json`` is owned by the history module and the ``_scratchpad``
# notes file is owned by write_scratchpad — don't let the LLM clobber
# either from the generic file tools.
RESERVED_FILE_NAMES = frozenset({"state.json"})

# Cognition wiki tarball is fetched once per round and cached in scratch.
COGNITION_WIKI_CACHE_DIR = "_cognition_wiki"  # subdir of scratch_dir
COGNITION_WIKI_FETCH_TIMEOUT = 20  # seconds
COGNITION_WIKI_MAX_BYTES = 5 * 1024 * 1024  # 5 MB sanity cap


def _ensure_wiki_extracted(client, url: str, scratch_dir: str | None) -> Path | None:
    """Fetch the wiki tarball once per round, extract it, return the dir.

    Cached across calls in the same round via scratch_dir. Returns None
    if any step fails — caller must handle gracefully.
    """
    if not url or client is None or scratch_dir is None:
        return None
    cache_dir = Path(scratch_dir) / COGNITION_WIKI_CACHE_DIR
    if cache_dir.exists() and (cache_dir / "_index.md").is_file():
        return cache_dir
    try:
        data = call_with_timeout(
            client.get, args=(url,), timeout=COGNITION_WIKI_FETCH_TIMEOUT,
        )
        if not isinstance(data, (bytes, bytearray)):
            return None
        if len(data) > COGNITION_WIKI_MAX_BYTES:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                target = (cache_dir / member.name).resolve()
                if (cache_dir.resolve() not in target.parents
                        and target != cache_dir.resolve()):
                    return None
            tar.extractall(cache_dir)
        if not (cache_dir / "_index.md").is_file():
            return None
        return cache_dir
    except (TimeoutError, tarfile.TarError, OSError, Exception):
        return None


# ── Schema (OpenAI function-calling format) ───────────────────────────

TOOLS: list[dict] = [
    # ── Research ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Arxiv search by query. Returns titles + abstracts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                        "description": (
                            "Search query (e.g. 'efficient <task_name> "
                            "neural network')"
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 20,
                        "default": 10,
                        "description": "Maximum results (10-20, default 10)",
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
                "Experiment DB (read-only). ~60 calls/min budget; calls are "
                "logged under your miner ID and shown on the public dashboard.\n"
                "Frontier+listing: /frontier?task=, /experiments/pareto?task=, "
                "/experiments/recent?n=, /experiments/failures?n=, "
                "/experiments/families?task=, /experiments/stats?task=, "
                "/experiments/tasks, /challenge.\n"
                "Per-experiment: /experiments/{idx}, "
                "/experiments/{idx}/diff (credits read), "
                "/experiments/lineage/{idx}, "
                "/experiments/{idx}/lineage_diffs, "
                "/experiments/diff/{a}/{b} (does NOT credit reads), "
                "POST /experiments/search body {\"query\": \"...\"}.\n"
                "Provenance: /provenance/{idx}/similar?top_k= "
                "(novelty check before submit), "
                "/provenance/{idx}/influences, /provenance/{idx}/impact, "
                "/provenance/{idx}/graph, /provenance/components, "
                "/provenance/component_stats, /provenance/dead_ends.\n"
                "Responses >256 KB are returned but don't credit reads."
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
                        "description": (
                            "API path, e.g. '/experiments/recent?n=10'"
                        ),
                    },
                    "body": {
                        "type": "object",
                        "description": "JSON body for POST (optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cognition_wiki_index",
            "description": (
                "Curated task-specific design-recipe wiki TOC. Try "
                "before search_papers."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cognition_wiki_read",
            "description": "Read one wiki entry by slug from cognition_wiki_index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {
                        "type": "string",
                        "description": (
                            "Entry slug from cognition_wiki_index "
                            "(e.g. 'patchtst', 'revin', "
                            "'medium-model-recipe')"
                        ),
                    },
                },
                "required": ["slug"],
            },
        },
    },
    # ── Analysis ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analyze_task",
            "description": (
                "Task spec, params, constraints, objectives, FLOPs "
                "budget as JSON."
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
            "name": "estimate_layer_flops",
            "description": (
                "Forward-pass FLOPs for one layer/module. Code must "
                "assign to `layer`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code that defines and instantiates a "
                            "module. Must assign to `layer`. Example: "
                            "'import torch.nn as nn\\nlayer = nn.Conv1d(7, 32, 5)'"
                        ),
                    },
                    "input_shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Input tensor shape incl. batch dim, e.g. [1, 512, 7]"
                        ),
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
                "Probe a build_model: FLOPs, per-layer trace, "
                "output-shape check. Cheaper than validate_code. "
                "Returns a candidate_id (cand_<hex>) accepted by "
                "validate_code and submit in place of source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code containing a `def build_model(...)` "
                            "that returns an nn.Module. No need to include "
                            "build_optimizer — one is added automatically."
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
            "name": "estimate_flops",
            "description": "Run build_model and measure forward-pass FLOPs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Full Python source containing build_model."
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
                "List frontier members (index + objectives). Use "
                "get_frontier_member for code."
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
                "One frontier member (code + metrics) by index from "
                "list_frontier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "idx": {
                        "type": "integer",
                        "description": "Member index from list_frontier.",
                    },
                },
                "required": ["idx"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "size_to_flops",
            "description": (
                "Sweep one scalar knob (use {{SIZE}} placeholder in "
                "code) to land inside the FLOPs gate. Runs up to "
                f"{MAX_PROBES} measurements; returns best (size, "
                "flops) pair."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code_template": {
                        "type": "string",
                        "description": (
                            "Python source containing `build_model` "
                            "with a `{{SIZE}}` placeholder. The "
                            "placeholder is substituted with each "
                            "probe's integer. Example: "
                            "`hidden = {{SIZE}}` inside build_model."
                        ),
                    },
                    "size_min": {
                        "type": "integer",
                        "description": (
                            "Smallest size to probe (must be > 0)."
                        ),
                    },
                    "size_max": {
                        "type": "integer",
                        "description": (
                            "Largest size to probe (must be >= size_min)."
                        ),
                    },
                    "target_flops": {
                        "type": "integer",
                        "description": (
                            "Target FLOPs count. Defaults to 60% of the "
                            "challenge's max_flops_equivalent when 0 or "
                            "omitted."
                        ),
                    },
                },
                "required": ["code_template", "size_min", "size_max"],
            },
        },
    },
    # ── Validation ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "trace_architecture",
            "description": (
                "Op-by-op trace of build_model: name, class, input "
                "shape, output shape, param count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Full Python source containing build_model."
                        ),
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": (
                            "Max trace rows to include in the rendered "
                            "output (default 60)."
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
                "Run build_model and compare output shape against the "
                "task constraint."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Full Python source containing build_model."
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
            "name": "validate_code",
            "description": (
                "Final pre-submission check (syntax + build_model/"
                "build_optimizer + FLOPs gate + output shape). "
                "Required before submit. Accepts a candidate_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Full Python source to validate. Optional "
                            "when ``candidate_id`` is supplied."
                        ),
                    },
                    "candidate_id": {
                        "type": "string",
                        "description": (
                            "Optional. ID of a stored candidate "
                            "(returned by ``sketch_architecture``). "
                            "When provided, the code is loaded from "
                            "state and the ``code`` argument is "
                            "ignored."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    # ── Composing tools ─────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "define_macro",
            "description": (
                "Save a named tool sequence for replay. Steps: "
                "{tool, args, output_to?}; refer to args via "
                "${args.X} and prior outputs via ${var}. submit, "
                "define_macro, run_macro can't appear inside. Up to "
                f"{MAX_MACRO_STEPS} steps, {MAX_MACROS} macros."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Macro name (alnum / underscore). "
                            "Re-using a name overwrites the macro."
                        ),
                    },
                    "sequence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string"},
                                "args": {"type": "object"},
                                "output_to": {"type": "string"},
                            },
                            "required": ["tool"],
                        },
                        "description": (
                            "Ordered list of steps. Example: "
                            "[{\"tool\": \"list_frontier\", "
                            "\"args\": {}, \"output_to\": \"fl\"}]"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Optional human-readable description, "
                            "shown by list_macros."
                        ),
                    },
                },
                "required": ["name", "sequence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_macro",
            "description": (
                "Execute a saved macro by name. Halts on the first "
                "error and returns the concatenated step outputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Macro name to execute.",
                    },
                    "args": {
                        "type": "object",
                        "description": (
                            "Optional dict of run-time arguments "
                            "referenced inside the macro via "
                            "``${args.foo}``."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_macros",
            "description": "List saved macros (name, description, steps).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ── State ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_my_submissions",
            "description": (
                "Your prior submissions, newest first, with full "
                "code (n=1) or truncated to ~40 lines (n>1)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 3,
                        "description": (
                            "Most-recent N submissions to show "
                            "(default 3). ``n=1`` returns full code "
                            "for the latest; larger values truncate "
                            "per entry."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_scratchpad",
            "description": (
                "Prior-round notes (hypotheses, dead_ends, "
                "observations) + submission history."
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
                "Record one note: hypothesis (optional candidate_id "
                "link), dead_end + reason, or observation. Or "
                "attach inline via submit(note=...)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": (
                            "Appended to open_hypotheses. Example: "
                            "'Try depthwise-sep convs for the same "
                            "receptive field at lower FLOPs.'"
                        ),
                    },
                    "candidate_id": {
                        "type": "string",
                        "description": (
                            "Optional. Only meaningful with "
                            "`hypothesis`. Links the hypothesis to "
                            "a candidate (e.g. 'cand_a3f24c1d') so "
                            "the next round's read_scratchpad can "
                            "show what the candidate's outcome was. "
                            "If a hypothesis with the same text "
                            "already exists, the id is appended to "
                            "its candidate_ids list."
                        ),
                    },
                    "dead_end": {
                        "type": "string",
                        "description": (
                            "Short name of an approach that failed. "
                            "Pair with `reason`."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the dead_end failed.",
                    },
                    "observation": {
                        "type": "string",
                        "description": (
                            "A task-agnostic fact learned this round "
                            "(e.g. 'output shape inferred as (B, N, K)'). "
                            "Appended to task_observations."
                        ),
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "(DEPRECATED) Free-form notes string. "
                            "Prefer hypothesis / dead_end / observation."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_hypothesis",
            "description": (
                "Late-bind a candidate_id or verdict (supported/"
                "refuted/inconclusive) to an existing hypothesis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": (
                            "Exact text of an existing hypothesis. "
                            "Call read_scratchpad to see what's "
                            "stored."
                        ),
                    },
                    "candidate_id": {
                        "type": "string",
                        "description": (
                            "Optional. Candidate id to attach. "
                            "Appended to the hypothesis's "
                            "candidate_ids list (deduped)."
                        ),
                    },
                    "verdict": {
                        "type": "string",
                        "enum": list(HYPOTHESIS_VERDICTS),
                        "description": (
                            "Optional. Manual judgement. Sets "
                            "outcome.verdict; score / rank are left "
                            "null until the next round's "
                            "previous_results fills them in."
                        ),
                    },
                },
                "required": ["hypothesis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the scratchpad directory.",
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
            "name": "read_file",
            "description": "Read a scratchpad file by basename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "File name (basename only, no slashes or "
                            "parent references). Example: 'design.md'."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a text file to the scratchpad directory "
                "(persists across rounds; overwrites)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "File name (basename only, no slashes or "
                            "parent references). Example: 'design.md'."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents as text.",
                    },
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Case-insensitive substring search across scratchpad "
                "files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Substring to search for (case-insensitive)."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    # ── Control ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Ship validated code in one call. Pass code OR a "
                "candidate_id from validate_code. Optional note "
                "records a scratchpad entry inline (hypothesis if "
                "candidate_id is set, else observation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Final validated Python source. Optional "
                            "when ``candidate_id`` is supplied."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Short descriptive name for this submission."
                        ),
                    },
                    "motivation": {
                        "type": "string",
                        "description": (
                            "Why this design — what it improves."
                        ),
                    },
                    "candidate_id": {
                        "type": "string",
                        "description": (
                            "Optional. ID of a stored candidate to "
                            "ship (e.g. 'cand_a3f24c1d'). When "
                            "provided, the candidate's stored code is "
                            "submitted and the ``code`` argument is "
                            "ignored."
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "Optional. Scratchpad note recorded before "
                            "shipping. When ``candidate_id`` is also "
                            "supplied the note is stored as a "
                            "hypothesis linked to that candidate so "
                            "next round's read_scratchpad can resolve "
                            "the outcome; otherwise it's stored as a "
                            "task observation. Use this to skip a "
                            "separate write_scratchpad call."
                        ),
                    },
                },
                "required": ["name", "motivation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time_remaining",
            "description": "Seconds left in this round.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── Circuit breaker ───────────────────────────────────────────────────

# After this many identical error responses in a row, subsequent calls to
# the same tool short-circuit to a hard-stop message instead of running.
CIRCUIT_BREAKER_THRESHOLD = 2
CIRCUIT_BREAKER_ERROR_MAX_LEN = 200
_CIRCUIT_BREAKER_ERROR_MARKERS = (
    "error", "errors:", "failed", "not found", "rejected", "out of range",
    "unavailable", "timeout", "timed out",
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _looks_like_error(result: str) -> bool:
    if len(result) <= CIRCUIT_BREAKER_ERROR_MAX_LEN:
        return True
    low = result.lower()
    return any(m in low for m in _CIRCUIT_BREAKER_ERROR_MARKERS)


def _make_circuit_breaker():
    """Per-round circuit breaker state. Returned breaker takes
    ``(tool_name, result_str)`` and returns the original result OR a
    hard-stop message if the same error has now repeated past threshold.
    """
    counters: dict = {}

    def apply(tool_name: str, result: str) -> str:
        if not _looks_like_error(result):
            counters.pop(tool_name, None)
            return result
        state = counters.get(tool_name)
        if state is None or state["last"] != result:
            counters[tool_name] = {"last": result, "count": 1}
            return result
        state["count"] += 1
        if state["count"] > CIRCUIT_BREAKER_THRESHOLD:
            n = state["count"]
            _log(
                f"[tools] circuit breaker tripped on {tool_name} "
                f"after {n} identical errors"
            )
            return (
                f"[CIRCUIT OPEN] Tool {tool_name} returned identical errors "
                f"{n} times. Do NOT call this tool again — try a different "
                "approach."
            )
        return result

    return apply


# ── Submit signal ─────────────────────────────────────────────────────

class SubmitSignal(Exception):
    """Raised by the submit handler to terminate the loop with a result."""

    def __init__(self, code: str, name: str, motivation: str):
        super().__init__(f"submit:{name}")
        self.code = code
        self.name = name
        self.motivation = motivation


# ── Handlers ──────────────────────────────────────────────────────────

def build_handlers(
    challenge: dict,
    *,
    client=None,
    scratch_dir: str | None = None,
    deadline: float | None = None,
    state: dict | None = None,
) -> dict:
    """Build a ``{name: callable(**kwargs) -> str}`` mapping for the round.

    ``client`` is a ``GatedClient``-compatible object (has
    ``get_json``/``post_json``) used for paper search and DB queries.
    ``scratch_dir`` is a filesystem path where scratchpad state lives.
    ``deadline`` is the monotonic wall-clock the round must finish by;
    ``time_remaining`` reports against it. ``state``, when provided,
    overrides the on-disk load — used by callers that need to merge
    round-result feedback before the handlers see the state.

    Any of these may be ``None`` — the affected tools return an
    "unavailable" string instead of raising. Handlers are wrapped with
    the circuit breaker so the LLM can't burn tool budget calling the
    same broken tool over and over.
    """
    breaker = _make_circuit_breaker()
    # Per-round telemetry — every wrapped handler bumps this. The agent
    # reads it at end-of-round to log a usage summary. Handlers see it
    # via closure only; nothing inside a handler should mutate it.
    _call_counts: dict[str, int] = {}

    db_url = (challenge.get("db_url") or "").rstrip("/")
    desearch_url = (challenge.get("desearch_url") or "").rstrip("/")
    flops_min, flops_max = extract_flops_budget(challenge)
    target_flops = int(flops_max * 0.6) if flops_max else 0
    round_id = str(
        challenge.get("round_id")
        or challenge.get("challenge_id")
        or ""
    )

    if state is None:
        state = load_state(scratch_dir) if scratch_dir else {}
    # ``wrote_this_round`` is informational only — read by ``_submit``
    # to log a stderr warning when shipping without a fresh note.
    state_holder: dict = {
        "state": state,
        "wrote_this_round": False,
    }

    def _frontier() -> list:
        frontier = challenge.get("feasible_frontier") or []
        if not frontier:
            frontier = challenge.get("pareto_frontier") or []
        return frontier if isinstance(frontier, list) else []

    def _fmt(data) -> str:
        try:
            return json.dumps(data, default=str, indent=2)
        except (TypeError, ValueError):
            return str(data)

    # ── Research ─────────────────────────────────────────────────

    def _search_papers(query: str = "", max_results: int = 10,
                       tool: str | None = None,
                       date_filter: str | None = None,
                       **_kwargs) -> str:
        if not query:
            return "error: empty query"
        if client is None or not desearch_url:
            return "paper search unavailable (no client or desearch_url)"
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
            results = resp.get("results", []) if isinstance(resp, dict) else []
            if not results:
                return "no papers found"
            lines = []
            for r in results[:max_results]:
                lines.append(f"**{r.get('title', 'untitled')}**")
                abstract = r.get("abstract", "")
                if abstract:
                    lines.append(abstract[:500])
                lines.append("")
            return "\n".join(lines)
        except TimeoutError:
            return f"paper search timed out after {TOOL_HTTP_TIMEOUT}s"
        except Exception as exc:
            return f"paper search failed: {exc}"

    def _query_db(path: str = "", method: str = "GET",
                  body: dict | None = None, **_kwargs) -> str:
        if not path:
            return "error: empty path"
        if client is None or not db_url:
            return "experiment DB unavailable (no client or db_url)"
        if not path.startswith("/"):
            path = "/" + path
        url = f"{db_url}{path}"
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
            if len(formatted) > 8000:
                formatted = (
                    formatted[:8000]
                    + "\n... (truncated, try a more specific query)"
                )
            return formatted
        except TimeoutError:
            return f"DB query timed out after {TOOL_HTTP_TIMEOUT}s"
        except Exception as exc:
            return f"DB query failed: {exc}"

    wiki_url = (challenge.get("cognition_wiki_url") or "").strip()

    def _cognition_wiki_index(**_kwargs) -> str:
        if not wiki_url:
            return "cognition wiki not published for this task"
        wiki_dir = _ensure_wiki_extracted(client, wiki_url, scratch_dir)
        if wiki_dir is None:
            return "cognition wiki unavailable (fetch or extract failed)"
        try:
            return (wiki_dir / "_index.md").read_text(encoding="utf-8")
        except OSError as exc:
            return f"cognition wiki index read failed: {exc}"

    def _cognition_wiki_read(slug: str = "", **_kwargs) -> str:
        if not slug:
            return "error: empty slug"
        safe = "".join(c for c in slug if c.isalnum() or c in "-_")
        if not safe or safe != slug:
            return (
                "error: slug must be alphanumeric, hyphen, or underscore only"
            )
        if not wiki_url:
            return "cognition wiki not published for this task"
        wiki_dir = _ensure_wiki_extracted(client, wiki_url, scratch_dir)
        if wiki_dir is None:
            return "cognition wiki unavailable (fetch or extract failed)"
        entry = wiki_dir / f"{safe}.md"
        if not entry.is_file():
            return (
                f"slug '{safe}' not found in wiki — call "
                "cognition_wiki_index for the list"
            )
        try:
            return entry.read_text(encoding="utf-8")
        except OSError as exc:
            return f"cognition wiki read failed: {exc}"

    # ── Analysis ─────────────────────────────────────────────────

    def _analyze_task(**_kwargs) -> str:
        task = challenge.get("task", {}) or {}
        summary = {
            "name": task.get("name"),
            "description": task.get("description"),
            "task_params": task.get("task_params", {}),
            "constraints": task.get("constraints", []),
            "objectives": task.get("objectives", []),
            "anti_patterns": task.get("anti_patterns", []),
            "flops_range": [flops_min, flops_max],
        }
        frontier = _frontier()
        if frontier:
            summary["frontier_ops"] = scan_frontier_ops(frontier)
        return json.dumps(summary, indent=2, default=str)

    def _estimate_layer_flops(code: str = "",
                              input_shape: list | None = None,
                              **_kwargs) -> str:
        if not code or not code.strip():
            return "error: empty code"
        if not input_shape:
            return "error: input_shape is required"
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            return "error: FlopCounterMode not available in this torch version"

        namespace: dict = {"torch": torch, "nn": nn}
        try:
            exec(compile(code, "<layer_snippet>", "exec"), namespace)
        except Exception as exc:
            return f"layer code execution failed: {exc}"

        layer = namespace.get("layer")
        if layer is None:
            return (
                "error: snippet did not assign a variable named `layer`. "
                "Example: `layer = nn.Conv1d(7, 32, 5)`."
            )
        if not isinstance(layer, nn.Module):
            return f"error: `layer` is not an nn.Module (got {type(layer).__name__})"

        shape = [int(d) for d in input_shape]
        try:
            dummy = torch.randn(*shape)
        except Exception as exc:
            return f"failed to build dummy input of shape {shape}: {exc}"

        try:
            layer.eval()
            with torch.no_grad():
                counter = FlopCounterMode(display=False)
                with counter:
                    out = layer(dummy)
                total = int(counter.get_total_flops())
        except Exception as exc:
            return f"forward pass failed: {exc}"

        out_shape = (
            tuple(out.shape) if isinstance(out, torch.Tensor) else "(non-tensor)"
        )
        param_count = sum(p.numel() for p in layer.parameters())
        lines = [
            f"measured_flops: {total:,}",
            f"output_shape: {out_shape}",
            f"params: {param_count:,}",
        ]
        if target_flops:
            pct = 100.0 * total / target_flops
            lines.append(
                f"budget_usage: {pct:.1f}% of target {target_flops:,}"
            )
        return "\n".join(lines)

    def _sketch_architecture(code: str = "", **_kwargs) -> str:
        if not code or not code.strip():
            return "error: empty code"

        if "def build_optimizer" not in code:
            code = code.rstrip() + (
                "\n\ndef build_optimizer(model):\n"
                "    import torch\n"
                "    return torch.optim.Adam(model.parameters(), lr=1e-3)\n"
            )

        sink: list = []
        estimated, err = estimate_flops(code, challenge, sink)
        if err:
            return f"sketch forward pass failed: {err}"

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
                + (
                    f" ({pct:.0f}% of target {target_flops:,})"
                    if target_flops else ""
                )
            )
            if flops_min and flops_max:
                gate_min = int(flops_min * 0.9)
                gate_max = int(flops_max * 1.1)
                if estimated < gate_min:
                    lines.append(
                        f"STATUS: below hard gate [{gate_min:,}, "
                        f"{gate_max:,}] — increase capacity"
                    )
                    hint = suggest_resize(
                        estimated, gate_min, gate_max, target_flops,
                    )
                    if hint:
                        lines.append(hint)
                elif estimated > gate_max:
                    lines.append(
                        f"STATUS: above hard gate [{gate_min:,}, "
                        f"{gate_max:,}] — reduce capacity"
                    )
                    hint = suggest_resize(
                        estimated, gate_min, gate_max, target_flops,
                    )
                    if hint:
                        lines.append(hint)
                else:
                    lines.append(
                        f"STATUS: within hard gate [{gate_min:,}, "
                        f"{gate_max:,}]"
                    )

        task = challenge.get("task", {}) or {}
        tp = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        expected = infer_output_shape(tp, constraints)
        if expected is not None and sink:
            shape_err = verify_output_shape(sink[0], expected)
            actual = tuple(sink[0])
            if shape_err:
                lines.append(
                    f"OUTPUT SHAPE: {actual} — MISMATCH: {shape_err}"
                )
            else:
                lines.append(
                    f"OUTPUT SHAPE: {actual} — matches expected"
                )
        elif sink:
            lines.append(
                f"OUTPUT SHAPE: {tuple(sink[0])} (no declared expectation)"
            )

        lines.append(
            "\nNote: sketch_architecture does NOT reject on FLOPs bounds. "
            "Use validate_code for the final pre-submission check."
        )

        trace_text = (
            format_trace(entries, max_rows=40)
            if not trace_err and entries else None
        )
        cid = upsert_candidate(
            state_holder["state"],
            code=code,
            flops=int(estimated) if estimated is not None else None,
            trace=trace_text,
        )
        lines.append(f"\ncandidate_id: {cid}")
        return "\n".join(lines)

    def _estimate_flops(code: str = "", **_kwargs) -> str:
        if not code:
            return "error: empty code"
        flops, err = estimate_flops(code, challenge)
        if err:
            return f"error: {err}"
        return f"estimated_flops: {flops:,}"

    def _size_to_flops(code_template: str = "", size_min: int = 0,
                       size_max: int = 0, target_flops: int = 0,
                       **_kwargs) -> str:
        if not code_template or not code_template.strip():
            return "error: empty code_template"
        if "{{SIZE}}" not in code_template:
            return (
                "error: code_template must contain a `{{SIZE}}` "
                "placeholder (the integer knob to sweep)"
            )
        try:
            size_min = int(size_min)
            size_max = int(size_max)
            target_flops = int(target_flops or 0)
        except (TypeError, ValueError):
            return "error: size_min/size_max/target_flops must be integers"
        if size_min <= 0:
            return f"error: size_min must be > 0 (got {size_min})"
        if size_max < size_min:
            return (
                f"error: size_max ({size_max}) < size_min ({size_min})"
            )
        if target_flops <= 0:
            target_flops = int(flops_max * 0.6) if flops_max else 0
        if target_flops <= 0:
            return (
                "error: no target_flops supplied and the challenge has "
                "no FLOPs budget to default from"
            )

        def _probe(size: int) -> int | None:
            code = code_template.replace("{{SIZE}}", str(size))
            flops, err = estimate_flops(code, challenge)
            if err or flops is None:
                return None
            return int(flops)

        result = sweep_sizes(
            _probe, size_min, size_max, target_flops,
        )
        probes = result["probes"]
        failures = result["failures"]
        best = result["best"]
        n_probes = result["n_probes"]

        lines = [
            f"target_flops: {target_flops:,}",
            f"search_range: [{size_min:,}, {size_max:,}]",
            f"probes: {n_probes} ({len(probes)} ok, {len(failures)} failed)",
        ]
        if best is None:
            lines.append(
                "best: NONE — every probe failed. Check that "
                "code_template defines build_model and the SIZE knob is "
                "wired to a real shape dimension."
            )
            if failures:
                lines.append(
                    f"failing sizes (sample): {failures[:5]}"
                )
        else:
            size, flops = best
            pct = 100.0 * flops / target_flops
            lines.append(
                f"best: size={size:,} flops={flops:,} "
                f"({pct:.1f}% of target)"
            )
            if flops_min and flops_max:
                gate_min = int(flops_min * 0.9)
                gate_max = int(flops_max * 1.1)
                if gate_min <= flops <= gate_max:
                    lines.append(
                        f"status: within hard gate "
                        f"[{gate_min:,}, {gate_max:,}]"
                    )
                else:
                    lines.append(
                        f"status: OUTSIDE hard gate "
                        f"[{gate_min:,}, {gate_max:,}] — widen the "
                        "search range or change the knob"
                    )
            if len(probes) > 1:
                sample = ", ".join(
                    f"{s}:{f:,}" for s, f in probes[:6]
                )
                lines.append(f"probe_samples: {sample}")
        return "\n".join(lines)

    def _list_frontier(**_kwargs) -> str:
        frontier = _frontier()
        if not frontier:
            return "frontier is empty (bootstrapping round)"
        items = [
            {
                "idx": i,
                "name": f.get("name", "?"),
                "objectives": f.get("objectives", {}),
            }
            for i, f in enumerate(frontier)
        ]
        return json.dumps(items, indent=2)

    def _get_frontier_member(idx: int = -1, **_kwargs) -> str:
        frontier = _frontier()
        if not frontier:
            return "error: frontier is empty"
        if not isinstance(idx, int) or idx < 0 or idx >= len(frontier):
            return f"error: idx {idx} out of range [0, {len(frontier)})"
        return json.dumps(frontier[idx], indent=2, default=str)

    # ── Validation ───────────────────────────────────────────────

    def _trace_architecture(code: str = "", max_rows: int = 60,
                            **_kwargs) -> str:
        if not code or not code.strip():
            return "error: empty code"
        entries, err = trace_architecture(code, challenge)
        if err:
            return f"trace failed: {err}"
        if not entries:
            return (
                "trace produced no entries — the model has no leaf modules "
                "with parameters, or the forward pass produced no tensor "
                "outputs through hooked modules."
            )
        return format_trace(entries, max_rows=max_rows)

    def _check_output_shape(code: str = "", **_kwargs) -> str:
        if not code or not code.strip():
            return "error: empty code"
        task = challenge.get("task", {}) or {}
        tp = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        expected = infer_output_shape(tp, constraints)
        if expected is None:
            return (
                "no parseable output-shape constraint in task.constraints; "
                "skipping shape check"
            )
        sink: list = []
        _, err = estimate_flops(code, challenge, sink)
        if err:
            return f"could not run forward pass: {err}"
        if not sink:
            return (
                "forward pass completed but no tensor output was captured. "
                "Add a concrete forward() that returns a torch.Tensor."
            )
        shape_err = verify_output_shape(sink[0], expected)
        if shape_err:
            return f"FAILED: {shape_err}"
        expected_pretty = ", ".join(
            str(e) if e >= 0 else "?" for e in expected
        )
        return (
            f"PASSED: output shape {tuple(sink[0])} matches expected "
            f"(B, {expected_pretty})."
        )

    def _validate_code(code: str = "", candidate_id: str = "",
                       **_kwargs) -> str:
        state = state_holder["state"]
        if candidate_id:
            record = find_candidate(state, candidate_id)
            if record is None:
                return (
                    f"errors: candidate_id {candidate_id!r} not found"
                )
            code = record.get("code") or ""
        if not code:
            return "errors: empty code"
        ok, errors = validate_code(code, challenge)
        if not ok:
            return "errors: " + "; ".join(errors)
        # Stash the most recent validated code so the agent can
        # auto-submit it if the LLM never calls submit explicitly.
        state_holder["last_validated_code"] = code
        cid = candidate_id or upsert_candidate(state, code=code)
        mark_candidate_validated(state, cid)
        state_holder["last_validated_candidate_id"] = cid
        return (
            "ok — code passed all checks. THIS IS YOUR FINAL "
            "ARTIFACT. Your next tool call should be "
            f"`submit(candidate_id={cid!r}, name=..., motivation=..., "
            "note=...)` — pass the scratchpad note inline so you only "
            "need one call. Do not call sketch_architecture, "
            "size_to_flops, or validate_code again unless you have a "
            "specific reason to revise.\n"
            f"candidate_id: {cid}"
        )

    # ── State ────────────────────────────────────────────────────

    READ_MY_SUBMISSIONS_TRUNC_LINES = 40

    def _read_my_submissions(n: int = 3, **_kwargs) -> str:
        try:
            n = max(1, min(int(n), 20))
        except (TypeError, ValueError):
            n = 3
        subs = get_submissions(state_holder["state"])
        if not subs:
            return "no submissions yet"
        # Newest first.
        recent = list(reversed(subs[-n:]))
        parts: list[str] = []
        for i, sub in enumerate(recent, 1):
            score = sub.get("score")
            if isinstance(score, (int, float)):
                score_text = f"{score:.4g}"
            else:
                score_text = "pending"
            rank = sub.get("rank")
            rank_total = sub.get("rank_total")
            if isinstance(rank, int):
                rank_text = (
                    f"{rank}/{rank_total}"
                    if isinstance(rank_total, int) else str(rank)
                )
            else:
                rank_text = "?"
            cid = sub.get("candidate_id") or "—"
            sub_round = sub.get("round_id") or "?"
            sub_name = sub.get("name") or "?"
            motivation = sub.get("motivation") or ""
            code = sub.get("code") or ""
            code_lines = code.splitlines()
            # Full code only when n=1; otherwise truncate per entry.
            if n > 1 and len(code_lines) > READ_MY_SUBMISSIONS_TRUNC_LINES:
                shown = "\n".join(
                    code_lines[:READ_MY_SUBMISSIONS_TRUNC_LINES]
                )
                more = len(code_lines) - READ_MY_SUBMISSIONS_TRUNC_LINES
                code_block = (
                    f"{shown}\n... ({more} more lines truncated; "
                    "call read_my_submissions(n=1) for full code)"
                )
            else:
                code_block = code
            header = (
                f"## Submission {i} of {len(recent)} "
                f"(round {sub_round}, name={sub_name})"
            )
            meta_lines = [
                f"score: {score_text} (rank {rank_text})",
                f"candidate_id: {cid}",
            ]
            if motivation:
                meta_lines.append(f"motivation: {motivation}")
            parts.append(
                f"{header}\n"
                + "\n".join(meta_lines)
                + f"\n```python\n{code_block}\n```"
            )
        return "\n\n".join(parts)

    # ── Macros ──────────────────────────────────────────────────
    #
    # Macro execution calls the same wrapped handler dict the LLM
    # uses, so circuit breakers and call counts apply uniformly to
    # macro-driven invocations. The dict isn't built yet at this
    # point in build_handlers — we stash a reference once it's
    # finalized below and read from the box at run time.

    handlers_box: dict = {}
    MACRO_FORBIDDEN_TOOLS = frozenset({
        "submit", "define_macro", "run_macro",
    })
    MACRO_NAME_RE = re.compile(r"^[A-Za-z_][\w]{0,63}$")
    MACRO_REF_RE = re.compile(r"\$\{([\w.]+)\}")

    def _resolve_ref(ref: str, run_args: dict, step_outputs: dict):
        if ref.startswith("args."):
            key = ref[len("args."):]
            if isinstance(run_args, dict) and key in run_args:
                return run_args[key], True
            return None, False
        if ref in step_outputs:
            return step_outputs[ref], True
        return None, False

    def _substitute(value, run_args: dict, step_outputs: dict):
        """Recursively substitute ``${...}`` refs in args. A whole-
        string ref preserves the source value's type; embedded refs
        stringify. Missing refs leave the literal ``${name}`` in
        place so the LLM can see what didn't resolve.
        """
        if isinstance(value, str):
            full = MACRO_REF_RE.fullmatch(value)
            if full is not None:
                resolved, ok = _resolve_ref(
                    full.group(1), run_args, step_outputs,
                )
                return resolved if ok else value

            def replace(m):
                resolved, ok = _resolve_ref(
                    m.group(1), run_args, step_outputs,
                )
                return str(resolved) if ok else m.group(0)

            return MACRO_REF_RE.sub(replace, value)
        if isinstance(value, dict):
            return {
                k: _substitute(v, run_args, step_outputs)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [_substitute(v, run_args, step_outputs) for v in value]
        return value

    def _validate_macro_sequence(
        sequence, available_tools: set,
    ) -> str | None:
        """Return None if sequence is valid, else an error string."""
        if not isinstance(sequence, list) or not sequence:
            return "error: sequence must be a non-empty list of steps"
        if len(sequence) > MAX_MACRO_STEPS:
            return (
                f"error: sequence has {len(sequence)} steps; max "
                f"is {MAX_MACRO_STEPS}"
            )
        for i, step in enumerate(sequence, 1):
            if not isinstance(step, dict):
                return f"error: step {i} is not a dict"
            tool = step.get("tool")
            if not isinstance(tool, str) or not tool:
                return f"error: step {i} missing 'tool' name"
            if tool in MACRO_FORBIDDEN_TOOLS:
                return (
                    f"error: step {i} calls {tool!r} which is not "
                    "allowed in a macro (submit ships code; "
                    "define_macro and run_macro would let macros "
                    "recurse)"
                )
            if tool not in available_tools:
                return (
                    f"error: step {i} references unknown tool "
                    f"{tool!r}"
                )
            args = step.get("args", {})
            if args is not None and not isinstance(args, dict):
                return (
                    f"error: step {i} args must be a dict (got "
                    f"{type(args).__name__})"
                )
            output_to = step.get("output_to")
            if output_to is not None and (
                not isinstance(output_to, str) or not output_to
            ):
                return (
                    f"error: step {i} output_to must be a non-empty "
                    "string when set"
                )
        return None

    def _define_macro(name: str = "", sequence: list | None = None,
                      description: str = "", **_kwargs) -> str:
        if not isinstance(name, str) or not MACRO_NAME_RE.match(name):
            return (
                "error: macro name must match [A-Za-z_]\\w{0,63} "
                "(alphanumeric / underscore, up to 64 chars, must "
                "not start with a digit)"
            )
        # The full set of tools the LLM can dispatch — derived from
        # the wrapped dict once it's finalized; before that we fall
        # back to the raw dict's keys.
        wrapped = handlers_box.get("handlers")
        available = set(
            wrapped.keys() if wrapped is not None else raw.keys()
        )
        err = _validate_macro_sequence(sequence, available)
        if err:
            return err
        # Strip unknown step keys; keep only the schema-blessed
        # subset so persisted state stays clean.
        clean_sequence = []
        for step in sequence:
            entry = {
                "tool": step["tool"],
                "args": step.get("args") or {},
            }
            if step.get("output_to"):
                entry["output_to"] = step["output_to"]
            clean_sequence.append(entry)
        add_macro(
            state_holder["state"],
            name=name,
            sequence=clean_sequence,
            description=str(description or ""),
        )
        signature = ", ".join(
            f"{i + 1}.{s['tool']}" for i, s in enumerate(clean_sequence)
        )
        return (
            f"macro {name!r} stored ({len(clean_sequence)} step(s): "
            f"{signature})"
        )

    def _run_macro(name: str = "", args: dict | None = None,
                   **_kwargs) -> str:
        if not isinstance(name, str) or not name:
            return "error: macro name is required"
        macro = find_macro(state_holder["state"], name)
        if macro is None:
            return f"error: macro {name!r} not found"
        sequence = macro.get("sequence") or []
        run_args = args if isinstance(args, dict) else {}
        step_outputs: dict = {}
        wrapped = handlers_box.get("handlers") or {}
        parts: list[str] = []

        for i, step in enumerate(sequence, 1):
            tool = step.get("tool", "")
            raw_args = step.get("args") or {}
            output_to = step.get("output_to")
            resolved = _substitute(raw_args, run_args, step_outputs)

            label = f"[step {i} {tool}]"
            handler = wrapped.get(tool)
            if handler is None:
                parts.append(f"{label}\nerror: tool not found")
                return "\n\n".join(parts)
            if not isinstance(resolved, dict):
                parts.append(
                    f"{label}\nerror: args resolved to "
                    f"{type(resolved).__name__}, expected dict"
                )
                return "\n\n".join(parts)

            try:
                result = handler(**resolved)
            except SubmitSignal:
                # Should be unreachable — submit is banned at define
                # time — but if a caller injected a macro directly,
                # don't let SubmitSignal silently ship.
                parts.append(
                    f"{label}\nerror: submit is not allowed in macros"
                )
                return "\n\n".join(parts)
            except Exception as exc:
                parts.append(f"{label}\nerror: {exc}")
                return "\n\n".join(parts)

            result_str = str(result)
            parts.append(f"{label}\n{result_str}")
            if output_to:
                step_outputs[output_to] = result_str
            if result_str.lower().startswith(("error:", "errors:")):
                return "\n\n".join(parts)

        return "\n\n".join(parts)

    def _list_macros(**_kwargs) -> str:
        macros = get_macros(state_holder["state"])
        if not macros:
            return "no macros defined yet"
        items = []
        for name, m in macros.items():
            seq = m.get("sequence") or []
            items.append({
                "name": name,
                "description": m.get("description") or "",
                "n_steps": len(seq),
                "tools": [s.get("tool") for s in seq],
            })
        return json.dumps(items, indent=2, default=str)

    def _read_scratchpad(**_kwargs) -> str:
        state = state_holder["state"]
        history_entries = get_history(state)
        legacy_notes = state.get("agent_notes", "")
        score_direction = (
            challenge.get("score_direction") or "minimize"
        )
        parts = []
        # One-line situational summary at the top so the agent gets a
        # quick read on what's been tried before scrolling the details.
        parts.append(format_scratchpad_summary(state, score_direction))
        structured = format_notes(state)
        if structured:
            parts.append(structured)
        if legacy_notes:
            parts.append(f"## Free-form Notes (deprecated)\n{legacy_notes}")
        if history_entries:
            parts.append(
                "## Submission History\n"
                + format_history(
                    history_entries,
                    max_entries=10,
                    score_direction=score_direction,
                )
            )
        # The summary line alone isn't enough to imply "scratchpad is
        # populated" — if it's the only part we have, treat the round as
        # the first one.
        if len(parts) <= 1:
            return "scratchpad is empty — this is your first round"
        return "\n\n".join(parts)

    def _write_scratchpad(hypothesis: str = "", dead_end: str = "",
                          reason: str = "", observation: str = "",
                          candidate_id: str = "", notes: str = "",
                          **_kwargs) -> str:
        state = state_holder["state"]
        wrote: list[str] = []
        if hypothesis and hypothesis.strip():
            add_hypothesis(
                state, text=hypothesis,
                candidate_id=candidate_id or None,
            )
            wrote.append("hypothesis")
        if dead_end and dead_end.strip():
            combined = (
                f"{dead_end.strip()} — {reason.strip()}"
                if reason and reason.strip()
                else dead_end.strip()
            )
            add_note(state, "dead_ends", combined)
            wrote.append("dead_end")
        if observation and observation.strip():
            add_note(state, "task_observations", observation)
            wrote.append("observation")
        if notes and notes.strip():
            # Deprecated path — still honored so in-flight rounds don't
            # crash. Stash alongside the structured notes and warn.
            state["agent_notes"] = notes.strip()
            wrote.append("notes(deprecated)")
            _log(
                "[tools] write_scratchpad(notes=...) is deprecated — use "
                "hypothesis/dead_end/observation instead"
            )
        if not wrote:
            return (
                "error: nothing to write — pass at least one of "
                "hypothesis, dead_end (+reason), observation, or notes"
            )
        state_holder["state"] = state
        state_holder["wrote_this_round"] = True
        return f"scratchpad updated ({', '.join(wrote)})"

    def _link_hypothesis(hypothesis: str = "", candidate_id: str = "",
                         verdict: str = "", **_kwargs) -> str:
        if not hypothesis or not hypothesis.strip():
            return "error: hypothesis text is required"
        if verdict and verdict not in HYPOTHESIS_VERDICTS:
            return (
                "error: verdict must be one of "
                f"{', '.join(HYPOTHESIS_VERDICTS)} (got {verdict!r})"
            )
        if not candidate_id and not verdict:
            return "error: pass candidate_id, verdict, or both"
        record = link_hypothesis(
            state_holder["state"],
            text=hypothesis,
            candidate_id=candidate_id or None,
            verdict=verdict or None,
        )
        if record is None:
            return (
                f"error: hypothesis {hypothesis.strip()!r} not found "
                "— call read_scratchpad to see what's stored"
            )
        parts = [f"text={record['text']!r}"]
        cids = record.get("candidate_ids") or []
        if cids:
            parts.append(f"candidate_ids={cids}")
        outcome = record.get("outcome") or {}
        if outcome.get("verdict"):
            parts.append(f"verdict={outcome['verdict']}")
        return "linked — " + ", ".join(parts)

    # ── Files (scratchpad directory) ─────────────────────────────

    def _safe_file_path(name: str) -> tuple[str | None, str | None]:
        """Resolve ``name`` to an absolute path inside ``scratch_dir``.

        Returns ``(path, None)`` on success or ``(None, error)`` when the
        name is unsafe or the scratch directory is unavailable. Only
        simple basenames are allowed — no slashes, no parent refs, no
        absolute paths. Reserved names are rejected.
        """
        if scratch_dir is None:
            return None, "files unavailable (no scratchpad directory)"
        if not name or not isinstance(name, str):
            return None, "error: name is required"
        if name != os.path.basename(name) or name in (".", ".."):
            return None, (
                f"error: {name!r} is not a plain filename "
                "(no path separators, no '.' or '..')"
            )
        if name in RESERVED_FILE_NAMES:
            return None, f"error: '{name}' is reserved"
        return os.path.join(scratch_dir, name), None

    def _list_files(**_kwargs) -> str:
        if scratch_dir is None:
            return "files unavailable (no scratchpad directory)"
        try:
            entries = sorted(os.listdir(scratch_dir))
        except FileNotFoundError:
            return "no files yet"
        except OSError as exc:
            return f"error: could not list files: {exc}"
        items = []
        for entry in entries:
            if entry in RESERVED_FILE_NAMES:
                continue
            full = os.path.join(scratch_dir, entry)
            if not os.path.isfile(full):
                continue
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            items.append({"name": entry, "size": size})
        if not items:
            return "no files yet"
        return json.dumps(items, indent=2)

    def _read_file(name: str = "", **_kwargs) -> str:
        path, err = _safe_file_path(name)
        if err:
            return err
        if not os.path.isfile(path):
            return f"error: '{name}' not found"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read(FILE_MAX_BYTES + 1)
        except OSError as exc:
            return f"error: could not read '{name}': {exc}"
        except UnicodeDecodeError:
            return f"error: '{name}' is not valid UTF-8 text"
        if len(data) > FILE_MAX_BYTES:
            return (
                data[:FILE_MAX_BYTES]
                + f"\n... (truncated at {FILE_MAX_BYTES} bytes)"
            )
        return data

    def _search_files(query: str = "", **_kwargs) -> str:
        if scratch_dir is None:
            return "files unavailable (no scratchpad directory)"
        if not query or not isinstance(query, str):
            return "error: query is required"
        needle = query.lower()
        try:
            entries = sorted(os.listdir(scratch_dir))
        except FileNotFoundError:
            return "no files yet"
        except OSError as exc:
            return f"error: could not list files: {exc}"

        lines_out: list[str] = []
        total = 0
        for entry in entries:
            if total >= FILE_SEARCH_MAX_TOTAL_MATCHES:
                lines_out.append("... (match cap reached)")
                break
            if entry in RESERVED_FILE_NAMES:
                continue
            full = os.path.join(scratch_dir, entry)
            if not os.path.isfile(full):
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(FILE_MAX_BYTES)
            except OSError:
                continue
            hits: list[str] = []
            for i, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    hits.append(f"  {i}: {line.strip()[:200]}")
                    if len(hits) >= FILE_SEARCH_MAX_MATCHES_PER_FILE:
                        hits.append("  ... (more matches in this file)")
                        break
            if hits:
                lines_out.append(f"[{entry}]")
                lines_out.extend(hits)
                total += len(hits)
        if not lines_out:
            return f"no matches for {query!r}"
        return "\n".join(lines_out)

    def _write_file(name: str = "", content: str = "", **_kwargs) -> str:
        path, err = _safe_file_path(name)
        if err:
            return err
        if not isinstance(content, str):
            return "error: content must be a string"
        encoded = content.encode("utf-8")
        if len(encoded) > FILE_MAX_BYTES:
            return (
                f"error: content is {len(encoded)} bytes, max is "
                f"{FILE_MAX_BYTES}"
            )
        # Enforce max file count only when creating a new file.
        if not os.path.exists(path):
            try:
                existing = [
                    e for e in os.listdir(scratch_dir)
                    if e not in RESERVED_FILE_NAMES
                    and os.path.isfile(os.path.join(scratch_dir, e))
                ]
            except OSError:
                existing = []
            if len(existing) >= FILE_MAX_COUNT:
                return (
                    f"error: file limit reached ({FILE_MAX_COUNT}); "
                    "delete or overwrite an existing file"
                )
        try:
            os.makedirs(scratch_dir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(encoded)
        except OSError as exc:
            return f"error: could not write '{name}': {exc}"
        return f"wrote {name} ({len(encoded)} bytes)"

    # ── Control ──────────────────────────────────────────────────

    def _submit(code: str = "", name: str = "", motivation: str = "",
                candidate_id: str = "", note: str = "",
                **_kwargs) -> str:
        # Inline scratchpad note. Recorded BEFORE raising SubmitSignal
        # so the persisted state captures the note even though the
        # submit terminates the loop. When the submission is tied to a
        # candidate_id the note is filed as a hypothesis linked to the
        # candidate (so next round's read_scratchpad shows what
        # actually shipped). Otherwise it's a generic observation.
        if note and isinstance(note, str) and note.strip():
            text = note.strip()
            state = state_holder["state"]
            if candidate_id:
                add_hypothesis(state, text=text, candidate_id=candidate_id)
            else:
                add_note(state, "task_observations", text)
            state_holder["state"] = state
            state_holder["wrote_this_round"] = True
        # Soft incentive only: if the LLM hasn't written any scratchpad
        # note this round, log a warning and let the submit go through.
        # The harness no longer blocks shipping — notes-for-future-rounds
        # is the LLM's responsibility, not ours.
        if not state_holder.get("wrote_this_round"):
            print(
                "[agent] submit without scratchpad note this round",
                file=sys.stderr,
                flush=True,
            )
        if candidate_id:
            record = find_candidate(state_holder["state"], candidate_id)
            if record is None:
                return f"error: candidate_id {candidate_id!r} not found"
            code = record.get("code") or code
            mark_candidate_submitted(state_holder["state"], candidate_id)
        if not code:
            return (
                "error: nothing to submit — pass either ``code`` or a "
                "stored ``candidate_id``"
            )
        # Time-gate: only ship when we're inside the late-round window.
        # Before that, stash the candidate as best-so-far and tell the
        # LLM to keep iterating. The state_holder persists across tool
        # calls, so the stashed candidate is recoverable if the LLM
        # never beats it.
        remaining = (deadline - time.monotonic()) if deadline is not None else 0.0

        state = state_holder["state"]
        state["best_so_far"] = {
            "code": code,
            "name": name,
            "motivation": motivation,
            "candidate_id": candidate_id or None,
            "round_id": round_id,
        }
        state_holder["state"] = state

        if remaining > EARLY_SUBMIT_GATE_SECONDS:
            mins_left = int(remaining // 60)
            return (
                f"stashed as best-so-far (~{mins_left}m left). Keep iterating: "
                "try a structurally different architecture, tune sizing, or pick "
                "a different idea. Your stashed candidate ships automatically if "
                "you don't beat it. Call `submit` again inside the last 5 minutes "
                "to lock in your final choice."
            )

        # Inside the late-round window — record the submission (so
        # read_my_submissions / merge_results_into_state see what
        # actually shipped) and ship.
        add_submission(
            state_holder["state"],
            code=code,
            name=name,
            motivation=motivation,
            candidate_id=candidate_id or None,
            round_id=round_id,
        )
        raise SubmitSignal(code=code, name=name, motivation=motivation)

    def _time_remaining(**_kwargs) -> str:
        if deadline is None:
            return "no deadline set"
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 60:
            return (
                f"{remaining:.0f}s remaining "
                f"({remaining / 60:.1f} minutes)"
            )
        if remaining > 0:
            return f"{remaining:.0f}s remaining — wrap up soon"
        return "time is up — submit now or the round ends without a submission"

    raw = {
        "search_papers": _search_papers,
        "query_db": _query_db,
        "cognition_wiki_index": _cognition_wiki_index,
        "cognition_wiki_read": _cognition_wiki_read,
        "analyze_task": _analyze_task,
        "estimate_layer_flops": _estimate_layer_flops,
        "sketch_architecture": _sketch_architecture,
        "estimate_flops": _estimate_flops,
        "size_to_flops": _size_to_flops,
        "list_frontier": _list_frontier,
        "get_frontier_member": _get_frontier_member,
        "trace_architecture": _trace_architecture,
        "check_output_shape": _check_output_shape,
        "validate_code": _validate_code,
        "define_macro": _define_macro,
        "run_macro": _run_macro,
        "list_macros": _list_macros,
        "read_scratchpad": _read_scratchpad,
        "read_my_submissions": _read_my_submissions,
        "write_scratchpad": _write_scratchpad,
        "link_hypothesis": _link_hypothesis,
        "list_files": _list_files,
        "read_file": _read_file,
        "write_file": _write_file,
        "search_files": _search_files,
        "submit": _submit,
        "time_remaining": _time_remaining,
    }

    def _wrap(name, fn):
        def wrapped(**kwargs):
            _call_counts[name] = _call_counts.get(name, 0) + 1
            # SubmitSignal must propagate untouched — do not pass it
            # through the breaker.
            try:
                result = fn(**kwargs)
            except SubmitSignal:
                raise
            except Exception as exc:
                result = f"error: {exc}"
            return breaker(name, str(result))

        return wrapped

    class _SubmitWrapper:
        """Callable wrapper around ``_submit`` that exposes per-round state.

        The agent reads ``_state_holder`` / ``_call_counts`` after the
        loop to persist scratchpad notes and log a tool-usage summary.
        ``_has_validated`` / ``_last_validated_code`` let the agent
        recover the most recent validated code if the LLM never calls
        submit explicitly (the "analysis paralysis" failure mode).
        """

        def __init__(self):
            self._state_holder = state_holder
            self._call_counts = _call_counts

        def __call__(self, **kwargs):
            _call_counts["submit"] = _call_counts.get("submit", 0) + 1
            try:
                result = _submit(**kwargs)
            except SubmitSignal:
                raise
            except Exception as exc:
                result = f"error: {exc}"
            return breaker("submit", str(result))

        @property
        def _has_validated(self) -> bool:
            return bool(state_holder.get("last_validated_code"))

        @property
        def _last_validated_code(self) -> str:
            return state_holder.get("last_validated_code") or ""

    handlers = {name: _wrap(name, fn) for name, fn in raw.items()}
    handlers["submit"] = _SubmitWrapper()
    # Make the wrapped dict visible to _run_macro so macro-driven
    # invocations go through the same circuit breaker / counter path
    # as direct LLM calls.
    handlers_box["handlers"] = handlers
    return handlers


# Role → tool subset. Schemas stay byte-identical with the openai_sdk
# version; ``role`` only filters which tools are exposed to a given
# subagent. ``time_remaining`` is included in both subsets so each
# subagent can budget its own deadline.
ROLE_TOOLS: dict[str, frozenset[str]] = {
    "researcher": frozenset({
        "search_papers", "query_db", "list_frontier", "analyze_task",
        "cognition_wiki_index", "cognition_wiki_read",
        "time_remaining",
    }),
    "designer": frozenset({
        "sketch_architecture", "estimate_layer_flops", "validate_code",
        "submit", "time_remaining",
    }),
}


def build_tools(
    challenge: dict | None = None,
    role: str | None = None,
) -> list[dict]:
    """Return a tools list customized for this challenge.

    Swaps the placeholder in the ``search_papers`` description for the
    task's actual name so the LLM sees a relevant example query.

    When ``role`` is one of ``ROLE_TOOLS`` keys, only the tools in that
    subset are returned — the schemas themselves are not modified, so a
    handler dispatched by name still resolves identically.
    """
    if not challenge:
        tools = list(TOOLS)
    else:
        task = challenge.get("task", {}) or {}
        task_name = task.get("name", "") or "neural network"
        tools = copy.deepcopy(TOOLS)
        for t in tools:
            fn = t.get("function", {})
            if fn.get("name") == "search_papers":
                props = fn.get("parameters", {}).get("properties", {})
                q = props.get("query", {})
                q["description"] = (
                    f"Search query (e.g. 'efficient {task_name} "
                    "neural network')"
                )
    if role is None:
        return tools
    allowed = ROLE_TOOLS.get(role)
    if allowed is None:
        raise ValueError(
            f"unknown role {role!r}; expected one of {sorted(ROLE_TOOLS)}"
        )
    return [
        t for t in tools
        if t.get("function", {}).get("name") in allowed
    ]
