"""Claude-Code-style multi-subagent miner agent — v3.

Fork of ``miners/claude_style_v2_gepa``. Adds an **analyst** subagent
that runs before the researcher, a **two-phase researcher** (high-
temperature brainstorm → tool-driven brief), and **primitive
injection** — a per-round sample from ``core.primitives.PRIMITIVE_POOL``
threaded through analyst and researcher as required ingredients.

Pipeline: ``analyst → (phase A brainstorm) → researcher → designer ↔ critic``.

Prompt-population surface has four slots in v3:
``metadata.slot`` ∈ {``analyst``, ``researcher``, ``designer``,
``critic``}. Each subagent's GEPA-evolved directive replaces the
``## Principles`` body section of its hardcoded system prompt.
Submission ``prompt_id`` is the composite
``a:<aid>|r:<rid>|d:<did>|c:<cid>``. See ``README.md`` for full
wiring.

A Claude-Code-inspired harness on top of the OpenAI-compatible LLM
transport: an orchestrator coordinates four specialist subagents
(analyst, researcher, designer, critic) with their own message lists
and tool subsets. The Claude Agent SDK / `claude` CLI / `anthropic`
package are NOT used — the sandbox doesn't have them. We borrow
only the patterns (subagent split, structured plan, pre-tool-call
hooks, context isolation).

This package is structured so it can be deployed two ways:

1. Imported as a package (``from agents.claude_style import agent``) — the
   ``__init__.py`` below injects this directory into ``sys.path`` so the
   sibling modules' ``from core.X import Y`` absolute imports resolve to
   ``agents/claude_style/core/``.

2. Copied flat to ``/workspace/agent/`` and loaded standalone by the
   harness via ``importlib.util.spec_from_file_location`` — in that mode
   this ``__init__.py`` does NOT run, but because ``core/`` sits right
   next to ``agent.py`` and the harness puts the agent's directory on
   ``sys.path``, ``from core.X import Y`` still resolves correctly. This
   matches how the autonomous agent is deployed.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
