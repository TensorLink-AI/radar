"""Claude-Code-style multi-subagent miner agent — v4.

Fork of ``miners/claude_style_v3``. Adds **training-recipe-aware
two-pass design** so the agent never has to balance architecture and
training recipe in a single decision:

1. **Analyst now reports `training_recipe_norm`** — what optimizer /
   LR / schedule / loss / AMP / batch the frontier members have in
   common. The analyst gains ``get_frontier_member`` to read 1-2
   members' recipe hooks. The new field rides through to the recipe-
   tuner pass without leaking into the researcher's brief (which
   stays architecture-only).

2. **In-round recipe-tuner pass** runs after the designer's
   architecture pass on fresh rounds. It reuses the continuation
   machinery: the just-validated architecture is extracted via
   ``continuation.extract_arch_source`` and pinned into a second
   designer call as a frozen block under ``_inround_recipe_context``.
   The designer sees a single-focus recipe-only preamble — the same
   framing that already works well for continuation rounds.
   Continuation rounds skip this pass (the continuation flow already
   IS the recipe pass).

3. **Fallback templates ship a strong default recipe** — AdamW with
   weight decay + linear-warmup-cosine scheduler + bfloat16 AMP +
   grad clip + sensible ``training_config``. Raises the floor for
   any submission that doesn't override the recipe, and gives the
   LLM a stronger prior to start from.

Pipeline (fresh round):
  ``analyst → (phase A) → researcher → designer (arch) → recipe-tuner``

Pipeline (continuation round):
  ``designer (recipe-only)`` — unchanged from v3.

Inherits everything else from v3 verbatim (analyst axes-of-variation
digest, two-phase researcher, primitive injection, single-slot
``_operator_prompt`` on the designer). See ``README.md`` for the full
wiring.

A Claude-Code-inspired harness on top of the OpenAI-compatible LLM
transport: an orchestrator coordinates four specialist subagents
(analyst, researcher, designer, critic) with their own message lists
and tool subsets. The Claude Agent SDK / `claude` CLI / `anthropic`
package are NOT used — the sandbox doesn't have them. We borrow only
the patterns (subagent split, structured plan, pre-tool-call hooks,
context isolation).

This package is structured so it can be deployed two ways:

1. Imported as a package (``from agents.claude_style_v4 import agent``)
   — the ``__init__.py`` below injects this directory into ``sys.path``
   so the sibling modules' ``from core.X import Y`` absolute imports
   resolve to ``agents/claude_style_v4/core/``.

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
