"""Claude-Code-style multi-subagent miner agent — v5.

Fork of ``miners/claude_style_v4``. v5 keeps v4's analyst → researcher
→ designer → recipe-tuner pipeline verbatim and adds three focused
mechanisms aimed at *fewer failures, more novel designs, more SOTA
candidates*:

1. **Static recipe-sanity inspector** (``core.recipe_sanity``). A
   pure-AST inspector that flags known-bad recipe patterns the runtime
   ``validate_code`` smoke test can't catch — lr outside [1e-5, 5e-2],
   AdamW with weight_decay=0, missing grad clip on Adam-family
   optimizers, COMPILE=True without bf16 AMP, oversized batches on
   long-context tasks. Findings are appended to the critic's message
   so the designer sees them inline on the next turn (no extra LLM
   call). The runtime validator catches code that crashes; the
   inspector catches code that trains fine but produces weak models.

2. **Divergence target in the brief** (``BRIEF_SCHEMA_EXAMPLE`` gains
   a ``divergence_axes`` field). The researcher now MUST name at
   least one axis from the analyst's ``axes_fixed_across_frontier``
   where the lead idea explicitly breaks with the frontier consensus.
   The designer system prompt cites the brief's divergence axes as a
   hard requirement. A candidate that ties the frontier on every
   fixed axis loses the Pareto bonus — making the divergence explicit
   at brief-time pushes the designer past local LR/depth jitter and
   toward structural variation.

3. **DIVERGE line in the critic** (four-line KEEP / CHANGE / DROP /
   DIVERGE template). The critic now grades the candidate on
   architectural + recipe divergence on every iteration, not only on
   validation outcome. If the candidate matches the frontier on every
   axis, the critic is required to say so and the CHANGE line must
   propose a structural pivot rather than a hyperparam tweak.

Plus a smaller continuation-readiness nudge in the designer prompt:
``build_model`` / ``init_weights`` / ``COMPILE`` must stay cleanly
separable from the recipe hooks so a future continuation round can
lift the architecture verbatim via
``continuation.extract_arch_source``.

Pipeline is unchanged from v4:
  ``analyst → (phase A) → researcher → designer (arch) → recipe-tuner``

Inherits everything else from v4 verbatim (``training_recipe_norm``
field on the analyst, in-round recipe-tuner second pass, strong-default
fallback templates, primitive injection, ``_operator_prompt`` slot).
See ``README.md`` for the full wiring and the v4 → v5 deltas.

A Claude-Code-inspired harness on top of the OpenAI-compatible LLM
transport: an orchestrator coordinates four specialist subagents
(analyst, researcher, designer, critic) with their own message lists
and tool subsets. The Claude Agent SDK / `claude` CLI / `anthropic`
package are NOT used — the sandbox doesn't have them. We borrow only
the patterns (subagent split, structured plan, pre-tool-call hooks,
context isolation).

This package is structured so it can be deployed two ways:

1. Imported as a package (``from agents.claude_style_v5 import agent``)
   — the ``__init__.py`` below injects this directory into ``sys.path``
   so the sibling modules' ``from core.X import Y`` absolute imports
   resolve to ``agents/claude_style_v5/core/``.

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
