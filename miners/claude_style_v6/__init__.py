"""Claude-Code-style multi-subagent miner agent — v6.

Fork of ``miners/claude_style_v5``. v6 keeps the entire v5 stack
verbatim for the architecture tasks (analyst → researcher → designer →
recipe-tuner, the static recipe-sanity inspector, the divergence target,
the DIVERGE critic line) and concentrates its additions on the
**data-generator tasks** (``ts_data_pipeline`` and
``synthetic_data_generator``), where v5 left two problems on the table:
valid submissions could be silently rejected, and the designer got no
signal about whether its generator was any *good* — only whether the
shapes were right. v6's three deltas:

1. **synthetic_data_generator parity (robustness).** v5 routed both
   pipeline tasks to the pipeline-designer flow, but ``validate_code``
   and the fallback generator only recognised the literal
   ``ts_data_pipeline``. On a ``synthetic_data_generator`` round a valid
   ``build_pipeline`` submission failed the agent's own validation (it
   demanded ``build_model``/``build_optimizer``) and the fallback shipped
   a model with the wrong contract — a guaranteed failed round. v6
   centralises the pipeline-task set in ``core`` (``PIPELINE_TASKS`` /
   ``is_pipeline_task``) so routing, validation, fallback, and tool
   gating all agree.

2. **Quality feedback, not just shapes (better ideas).**
   ``core.pipeline_probe`` now pulls a few extra batches and computes
   cheap descriptive statistics — per-channel variance, constant-channel
   fraction, lag-1 autocorrelation, value range, cross-batch variety,
   finite fraction — and flags degenerate generators (all-constant,
   near-zero variance, identical batches across draws, clipped range).
   ``pipeline_smoke_test`` surfaces these so the designer can iterate
   toward a richer distribution in-round instead of discovering a
   collapsed generator only after it scores at the GIFT gate.

3. **A sharper data-design prompt (better ideas).** The
   pipeline-designer system prompt's design space is rewritten from a
   four-bullet menu into concrete, GIFT-Eval-aware generation recipes
   (trend + multi-scale seasonality, regime/change-point switching,
   heavy-tail + heteroscedastic noise, varied scales/normalisation) with
   an explicit *diversity-first* mandate and instructions to read the
   new quality stats and the frontier before committing.

Everything else — the orchestration, the architecture-task subagents,
the recipe-tuner, primitive injection, the ``_operator_prompt`` slot —
is inherited from v5 unchanged. See ``README.md`` for the v5 → v6 deltas.

A Claude-Code-inspired harness on top of the OpenAI-compatible LLM
transport: an orchestrator coordinates four specialist subagents
(analyst, researcher, designer, critic) with their own message lists
and tool subsets. The Claude Agent SDK / `claude` CLI / `anthropic`
package are NOT used — the sandbox doesn't have them. We borrow only
the patterns (subagent split, structured plan, pre-tool-call hooks,
context isolation).

This package is structured so it can be deployed two ways:

1. Imported as a package (``from agents.claude_style_v6 import agent``)
   — the ``__init__.py`` below injects this directory into ``sys.path``
   so the sibling modules' ``from core.X import Y`` absolute imports
   resolve to ``agents/claude_style_v6/core/``.

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
