"""Claude-Code-style multi-subagent miner agent.

A Claude-Code-inspired harness on top of the OpenAI-compatible LLM
transport: an orchestrator coordinates three specialist subagents
(researcher, designer, critic) with their own message lists and tool
subsets. The Claude Agent SDK / `claude` CLI / `anthropic` package are
NOT used — the sandbox doesn't have them. We borrow only the patterns
(subagent split, structured plan, pre-tool-call hooks, context isolation).

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
