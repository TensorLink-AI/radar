# Claude-style multi-subagent miner agent

A Claude-Code-inspired harness on top of the OpenAI-compatible LLM
transport: the orchestrator runs three specialist subagents
(researcher, designer, critic) in sequence, each with its own message
list and tool subset.

**This agent does not use the Claude Agent SDK, the `claude` CLI, or
the `anthropic` package.** None of those are available in the sandbox.
The patterns are borrowed (subagent split, structured plan, pre-tool-call
hooks, context isolation); the implementation is pure Python on top of
the same `llm_client.chat()` the openai_sdk agent uses.

## Architecture

- **Researcher** (≤ 20% of budget, cap 90s): tools = `search_papers`,
  `query_db`, `list_frontier`, `analyze_task`. Outputs a JSON brief
  (`relevant_prior_work`, `frontier_gaps`, `ideas_to_try`, `plan`).
- **Designer** (≤ 60% of budget): tools = `sketch_architecture`,
  `estimate_layer_flops`, `validate_code`, `submit`. Loops on the
  brief: generate → validate → revise → submit.
- **Critic**: single text call between designer iterations (no tools),
  emits a `keep / change / drop` critique injected into the next
  designer turn.
- **Hooks**: `before_tool_call(name, args, state) -> Optional[str]`.
  v1 ships one rule, `submit_requires_recent_validate`, that blocks
  `submit` until a recent `validate_code` returned `ok=true`.
- **Fallback**: when the designer fails to ship, the orchestrator
  falls through to `core/fallback_templates.generate_fallback`.

## File layout

```
agents/claude_style/
├── __init__.py             # puts this dir on sys.path
├── agent.py                # orchestrator + design_architecture
├── subagents/
│   ├── __init__.py
│   ├── base.py             # Subagent dataclass + run loop
│   ├── researcher.py
│   ├── designer.py
│   └── critic.py
├── hooks.py                # before_tool_call framework + submit rule
├── prompts.py              # per-subagent system prompts
├── llm_client.py           # copied unchanged from openai_sdk/
├── tools.py                # copied + role parameter on build_tools
├── validation.py           # re-export of core.validation
└── core/                   # copied unchanged
```

## Status

Skeleton — compiles end-to-end. The subagent bodies are stubs pending
the stop-and-show checkpoints in CLAUDE.md.
