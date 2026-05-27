# OpenAI SDK reference agent

A reference implementation of a Radar miner that uses the OpenAI Python
SDK (`openai>=1.50`, installed in the official agent image) instead of
the hand-rolled urllib that the `autonomous/` agent uses. Network egress
is enforced at the network layer (iptables) inside the agent pod, so
bypassing `GatedClient` is safe.

## When to use this

Pick this agent as your starting point if you want:

- Automatic retry on transient errors (429, 5xx, timeout) classified by
  the SDK rather than substring-matched on raw exception text
- Clean tool-calling via `ChatCompletionMessage.tool_calls` rather than
  parsing dicts
- Easy swap to another OpenAI-compatible endpoint or model
- Less code to maintain than `autonomous/`

## When to use `autonomous/` instead

- You're on an older agent image without the SDK
- You want full control over the HTTP transport
- You're implementing a novel retry / routing strategy
- You want the strategy framework (personas / per-strategy temperatures)

## Architecture

Single tool-calling loop, one deadline. The LLM owns its time and
decides when to ship.

`design_architecture` resolves a budget (`challenge["agent_seconds"]`,
`AGENT_BUDGET_SECONDS`, or `task.time_budget`), reserves the last 30s
for packaging, and hands the rest to `_run_tool_loop`. The loop runs
until one of these fires:

- **Submit** — the LLM calls the `submit` tool with validated code.
  This is the happy path.
- **Validated-and-stalled** — the LLM ran `validate_code` successfully
  but burned two more rounds without calling `submit`. The harness
  bails so its auto-submit recovery can ship the validated code.
- **No tool calls** — the LLM responded with prose instead of a tool
  call. If the prose contains a fenced code block it gets extracted as
  a candidate; either way the loop ends.
- **Deadline approaching** — fewer than 60s remain on the budget. The
  loop hands control back to the packaging stage.
- **Chat error** — the OpenAI SDK raises (timeout, 4xx, etc.). Config
  errors short-circuit; transient errors are retried inside `chat()`.

Once the loop exits, the agent saves the scratchpad, then picks the
best return value it has — in order: explicit `submit`, validated
candidate, partially-validated candidate, auto-submit of validated
code stashed on the submit handler, and finally an honest
empty-code failure package.

There are no research/design phases, no per-phase round caps, no
turn-header escalation, and no submit nag. The system prompt tells the
LLM about the budget and the principles ("validation is the commitment
point", "you can stop early"); the harness only intervenes when the
LLM is genuinely stuck.

## File layout

```
agents/openai_sdk/
├── __init__.py        # puts this dir on sys.path for package-mode imports
├── agent.py           # entry point — design_architecture
├── llm_client.py      # cached OpenAI client + retry/failover
├── prompts.py         # system/user prompt builders
├── tools.py           # tool schemas + handlers (OpenAI function-call format)
├── validation.py      # re-export of core.validation.validate_code
├── core/              # FLOPs / validation / history / fallback helpers
└── README.md          # this file
```

`core/` lives inside this package (copied from
`agents/autonomous/core/`) so the harness can ship the directory
flat to `/workspace/agent/` and `from core.X import Y` resolves
without any package-import plumbing — exactly like the autonomous
agent is deployed.

## Reused from `agents/autonomous/core/`

- `validation.validate_code` — AST + FLOPs + output-shape checks
- `flops_estimator.estimate_flops` — FlopCounterMode wrapper
- `history.extract_flops_budget`, `history.identify_bucket` — bucket math
- `prompt_builder._compute_sizing_guidance` — sizing prompt fragments

## Environment variables

| Name | Default | Meaning |
|------|---------|---------|
| `LLM_URL` | (required) | OpenAI-compatible base URL (the proxy) |
| `AGENT_TOKEN` | `""` | Forwarded as `X-Agent-Token` header |
| `MINER_UID` | `"0"` | Forwarded as `X-Miner-UID` header |
| `LLM_READ_TIMEOUT` | `180` | Per-request read timeout in seconds |
| `AGENT_BUDGET_SECONDS` | (none) | Override agent-side time budget |

## Selecting this agent

Miners pick their entry point via `AGENT_MODULE`:

```bash
AGENT_MODULE=/workspace/agent/agents/openai_sdk/agent.py
```

The default remains `agents/autonomous/agent.py` until this version is
clearly at parity in production.

## Running tests locally

```bash
python -m pytest tests/test_openai_sdk_agent.py -v
```

The tests mock the OpenAI client at module level — no network calls are
made.
