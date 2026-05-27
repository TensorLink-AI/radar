# local/ — single-laptop radar stack

A minimal validator + miner pair that runs on one machine with SQLite
as the broker. No HTTP, no Docker, no HMAC, no chain, no Hippius/R2.
Useful for understanding the phase A → B → C flow without spinning up
Postgres and a fleet of trainer pods.

## What it preserves

| Real radar | Local stack |
|---|---|
| Phase A: agent designs architecture | `local/agent.py::design_architecture` |
| Phase B: train on miner pod | `local/trainer.py::run_training` (in-process) |
| Phase C: validator evals checkpoint | Same function — held-out test split |
| Postgres single source of truth | `local/store.py` (SQLite, WAL) |
| HMAC + bearer tokens | None — trust the machine |
| Size buckets (FLOPs-equivalent) | `local/task.py::SIZE_BUCKETS` |
| Scoring: size gate + frontier + Pareto bonus | `local/scoring.py` |
| Validator ↔ DB HTTP API | `local/services.py` on 127.0.0.1 |
| Agent's `GatedClient` for db/llm/desearch/wiki | Real `shared.url_gate.GatedClient` pointed at the local services URL |
| Validator ↔ miner via challenge proxy | Validator writes `challenges` row; miner polls |
| Miner ↔ trainer pod via Targon/Basilica | Validator runs trainer in-process |

## What it drops

- Docker / image hardening / boot proof
- Targon, RunPod, Basilica hosting
- Cloudflare R2 / Hippius artifact storage
- Operator CLI, miner bearer tokens
- The full `runner/timeseries_forecast` (GIFT-Eval + safetensors checkpoints)
- The FastAPI dashboard
- Postgres FTS / provenance audit / access logs

## Task

A synthetic 8-dim regression with a small MLP ground truth + light
Gaussian noise. The miner emits a numpy MLP submission (no torch
required); the validator trains it under fixed gradient descent and
reports MSE on a held-out test split. Lower is better.

Size buckets are param-count × 2 (close enough to FLOPs-per-forward
for an MLP). Default buckets are tiny / small / medium / large; the
validator rotates through them by `round_id`.

## Quickstart

```bash
# Only dependency: numpy. No torch, no Postgres, no Docker.
pip install numpy

# Run validator + 1 miner together, 3 rounds, then exit.
python local/run.py --rounds 3

# Or run them in two terminals (real role separation):
python -m local.validator --rounds 3      # terminal 1
python -m local.miner --rounds 3          # terminal 2

# Multiple miners against one validator
python local/run.py --miners 3 --rounds 5
```

Both processes write into `local/radar_local.db`. Inspect it with
`sqlite3 local/radar_local.db` — schema in `local/store.py`.

## Plugging in your own agent

Same interface as `miner/neuron.py --agent_dir` in real radar. Point
either run.py or the miner directly at a directory containing
`agent.py`:

```bash
python local/run.py --agent_dir path/to/your/agent_dir/
# or
python -m local.miner --agent_dir path/to/your/agent_dir/
```

Sibling files in that dir are importable from `agent.py`.
`--agent_module path/to/single.py` works for one-file agents.

The agent's signature can be either form — the miner auto-detects:

```python
def design_architecture(challenge, client):   # like the distributed harness
    ...

def design_architecture(challenge):           # if you don't need the client
    ...
```

Returns `{code, name, motivation, reasoning, tool_calls, prompt_id}`.
The `code` field is a Python source string that must define
`build_model(input_dim, output_dim)` returning an object with
`hidden_sizes`, `activation`, `learning_rate`, `epochs` attributes —
see `local/agent.py` for the reference emitter.

## Agent-facing services

When the validator starts it stands up a localhost HTTP server with
the same endpoint shape miner agents see in real radar. The challenge
carries the URLs and `allowed_urls`, and the miner builds a real
`shared.url_gate.GatedClient` pointed at them:

| Endpoint | Use |
|---|---|
| `GET  {db_url}/experiments/recent?limit=N` | Past experiment rows (SQLite) |
| `GET  {db_url}/experiments/{id}` | One experiment by id |
| `GET  {db_url}/frontier` | Current Pareto front |
| `POST {llm_url}/chat`  `{model?, messages, max_tokens?, temperature?}` | LLM |
| `GET  {llm_url}/models` | Available models |
| `POST {desearch_url}/search`  `{query, max_results?}` | Arxiv via `export.arxiv.org` |
| `GET  {cognition_wiki_url}` | List markdown files |
| `GET  {cognition_wiki_url}/<path>` | Raw markdown content |

LLM provider rules (no key required for the stack to *run*):

| Env var | What happens |
|---|---|
| `CHUTES_API_KEY` | Real proxy to Chutes AI (`https://llm.chutes.ai/v1`) — same provider real radar uses; default model `deepseek-ai/DeepSeek-V3-0324` |
| `OPENAI_API_KEY` | Real proxy to OpenAI Chat Completions |
| neither | Deterministic stub — agent can still test the call path |

Both providers expose the same OpenAI-compatible `POST /llm/chat` shape
(`{model?, messages, temperature?, max_tokens?}`), so the agent code
that ships in the other repo's example miner doesn't need to know
which one is configured.

Wiki: pass `--wiki_dir <path>` to expose any markdown directory.

```bash
CHUTES_API_KEY=cpk_... python local/run.py \
    --agent_dir /path/to/my_agent \
    --wiki_dir  /path/to/notes
```

## Prompt-population optimizer (GEPA / random_mutate)

The local stack reuses the existing `miner_template.optimizers`
plugins against the local SQLite store. `prompt_id` round-trips
end-to-end (`agent → proposal → experiments.prompt_id`) so the
optimizer can attribute Phase C scores back to the variant that
produced each architecture.

```bash
# one-shot (no LLM needed — CI-safe)
python -m local.optimize --agent_dir /path/to/agent \
    --optimizer random_mutate --population 6

# daemon — re-runs whenever a new experiment lands
CHUTES_API_KEY=cpk_... python -m local.optimize \
    --agent_dir /path/to/agent --optimizer gepa --watch
```

Layout written under `<agent_dir>/prompts/` (override with
`--prompts_dir`):

```
prompts/
  active.json          # current population the agent rotates through
  history/
    gen_001.json
    gen_002.json
```

The default `local/agent.py` reads `active.json` via
`miner_template.prompts.load_active` and picks one variant per round
with `pick_for_round(pop, round_id)` — same logic as
`miner_template/agent.py` in real radar.

For GEPA the reflector LM is auto-configured from `CHUTES_API_KEY`
(preferred — points DSPy at `https://llm.chutes.ai/v1`) or
`OPENAI_API_KEY`. Without either, GEPA refuses to run; use
`--optimizer random_mutate` for a no-key loop. Install DSPy with
`pip install dspy` before using `--optimizer gepa`.

## Where to look next

- `local/validator.py` — the round loop. Mirrors the structure of
  `validator/neuron.py` collapsed to single-process.
- `local/trainer.py` — Phase B/C combined. Mirrors
  `runner/timeseries_forecast/harness.py` for the regression task.
- `local/scoring.py` — size gate + sigmoid-improvement + Pareto
  bonus. Same shape as `shared/scoring.py`, smaller surface.
