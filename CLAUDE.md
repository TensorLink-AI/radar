# CLAUDE.md — radar-local

## What this is

A single-laptop validator + miner pair that runs phase A → B → C of
the (formerly distributed) radar workflow against a SQLite broker.
Two OS processes, no Docker, no Postgres, no chain, no HMAC. See
`README.md` for the user-facing intro and `local/README.md` for the
full surface.

## Layout

```
local/           The whole stack (one file per responsibility).
miners/          Self-contained miner agents (autonomous,
                 claude_style[_v2], openai_sdk[_v2], patch_decoder).
                 Each subdir is a valid --agent_dir target.
miner_template/  Trimmed to prompts.py + optimizers/ for the
                 prompt-evolution loop.
shared/          url_gate.py (GatedClient) + gift_eval.py (manifest,
                 R2 download, Arrow loader) + r2_audit.py
                 (HippiusStorage / R2AuditLog S3 client).
tests/           pytest suite for the shared/ modules.
```

## GIFT-Eval / R2 access

`shared.r2_audit.HippiusStorage` (alias `R2AuditLog`) is the
S3-compatible client. It reads creds from env in this order:
`HIPPIUS_*` first, then legacy `R2_*` (`R2_ACCOUNT_ID` derives the
per-account endpoint). The default Hippius endpoint is
`https://s3.hippius.com`; pointing at `*.r2.cloudflarestorage.com`
forces region `auto` instead of `decentralized`.

`shared.gift_eval` owns the GIFT-Eval manifest (key → R2 subpath),
the SHORT/MED_LONG leaderboard lists, deterministic per-round
dataset selection, Arrow IPC parsing, and rolling-origin window
construction. The R2 prefix is
`gift-eval-benchmark/gift-eval-full/{subpath}/data-00000-of-00001.arrow`.

`python -m local.fetch_gift_eval` is the CLI wrapper over
`ensure_datasets_cached()` — use it to prefetch Arrow files into
`$RADAR_GIFT_EVAL_CACHE` (default `/tmp/radar_gift_eval`). Requires
`pip install -e .[gift_eval]` (pulls boto3 + pyarrow).

## Key files

| File | Purpose |
|------|---------|
| `local/store.py` | SQLite (WAL) broker — challenges, proposals, experiments. |
| `local/services.py` | Threaded stdlib HTTP server on 127.0.0.1 — `/experiments`, `/llm/chat`, `/desearch/search`, `/wiki`. |
| `local/providers.py` | LLM (Chutes → OpenAI → stub), arxiv via `export.arxiv.org`, file-backed wiki. |
| `local/validator.py` | Round loop: picks bucket → publishes challenge → drains proposals → trains + evaluates → scores. |
| `local/miner.py` | Polls for challenges, loads the agent from `--agent_dir`, calls `design_architecture(challenge, client?)`. |
| `local/agent.py` | Default miner agent. Reads the active prompt via `miner_template.prompts`. |
| `local/trainer.py` | Frozen Phase B+C: numpy MLP forward/backward, MSE on held-out test split. |
| `local/scoring.py` | Size gate + sigmoid-improvement + Pareto bonus. |
| `local/task.py` | Synthetic 8-dim regression + FLOPs-equivalent size buckets. |
| `local/optimize.py` | Prompt-population CLI (`gepa` / `random_mutate`). |
| `local/run.py` | Launches validator + N miners as subprocesses. |
| `miner_template/prompts.py` | `active.json` + `history/gen_NNN.json` atomic-write population store. |
| `miner_template/optimizers/` | Pluggable optimizer registry (`gepa`, `random_mutate`, `pkg.mod:func`). |
| `shared/url_gate.py` | `GatedClient` enforcing the per-challenge `allowed_urls`. |

## Commands

```bash
# Install
pip install -e .

# Single-laptop run
python local/run.py --rounds 5
python local/run.py --agent_dir /path/to/agent --wiki_dir /path/to/notes --miners 3

# Prompt evolution
python -m local.optimize --agent_dir /path/to/agent --optimizer random_mutate
CHUTES_API_KEY=cpk_... python -m local.optimize \
    --agent_dir /path/to/agent --optimizer gepa --watch
```

## Code style

- No file over 300 lines.
- Type hints on public functions.
- `@dataclass` for plain data.
- `logger = logging.getLogger(__name__)`.
- The stack must run with only `numpy` installed; everything else is
  optional (DSPy for GEPA, a provider key for the real LLM proxy).
