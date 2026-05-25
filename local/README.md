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
| Validator ↔ DB HTTP API | Direct SQLite reads/writes |
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

Drop a file with a top-level `design_architecture(challenge: dict) -> dict`
and point the miner at it:

```bash
python -m local.miner --agent_module my_agent.py
```

The contract matches `miner_template/agent.py`: return
`{code, name, motivation, reasoning, tool_calls, prompt_id}`. The
`code` field is a Python source string that must define
`build_model(input_dim, output_dim)` returning an object with
`hidden_sizes`, `activation`, `learning_rate`, `epochs` attributes —
see `local/agent.py` for the reference emitter.

## Where to look next

- `local/validator.py` — the round loop. Mirrors the structure of
  `validator/neuron.py` collapsed to single-process.
- `local/trainer.py` — Phase B/C combined. Mirrors
  `runner/timeseries_forecast/harness.py` for the regression task.
- `local/scoring.py` — size gate + sigmoid-improvement + Pareto
  bonus. Same shape as `shared/scoring.py`, smaller surface.
