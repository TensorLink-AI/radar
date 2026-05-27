# radar-local

A single-laptop stack for autonomous-ML-research rounds: validator
designs a challenge, a miner agent picks an architecture for the
bucket, a frozen training harness trains and evaluates it on a
held-out split, the validator scores against a size-gated Pareto
frontier. SQLite is the only state store.

```bash
pip install numpy
python local/run.py --rounds 5
```

That spawns two OS processes — `local/validator.py` and
`local/miner.py` — talking through `local/radar_local.db`. Multiple
miners with `--miners 3`; point at your own agent dir with
`--agent_dir`; expose markdown notes to the agent with
`--wiki_dir`.

For LLM access, arxiv search, and the prompt-evolution loop see
[`local/README.md`](local/README.md).

## What's in here

| Path | What it is |
|---|---|
| `local/` | The whole stack — validator, miner, trainer, SQLite store, agent-facing services |
| `miner_template/prompts.py` | Atomic-write prompt-population store (`active.json` + history) |
| `miner_template/optimizers/` | Pluggable optimizer registry; built-ins `random_mutate` and `gepa` |
| `shared/url_gate.py` | `GatedClient` the miner uses to reach validator services |
| `pyproject.toml` | One dependency: `numpy`. GEPA needs `dspy` (extra). |

This repo started as a pruned-down fork of the distributed radar
subnet (Postgres + Targon/RunPod hosting + image hardening + chain
commitments). All of that machinery was removed when we shrank to the
laptop scope — see PR #175 for the diff if you need the history.
