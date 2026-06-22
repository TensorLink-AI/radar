# meta — orchestrator for radar experimental units

Sibling to `local/`. Treats one radar-local stack (one validator +
N miners on one host) as the "experimental unit", spins them up on
RunPod / Vast / Lambda, monitors them, budget-caps them, and aggregates
results across experiments so a higher-level Claude Code session (the
"Ralph loop") can run a continuous research loop.

**Does not modify `local/`, `miners/`, `miner_template/`, or `shared/`.**
The orchestrator talks to those packages the same way an external user
would: launching `python local/run.py` over SSH and reading the R2
paths the existing code already writes (`local/backup.py`,
`local/artifacts.py`, `local/export_events.py`).

## Quick start

```bash
# Install the meta extra (PyYAML + paramiko + httpx for RunPod):
pip install -e .[meta]

# Submit a YAML spec (queues each unit, doesn't spawn pods yet):
python -m meta submit meta/experiments/example.yaml

# Start the orchestrator daemon (scheduler + poller + ingester threads):
RUNPOD_API_KEY=... HIPPIUS_ACCESS_KEY_ID=... HIPPIUS_SECRET_ACCESS_KEY=... \
  python -m meta daemon --orch-id myorch --ssh-key-path ~/.ssh/runpod_key

# Watch state:
python -m meta status                                          # all experiments
python -m meta status --experiment cont-equilibrium-sweep      # one experiment + units
python -m meta xcompare                                        # cross-experiment digest
python -m meta logs --experiment X --unit eq70 --tail 200      # tail validator log
python -m meta compare --experiment X                          # per-unit frontier diff
python -m meta export --experiment X                           # write report.md
python -m meta kill --experiment X --unit eq70 --reason "manual"

# Dashboard:
python -m meta.dashboard --port 8766
# → open http://127.0.0.1:8766/

# Ralph loop (continuous meta-optimizer):
python -m meta ralph \
    --prompt meta/prompts/researcher.md \
    --interval 30m \
    --max-concurrent-experiments 4 \
    --total-budget-usd 500
```

## Layout

```
meta/
  __init__.py / __main__.py    CLI dispatch
  spec.py                      YAML schema + validation
  store.py                     registry.db + per-experiment meta.db
  providers/                   base, mock, runpod
  ssh.py                       paramiko/subprocess SSH wrapper
  ingest.py                    R2 latest.db.gz → frontier table
  orchestrator.py              scheduler / poller / ingester threads
  xquery.py                    cross-experiment SQL
  ralph.py                     headless Claude Code driver
  dashboard.py / dashboard.html  read-only HTTP dashboard
  prompts/researcher.md        the meta-optimizer prompt
  experiments/                 versioned ablation YAMLs
  data/                        runtime state (registry.db + experiments/)
  settings.example.json        Claude permission template for Ralph
```

## Storage layout (runtime)

```
meta/data/
  registry.db                          # tiny: list of experiments
  experiments/
    cont-equilibrium-sweep-2026-06/
      meta.db                          # units + meta_events + frontier
      units/
        eq50/local_snapshot.db         # gunzipped radar_local.db (read-only)
        eq70/local_snapshot.db
      report.md                        # Claude's writeup
      spec.yaml                        # audit copy of the submitted YAML
  scratch/                             # Ralph session contexts + logs
```

One SQLite per experiment, plus a tiny global registry — mirrors how
`local/` already works (one DB per validator). Dropping one
experiment's DB never touches any other.

## Cost discipline

`pod.gpu` selects a GPU family. `cost.max_usd_per_hour` is a hard
filter on pod-types — the scheduler asks the provider for the cheapest
match at or below that rate and **rejects the experiment** if nothing
qualifies. `cost.budget_usd` caps total spend per experiment; the
poller bumps `units.cost_usd` each heartbeat and SIGTERMs any unit
once the budget is crossed. The Ralph loop has its own
`--total-budget-usd` cap on top — three layers of brakes.

## Isolation from other R2 producers

The orchestrator mints instance IDs as
`orch-<orch_id>-<experiment>-<unit>` and only reads R2 keys for
instance IDs it minted. It never lists buckets or directories — so a
teammate's instances on the same bucket are untouched. `xcompare` is
local-only: it queries `registry.db` + the experiment `meta.db` files
this orchestrator maintains.

## Permissions (Ralph session)

The Ralph-loop Claude session runs under `meta/settings.example.json`
(copied to `~/.claude/settings.json` on the orchestrator pod). It can
write only:

- `meta/experiments/*.yaml`         (new ablation specs)
- `meta/data/experiments/*/report.md` (analysis writeups)
- `meta/data/scratch/*.md`          (scratchpad)
- `miners/ralph_*/**`               (new agents it scaffolds — never existing miners)

and call only the allowlisted `meta` CLI verbs. `local/`, `shared/`,
`miner_template/`, the orchestrator's own code, and any non-`ralph_*`
miner are hard-denied.
