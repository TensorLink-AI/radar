# Radar Subnet — CLAUDE.md

## What This Is

A Bittensor subnet for autonomous ML research using phase-split validation. Miners host agents that design architectures. Validators coordinate training and independently evaluate checkpoints for consensus.

## Architecture — Phase-Split Validation

```
Phase A (Agent):     Miner agents design architectures (~10 min)
Phase B (Training):  Cross-eval training on miner Basilica pods (~30 min, runs ONCE)
Phase C (Evaluation): Every validator evals every checkpoint (seconds, TRUST ANCHOR)
```

### Database Architecture

```
Subnet Owner (database/ module) — one binary, three deploy modes
                                  (RADAR_NEURON_MODE={validator,dashboard,all})
  ├── Runs Postgres (single source of truth, shared across modes)
  ├── FastAPI on RADAR_DB_API_PORT (default 8090)
  │   ├── validator mode  — Epistula-authed validator/miner surface
  │   │   ├── /experiments/*, /challenge, /frontier
  │   │   ├── /provenance/*, /agent_code/*
  │   │   ├── /desearch/*, /llm/*  (optional proxies)
  │   │   └── Internal Jinja operator UI at /dashboard/* when
  │   │       RADAR_DASHBOARD_ENABLED=true (cookie-gated)
  │   ├── dashboard mode  — Public JSON API for radarnet.io SPA
  │   │   └── /dashboard/api/* (no auth, CORS via RADAR_DASHBOARD_CORS_ORIGINS)
  │   └── all mode (default, dev/legacy)  — every surface on one process
  ├── Epistula auth: validator surface only. JSON API is public by design.
  └── Chain sync (metagraph + round loop) runs in {validator, all} modes
      only. Dashboard mode never imports bittensor or touches a wallet.

Public dashboard JSON API (/dashboard/api/*) is PUBLIC by design — anyone
on the internet can read agent source code, stdout, experiments, and
frontier data. Do not add fields to those endpoints that shouldn't be
world-readable.

Validators (validator/ module)
  ├── Phase A, B, C logic unchanged
  ├── NO local database — HTTP client of database server
  ├── Reverse-proxy FastAPI on RADAR_PROXY_PORT (default 8080)
  │   ├── Proxies /experiments/*, /challenge, /frontier → database server
  │   ├── Hosts /desearch/* locally (desearch_proxy.py)
  │   └── Rate limits per miner
  └── After Phase C: POST experiments via DatabaseClient

Miners (unchanged)
  └── Query validator proxy URL from challenge["db_url"]
```

```
database/        Centralized Postgres DB server (subnet owner)
shared/          Core libraries shared by all components
validator/       Validator neuron + proxy + Phase C evaluator
miner/           Miner neuron (Basilica deployment)
miner_template/  Starter kit for miners (agent + deploy scripts)
runner/          Per-task frozen environments (runner/timeseries_forecast/)
tasks/           Task definitions (YAML)
tests/           Unit + integration tests
scripts/         Localnet test script + Postgres startup
```

## Key Files

| File | Purpose |
|------|---------|
| `shared/protocol.py` | Challenge/Proposal wire format (JSON serializable) |
| `shared/auth.py` | Epistula signing/verification |
| `shared/challenge.py` | Deterministic challenge generation, phase timing, size buckets |
| `shared/task.py` | TaskSpec, Objective, YAML loader |
| `shared/database.py` | DataElement dataclass + deprecated ExperimentDB (JSON) |
| `shared/pg_schema.py` | Postgres DDL, row conversion, diff helpers |
| `shared/pg_store.py` | PgExperimentStore — async Postgres experiment store |
| `shared/pg_provenance.py` | PgProvenanceQuery — async Postgres provenance |
| `shared/pg_access_logger.py` | PgAccessLogger — async Postgres access logger |
| `shared/db_client.py` | DatabaseClient — HTTP client for validators → DB server |
| `shared/provenance.py` | Pure-Python helpers (detect_components, compute_similarity) |
| `shared/access_logger.py` | Pure-Python helper (_extract_experiment_ids) |
| `shared/pareto.py` | ParetoFront — non-dominated sorting, UCT sampling |
| `shared/dedup.py` | Code similarity (provenance queries) |
| `shared/scoring.py` | Size-gated Pareto frontier scoring (Phase C) |
| `shared/commitment.py` | On-chain Docker image + endpoint commitment |
| `shared/r2_audit.py` | R2 storage for checkpoints, snapshots, dispatch records |
| `database/server.py` | Centralized DB API (FastAPI, all experiment routes) |
| `database/neuron.py` | Subnet owner process (Postgres + API server) |
| `validator/neuron.py` | Main validator loop (3-phase: collect → train → evaluate) |
| `validator/db_proxy.py` | Reverse proxy for miners (forwards to DB server) |
| `validator/collection.py` | Phase A: collect submissions from miner agents |
| `validator/coordinator.py` | Phase B: deterministic job assignment, dispatch, R2 I/O |
| `validator/evaluator.py` | Phase C: evaluate checkpoints (trust anchor) |
| `validator/desearch_proxy.py` | Rate-limited arxiv search proxy |
| `validator/analyzer.py` | Template-based experiment analysis |
| `validator/pod_manager.py` | Affinetes pod lifecycle + code pre-validation |
| `miner/neuron.py` | Miner neuron (deploy agent + trainer on Basilica) |
| `miner_template/agent.py` | Starter miner agent |
| `runner/timeseries_forecast/harness.py` | Frozen training loop with recipe hooks |
| `runner/timeseries_forecast/server.py` | Trainer HTTP endpoint (POST /train) |
| `runner/timeseries_forecast/flops.py` | FLOPs-equivalent wallclock calibration |
| `runner/timeseries_forecast/prepare.py` | Data pipeline + validate() |
| `runner/timeseries_forecast/evaluate.py` | CRPS/MASE computation |
| `config.py` | Central config (RADAR_* env vars, round timing, scoring) |

## Commands

```bash
# Install
pip install -e .

# Start Postgres (local dev)
scripts/start_pg.sh

# Run tests (315+ tests, all passing; Postgres tests need TEST_PG_DSN)
python -m pytest tests/ -v

# Start database server — default (all) mode: validator surface + public JSON
python database/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <name>

# Validator-only mode (no public JSON API; internal Jinja UI optional)
RADAR_NEURON_MODE=validator \
  python database/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <name>

# Public dashboard-only mode (no wallet, no chain sync, no Epistula)
RADAR_NEURON_MODE=dashboard \
  RADAR_DASHBOARD_CORS_ORIGINS=https://radarnet.io \
  python database/neuron.py --port 8091

# Running both locally on the same Postgres (dev)
# Terminal 1: validator surface on 8090
RADAR_NEURON_MODE=validator python database/neuron.py --netuid 1 --port 8090
# Terminal 2: public JSON API on 8091 pointing at the same Postgres
RADAR_NEURON_MODE=dashboard python database/neuron.py --port 8091

# Start validator (requires subtensor or mainnet)
python validator/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <name>

# Start miner
python miner/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <name> \
    --agent_image myagent:latest --agent_url <url> --trainer_url <url>

# Build trainer image
docker build -t ts-runner:latest runner/timeseries_forecast/
```

## Postgres Setup

```bash
# Local dev (Docker)
docker run -d --name radar-pg \
  -e POSTGRES_USER=radar -e POSTGRES_PASSWORD=radar -e POSTGRES_DB=radar \
  -p 5432:5432 -v radar-pg-data:/var/lib/postgresql/data \
  postgres:16-alpine

# Set in .env
RADAR_PG_DSN=postgresql://radar:radar@localhost:5432/radar

# Production (Supabase or managed Postgres)
RADAR_PG_DSN=postgresql://postgres.[ref]:[password]@db.[ref].supabase.co:5432/postgres
```

## Scoring Formula

```
# Phase C scoring (all metrics from validator-side evaluation):

# 1. Size gate (hard): flops_equivalent_size in [min, max] for this round's bucket
# 2. Frontier comparison:
#    - No frontier in bucket? Pure relative ranking (bootstrapping)
#    - Frontier exists? Sigmoid of improvement over best frontier CRPS
# 3. Pareto dominance bonus: 1.5x if dominates existing front members
# 4. Penalties: trainer FLOPs mismatch (0.3), trainer failure/timeout (0.5)
# 5. Cross-miner: softmax(temperature=0.1) then EMA(alpha=0.3) before setting weights
```

## Size Buckets (FLOPs-equivalent)

| Bucket | Min | Max |
|--------|-----|-----|
| Tiny | 100K | 500K |
| Small | 500K | 2M |
| Medium-small | 2M | 10M |
| Medium | 10M | 50M |
| Large | 50M | 125M |

Each round targets one bucket deterministically from the block hash. FLOPs measured via analytical counting (torch.utils.flop_counter) with wallclock calibration fallback. 10% tolerance.

## Round Timing

Three layers of timing, intentionally separated:

1. **Block windows** (on-chain, validator-global): define WHEN phases start/end. Consensus-driven via block height (~12s/block). Rigid boundaries. Env: `RADAR_SUBMISSION_WINDOW`, `RADAR_TRAINING_WINDOW`, `RADAR_EVAL_WINDOW`, `RADAR_FALLBACK_WINDOW`.
2. **Validator operational timeouts** (seconds, validator-global): HTTP / R2 polling guardrails. E.g. `TRAINER_PREPARE_TIMEOUT`.
3. **Per-task second budgets** (seconds, set in `tasks/<task>/<task>.yaml`): different tasks can demand different amounts of work per phase.
   - `agent_seconds` → Phase A wall-clock for the **agent pod** (0/unset = inherit `Config.AGENT_TIMEOUT`)
   - `time_budget` → Phase B wall-clock for the **trainer's training loop**
   - `kill_timeout` → Phase B hard subprocess kill (outer safety net)

| Phase | Block Window | Per-task budget | Global default | Controls |
|-------|-------------|-----------------|----------------|----------|
| Submission (A) | 50 blocks (~10 min) | `agent_seconds` | `AGENT_TIMEOUT` (600s) | Agent pod wall-clock |
| Training (B)   | 150 blocks (~30 min) | `time_budget` / `kill_timeout` | 300s / 600s | Trainer training loop + kill |
| Evaluation (C) | 25 blocks (~5 min) | — | `EVAL_WINDOW_BLOCKS × 12` (300s) | R2 checkpoint polling |
| Fallback/Scoring | 50 blocks (~10 min) | — | `FALLBACK_WINDOW_BLOCKS × 12` (600s) | Re-dispatch polling |
| **Total** | **275 (~55 min)** | | | |

Note: `RADAR_TIME_BUDGET` used to silently override every task's `time_budget`; it has been removed. Edit the per-task YAML instead.

## R2 Bucket Path Convention

```
snapshots/round_{round_id}/db.json                         # DB snapshot (Phase A)
round_{round_id}/miner_{hotkey}/checkpoint.safetensors       # Model weights (Phase B)
round_{round_id}/miner_{hotkey}/architecture.py             # Architecture code (Phase B)
round_{round_id}/miner_{hotkey}/training_meta.json          # Training metadata (Phase B)
round_{round_id}/dispatch/vali_{hotkey}.json                # Dispatch records (Phase B)
frontier/latest.json                                        # Current Pareto frontier
```

## What's Done

- [x] Phase-split validation pipeline (A -> B -> C)
- [x] Epistula authentication (`shared/auth.py`)
- [x] Deterministic challenge generation with size buckets (`shared/challenge.py`)
- [x] Size-gated Pareto frontier scoring (`shared/scoring.py`)
- [x] FLOPs-equivalent wallclock calibration (`runner/timeseries_forecast/flops.py`)
- [x] Trainer HTTP server (`runner/timeseries_forecast/server.py`)
- [x] Phase C validator-side evaluation (`validator/evaluator.py`)
- [x] Submission collection from miner agents (`validator/collection.py`)
- [x] Training coordinator with cross-eval (`validator/coordinator.py`)
- [x] DB snapshot for frozen miner state
- [x] Miner template (starter agent + deploy)
- [x] 315+ unit + integration tests all passing
- [x] Centralized Postgres DB with async API (`database/`, `shared/pg_*.py`)
- [x] Validator reverse proxy for miners (`validator/db_proxy.py`)
- [x] DatabaseClient for validator -> DB server communication (`shared/db_client.py`)
- [x] Validator / dashboard deploy split (`RADAR_NEURON_MODE`)

## Neuron Mode Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `RADAR_NEURON_MODE` | `all` | `validator` (Epistula API + chain), `dashboard` (open public JSON), or `all` (both on one process). |
| `RADAR_PG_POOL_MIN` | `2` | asyncpg pool min size. Dashboard deploys typically leave as-is; validator deploys can bump. |
| `RADAR_PG_POOL_MAX` | `10` | asyncpg pool max size. |
| `RADAR_PG_STATEMENT_TIMEOUT_MS` | `0` | `SET statement_timeout` for every pool connection. `0` disables. Set to `5000` on dashboard deploys so a slow public query can't pressure the shared cluster. |
| `RADAR_PG_STARTUP_RETRIES` | `6` | Retries for the asyncpg bootstrap connect + pool create, so the container doesn't crash-loop while Postgres is still coming up. |
| `RADAR_PG_STARTUP_BACKOFF_INITIAL_S` | `1.0` | Initial wait between startup retries (doubles each attempt). |
| `RADAR_PG_STARTUP_BACKOFF_MAX_S` | `30.0` | Cap on the exponential backoff between startup retries. |
| `RADAR_DASHBOARD_CORS_ORIGINS` | `""` | Comma-separated CORS origins for `/dashboard/api/*`. Empty disables CORS. Production dashboard sets this to `https://radarnet.io`. |
| `RADAR_DASHBOARD_ENABLED` | `false` | Mounts the cookie-gated internal Jinja operator UI at `/dashboard/*` (validator / all modes only). |
| `RADAR_DASHBOARD_KEY` | `""` | Shared key required to log into the Jinja UI. |

The `/dashboard/api/*` JSON endpoints are **PUBLIC**. Do not add any field
to them that shouldn't be world-readable. The Jinja `/dashboard/*` HTML
pages remain operator-gated by cookie auth.

In production these modes run as separate deployments with independent
lifecycles (validator on a VPS, dashboard on Railway, SPA on Vercel).

## What's Outstanding

- [ ] **Real Basilica deployment** — integrate Basilica API for pod lifecycle
- [ ] **Spot checking Phase A** — 10-20% of rounds re-audit agent submissions
- [ ] **Subnet LLM** — provide shared LLM endpoint to miners
- [ ] **Real GIFT-Eval data** — swap placeholder prepare.py with real data pipeline
- [ ] **Mainnet registration** — register subnet, set hyperparameters, deploy
- [ ] **Docker network isolation** — whitelist network for trainer containers
- [ ] **Cross-tempo EMA** — weight smoothing across rounds

## Bittensor SDK

Using `bittensor>=10.1.0`. Key classes: `bt.Wallet`, `bt.Subtensor`, `bt.Keypair`.

## Code Style

- No file over 300 lines
- Tests for every module in `tests/`
- Type hints on all public functions
- `@dataclass` for data types (not Pydantic, matching protocol.py)
- Config via `os.getenv("RADAR_*", default)` in `config.py`
- Logging via `logger = logging.getLogger(__name__)`
