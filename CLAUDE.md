# Radar Subnet — CLAUDE.md

## What This Is

A Bittensor subnet for autonomous ML research using phase-split validation. Miners host agents that design architectures. Validators coordinate training and independently evaluate checkpoints for consensus.

## Architecture — Phase-Split Validation

```
Phase A (Agent):     Miner agents design architectures (~10 min)
Phase B (Training):  Cross-eval training on miner Basilica pods (~30 min, runs ONCE)
Phase C (Evaluation): Every validator evals every checkpoint (seconds, TRUST ANCHOR)
```

```
shared/          Core libraries shared by validator and miner
validator/       Validator neuron + DB server + Phase C evaluator
miner/           Miner neuron (Basilica deployment)
miner_template/  Starter kit for miners (agent + deploy scripts)
runner/          Per-task frozen environments (runner/timeseries_forecast/)
tasks/           Task definitions (YAML)
tests/           Unit + integration tests
scripts/         Localnet test script
```

## Key Files

| File | Purpose |
|------|---------|
| `shared/protocol.py` | Challenge/Proposal wire format (JSON serializable) |
| `shared/auth.py` | Epistula signing/verification (extracted from gossip.py) |
| `shared/challenge.py` | Deterministic challenge generation, phase timing, size buckets |
| `shared/task.py` | TaskSpec, Objective, YAML loader |
| `shared/database.py` | ExperimentDB — append-only store with flops range + snapshot |
| `shared/pareto.py` | ParetoFront — non-dominated sorting, UCT sampling, get_feasible() |
| `shared/dedup.py` | Code similarity / deduplication |
| `shared/scoring.py` | Size-gated Pareto frontier scoring (Phase C) |
| `shared/commitment.py` | On-chain Docker image + endpoint commitment |
| `shared/r2_audit.py` | R2 storage for checkpoints, snapshots, dispatch records |
| `validator/neuron.py` | Main validator loop (3-phase: collect → train → evaluate) |
| `validator/collection.py` | Phase A: collect submissions from miner agents |
| `validator/coordinator.py` | Phase B: deterministic job assignment, dispatch, R2 I/O |
| `validator/evaluator.py` | Phase C: evaluate checkpoints (trust anchor) |
| `validator/db_server.py` | FastAPI experiment DB with auth + rate limiting |
| `validator/analyzer.py` | Template-based experiment analysis |
| `validator/pod_manager.py` | Affinetes pod lifecycle + code pre-validation |
| `miner/neuron.py` | Miner neuron (deploy agent + trainer on Basilica) |
| `miner_template/agent.py` | Starter miner agent (FastAPI GET /submission/{round_id}) |
| `runner/timeseries_forecast/harness.py` | Frozen training loop (no eval — Phase C does that) |
| `runner/timeseries_forecast/server.py` | Trainer HTTP endpoint (POST /train) |
| `runner/timeseries_forecast/flops.py` | FLOPs-equivalent wallclock calibration |
| `runner/timeseries_forecast/prepare.py` | Data pipeline + validate() |
| `runner/timeseries_forecast/evaluate.py` | CRPS/MASE computation |
| `config.py` | Central config (RADAR_* env vars, round timing, scoring) |

## Commands

```bash
# Install
pip install -e .

# Run tests (185 tests, all passing)
python -m pytest tests/ -v

# Start validator (requires subtensor or mainnet)
python validator/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <name>

# Start miner
python miner/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <name> \
    --agent_image myagent:latest --agent_url <url> --trainer_url <url>

# Build trainer image
docker build -t ts-runner:latest runner/timeseries_forecast/
```

## Scoring Formula

```
# Phase C scoring (all metrics from validator-side evaluation):

# 1. Size gate (hard): flops_equivalent_size in [min, max] for this round's bucket
# 2. Frontier comparison:
#    - No frontier in bucket? Pure relative ranking (bootstrapping)
#    - Frontier exists? Sigmoid of improvement over best frontier CRPS
# 3. Pareto dominance bonus: 1.5× if dominates existing front members
# 4. Penalties: trainer FLOPs mismatch (0.3), trainer failure/timeout (0.5)
# 5. Cross-miner: softmax(temperature=0.1) then EMA(α=0.3) before setting weights
```

## Size Buckets (FLOPs-equivalent)

| Bucket | Min | Max |
|--------|-----|-----|
| Tiny | 100K | 500K |
| Small | 500K | 2M |
| Medium-small | 2M | 10M |
| Medium | 10M | 50M |
| Large | 50M | 125M |

Each round targets one bucket deterministically from the block hash. A 5% tolerance is applied to both the trainer size gate and validator scoring to account for wallclock measurement variance.

## Round Timing

| Phase | Blocks | Duration |
|-------|--------|----------|
| Submission (A) | 50 | ~10 min |
| Training (B) | 150 | ~30 min |
| Evaluation (C) | 25 | ~5 min |
| Fallback/Scoring | 50 | ~10 min |
| **Total** | **275** | **~55 min** |

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

- [x] Phase-split validation pipeline (A → B → C)
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
- [x] 185 unit + integration tests all passing
- [x] DB server with Epistula auth + rate limiting

## What's Outstanding

- [ ] **Real Basilica deployment** — integrate Basilica API for pod lifecycle
- [ ] **Spot checking Phase A** — 10-20% of rounds re-audit agent submissions
- [ ] **Subnet LLM** — provide shared LLM endpoint to miners
- [ ] **Real GIFT-Eval data** — swap placeholder prepare.py with real data pipeline
- [ ] **Mainnet registration** — register subnet, set hyperparameters, deploy
- [ ] **Docker network isolation** — whitelist network for trainer containers
- [ ] **Persistent DB storage** — move from JSON to proper storage
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
