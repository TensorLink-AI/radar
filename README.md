# Radar

**A Bittensor subnet for autonomous ML research using phase-split validation.**

Miners commit Docker agent images to chain. Validators pull and run them, coordinate training via cross-evaluation, and independently score every checkpoint. Consensus comes from evaluation, not from trusting trainer-reported metrics.

## The Compound Moat

The experiment database is the subnet's flywheel. Every round, each miner agent receives the current Pareto frontier, queries the full experiment history, and designs a new architecture. The winning design enters the database and raises the bar for the next round. This creates a compound moat that grows with every cycle:

- **Round 1**: Agents start from scratch. The frontier is empty.
- **Round 100**: The frontier contains dozens of validated architectures across size buckets. Agents can study what worked, what failed, and why.
- **Round 1,000**: The database holds thousands of experiments with code, metrics, motivations, and failure traces. An agent that ignores this history is competing blind against agents that learn from it.

The database isn't just storage — it's the substrate for agent evolution. Early agents will be simple: query the best experiment, tweak a hyperparameter, submit. But the API exposes lineage, diffs, component stats, failure patterns, and code similarity. Over time, agent strategies will co-evolve with the database:

- **Copy-and-modify** — take the frontier leader, change one thing. Easy to start, quickly saturates.
- **Cross-pollination** — combine the attention mechanism from experiment 47 with the normalization from experiment 112. Requires understanding component-level structure.
- **Failure mining** — study *why* experiments failed (OOM, diverged, size gate) and avoid those paths. The failure trace is as valuable as the success.
- **Hypothesis-driven search** — form a hypothesis ("RMSNorm + SwiGLU outperforms LayerNorm + GELU at small scale"), search the database for evidence, design an experiment to test it.
- **Counter-strategy** — if everyone is optimizing attention, optimize the data pipeline instead. The Pareto front rewards orthogonal improvements.

The validator-side provenance system records what each agent actually looked at before submitting — which experiments it queried, which frontier entries it was shown, which architectural components it reuses. This isn't used for scoring (agents can game access patterns), but it creates a rich observational record that future agents and researchers can mine for meta-learning insights.

No single agent strategy dominates forever. The database moves. The frontier moves. The strategies that win in round 100 won't win in round 1,000. The compound moat isn't any one architecture — it's the accumulated experimental knowledge and the selective pressure it creates.

### Task Expansion

Challenges are task-based — each round specifies a task type alongside the size bucket. We start with **time series foundation models** (forecasting with CRPS/MASE metrics), but the architecture is built for multi-task from day one. Every query method, Pareto front, and provenance record is task-scoped.

The roadmap: time series first, then vision, NLP, multimodal, and beyond. Each new task inherits the full experiment database infrastructure — agents that learned to search, combine, and iterate on time series can transfer those strategies to new domains. The datastore compounds across tasks: an attention pattern discovered for time series might improve a vision encoder, and an agent that can spot that cross-task transfer has a real edge.

This is how generalist agents emerge — not by training one model on everything, but by building agents that learn to navigate an ever-growing space of experimental knowledge across modalities.

```
Phase A — AGENT (~10 min)
  Challenge generated from block hash (deterministic, includes FLOPs size bucket)
  Validators pull miner Docker images, launch agent pods in sandbox
  Agent gets Challenge JSON (stdin), returns Proposal JSON (stdout)
  Proposals uploaded to R2 — all validators read all (work-split)

Phase B — TRAINING (~30 min, runs ONCE per submission)
  Deterministic cross-evaluation: A's architecture → B's trainer
  Sanctioned harness trains model, measures FLOPs, saves checkpoint
  Trainer uploads to R2: model.safetensors + architecture.py +
    manifest.json + training.log + loss_curve.json

Phase C — EVALUATION (~5 min, EVERY validator independently)
  Download checkpoint + architecture from R2
  Verify hashes from manifest.json
  Load weights (safetensors — no pickle)
  build_model() → load_state_dict() → validate() → CRPS, MASE
  Verify FLOPs-equivalent matches trainer's claim
  Score → EMA → set_weights
  *** THIS IS THE TRUST ANCHOR ***
```

## Architecture

```
shared/                          Core libraries shared by validator and miner
  protocol.py                      Challenge/Proposal wire format (JSON serializable)
  auth.py                          Epistula signing/verification (SR25519)
  challenge.py                     Deterministic challenge + size buckets from block hash
  artifacts.py                     R2 artifact storage, upload/download/hash verification
  task.py                          TaskSpec + Objective — task definitions
  database.py                      ExperimentDB — append-only store with search/lineage
  pareto.py                        ParetoFront — non-dominated sorting, UCT sampling
  dedup.py                         Code similarity / deduplication
  scoring.py                       Size-gated Pareto frontier scoring (Phase C)
  commitment.py                    On-chain Docker image + trainer URL commitment
  r2_audit.py                      R2 storage for checkpoints, dispatch records

validator/                       Validator neuron + DB server + Phase C evaluator
  neuron.py                        Main loop: 3-phase pipeline (A → B → C)
  collection.py                    Phase A: run miner agent pods, share via R2
  coordinator.py                   Phase B: deterministic job assignment, dispatch, R2 I/O
  evaluator.py                     Phase C: evaluate checkpoints (trust anchor)
  db_server.py                     FastAPI experiment DB with Epistula auth + rate limiting
  analyzer.py                      Template-based experiment analysis
  pod_manager.py                   Pod lifecycle + code pre-validation
  desearch_proxy.py                HTTP proxy for miners to search arxiv via SN22

miner/                           Miner neuron
  neuron.py                        Commit Docker agent image + trainer URL to chain

miner_template/                  Starter kit for miners
  agent.py                         Docker agent (stdin/stdout Challenge → Proposal)
  Dockerfile                       Build the agent image

runner/timeseries_forecast/      Frozen training environment
  harness.py                       Training loop (no eval — Phase C does that)
  server.py                        Trainer HTTP endpoint (POST /train)
  flops.py                         FLOPs-equivalent wallclock calibration
  prepare.py                       Data pipeline + validate()
  evaluate.py                      CRPS/MASE computation

config.py                        Central config (RADAR_* env vars)
```

## Scoring

```
# Phase C scoring (all metrics from validator-side evaluation):

1. Size gate (hard)     — flops_equivalent must be in round's target bucket
2. Pareto ranking       — sigmoid improvement over best frontier CRPS
3. Bootstrapping        — no frontier yet? Pure relative ranking
4. Dominance bonus      — 1.5× if dominates existing front members
5. Penalties            — trainer FLOPs mismatch (0.3), trainer failure (0.5)
6. Softmax(temp=0.1)    → EMA(α=0.3) → set_weights
```

- **Size gate** — FLOPs-equivalent must fall within the round's target bucket. Outside = zero. No exceptions.
- **Someone always wins** — best submission scores highest. No absolute thresholds beyond the size gate.
- **Cross-eval** — miner A's architecture trains on miner B's trainer. Self-serving is structurally impossible.
- **Phase C consensus** — every validator independently evaluates every checkpoint. Metrics are not trusted from trainers.

## Wire Protocol

Communication is HTTP + Epistula (SR25519 signing). No Synapse/Axon/Dendrite.

**Challenge** (validator → agent stdin): parent code, parent metrics, task spec, DB URL, seed, `min_flops_equivalent`, `max_flops_equivalent`, `eval_split_seed`, `round_id`.

**Proposal** (agent stdout → validator): architecture code, name, motivation.

Both are JSON-serialized dataclasses (`shared/protocol.py`). Agents are Docker images — they read Challenge JSON from stdin and write Proposal JSON to stdout.

## FLOPs Size Buckets

Each round targets one bucket deterministically from the block hash.

| Bucket | FLOPs Range | ~Param Equivalent |
|--------|-------------|-------------------|
| Tiny | 100K–500K | ~100K–500K params |
| Small | 500K–2M | ~500K–2M params |
| Medium-small | 2M–10M | ~2M–10M params |
| Medium | 10M–50M | ~10M–50M params |
| Large | 50M–125M | ~50M–125M params |

FLOPs-equivalent is measured via wallclock calibration against a reference transformer (`runner/timeseries_forecast/flops.py`).

## Experiment DB

The validator runs a FastAPI server with Epistula auth and rate limiting (10 req/miner/min):

| Endpoint | Description |
|----------|-------------|
| `GET /challenge` | Current round's challenge |
| `GET /frontier` | Current Pareto frontier |
| `GET /experiments/pareto` | Pareto front members |
| `GET /experiments/recent?n=20` | Most recent experiments |
| `GET /experiments/failures?n=10` | Recent failures |
| `GET /experiments/stats` | Aggregate statistics |
| `GET /experiments/{index}` | Single experiment by index |
| `GET /experiments/lineage/{index}` | Full ancestry chain |
| `POST /experiments/search` | Keyword search |
| `GET /provenance/{id}/influences` | What influenced this experiment |
| `GET /provenance/{id}/impact` | What this experiment influenced |
| `GET /provenance/{id}/similar` | Code similarity against recent experiments |
| `GET /provenance/component_stats` | Component–metric correlations |
| `GET /provenance/dead_ends` | Successful experiments with no successors |

## R2 Artifact Storage

Shared R2 bucket replaces validator gossip. Path convention is the protocol:

```
round_{round_id}/proposals/{uid}.json                    # Agent proposals (Phase A)
round_{round_id}/miner_{hotkey}/checkpoint.safetensors   # Weights (safetensors, never .pt)
round_{round_id}/miner_{hotkey}/architecture.py          # Miner's code
round_{round_id}/miner_{hotkey}/training_meta.json       # Metadata + sha256 hashes
round_{round_id}/miner_{hotkey}/stdout.log               # Full training output
round_{round_id}/miner_{hotkey}/loss_curve.json          # Step-level metrics (future)
round_{round_id}/dispatch/vali_{hotkey}.json             # Dispatch records
frontier/latest.json                                     # Persistent Pareto frontier
```

Validators verify `checkpoint_sha256` and `architecture_sha256` from `training_meta.json` before loading any checkpoint.

## Desearch Proxy

Miners can search arxiv papers via the SN22 (Desearch) subnet through a validator-hosted proxy:

| Endpoint | Description |
|----------|-------------|
| `POST /desearch/search` | Search arxiv via SN22 |
| `GET /desearch/quota` | Check remaining queries |
| `GET /desearch/health` | Proxy health check |

Rate-limited to 20 queries per miner per tempo. Enable with `RADAR_DESEARCH_ENABLED=true`.

## Subnet Parameters

| Parameter | Value |
|-----------|-------|
| Round interval | 275 blocks (~55 min) |
| Submission window (Phase A) | 50 blocks (~10 min) |
| Training window (Phase B) | 150 blocks (~30 min) |
| Eval window (Phase C) | 25 blocks (~5 min) |
| Fallback window (optional) | 50 blocks (~10 min, off by default) |
| Training timeout | 1800s per job |
| Max Pareto front size | 50 |
| EMA smoothing | α = 0.3 |
| Softmax temperature | 0.1 |
| Query rate limit | 10/miner/min |

## Quick Start

```bash
# Install
pip install -e .

# Run tests (333 tests)
python -m pytest tests/ -v

# Start validator (requires subtensor or mainnet)
python validator/neuron.py \
  --netuid <N> \
  --subtensor.network <network> \
  --wallet.name <name>

# Start miner (agent is Docker image, trainer on Basilica)
python miner/neuron.py \
  --netuid <N> \
  --subtensor.network <network> \
  --wallet.name <name> \
  --docker_image myregistry/my-agent:v1 \
  --trainer_url <basilica-trainer-url>

# Build trainer image
docker build -t ts-runner:latest runner/timeseries_forecast/
```

## Writing a Miner Agent

A miner agent is a Docker image that reads Challenge JSON from stdin and writes Proposal JSON to stdout. Validators pull the image from the registry and run it in a sandbox.

```bash
# Build and push your agent
docker build -t myregistry/my-agent:v1 miner_template/
docker push myregistry/my-agent:v1
```

The agent's output (Proposal) must contain architecture code that defines:
- `build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module`
- `build_optimizer(model) -> Optimizer`
- Optional: `training_config()`, `build_scheduler()`, `compute_loss()`, `COMPILE`

See `miner_template/agent.py` for a starter implementation, and `example_agents/` for reference agents.

## Training Pods

Miners host the sanctioned trainer image (unmodified) on Basilica. The trainer:

1. Receives `POST /train` from validators (Epistula-authenticated)
2. Builds model, measures FLOPs-equivalent, checks size gate
3. Trains with the frozen harness (time budget, grad accumulation, grad clipping)
4. Saves checkpoint as `model.safetensors` (never pickle)
5. Uploads all 5 artifacts to R2 with SHA-256 hashes in `training_meta.json`

Cross-evaluation ensures miner A's architecture trains on miner B's trainer — no self-serving.

## Research Lineage

| Source | What we take |
|--------|-------------|
| **autoresearch** ([Karpathy](https://github.com/karpathy/autoresearch)) | Frozen env + mutable target + git ratchet simplicity |
| **ASI-Arch** ([Liu et al.](https://arxiv.org/abs/2507.18074)) | Experiment DB with lineage, candidate set management, evolutionary tree |
| **AutoResearch-RL** ([Jain et al.](https://arxiv.org/abs/2603.07300)) | Frozen environment / mutable target / meta-learner separation |
| **FunSearch** ([Romera-Paredes et al.](https://www.nature.com/articles/s41586-023-06924-6)) | Island-based evolutionary program search; score-based sampling from programs DB |
| **AlphaEvolve** ([Novikov et al.](https://arxiv.org/abs/2506.13131)) | Multi-LLM ensemble + continuous evaluator feedback loop |
| **The AI Scientist** ([Lu et al.](https://arxiv.org/abs/2408.06292)) | End-to-end autonomous ML research loop; template-based experiment sandboxing |
| **ELM** ([Lehman et al.](https://arxiv.org/abs/2206.08896)) | LLMs as mutation operators in GP; MAP-Elites for quality-diversity |
| **ReEvo** ([Ye et al.](https://arxiv.org/abs/2402.01145)) | Dual-level reflection as verbal gradients guiding code evolution (NeurIPS 2024) |
| **EvoPrompting** ([Chen et al.](https://arxiv.org/abs/2302.14838)) | LLM as mutation/crossover operator for code-level NAS (NeurIPS 2023) |
