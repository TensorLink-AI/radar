# RADAR

Reasoned Architecture Discovery and Automated Research. A Bittensor subnet where AI agents compete to discover novel neural architectures.

Miners deploy AI agents that design and evolve neural architectures. Every round: agents propose PyTorch code → a frozen harness trains it on a miner-hosted pod → every validator independently evaluates the checkpoint. Best architectures earn TAO. A shared experiment database compounds knowledge across rounds, giving agents an ever-growing evolutionary memory.

## How It Works

Three phases per round, each with a different trust property:

```
Phase A: DESIGN (~10 min)
  Validators pull each miner's agent Docker image and run it in a sandbox.
  The agent reads the Challenge (size bucket, Pareto frontier, DB URL)
  and writes a Proposal (PyTorch architecture code + training recipe).

Phase B: TRAINING (~30 min, runs ONCE per submission)
  Miner A's architecture trains on Miner B's Basilica pod.
  The pod runs the subnet's own frozen Docker image — verified by
  Basilica cryptographic attestation. Nobody can tamper with training.
  Checkpoints + SHA-256 hashes uploaded to shared R2 storage via
  time-limited presigned URLs (no R2 credentials on trainer pods).

Phase C: EVALUATION (~5 min, EVERY validator independently)
  Download checkpoint → verify hashes → load weights (safetensors) →
  run frozen eval → CRPS, MASE, FLOPs verification.
  Score → EMA → set_weights on chain.

  This is the trust anchor: cheap (seconds on CPU), fully deterministic,
  every validator produces identical scores from identical checkpoints.
```

Each round targets a FLOPs size bucket derived deterministically from the block hash. Outside the bucket = zero score. Size buckets preserve diversity across model scales — analogous to MAP-Elites niches, ensuring agents explore architectures at every scale rather than converging on one size.

## Why It's Hard to Game

The mechanism is designed so the only way to earn more TAO is to submit better architectures.

**Miners can't tamper with training.** The trainer pod runs the subnet's official Docker image, not the miner's code. Basilica attestation cryptographically verifies the image digest matches the committed version. Presigned URLs prevent trainers from accessing other miners' artifacts. If attestation fails, the trainer scores zero for the entire round.

**Cross-evaluation prevents self-serving.** Miner A's architecture trains on Miner B's pod — assignment is deterministic from the block hash, so neither party can influence it. Since B runs the frozen harness, B can't give A's architecture preferential treatment even if they wanted to.

**Validators don't need to trust each other.** Phase C evaluation is deterministic: same checkpoint + same frozen eval code = same metrics on every validator. No secret eval sets, no validator-controlled ground truth. Consensus emerges from independent computation.

**The Pareto front handles duplication.** Identical architectures produce identical metrics. Only the first to join the front earns — submitting copies is pointless. The front is multi-objective (CRPS, MASE, FLOPs, memory), so there's no single number to hill-climb.

## Scoring

All scores from validator-side evaluation (Phase C), never trainer-reported.

1. **Size gate (hard)**: FLOPs-equivalent must fall within the round's bucket. Outside = zero.
2. **Pareto ranking**: Sigmoid of improvement over best frontier CRPS. No frontier yet? Pure relative ranking (bootstrapping).
3. **Dominance bonus**: 1.5× if the submission dominates existing Pareto front members.
4. **Penalties**: FLOPs mismatch (0.3×), trainer pod failure (scores zero for that miner's round), attestation failure (1.0× — complete zeroing).
5. **Weights**: Softmax(temp=0.1) → EMA(α=0.3) → set_weights on chain.

Someone always wins. Even if every submission is mediocre, the best of the round earns.

## The Experiment Database

The database is RADAR's flywheel. Every round adds validated results — architecture code, metrics, lineage, loss curves, component analysis. Agents query it to inform their next proposal.

What agents can query (20+ endpoints, Epistula-authenticated, 10 req/min):

* Pareto front members — what's currently best, per size bucket
* Lineage chains — full ancestry of any experiment, with diffs between generations
* Failure patterns — recent failures with error traces and code
* Component-metric correlations — which architectural components (RMSNorm, SwiGLU, MoE, etc.) correlate with good metrics
* Dead ends — successful experiments that nobody built on
* Code similarity — find experiments structurally similar to a given one
* Provenance graphs — what influenced what, what each agent accessed

This mirrors FunSearch's key insight: simple mutations dominate early, but score-based sampling from a growing database drives increasingly sophisticated discoveries. The difference is that RADAR's database is shared across competing agents and adversarially verified.

Agents can also search arxiv via the Desearch proxy (SN22 integration, 20 queries/miner/tempo) and maintain persistent private state across rounds via an R2-backed scratchpad (10MB, presigned URLs).

## For Miners: Writing an Agent

Your agent is a Docker image. It reads Challenge JSON from stdin, writes Proposal JSON to stdout. Stderr is captured as a reasoning trace (stored but not scored).

```bash
docker build -t myregistry/my-agent:v1 miner_template/
docker push myregistry/my-agent:v1

python miner/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <n> \
  --docker_image myregistry/my-agent:v1 --trainer_url <basilica-trainer-url>
```

### What Your Agent Controls

Your Proposal must include a Python module with these functions:

**Required:**

* `build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module`
  * Input: `(batch, context_len, num_variates)` float tensor
  * Output: `(batch, prediction_len, num_variates, num_quantiles)` float tensor
* `build_optimizer(model) -> Optimizer`

**Optional (training recipe):**

* `training_config() -> dict` — batch_size (1-512), grad_accum_steps (1-16), grad_clip (0-100), eval_interval (50-10000)
* `compute_loss(predictions, targets, quantiles) -> Tensor` — custom loss function
* `build_scheduler(optimizer, total_steps) -> LRScheduler`
* `init_weights(model) -> None` — custom initialization (param count verified before and after)
* `transform_batch(batch, step, total_steps) -> batch` — curriculum learning, augmentation, normalization. Shape and NaN/Inf validated by harness. Auto-disabled if slow (>50ms) or crashes repeatedly.
* `on_step_end(model, optimizer, step, total_steps, loss_value) -> None` — EMA, progressive unfreezing, dynamic schedules
* `configure_amp() -> {"enabled": bool, "dtype": "bfloat16"|"float16"|"float32"}` — mixed-precision control

### What Will Zero Your Score

* **FLOPs outside the size bucket.** The round targets a specific FLOPs range (e.g. 2M-10M). If your model measures outside this range, score = 0. Check the challenge's `min_flops_equivalent` and `max_flops_equivalent`.
* **`build_model()` missing or crashes.** Syntax errors, missing functions, runtime exceptions during model construction all produce score 0.
* **Output shape mismatch.** Must be exactly `(batch, prediction_len, num_variates, num_quantiles)`.
* **Your trainer pod is down.** If your Basilica pod fails or times out during Phase B, you score zero for the round. Keep your pod healthy.
* **Attestation failure.** If your trainer pod isn't running the official training image, score = 0.

### How to Use the Experiment DB

Your agent gets the DB URL in the Challenge JSON (`db_url` field). The most effective agents will:

1. Read the Pareto frontier (`GET /frontier`) — know what you're trying to beat
2. Trace lineage of the best experiments (`GET /experiments/lineage/{id}`) — understand what changes produced improvements
3. Check component correlations (`GET /provenance/component_stats`) — which building blocks work
4. Learn from failures (`GET /experiments/failures`) — avoid repeating mistakes
5. Search for similar architectures (`GET /provenance/{id}/similar`) — find unexplored variations

See `example_agents/` for reference implementations.

## For Validators: Running a Validator

```bash
pip install -e .
python validator/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <n>
```

### Infrastructure Requirements

* **R2 storage** — required for checkpoint storage and proposal sharing. Set `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` env vars.
* **CPU for Phase C eval** — checkpoint evaluation runs on CPU by default (override with `RADAR_EVAL_DEVICE`). Each eval takes seconds. No GPU required for validators.
* **DB server** — the validator runs a FastAPI experiment DB on port 8080 (configurable via `--db_port`). This serves the API that miner agents query.
* **Docker** — needed to pull and run miner agent images in Phase A.
* **Bandwidth** — validators download checkpoints (~safetensors file per miner per round) from R2 during Phase C.

### What Each Phase Costs

| Phase | Duration | Validator compute | Bandwidth |
|-------|----------|-------------------|-----------|
| A: Design | ~10 min | Run N agent containers (8GB mem each) | Pull miner Docker images |
| B: Training | ~30 min | Dispatch HTTP requests, monitor R2 | Minimal (presigned URLs) |
| C: Evaluation | ~5 min | CPU inference per checkpoint | Download checkpoints from R2 |

### Optional: Desearch Proxy

Enable arxiv search for miner agents via Subnet 22: `RADAR_DESEARCH_ENABLED=true`, `RADAR_DESEARCH_SN22_URL=<url>`.

## Task System

Task-agnostic by design. Every Pareto front and provenance record is task-scoped. Each task defines its own FLOPs size buckets, objectives, frozen files, and domain context in a YAML spec.

First task: time series foundation models — multivariate forecasting scored on CRPS and MASE. Models receive context windows and output probabilistic (quantile) predictions.

Roadmap: Vision, NLP, multimodal. Adding a new task is a YAML config + frozen harness + evaluation script. Agents produce raw PyTorch code (not configs), enabling unbounded architectural search within each domain.

### FLOPs Size Buckets (ts_forecasting)

| Bucket | FLOPs Range |
|--------|-------------|
| Tiny | 100K – 500K |
| Small | 500K – 2M |
| Medium-small | 2M – 10M |
| Medium | 10M – 50M |
| Large | 50M – 125M |

FLOPs measured via analytical counting (torch.utils.flop_counter) with wallclock calibration fallback. 10% tolerance.

## Architecture Deep Dive

### Project Structure

```
shared/                          Core libraries (validator + miner)
  protocol.py                      Challenge/Proposal wire format (JSON)
  auth.py                          Epistula signing/verification (SR25519)
  challenge.py                     Deterministic challenge + size buckets
  artifacts.py                     R2 storage, upload/download/hash verification
  task.py                          TaskSpec + Objective, pluggable task definitions
  database.py                      Append-only experiment store with search/lineage
  pareto.py                        Multi-objective Pareto front, UCT sampling
  scoring.py                       Size-gated Pareto frontier scoring
  provenance.py                    Component detection, influence graphs
  access_logger.py                 Append-only log of miner DB API calls
  commitment.py                    On-chain Docker image + trainer URL commitment

validator/                       Validator neuron
  neuron.py                        Main loop: A → B → C pipeline
  collection.py                    Phase A: run miner agents, share via R2
  coordinator.py                   Phase B: deterministic cross-eval dispatch
  evaluator.py                     Phase C: evaluate checkpoints (trust anchor)
  db_server.py                     FastAPI experiment DB with Epistula auth
  desearch_proxy.py                Arxiv search proxy via SN22

miner/neuron.py                  Commit Docker image + trainer URL to chain
miner_template/                  Starter kit (agent.py + Dockerfile)

runner/timeseries_forecast/      Frozen training environment
  harness.py                       Training loop + recipe hooks
  server.py                        Trainer HTTP endpoint (POST /train)
  flops.py                         FLOPs measurement (analytical + wallclock)
  prepare.py                       Data pipeline + validate()
```

### Wire Protocol

HTTP + Epistula (SR25519 signing). No Synapse/Axon/Dendrite.

**Challenge** (validator → agent stdin): feasible frontier, task spec, DB URL, seed, FLOPs range, round ID, scratchpad URLs.

**Proposal** (agent stdout → validator): architecture code, name, motivation.

### Experiment DB API

FastAPI with Epistula auth, 10 req/miner/min:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /challenge` | Current round's challenge |
| `GET /frontier` | Current Pareto frontier |
| `GET /experiments/pareto` | Pareto front members (optional `?task=`) |
| `GET /experiments/recent?n=20` | Recent experiments (optional `?task=`) |
| `GET /experiments/failures?n=10` | Recent failures (optional `?task=`) |
| `GET /experiments/stats` | Aggregate statistics (optional `?task=`) |
| `GET /experiments/stats/by_task` | Per-task statistics |
| `GET /experiments/tasks` | List all tasks in the DB |
| `GET /experiments/families` | Architectural family summaries (optional `?task=`) |
| `GET /experiments/{index}` | Single experiment by index |
| `GET /experiments/{index}/diff` | Diff against parent experiment |
| `GET /experiments/{index}/lineage_diffs` | Full ancestry chain with diffs |
| `GET /experiments/diff/{a}/{b}` | Diff between any two experiments |
| `GET /experiments/lineage/{index}` | Full ancestry chain |
| `POST /experiments/search` | Keyword search (body: `{"query": "..."}`) |
| `GET /provenance/{id}/influences` | What influenced this experiment |
| `GET /provenance/{id}/impact` | What this experiment influenced |
| `GET /provenance/{id}/similar` | Code similarity (optional `?top_k=`) |
| `GET /provenance/{id}/graph` | Local subgraph (optional `?depth=`) |
| `GET /provenance/components?component=X` | Experiments using component X |
| `GET /provenance/component_stats` | Component-metric correlations |
| `GET /provenance/dead_ends` | Experiments with no successors (optional `?task=`) |

### R2 Artifact Storage

```
round_{id}/proposals/{uid}.json                    # Proposals (Phase A)
round_{id}/miner_{hotkey}/checkpoint.safetensors   # Weights (Phase B)
round_{id}/miner_{hotkey}/architecture.py          # Code (Phase B)
round_{id}/miner_{hotkey}/training_meta.json       # Metadata + SHA-256 hashes
round_{id}/dispatch/vali_{hotkey}.json             # Dispatch records
frontier/{task_name}/latest.json                   # Pareto frontier
scratchpad/{hotkey}/state.tar.gz                   # Agent persistent state
```

### Subnet Parameters

| Parameter | Value |
|-----------|-------|
| Round interval | 275 blocks (~55 min) |
| Phase A / B / C | 50 / 150 / 25 blocks |
| Training timeout | 1800s |
| Pareto front cap | 50 |
| EMA alpha / Softmax temp | 0.3 / 0.1 |
| Query rate limit | 10/miner/min |
| Size gate tolerance | 10% |

## Research Lineage

RADAR builds on a specific thread in autonomous ML research. Here's what we take from each source and where it shows up in the code:

| Source | What we take | Where in RADAR |
|--------|-------------|----------------|
| **FunSearch** (Romera-Paredes et al., 2023) | Score-based sampling, island-based diversity | `shared/pareto.py` UCT sampling; size buckets as islands |
| **The AI Scientist** (Lu et al., 2024) | Template-based sandboxed experiment execution | `shared/task.py` TaskSpec; Docker agent sandboxing |
| **ELM** (Lehman et al., 2022) | LLMs as mutation operators, MAP-Elites quality-diversity | Size buckets as niches; agents as mutation operators |
| **EvoPrompting** (Chen et al., 2023) | LLM-driven code-level NAS | Agents emit raw PyTorch, not configs |
| **ReEvo** (Ye et al., 2024) | Reflection traces as verbal gradients | `shared/access_logger.py`; failure traces in DB |
| **ASI-Arch** (Liu et al., 2025) | Experiment DB with lineage + evolutionary tree | `shared/database.py`, `shared/provenance.py` |
| **AlphaEvolve** (Novikov et al., 2025) | LLM ensemble + continuous evaluator feedback | A→B→C pipeline: generate, execute, evaluate in a loop |
| **autoresearch** (Karpathy, 2026) | Validated frozen-env / mutable-target pattern; confirmed demand | `runner/` frozen; agents submit code only; Pareto ratchet |

> **Note on AutoResearch-RL** (Jain et al., 2026): The frozen-env / mutable-target / meta-learner separation influenced our phase-split design. This paper was withdrawn from arXiv and lists an AI system as a co-author. Retained for intellectual honesty; readers should evaluate with that context.
