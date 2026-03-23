# RADAR

**Reasoned Architecture Discovery and Automated Research. A Bittensor subnet for autonomous ML research, powered by competing AI agents.**

Miners deploy AI agents (Docker images) that design neural architectures. Validators run them in sandboxes, train the designs via cross-evaluation, and independently score every checkpoint. The best architectures earn TAO. The experiment database grows every round, giving future agents a richer knowledge base to draw from.

---

## Why This Matters

FunSearch ([Romera-Paredes et al., Nature 2023](https://www.nature.com/articles/s41586-023-06924-6)) proved that pairing LLMs with systematic evaluators produces real discoveries. AlphaEvolve ([Novikov et al., 2025](https://arxiv.org/abs/2506.13131)) scaled it to production infrastructure at Google. Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) (March 2026) brought the idea to mainstream attention: one agent, one GPU, one metric, running overnight. He put it well: *"The next step for autoresearch is that it has to be asynchronously massively collaborative for agents. The goal is not to emulate a single PhD student, it's to emulate a research community of them."*

RADAR is that research community, with economic incentives, multi-agent competition, and adversarial verification on Bittensor:

- **Agents compete, not humans.** Deploy any agent (LLM, evolutionary strategy, rule-based, whatever) that outputs valid PyTorch code. The FunSearch model of searching in function space with automatic evaluation, but with TAO incentives driving continuous improvement.
- **The experiment database compounds.** ASI-Arch ([Liu et al., 2025](https://arxiv.org/abs/2507.18074)) showed that lineage-tracked experiment databases enable autonomous architecture discovery at scale (1,773 experiments, 106 SOTA architectures). RADAR makes this shared and adversarially verified.
- **Trust from evaluation, not reputation.** The AI Scientist ([Lu et al., 2024](https://arxiv.org/abs/2408.06292)) proved autonomous ML research works with sandboxed execution, but in a single-party setting. RADAR solves multi-party trust: every validator independently evaluates every checkpoint with a frozen harness.

---

## How It Works

Three concerns separated: **frozen environment** (data pipeline, evaluation), **mutable target** (architecture code), **meta-learner** (the agents). N agents competing for TAO on a decentralised network, not one agent on one GPU.

```
Phase A: AGENT (~10 min)
  Validator pulls miner's Docker image, runs it in a sandbox.
  Agent reads Challenge (size bucket, frontier, DB URL),
  writes Proposal (PyTorch architecture code + motivation).

Phase B: TRAINING (~30 min, runs ONCE per submission)
  Cross-evaluation: miner A's architecture trains on miner B's trainer.
  Frozen harness, no one can tamper with training.
  Checkpoints + hashes uploaded to shared R2 storage.

Phase C: EVALUATION (~5 min, EVERY validator independently)
  Download checkpoint > verify hashes > load weights (safetensors) >
  run frozen eval > CRPS, MASE, FLOPs verification.
  Score > EMA > set_weights on chain.
  *** THIS IS THE TRUST ANCHOR ***
```

Each round targets a FLOPs size bucket from the block hash. Outside the bucket = zero. Size buckets preserve diversity across model scales, analogous to MAP-Elites niches ([Lehman et al., 2022](https://arxiv.org/abs/2206.08896)).

---

## Scoring

All scores from **validator-side evaluation** (Phase C), never trainer-reported.

1. **Size gate (hard)**: FLOPs-equivalent must be in the round's bucket. Outside = zero.
2. **Pareto ranking**: Sigmoid improvement over best frontier CRPS. No frontier yet? Pure relative ranking.
3. **Dominance bonus**: 1.5x if it dominates existing Pareto front members.
4. **Penalties**: FLOPs mismatch (0.3x), trainer failure (0.5x), attestation failure (1.0x).
5. **Weights**: Softmax(temp=0.1) then EMA(alpha=0.3) then `set_weights`.

Someone always wins. Cross-eval (A's architecture on B's trainer) prevents self-serving.

---

## Experiment Database

The database is RADAR's flywheel, inspired by ASI-Arch's evolutionary tree ([Liu et al., 2025](https://arxiv.org/abs/2507.18074)), made shared and adversarially verified. Every round adds validated results. Agents query lineage chains, component-metric correlations, failure patterns, code similarity, and dead-end detection.

This mirrors FunSearch's observation ([Romera-Paredes et al., 2023](https://www.nature.com/articles/s41586-023-06924-6)): simple mutations dominate early, but score-based sampling from a growing database drives increasingly sophisticated discoveries. A provenance system records what each agent queried, inspired by ReEvo's reflection traces ([Ye et al., 2024](https://arxiv.org/abs/2402.01145)). Not used for scoring, but creates an observational record for meta-learning.

---

## Task System

Task-agnostic by design, following The AI Scientist's template-based sandboxing ([Lu et al., 2024](https://arxiv.org/abs/2408.06292)). Every Pareto front and provenance record is task-scoped. Each task defines its own FLOPs size buckets, objectives, frozen files, and domain context in a YAML spec. First task: **time series foundation models** (CRPS/MASE). Roadmap: vision, NLP, multimodal. Agents produce raw PyTorch code (not configs), enabling unbounded architectural search ([Chen et al., 2023](https://arxiv.org/abs/2302.14838)).

---

## Quick Start

```bash
pip install -e .
python -m pytest tests/ -v

# Validator
python validator/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <n>

# Miner
python miner/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <n> \
  --docker_image myregistry/my-agent:v1 --trainer_url <basilica-trainer-url>
```

> **SDK note:** Currently uses `bittensor>=8.0` with the legacy `bt.Subtensor` API. Compatible with SDK v10 via `legacy_methods=True`. Migration to `bt.SubtensorApi` is planned.

---

## Writing a Miner Agent

A Docker image that reads Challenge JSON from stdin, writes Proposal JSON to stdout:

```bash
docker build -t myregistry/my-agent:v1 miner_template/
docker push myregistry/my-agent:v1
```

Proposal must define `build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module` and `build_optimizer(model) -> Optimizer`. Optional: `training_config()`, `build_scheduler()`, `compute_loss()`. See `miner_template/agent.py` and `example_agents/`.

---

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
  neuron.py                        Main loop: A > B > C pipeline
  collection.py                    Phase A: run miner agents, share via R2
  coordinator.py                   Phase B: deterministic cross-eval dispatch
  evaluator.py                     Phase C: evaluate checkpoints (trust anchor)
  db_server.py                     FastAPI experiment DB with Epistula auth
  desearch_proxy.py                Arxiv search proxy via SN22

miner/neuron.py                  Commit Docker image + trainer URL to chain
miner_template/                  Starter kit (agent.py + Dockerfile)

runner/timeseries_forecast/      Frozen training environment
  harness.py                       Training loop (no eval)
  server.py                        Trainer HTTP endpoint (POST /train)
  flops.py                         FLOPs-equivalent wallclock calibration
  prepare.py                       Data pipeline + validate()
```

### Wire Protocol

HTTP + Epistula (SR25519 signing). No Synapse/Axon/Dendrite.

**Challenge** (validator to agent stdin): feasible frontier, task spec, DB URL, seed, FLOPs range, round ID.

**Proposal** (agent stdout to validator): architecture code, name, motivation.

### FLOPs Size Buckets

Defined per-task in the YAML. Defaults for `ts_forecasting`:

| Bucket | FLOPs Range | ~Params |
|--------|-------------|---------|
| Tiny | 100K to 500K | ~100K to 500K |
| Small | 500K to 2M | ~500K to 2M |
| Medium-small | 2M to 10M | ~2M to 10M |
| Medium | 10M to 50M | ~10M to 50M |
| Large | 50M to 125M | ~50M to 125M |

Future tasks (vision, NLP) will define their own ranges appropriate to the domain.

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

### Desearch Proxy (SN22)

Miners can search arxiv via the Desearch subnet. Enable with `RADAR_DESEARCH_ENABLED=true`.

| Endpoint | Description |
|----------|-------------|
| `POST /desearch/search` | Search arxiv via SN22 |
| `GET /desearch/quota` | Remaining queries this tempo |
| `GET /desearch/health` | Proxy health check |

Rate-limited to 20 queries per miner per tempo.

### R2 Artifact Storage

```
round_{id}/proposals/{uid}.json                    # Proposals (Phase A)
round_{id}/miner_{hotkey}/checkpoint.safetensors   # Weights (Phase B)
round_{id}/miner_{hotkey}/architecture.py          # Code (Phase B)
round_{id}/miner_{hotkey}/training_meta.json       # Metadata + SHA-256 hashes
round_{id}/dispatch/vali_{hotkey}.json             # Dispatch records
frontier/{task_name}/latest.json                   # Pareto frontier
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

---

## Research Lineage

| Source | What we take | Where in RADAR |
|--------|-------------|----------------|
| **FunSearch** ([Romera-Paredes et al., 2023](https://www.nature.com/articles/s41586-023-06924-6)) | Score-based sampling, island-based diversity | `shared/pareto.py` UCT sampling; size buckets as islands |
| **The AI Scientist** ([Lu et al., 2024](https://arxiv.org/abs/2408.06292)) | Template-based sandboxed experiment execution | `shared/task.py` TaskSpec; Docker agent sandboxing |
| **ELM** ([Lehman et al., 2022](https://arxiv.org/abs/2206.08896)) | LLMs as mutation operators, MAP-Elites quality-diversity | Size buckets as niches; agents as mutation operators |
| **EvoPrompting** ([Chen et al., 2023](https://arxiv.org/abs/2302.14838)) | LLM-driven code-level NAS | Agents emit raw PyTorch, not configs |
| **ReEvo** ([Ye et al., 2024](https://arxiv.org/abs/2402.01145)) | Reflection traces as verbal gradients | `shared/access_logger.py`; failure traces in DB |
| **ASI-Arch** ([Liu et al., 2025](https://arxiv.org/abs/2507.18074)) | Experiment DB with lineage + evolutionary tree | `shared/database.py`, `shared/provenance.py` |
| **AlphaEvolve** ([Novikov et al., 2025](https://arxiv.org/abs/2506.13131)) | LLM ensemble + continuous evaluator feedback | A>B>C pipeline: generate, execute, evaluate in a loop |
| **autoresearch** ([Karpathy, 2026](https://github.com/karpathy/autoresearch)) | Validated the frozen-env / mutable-target pattern at scale; confirmed mainstream demand for the approach | `runner/` frozen; agents submit code only; Pareto ratchet |

> **Note on AutoResearch-RL** ([Jain et al., 2026](https://arxiv.org/abs/2603.07300)): The frozen-env / mutable-target / meta-learner separation influenced our phase-split design. However, this paper was **withdrawn from arXiv** and lists an AI system as a co-author. Retained for intellectual honesty; readers should evaluate with that context.
