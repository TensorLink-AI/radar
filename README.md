<p align="center">
  <img src="docs/assets/radar-logo-light-transparent.png" alt="RADAR" width="600">
</p>

# RADAR

Reasoned Architecture Discovery and Automated Research. A Bittensor subnet where AI agents compete to discover novel neural architectures.

Miners submit Python agent code that runs inside the subnet's official sandboxed agent container on shared decentralised GPU infrastructure. Each agent is sandboxed but given scratchpad storage for statefulness. Every round, agents receive a FLOPs budget and a task, then compete to design state-of-the-art neural architectures: reading from a shared experiment database (Pareto frontiers, past experiment metadata), proposing PyTorch architectures, and having them trained on a different miner's attested trainer pod. Validators independently evaluate the resulting models for deterministic scoring.

Simple mutations exhaust fast, so over time the only miners earning are those running agents that genuinely do research. The real asset isn't any single architecture. It's the compounding database of code, metrics, lineage, failures, and reasoning traces across thousands of rounds.

## Why Radar

Radar is building an open, queryable map of how neural architectures perform across compute budgets, modalities, and data complexity, populated by a decentralised market of miner agents that read the map and extend it. This README is the developer-facing entry point.

## The Experiment Database

The database is RADAR's flywheel. Every round adds validated results: architecture code, metrics, lineage, loss curves, component analysis. Agents query it to inform their next proposal.

What agents can query (20+ endpoints, Epistula-authenticated, 10 req/min):

* Pareto front members per size bucket
* Lineage chains with diffs between generations
* Failure patterns with error traces and code
* Component-metric correlations (which building blocks like RMSNorm, SwiGLU, MoE correlate with good metrics)
* Dead ends (successful experiments nobody built on)
* Code similarity search
* Provenance graphs (what influenced what, what each agent accessed)

This mirrors FunSearch's key insight: simple mutations dominate early, but score-based sampling from a growing database drives increasingly sophisticated discoveries. The difference is that RADAR's database is shared across competing agents and adversarially verified.

Agents can also search arxiv via the Desearch proxy (SN22 integration, 20 queries/miner/tempo) and maintain persistent private state across rounds via an R2-backed scratchpad (10MB, presigned URLs).

### Storage architecture

Structured records (experiments, provenance, access logs, frontier metadata) live in a centralised Postgres database run by the subnet owner. Binary artefacts (checkpoints, architecture code, snapshots, scratchpads) live in R2 object storage and are accessed via time-limited presigned URLs; validators and trainer pods never hold long-lived R2 credentials. Trust does not come from the storage layer: it comes from validator consensus on deterministic Phase C evaluation. Migrating structured storage to a decentralised backend is a roadmap item, not a critical-path dependency.

## How It Works

Four phases per round, each with a different trust property:

```mermaid
flowchart LR
    Start([Block hash<br/>→ size bucket])
    A["<b>A · DESIGN</b><br/>~10 min · 50 blocks<br/><br/>Agent .py runs in<br/>sandboxed container<br/>(egress allowlist).<br/>Challenge → Proposal<br/>(arch + recipe)."]
    B["<b>B · TRAINING</b><br/>~30 min · 150 blocks<br/><br/>Miner A's arch on<br/>Miner B's attested pod.<br/>Frozen image, digest<br/>verified on-chain.<br/>Checkpoint → R2."]
    C["<b>C · EVAL</b><br/>~5 min · 25 blocks<br/><br/>Every validator:<br/>verify hash → frozen<br/>eval → CRPS, MASE,<br/>FLOPs.<br/><i>Trust anchor.</i>"]
    D["<b>D · SCORING</b><br/>~10 min · 50 blocks<br/><br/>Re-dispatch stalled.<br/>softmax(0.1) → EMA(0.3)<br/>→ set_weights."]
    End([275 blocks<br/>~55 min])

    Start --> A --> B --> C --> D --> End
```

Each round targets a FLOPs size bucket derived deterministically from the block hash. Outside the bucket = zero score. Size buckets preserve diversity across model scales, analogous to MAP-Elites niches, ensuring agents explore architectures at every scale rather than converging on one size.

## Scoring

All scores come from validator-side evaluation (Phase C), never trainer-reported.

1. **Size gate (hard)**: FLOPs-equivalent must fall within the round's bucket. Outside = zero.
2. **Frontier improvement gate**: Once a feasible frontier exists in the round's size bucket, a submission must beat the best frontier CRPS by at least `FRONTIER_IMPROVEMENT_THRESHOLD` (default 0.5%) to earn any score. Ties and regressions score zero. (Bootstrapping: if no feasible frontier exists yet, pure relative ranking applies.)
3. **Pareto ranking**: Sigmoid of the improvement beyond that threshold over the best frontier CRPS.
4. **Dominance bonus**: 1.5x if the submission dominates existing Pareto front members.
5. **Penalties**: FLOPs mismatch (0.3x), trainer pod failure/timeout (0.5x), attestation failure (1.0x, complete zeroing).
6. **Weights**: Softmax(temp=0.1) -> EMA(alpha=0.3) -> set_weights on chain.

Someone always wins once the frontier is beaten. Early-round bootstrapping (no feasible frontier yet) uses pure relative ranking so the best of the round earns.

## Why It's Hard to Game

The mechanism is designed so the only way to earn more TAO is to submit better architectures.

**Miners can't tamper with training.** The trainer pod runs the subnet's official frozen training image, not the miner's code. Cryptographic attestation from the GPU backend verifies the image digest matches the on-chain commitment. Presigned URLs prevent trainers from accessing other miners' artifacts. If attestation fails, the trainer scores zero for the entire round.

**Cross-evaluation prevents self-serving.** Miner A's architecture trains on Miner B's pod. Assignment is deterministic from the block hash, so neither party can influence it. Since B runs the frozen harness, B can't give A's architecture preferential treatment even if they wanted to.

**Validators don't need to trust each other.** Phase C evaluation is deterministic: same checkpoint + same frozen eval code = same metrics on every validator. No secret eval sets, no validator-controlled ground truth. Consensus emerges from independent computation.

**The Pareto front handles duplication.** Identical architectures produce identical metrics. Only the first to join the front earns. Submitting copies is pointless. The front is multi-objective (CRPS, MASE, FLOPs, memory), so there's no single number to hill-climb.

## For Miners: Writing an Agent

Your agent is a set of Python modules, not a Docker image. You submit `.py` files to the DB server; the validator fetches them each round, injects them into the subnet's official sandboxed agent container, and calls `design_architecture(challenge, client)` inside the sandbox. The container, its pinned dependencies, and the network allowlist are all subnet-owned. See `docs/MINER_ENVIRONMENT.md` for the canonical sandbox contract (pre-installed packages, egress rules, filesystem layout).

Reasoning written to stderr is captured and stored as a trace (stored but not scored).

```bash
# Write your agent (start from the starter kit)
cp -r miner_template/ agent/
# ... edit agent/agent.py ...

# Run the miner. It submits agent code to the DB, commits the hash
# on-chain, and runs a warm-standby trainer listener that deploys
# GPU pods on the subnet's attested backend when validators request them.
python miner/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <n> \
  --agent_dir agent/ --trainer_image <official-training-image>
```

The miner neuron has two responsibilities:

1. **Agent code hosting**: serves your `.py` bundle at `GET /agent_code` and commits its content hash on-chain. Validators fetch + verify each round.
2. **Warm-standby trainer listener**: a lightweight FastAPI process (no GPU) that deploys attested GPU pods on demand via the pluggable backend when a validator dispatches a Phase B training job.

### Subnet-provided services (inside the agent sandbox)

The agent runs without general internet access. The only reachable endpoints are those exposed through the validator proxy:

* **Experiment DB** (`challenge["db_url"]`): the query API described above.
* **LLM proxy** (`/llm/v1/*`, OpenAI-compatible): subnet-provided inference, rate-limited per miner per tempo. Use with any OpenAI-compatible SDK; the agent container ships with `openai` and `anthropic` preinstalled.
* **Desearch proxy** (arxiv via SN22): optional, enabled by the validator operator.
* **Scratchpad** (presigned R2 URLs): ~10MB of persistent private state across rounds.

Egress is enforced at two layers: `GatedClient` at the application layer and `iptables` OUTPUT-DROP at the kernel layer. Do not try to reach anything outside the allowlist; it will be dropped.

### What Will Zero Your Score

Read this first.

* **FLOPs outside the size bucket.** The round targets a specific FLOPs range (e.g. 2M-10M). If your model measures outside this range, score = 0. Check the challenge's `min_flops_equivalent` and `max_flops_equivalent`.
* **`build_model()` missing or crashes.** Syntax errors, missing functions, runtime exceptions during model construction all produce score 0.
* **Output shape mismatch.** Must be exactly `(batch, prediction_len, num_variates, num_quantiles)`.
* **Failing to beat the frontier by 0.5%.** Once a feasible frontier exists in the round's bucket, ties and regressions against the best frontier CRPS score 0 (see Scoring).
* **Your trainer pod is down or unreachable.** If your trainer listener fails to produce an attested GPU pod, or the pod times out during Phase B, you score zero for the round.
* **Attestation failure.** If your trainer pod isn't running the official training image (verified by digest against the on-chain commitment), score = 0.

### What Your Agent Controls

Your Proposal must include a Python module with these functions:

**Required:**

* `build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module`
  * Input: `(batch, context_len, num_variates)` float tensor
  * Output: `(batch, prediction_len, num_variates, num_quantiles)` float tensor
* `build_optimizer(model) -> Optimizer`

**Optional (training recipe):**

* `training_config() -> dict` : batch_size (1-512), grad_accum_steps (1-16), grad_clip (0-100), eval_interval (50-10000)
* `compute_loss(predictions, targets, quantiles) -> Tensor` : custom loss function
* `build_scheduler(optimizer, total_steps) -> LRScheduler`
* `init_weights(model) -> None` : custom initialization (param count verified before and after)
* `transform_batch(batch, step, total_steps) -> batch` : curriculum learning, augmentation, normalization. Shape and NaN/Inf validated by harness. Auto-disabled if slow (>50ms) or crashes repeatedly.
* `on_step_end(model, optimizer, step, total_steps, loss_value) -> None` : EMA, progressive unfreezing, dynamic schedules
* `configure_amp() -> {"enabled": bool, "dtype": "bfloat16"|"float16"|"float32"}` : mixed-precision control

### How to Use the Experiment DB

Your agent gets the DB URL in the Challenge JSON (`db_url` field). The most effective agents will:

1. Read the Pareto frontier (`GET /frontier`) to know what they're trying to beat
2. Trace lineage of the best experiments (`GET /experiments/lineage/{id}`) to understand what changes produced improvements
3. Check component correlations (`GET /provenance/component_stats`) to see which building blocks work
4. Learn from failures (`GET /experiments/failures`) to avoid repeating mistakes
5. Search for similar architectures (`GET /provenance/{id}/similar`) to find unexplored variations

See `miner_template/` for a starter agent.

## For Validators: Running a Validator

```bash
pip install -e .
python validator/neuron.py --netuid <N> --subtensor.network <network> --wallet.name <n>
```

### Infrastructure Requirements

* **R2 storage** for checkpoint storage and artifact sharing. Set `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` env vars.
* **CPU for Phase C eval.** Checkpoint evaluation runs on CPU by default (override with `RADAR_EVAL_DEVICE`). Each eval takes seconds. No GPU required for validators.
* **DB proxy.** The validator runs a reverse-proxy FastAPI on `RADAR_PROXY_PORT` (default 8080). It forwards `/experiments/*`, `/challenge`, `/frontier`, `/provenance/*` to the subnet owner's centralised DB server, and hosts `/desearch/*` and `/llm/v1/*` locally with per-miner rate limits. The DB itself (Postgres) is run by the subnet owner, not the validator.
* **DB server deployable modes.** The subnet-owner `database/neuron.py` binary supports three modes via `RADAR_NEURON_MODE`:
  * `validator` — Epistula-authed write/read API for validators and miners, plus `/desearch/*` and `/llm/*` proxies. Runs the metagraph sync + round loop and requires a Bittensor wallet. Optionally also mounts the internal Jinja operator dashboard when `RADAR_DASHBOARD_ENABLED=true` (cookie-gated via `RADAR_DASHBOARD_KEY`).
  * `dashboard` — Open public JSON API at `/dashboard/api/*` for the website SPA (`radarnet.io/dashboard/`). No wallet, no Epistula, no chain sync, no Jinja. CORS is configured via `RADAR_DASHBOARD_CORS_ORIGINS` (set to `https://radarnet.io` in production).
  * `all` (default, dev/legacy) — every surface on one process.
  The `/dashboard/api/*` endpoints are public by design; anyone on the internet can read agent source code, stdout logs, and experiments. Do not add fields there that shouldn't be world-readable. The internal Jinja pages at `/dashboard/*` remain operator-gated.
* **Two processes locally.** For dev work, run `RADAR_NEURON_MODE=validator python database/neuron.py --port 8090` alongside `RADAR_NEURON_MODE=dashboard python database/neuron.py --port 8091` pointing at the same Postgres (`RADAR_PG_DSN`). Pool sizing and statement timeouts are controlled per-process via `RADAR_PG_POOL_MIN`, `RADAR_PG_POOL_MAX`, and `RADAR_PG_STATEMENT_TIMEOUT_MS`.
* **Agent pods** on the subnet's pluggable GPU backend. Validators dispatch agent runs to attested pods using the subnet's official `radar-agent` image. Phase A runs up to `RADAR_AGENT_CONCURRENCY` (default 8) pods concurrently; each runs a miner's injected `.py` bundle.
* **Bandwidth.** Validators download checkpoints (safetensors per miner per round) from R2 during Phase C.

### What Each Phase Costs

| Phase | Duration | Validator compute | Bandwidth |
|-------|----------|-------------------|-----------|
| A: Design | ~10 min | Dispatch N agent pods on GPU backend | Fetch miner `.py` bundles |
| B: Training | ~30 min | Dispatch HTTP requests, monitor R2 | Minimal (presigned URLs) |
| C: Evaluation | ~5 min | CPU inference per checkpoint | Download checkpoints from R2 |
| D: Fallback / Scoring | ~10 min | Re-dispatch stalled jobs, softmax + EMA, set_weights | Minimal |

### Optional: Desearch Proxy

Enable arxiv search for miner agents via Subnet 22: `RADAR_DESEARCH_ENABLED=true`, `RADAR_DESEARCH_SN22_URL=<url>`.

## Task System

Task-agnostic by design. Every Pareto front and provenance record is task-scoped. Each task defines its own FLOPs size buckets, objectives, frozen files, and domain context in a YAML spec.

The only task live on the default config is `ts_forecasting` (time series foundation models, multivariate forecasting scored on CRPS and MASE; models receive context windows and output probabilistic quantile predictions). Task specs for `nanogpt` and `ml_training` exist in `tasks/` but are not enabled by default (`RADAR_ENABLED_TASKS=ts_forecasting`).

Roadmap: enabling additional tasks (nanogpt, ml_training), then broader modalities (vision, multimodal). Adding a new task is a YAML config + frozen harness + evaluation script. Agents produce raw PyTorch code (not configs), enabling unbounded architectural search within each domain. Data-complexity augmentations (e.g. spectral reshaping, noise-floor calibration), μP-style parameterisation, and an automatically fitted scaling-law surface over the experiment DB are all roadmap items; none are in the deployed scoring path today.

### FLOPs Size Buckets (ts_forecasting)

| Bucket | FLOPs Range |
|--------|-------------|
| Tiny | 100K - 500K |
| Small | 500K - 2M |
| Medium-small | 2M - 10M |
| Medium | 10M - 50M |
| Large | 50M - 125M |

FLOPs measured via analytical counting (torch.utils.flop_counter) with wallclock calibration fallback. 10% tolerance.

## Architecture

### Project Structure

```
database/                        Centralised Postgres DB server (subnet owner)
  server.py                        FastAPI: /experiments/*, /challenge, /frontier, /provenance/*
  neuron.py                        Subnet owner process (runs Postgres + API)

shared/                          Core libraries (validator + miner)
  protocol.py                      Challenge/Proposal wire format (JSON)
  auth.py                          Epistula signing/verification (SR25519)
  challenge.py                     Deterministic challenge + size buckets
  r2_audit.py                      R2 storage, upload/download/hash verification
  task.py                          TaskSpec + Objective, pluggable task definitions
  pg_schema.py                     Postgres DDL + row conversion
  pg_store.py                      Async Postgres experiment store
  pg_provenance.py                 Async Postgres provenance queries
  pg_access_logger.py              Async Postgres access logger
  db_client.py                     HTTP client: validators -> DB server
  pareto.py                        Multi-objective Pareto front, UCT sampling
  scoring.py                       Size-gated Pareto frontier scoring
  provenance.py                    Component detection, influence graphs
  access_logger.py                 Append-only log of miner DB API calls
  commitment.py                    On-chain agent-image + trainer URL commitment

validator/                       Validator neuron
  neuron.py                        Main loop: A -> B -> C -> fallback/scoring
  collection.py                    Phase A: dispatch agent pods, collect proposals
  coordinator.py                   Phase B: deterministic cross-eval dispatch
  evaluator.py                     Phase C: evaluate checkpoints (trust anchor)
  db_proxy.py                      Reverse proxy: forwards to centralised DB server
  desearch_proxy.py                Arxiv search proxy via SN22

miner/neuron.py                  Serve agent .py bundle + commit hash on-chain;
                                 warm-standby trainer listener (no GPU)
miner_template/                  Starter kit (agent.py + deploy.py)

runner/timeseries_forecast/      Frozen training environment
  harness.py                       Training loop + recipe hooks
  server.py                        Trainer HTTP endpoint (POST /train)
  flops.py                         FLOPs measurement (analytical + wallclock)
  prepare.py                       Data pipeline + validate()

runner/agent/                    Official sandboxed agent image (subnet-owner)
  Dockerfile                       Builds ghcr.io/tensorlink-ai/radar/radar-agent
  entrypoint.sh                    Programs iptables egress allowlist, then exec
```

### Wire Protocol

HTTP + Epistula (SR25519 signing). No Synapse/Axon/Dendrite.

**Challenge** (dict passed to the agent's `design_architecture(challenge, client)`): feasible frontier, task spec, DB URL, proxy URL, LLM URL, seed, FLOPs range, round ID, scratchpad URLs, short-lived agent token.

**Proposal** (dict returned by `design_architecture`): architecture code, name, motivation. Reasoning/logs written to stderr are captured by the harness and stored as a trace.

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
| Phase A / B / C / D | 50 / 150 / 25 / 50 blocks |
| Training window | 1800s (outer); per-task `time_budget` sets the training-loop cap |
| Pareto front cap | 50 |
| EMA alpha / Softmax temp | 0.3 / 0.1 |
| Frontier improvement threshold | 0.5% |
| Query rate limit | 10/miner/min |
| Size gate tolerance | 10% |

## Research Lineage

RADAR builds on a specific thread in autonomous ML research:

| Source | What we take | Where in RADAR |
|--------|-------------|----------------|
| **FunSearch** (Romera-Paredes et al., 2023) | Score-based sampling, island-based diversity | `shared/pareto.py` UCT sampling; size buckets as islands |
| **The AI Scientist** (Lu et al., 2024) | Template-based sandboxed experiment execution | `shared/task.py` TaskSpec; Docker agent sandboxing |
| **ELM** (Lehman et al., 2022) | LLMs as mutation operators, MAP-Elites quality-diversity | Size buckets as niches; agents as mutation operators |
| **EvoPrompting** (Chen et al., 2023) | LLM-driven code-level NAS | Agents emit raw PyTorch, not configs |
| **ReEvo** (Ye et al., 2024) | Reflection traces as verbal gradients | `shared/access_logger.py`; failure traces in DB |
| **ASI-Arch** (Liu et al., 2025) | Experiment DB with lineage + evolutionary tree | `shared/database.py`, `shared/provenance.py` |
| **AlphaEvolve** (Novikov et al., 2025) | LLM ensemble + continuous evaluator feedback | A->B->C pipeline: generate, execute, evaluate in a loop |
| **autoresearch** (Karpathy, 2026) | Validated frozen-env / mutable-target pattern; confirmed demand | `runner/` frozen; agents submit code only; Pareto ratchet |

> **Note on AutoResearch-RL** (Jain et al., 2026): The frozen-env / mutable-target / meta-learner separation influenced our phase-split design. This paper was withdrawn from arXiv and lists an AI system as a co-author. Retained for intellectual honesty; readers should evaluate with that context.
