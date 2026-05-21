# Radar — CLAUDE.md

## What This Is

A platform for autonomous ML research using phase-split validation. Miners host agents that design architectures. Validators coordinate training and independently evaluate checkpoints for consensus.

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
  ├── HMAC auth: validator surface only. JSON API is public by design.
  └── Peer-refresh + round loop runs in {validator, all} modes only.
      Dashboard mode is identity-free.

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
| `shared/auth.py` | HMAC-SHA256 signing/verification keyed by `RADAR_SHARED_SECRET` |
| `shared/peers.py` | Static peer registry loaded from `MINERS_CONFIG_PATH` (`miners.json`) |
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
| `shared/r2_audit.py` | S3-compatible artifact storage. Hippius (Substrate) is the primary backend; Cloudflare R2 stays supported as a legacy fallback. Exports `HippiusStorage` (preferred) and `R2AuditLog` (alias). |
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

# Point everything at a JSON peer registry and a shared HMAC secret
export MINERS_CONFIG_PATH=$PWD/miners.json   # see miners.example.json
export RADAR_SHARED_SECRET="$(openssl rand -hex 32)"

# Start database server — default (all) mode: validator surface + public JSON
python database/neuron.py

# Validator-only mode (no public JSON API; internal Jinja UI optional)
RADAR_NEURON_MODE=validator python database/neuron.py

# Public dashboard-only mode
RADAR_NEURON_MODE=dashboard \
  RADAR_DASHBOARD_CORS_ORIGINS=https://radarnet.io \
  python database/neuron.py --port 8091

# Running both locally on the same Postgres (dev)
# Terminal 1: validator surface on 8090
RADAR_NEURON_MODE=validator python database/neuron.py --port 8090
# Terminal 2: public JSON API on 8091 pointing at the same Postgres
RADAR_NEURON_MODE=dashboard python database/neuron.py --port 8091

# Start validator (peer-refresh loop)
python validator/neuron.py

# Start validator (peer-refresh loop) + proxy
python validator/neuron.py &
uvicorn validator.db_proxy:app --host 0.0.0.0 --port 8080

# Start miner (peer-refresh loop)
python miner/neuron.py

# Start the runner that hosts /train (used by trainer pods)
python -m runner.server

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

1. **Block windows** (validator-global): define WHEN phases start/end (~12s/block). Rigid boundaries. Env: `RADAR_SUBMISSION_WINDOW`, `RADAR_TRAINING_WINDOW`, `RADAR_EVAL_WINDOW`, `RADAR_FALLBACK_WINDOW`.
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
- [x] HMAC shared-secret authentication (`shared/auth.py`, `RADAR_SHARED_SECRET`)
- [x] Static peer registry (`shared/peers.py`, `MINERS_CONFIG_PATH`)
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
- [x] Hippius (Substrate) artifact backend with R2 legacy fallback (`shared/r2_audit.py`)

## Artifact Storage (Hippius / R2)

Primary backend is **Hippius**, a Substrate-based decentralized object store
with an S3-compatible API at `https://s3.hippius.com`. Cloudflare R2 stays
supported as a legacy fallback so existing deployments keep working without
an env-var change. The client (`shared.r2_audit.HippiusStorage`, aliased as
`R2AuditLog`) reads `HIPPIUS_*` first, then falls back to `R2_*`:

| Var | Default | Purpose |
|-----|---------|---------|
| `HIPPIUS_ACCESS_KEY_ID` | `R2_ACCESS_KEY_ID` | Hippius access key (begins with `hip_`). Falls back to the legacy R2 var. |
| `HIPPIUS_SECRET_ACCESS_KEY` | `R2_SECRET_ACCESS_KEY` | Hippius secret. Falls back to the legacy R2 var. |
| `HIPPIUS_BUCKET` | `R2_BUCKET` | Bucket name for experiment artifacts. |
| `HIPPIUS_REGION` | `decentralized` | Required by Hippius; leave as default. |
| `HIPPIUS_ENDPOINT_URL` | `https://s3.hippius.com` | Override for private gateways. The legacy `R2_ACCOUNT_ID`-derived endpoint is only used when no `HIPPIUS_*` vars are set. |
| `MOCK_S3_ENDPOINT` | unset | Localnet override (`scripts/mock_r2_server.py`). `MOCK_R2_ENDPOINT` remains an alias for back-compat. |

Operators migrating from R2: leave the existing `R2_*` vars in place and
add `HIPPIUS_*` once your Hippius keys are issued — the client will pick up
the new backend automatically. To cut over fully, also set
`HIPPIUS_BUCKET` (or unset `R2_BUCKET`) so dispatched URLs point at Hippius.

## Non-competitive Mode

Set `RADAR_NONCOMPETITIVE=true` for a non-subnet deployment:

- Validator runs in HMAC-only mode (no weight setting, no on-chain
  commitments). Existing competitive deployments leave it false.
- Auth uses HMAC service key (`RADAR_SERVICE_KEY`) for validator ↔ trainer
  ↔ DB. Headers: `X-Radar-{Signature,Timestamp,Key-Id}`.
- Miner identity is bearer tokens issued via the operator CLI:
  ```
  python -m database.operator_cli register --name alice [--hotkey ss58]
  python -m database.operator_cli issue-key --miner-id <id> --label prod
  python -m database.operator_cli rotate-service-key
  python -m database.operator_cli list-miners | list-keys --miner-id <id>
  python -m database.operator_cli revoke-key --key-id <id>
  ```
- Miners pull scored history + evolve prompts locally via the miner CLI:
  ```
  RADAR_DB_URL=http://db:8090 RADAR_MINER_API_KEY=rdrk_... \
    python miner/neuron.py optimize --optimizer gepa --seed --watch
  python miner/neuron.py results --json
  python miner/neuron.py prompts list | history | rollback <gen>
  ```
- `prompts/active.json` (atomic-write) drives which prompt variant the
  agent picks each round; `prompt_id` round-trips back via
  `experiments.prompt_id` so Phase C scores attribute to the variant.

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

## Image Hardening (trainer container)

Miners deploy the trainer image on Targon, whose `ContainerConfig` API
lets the operator override `command`, `args`, `env`, `working_dir`, and
`security_context` without changing the image digest. The image
defends itself in a chain so a tampered deploy can't pretend to run
our harness:

1. **`/usr/local/bin/radar-entrypoint.sh`** (chmod 555) is the Docker
   ENTRYPOINT. Refuses to run if `$# > 0` (any operator-supplied
   `command`/`args`), refuses if any preload-style env var is set
   (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `LD_AUDIT`, `PYTHONPATH`,
   `PYTHONHOME`, `PYTHONSTARTUP`, `PYTHONINSPECT`, `PYTHONUSERBASE`),
   pins `PYTHONNOUSERSITE=1`, unsets every env var not on the
   allow-list (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TZ`, CUDA vars,
   trainer config, plus `WALLET_*` / `BT_*` / `RADAR_*` / `R2_*` /
   `S3_*` / `AWS_*` / `HIPPIUS_*` / `MOCK_*` prefixes), and
   `exec`s `python3 /workspace/_bootstrap.py`.
2. **`/workspace/_bootstrap.py`** (chmod 444) reads
   `/workspace/_bootstrap_hashes.json`, sha256-verifies every
   load-bearing file, refuses if any hash mismatches, refuses if any
   unexpected file appears in the protected dirs (`/workspace`,
   `/workspace/runner`, `/workspace/runner/timeseries_forecast`,
   `/workspace/shared`, `/workspace/frozen`), writes
   `/tmp/boot_proof.json`, then `os.execvp`s into
   `/workspace/server.py`. Stdlib-only — never imports torch / fastapi /
   anything redirectable.
3. **Build-time hash table** is generated by
   `/workspace/_gen_hashes.py` during `docker build` (single source of
   truth — the bootstrap reads but does not duplicate the file list).
   Build fails loudly if a COPY produced an extra or missing file.
4. **Per-job state lives outside `/workspace`** at
   `/var/radar/sandbox/`. `/workspace` itself is `chmod 555` so the
   bootstrap's protected-dirs check stays sound.
5. **`GET /boot_proof`** on the trainer returns the contents of
   `/tmp/boot_proof.json` along with a hotkey signature over its
   canonical-JSON encoding. Fields: `boot_time`, `files_hashed`,
   `file_count`, `hashes_root_sha256` (sha256 of the canonical-encoded
   expected hash table — one value validators can pin), and
   `bootstrap_version`. **If the proof file is missing the endpoint
   returns 503** — that's the bypass-detection signal validators
   consume when a miner overrode `command`/`args` to skip the
   entrypoint.

This image-level defense is one of several layers; behavioral
attestation, spot re-execution, canary tasks, and stamp streaming are
separate validator-side concerns and do not live here.

## Targon migration (Basilica → Targon hosting)

Trainer pods are migrating from Basilica to **Targon**, a confidential
compute platform that runs containers on Intel TDX CVMs with NVIDIA
H100/H200/B200 GPUs in CC mode. Targon provides hardware-rooted
attestation (TDX quote signed by the CPU + NRAS GPU token signed by
NVIDIA) verifiable through `https://tower.targon.com`.

Both miner and validator pick the backend via
`RADAR_HOSTING_BACKEND=basilica|targon` (default `basilica` during
rollout). A deployment is fully one or the other — no half-and-half.

### What Targon's attestation proves

| Attestation says | Targon proves it | Targon does NOT prove it |
|---|---|---|
| Image bytes | ✅ workload runs digest X | — |
| GPU is in CC mode | ✅ NRAS token | — |
| TDX hardware identity | ✅ Intel quote | — |
| Launch config (command/args/env) | ❌ — only image bytes | image hardening's job |
| Behavior matches intent | ❌ | spot re-execution + canary tasks |

The image-hardening layer (`runner/entrypoint.sh` + `_bootstrap.py` +
`/boot_proof`) closes the launch-config gap. Targon attestation +
boot proof together cover the full image-level trust chain.

### Verification stack (validator side)

For each round, after a miner posts `TrainerReady`:

1. **Workload digest** — `targon.verify_image_digest(uid, OFFICIAL_TRAINING_IMAGE_DIGEST)`.
2. **TDX + NRAS attestation** — `targon.verify_attestation(cvm_ip, miner_hk, validator_hk, nonce)` end-to-end via tower; cross-checks the GPU class miner declared.
3. **Boot proof** — `GET {trainer_url}/boot_proof`, validate signature. Feature-flagged via `RADAR_REQUIRE_BOOT_PROOF` (off until hardened image is universal). 503 = bootstrap didn't run = launch-config bypass.

All three must pass. Failures exclude the miner from the round.

### Mid-round re-verification

While Phase B runs, validators re-execute checks (1) and (3) at
2–3 deterministic offsets within the training window. Offsets are
seeded from `block_hash XOR round_id XOR miner_uid` — every validator
agrees on when checks happen but the miner can't predict them. Skip
(2) on re-verify because the workload UID is already pinned.
Mid-run failures mark the miner `compromised` for the round.

### What's per-round (off-chain) vs persistent (on-chain)

- **On-chain via `ImageCommitment`**: `code_hash`, `listener_url`,
  `subnet_version`. The same set as today — Targon does NOT add new
  on-chain fields. The trainer image digest is *not* on chain because
  every miner runs the same subnet-owner-blessed digest, so there's
  no per-miner variation worth committing to (and the 128-byte chain
  budget can't hold it anyway). Validators read the expected digest
  from `Config.OFFICIAL_TRAINING_IMAGE_DIGEST`.
- **Per-round via signed `TrainerReady`**: `targon_workload_uid`,
  `cvm_ip`, `gpu_class`, `image_digest`. Ephemeral, Epistula-signed
  by the miner hotkey, scoped to one round.

### Hybrid fallback when Targon is down

If the circuit breaker opens (5 consecutive failures within 60s
default), validators tag affected miners with `targon_unavailable`
on that round, accept their dispatches, and apply
`Config.TARGON_UNAVAILABLE_SCORE_MULTIPLIER` (default 0.5) to their
final score. This keeps the subnet alive during a Targon outage
without fully rewarding miners who couldn't be attested in real-time.

The 0.5 multiplier is a deliberate policy trade-off: full weight
during outages would let attackers wait for / trigger Targon
unavailability to launder unattested runs; zero weight would brick
the subnet during routine API maintenance. Half-weight stays open
for honest miners and rejects strategic exploitation. Tunable via
the env var if the trade-off needs adjusting.

### Operator runbook

**Miner**:

1. Get a Targon API key (https://docs.targon.com — separate account from validators).
2. Push the trainer image to a registry Targon can reach. Public GHCR works; for private registries set `RADAR_TARGON_REGISTRY_USERNAME`/`PASSWORD`/`SERVER`.
3. Set on the miner host:
   ```
   RADAR_HOSTING_BACKEND=targon
   TARGON_API_KEY=tg_...
   OFFICIAL_TRAINING_IMAGE=ghcr.io/...   # subnet-owner-blessed image
   OFFICIAL_TRAINING_IMAGE_DIGEST=sha256:...
   RADAR_TRAINER_GPU_MODELS=H200          # or H100, B200 — first entry wins
   ```
4. Restart the miner. The startup sequence under `RADAR_HOSTING_BACKEND=targon`:
   - Refuses to boot without `TARGON_API_KEY` (clear error pointing to docs).
   - Validates the key via a cheap `list_workloads` call before joining the peer set.
   - Lists any workloads owned by this account from a prior process and tears them down (`ORPHAN_TEARDOWN` log line per uid).

   While running:
   - Per round: deploy → poll `/health` + CVM `/api/v1/evidence` for up to `RADAR_TARGON_READINESS_TIMEOUT` seconds (default 180; TDX boot adds 60–120s) → tear down previous round's workload synchronously (3 retries, 1s/2s/4s backoff) → post `TrainerReady`.
   - On readiness timeout the workload is torn down and the round is dropped — we never advertise a half-ready URL.
   - A background `HealthMonitor` polls `/health` every 30s during the round; >2 consecutive minutes failed marks the round locally compromised (`HEALTH_COMPROMISED` log) but does NOT trigger redeploy. The validator's mid-run reverify is the authoritative signal — miner-side health is defensive logging only.
   - On SIGTERM/SIGINT/atexit, every active workload is torn down before the process exits. On crash, the next startup's orphan reaper catches the leak.

   Telemetry tags every Targon-touching log line with structured prefixes:
   `DEPLOY_OK`, `DEPLOY_FAILED`, `READINESS_OK`, `READINESS_TIMEOUT`,
   `TRAINER_READY`, `POST_READY_FAILED`, `PRIOR_TEARDOWN_LEAK`,
   `SHUTDOWN_TEARDOWN`, `ORPHAN_TEARDOWN`, `HEALTH_DEGRADED`,
   `HEALTH_RECOVERED`, `HEALTH_COMPROMISED` — plus `round_id`, `uid`,
   and elapsed seconds. `DEPLOY_FAILED` (code/auth) and `POST_READY_FAILED`
   (network/HTTP) are distinct log lines so operators can debug the right
   layer.

**Validator**:

1. Get a Targon API key (separate account from miners).
2. Set:
   ```
   RADAR_HOSTING_BACKEND=targon
   TARGON_API_KEY=tg_...
   OFFICIAL_TRAINING_IMAGE_DIGEST=sha256:...   # must match miner
   RADAR_REQUIRE_BOOT_PROOF=false              # flip to true once hardened image is universal
   ```
3. Restart the validator. The Targon client is constructed lazily —
   Basilica deployments don't load `targon-sdk` at runtime.

## What's Outstanding

- [ ] **Real Basilica deployment** — integrate Basilica API for pod lifecycle
- [ ] **Spot checking Phase A** — 10-20% of rounds re-audit agent submissions
- [ ] **Subnet LLM** — provide shared LLM endpoint to miners
- [ ] **Real GIFT-Eval data** — swap placeholder prepare.py with real data pipeline
- [ ] **Mainnet registration** — register subnet, set hyperparameters, deploy
- [ ] **Docker network isolation** — whitelist network for trainer containers
- [ ] **Cross-tempo EMA** — weight smoothing across rounds

## Code Style

- No file over 300 lines
- Tests for every module in `tests/`
- Type hints on all public functions
- `@dataclass` for data types (not Pydantic, matching protocol.py)
- Config via `os.getenv("RADAR_*", default)` in `config.py`
- Logging via `logger = logging.getLogger(__name__)`
