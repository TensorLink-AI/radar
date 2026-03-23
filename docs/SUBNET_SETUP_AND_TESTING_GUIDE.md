# EvoLoop Subnet — Setup & Testing Guide

Hands-on guide for running the EvoLoop subnet on a local chain or Bittensor testnet with multiple miners and validators, and verifying that every component works.

---

## Table of Contents

1. [Compute Requirements](#1-compute-requirements)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Environment Configuration](#3-environment-configuration)
4. [Quick Start: Automated Multi-Node Test](#4-quick-start-automated-multi-node-test)
5. [Local Subtensor (Dev Chain)](#5-local-subtensor-dev-chain)
6. [Single Miner + Single Validator (Smoke Test)](#6-single-miner--single-validator-smoke-test)
7. [Multiple Miners Setup](#7-multiple-miners-setup)
8. [Multiple Validators Setup](#8-multiple-validators-setup)
9. [Full Multi-Node Testbed (3 Miners + 2 Validators)](#9-full-multi-node-testbed-3-miners--2-validators)
10. [Testing the Agent (Phase A)](#10-testing-the-agent-phase-a)
11. [Testing the Trainer (Phase B)](#11-testing-the-trainer-phase-b)
12. [Testing the Evaluator (Phase C)](#12-testing-the-evaluator-phase-c)
13. [Testing Scoring & Weights](#13-testing-scoring--weights)
14. [Testing the DB Server API](#14-testing-the-db-server-api)
15. [Testing R2 Artifact Storage](#15-testing-r2-artifact-storage)
16. [Testing Cross-Eval & Work Splitting](#16-testing-cross-eval--work-splitting)
17. [Testing Fallback & Validator Failure](#17-testing-fallback--validator-failure)
18. [Deploying to Bittensor Testnet](#18-deploying-to-bittensor-testnet)
19. [Monitoring & Verification Checklist](#19-monitoring--verification-checklist)
20. [Troubleshooting](#20-troubleshooting)

---

## 1. Compute Requirements

### Deployment Modes: Local GPU vs Basilica

EvoLoop supports two training backends. This affects what hardware **you** need:

| | Local (`RADAR_RUNNER_BACKEND=local`) | Basilica (`RADAR_RUNNER_BACKEND=basilica`) |
|---|---|---|
| **Training (Phase B)** | Runs on **your** GPU | Runs on **Basilica remote GPU pods** |
| **Your server needs GPU?** | Yes (8+ GB VRAM) | **No** |
| **Agent collection (Phase A)** | Local Docker (CPU) | Local Docker (CPU) |
| **Evaluation (Phase C)** | Local CPU (default) | Local CPU (default) |
| **Use case** | Dev iteration, localnet testing | **Production, testnet, mainnet** |

**With Basilica, a CPU-only VPS is sufficient for validators and miners.** Basilica provides the GPU for training — you pay per-pod instead of provisioning your own hardware.

### Per-Component Resource Usage

| Component | Runs Where | CPU | RAM | GPU | Duration |
|-----------|-----------|-----|-----|-----|----------|
| **Subtensor (localnet)** | Local Docker | 1 core | 2 GB | No | Always on |
| **Miner process** | Local | 0.5 core | 0.5 GB | No | Always on |
| **Validator process** | Local | 1 core | 1-2 GB | No | Always on |
| **Agent container (Phase A)** | Local Docker | 2 cores | 8 GB max | No | 120s per agent |
| **Training (Phase B) — local mode** | Local Docker | 2 cores | 8 GB | **Yes (8+ GB VRAM)** | 300s per job |
| **Training (Phase B) — Basilica mode** | **Basilica remote pod** | — | — | **Their GPU** | 300s per job |
| **Evaluation (Phase C)** | Local (CPU) | 1 core | 1-4 GB | No | 5-30s per model |

### Recommended Server Configs

**With Basilica (production path — no local GPU needed):**

| Setup | CPU | RAM | GPU | Disk | What You Can Run |
|-------|-----|-----|-----|------|------------------|
| **Validator** | 4 cores | 16 GB | None | 20 GB | Full pipeline: collect agents, dispatch training to Basilica, evaluate checkpoints, set weights |
| **Miner** | 2 cores | 4 GB | None | 10 GB | Agent container + commitment |
| **Multi-validator testbed** | 8 cores | 32 GB | None | 40 GB | 3+ validators + 5+ miners, full round verification |

**With local training (dev/iteration):**

| Setup | CPU | RAM | GPU | Disk | What You Can Run |
|-------|-----|-----|-----|------|------------------|
| **Minimal (CPU-only)** | 4 cores | 16 GB | None | 20 GB | Everything except Phase B training |
| **Standard** | 8 cores | 32 GB | 1x GPU (8+ GB) | 50 GB | Full pipeline with local training |
| **Production-like** | 16 cores | 64 GB | 1-2x GPU (16+ GB) | 100 GB | 5+ miners, 3+ validators, multiple rounds |

### How Resources Scale

```
# With Basilica (no local GPU):
Total RAM ≈ 4 GB (subtensor + base) + 0.5 GB × miners + 2 GB × validators + 8 GB (shared agent pool)

Examples:
  3 miners + 2 validators: ~4 + 1.5 + 4 + 8 = ~18 GB
  5 miners + 3 validators: ~4 + 2.5 + 6 + 8 = ~21 GB
  8 miners + 4 validators: ~4 + 4   + 8 + 8 = ~24 GB
```

Agent containers run **sequentially** per validator (one at a time), so the 8 GB agent memory is shared, not multiplied by miner count.

### CPU-Only Testing

You can test everything **except Phase B training** without a GPU or Basilica:

```bash
bash scripts/multi_node_test.sh --cpu-only
```

This tests: agent collection (Phase A), scoring pipeline, work splitting across validators, EMA weight updates, DB API, Pareto frontier, cross-eval job assignment, and weight setting. Training is skipped but the rest of the pipeline runs.

With Basilica, `--cpu-only` is unnecessary — training runs remotely:

```bash
bash scripts/multi_node_test.sh --basilica
```

### Cloud Instance Recommendations

**With Basilica (CPU-only servers):**

| Provider | Instance | Specs | Cost (approx) | Good For |
|----------|----------|-------|----------------|----------|
| Any VPS | 4+ vCPU, 16+ GB | CPU only | ~$0.05-0.15/hr | Single validator or miner |
| AWS | `c6i.2xlarge` | 8 vCPU, 16 GB | ~$0.34/hr | Multi-node testbed |
| GCP | `e2-standard-8` | 8 vCPU, 32 GB | ~$0.27/hr | Multi-node testbed |
| Hetzner | `CPX41` | 8 vCPU, 16 GB | ~$0.04/hr | Budget multi-node |

**With local GPU (dev iteration):**

| Provider | Instance | Specs | Cost (approx) |
|----------|----------|-------|----------------|
| AWS | `g5.2xlarge` | 8 vCPU, 32 GB, 1x A10G (24 GB) | ~$1.20/hr |
| GCP | `g2-standard-8` | 8 vCPU, 32 GB, 1x L4 (24 GB) | ~$0.95/hr |
| Lambda | `gpu_1x_a10` | 30 vCPU, 200 GB, 1x A10 (24 GB) | ~$0.75/hr |

---

## 2. Prerequisites & Installation

```bash
# Python 3.10+
python3 --version

# Docker (daemon must be running)
docker info

# GITHUB_TOKEN for pulling subtensor localnet image from GHCR
export GITHUB_TOKEN=ghp_...

# Clone and install
git clone https://github.com/TensorLink-AI/radar.git
cd radar
pip install -e ".[dev]"
```

Verify:

```bash
python3 -c "
import shared.protocol, shared.task, shared.database, shared.scoring, shared.auth
import bittensor as bt
print(f'OK — bittensor {bt.__version__}')
"
```

---

## 3. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` for your deployment mode:

**Basilica mode (production — no local GPU required):**

```bash
# Use Basilica for training (remote GPU pods)
RADAR_RUNNER_BACKEND=basilica
BASILICA_API_TOKEN=your-basilica-token

# Affinetes mode controls how pods are launched
RADAR_AFFINETES_MODE=basilica

# Evaluation runs on CPU (no GPU needed)
RADAR_EVAL_DEVICE=cpu

# R2 for checkpoint/artifact storage
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-key
R2_SECRET_ACCESS_KEY=your-secret
R2_BUCKET=your-bucket

# Disable desearch for testing
RADAR_DESEARCH_ENABLED=false
```

**Local mode (dev iteration with local GPU):**

```bash
# Local Docker execution
RADAR_RUNNER_BACKEND=local
RADAR_AFFINETES_MODE=docker
RADAR_EVAL_DEVICE=cpu
RADAR_DESEARCH_ENABLED=false

# R2 (skip for localnet — uses file fallback)
# R2_ACCOUNT_ID=
# R2_ACCESS_KEY_ID=
# R2_SECRET_ACCESS_KEY=
# R2_BUCKET=
```

Full config reference in `config.py` — every value is set via `RADAR_*` env vars.

---

## 4. Quick Start: Automated Multi-Node Test

The fastest way to test the full subnet. One script handles everything:

```bash
# Default: 3 miners + 2 validators on localnet, wait for 1 round
bash scripts/multi_node_test.sh

# Customize the topology
bash scripts/multi_node_test.sh --miners 5 --validators 3
bash scripts/multi_node_test.sh --miners 8 --validators 4 --rounds 2

# Basilica mode — training on remote GPU pods, no local GPU needed
bash scripts/multi_node_test.sh --basilica

# CPU-only mode (no GPU, no Basilica — skips training, tests everything else)
bash scripts/multi_node_test.sh --cpu-only

# Testnet — uses Bittensor testnet instead of local subtensor (implies --basilica)
# First create wallets + subnet: bash scripts/create_test_wallets.sh --network test
# Then pass the netuid it assigned:
bash scripts/multi_node_test.sh --testnet --netuid <NETUID>

# Reuse an already-running subtensor
bash scripts/multi_node_test.sh --skip-subtensor

# Skip unit tests for faster iteration
bash scripts/multi_node_test.sh --skip-tests
```

### What the Script Does

1. **Preflight** — checks Python, Docker, GPU, available RAM/disk
2. **Unit tests** — runs the full test suite for fast feedback
3. **Builds** 3 agent Docker images (systematic, failure analyst, lineage tracker)
4. **Starts** local subtensor via Docker
5. **Creates** all wallets, funds from Alice dev account, registers neurons, stakes validators
6. **Starts** N miners (each with a different agent strategy, cycling through the 3 agents)
7. **Starts** M validators (each on a different DB port: 8080, 8081, ...)
8. **Waits** for round completion (monitors logs + DB API)
9. **Verifies**: neuron registration, commitments, DB experiments, Pareto front, cross-eval invariants, validator consistency

### Output

```
╔═══════════════════════════════════════════════════════════════╗
║                     TEST RESULTS                             ║
╠═══════════════════════════════════════════════════════════════╣
║  Miners:     3                                               ║
║  Validators: 2                                               ║
║  Rounds:     1 / 1 completed                                 ║
║  Checks:     6 passed, 0 failed                              ║
╠═══════════════════════════════════════════════════════════════╣
║  Log directory: /tmp/radar_test_12345/                     ║
║  DB APIs:                                                    ║
║    Validator 0: http://localhost:8080                         ║
║    Validator 1: http://localhost:8081                         ║
╚═══════════════════════════════════════════════════════════════╝
```

### Log Files

All logs are written to `/tmp/radar_test_<PID>/`:

```
/tmp/radar_test_12345/
├── validator0.log        # Validator 0 full log
├── validator1.log        # Validator 1 full log
├── miner0.log            # Miner 0 (systematic agent)
├── miner1.log            # Miner 1 (failure analyst)
├── miner2.log            # Miner 2 (lineage tracker)
├── experiments_v0/       # Validator 0 experiment DB
└── experiments_v1/       # Validator 1 experiment DB
```

The rest of this guide walks through each step manually if you want to test individual components or understand what the script does.

---

## 5. Local Subtensor (Dev Chain)

This simulates the Bittensor blockchain locally. Every command below uses `--subtensor.network local`.

### Start Subtensor

```bash
# Authenticate with GHCR
echo "$GITHUB_TOKEN" | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Pull and run
docker pull ghcr.io/opentensor/subtensor-localnet:devnet-ready
docker run -d \
    --name radar-subtensor \
    -p 9944:9944 \
    -p 9945:9945 \
    ghcr.io/opentensor/subtensor-localnet:devnet-ready
```

### Wait for It

```bash
for i in $(seq 1 30); do
    python3 -c "
import bittensor as bt
s = bt.Subtensor(network='local')
print(f'Block: {s.block}')
" 2>/dev/null && break
    echo "Waiting... ($i/30)"
    sleep 1
done
```

### Create Owner Wallet + Subnet

```python
import bittensor as bt

sub = bt.Subtensor(network='local')

# Alice is pre-funded on devnet
alice = bt.Wallet(name='alice-dev')
alice.create_coldkey_from_uri('//Alice', use_password=False, overwrite=True, suppress=True)
alice.create_new_hotkey(use_password=False, overwrite=True, suppress=True)
print(f'Alice balance: {sub.get_balance(alice.coldkeypub.ss58_address)}')

# Create owner wallet and fund it
owner = bt.Wallet(name='owner')
owner.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
sub.transfer(wallet=alice, destination_ss58=owner.coldkeypub.ss58_address,
             amount=bt.Balance.from_tao(100_000),
             wait_for_inclusion=True, wait_for_finalization=True)

# Create subnet
sub.register_subnet(wallet=owner, mev_protection=False)
print('Subnet created on netuid 1')
```

### Manage Subtensor

```bash
docker stop radar-subtensor    # pause
docker start radar-subtensor   # resume
docker rm -f radar-subtensor   # destroy
```

---

## 6. Single Miner + Single Validator (Smoke Test)

The minimal test to verify the pipeline works end-to-end.

### Step 1: Create Wallets and Register

```python
import bittensor as bt

sub = bt.Subtensor(network='local')
alice = bt.Wallet(name='alice-dev')
alice.create_coldkey_from_uri('//Alice', use_password=False, overwrite=True, suppress=True)
alice.create_new_hotkey(use_password=False, overwrite=True, suppress=True)

for name in ['validator', 'miner']:
    w = bt.Wallet(name=name)
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)

    # Fund
    sub.transfer(wallet=alice, destination_ss58=w.coldkeypub.ss58_address,
                 amount=bt.Balance.from_tao(100_000),
                 wait_for_inclusion=True, wait_for_finalization=True)

    # Register
    sub.burned_register(wallet=w, netuid=1, mev_protection=False)
    print(f'{name} registered, balance: {sub.get_balance(w.coldkeypub.ss58_address)}')

# Stake on validator
val = bt.Wallet(name='validator')
sub.add_stake(wallet=val, netuid=1, hotkey_ss58=val.hotkey.ss58_address,
              amount=bt.Balance.from_tao(1000), mev_protection=False)
```

### Step 2: Build Agent Image

```bash
docker build -t systematic:latest example_agents/systematic/
```

There are 3 example agents to choose from:

| Agent | Strategy | Location |
|-------|----------|----------|
| `systematic` | Pattern mine from top performers | `example_agents/systematic/` |
| `failure_analyst` | Avoid patterns from failures | `example_agents/failure_analyst/` |
| `lineage_tracker` | Trace evolutionary lineages | `example_agents/lineage_tracker/` |

### Step 3: Start Miner

```bash
python miner/neuron.py \
    --netuid 1 \
    --subtensor.network local \
    --wallet.name miner \
    --docker_image systematic:latest \
    --axon.external_ip 127.0.0.1 \
    > /tmp/radar_miner.log 2>&1 &
echo "Miner PID: $!"
```

The miner:
1. Gets the Docker image digest via `docker inspect`
2. Creates an `ImageCommitment` (image URL + digest + trainer URL)
3. Commits to chain (falls back to `/tmp/radar_commitments/` on localnet)
4. Stays alive, periodically syncing the metagraph

### Step 4: Start Validator

```bash
export RADAR_RUNNER_BACKEND=local
export RADAR_DESEARCH_ENABLED=false

python validator/neuron.py \
    --netuid 1 \
    --subtensor.network local \
    --wallet.name validator \
    --db_dir ./experiments \
    --db_port 8080 \
    > /tmp/radar_validator.log 2>&1 &
echo "Validator PID: $!"
```

### Step 5: Monitor

```bash
# Watch validator progress
tail -f /tmp/radar_validator.log | grep -E "(Phase|Round|Score|Weight|Error|DB)"

# Check DB API
curl -s http://localhost:8080/experiments/stats | python3 -m json.tool
curl -s http://localhost:8080/experiments/recent | python3 -m json.tool
```

### Step 6: Verify One Round Completed

Look for these in the validator log:
- `"Phase A: X proposals"` — agents submitted architectures
- `"Phase C: evaluated X checkpoints"` — evaluation ran
- `"Weights set for X UIDs"` — weights committed to chain

Or check programmatically:

```bash
# Experiments exist in DB
curl -s http://localhost:8080/experiments/stats

# Weights set on chain
python3 -c "
import bittensor as bt
sub = bt.Subtensor(network='local')
meta = sub.metagraph(netuid=1)
print(f'Neurons: {meta.n}')
for uid in range(meta.n):
    print(f'  UID {uid}: stake={meta.S[uid]:.1f} trust={meta.T[uid]:.3f}')
"
```

### Automated Smoke Test

The script does all of the above automatically:

```bash
bash scripts/test_localnet.sh          # full end-to-end
bash scripts/test_localnet.sh --quick  # unit tests only (no subtensor)
```

---

## 7. Multiple Miners Setup

Running 3 miners simultaneously, each with a different agent strategy.

### Create and Fund Miner Wallets

```python
import bittensor as bt

sub = bt.Subtensor(network='local')
alice = bt.Wallet(name='alice-dev')
alice.create_coldkey_from_uri('//Alice', use_password=False, overwrite=True, suppress=True)
alice.create_new_hotkey(use_password=False, overwrite=True, suppress=True)

for i in range(3):
    name = f'miner{i}'
    w = bt.Wallet(name=name)
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    sub.transfer(wallet=alice, destination_ss58=w.coldkeypub.ss58_address,
                 amount=bt.Balance.from_tao(100_000),
                 wait_for_inclusion=True, wait_for_finalization=True)
    sub.burned_register(wallet=w, netuid=1, mev_protection=False)
    print(f'{name} registered')
```

### Build 3 Different Agent Images

```bash
docker build -t agent-systematic:latest example_agents/systematic/
docker build -t agent-failure:latest    example_agents/failure_analyst/
docker build -t agent-lineage:latest    example_agents/lineage_tracker/
```

### Start All 3 Miners

```bash
# Miner 0: systematic agent
python miner/neuron.py \
    --netuid 1 --subtensor.network local \
    --wallet.name miner0 \
    --docker_image agent-systematic:latest \
    --axon.external_ip 127.0.0.1 \
    > /tmp/radar_miner0.log 2>&1 &
echo "Miner 0 PID: $!"

# Miner 1: failure analyst
python miner/neuron.py \
    --netuid 1 --subtensor.network local \
    --wallet.name miner1 \
    --docker_image agent-failure:latest \
    --axon.external_ip 127.0.0.1 \
    > /tmp/radar_miner1.log 2>&1 &
echo "Miner 1 PID: $!"

# Miner 2: lineage tracker
python miner/neuron.py \
    --netuid 1 --subtensor.network local \
    --wallet.name miner2 \
    --docker_image agent-lineage:latest \
    --axon.external_ip 127.0.0.1 \
    > /tmp/radar_miner2.log 2>&1 &
echo "Miner 2 PID: $!"
```

### Verify All Commitments

```bash
# All 3 miners should have committed their images
ls /tmp/radar_commitments/1/   # one JSON file per miner hotkey
```

---

## 8. Multiple Validators Setup

Running 2 validators that split work and independently evaluate checkpoints.

### Create and Fund Validator Wallets

```python
import bittensor as bt

sub = bt.Subtensor(network='local')
alice = bt.Wallet(name='alice-dev')
alice.create_coldkey_from_uri('//Alice', use_password=False, overwrite=True, suppress=True)
alice.create_new_hotkey(use_password=False, overwrite=True, suppress=True)

for i in range(2):
    name = f'validator{i}'
    w = bt.Wallet(name=name)
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    sub.transfer(wallet=alice, destination_ss58=w.coldkeypub.ss58_address,
                 amount=bt.Balance.from_tao(100_000),
                 wait_for_inclusion=True, wait_for_finalization=True)
    sub.burned_register(wallet=w, netuid=1, mev_protection=False)
    # Stake
    sub.add_stake(wallet=w, netuid=1, hotkey_ss58=w.hotkey.ss58_address,
                  amount=bt.Balance.from_tao(1000), mev_protection=False)
    print(f'{name} registered and staked')
```

### Start Both Validators (Different Ports)

```bash
export RADAR_RUNNER_BACKEND=local
export RADAR_DESEARCH_ENABLED=false

# Validator 0
python validator/neuron.py \
    --netuid 1 --subtensor.network local \
    --wallet.name validator0 \
    --db_dir ./experiments_v0 \
    --db_port 8080 \
    > /tmp/radar_validator0.log 2>&1 &
echo "Validator 0 PID: $!"

# Validator 1
python validator/neuron.py \
    --netuid 1 --subtensor.network local \
    --wallet.name validator1 \
    --db_dir ./experiments_v1 \
    --db_port 8081 \
    > /tmp/radar_validator1.log 2>&1 &
echo "Validator 1 PID: $!"
```

### Verify Work Splitting

Both validators compute the same challenge but split agent collection work:

```bash
# Both should show which miners they were assigned
grep "assigned\|my_assignments\|Phase A" /tmp/radar_validator0.log
grep "assigned\|my_assignments\|Phase A" /tmp/radar_validator1.log
```

Key behavior to verify:
- Each validator collects proposals from a **subset** of miners (work splitting)
- Both validators read **all** proposals from R2 after collection
- Both validators **independently evaluate every checkpoint** (Phase C)
- Both validators compute **identical scores** (deterministic)
- Both validators **set weights independently** on chain

### Compare Validator Outputs

```bash
# Both validators' DBs should converge to the same experiments
curl -s http://localhost:8080/experiments/stats | python3 -m json.tool
curl -s http://localhost:8081/experiments/stats | python3 -m json.tool

# Both should produce identical Pareto fronts
curl -s http://localhost:8080/experiments/pareto | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"
curl -s http://localhost:8081/experiments/pareto | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"
```

---

## 9. Full Multi-Node Testbed (3 Miners + 2 Validators)

This is the full realistic setup combining sections 7 and 8. You can also just run the automated script from section 4:

```bash
bash scripts/multi_node_test.sh --miners 3 --validators 2
```

Or do it manually:

```bash
#!/bin/bash
set -euo pipefail

# Prerequisites: subtensor running, wallets created & registered (see sections 4-7)

export RADAR_RUNNER_BACKEND=local
export RADAR_DESEARCH_ENABLED=false

PIDS=()
cleanup() {
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# ── Build agent images ──
docker build -t agent-systematic:latest example_agents/systematic/
docker build -t agent-failure:latest    example_agents/failure_analyst/
docker build -t agent-lineage:latest    example_agents/lineage_tracker/

# ── Start 3 miners ──
AGENTS=("agent-systematic:latest" "agent-failure:latest" "agent-lineage:latest")
for i in 0 1 2; do
    python miner/neuron.py \
        --netuid 1 --subtensor.network local \
        --wallet.name miner${i} \
        --docker_image "${AGENTS[$i]}" \
        --axon.external_ip 127.0.0.1 \
        > /tmp/radar_miner${i}.log 2>&1 &
    PIDS+=($!)
    echo "Miner ${i} started (PID $!) — ${AGENTS[$i]}"
done

sleep 5  # Let miners commit their images

# ── Start 2 validators ──
for i in 0 1; do
    PORT=$((8080 + i))
    python validator/neuron.py \
        --netuid 1 --subtensor.network local \
        --wallet.name validator${i} \
        --db_dir ./experiments_v${i} \
        --db_port ${PORT} \
        > /tmp/radar_validator${i}.log 2>&1 &
    PIDS+=($!)
    echo "Validator ${i} started (PID $!) — port ${PORT}"
done

echo ""
echo "All processes running. Log files:"
echo "  Miners:     /tmp/radar_miner{0,1,2}.log"
echo "  Validators: /tmp/radar_validator{0,1}.log"
echo ""
echo "Monitor with:"
echo "  tail -f /tmp/radar_validator0.log"
echo ""
echo "Check DB APIs:"
echo "  curl -s http://localhost:8080/experiments/stats | python3 -m json.tool"
echo "  curl -s http://localhost:8081/experiments/stats | python3 -m json.tool"
echo ""
echo "Press Ctrl+C to stop all processes."

# Wait for all processes
wait
```

### What to Expect

| Phase | What Happens | Duration |
|-------|-------------|----------|
| Startup | Miners commit images, validators start DB servers | ~10 sec |
| Phase A | Validators split 3 miners between them, pull Docker images, run agents | ~10 min |
| Phase B | Cross-eval: each miner's architecture trained on a different miner's trainer | ~30 min |
| Phase C | Both validators independently evaluate all checkpoints | ~5 min |
| Scoring | Size gate → Pareto ranking → softmax → EMA → set weights | ~1 min |

### Verification Checklist

```bash
# 1. All miners committed images
ls /tmp/radar_commitments/1/

# 2. Validators found all commitments
grep "commitments" /tmp/radar_validator0.log

# 3. Phase A collected proposals
grep "Phase A" /tmp/radar_validator0.log /tmp/radar_validator1.log

# 4. Cross-eval: no miner trained their own architecture
grep "arch_owner\|trainer_uid\|cross.eval" /tmp/radar_validator0.log

# 5. Phase C evaluated checkpoints
grep "Phase C" /tmp/radar_validator0.log

# 6. Weights set
grep "Weights set" /tmp/radar_validator0.log /tmp/radar_validator1.log

# 7. Metagraph reflects weights
python3 -c "
import bittensor as bt
sub = bt.Subtensor(network='local')
meta = sub.metagraph(netuid=1)
for uid in range(meta.n):
    print(f'UID {uid}: stake={meta.S[uid]:.1f} emission={meta.E[uid]:.6f}')
"
```

---

## 10. Testing the Agent (Phase A)

The agent is a Docker container. It reads a Challenge JSON from stdin and writes a Proposal JSON to stdout.

### Test an Agent Manually

```bash
# Create a test challenge
cat << 'EOF' > /tmp/test_challenge.json
{
  "challenge_id": "test-001",
  "seed": 42,
  "round_id": 1,
  "min_flops_equivalent": 100000,
  "max_flops_equivalent": 500000,
  "eval_split_seed": 42,
  "task": {"name": "ts_forecasting"},
  "db_url": "",
  "desearch_url": "",
  "feasible_frontier": []
}
EOF

# Run the agent (should output Proposal JSON to stdout)
cat /tmp/test_challenge.json | docker run -i --rm systematic:latest
```

Expected output: a JSON with `code`, `name`, and `motivation` fields:

```json
{
  "code": "import torch\nimport torch.nn as nn\n...\ndef build_model(...): ...\ndef build_optimizer(model): ...",
  "name": "my_architecture",
  "motivation": "Why this design works..."
}
```

### Verify Agent Code is Valid

The submitted `code` must contain these functions:

```python
# Extract the code from the proposal and validate
python3 << 'PYEOF'
import json, sys

# Read proposal output from agent
proposal = json.loads(open("/tmp/test_proposal.json").read())
code = proposal["code"]

# Write to a temp file and try to import
with open("/tmp/test_submission.py", "w") as f:
    f.write(code)

# Check required functions exist
import importlib.util
spec = importlib.util.spec_from_file_location("sub", "/tmp/test_submission.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assert hasattr(mod, "build_model"), "Missing build_model()"
assert hasattr(mod, "build_optimizer"), "Missing build_optimizer()"
print("Agent code is valid")
PYEOF
```

### Test Agent with Frontier Context

Agents work better when given the current Pareto frontier:

```bash
# With a running validator, include real frontier data
FRONTIER=$(curl -s http://localhost:8080/frontier)
python3 -c "
import json
challenge = json.load(open('/tmp/test_challenge.json'))
frontier = json.loads('${FRONTIER}')
challenge['feasible_frontier'] = frontier.get('frontier', [])
challenge['db_url'] = 'http://localhost:8080'
print(json.dumps(challenge))
" | docker run -i --rm systematic:latest
```

### Test Agent Timeout Handling

The validator gives agents 120 seconds. Test what happens when an agent is slow:

```bash
# The validator's collection module uses a 120s timeout
# An agent that exceeds this gets killed and receives no score
timeout 120 bash -c 'cat /tmp/test_challenge.json | docker run -i --rm systematic:latest'
```

---

## 11. Testing the Trainer (Phase B)

The trainer is the frozen training loop that runs inside a Docker container.

### Build and Start the Trainer

```bash
docker build -t ts-runner:latest runner/timeseries_forecast/
docker run --rm -p 8081:8081 ts-runner:latest
```

### Test the Training Endpoint

```bash
curl -X POST http://localhost:8081/train \
  -H "Content-Type: application/json" \
  -d '{
    "architecture": "import torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 5)\n    def forward(self, x):\n        return self.linear(x)\ndef build_model(context_len=96, prediction_len=24, num_variates=1, quantiles=None):\n    return Model()\ndef build_optimizer(model):\n    return torch.optim.Adam(model.parameters(), lr=0.001)",
    "seed": 42,
    "round_id": 1,
    "min_flops_equivalent": 100000,
    "max_flops_equivalent": 500000,
    "miner_hotkey": "test_hotkey",
    "time_budget": 60
  }'
```

Expected response:

```json
{
  "round_id": 1,
  "miner_hotkey": "test_hotkey",
  "status": "success",
  "flops_equivalent_size": 150000,
  "training_time_seconds": 45.2,
  "num_steps": 1234,
  "checkpoint_key": "round_1/miner_test_hotkey/checkpoint.safetensors",
  "architecture_key": "round_1/miner_test_hotkey/architecture.py"
}
```

### Test Size Gate Rejection

Submit an architecture that's too large for the bucket:

```bash
curl -X POST http://localhost:8081/train \
  -H "Content-Type: application/json" \
  -d '{
    "architecture": "import torch\nimport torch.nn as nn\nclass HugeModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(100)])\n    def forward(self, x):\n        return self.layers(x)\ndef build_model(**kwargs):\n    return HugeModel()\ndef build_optimizer(model):\n    return torch.optim.Adam(model.parameters())",
    "seed": 42,
    "round_id": 1,
    "min_flops_equivalent": 100000,
    "max_flops_equivalent": 500000,
    "miner_hotkey": "test",
    "time_budget": 30
  }'
```

Should return `"status": "size_violation"`.

### Test Invalid Code Rejection

```bash
curl -X POST http://localhost:8081/train \
  -H "Content-Type: application/json" \
  -d '{
    "architecture": "this is not valid python",
    "seed": 42,
    "round_id": 1,
    "min_flops_equivalent": 100000,
    "max_flops_equivalent": 500000,
    "miner_hotkey": "test",
    "time_budget": 30
  }'
```

Should return `"status": "build_failed"`.

---

## 12. Testing the Evaluator (Phase C)

Phase C is the trust anchor — every validator independently evaluates checkpoints.

### Run Evaluation Unit Tests

```bash
python -m pytest tests/test_evaluator.py tests/test_evaluate.py -v
```

### Verify FLOPs Measurement

The evaluator checks that the trainer's claimed FLOPs match the validator's measurement:

```python
from validator.evaluator import verify_flops_claim

# 2% tolerance
assert verify_flops_claim(claimed=200_000, measured=200_000)  # exact match
assert verify_flops_claim(claimed=200_000, measured=203_000)  # within 2%
assert not verify_flops_claim(claimed=200_000, measured=250_000)  # too far off
```

### Verify Checkpoint Format

Checkpoints must be safetensors (no pickle for security):

```python
from safetensors.torch import save_file, load_file
import torch

# Valid: safetensors
model = torch.nn.Linear(10, 5)
save_file(model.state_dict(), "/tmp/test_checkpoint.safetensors")
loaded = load_file("/tmp/test_checkpoint.safetensors")
print(f"Loaded {len(loaded)} tensors from safetensors")

# Invalid: pickle-based .pt files are rejected by the evaluator
```

---

## 13. Testing Scoring & Weights

### Score a Round with Multiple Miners

```python
from shared.scoring import score_round, scores_to_weights, ema_update
from shared.pareto import ParetoFront
from shared.task import Objective

class MockChallenge:
    min_flops_equivalent = 100_000
    max_flops_equivalent = 500_000

objectives = [
    Objective(name="crps", pattern=r"crps:\s*([\d.]+)", lower_is_better=True, primary=True),
]

# Simulate 4 miners with different CRPS scores
eval_results = {
    0: {"crps": 0.80, "flops_equivalent_size": 200_000, "passed_size_gate": True},
    1: {"crps": 0.65, "flops_equivalent_size": 300_000, "passed_size_gate": True},
    2: {"crps": 0.90, "flops_equivalent_size": 150_000, "passed_size_gate": True},
    3: {"crps": 0.50, "flops_equivalent_size": 600_000, "passed_size_gate": False},  # fails size gate
}

pareto = ParetoFront(max_size=50)
scores = score_round(eval_results, MockChallenge(), pareto, objectives, {})

print("Scores:")
for uid, score in sorted(scores.items()):
    print(f"  UID {uid}: {score:.4f}")

# UID 3 should be 0 (failed size gate)
assert scores[3] == 0.0
# UID 1 should score highest (best CRPS among valid)
assert scores[1] == max(scores.values())

# Convert to weights
uids, weights = scores_to_weights(scores, temperature=0.1)
print(f"\nWeights: {dict(zip(uids, [f'{w:.4f}' for w in weights]))}")
print(f"Sum: {sum(weights):.6f}")
```

### Test EMA Over Multiple Rounds

```python
ema = {}
all_uids = [0, 1, 2, 3]

# Simulate 5 rounds where UID 1 consistently performs best
for round_num in range(5):
    round_scores = {0: 0.3, 1: 0.9, 2: 0.5, 3: 0.1}
    ema = ema_update(ema, round_scores, all_uids, alpha=0.3)
    print(f"Round {round_num}: EMA = {ema}")

# UID 1 should converge toward 0.9
assert ema[1] > 0.8
print(f"\nFinal EMA: {ema}")
```

---

## 14. Testing the DB Server API

The validator runs a FastAPI server for miners to query experiment history.

### Start Standalone DB Server

```bash
python3 -c "
from validator.db_server import app, set_db
from shared.database import ExperimentDB, DataElement
import uvicorn, tempfile

db = ExperimentDB(db_dir=tempfile.mkdtemp())
# Add some seed data
for i in range(5):
    db.add(DataElement(name=f'test_{i}', metric=0.5+i*0.1, success=True,
                       objectives={'crps': 0.5+i*0.1, 'flops_equivalent_size': 200000+i*50000}))
set_db(db)
uvicorn.run(app, host='0.0.0.0', port=8080)
" &
sleep 2
```

### Test All Endpoints

```bash
# Health check
curl -s http://localhost:8080/health

# DB statistics
curl -s http://localhost:8080/experiments/stats | python3 -m json.tool

# Recent experiments
curl -s http://localhost:8080/experiments/recent?n=3 | python3 -m json.tool

# Single experiment by index
curl -s http://localhost:8080/experiments/0 | python3 -m json.tool

# Experiment lineage
curl -s http://localhost:8080/experiments/lineage/0 | python3 -m json.tool

# Failed experiments
curl -s http://localhost:8080/experiments/failures | python3 -m json.tool

# Pareto front members
curl -s http://localhost:8080/experiments/pareto | python3 -m json.tool

# Search experiments by keyword
curl -s -X POST http://localhost:8080/experiments/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# Current challenge (empty if no round active)
curl -s http://localhost:8080/challenge | python3 -m json.tool

# Current frontier
curl -s http://localhost:8080/frontier | python3 -m json.tool
```

### Test Rate Limiting

The DB server rate-limits miners to 10 requests/minute per hotkey:

```bash
# Rapid-fire 15 requests — last few should be rate-limited
for i in $(seq 1 15); do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/experiments/stats)
    echo "Request $i: HTTP $STATUS"
done
```

---

## 15. Testing R2 Artifact Storage

R2 stores checkpoints, proposals, dispatch records, and the frontier.

### With Real R2 Credentials

```bash
# Set credentials
export R2_ACCOUNT_ID=your-account-id
export R2_ACCESS_KEY_ID=your-key
export R2_SECRET_ACCESS_KEY=your-secret
export R2_BUCKET=your-bucket
```

```python
from shared.r2_audit import R2AuditLog

r2 = R2AuditLog()

# Upload a test artifact
r2.upload_json("test/hello.json", {"message": "works"})
data = r2.download_json("test/hello.json")
print(f"Downloaded: {data}")

# List artifacts
keys = r2.list_keys("test/")
print(f"Keys: {keys}")
```

### Without R2 (Localnet)

On localnet without R2 credentials, the system uses file-based fallback. Verify:

```bash
# Commitments stored as files
ls /tmp/radar_commitments/1/

# Validator should log this
grep "R2.*disabled\|file fallback" /tmp/radar_validator.log
```

### Run R2 Tests (In-Memory Mock)

```bash
python -m pytest tests/test_r2_audit.py tests/test_r2_audit_extended.py tests/test_artifacts.py -v
```

---

## 16. Testing Cross-Eval & Work Splitting

These are critical invariants — verify them with both unit tests and live observation.

### Unit Tests

```bash
python -m pytest tests/test_coordinator.py tests/test_work_splitting.py tests/test_multi_miner_validator.py -v
```

### Cross-Eval Invariant

With 3+ miners, no miner should train their own architecture:

```python
from validator.coordinator import compute_assignments
from shared.protocol import Proposal

submissions = {i: Proposal(code=f"arch_{i}") for i in range(5)}
jobs = compute_assignments("a" * 64, submissions, [0,1,2,3,4], [100,101], round_id=1)

for job in jobs:
    print(f"Arch owner={job.arch_owner} trained by={job.trainer_uid} dispatched by={job.dispatcher}")
    assert job.arch_owner != job.trainer_uid, "Self-training detected!"
```

### Work Splitting Invariant

Every miner assigned to exactly one validator, no overlaps:

```python
from validator.neuron import get_my_assignments

miners = list(range(10))
validators = [100, 101, 102]

all_assigned = []
for v in validators:
    assigned = get_my_assignments(miners, validators, v, seed=42)
    print(f"Validator {v} assigned miners: {assigned}")
    all_assigned.extend(assigned)

assert sorted(all_assigned) == sorted(miners), "Not all miners covered!"
assert len(all_assigned) == len(set(all_assigned)), "Overlap detected!"
```

### Live Verification

With the multi-node testbed running (section 8):

```bash
# Check that validators split work
grep "assigned\|my_assignments" /tmp/radar_validator0.log
grep "assigned\|my_assignments" /tmp/radar_validator1.log

# Check cross-eval in coordinator logs
grep "arch_owner.*trainer_uid" /tmp/radar_validator0.log
```

---

## 17. Testing Fallback & Validator Failure

### Unit Test

```bash
python -m pytest tests/test_multi_miner_validator.py::TestMultiValidatorFallback -v
```

### Simulate Validator Failure

With the multi-node testbed running:

```bash
# Kill validator 1
kill $(pgrep -f "wallet.name validator1")

# Watch validator 0's logs for fallback behavior
tail -f /tmp/radar_validator0.log | grep -i "fallback\|missing\|reassign"
```

If `RADAR_FALLBACK_ENABLED=true`, validator 0 should pick up validator 1's orphaned jobs.

### Verify Programmatically

```python
from validator.coordinator import compute_fallback, Job

# 6 jobs split across 3 validators
jobs = [
    Job(arch_owner=i, trainer_uid=(i+1)%6, dispatcher=100+(i%3), round_id=1)
    for i in range(6)
]

# Validators 100 and 101 go offline
reassigned = compute_fallback("a"*64, [100, 101], jobs, [102])

print(f"Reassigned {len(reassigned)} jobs to remaining validator(s)")
for j in reassigned:
    print(f"  Arch {j.arch_owner} → trainer {j.trainer_uid}, dispatcher {j.dispatcher}")
    assert j.dispatcher == 102, "Should go to the only remaining validator"
```

---

## 18. Deploying to Bittensor Testnet

On testnet and mainnet, use **Basilica for training** — no local GPU needed.

### Quick Path: Automated Wallet + Subnet Setup

The wallet script handles everything — owner, subnet, miners, validators:

```bash
# 1. Fund the owner wallet first (wallets are created with no password)
bash scripts/create_test_wallets.sh --network test --skip-register

# 2. Fund all wallets via faucet (the script prints the commands)
btcli wallet faucet --wallet.name owner --subtensor.network test
btcli wallet faucet --wallet.name validator0 --subtensor.network test
btcli wallet faucet --wallet.name validator1 --subtensor.network test
btcli wallet faucet --wallet.name miner0 --subtensor.network test
btcli wallet faucet --wallet.name miner1 --subtensor.network test
btcli wallet faucet --wallet.name miner2 --subtensor.network test

# 3. Now create subnet + register all neurons (owner creates the subnet)
bash scripts/create_test_wallets.sh --network test
# Output: "Subnet created — netuid <N>"
# Note the netuid!

# 4. Run the test
bash scripts/multi_node_test.sh --testnet --netuid <NETUID> --miners 3 --validators 2
```

### Manual Path (Step by Step)

#### Step 1: Create Owner Wallet + Subnet

```bash
# Create and fund the owner wallet
btcli wallet create --wallet.name owner
btcli wallet faucet --wallet.name owner --subtensor.network test

# Create the subnet — the chain assigns the netuid
btcli subnet create --wallet.name owner --subtensor.network test
# Output: "Registered subnet with netuid: <N>"
# USE THIS NETUID IN ALL COMMANDS BELOW
```

#### Step 2: Create, Fund, and Register Neurons

```bash
NETUID=<N>  # from step 1

# Create wallets (no password)
for NAME in validator0 validator1 miner0 miner1 miner2; do
    btcli wallet create --wallet.name $NAME
    btcli wallet faucet --wallet.name $NAME --subtensor.network test
    btcli subnet register --wallet.name $NAME --netuid $NETUID --subtensor.network test
done

# Stake on validators
btcli stake add --wallet.name validator0 --netuid $NETUID --subtensor.network test --amount 100
btcli stake add --wallet.name validator1 --netuid $NETUID --subtensor.network test --amount 100
```

#### Step 3: Push Docker Images to Registry

```bash
docker tag agent-systematic:latest ghcr.io/YOUR_ORG/agent-systematic:v1
docker tag agent-failure:latest    ghcr.io/YOUR_ORG/agent-failure:v1
docker tag agent-lineage:latest    ghcr.io/YOUR_ORG/agent-lineage:v1
docker push ghcr.io/YOUR_ORG/agent-systematic:v1
docker push ghcr.io/YOUR_ORG/agent-failure:v1
docker push ghcr.io/YOUR_ORG/agent-lineage:v1
```

#### Step 4: Configure Environment

```bash
NETUID=<N>  # from step 1

# Basilica mode — no local GPU required
export RADAR_RUNNER_BACKEND=basilica
export RADAR_AFFINETES_MODE=basilica
export BASILICA_API_TOKEN=your-token
export RADAR_EVAL_DEVICE=cpu
export RADAR_DESEARCH_ENABLED=false

# R2 for artifact storage
export R2_ACCOUNT_ID=your-account-id
export R2_ACCESS_KEY_ID=your-key
export R2_SECRET_ACCESS_KEY=your-secret
export R2_BUCKET=your-bucket
```

#### Step 5: Start Miners (CPU-only server is fine)

```bash
AGENTS=("agent-systematic" "agent-failure" "agent-lineage")
for i in 0 1 2; do
    python miner/neuron.py \
        --netuid $NETUID \
        --subtensor.network test \
        --wallet.name miner${i} \
        --docker_image ghcr.io/YOUR_ORG/${AGENTS[$i]}:v1 \
        > /tmp/radar_miner${i}.log 2>&1 &
done
```

#### Step 6: Start Validators (CPU-only server is fine)

```bash
for i in 0 1; do
    python validator/neuron.py \
        --netuid $NETUID \
        --subtensor.network test \
        --wallet.name validator${i} \
        --db_dir ./experiments_v${i} \
        --db_port $((8080 + i)) \
        > /tmp/radar_validator${i}.log 2>&1 &
done
```

### Automated Testnet Multi-Node Test

If wallets are already funded and registered:

```bash
bash scripts/multi_node_test.sh --testnet --netuid <NETUID> \
    --miners 3 --validators 2 --skip-tests
```

This skips localnet subtensor setup and connects directly to testnet.

---

## 19. Monitoring & Verification Checklist

### Per-Round Checks

Run after each round (~55 minutes) to verify everything is working:

```bash
#!/bin/bash
# monitoring_check.sh — run periodically

NETUID=1
NETWORK=local  # or "test"
DB_PORT=8080

echo "=== Metagraph ==="
python3 -c "
import bittensor as bt
sub = bt.Subtensor(network='${NETWORK}')
meta = sub.metagraph(netuid=${NETUID})
print(f'Neurons: {meta.n}')
for uid in range(meta.n):
    print(f'  UID {uid}: stake={meta.S[uid]:.1f} trust={meta.T[uid]:.3f} emission={meta.E[uid]:.6f}')
"

echo ""
echo "=== DB Stats ==="
curl -s http://localhost:${DB_PORT}/experiments/stats | python3 -m json.tool

echo ""
echo "=== Pareto Front ==="
curl -s http://localhost:${DB_PORT}/experiments/pareto | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'{len(data)} members on Pareto front')
for e in data[:5]:
    print(f'  {e.get(\"name\", \"?\")} — metric={e.get(\"metric\", \"?\")}, flops={e.get(\"objectives\", {}).get(\"flops_equivalent_size\", \"?\")}')
"

echo ""
echo "=== Recent Experiments ==="
curl -s http://localhost:${DB_PORT}/experiments/recent?n=5 | python3 -c "
import sys, json
data = json.load(sys.stdin)
for e in data:
    status = '✓' if e.get('success') else '✗'
    print(f'  {status} {e.get(\"name\", \"?\")} — metric={e.get(\"metric\", \"?\")}')
"

echo ""
echo "=== Last 5 Log Lines ==="
tail -5 /tmp/radar_validator0.log 2>/dev/null
```

### Things That Should Be True

| Check | How to Verify |
|-------|--------------|
| All miners committed images | `ls /tmp/radar_commitments/1/` has one file per miner |
| Validators found commitments | Log shows `"Found N commitments"` |
| Phase A collected proposals | Log shows `"Phase A: N proposals"` |
| Cross-eval enforced | No `arch_owner == trainer_uid` in job assignments |
| Work split across validators | Each validator's log shows different assigned miner UIDs |
| Phase C evaluated checkpoints | Log shows `"Phase C: evaluated N checkpoints"` |
| Scoring produced weights | Log shows `"Weights set for N UIDs"` |
| DB has experiments | `/experiments/stats` returns `total > 0` |
| Pareto front growing | `/experiments/pareto` returns non-empty list |
| Weights set on chain | `metagraph.E[uid] > 0` for at least one miner |
| Both validators agree | Same Pareto front members on both DB APIs |

---

## 20. Troubleshooting

### Subtensor Won't Start

```bash
docker logs radar-subtensor --tail 30
# Common fix: port 9944 already in use
docker rm -f radar-subtensor
docker run -d --name radar-subtensor -p 9944:9944 -p 9945:9945 \
    ghcr.io/opentensor/subtensor-localnet:devnet-ready
```

### Miner Won't Commit Image

```bash
# Check if Docker image exists
docker images | grep systematic

# Check miner logs
tail -20 /tmp/radar_miner0.log

# Verify commitment file was written (localnet fallback)
ls -la /tmp/radar_commitments/1/
```

### Validator Can't Find Commitments

```bash
# Commitments use file fallback on localnet
ls /tmp/radar_commitments/1/
# Should have one .json file per miner hotkey

# Check validator log
grep "commitment" /tmp/radar_validator0.log
```

### Agent Timeout / No Proposal

```bash
# Test agent manually
cat /tmp/test_challenge.json | timeout 120 docker run -i --rm systematic:latest

# Check Docker resource limits
docker stats --no-stream
```

### Validator DB API Returns Empty

```bash
# Validator needs to complete at least one full round (~55 min)
grep "Round.*complete\|DB size" /tmp/radar_validator0.log

# Check DB directory has data
ls -la ./experiments_v0/
```

### Port Conflicts

```bash
# Find what's using a port
lsof -i :8080

# Use different ports for each validator
# Validator 0: --db_port 8080
# Validator 1: --db_port 8081
```

### Killing Everything

```bash
# Stop all radar processes
pkill -f "miner/neuron.py"
pkill -f "validator/neuron.py"
docker stop radar-subtensor

# Clean up temp files
rm -rf /tmp/radar_*.log /tmp/radar_commitments
rm -rf ./experiments_v0 ./experiments_v1
```

### Running Unit Tests for Confidence

Before live testing, run the full suite:

```bash
# All 240+ tests
python -m pytest tests/ -v

# Just the multi-miner/multi-validator tests
python -m pytest tests/test_multi_miner_validator.py -v
```
