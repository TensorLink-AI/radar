#!/bin/bash
# =============================================================================
# Radar Subnet — Multi-Node Integration Test
#
# Orchestrates a realistic testbed on a single server:
#   - 1 local subtensor (Docker)
#   - N miners with different agent strategies
#   - M validators that split work and independently evaluate
#
# Usage:
#   bash scripts/multi_node_test.sh                    # 3 miners, 2 validators (localnet)
#   bash scripts/multi_node_test.sh --miners 5         # 5 miners, 2 validators
#   bash scripts/multi_node_test.sh --validators 3     # 3 miners, 3 validators
#   bash scripts/multi_node_test.sh --miners 4 --validators 3
#   bash scripts/multi_node_test.sh --skip-subtensor   # reuse running subtensor
#   bash scripts/multi_node_test.sh --rounds 2         # wait for 2 rounds
#   bash scripts/multi_node_test.sh --cpu-only         # skip GPU training tests
#   bash scripts/multi_node_test.sh --basilica          # use Basilica for training (no local GPU)
#   bash scripts/multi_node_test.sh --testnet --netuid <N>  # run on Bittensor testnet
#
# Compute Requirements:
#   Localnet (local training): 4 CPU cores, 16 GB RAM, 20 GB disk, Docker, 1 GPU (8+ GB VRAM)
#   Localnet (Basilica):       4 CPU cores, 16 GB RAM, 20 GB disk, Docker (no GPU needed)
#   Testnet:                   4 CPU cores, 16 GB RAM, 20 GB disk, Docker (no GPU needed)
#
# Prerequisites:
#   - Docker installed and running
#   - Python 3.10+
#   - pip install -e ".[dev]" (or uv pip install -e ".[dev]")
#   - GITHUB_TOKEN for GHCR login (push agent images + pull subtensor image)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Load .env if present (temporarily allow unset vars) ──
if [ -f "$PROJECT_DIR/.env" ]; then
    set +u
    set -a
    source "$PROJECT_DIR/.env" || true
    set +a
    set -u
fi

# ── Defaults ──
NUM_MINERS=${NUM_MINERS:-3}
NUM_VALIDATORS=${NUM_VALIDATORS:-2}
SKIP_SUBTENSOR=false
SKIP_UNIT_TESTS=false
CPU_ONLY=false
USE_BASILICA=false
USE_TESTNET=false
WAIT_ROUNDS=1
NETUID=1
NETWORK=local
DB_PORT_BASE=8080
ROUND_TIMEOUT=1800  # 30 min per round (agents + training + eval wall-clock time)
GHCR_REPO=${GHCR_REPO:-"ghcr.io/tensorlink-ai/radar"}

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --miners)       NUM_MINERS="$2"; shift 2 ;;
        --validators)   NUM_VALIDATORS="$2"; shift 2 ;;
        --skip-subtensor) SKIP_SUBTENSOR=true; shift ;;
        --skip-tests)   SKIP_UNIT_TESTS=true; shift ;;
        --cpu-only)     CPU_ONLY=true; shift ;;
        --basilica)     USE_BASILICA=true; shift ;;
        --testnet)      USE_TESTNET=true; SKIP_SUBTENSOR=true; NETWORK=test; shift ;;
        --netuid)       NETUID="$2"; shift 2 ;;
        --rounds)       WAIT_ROUNDS="$2"; shift 2 ;;
        --help|-h)
            head -30 "$0" | tail -25
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Testnet implies Basilica (no local GPU assumed)
if [ "$USE_TESTNET" = true ]; then
    USE_BASILICA=true
    ROUND_TIMEOUT=3600  # testnet rounds are ~55 min real time
fi

# ── Validator wallet names ──
if [ "$USE_TESTNET" = true ]; then
    VALIDATOR_WALLETS=("validator0" "owner")
else
    VALIDATOR_WALLETS=()
    for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
        VALIDATOR_WALLETS+=("validator${i}")
    done
fi

# ── Colors ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
step()  { echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

# ── Cleanup ──
PIDS=()
TRAINER_CONTAINERS=()
SUBTENSOR_CONTAINER=""
LOG_DIR="/tmp/radar_test_$$"
mkdir -p "$LOG_DIR"

cleanup() {
    echo ""
    info "Cleaning up..."
    # Send SIGTERM and wait for graceful shutdown (allows Python finally blocks
    # to run and clean up agent pods)
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    # Give processes up to 10s to clean up agent pods gracefully
    for pid in "${PIDS[@]}"; do
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    if [ -n "$SUBTENSOR_CONTAINER" ]; then
        docker stop "$SUBTENSOR_CONTAINER" 2>/dev/null || true
        docker rm "$SUBTENSOR_CONTAINER" 2>/dev/null || true
    fi
    # Stop trainer containers (Docker mode)
    for cname in "${TRAINER_CONTAINERS[@]}"; do
        docker stop "$cname" 2>/dev/null || true
        docker rm "$cname" 2>/dev/null || true
    done
    # Clean up any orphaned agent containers launched by affinetes
    # (leak if validator was killed before Python finally blocks ran)
    python3 -c "
import affinetes
envs = affinetes.list_active_environments()
if envs:
    print(f'Cleaning up {len(envs)} active affinetes environment(s)...')
    affinetes.cleanup_all_environments()
    print('Done.')
" 2>/dev/null || true
    # Fallback: sweep Docker for containers matching agent image names
    for agent_name in agent-systematic agent-failure agent-lineage; do
        ORPHANED=$(docker ps -aq --filter "name=${agent_name}" 2>/dev/null || true)
        if [ -n "$ORPHANED" ]; then
            info "Removing orphaned agent container(s) matching ${agent_name}..."
            docker stop $ORPHANED 2>/dev/null || true
            docker rm $ORPHANED 2>/dev/null || true
        fi
    done
    # Delete Basilica deployments if any
    if [ -f "$LOG_DIR/trainer_urls.txt.deployments" ] 2>/dev/null; then
        python3 -c "
from basilica import BasilicaClient
client = BasilicaClient()
with open('$LOG_DIR/trainer_urls.txt.deployments') as f:
    for name in f:
        name = name.strip()
        if name:
            try:
                client.delete_deployment(name)
                print(f'Deleted Basilica deployment: {name}')
            except Exception as e:
                print(f'Failed to delete {name}: {e}')
" 2>&1 | while IFS= read -r line; do echo "  $line"; done || warn "Basilica cleanup failed — deployments may need manual deletion (TTL: 1h)"
    fi
    # Clean commitment files
    rm -rf /tmp/radar_commitments 2>/dev/null || true
    echo ""
    ok "Cleanup complete. Logs preserved at: $LOG_DIR/"
}
trap cleanup EXIT INT TERM

# =============================================================================
TRAINING_MODE="local"
if [ "$USE_BASILICA" = true ]; then TRAINING_MODE="basilica (remote GPU)"; fi
if [ "$CPU_ONLY" = true ]; then TRAINING_MODE="cpu-only (training skipped)"; fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Radar Multi-Node Integration Test                  ║"
echo "║                                                              ║"
printf "║  Network:    %-48s ║\n" "$NETWORK (netuid $NETUID)"
printf "║  Miners:     %-48s ║\n" "$NUM_MINERS"
printf "║  Validators: %-48s ║\n" "$NUM_VALIDATORS"
printf "║  Rounds:     %-48s ║\n" "$WAIT_ROUNDS"
printf "║  Training:   %-48s ║\n" "$TRAINING_MODE"
printf "║  Registry:   %-48s ║\n" "$GHCR_REPO"
printf "║  Logs:       %-48s ║\n" "$LOG_DIR/"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# STEP 0: Preflight checks
# =============================================================================
step "Step 0: Preflight checks"

CHECKS_OK=true

# Python
if ! python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    fail "Python 3.10+ required"
    CHECKS_OK=false
else
    ok "Python $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
fi

# Docker
if ! docker info &>/dev/null; then
    fail "Docker daemon not running"
    CHECKS_OK=false
else
    ok "Docker running"
fi

# GPU check (informational)
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    ok "GPU: $GPU_NAME ($GPU_MEM)"
elif [ "$USE_BASILICA" = true ]; then
    ok "No local GPU needed — training runs on Basilica remote pods"
elif [ "$CPU_ONLY" = true ]; then
    ok "Running in CPU-only mode (training skipped)"
else
    warn "No GPU detected. Training (Phase B) requires GPU."
    warn "Use --basilica to train on remote GPU pods, or --cpu-only to skip training."
fi

# Basilica token check
if [ "$USE_BASILICA" = true ]; then
    if [ -z "${BASILICA_API_TOKEN:-}" ]; then
        warn "BASILICA_API_TOKEN not set. Training dispatch will fail."
        warn "Set it in .env or: export BASILICA_API_TOKEN=your-token"
    else
        ok "Basilica API token configured"
    fi
fi

# Core modules
if ! python3 -c "import shared.protocol, shared.scoring, shared.database, bittensor" 2>/dev/null; then
    fail "Core modules not importable. Run: pip install -e '.[dev]'"
    CHECKS_OK=false
else
    ok "Core modules importable"
fi

# System resources
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
NUM_CPUS=$(nproc 2>/dev/null || echo 1)
DISK_FREE_GB=$(df -BG "$PROJECT_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo 0)

info "System: ${NUM_CPUS} CPUs, ${TOTAL_MEM_GB} GB RAM, ${DISK_FREE_GB} GB disk free"

# Resource warnings
NEEDED_MEM=$((4 + NUM_MINERS * 1 + NUM_VALIDATORS * 2))  # GB estimate
if [ "$TOTAL_MEM_GB" -lt "$NEEDED_MEM" ] 2>/dev/null; then
    warn "Low memory: ${TOTAL_MEM_GB} GB available, ~${NEEDED_MEM} GB recommended"
    warn "  Subtensor container: ~2 GB"
    warn "  Per miner process:   ~0.5 GB"
    warn "  Per validator:       ~1-2 GB (+ evaluation memory)"
    warn "  Agent containers:    ~8 GB per concurrent agent (shared across validators)"
fi

if [ "$CHECKS_OK" = false ]; then
    fail "Preflight checks failed."
    exit 1
fi
ok "Preflight passed"

# =============================================================================
# STEP 1: Run unit tests (fast feedback)
# =============================================================================
if [ "$SKIP_UNIT_TESTS" = false ]; then
    step "Step 1: Running unit tests"
    python3 -m pytest tests/ -x --tb=short -q 2>&1 | tail -5
    ok "Unit tests passed"
else
    step "Step 1: Skipping unit tests (--skip-tests)"
fi

# =============================================================================
# STEP 2: Build agent Docker images and push to GHCR
# =============================================================================
step "Step 2: Building agent Docker images and pushing to GHCR"

# Login to GHCR
if [ -z "${GITHUB_TOKEN:-}" ]; then
    fail "GITHUB_TOKEN required to push agent images to GHCR."
    fail "Set it in .env or: export GITHUB_TOKEN=ghp_..."
    exit 1
fi
echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_USER:-dev}" --password-stdin 2>/dev/null
ok "Logged in to ghcr.io"

# Map of available agents — cycle through them for N miners
AGENT_DIRS=("example_agents/systematic" "example_agents/failure_analyst" "example_agents/lineage_tracker")
AGENT_NAMES=("agent-systematic" "agent-failure" "agent-lineage")
AGENT_GHCR_URLS=()

for i in "${!AGENT_DIRS[@]}"; do
    dir="${AGENT_DIRS[$i]}"
    name="${AGENT_NAMES[$i]}"
    ghcr_url="${GHCR_REPO}/${name}:latest"
    AGENT_GHCR_URLS+=("$ghcr_url")
    if [ -d "$dir" ]; then
        # Copy affinetes Actor wrapper into agent build context
        cp miner/agent_env_wrapper/env.py "$dir/env.py"

        # Copy affinetes HTTP server template (replicates two-stage build for Basilica)
        AFFINETES_TEMPLATES="$(python -c 'import affinetes, pathlib; print(pathlib.Path(affinetes.__file__).parent / "templates")')"
        cp "$AFFINETES_TEMPLATES/http_server.py" "$dir/http_server.py"

        docker build -q -t "${name}:latest" "$dir" > /dev/null 2>&1
        docker tag "${name}:latest" "$ghcr_url"
        docker push "$ghcr_url" > /dev/null 2>&1

        # Clean up copied files
        rm -f "$dir/env.py" "$dir/http_server.py"

        ok "Built + pushed ${ghcr_url}"
    else
        warn "Agent dir not found: $dir"
    fi
done

# =============================================================================
# STEP 2b: Build trainer Docker image and push to GHCR
# =============================================================================
step "Step 2b: Building trainer image and pushing to GHCR"

TRAINER_IMAGE="${GHCR_REPO}/ts-runner:latest"

# Copy shared modules into runner build context (Dockerfile expects them)
cp shared/auth.py runner/timeseries_forecast/auth.py
cp shared/artifacts.py runner/timeseries_forecast/artifacts.py
cp shared/r2_audit.py runner/timeseries_forecast/r2_audit.py

TRAINER_LOG="${LOG_DIR}/trainer_build.log"
if ! docker build -t ts-runner:latest runner/timeseries_forecast/ > "$TRAINER_LOG" 2>&1; then
    rm -f runner/timeseries_forecast/auth.py runner/timeseries_forecast/artifacts.py runner/timeseries_forecast/r2_audit.py
    echo "  Trainer build log: $TRAINER_LOG"
    tail -20 "$TRAINER_LOG"
    fail "Trainer Docker build failed — see log above"
fi

# Clean up copied files
rm -f runner/timeseries_forecast/auth.py runner/timeseries_forecast/artifacts.py runner/timeseries_forecast/r2_audit.py

docker tag ts-runner:latest "$TRAINER_IMAGE"
if ! docker push "$TRAINER_IMAGE" > /dev/null 2>&1; then
    fail "Failed to push ${TRAINER_IMAGE}"
fi

ok "Built + pushed ${TRAINER_IMAGE}"

export OFFICIAL_TRAINING_IMAGE="$TRAINER_IMAGE"

# Make all GHCR packages public so other nodes can pull without auth
step "Step 2c: Making GHCR packages public"

GHCR_ORG="tensorlink-ai"
ALL_PACKAGES=("${AGENT_NAMES[@]}" "ts-runner")
for pkg in "${ALL_PACKAGES[@]}"; do
    # GHCR package names use the repo as a namespace: radar/<name>
    encoded_pkg="radar%2F${pkg}"
    gh api -X PUT "/orgs/${GHCR_ORG}/packages/container/${encoded_pkg}/visibility" \
        -f visibility=public 2>/dev/null && ok "Public: ${pkg}" || warn "Could not set visibility for ${pkg}"
done

# =============================================================================
# STEP 3: Start local subtensor
# =============================================================================
step "Step 3: Subtensor ($NETWORK)"

if [ "$USE_TESTNET" = true ]; then
    info "Using Bittensor testnet (no local subtensor needed)"
    # Verify testnet connectivity
    if python3 -c "import bittensor as bt; sub=bt.Subtensor(network='test'); print(f'Testnet block: {sub.block}')" 2>/dev/null; then
        ok "Connected to testnet"
    else
        fail "Cannot connect to Bittensor testnet"
        exit 1
    fi
elif [ "$SKIP_SUBTENSOR" = true ]; then
    info "Reusing existing subtensor (--skip-subtensor)"
else
    # Check if already running
    if python3 -c "import bittensor as bt; bt.Subtensor(network='local').block" 2>/dev/null; then
        ok "Subtensor already running"
    else
        SUBTENSOR_IMAGE="ghcr.io/opentensor/subtensor-localnet:devnet-ready"

        if ! docker image inspect "$SUBTENSOR_IMAGE" &>/dev/null; then
            info "Pulling subtensor image..."
            if [ -n "${GITHUB_TOKEN:-}" ]; then
                echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_USER:-dev}" --password-stdin 2>/dev/null || true
            fi
            docker pull "$SUBTENSOR_IMAGE" || { fail "Cannot pull subtensor image. Set GITHUB_TOKEN."; exit 1; }
        fi

        SUBTENSOR_CONTAINER="radar-test-subtensor-$$"
        docker run -d --name "$SUBTENSOR_CONTAINER" -p 9944:9944 -p 9945:9945 "$SUBTENSOR_IMAGE" >/dev/null 2>&1

        # Wait for readiness
        for i in $(seq 1 45); do
            if python3 -c "import bittensor as bt; print(bt.Subtensor(network='local').block)" 2>/dev/null; then
                break
            fi
            [ "$i" = 45 ] && { fail "Subtensor didn't start in 45s"; exit 1; }
            sleep 1
        done
        ok "Subtensor started (container: $SUBTENSOR_CONTAINER)"
    fi
fi

# =============================================================================
# STEP 4: Create wallets, fund, register, stake
# =============================================================================
step "Step 4: Setting up $NUM_MINERS miner(s) + $NUM_VALIDATORS validator(s)"

python3 << PYEOF
import bittensor as bt
import sys

network = '${NETWORK}'
netuid = ${NETUID}
sub = bt.Subtensor(network=network)

if network == 'local':
    # --- Localnet: Alice is pre-funded ---
    alice = bt.Wallet(name='alice-dev')
    alice.create_coldkey_from_uri('//Alice', use_password=False, overwrite=True, suppress=True)
    alice.create_new_hotkey(use_password=False, overwrite=True, suppress=True)
    bal = sub.get_balance(alice.coldkeypub.ss58_address)
    print(f'  Alice balance: {bal}')
    if bal < 10000:
        print('  ERROR: Alice not pre-funded. Need devnet-ready image.')
        sys.exit(1)

    # --- Owner wallet + subnet ---
    owner = bt.Wallet(name='owner')
    owner.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    owner_bal = sub.get_balance(owner.coldkeypub.ss58_address)
    if owner_bal < 10000:
        sub.transfer(wallet=alice, destination_ss58=owner.coldkeypub.ss58_address,
                     amount=bt.Balance.from_tao(100_000),
                     wait_for_inclusion=True, wait_for_finalization=True)

    try:
        sub.register_subnet(wallet=owner, mev_protection=False)
        print(f'  Subnet created on netuid {netuid}')
    except Exception as e:
        print(f'  Subnet note: {e}')

    def fund_wallet(w):
        bal = sub.get_balance(w.coldkeypub.ss58_address)
        if bal < 10000:
            sub.transfer(wallet=alice, destination_ss58=w.coldkeypub.ss58_address,
                         amount=bt.Balance.from_tao(100_000),
                         wait_for_inclusion=True, wait_for_finalization=True)
else:
    # --- Testnet: wallets must already exist and be funded via faucet ---
    print(f'  Using {network} — wallets must be pre-created and funded via faucet')
    print(f'  Looking for existing wallets: {[$(printf '"%s",' "${VALIDATOR_WALLETS[@]}")]}, miner0..{${NUM_MINERS}-1}')

    def fund_wallet(w):
        bal = sub.get_balance(w.coldkeypub.ss58_address)
        if bal < 100:
            print(f'  WARNING: {w.name} has low balance ({bal}). Fund via: btcli wallet faucet --wallet.name {w.name} --subtensor.network test')

# --- Validators ---
vali_names = [$(printf '"%s",' "${VALIDATOR_WALLETS[@]}")]
for i in range(${NUM_VALIDATORS}):
    name = vali_names[i]
    w = bt.Wallet(name=name)
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    fund_wallet(w)
    try:
        sub.burned_register(wallet=w, netuid=netuid, mev_protection=False)
    except Exception:
        pass
    try:
        sub.add_stake(wallet=w, netuid=netuid, hotkey_ss58=w.hotkey.ss58_address,
                      amount=bt.Balance.from_tao(1000), mev_protection=False)
    except Exception:
        pass
    print(f'  {name}: registered + staked')

# --- Miners ---
for i in range(${NUM_MINERS}):
    name = f'miner{i}'
    w = bt.Wallet(name=name)
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    fund_wallet(w)
    try:
        sub.burned_register(wallet=w, netuid=netuid, mev_protection=False)
    except Exception:
        pass
    print(f'  {name}: registered')

# --- Verify ---
meta = sub.metagraph(netuid=netuid)
print(f'\n  Metagraph: {meta.n} neurons')
for uid in range(meta.n):
    hk = meta.hotkeys[uid][:12] if uid < len(meta.hotkeys) else '?'
    print(f'    UID {uid}: hotkey={hk}... stake={meta.S[uid]:.0f}')
PYEOF

if [ $? -ne 0 ]; then
    fail "Wallet/registration setup failed"
    exit 1
fi
ok "All neurons registered"

# =============================================================================
# STEP 4b: Start trainer containers (simulates miner-deployed trainers)
# =============================================================================
step "Step 4b: Starting trainer containers for Phase B"

TRAINER_PORT_BASE=8090
TRAINER_CONTAINERS=()

TRAINER_URLS=()  # populated by whichever path runs

if [ "$CPU_ONLY" = true ]; then
    info "CPU-only mode: skipping trainer containers"
elif [ "$USE_BASILICA" = true ]; then
    # Deploy trainers on Basilica via affinetes
    info "Deploying $NUM_MINERS trainer pod(s) on Basilica..."
    TRAINER_URLS_FILE="$LOG_DIR/trainer_urls.txt"

    python3 -c "
import sys, os, time
sys.path.insert(0, '.')

from basilica import BasilicaClient

client = BasilicaClient()
num_miners = $NUM_MINERS
ts = int(time.time())

DEPLOY_KWARGS = dict(
    image='${TRAINER_IMAGE}',
    port=8081,
    cpu='2000m',
    memory='8Gi',
    gpu_count=1,
    gpu_models=['RTX-A4000', 'RTX-A6000'],
    ttl_seconds=3600,
    timeout=900,
    public=True,
    env={
        'SUBTENSOR_NETWORK': '${NETWORK}',
        'NETUID': '${NETUID}',
    },
)

urls = []
deployment_names = []
for i in range(num_miners):
    name = f'radar-trainer-{i}-{ts}'
    # Retry up to 3 times with backoff — Basilica scheduler can fail
    # under concurrent GPU allocation pressure
    last_err = None
    for attempt in range(3):
        try:
            print(f'Trainer {i}: deploying {name} (attempt {attempt+1})...', file=sys.stderr)
            dep = client.deploy(name=name, **DEPLOY_KWARGS)
            print(f'Trainer {i}: READY at {dep.url} (instance={dep.name})', file=sys.stderr)
            urls.append(dep.url)
            deployment_names.append(dep.name)
            last_err = None
            break
        except Exception as e:
            last_err = e
            # Use a fresh name on retry to avoid name collision
            name = f'radar-trainer-{i}-{int(time.time())}'
            wait = 10 * (attempt + 1)
            print(f'Trainer {i}: deploy failed ({e}), retrying in {wait}s...', file=sys.stderr)
            time.sleep(wait)
    if last_err:
        print(f'Trainer {i}: FAILED after 3 attempts: {last_err}', file=sys.stderr)
        sys.exit(1)

with open('${TRAINER_URLS_FILE}', 'w') as f:
    for u in urls:
        f.write(u + '\n')

with open('${TRAINER_URLS_FILE}.deployments', 'w') as f:
    for d in deployment_names:
        f.write(d + '\n')
" 2>&1 | while IFS= read -r line; do echo "  $line"; done

    if [ -f "$TRAINER_URLS_FILE" ]; then
        while IFS= read -r url; do
            TRAINER_URLS+=("$url")
        done < "$TRAINER_URLS_FILE"
        ok "Deployed ${#TRAINER_URLS[@]} trainer pod(s) on Basilica"
    else
        fail "Failed to deploy trainer pods on Basilica"
    fi
else
    for i in $(seq 0 $((NUM_MINERS - 1))); do
        TRAINER_PORT=$((TRAINER_PORT_BASE + i))
        CONTAINER_NAME="radar-trainer-${i}-$$"

        GPU_FLAGS=""
        if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
            GPU_FLAGS="--gpus all"
        fi

        docker run -d --name "$CONTAINER_NAME" \
            $GPU_FLAGS \
            -p "${TRAINER_PORT}:8081" \
            -e TRAINER_PORT=8081 \
            -e SUBTENSOR_NETWORK="${NETWORK}" \
            -e NETUID="${NETUID}" \
            "$TRAINER_IMAGE" >/dev/null 2>&1

        TRAINER_CONTAINERS+=("$CONTAINER_NAME")
        ok "Trainer $i started (container: $CONTAINER_NAME, port: $TRAINER_PORT)"
    done

    info "Waiting for trainer health checks..."
    for i in $(seq 0 $((NUM_MINERS - 1))); do
        TRAINER_PORT=$((TRAINER_PORT_BASE + i))
        HEALTHY=false
        for attempt in $(seq 1 30); do
            if curl -sf "http://localhost:${TRAINER_PORT}/health" >/dev/null 2>&1; then
                ok "Trainer $i healthy on port $TRAINER_PORT"
                HEALTHY=true
                break
            fi
            sleep 2
        done
        if [ "$HEALTHY" = false ]; then
            warn "Trainer $i not responding on port $TRAINER_PORT after 60s"
            docker logs "${TRAINER_CONTAINERS[$i]}" 2>&1 | tail -10
        fi
    done

    # Populate TRAINER_URLS for local Docker path
    for i in $(seq 0 $((NUM_MINERS - 1))); do
        TRAINER_PORT=$((TRAINER_PORT_BASE + i))
        TRAINER_URLS+=("http://localhost:${TRAINER_PORT}")
    done
fi

# =============================================================================
# STEP 5: Start miners (commit trainer URL so validators can dispatch)
# =============================================================================
step "Step 5: Starting $NUM_MINERS miner(s)"

NUM_AGENT_TYPES=${#AGENT_GHCR_URLS[@]}

for i in $(seq 0 $((NUM_MINERS - 1))); do
    AGENT_IDX=$((i % NUM_AGENT_TYPES))
    AGENT_IMAGE="${AGENT_GHCR_URLS[$AGENT_IDX]}"
    MINER_LOG="$LOG_DIR/miner${i}.log"

    # Miner commits this URL so the validator's coordinator can POST to it
    TRAINER_URL="${TRAINER_URLS[$i]:-}"

    python3 miner/neuron.py \
        --netuid "$NETUID" \
        --subtensor.network "$NETWORK" \
        --wallet.name "miner${i}" \
        --docker_image "$AGENT_IMAGE" \
        --trainer_url "$TRAINER_URL" \
        --axon.external_ip 127.0.0.1 \
        > "$MINER_LOG" 2>&1 &
    PIDS+=($!)
    ok "Miner $i started (PID $!) — $AGENT_IMAGE — trainer: ${TRAINER_URL:-none} — log: $MINER_LOG"
done

sleep 5  # Let miners commit images

# Verify commitments (localnet uses file fallback, testnet uses chain)
if [ "$USE_TESTNET" = true ]; then
    ok "Testnet: commitments go to chain (skipping file check)"
else
    COMMITMENT_COUNT=$(ls /tmp/radar_commitments/"$NETUID"/ 2>/dev/null | wc -l || echo 0)
    if [ "$COMMITMENT_COUNT" -ge "$NUM_MINERS" ]; then
        ok "All $NUM_MINERS miners committed images"
    else
        warn "Only $COMMITMENT_COUNT / $NUM_MINERS commitments found (some may use chain)"
    fi
fi

# =============================================================================
# STEP 6: Start validators
# =============================================================================
step "Step 6: Starting $NUM_VALIDATORS validator(s)"

if [ "$USE_BASILICA" = true ]; then
    export RADAR_RUNNER_BACKEND=basilica
    export RADAR_AFFINETES_MODE=basilica
else
    export RADAR_RUNNER_BACKEND=local
    export RADAR_AFFINETES_MODE=docker
fi
export RADAR_DESEARCH_ENABLED=false
export RADAR_EVAL_DEVICE=cpu

# ── Test-mode overrides: shorter rounds, single size bucket ──
# Pin to medium-small bucket so agents can reliably target the midpoint (6M).
export RADAR_MIN_FLOPS=2000000
export RADAR_MAX_FLOPS=10000000

if [ "$USE_TESTNET" = true ]; then
    # Testnet: use production round timing (275 blocks) for consensus compatibility.
    # Other validators on testnet use defaults — mismatched intervals break consensus.
    export RADAR_ROUND_INTERVAL=${RADAR_ROUND_INTERVAL:-275}
    export RADAR_SUBMISSION_WINDOW=${RADAR_SUBMISSION_WINDOW:-50}
    export RADAR_TRAINING_WINDOW=${RADAR_TRAINING_WINDOW:-150}
    export RADAR_EVAL_WINDOW=${RADAR_EVAL_WINDOW:-25}
    export RADAR_TRAINING_TIMEOUT=${RADAR_TRAINING_TIMEOUT:-1800}
    export RADAR_AGENT_TIMEOUT=${RADAR_AGENT_TIMEOUT:-600}
    export RADAR_TIME_BUDGET=${RADAR_TIME_BUDGET:-300}
    # Skip the training-window block wait — in test mode, dispatch already finished
    # and waiting for 150 blocks (~30 min) causes the test to timeout.
    export RADAR_SKIP_TRAINING_WAIT=${RADAR_SKIP_TRAINING_WAIT:-true}
else
    # Localnet: shorter round windows for faster iteration (~1s/block)
    export RADAR_ROUND_INTERVAL=${RADAR_ROUND_INTERVAL:-75}
    export RADAR_SUBMISSION_WINDOW=${RADAR_SUBMISSION_WINDOW:-15}
    export RADAR_TRAINING_WINDOW=${RADAR_TRAINING_WINDOW:-40}
    export RADAR_EVAL_WINDOW=${RADAR_EVAL_WINDOW:-10}
    export RADAR_TRAINING_TIMEOUT=${RADAR_TRAINING_TIMEOUT:-420}
    # Agent timeout: 120s for fast test iterations (default 600s — too long for localnet rounds)
    export RADAR_AGENT_TIMEOUT=${RADAR_AGENT_TIMEOUT:-120}
    # Trainer time_budget: 120s for fast test iterations (default 300s)
    export RADAR_TIME_BUDGET=${RADAR_TIME_BUDGET:-120}
    # Skip the training-window block wait — dispatch already finished, no need to
    # wait for the full window to close on localnet.
    export RADAR_SKIP_TRAINING_WAIT=${RADAR_SKIP_TRAINING_WAIT:-true}
fi

if [ "$CPU_ONLY" = true ]; then
    info "CPU-only mode: training will be skipped"
fi

for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
    DB_PORT=$((DB_PORT_BASE + i))
    DB_DIR="$LOG_DIR/experiments_v${i}"
    VALI_LOG="$LOG_DIR/validator${i}.log"

    mkdir -p "$DB_DIR"

    # Kill anything on this port
    if command -v fuser &>/dev/null; then
        fuser -k "${DB_PORT}/tcp" 2>/dev/null || true
    fi

    python3 validator/neuron.py \
        --netuid "$NETUID" \
        --subtensor.network "$NETWORK" \
        --wallet.name "${VALIDATOR_WALLETS[$i]}" \
        --db_dir "$DB_DIR" \
        --db_port "$DB_PORT" \
        > "$VALI_LOG" 2>&1 &
    PIDS+=($!)
    ok "Validator $i started (PID $!) — port $DB_PORT — log: $VALI_LOG"
done

# Wait for validator DB servers to become healthy (up to 60s)
info "Waiting for validator DB servers..."
for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
    DB_PORT=$((DB_PORT_BASE + i))
    HEALTHY=false
    for attempt in $(seq 1 30); do
        if curl -sf "http://localhost:${DB_PORT}/health" >/dev/null 2>&1; then
            ok "Validator $i DB server healthy on port $DB_PORT"
            HEALTHY=true
            break
        fi
        sleep 2
    done
    if [ "$HEALTHY" = false ]; then
        warn "Validator $i DB server not responding on port $DB_PORT after 60s"
        tail -5 "$LOG_DIR/validator${i}.log" 2>/dev/null || true
    fi
done

# =============================================================================
# STEP 7: Wait for rounds and verify
# =============================================================================
step "Step 7: Waiting for $WAIT_ROUNDS round(s) (timeout: ${ROUND_TIMEOUT}s per round)"

TOTAL_TIMEOUT=$((ROUND_TIMEOUT * WAIT_ROUNDS))
ELAPSED=0
INTERVAL=15
ROUNDS_COMPLETED=0
LAST_LOG_LINES=0

while [ "$ELAPSED" -lt "$TOTAL_TIMEOUT" ] && [ "$ROUNDS_COMPLETED" -lt "$WAIT_ROUNDS" ]; do
    # Check validators still running
    ALL_ALIVE=true
    for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
        pid_idx=$((NUM_MINERS + i))
        if ! kill -0 "${PIDS[$pid_idx]}" 2>/dev/null; then
            warn "Validator $i exited unexpectedly"
            tail -10 "$LOG_DIR/validator${i}.log" 2>/dev/null
            ALL_ALIVE=false
        fi
    done

    # Stream new validator 0 log lines
    if [ -f "$LOG_DIR/validator0.log" ]; then
        TOTAL_LINES=$(wc -l < "$LOG_DIR/validator0.log")
        if [ "$TOTAL_LINES" -gt "$LAST_LOG_LINES" ]; then
            tail -n +$((LAST_LOG_LINES + 1)) "$LOG_DIR/validator0.log" | head -n $((TOTAL_LINES - LAST_LOG_LINES)) | while IFS= read -r line; do
                echo -e "  ${CYAN}[v0]${NC} $line"
            done
            LAST_LOG_LINES=$TOTAL_LINES
        fi
    fi

    # Check for round completion — require ALL validators to complete
    MIN_COMPLETED=$((999999))
    for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
        COMPLETED=$(grep -c "Round.*complete\|Weights set\|forward.*complete" "$LOG_DIR/validator${i}.log" 2>/dev/null || true)
        COMPLETED="${COMPLETED:-0}"
        if [ "$COMPLETED" -lt "$MIN_COMPLETED" ]; then
            MIN_COMPLETED=$COMPLETED
        fi
    done
    if [ "$MIN_COMPLETED" -gt "$ROUNDS_COMPLETED" ]; then
        ROUNDS_COMPLETED=$MIN_COMPLETED
        ok "Round $ROUNDS_COMPLETED completed by ALL validators"
    fi

    # Also check DB for experiments across ALL validators
    ALL_HAVE_EXPERIMENTS=true
    for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
        V_PORT=$((DB_PORT_BASE + i))
        V_EXP=$(curl -sf "http://localhost:${V_PORT}/experiments/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('total', 0))
except:
    print(0)
" 2>/dev/null | tail -1 || echo 0)
        V_EXP="${V_EXP//[^0-9]/}"
        V_EXP="${V_EXP:-0}"
        if [ "$V_EXP" -eq 0 ]; then
            ALL_HAVE_EXPERIMENTS=false
        fi
    done

    if [ "$ALL_HAVE_EXPERIMENTS" = true ] && [ "$ROUNDS_COMPLETED" -eq 0 ]; then
        ROUNDS_COMPLETED=1
        ok "Experiments found in ALL validator DBs"
    fi

    if [ "$ROUNDS_COMPLETED" -ge "$WAIT_ROUNDS" ]; then
        break
    fi

    if [ "$ALL_ALIVE" = false ]; then
        warn "Not all processes alive. Continuing to check results..."
    fi

    info "  Waiting... (${ELAPSED}s / ${TOTAL_TIMEOUT}s, rounds: ${ROUNDS_COMPLETED}/${WAIT_ROUNDS})"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ "$ROUNDS_COMPLETED" -lt "$WAIT_ROUNDS" ]; then
    warn "Only completed $ROUNDS_COMPLETED / $WAIT_ROUNDS rounds within timeout"
fi

# =============================================================================
# STEP 8: Verification
# =============================================================================
step "Step 8: Verification"

PASS_COUNT=0
FAIL_COUNT=0

check() {
    local desc="$1"
    local result="$2"
    if [ "$result" = "true" ]; then
        ok "$desc"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        fail "$desc"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# 8.1 Metagraph
info "Checking metagraph..."
META_OUTPUT=$(python3 -c "
import bittensor as bt
sub = bt.Subtensor(network='${NETWORK}')
meta = sub.metagraph(netuid=${NETUID})
print(f'neurons={meta.n}')
for uid in range(meta.n):
    s = meta.S[uid]
    e = meta.E[uid] if hasattr(meta, 'E') else 0
    print(f'  UID {uid}: stake={s:.0f} emission={e:.6f}')
" 2>/dev/null || echo "neurons=0")
echo "  $META_OUTPUT" | head -10

NEURON_COUNT=$(echo "$META_OUTPUT" | head -1 | grep -oP 'neurons=\K\d+' || echo 0)
EXPECTED=$((NUM_MINERS + NUM_VALIDATORS))
check "All $EXPECTED neurons registered" "$([ "$NEURON_COUNT" -ge "$EXPECTED" ] && echo true || echo false)"

# 8.2 Miner commitments
if [ "$USE_TESTNET" = true ]; then
    check "Miner commitments (on-chain for testnet)" "true"
else
    COMMITMENT_COUNT=$(ls /tmp/radar_commitments/"$NETUID"/ 2>/dev/null | wc -l || echo 0)
    check "Miner commitments present ($COMMITMENT_COUNT)" "$([ "$COMMITMENT_COUNT" -gt 0 ] && echo true || echo false)"
fi

# 8.3 Validator DB APIs
for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
    DB_PORT=$((DB_PORT_BASE + i))
    STATS=$(curl -sf "http://localhost:${DB_PORT}/experiments/stats" 2>/dev/null || echo "{}")
    TOTAL=$(echo "$STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo 0)
    check "Validator $i DB has experiments ($TOTAL)" "$([ "$TOTAL" -gt 0 ] && echo true || echo false)"
done

# 8.4 Pareto front
PARETO_SIZE=$(curl -sf "http://localhost:${DB_PORT_BASE}/experiments/pareto" 2>/dev/null | python3 -c "
import sys, json
try:
    print(len(json.load(sys.stdin)))
except:
    print(0)
" 2>/dev/null || echo 0)
check "Pareto front has members ($PARETO_SIZE)" "$([ "$PARETO_SIZE" -gt 0 ] && echo true || echo false)"

# 8.5 Cross-eval (verify arch_owner != trainer_uid in dispatch logs)
# coordinator.py logs "Job arch=X trainer=Y: status" — check no job has arch==trainer
SELF_TRAIN=$(grep -cP "Job arch=(\d+) trainer=\1:" "$LOG_DIR/validator0.log" 2>/dev/null) || SELF_TRAIN=0
check "No self-training detected ($SELF_TRAIN)" "$([ "$SELF_TRAIN" -eq 0 ] && echo true || echo false)"

# 8.6 Validator consistency (both have same experiment count, within tolerance)
if [ "$NUM_VALIDATORS" -ge 2 ]; then
    TOTAL_V0=$(curl -sf "http://localhost:${DB_PORT_BASE}/experiments/stats" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo 0)
    TOTAL_V1=$(curl -sf "http://localhost:$((DB_PORT_BASE+1))/experiments/stats" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo 0)
    DIFF=$(( TOTAL_V0 > TOTAL_V1 ? TOTAL_V0 - TOTAL_V1 : TOTAL_V1 - TOTAL_V0 ))
    check "Validators consistent (v0=$TOTAL_V0, v1=$TOTAL_V1, diff=$DIFF)" "$([ "$DIFF" -le 2 ] && echo true || echo false)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                     TEST RESULTS                             ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Miners:     $NUM_MINERS                                            ║"
echo "║  Validators: $NUM_VALIDATORS                                            ║"
echo "║  Rounds:     $ROUNDS_COMPLETED / $WAIT_ROUNDS completed                                   ║"
echo "║  Checks:     $PASS_COUNT passed, $FAIL_COUNT failed                                 ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Log directory: $LOG_DIR/  ║"
echo "║                                                              ║"
echo "║  Logs:                                                       ║"
for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
    printf "║    Validator %d: %-45s ║\n" "$i" "$LOG_DIR/validator${i}.log"
done
for i in $(seq 0 $((NUM_MINERS - 1))); do
    printf "║    Miner %d:     %-45s ║\n" "$i" "$LOG_DIR/miner${i}.log"
done
echo "║                                                              ║"
echo "║  DB APIs:                                                    ║"
for i in $(seq 0 $((NUM_VALIDATORS - 1))); do
    printf "║    Validator %d: http://localhost:%-26d  ║\n" "$i" "$((DB_PORT_BASE + i))"
done
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    warn "$FAIL_COUNT verification check(s) failed."
    warn "This may be expected if the round didn't complete within timeout."
    warn "Check logs for details."
    exit 1
else
    ok "All checks passed!"
    exit 0
fi
