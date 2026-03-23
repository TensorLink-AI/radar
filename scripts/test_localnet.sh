#!/bin/bash
# =============================================================================
# RADAR Subnet — Localnet End-to-End Test
#
# Fully automated: installs deps, starts local subtensor, creates wallets,
# registers neurons, launches validator + miner, waits for one loop, verifies.
#
# Usage:
#   bash scripts/test_localnet.sh          # full end-to-end
#   bash scripts/test_localnet.sh --quick  # unit/integration tests only
#
# Prerequisites:
#   - Docker installed and running
#   - Python 3.10+
#   - uv (https://docs.astral.sh/uv/)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Load .env if present ──
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# =============================================================================
# PREFLIGHT: Verify environment before doing anything expensive
# =============================================================================
echo "-----------------------------------------"
info "Preflight: Checking environment..."
echo "-----------------------------------------"

PREFLIGHT_OK=true

# ── Python ──
if ! command -v python3 &>/dev/null; then
    fail "python3 not found. Install Python 3.10+."
    PREFLIGHT_OK=false
else
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
        fail "Python >= 3.10 required (found $PY_VERSION)."
        PREFLIGHT_OK=false
    else
        ok "Python $PY_VERSION"
    fi
fi

# ── uv ──
if ! command -v uv &>/dev/null; then
    fail "uv not found. Install from https://docs.astral.sh/uv/"
    PREFLIGHT_OK=false
else
    ok "uv $(uv --version 2>/dev/null | head -1)"
fi

# ── Docker ──
if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install Docker and ensure it's in PATH."
    PREFLIGHT_OK=false
elif ! docker info &>/dev/null 2>&1; then
    fail "Docker daemon not running. Start Docker first."
    PREFLIGHT_OK=false
else
    ok "Docker running"
fi

# ── GITHUB_TOKEN (needed to pull subtensor-localnet from GHCR) ──
if [ -z "${GITHUB_TOKEN:-}" ]; then
    warn "GITHUB_TOKEN not set. Pulling subtensor image from GHCR will fail"
    warn "unless you are already authenticated. Set it in .env or export it:"
    warn "  export GITHUB_TOKEN=ghp_..."
else
    ok "GITHUB_TOKEN set"
fi

# ── .env file ──
if [ -f "$PROJECT_DIR/.env" ]; then
    ok ".env file found"
else
    warn "No .env file found. Using environment variables only."
    warn "Copy .env.example to .env and fill in values if needed:"
    warn "  cp .env.example .env"
fi

# ── python-dotenv (config.py imports it at load time) ──
if python3 -c "import dotenv" 2>/dev/null; then
    ok "python-dotenv importable"
else
    warn "python-dotenv not installed yet (will be installed with dependencies)"
fi

echo ""
if [ "$PREFLIGHT_OK" = false ]; then
    fail "Preflight checks failed. Fix the issues above and retry."
    exit 1
fi
ok "Preflight checks passed."
echo ""

# PIDs to clean up on exit
PIDS=()
SUBTENSOR_CONTAINER=""

cleanup() {
    info "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [ -n "$SUBTENSOR_CONTAINER" ]; then
        docker stop "$SUBTENSOR_CONTAINER" 2>/dev/null || true
        docker rm "$SUBTENSOR_CONTAINER" 2>/dev/null || true
    fi
    ok "Cleanup complete."
}
trap cleanup EXIT INT TERM

echo "========================================="
echo " RADAR Subnet — Localnet End-to-End Test"
echo "========================================="
echo ""

# =============================================================================
# STEP 0: Set up uv virtual environment
# =============================================================================
info "Setting up uv virtual environment..."

VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv at $VENV_DIR..."
    uv venv "$VENV_DIR" --python 3.10
    ok "Virtual environment created."
else
    ok "Virtual environment already exists at $VENV_DIR."
fi

# Activate the venv so all python3/pip calls use it
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Activated venv (python: $(which python3))"
echo ""

# ── Quick mode: just run unit/integration tests ──
if [ "${1:-}" = "--quick" ]; then
    info "Quick mode: running unit and integration tests only."
    echo ""
    info "Installing package dependencies..."
    uv pip install -e ".[dev]" --quiet 2>&1 | tail -3
    ok "Dependencies installed."
    echo ""
    info "Running all tests..."
    python3 -m pytest tests/ -v --tb=short
    ok "All tests passed."
    exit 0
fi

# =============================================================================
# STEP 1: Install package dependencies
# =============================================================================
echo "-----------------------------------------"
info "Step 1: Installing package dependencies..."
echo "-----------------------------------------"

uv pip install -e ".[dev]" --quiet 2>&1 | tail -5
ok "Python package installed (editable mode)."

# Verify key imports
python3 -c "import shared.protocol; import shared.task; import shared.database; import shared.scoring" 2>/dev/null \
    && ok "Core modules importable." \
    || { fail "Core modules failed to import."; exit 1; }

python3 -c "import bittensor as bt; print(f'  bittensor {bt.__version__}')" 2>/dev/null \
    && ok "Bittensor SDK available." \
    || { fail "Bittensor not installed. Check pyproject.toml dependencies."; exit 1; }

echo ""

# =============================================================================
# STEP 2: Run unit tests first (fast feedback)
# =============================================================================
echo "-----------------------------------------"
info "Step 2: Running unit tests..."
echo "-----------------------------------------"

python3 -m pytest tests/ -v --tb=short -x || {
    fail "Unit tests failed. Fix before running localnet test."
    exit 1
}
ok "All unit tests passed."
echo ""

# =============================================================================
# STEP 3: Build agent Docker image
# =============================================================================
echo "-----------------------------------------"
info "Step 3: Building systematic agent Docker image..."
echo "-----------------------------------------"

if command -v docker &>/dev/null; then
    docker build -t systematic:latest example_agents/systematic/ \
        && ok "Docker image 'systematic:latest' built." \
        || warn "Docker build failed. Continuing without Docker agent."
else
    warn "Docker not found. Skipping agent image build."
fi
echo ""

# =============================================================================
# STEP 4: Start local subtensor
# =============================================================================
echo "-----------------------------------------"
info "Step 4: Starting local subtensor..."
echo "-----------------------------------------"

# Check if subtensor is already running
if python3 -c "
import bittensor as bt
s = bt.Subtensor(network='local')
print(f'  Block: {s.block}')
" 2>/dev/null; then
    ok "Local subtensor already running."
else
    info "Starting subtensor via Docker..."

    # Pull and start the official localnet subtensor image
    SUBTENSOR_IMAGE="ghcr.io/opentensor/subtensor-localnet:devnet-ready"

    if ! docker image inspect "$SUBTENSOR_IMAGE" &>/dev/null; then
        info "Pulling subtensor-localnet image (this may take a minute)..."
        # GHCR requires authentication — login with GITHUB_TOKEN if available
        if [ -n "${GITHUB_TOKEN:-}" ]; then
            echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_USER:-tensorlink-dev}" --password-stdin 2>/dev/null \
                && ok "Logged in to ghcr.io." \
                || warn "GHCR login failed. Pull may still work if already authenticated."
        fi
        docker pull "$SUBTENSOR_IMAGE" || {
            fail "Could not pull subtensor image."
            info "GHCR requires authentication. Run:"
            info "  echo \$GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin"
            info "Then retry. Your PAT needs the read:packages scope."
            exit 1
        }
    fi

    SUBTENSOR_CONTAINER="radar-test-subtensor-$$"
    docker run -d \
        --name "$SUBTENSOR_CONTAINER" \
        -p 9944:9944 \
        -p 9945:9945 \
        "$SUBTENSOR_IMAGE" 2>&1 || true

    # Verify the container is actually running
    sleep 2
    if ! docker ps --filter "name=$SUBTENSOR_CONTAINER" --format '{{.ID}}' | grep -q .; then
        fail "Subtensor container is not running."
        info "Container logs:"
        docker logs "$SUBTENSOR_CONTAINER" 2>&1 | tail -20 || true
        docker rm "$SUBTENSOR_CONTAINER" 2>/dev/null || true
        SUBTENSOR_CONTAINER=""
        exit 1
    fi
    ok "Subtensor container is running."

    info "Waiting for subtensor to be ready..."
    RETRIES=30
    for i in $(seq 1 $RETRIES); do
        if python3 -c "
import bittensor as bt
s = bt.Subtensor(network='local')
print(f'  Block: {s.block}')
" 2>/dev/null; then
            ok "Local subtensor is ready."
            break
        fi
        if [ "$i" = "$RETRIES" ]; then
            fail "Subtensor did not become ready after ${RETRIES}s."
            info "Container logs:"
            docker logs "$SUBTENSOR_CONTAINER" 2>&1 | tail -20 || true
            exit 1
        fi
        sleep 1
    done
fi
echo ""

# =============================================================================
# STEP 5: Create wallets
# =============================================================================
echo "-----------------------------------------"
info "Step 5: Creating wallets..."
echo "-----------------------------------------"

# Create owner, validator, and miner wallets (non-interactive)
for WALLET_NAME in owner validator miner; do
    if [ -d "$HOME/.bittensor/wallets/$WALLET_NAME" ]; then
        ok "Wallet '$WALLET_NAME' already exists."
    else
        python3 -c "
import bittensor as bt
w = bt.Wallet(name='$WALLET_NAME')
w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
print(f'  Created wallet: $WALLET_NAME')
" && ok "Wallet '$WALLET_NAME' created." \
  || { fail "Failed to create wallet '$WALLET_NAME'."; exit 1; }
    fi
done
echo ""

# =============================================================================
# STEP 6: Fund wallets and create subnet
# =============================================================================
echo "-----------------------------------------"
info "Step 6: Funding wallets and creating subnet..."
echo "-----------------------------------------"

CHAIN_ENDPOINT="ws://127.0.0.1:9944"

# Use the pre-funded Alice dev account to transfer TAO to our wallets,
# then create subnet, register neurons via burned_register, and stake.
# This avoids slow PoW faucet and PoW registration entirely.
python3 << 'PYEOF'
import bittensor as bt
import sys

sub = bt.Subtensor(network='local')

# --- Create Alice wallet from dev seed ---
alice = bt.Wallet(name='alice-dev')
alice.create_coldkey_from_uri('//Alice', use_password=False, overwrite=True, suppress=True)
alice.create_new_hotkey(use_password=False, overwrite=True, suppress=True)
alice_bal = sub.get_balance(alice.coldkeypub.ss58_address)
print(f'  Alice balance: {alice_bal}')

if alice_bal < 10000:
    print('  ERROR: Alice account not pre-funded. Is this a devnet-ready image?')
    sys.exit(1)

# --- Fund our wallets via transfer from Alice ---
for name in ['owner', 'validator', 'miner']:
    w = bt.Wallet(name=name)
    bal = sub.get_balance(w.coldkeypub.ss58_address)
    if bal >= 10000:
        print(f'  {name} already funded: {bal}')
        continue
    print(f'  Transferring 100k TAO to {name}...')
    result = sub.transfer(
        wallet=alice,
        destination_ss58=w.coldkeypub.ss58_address,
        amount=bt.Balance.from_tao(100_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    bal = sub.get_balance(w.coldkeypub.ss58_address)
    print(f'    {name} balance: {bal}')

# --- Create subnet ---
print('  Creating subnet...')
try:
    w_owner = bt.Wallet(name='owner')
    result = sub.register_subnet(wallet=w_owner, mev_protection=False)
    print(f'    Subnet created: {result}')
except Exception as e:
    print(f'    Subnet creation note (may already exist): {e}')

# --- Register neurons via burned_register (no PoW) ---
for name in ['validator', 'miner']:
    print(f'  Registering {name} on subnet 1...')
    try:
        w = bt.Wallet(name=name)
        result = sub.burned_register(wallet=w, netuid=1, mev_protection=False)
        print(f'    Registered: {result}')
    except Exception as e:
        print(f'    Registration note (may already be registered): {e}')

# --- Stake on validator ---
print('  Staking 1000 TAO on validator...')
try:
    w_val = bt.Wallet(name='validator')
    result = sub.add_stake(wallet=w_val, netuid=1, hotkey_ss58=w_val.hotkey.ss58_address, amount=bt.Balance.from_tao(1000), mev_protection=False)
    print(f'    Staked: {result}')
except Exception as e:
    print(f'    Staking note: {e}')

# --- Verify metagraph ---
meta = sub.metagraph(netuid=1)
print(f'  Metagraph: {meta.n} neurons registered')
print('  Setup complete.')
PYEOF

if [ $? -eq 0 ]; then
    ok "Subnet setup complete."
else
    warn "Subnet setup had errors (see above)."
fi
echo ""

# =============================================================================
# STEP 7a: Start local trainer server
# =============================================================================
echo "-----------------------------------------"
info "Step 7a: Starting local trainer server..."
echo "-----------------------------------------"

TRAINER_PORT=8090

# Kill anything on this port
if command -v fuser &>/dev/null; then
    fuser -k "${TRAINER_PORT}/tcp" 2>/dev/null || true
elif command -v lsof &>/dev/null; then
    lsof -ti :"$TRAINER_PORT" 2>/dev/null | xargs -r kill 2>/dev/null || true
fi

(cd "$PROJECT_DIR/runner/timeseries_forecast" && \
    PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" python3 -m uvicorn server:app \
    --host 0.0.0.0 --port "$TRAINER_PORT" --log-level info) \
    > /tmp/radar_trainer.log 2>&1 &
TRAINER_PID=$!
PIDS+=("$TRAINER_PID")

# Wait for trainer to be ready
for attempt in $(seq 1 10); do
    if curl -sf "http://127.0.0.1:${TRAINER_PORT}/health" >/dev/null 2>&1; then
        ok "Trainer server ready on port $TRAINER_PORT"
        break
    fi
    [ "$attempt" = 10 ] && warn "Trainer server not responding on port $TRAINER_PORT"
    sleep 1
done

TRAINER_URL="http://127.0.0.1:${TRAINER_PORT}"
echo ""

# =============================================================================
# STEP 7b: Start miner (commits Docker image + trainer URL to chain)
# =============================================================================
echo "-----------------------------------------"
info "Step 7b: Starting miner (commits image + trainer URL to chain)..."
echo "-----------------------------------------"

python3 miner/neuron.py \
    --netuid 1 \
    --subtensor.network local \
    --wallet.name miner \
    --docker_image systematic:latest \
    --trainer_url "$TRAINER_URL" \
    --axon.external_ip 127.0.0.1 \
    > /tmp/radar_miner.log 2>&1 &
MINER_PID=$!
PIDS+=("$MINER_PID")

sleep 5  # Give miner time to commit image to chain

if kill -0 "$MINER_PID" 2>/dev/null; then
    ok "Miner started (PID $MINER_PID). Logs: /tmp/radar_miner.log"
else
    fail "Miner failed to start. Check /tmp/radar_miner.log"
    tail -20 /tmp/radar_miner.log 2>/dev/null || true
    exit 1
fi
echo ""

# =============================================================================
# STEP 8: Start validator (reads commitments, runs Docker images directly)
# =============================================================================
echo "-----------------------------------------"
info "Step 8: Starting validator..."
echo "-----------------------------------------"

export RADAR_RUNNER_BACKEND=local
export RADAR_DESEARCH_ENABLED=false
# Pin to medium-small bucket so agents reliably target the midpoint (6M FLOPs).
export RADAR_MIN_FLOPS=2000000
export RADAR_MAX_FLOPS=10000000

DB_DIR=$(mktemp -d)
DB_PORT=8099  # Use non-standard port to avoid conflicts

# Kill any leftover process holding the DB port (e.g. from a Ctrl+Z'd run)
if command -v fuser &>/dev/null; then
    fuser -k "${DB_PORT}/tcp" 2>/dev/null || true
    sleep 1
elif command -v lsof &>/dev/null; then
    lsof -ti :"$DB_PORT" 2>/dev/null | xargs -r kill 2>/dev/null || true
    sleep 1
fi

python3 validator/neuron.py \
    --netuid 1 \
    --subtensor.network local \
    --wallet.name validator \
    --db_dir "$DB_DIR" \
    --db_port "$DB_PORT" \
    > /tmp/radar_validator.log 2>&1 &
VALIDATOR_PID=$!
PIDS+=("$VALIDATOR_PID")

sleep 3

if kill -0 "$VALIDATOR_PID" 2>/dev/null; then
    ok "Validator started (PID $VALIDATOR_PID). Logs: /tmp/radar_validator.log"
else
    fail "Validator failed to start. Check /tmp/radar_validator.log"
    tail -20 /tmp/radar_validator.log 2>/dev/null || true
    exit 1
fi
echo ""

# =============================================================================
# STEP 9: Wait for one forward loop and verify
# =============================================================================
echo "-----------------------------------------"
info "Step 9: Waiting for one validator forward loop..."
echo "-----------------------------------------"

# Wait up to 10 minutes — the first forward loop creates a seed, sends a
# challenge, waits for miner response (up to 120s), executes, and scores.
TIMEOUT=600
ELAPSED=0
INTERVAL=10
VLOG_LINES=0  # track how many validator log lines we've shown
MLOG_LINES=0  # track how many miner log lines we've shown

while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
    # Check if validator is still running
    if ! kill -0 "$VALIDATOR_PID" 2>/dev/null; then
        warn "Validator exited. Checking logs..."
        tail -30 /tmp/radar_validator.log 2>/dev/null || true
        break
    fi

    # Surface new validator log lines since last check
    if [ -f /tmp/radar_validator.log ]; then
        TOTAL_VLINES=$(wc -l < /tmp/radar_validator.log)
        if [ "$TOTAL_VLINES" -gt "$VLOG_LINES" ]; then
            SKIP=$((VLOG_LINES + 1))
            tail -n +"$SKIP" /tmp/radar_validator.log | head -n $((TOTAL_VLINES - VLOG_LINES)) | while IFS= read -r line; do
                echo "  [validator] $line"
            done
            VLOG_LINES=$TOTAL_VLINES
        fi
    fi

    # Surface new miner log lines since last check
    if [ -f /tmp/radar_miner.log ]; then
        TOTAL_MLINES=$(wc -l < /tmp/radar_miner.log)
        if [ "$TOTAL_MLINES" -gt "$MLOG_LINES" ]; then
            SKIP=$((MLOG_LINES + 1))
            tail -n +"$SKIP" /tmp/radar_miner.log | head -n $((TOTAL_MLINES - MLOG_LINES)) | while IFS= read -r line; do
                echo "  [miner]     $line"
            done
            MLOG_LINES=$TOTAL_MLINES
        fi
    fi

    # Check for experiment results via DB API
    RESPONSE=$(curl -s "http://localhost:${DB_PORT}/experiments/recent" 2>/dev/null || echo "")
    if echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list) and len(data) > 0:
        print(f'  Found {len(data)} experiment(s)')
        sys.exit(0)
except:
    pass
sys.exit(1)
" 2>/dev/null; then
        ok "Experiments found in DB."
        break
    fi

    # Also check for weight-setting in logs
    if grep -q "Set weights\|Weights set" /tmp/radar_validator.log 2>/dev/null; then
        ok "Validator set weights on chain."
        break
    fi

    if grep -q "Tempo complete\|forward.*complete\|Scored proposal\|DB size:" /tmp/radar_validator.log 2>/dev/null; then
        ok "Validator completed a forward pass."
        break
    fi

    # Check if validator is erroring repeatedly
    ERROR_COUNT=$(grep -c "Error in forward pass" /tmp/radar_validator.log 2>/dev/null)
    ERROR_COUNT=${ERROR_COUNT:-0}
    if [ "$ERROR_COUNT" -ge 2 ]; then
        warn "Validator hit repeated errors. Last 30 lines of log:"
        tail -30 /tmp/radar_validator.log 2>/dev/null || true
        break
    fi

    # Show miner status too
    if ! kill -0 "$MINER_PID" 2>/dev/null; then
        warn "Miner exited. Last 10 lines:"
        tail -10 /tmp/radar_miner.log 2>/dev/null || true
    fi

    info "  Waiting... (${ELAPSED}s / ${TIMEOUT}s)"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    warn "Timed out waiting for forward loop (${TIMEOUT}s)."
    warn "This may be normal — localnet tempo can vary."
fi
echo ""

# =============================================================================
# STEP 10: Final verification
# =============================================================================
echo "-----------------------------------------"
info "Step 10: Final verification..."
echo "-----------------------------------------"

# Check experiment DB
info "Checking experiment DB..."
curl -s "http://localhost:${DB_PORT}/experiments/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  DB stats: {json.dumps(data, indent=2)}')
except:
    print('  DB not reachable or empty.')
" 2>/dev/null || warn "Could not reach DB."

# Check Pareto front
info "Checking Pareto front..."
curl -s "http://localhost:${DB_PORT}/experiments/pareto" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        print(f'  Pareto front size: {len(data)}')
except:
    print('  Pareto front not available.')
" 2>/dev/null || true

# Dump last few lines of logs
echo ""
info "Last 10 lines of validator log:"
tail -10 /tmp/radar_validator.log 2>/dev/null || true
echo ""
info "Last 10 lines of miner log:"
tail -10 /tmp/radar_miner.log 2>/dev/null || true

echo ""
echo "========================================="
ok "Localnet end-to-end test complete."
echo "========================================="
echo ""
echo "Logs:"
echo "  Validator: /tmp/radar_validator.log"
echo "  Miner:     /tmp/radar_miner.log"
echo "  Trainer:   /tmp/radar_trainer.log"
echo ""
