#!/usr/bin/env bash
# Deploy the centralized Postgres DB server for testnet.
#
# What this does:
#   1. Starts Postgres in Docker (if not already running)
#   2. Waits for Postgres to accept connections
#   3. Starts the database/neuron.py FastAPI server
#
# Usage:
#   ./scripts/deploy_db_testnet.sh 
#
# Environment (optional overrides):
#   RADAR_PG_DSN          Postgres connection string (default: local Docker)
#   RADAR_DB_API_PORT     API port (default: 8090)
#   RADAR_PG_PORT         Postgres port (default: 5432)
#   RADAR_NETWORK         Postgres schema to write into (default: testnet)
#                         — a single DB holds BOTH testnet and mainnet data,
#                           strictly isolated by schema. DO NOT set this to
#                           "mainnet" unless you really mean it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PG_PORT="${RADAR_PG_PORT:-5432}"
API_PORT="${RADAR_DB_API_PORT:-8090}"
PG_USER="radar"
PG_PASS="radar"
PG_DB="radar"
CONTAINER_NAME="radar-pg"
NETWORK="${RADAR_NETWORK:-testnet}"

# ── Colors ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 1. Start Postgres ────────────────────────────────────
info "Step 1/3: Starting Postgres..."

if [ -n "${RADAR_PG_DSN:-}" ]; then
    info "Using existing RADAR_PG_DSN (skipping Docker)"
    DSN="${RADAR_PG_DSN}"
else
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            info "Postgres already running (${CONTAINER_NAME})"
        else
            info "Starting existing container ${CONTAINER_NAME}..."
            docker start "${CONTAINER_NAME}"
        fi
    else
        info "Creating Postgres container ${CONTAINER_NAME}..."
        docker run -d --name "${CONTAINER_NAME}" \
            -e POSTGRES_USER="${PG_USER}" \
            -e POSTGRES_PASSWORD="${PG_PASS}" \
            -e POSTGRES_DB="${PG_DB}" \
            -p "${PG_PORT}:5432" \
            -v radar-pg-data:/var/lib/postgresql/data \
            postgres:16-alpine
    fi
    DSN="postgresql://${PG_USER}:${PG_PASS}@localhost:${PG_PORT}/${PG_DB}"
    export RADAR_PG_DSN="${DSN}"
fi

# ── 2. Wait for Postgres ─────────────────────────────────
info "Step 2/3: Waiting for Postgres to accept connections..."

MAX_RETRIES=30
for i in $(seq 1 $MAX_RETRIES); do
    if docker exec "${CONTAINER_NAME}" pg_isready -U "${PG_USER}" -d "${PG_DB}" >/dev/null 2>&1; then
        info "Postgres is ready"
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        error "Postgres did not become ready after ${MAX_RETRIES}s"
        exit 1
    fi
    sleep 1
done

# ── 3. Start DB server ───────────────────────────────────
info "Step 3/3: Starting database server on port ${API_PORT}..."
echo ""

# Make the schema selection impossible to miss. Operators have accidentally
# written mainnet rows into a "testnet" DB when this was silent — don't let
# that happen here.
if [ "${NETWORK}" = "mainnet" ]; then
    warn "╔════════════════════════════════════════════════════╗"
    warn "║  RADAR_NETWORK=mainnet — writing to MAINNET schema ║"
    warn "╚════════════════════════════════════════════════════╝"
else
    info "╔════════════════════════════════════════════════════╗"
    info "║  RADAR_NETWORK=${NETWORK} — writing to ${NETWORK} schema"
    info "╚════════════════════════════════════════════════════╝"
fi

info "Config:"
info "  RADAR_PG_DSN=${DSN}"
info "  RADAR_DB_API_PORT=${API_PORT}"
info "  RADAR_NETWORK=${NETWORK}"
info "  Passing args: $*"
echo ""
info "Validators should set:"
info "  RADAR_DB_API_URL=http://<this-host>:${API_PORT}"
echo ""

export RADAR_NETWORK="${NETWORK}"

cd "${PROJECT_DIR}"
exec python database/neuron.py --port "${API_PORT}" "$@"
