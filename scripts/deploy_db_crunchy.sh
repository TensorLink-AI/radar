#!/usr/bin/env bash
# Deploy the centralized Postgres DB server against a managed Postgres
# (Crunchy Data, Supabase, RDS, etc). No local Docker Postgres is started —
# the DSN points at an external, already-provisioned database.
#
# What this does:
#   1. Validates the env vars needed to reach the managed DB
#   2. Starts the database/neuron.py FastAPI server
#
# Usage:
#   export RADAR_PG_DSN='postgresql://user:pass@host.crunchybridge.com:5432/radar'
#   export RADAR_PG_SSL=require
#   export RADAR_NETWORK=testnet  # or mainnet
#   ./scripts/deploy_db_crunchy.sh --netuid 279 --subtensor.network test
#
# Environment:
#   RADAR_PG_DSN          Postgres connection string (REQUIRED)
#   RADAR_PG_SSL          "require" for managed Postgres (default: require)
#   RADAR_DB_API_PORT     API port (default: 8090)
#   RADAR_NETWORK         Postgres schema to write into (default: testnet)
#                         — a single DB holds BOTH testnet and mainnet data,
#                           strictly isolated by schema. DO NOT set this to
#                           "mainnet" unless you really mean it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

API_PORT="${RADAR_DB_API_PORT:-8090}"
NETWORK="${RADAR_NETWORK:-testnet}"
PG_SSL="${RADAR_PG_SSL:-require}"

# ── Colors ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 1. Validate env ──────────────────────────────────────
info "Step 1/2: Validating managed-Postgres configuration..."

if [ -z "${RADAR_PG_DSN:-}" ]; then
    error "RADAR_PG_DSN is not set. Point it at your managed Postgres, e.g.:"
    error "  export RADAR_PG_DSN='postgresql://user:pass@host:5432/radar'"
    exit 1
fi

# Strip any credentials before logging. DSNs look like
#   postgresql://user:pass@host:port/db
# so we print everything after the '@'.
DSN_SANITIZED="$(echo "${RADAR_PG_DSN}" | sed -E 's|://[^@]*@|://<redacted>@|')"

export RADAR_PG_SSL="${PG_SSL}"
export RADAR_NETWORK="${NETWORK}"

# ── 2. Start DB server ───────────────────────────────────
info "Step 2/2: Starting database server on port ${API_PORT}..."
echo ""

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
info "  RADAR_PG_DSN=${DSN_SANITIZED}"
info "  RADAR_PG_SSL=${PG_SSL}"
info "  RADAR_DB_API_PORT=${API_PORT}"
info "  RADAR_NETWORK=${NETWORK}"
info "  Passing args: $*"
echo ""
info "Validators should set:"
info "  RADAR_DB_API_URL=http://<this-host>:${API_PORT}"
echo ""

cd "${PROJECT_DIR}"
exec python database/neuron.py --port "${API_PORT}" "$@"
