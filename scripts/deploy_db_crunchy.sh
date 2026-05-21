#!/usr/bin/env bash
# Deploy the centralized Postgres DB server against a managed Postgres
# (Crunchy Bridge, Supabase, RDS, etc). No local Docker Postgres is started —
# the DSN points at an external, already-provisioned database.
#
# What this does:
#   1. Validates required env vars
#   2. Runs a psql connectivity precheck (fails fast before uvicorn boots)
#   3. Exec's database/neuron.py on the requested port / netuid
#
# Usage:
#   export RADAR_PG_DSN='postgresql://radar:PW@p.abc.db.postgresbridge.com:5432/radar?sslmode=require'
#   export RADAR_PG_SSL=verify
#   export RADAR_NETWORK=testnet        # or mainnet
#   ./scripts/deploy_db_crunchy.sh 
#
# Required env:
#   RADAR_PG_DSN          Full connection string (include ?sslmode=require)
#   RADAR_PG_SSL          "verify" (recommended) or "require" (deprecated)
#   RADAR_NETWORK         Postgres schema — "testnet" or "mainnet"
#                         A single DB holds BOTH networks, isolated by
#                         schema. DO NOT set this to "mainnet" unless
#                         you really mean it.
#
# Optional env:
#   RADAR_DB_API_PORT     API port (default: 8090)
#   RADAR_PG_POOL_MIN     default 2 (config.py)
#   RADAR_PG_POOL_MAX     default 10 (config.py)
#
# Exits non-zero if:
#   - required env vars missing
#   - RADAR_NETWORK isn't testnet/mainnet
#   - Crunchy is unreachable (DNS, firewall, credentials)
#   - psql not installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

API_PORT="${RADAR_DB_API_PORT:-8090}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 1. Validate required env ─────────────────────────────
info "Step 1/3: Validating environment"

missing=()
[ -z "${RADAR_PG_DSN:-}" ] && missing+=("RADAR_PG_DSN")
[ -z "${RADAR_NETWORK:-}" ] && missing+=("RADAR_NETWORK")
[ -z "${RADAR_PG_SSL:-}" ] && missing+=("RADAR_PG_SSL")

if [ ${#missing[@]} -gt 0 ]; then
    error "Missing required env vars: ${missing[*]}"
    error "See comment block at top of this script for the full list."
    exit 2
fi

case "${RADAR_NETWORK}" in
    testnet|mainnet) ;;
    *)
        error "RADAR_NETWORK must be 'testnet' or 'mainnet', got '${RADAR_NETWORK}'"
        exit 2
        ;;
esac

case "${RADAR_PG_SSL}" in
    verify) ;;
    require)
        warn "RADAR_PG_SSL=require skips hostname/cert verification (deprecated)."
        warn "Prefer RADAR_PG_SSL=verify for Crunchy Bridge."
        ;;
    *)
        error "RADAR_PG_SSL must be 'verify' or 'require' for managed Postgres, got '${RADAR_PG_SSL}'"
        exit 2
        ;;
esac

if ! command -v psql >/dev/null 2>&1; then
    error "psql not found on PATH. Install postgresql-client-18 (matches Crunchy PG18)."
    exit 2
fi

# Strip credentials before logging. DSNs look like
#   postgresql://user:pass@host:port/db
# so we print everything after the '@'.
DSN_SANITIZED="$(echo "${RADAR_PG_DSN}" | sed -E 's|://[^@]*@|://<redacted>@|')"

# ── 2. Connectivity precheck ─────────────────────────────
info "Step 2/3: Probing managed Postgres"

# \conninfo proves DNS, TLS, credentials, and role all work.
# Timeout prevents a hung socket from blocking service startup forever.
if ! timeout 15 psql "${RADAR_PG_DSN}" -c '\conninfo' >/dev/null 2>&1; then
    error "Cannot connect to managed Postgres via RADAR_PG_DSN (${DSN_SANITIZED})."
    error "Run this to see the actual error:"
    error "  psql \"\${RADAR_PG_DSN}\" -c '\\conninfo'"
    error "Common causes: wrong password, IP not on cluster allowlist, DNS, expired cert."
    exit 3
fi

server_version=$(timeout 15 psql "${RADAR_PG_DSN}" -At -c 'SHOW server_version' 2>/dev/null || echo "unknown")
info "Connected (PostgreSQL ${server_version})"

# ── 3. Launch DB server ──────────────────────────────────
info "Step 3/3: Starting database server on port ${API_PORT}"

if [ "${RADAR_NETWORK}" = "mainnet" ]; then
    warn "╔════════════════════════════════════════════════════╗"
    warn "║  RADAR_NETWORK=mainnet — writing to MAINNET schema ║"
    warn "╚════════════════════════════════════════════════════╝"
else
    info "╔════════════════════════════════════════════════════╗"
    info "║  RADAR_NETWORK=${RADAR_NETWORK} — writing to ${RADAR_NETWORK} schema"
    info "╚════════════════════════════════════════════════════╝"
fi

info "Config:"
info "  RADAR_PG_DSN=${DSN_SANITIZED}"
info "  RADAR_PG_SSL=${RADAR_PG_SSL}"
info "  RADAR_NETWORK=${RADAR_NETWORK}"
info "  RADAR_DB_API_PORT=${API_PORT}"
info "  Forwarding args: $*"
echo ""
info "Validators should set:"
info "  RADAR_DB_API_URL=http://<this-host>:${API_PORT}"
echo ""

cd "${PROJECT_DIR}"
exec python database/neuron.py --port "${API_PORT}" "$@"
