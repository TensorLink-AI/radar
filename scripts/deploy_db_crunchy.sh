#!/usr/bin/env bash
# Deploy the centralized Postgres DB server against a Crunchy Bridge cluster.
#
# What this does:
#   1. Reads RADAR_PG_DSN + RADAR_NETWORK from env
#   2. Runs a connectivity precheck (psql \conninfo) before launching
#   3. Exec's database/neuron.py on the requested port / netuid
#
# Usage:
#   RADAR_PG_DSN=postgresql://radar:PW@p.abc.db.postgresbridge.com:5432/radar?sslmode=require \
#   RADAR_PG_SSL=verify \
#   RADAR_NETWORK=testnet \
#   ./scripts/deploy_db_crunchy.sh --netuid 279 --subtensor.network test
#
# Required env:
#   RADAR_PG_DSN          Full Crunchy connection string (include ?sslmode=require)
#   RADAR_NETWORK         Schema name: testnet | mainnet
#   RADAR_PG_SSL          "verify" (recommended) or "require" (deprecated)
#
# Optional env:
#   RADAR_DB_API_PORT     API port (default: 8090)
#   RADAR_PG_POOL_MIN     default 2 (config.py)
#   RADAR_PG_POOL_MAX     default 10 (config.py)
#
# Exits non-zero if:
#   - required env vars missing
#   - Crunchy is unreachable (DNS, firewall, credentials)
#   - psql is not installed

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
        error "RADAR_PG_SSL must be 'verify' or 'require' for Crunchy, got '${RADAR_PG_SSL}'"
        exit 2
        ;;
esac

if ! command -v psql >/dev/null 2>&1; then
    error "psql not found on PATH. Install postgresql-client-18 (matches Crunchy PG18)."
    exit 2
fi

# ── 2. Connectivity precheck ─────────────────────────────
info "Step 2/3: Probing Crunchy Bridge"

# \conninfo is cheap and proves DNS, TLS, credentials, and role all work.
# Timeout prevents a hung socket from blocking service startup forever.
if ! timeout 15 psql "${RADAR_PG_DSN}" -c '\conninfo' >/dev/null 2>&1; then
    error "Cannot connect to Crunchy via RADAR_PG_DSN."
    error "Run this to see the actual error:"
    error "  psql \"\${RADAR_PG_DSN}\" -c '\\conninfo'"
    error "Common causes: wrong password, IP not on Crunchy allowlist, DNS, expired cert."
    exit 3
fi

server_version=$(timeout 15 psql "${RADAR_PG_DSN}" -At -c 'SHOW server_version' 2>/dev/null || echo "unknown")
info "Connected to Crunchy (PostgreSQL ${server_version})"

# ── 3. Launch DB server ──────────────────────────────────
info "Step 3/3: Starting database server"
info "  RADAR_NETWORK=${RADAR_NETWORK}"
info "  RADAR_PG_SSL=${RADAR_PG_SSL}"
info "  RADAR_DB_API_PORT=${API_PORT}"
info "  Forwarding args: $*"
echo ""

cd "${PROJECT_DIR}"
exec python database/neuron.py --port "${API_PORT}" "$@"
