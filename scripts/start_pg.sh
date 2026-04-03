#!/usr/bin/env bash
# Start Postgres in Docker for local dev / testing.
# Idempotent — does nothing if the container already exists.

set -euo pipefail

CONTAINER_NAME="radar-pg"
PG_USER="radar"
PG_PASS="radar"
PG_DB="radar"
PG_PORT="${RADAR_PG_PORT:-5432}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Postgres already running (${CONTAINER_NAME})"
    else
        echo "Starting existing container ${CONTAINER_NAME}..."
        docker start "${CONTAINER_NAME}"
    fi
else
    echo "Creating Postgres container ${CONTAINER_NAME}..."
    docker run -d --name "${CONTAINER_NAME}" \
        -e POSTGRES_USER="${PG_USER}" \
        -e POSTGRES_PASSWORD="${PG_PASS}" \
        -e POSTGRES_DB="${PG_DB}" \
        -p "${PG_PORT}:5432" \
        -v radar-pg-data:/var/lib/postgresql/data \
        postgres:16-alpine
fi

DSN="postgresql://${PG_USER}:${PG_PASS}@localhost:${PG_PORT}/${PG_DB}"
echo ""
echo "Connection DSN:"
echo "  RADAR_PG_DSN=${DSN}"
echo "  TEST_PG_DSN=${DSN}"
