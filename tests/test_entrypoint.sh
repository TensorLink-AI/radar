#!/bin/bash
# Integration test for runner/agent/entrypoint.sh.
#
# Verifies three things by spinning up the agent image:
#   1. A host in RADAR_ALLOWED_URLS is reachable from inside the pod.
#   2. A host NOT in the allowlist is blocked.
#   3. Without CAP_NET_ADMIN the entrypoint degrades gracefully (container
#      still starts; app-layer GatedClient is the sole enforcer).
#
# Docker-gated: this test is NOT wired into the main pytest run. Invoke it
# manually in CI with `bash tests/test_entrypoint.sh` on a runner that has
# Docker available.

set -u
set -o pipefail

: "${IMAGE_TAG:=radar-agent-entrypoint-test:latest}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_DIR="$ROOT/runner/agent"

log() { echo "[test_entrypoint] $*" >&2; }
fail() { log "FAIL: $*"; exit 1; }

if ! command -v docker >/dev/null 2>&1; then
    log "SKIP: docker not available in this environment"
    exit 0
fi

log "Building image $IMAGE_TAG"
# The real image build uses a script that stages shared/ files into the
# build context. For this smoke test we only need a minimal image layered
# on top of the entrypoint; build a throwaway Dockerfile.
TMP_CTX="$(mktemp -d)"
trap 'rm -rf "$TMP_CTX"' EXIT
cp "$AGENT_DIR/entrypoint.sh" "$TMP_CTX/entrypoint.sh"
cat > "$TMP_CTX/Dockerfile" <<'DOCKERFILE'
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        iptables iproute2 curl dnsutils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["sleep", "infinity"]
DOCKERFILE
docker build -t "$IMAGE_TAG" "$TMP_CTX" >/dev/null || fail "docker build failed"

ALLOWED_HOST="example.com"
BLOCKED_IP="1.1.1.1"

run_container() {
    local name="$1" caps="$2"
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" $caps \
        -e RADAR_ALLOWED_URLS="https://${ALLOWED_HOST}/" \
        "$IMAGE_TAG" >/dev/null
    # Give the entrypoint a moment to program rules.
    sleep 2
}

# ── Case 1+2: full capability run — allowed reachable, blocked not ──────
CAPS="--cap-add=NET_ADMIN --cap-add=NET_RAW"
CN="radar-entrypoint-capped-$$"
run_container "$CN" "$CAPS"

log "Case 1: allowed host ($ALLOWED_HOST) should be reachable"
if ! docker exec "$CN" curl -fsS --max-time 10 \
        "https://${ALLOWED_HOST}/" >/dev/null 2>&1; then
    docker logs "$CN" >&2 || true
    docker rm -f "$CN" >/dev/null 2>&1 || true
    fail "allowed host unreachable"
fi
log "  ok"

log "Case 2: blocked host ($BLOCKED_IP) should fail"
if docker exec "$CN" curl -fsS --max-time 5 \
        "https://${BLOCKED_IP}/" >/dev/null 2>&1; then
    docker logs "$CN" >&2 || true
    docker rm -f "$CN" >/dev/null 2>&1 || true
    fail "blocked host was reachable — egress lockdown broken"
fi
log "  ok (connection refused/timeout as expected)"

docker rm -f "$CN" >/dev/null 2>&1 || true

# ── Case 3: no CAP_NET_ADMIN — container must still start ───────────────
UCN="radar-entrypoint-uncapped-$$"
run_container "$UCN" ""

log "Case 3: without CAP_NET_ADMIN the container should still be running"
if ! docker inspect -f '{{.State.Running}}' "$UCN" 2>/dev/null | grep -q true; then
    docker logs "$UCN" >&2 || true
    docker rm -f "$UCN" >/dev/null 2>&1 || true
    fail "container exited without CAP_NET_ADMIN"
fi
if ! docker logs "$UCN" 2>&1 | grep -q "iptables not available"; then
    docker logs "$UCN" >&2 || true
    docker rm -f "$UCN" >/dev/null 2>&1 || true
    fail "expected graceful degradation warning not found in logs"
fi
log "  ok (graceful fallback)"

docker rm -f "$UCN" >/dev/null 2>&1 || true

log "All entrypoint tests passed"
