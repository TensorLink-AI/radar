#!/bin/bash
# Agent pod entrypoint — enforces network-layer egress control before
# handing off to the agent server. Runs as root so it can program
# iptables; Basilica provides container-level isolation.
#
# Allowlist is read from RADAR_ALLOWED_URLS (JSON array or CSV of URL
# prefixes). Each host:port is resolved and allowed; everything else is
# dropped. DNS and established/related flows are always allowed.
#
# Degrades gracefully: if iptables is unavailable (no CAP_NET_ADMIN),
# we log a warning and fall through to the server, where the GatedClient
# app-layer gate still enforces the allowlist.

set -e

log() { echo "[entrypoint] $*" >&2; }

ALLOWED="${RADAR_ALLOWED_URLS:-${AGENT_ALLOWED_URLS:-}}"
if [ -z "$ALLOWED" ]; then
    log "WARNING: no RADAR_ALLOWED_URLS set — skipping egress lockdown"
    exec "$@"
fi

# Detect iptables capability up front. Without CAP_NET_ADMIN, every rule
# will fail; bail to app-layer gating rather than limping along half-
# programmed.
if ! iptables -L >/dev/null 2>&1; then
    log "WARNING: iptables not available (CAP_NET_ADMIN?) — falling back to app-layer gating only"
    exec "$@"
fi

# Parse allowlist: JSON array or comma-separated list of URL prefixes.
if [[ "$ALLOWED" == \[* ]]; then
    URLS=$(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "$ALLOWED")
else
    URLS=$(echo "$ALLOWED" | tr ',' '\n')
fi

log "Locking down egress..."
iptables -P OUTPUT DROP                                              || log "iptables policy DROP failed: $?"
iptables -A OUTPUT -o lo -j ACCEPT                                   || log "iptables loopback failed: $?"
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT                       || log "iptables udp dns failed: $?"
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT                       || log "iptables tcp dns failed: $?"
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT    || log "iptables established failed: $?"

# Allow each host:port in the allowlist.
while IFS= read -r url; do
    [ -z "$url" ] && continue
    eval "$(python3 -c "
from urllib.parse import urlparse
u = urlparse('''$url''')
host = u.hostname or ''
port = u.port or (443 if u.scheme == 'https' else 80)
print(f'HOST={host}')
print(f'PORT={port}')
")"
    [ -z "$HOST" ] && continue

    IPS=$(getent ahosts "$HOST" | awk '{print $1}' | sort -u)
    if [ -z "$IPS" ]; then
        log "WARN: could not resolve $HOST — skipping"
        continue
    fi
    for ip in $IPS; do
        iptables -A OUTPUT -d "$ip" -p tcp --dport "$PORT" -j ACCEPT \
            || log "iptables allow $ip:$PORT failed: $?"
        log "allowed: $HOST:$PORT ($ip)"
    done
done <<< "$URLS"

log "Egress lockdown complete. Rules:"
iptables -L OUTPUT -n --line-numbers >&2 || true

exec "$@"
