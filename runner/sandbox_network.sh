#!/bin/bash
# Run command with no external network. Loopback stays up (for future proxy).
# Falls back to user-switch-only if CAP_NET_ADMIN unavailable.
set -e
NS_NAME="radar_sandbox_$$"

if ip netns add "$NS_NAME" 2>/dev/null; then
    ip netns exec "$NS_NAME" ip link set lo up
    trap "ip netns delete '$NS_NAME' 2>/dev/null || true" EXIT
    ip netns exec "$NS_NAME" \
        su -s /bin/bash trainer -c "cd /workspace && $*"
else
    echo "[sandbox] WARNING: no CAP_NET_ADMIN, import blocker only" >&2
    su -s /bin/bash trainer -c "cd /workspace && $*"
fi
