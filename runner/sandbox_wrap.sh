#!/bin/sh
# Best-effort network-namespace wrapper for the trainer sandbox.
#
# When the host kernel allows unprivileged user namespaces (the default on
# every modern Linux distro and on Basilica's Docker hosts) we can drop the
# child into a fresh empty network namespace.  After that the only sockets
# available inside the sandbox are loopback, which we bring up so common
# Python init paths that resolve "127.0.0.1" don't crash.
#
# If unshare cannot create a netns (locked-down kernel, restricted Docker
# config) we still execute the child — the Python import blocker in
# sandbox_runner.py is the second line of defense.  We log the fallback to
# stderr so operators can spot it in pod logs.

set -eu

if command -v unshare >/dev/null 2>&1; then
    if unshare -n true >/dev/null 2>&1; then
        exec unshare -n -- /bin/sh -c 'ip link set lo up 2>/dev/null || true; exec "$@"' _ "$@"
    fi
    if unshare -rn true >/dev/null 2>&1; then
        exec unshare -rn -- /bin/sh -c 'ip link set lo up 2>/dev/null || true; exec "$@"' _ "$@"
    fi
fi

echo "[sandbox] WARNING: unshare unavailable — relying on Python import blocker only" >&2
exec "$@"
