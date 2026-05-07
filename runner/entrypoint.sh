#!/bin/sh
# Hardened container entrypoint for the RADAR trainer image.
#
# This script is the FIRST thing the container runs. It defends against
# Targon-style ContainerConfig overrides by:
#
#   1. Refusing to run if the operator passed positional args (the
#      Dockerfile sets CMD [] — anything in $# means `command`/`args`
#      were overridden by the deploy config).
#   2. Refusing to run if any preload-style or import-path-altering env
#      var is set (LD_PRELOAD, PYTHONPATH, etc.).
#   3. Unsetting every env var not on the allow-list before exec'ing
#      Python — so a miner can't slip behaviour through e.g. unknown
#      RADAR-shaped names.
#   4. Exec'ing (not forking) into the bootstrap so the bootstrap is
#      PID 1's direct successor.
#
# Bypass detection lives on the validator side: GET /boot_proof reads
# /tmp/boot_proof.json (only created if the bootstrap actually ran) and
# returns 503 otherwise.

set -u

if [ "$#" -ne 0 ]; then
    echo "radar-entrypoint: refusing to run — unexpected positional args ($#): $*" >&2
    exit 100
fi

# Refuse to run if any preload / import-path-altering env var is set.
# PYTHONNOUSERSITE is handled separately below (we always set it).
#
# LD_LIBRARY_PATH is NOT in this list because the nvidia/cuda base image
# bakes it into the image (LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# so the NVIDIA Container Toolkit's bind-mounted driver libs are
# resolvable). Refusing on that broke every container off the new
# hardened image. Instead the allow-list strip below drops whatever
# value reached us (operator-supplied or inherited), and we re-export
# the canonical NVIDIA path right before exec — attacker overrides
# never make it through, CUDA still works.
for var in LD_PRELOAD LD_AUDIT PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE; do
    eval "val=\${$var:-}"
    if [ -n "$val" ]; then
        echo "radar-entrypoint: refusing to run — banned env var $var is set" >&2
        exit 101
    fi
done

# Always pin this so user-site dirs can't shadow installed packages.
PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE

# Allow-list of env vars passed through to Python. Anything not matched
# by either the exact-name set or the prefix set is unset before exec.
allow_exact='^(PATH|HOME|LANG|LC_ALL|TZ|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|NVIDIA_DRIVER_CAPABILITIES|TRAINER_PORT|NETUID|SUBTENSOR_NETWORK|RADAR_LOCALNET|TRAINER_HOTKEY_COOLDOWN|METAGRAPH_REFRESH_INTERVAL|PYTHONNOUSERSITE)$'
# Prefix allow-list — covers wallet vars (WALLET_*, BT_*), trainer-side
# config (RADAR_*), artifact storage (R2_*, S3_*, AWS_*, HIPPIUS_*),
# and localnet test overrides (MOCK_*).
allow_prefix='^(WALLET_|BT_|RADAR_|R2_|S3_|AWS_|HIPPIUS_|MOCK_)'

names=$(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p')
for name in $names; do
    if echo "$name" | grep -Eq "$allow_exact"; then
        continue
    fi
    if echo "$name" | grep -Eq "$allow_prefix"; then
        continue
    fi
    unset "$name"
done

# Sane defaults for anything the allow-list left empty.
: "${PATH:=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
: "${HOME:=/tmp}"
: "${LANG:=C.UTF-8}"
: "${LC_ALL:=C.UTF-8}"
export PATH HOME LANG LC_ALL

# Re-export the canonical NVIDIA driver-library path. The allow-list
# strip above unset whatever value reached us, so this overwrites any
# operator-injected LD_LIBRARY_PATH while keeping CUDA usable. The
# value matches what nvidia/cuda:12.4.1-*-ubuntu22.04 bakes into the
# base image; if you bump the base image and that value drifts, change
# it here too.
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LD_LIBRARY_PATH

exec python3 /workspace/_bootstrap.py
