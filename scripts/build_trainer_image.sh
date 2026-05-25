#!/usr/bin/env bash
# Build + push the timeseries_forecast trainer image.
#
# Usage:
#   IMAGE=ghcr.io/tensorlink-ai/ts-runner:latest scripts/build_trainer_image.sh
#
# Assumes you are already logged in to the registry (docker login ...).

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/tensorlink-ai/ts-runner:latest}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$REPO_ROOT"

# Task package (runner/timeseries_forecast/*.py — skip __pycache__)
for f in runner/timeseries_forecast/*.py; do
    cp "$f" "$STAGE/"
done

# Generic runner pieces — names that match the Dockerfile's COPY srcs
cp runner/entrypoint.sh        "$STAGE/entrypoint.sh"
cp runner/_bootstrap.py        "$STAGE/_bootstrap.py"
cp runner/_gen_hashes.py       "$STAGE/_gen_hashes.py"
cp runner/server.py            "$STAGE/server.py"
cp runner/launcher.py          "$STAGE/launcher.py"
cp runner/handler.py           "$STAGE/handler.py"
cp runner/sandbox.py           "$STAGE/sandbox.py"
cp runner/sandbox_runner.py    "$STAGE/sandbox_runner.py"
cp runner/uploads.py           "$STAGE/uploads.py"
cp runner/sandbox_wrap.sh      "$STAGE/sandbox_wrap.sh"
cp runner/__init__.py          "$STAGE/__init__.py"
cp runner/harness.py           "$STAGE/runner_harness.py"

# Shared modules the Dockerfile expects at the staging root
cp shared/auth.py              "$STAGE/auth.py"
cp shared/artifacts.py         "$STAGE/artifacts.py"
cp shared/r2_audit.py          "$STAGE/r2_audit.py"
cp shared/pretrain_data.py     "$STAGE/pretrain_data.py"
cp shared/gift_eval.py         "$STAGE/gift_eval.py"

# Dockerfile last (so any of the above failing aborts before build)
cp runner/timeseries_forecast/Dockerfile "$STAGE/Dockerfile"

echo "Build context staged at $STAGE"
ls "$STAGE"

docker build --no-cache -t "$IMAGE" "$STAGE"
docker push "$IMAGE"

echo
echo "Pushed: $IMAGE"
docker inspect --format='{{index .RepoDigests 0}}' "$IMAGE"
