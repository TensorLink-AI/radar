FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# iproute2 + util-linux for the sandbox network namespace wrapper.
RUN apt-get update && apt-get install -y \
    python3 python3-pip git iproute2 util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir \
    bittensor cryptography sentencepiece \
    fastapi uvicorn httpx boto3 safetensors pyarrow pandas

# ── Generic harness ───────────────────────────────────────────────
RUN mkdir -p /workspace/runner
COPY __init__.py /workspace/runner/__init__.py
COPY harness.py /workspace/runner/harness.py
COPY server.py /workspace/runner/server.py
COPY server.py /workspace/server.py

# ── Sandbox runner (untrusted miner code lives here) ──────────────
COPY sandbox.py /workspace/runner/sandbox.py
COPY sandbox_runner.py /workspace/sandbox_runner.py
COPY uploads.py /workspace/runner/uploads.py
COPY sandbox_wrap.sh /usr/local/bin/sandbox_wrap.sh

# ── Timeseries forecast runner ─────────────────────────────────────
RUN mkdir -p /workspace/runner/timeseries_forecast
COPY timeseries_forecast/__init__.py /workspace/runner/timeseries_forecast/__init__.py
COPY timeseries_forecast/train.py /workspace/runner/timeseries_forecast/train.py
COPY timeseries_forecast/eval_template.py /workspace/runner/timeseries_forecast/eval_template.py
COPY timeseries_forecast/harness.py /workspace/runner/timeseries_forecast/harness.py
COPY timeseries_forecast/prepare.py /workspace/runner/timeseries_forecast/prepare.py
COPY timeseries_forecast/evaluate.py /workspace/runner/timeseries_forecast/evaluate.py
COPY timeseries_forecast/flops.py /workspace/runner/timeseries_forecast/flops.py
COPY timeseries_forecast/pretrain_loader.py /workspace/runner/timeseries_forecast/pretrain_loader.py

# Also place ts-forecasting files at /workspace for direct imports
# (frozen harness.py and prepare.py expect to be importable as top-level modules)
COPY timeseries_forecast/harness.py /workspace/harness.py
COPY timeseries_forecast/prepare.py /workspace/prepare.py
COPY timeseries_forecast/evaluate.py /workspace/evaluate.py
COPY timeseries_forecast/flops.py /workspace/flops.py
COPY timeseries_forecast/pretrain_loader.py /workspace/pretrain_loader.py

# Affinetes actor (mirrors timeseries_forecast/Dockerfile so EXPECTED_FILES
# lists the same set across both Dockerfiles).
COPY timeseries_forecast/env.py /workspace/env.py

# Frozen copies (read-only reference)
RUN mkdir -p /workspace/frozen
COPY timeseries_forecast/harness.py /workspace/frozen/harness.py
COPY timeseries_forecast/prepare.py /workspace/frozen/prepare.py
COPY timeseries_forecast/evaluate.py /workspace/frozen/evaluate.py
COPY timeseries_forecast/flops.py /workspace/frozen/flops.py
# Also include auth.py + server.py here so the frozen layout matches
# the timeseries_forecast Dockerfile and EXPECTED_FILES is one list.
COPY shared_auth.py /workspace/frozen/auth.py
COPY server.py /workspace/frozen/server.py

# ── Shared modules (auth, R2 upload) ──────────────────────────────
# These are copied into the build context by the build script with shared_ prefix
RUN mkdir -p /workspace/shared
COPY shared_auth.py /workspace/shared/auth.py
COPY shared_artifacts.py /workspace/shared/artifacts.py
COPY shared_r2_audit.py /workspace/shared/r2_audit.py
COPY shared_pretrain_data.py /workspace/shared/pretrain_data.py
COPY shared_gift_eval.py /workspace/shared/gift_eval.py
RUN touch /workspace/shared/__init__.py

# ── Hardened entrypoint + bootstrap (load-bearing for image attestation) ─
COPY entrypoint.sh /usr/local/bin/radar-entrypoint.sh
COPY _bootstrap.py /workspace/_bootstrap.py
COPY _gen_hashes.py /workspace/_gen_hashes.py
COPY boot_proof.py /workspace/runner/boot_proof.py

# Per-job sandbox lives OUTSIDE /workspace so the read-only lockdown
# below doesn't have to carve out an exception for it.
RUN mkdir -p /var/radar/sandbox/checkpoints \
             /var/radar/sandbox/shards \
             /var/radar/sandbox/val_shards \
             /var/radar/sandbox/gift_eval

# Generate the build-time hash table. This step runs AFTER every COPY so
# the hashes match the bytes the image actually ships. _gen_hashes.py
# itself is the single source of truth for the file list.
RUN python3 /workspace/_gen_hashes.py

# Lock down every load-bearing file so a runtime attacker (even if they
# get inside the container) can't replace harness, server, or bootstrap.
# Directories go 555 so files can't be added either.
RUN chmod 555 /usr/local/bin/radar-entrypoint.sh \
              /usr/local/bin/sandbox_wrap.sh \
    && chmod 444 /workspace/_bootstrap.py \
                 /workspace/_gen_hashes.py \
                 /workspace/_bootstrap_hashes.json \
                 /workspace/server.py \
                 /workspace/sandbox_runner.py \
                 /workspace/harness.py /workspace/prepare.py \
                 /workspace/evaluate.py /workspace/flops.py \
                 /workspace/pretrain_loader.py /workspace/env.py \
                 /workspace/runner/__init__.py \
                 /workspace/runner/boot_proof.py \
                 /workspace/runner/harness.py \
                 /workspace/runner/server.py \
                 /workspace/runner/sandbox.py \
                 /workspace/runner/uploads.py \
                 /workspace/runner/timeseries_forecast/__init__.py \
                 /workspace/runner/timeseries_forecast/train.py \
                 /workspace/runner/timeseries_forecast/eval_template.py \
                 /workspace/runner/timeseries_forecast/harness.py \
                 /workspace/runner/timeseries_forecast/prepare.py \
                 /workspace/runner/timeseries_forecast/evaluate.py \
                 /workspace/runner/timeseries_forecast/flops.py \
                 /workspace/runner/timeseries_forecast/pretrain_loader.py \
    && chmod -R 444 /workspace/frozen/ \
    && chmod -R 444 /workspace/shared/ \
    && chmod 555 /workspace \
                 /workspace/runner \
                 /workspace/runner/timeseries_forecast \
                 /workspace/shared \
                 /workspace/frozen

WORKDIR /workspace
EXPOSE 8081
ENTRYPOINT ["/usr/local/bin/radar-entrypoint.sh"]
CMD []
