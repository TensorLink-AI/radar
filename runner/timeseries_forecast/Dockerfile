FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git iproute2 util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir \
    cryptography sentencepiece \
    fastapi uvicorn httpx boto3 safetensors pyarrow pandas \
    runpod

# ── Generic harness package + top-level entrypoints ───────────────
RUN mkdir -p /workspace/runner/timeseries_forecast
COPY __init__.py /workspace/runner/__init__.py
COPY runner_harness.py /workspace/runner/harness.py
COPY server.py /workspace/runner/server.py
COPY server.py /workspace/server.py
COPY launcher.py /workspace/runner/launcher.py
COPY launcher.py /workspace/launcher.py
COPY handler.py /workspace/runner/handler.py
COPY sandbox.py /workspace/runner/sandbox.py
COPY sandbox_runner.py /workspace/sandbox_runner.py
COPY uploads.py /workspace/runner/uploads.py
COPY sandbox_wrap.sh /usr/local/bin/sandbox_wrap.sh

# ── Timeseries package ────────────────────────────────────────────
COPY __init__.py /workspace/runner/timeseries_forecast/__init__.py
COPY train.py /workspace/runner/timeseries_forecast/train.py
COPY eval_template.py /workspace/runner/timeseries_forecast/eval_template.py
COPY harness.py /workspace/runner/timeseries_forecast/harness.py
COPY prepare.py /workspace/runner/timeseries_forecast/prepare.py
COPY evaluate.py /workspace/runner/timeseries_forecast/evaluate.py
COPY flops.py /workspace/runner/timeseries_forecast/flops.py
COPY pretrain_loader.py /workspace/runner/timeseries_forecast/pretrain_loader.py

# Top-level imports (frozen harness.py / prepare.py expect importable as top-level)
COPY harness.py /workspace/harness.py
COPY prepare.py /workspace/prepare.py
COPY evaluate.py /workspace/evaluate.py
COPY flops.py /workspace/flops.py
COPY pretrain_loader.py /workspace/pretrain_loader.py
COPY env.py /workspace/env.py

# Frozen code (harness, auth, prepare, evaluate, flops, server)
RUN mkdir -p /workspace/frozen
COPY harness.py /workspace/frozen/harness.py
COPY auth.py /workspace/frozen/auth.py
COPY prepare.py /workspace/frozen/prepare.py
COPY evaluate.py /workspace/frozen/evaluate.py
COPY flops.py /workspace/frozen/flops.py
COPY server.py /workspace/frozen/server.py

# Shared modules (needed by server.py for R2 upload)
RUN mkdir -p /workspace/shared
COPY auth.py /workspace/shared/auth.py
COPY artifacts.py /workspace/shared/artifacts.py
COPY r2_audit.py /workspace/shared/r2_audit.py
COPY pretrain_data.py /workspace/shared/pretrain_data.py
COPY gift_eval.py /workspace/shared/gift_eval.py
RUN touch /workspace/shared/__init__.py

# ── Hardened entrypoint + bootstrap ───────────────────────────────
COPY entrypoint.sh /usr/local/bin/radar-entrypoint.sh
COPY _bootstrap.py /workspace/_bootstrap.py
COPY _gen_hashes.py /workspace/_gen_hashes.py
COPY boot_proof.py /workspace/runner/boot_proof.py

# Per-job sandbox lives outside /workspace.
RUN mkdir -p /var/radar/sandbox/checkpoints \
             /var/radar/sandbox/shards \
             /var/radar/sandbox/val_shards \
             /var/radar/sandbox/gift_eval

# Generate the build-time hash table.
RUN python3 /workspace/_gen_hashes.py

# Lock everything down.
RUN chmod 555 /usr/local/bin/radar-entrypoint.sh \
              /usr/local/bin/sandbox_wrap.sh \
    && chmod 444 /workspace/_bootstrap.py \
                 /workspace/_gen_hashes.py \
                 /workspace/_bootstrap_hashes.json \
                 /workspace/server.py \
                 /workspace/launcher.py \
                 /workspace/sandbox_runner.py \
                 /workspace/harness.py /workspace/prepare.py \
                 /workspace/evaluate.py /workspace/flops.py \
                 /workspace/pretrain_loader.py /workspace/env.py \
                 /workspace/runner/__init__.py \
                 /workspace/runner/harness.py \
                 /workspace/runner/server.py \
                 /workspace/runner/launcher.py \
                 /workspace/runner/handler.py \
                 /workspace/runner/sandbox.py \
                 /workspace/runner/uploads.py \
                 /workspace/runner/boot_proof.py \
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
