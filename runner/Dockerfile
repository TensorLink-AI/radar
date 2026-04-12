FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for sandbox
RUN useradd -m -s /bin/bash -u 1000 trainer

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

# ── Timeseries forecast runner ─────────────────────────────────────
RUN mkdir -p /workspace/runner/timeseries_forecast
COPY timeseries_forecast/__init__.py /workspace/runner/timeseries_forecast/__init__.py
COPY timeseries_forecast/train.py /workspace/runner/timeseries_forecast/train.py
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

# Frozen copies (read-only reference)
RUN mkdir -p /workspace/frozen
COPY timeseries_forecast/harness.py /workspace/frozen/harness.py
COPY timeseries_forecast/prepare.py /workspace/frozen/prepare.py
COPY timeseries_forecast/evaluate.py /workspace/frozen/evaluate.py
COPY timeseries_forecast/flops.py /workspace/frozen/flops.py

# ── Shared modules (auth, R2 upload) ──────────────────────────────
# These are copied into the build context by the build script with shared_ prefix
RUN mkdir -p /workspace/shared
COPY shared_auth.py /workspace/shared/auth.py
COPY shared_artifacts.py /workspace/shared/artifacts.py
COPY shared_r2_audit.py /workspace/shared/r2_audit.py
COPY shared_pretrain_data.py /workspace/shared/pretrain_data.py
COPY shared_gift_eval.py /workspace/shared/gift_eval.py
RUN touch /workspace/shared/__init__.py

# Lock down frozen files and harness code
RUN chmod -R 444 /workspace/frozen/
RUN chown -R root:root /workspace/runner /workspace/frozen \
        /workspace/shared /workspace/server.py \
        /workspace/harness.py /workspace/prepare.py \
        /workspace/evaluate.py /workspace/flops.py \
        /workspace/pretrain_loader.py \
    && chmod -R 555 /workspace/runner /workspace/frozen /workspace/shared \
    && chmod 444 /workspace/server.py

# Sandbox workspace (writable by trainer)
RUN mkdir -p /workspace/sandbox/checkpoints \
        /workspace/sandbox/shards \
        /workspace/sandbox/data \
    && chown -R trainer:trainer /workspace/sandbox

# Sandbox scripts
COPY sandbox_runner.py /workspace/sandbox_runner.py
COPY sandbox_network.sh /usr/local/bin/sandbox_network.sh
RUN chown root:root /workspace/sandbox_runner.py /usr/local/bin/sandbox_network.sh \
    && chmod 444 /workspace/sandbox_runner.py \
    && chmod 555 /usr/local/bin/sandbox_network.sh

WORKDIR /workspace
EXPOSE 8081
CMD ["python3", "/workspace/server.py"]
