FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
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

# Lock down frozen files
RUN chmod -R 444 /workspace/frozen/

WORKDIR /workspace
EXPOSE 8081
CMD ["python3", "/workspace/server.py"]
