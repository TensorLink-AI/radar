FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir \
    bittensor cryptography sentencepiece \
    fastapi uvicorn httpx boto3 safetensors

# Frozen code (harness, auth, prepare, evaluate, flops, server)
COPY harness.py /workspace/frozen/harness.py
COPY auth.py /workspace/frozen/auth.py
COPY prepare.py /workspace/frozen/prepare.py
COPY evaluate.py /workspace/frozen/evaluate.py
COPY flops.py /workspace/frozen/flops.py
COPY server.py /workspace/frozen/server.py

# Lock down frozen files
RUN chmod -R 444 /workspace/frozen/

# Also place copies in /workspace for direct imports
COPY harness.py /workspace/harness.py
COPY prepare.py /workspace/prepare.py
COPY evaluate.py /workspace/evaluate.py
COPY flops.py /workspace/flops.py
COPY server.py /workspace/server.py

# Shared modules (needed by server.py for R2 upload)
RUN mkdir -p /workspace/shared
COPY auth.py /workspace/shared/auth.py
COPY artifacts.py /workspace/shared/artifacts.py
COPY r2_audit.py /workspace/shared/r2_audit.py
RUN touch /workspace/shared/__init__.py

# Actor for affinetes
COPY env.py /workspace/env.py

WORKDIR /workspace
EXPOSE 8081
CMD ["python3", "/workspace/server.py"]
