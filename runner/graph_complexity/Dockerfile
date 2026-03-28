FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir \
    bittensor cryptography sentencepiece \
    fastapi uvicorn httpx boto3 safetensors scipy numpy

# Frozen code (harness, prepare, evaluate, complexity)
COPY harness.py /workspace/frozen/harness.py
COPY prepare.py /workspace/frozen/prepare.py
COPY evaluate.py /workspace/frozen/evaluate.py
COPY complexity.py /workspace/frozen/complexity.py

# Lock down frozen files
RUN chmod -R 444 /workspace/frozen/

# Also place copies in /workspace for direct imports
COPY harness.py /workspace/harness.py
COPY prepare.py /workspace/prepare.py
COPY evaluate.py /workspace/evaluate.py
COPY complexity.py /workspace/complexity.py

# Shared modules
RUN mkdir -p /workspace/shared
RUN touch /workspace/shared/__init__.py

WORKDIR /workspace
EXPOSE 8081
