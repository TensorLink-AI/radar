FROM python:3.11-slim

# ── ML deps (same as runner/agent image) ──────────────────────────
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy safetensors

# ── Frozen harness + shared modules ───────────────────────────────
RUN mkdir -p /workspace/shared /workspace/agent
COPY ../../runner/agent/harness.py /workspace/harness.py
COPY ../../shared/url_gate.py /workspace/shared/url_gate.py
RUN touch /workspace/shared/__init__.py

# Lock down harness (read-only)
RUN chmod 444 /workspace/harness.py /workspace/shared/url_gate.py

# ── Affinetes Actor (env.py) ──────────────────────────────────────
COPY env.py /app/env.py

WORKDIR /app
