FROM python:3.11-slim

# ── Minimal deps — NO requests, httpx, aiohttp, or urllib3 ──────────
# Miner code cannot bring its own HTTP client.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy safetensors

# ── Frozen harness + shared modules ─────────────────────────────────
RUN mkdir -p /workspace/shared /workspace/agent
COPY harness.py /workspace/harness.py
# url_gate.py is copied into the build context by the build script
COPY shared_url_gate.py /workspace/shared/url_gate.py
RUN touch /workspace/shared/__init__.py

# Lock down harness (read-only)
RUN chmod 444 /workspace/harness.py /workspace/shared/url_gate.py

# ── Miner's agent code is mounted at /workspace/agent/ at runtime ───
# The validator injects the miner's .py files here before starting.
# Default entrypoint: frozen harness reads stdin, loads miner module.
WORKDIR /workspace
ENTRYPOINT ["python", "/workspace/harness.py"]
