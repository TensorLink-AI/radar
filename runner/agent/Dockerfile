FROM python:3.11-slim

# ── Minimal deps — NO requests, httpx, aiohttp, or urllib3 ───────��──
# Miner code cannot bring its own HTTP client.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy safetensors

# ── affinetes HTTP server deps ─────────────────────────────────────
# iptables/iproute2 are used by entrypoint.sh to enforce network-layer egress.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates iptables iproute2 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir fastapi uvicorn[standard] httpx pydantic \
    && pip install --no-cache-dir 'openai>=1.50' 'anthropic>=0.39' \
    && pip install --no-cache-dir git+https://github.com/AffineFoundation/affinetes.git@main

# ── affinetes server template (replicates two-stage inject) ────────
RUN mkdir -p /app/_affinetes \
    && cp $(python -c "import affinetes.templates; import os; print(os.path.dirname(affinetes.templates.__file__))")/http_server.py /app/_affinetes/server.py \
    && cp $(python -c "import affinetes.templates; import os; print(os.path.dirname(affinetes.templates.__file__))")/request_logger.py /app/request_logger.py \
    && touch /app/_affinetes/__init__.py \
    && chmod -R 777 /app/_affinetes

# ── Frozen harness + shared modules ───────────���─────────────────────
RUN mkdir -p /workspace/shared /workspace/agent
COPY harness.py /workspace/harness.py
# url_gate.py is copied into the build context by the build script
COPY shared_url_gate.py /workspace/shared/url_gate.py
RUN touch /workspace/shared/__init__.py

# Lock down harness (read-only)
RUN chmod 444 /workspace/harness.py /workspace/shared/url_gate.py

# ── affinetes Actor (env.py at /app/ where the server expects it) ──
COPY env.py /app/env.py

# ── Network-layer egress lockdown entrypoint ───────────────────────
# Programs iptables OUTPUT DROP + per-host ACCEPT for RADAR_ALLOWED_URLS
# before exec'ing the server. Degrades gracefully without CAP_NET_ADMIN.
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# ── Miner's agent code is injected at runtime via process_challenge ─
WORKDIR /app
EXPOSE 8000
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["sh", "-c", "python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port ${AFFINETES_PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
