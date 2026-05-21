FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps:
#   build-essential, libpq-dev — wheels that may need to compile (asyncpg
#     usually has manylinux wheels).
#   git — affinetes is declared in pyproject as a git+https dependency
#     and pip needs git on PATH to clone it.
#   ca-certificates — TLS to Postgres / outbound HTTPS.
# CUDA / GPU deps deliberately omitted; the DB neuron is CPU-only.
# Training images live in runner/Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      git \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps from pyproject.toml. requirements.txt in this repo
# is a partial list (core libs only) and does not include fastapi,
# uvicorn, pydantic, jinja2, etc. that database/neuron.py imports —
# installing only requirements.txt yields an unbootable image.
COPY pyproject.toml requirements.txt ./
COPY shared/ shared/
COPY validator/ validator/
COPY miner/ miner/
COPY database/ database/
COPY config.py ./

RUN pip install -r requirements.txt \
    && pip install .

# Bring in the rest of the build context (tasks, scripts, etc.) so
# things like task YAMLs are resolvable at runtime.
COPY . .

# Default port for local runs; Railway overrides with $PORT at runtime.
ENV PORT=8090

# Non-root user for the runtime.
RUN useradd --create-home --shell /bin/bash radar \
    && chown -R radar:radar /app
USER radar

# Mode-specific behaviour (validator / dashboard / all) is selected via
# RADAR_NEURON_MODE at runtime — the image itself is mode-agnostic.
CMD ["sh", "-c", "python database/neuron.py --port ${PORT}"]
