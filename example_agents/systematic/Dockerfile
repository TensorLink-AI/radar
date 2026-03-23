FROM python:3.11-slim

# App dependencies
RUN pip install --no-cache-dir requests

# Agent code + affinetes Actor wrapper
COPY run.py /app/run.py
COPY env.py /app/env.py

# affinetes HTTP server layer (replicates two-stage build)
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx pydantic
RUN mkdir -p /app/_affinetes && echo "" > /app/_affinetes/__init__.py
COPY http_server.py /app/_affinetes/server.py

WORKDIR /app
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port ${AFFINETES_PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
