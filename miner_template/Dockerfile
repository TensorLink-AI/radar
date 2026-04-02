FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent.py .

# Chutes API key must be provided at runtime:
#   docker run -e CHUTES_API_KEY=<key> my-agent:latest
# Optional overrides:
#   CHUTES_MODEL (default: deepseek-ai/DeepSeek-V3-0324)
#   CHUTES_BASE_URL (default: https://llm.chutes.ai/v1)
#   LLM_TEMPERATURE (default: 0.7)
#   LLM_MAX_TOKENS (default: 8192)

CMD ["python", "agent.py"]
