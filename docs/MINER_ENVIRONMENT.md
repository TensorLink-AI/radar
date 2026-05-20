# Miner Agent Environment

The validator launches every miner agent inside the official `radar-agent`
Docker image (`runner/agent/Dockerfile`). The image is controlled by the
subnet owner; miners only supply `.py` files that are injected at runtime.
This document describes what the sandbox offers miners and how egress is
constrained.

## Pre-installed Python packages

The image ships with both major LLM SDKs pre-installed. Miners can pick
whichever they prefer — or none — without rebuilding the image:

| Package       | Version    | Purpose                               |
| ------------- | ---------- | ------------------------------------- |
| `openai`      | `>=1.50`   | OpenAI-compatible chat/completions    |
| `anthropic`   | `>=0.39`   | Anthropic Messages API                |
| `httpx`       | (latest)   | Underlying HTTP client (shared dep)   |
| `pydantic`    | (latest)   | Data models (shared dep)              |
| `fastapi`     | (latest)   | Affinetes HTTP server runtime         |
| `torch`       | cpu build  | Inference in the agent is discouraged |
| `numpy`       | (latest)   | General numeric work                  |
| `safetensors` | (latest)   | Weight (de)serialisation              |

The validator's LLM proxy is OpenAI-compatible and lives at
`/llm/v1/*` on the validator proxy URL passed to the agent through the
challenge. Typical use with each SDK:

```python
# OpenAI SDK (recommended for most agents)
from openai import OpenAI
client = OpenAI(
    base_url=challenge["proxy_url"] + "/llm/v1",
    api_key=challenge["agent_token"],
)
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "..."}],
)

# Anthropic SDK (works because the proxy speaks OpenAI wire format;
# use the OpenAI-compatible shim or call via the injected GatedClient
# if you need the exact Anthropic Messages schema).
```

**LangGraph / LangChain / DSPy** can be installed by the miner as part of
their bundle (the agent bundle supports additional `.py` files), but note
that bringing extra packages at runtime is not supported — anything not
pre-installed is unavailable inside the sandbox. File a request if you
need something added to the base image.

## Networking

The agent has **no general internet access**. Egress is enforced in two
layers:

1. **Application layer — `GatedClient`.** The harness injects a
   `GatedClient` into `design_architecture(challenge, client)`. Every
   request goes through the gate and is rejected unless it matches an
   allowed URL prefix. If you use `openai` or `anthropic` directly, they
   share the same underlying `httpx` stack; their base URLs still have to
   be in the allowlist.

2. **Network layer — `iptables`.** Before the harness starts, the pod
   entrypoint (`runner/agent/entrypoint.sh`) programs iptables with an
   OUTPUT-DROP default policy and `ACCEPT` rules for each host in
   `RADAR_ALLOWED_URLS`. This is defence-in-depth: even if a miner
   bypasses `GatedClient`, unauthorised destinations are dropped at the
   kernel.

The allowlist is the same variable in both layers:

```
RADAR_ALLOWED_URLS=https://validator-proxy.example/,https://other-allowed.example/
```

The validator builds this list once per round from
`Config.AGENT_ALLOWED_URLS` + per-round presigned URLs. Miners do not
need to set it themselves.

### Graceful degradation

If the pod runs without `CAP_NET_ADMIN` (Basilica may not always grant
it), `iptables` is unavailable and the entrypoint falls through with a
warning. Security is not lost — `GatedClient` still gates at the
application layer — but the defence-in-depth layer is absent.

## Environment variables

| Variable              | Set by    | Seen by                           |
| --------------------- | --------- | --------------------------------- |
| `RADAR_ALLOWED_URLS`  | validator | entrypoint.sh (iptables)          |
| `AGENT_ALLOWED_URLS`  | validator | harness.py → `GatedClient`        |
| `AGENT_MODULE`        | validator | harness.py (entry-point path)     |

R2 credentials, Basilica tokens, and arbitrary pass-through vars are
**not** forwarded into agent pods.

## Filesystem layout

```
/workspace/
  harness.py             # frozen, read-only (444)
  agent/                 # miner bundle is written here at runtime
    agent.py             # entry point by default
    ...
  shared/
    url_gate.py          # frozen (444)
  entrypoint.sh          # root-owned, programs iptables then exec's CMD
/app/
  env.py                 # Affinetes Actor
  _affinetes/server.py   # HTTP server that calls process_challenge()
/tmp/scratchpad/         # persistent state (uploaded via presigned URL)
```

## Writing an agent

```python
def design_architecture(challenge: dict, client) -> dict:
    # client is a GatedClient — your only way to reach the network.
    # For LLM calls, either:
    #   (a) use client directly (safest, no extra imports), or
    #   (b) use openai.OpenAI(base_url=..., http_client=client.as_httpx())
    ...
    return {"code": "...", "name": "...", "motivation": "..."}
```

The scratchpad helpers `load_scratchpad(challenge)` /
`save_scratchpad(challenge)` are injected into your module namespace by
the harness — you do not need to import them.
