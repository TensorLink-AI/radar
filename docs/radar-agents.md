# Radar Agents — What They Can and Can't Access

Miner agents run inside the official `radar-agent` Docker image (built from
`runner/agent/Dockerfile`). The image is owned and frozen by the subnet
owner; miners only ship `.py` files that the validator injects at runtime
into `/workspace/agent/`. The frozen harness (`runner/agent/harness.py`)
loads the module, calls `design_architecture(challenge, client)`, and
captures the returned dict as the round's `Proposal`.

This document lists every interface the agent has — and explicitly the
ones it does not.

## Entry point

```python
def design_architecture(challenge: dict, client: GatedClient) -> dict:
    return {
        "code": "...",        # required
        "name": "...",
        "motivation": "...",
        "reasoning": "...",   # optional, self-reported
        "tool_calls": [...],  # optional, self-reported
    }
```

`stdout` is reserved for the JSON proposal; the harness redirects miner
`print()` calls to `stderr` so they don't corrupt the wire output.

## What's in `challenge`

The dict is a JSON-decoded `shared.protocol.Challenge`. Fields the agent
may read (full schema in `shared/protocol.py`):

| Field | Purpose |
|---|---|
| `challenge_id`, `round_id`, `seed`, `eval_split_seed` | Round identity / determinism |
| `min_flops_equivalent`, `max_flops_equivalent` | Hard size-bucket bounds for this round |
| `task` | TaskSpec dict (objective, context/prediction lengths, quantiles, etc.) |
| `feasible_frontier` | Pareto frontier filtered to this size bucket — list of `{code, metric, objectives}` |
| `db_url` | Validator proxy base URL — `/experiments/*`, `/challenge`, `/frontier`, `/provenance/*` |
| `desearch_url` | Validator proxy base URL — `/desearch/*` (arxiv search) |
| `llm_url` | Validator proxy base URL — `/llm/*` (OpenAI-compatible chat/completions) |
| `agent_token` | Per-round ephemeral secret. Auto-injected as `X-Agent-Token` by the harness. |
| `agent_seconds` | Phase A wall-clock budget for the pod (default 600s) |
| `scratchpad_get_url`, `scratchpad_put_url` | Presigned per-miner URLs for private state |
| `scratchpad_max_mb` | Hard size limit on the uploaded tarball (default 10MB) |
| `cognition_wiki_url` | Optional presigned GET to a per-task markdown corpus |

## Network access — `GatedClient` only

The agent has **no general internet access**. The harness builds a
`shared.url_gate.GatedClient` and passes it as the second argument. It is
the *only* way for agent code to make HTTP requests:

- The image ships with `urllib` (stdlib) only as a real HTTP path; `requests`,
  `aiohttp`, and `urllib3` are not installed.
- `httpx`, `openai`, and `anthropic` are pre-installed but use the same
  underlying network layer — they still must hit allowlisted hosts.
- The pod entrypoint (`runner/agent/entrypoint.sh`) programs `iptables` with
  an OUTPUT-DROP default and `ACCEPT` rules for each allowlisted host,
  defence-in-depth at the kernel layer (graceful fallthrough if
  `CAP_NET_ADMIN` is missing).

The allowlist is built per round by `validator/collection.py::_build_allowed_urls`
and contains:

1. The validator-configured `Config.AGENT_ALLOWED_URLS` (`RADAR_AGENT_ALLOWED_URLS`).
2. The base URLs of `db_url`, `desearch_url`, `llm_url`.
3. The per-request scratchpad presigned URLs.

Anything else (raw GitHub, HuggingFace Hub, the open internet) is blocked
both at `GatedClient` and at iptables.

`GatedClient` API the agent sees:

```python
client.get(url) -> bytes
client.get_json(url) -> dict
client.post(url, data) -> bytes
client.post_json(url, payload) -> dict
client.put(url, data, content_type=...) -> int    # used by scratchpad upload
```

Default timeouts: 10s connect/read, 30s for `/llm/*` URLs; 1 retry on
transient failures, 0 retries on LLM endpoints (too expensive). Every
request automatically carries the `X-Agent-Token`, `X-Miner-UID`, and
`X-Miner-Hotkey` headers — provenance heatmaps depend on them.

## Validator proxy surface

The proxy (`validator/db_proxy.py`) authenticates the agent via the
ephemeral token and forwards a strict prefix list to the central database
server. Routes the agent can call:

### `db_url` — experiment / frontier / provenance reads

These all proxy to `database/server.py` validator routes:

| Method + Path | Purpose |
|---|---|
| `GET /challenge` | Current challenge state |
| `GET /frontier` | Global Pareto frontier |
| `GET /experiments/recent?limit=N` | Recent experiment rows |
| `GET /experiments/pareto` | Frontier members |
| `GET /experiments/failures` | Past failed runs |
| `GET /experiments/stats`, `/experiments/stats/by_task`, `/experiments/tasks`, `/experiments/families` | Aggregates |
| `GET /experiments/{index}` | A single experiment row |
| `GET /experiments/{index}/diff`, `/experiments/{index}/lineage_diffs` | Code diffs |
| `GET /experiments/diff/{a}/{b}` | Pairwise diff |
| `GET /experiments/lineage/{index}` | Provenance lineage |
| `GET /experiments/{index}/verify` | Reproducibility check status |
| `POST /experiments/search` | Filtered search |
| `GET /provenance/{experiment_id}/influences`, `/impact`, `/similar`, `/graph` | Provenance graph |
| `GET /provenance/components`, `/component_stats`, `/dead_ends` | Component-level stats |

### `desearch_url` — arxiv search proxy

`POST {desearch_url}/search` with `{"query": "...", "max_results": N}` —
hits the SN22 Desearch API behind the proxy. Rate-limited by
`Config.DESEARCH_MAX_QUERIES` per round (default 20, per-hour window).

### `llm_url` — LLM inference proxy

OpenAI-compatible. The `miner_template` example uses:

- `GET {llm_url}/models` — list available models
- `POST {llm_url}/chat` — `{"model", "messages", "temperature", "max_tokens"}`

The proxy lives at `/llm/v1/*` on the validator proxy when used through the
OpenAI SDK (`base_url=challenge["proxy_url"] + "/llm/v1"`,
`api_key=challenge["agent_token"]`). Rate-limited by `Config.LLM_MAX_QUERIES`
per round (default 50, per-hour window).

### Scratchpad — per-miner persistent state

The harness injects two helpers into the agent's module namespace:

```python
scratch_dir = load_scratchpad(challenge)        # downloads + extracts to /tmp/scratchpad
# ... read / write files in scratch_dir ...
save_scratchpad(challenge, scratch_dir)         # tar.gz + PUT, retried 3x
```

- Presigned URLs are unique per request and expire after the round.
- Size limit: `challenge["scratchpad_max_mb"]` (default 10MB) — enforced
  client-side; oversize tarballs are *not* uploaded.
- A 404 on the GET means "first round" (start fresh, no error).
- A 403 on PUT means the URL expired (retries skipped — they won't help).
- Storage is isolated per miner hotkey on the validator's artifact backend
  (Hippius / R2).

## Rate limits (per round, validator-side)

| Category | Default | Env var | Window |
|---|---|---|---|
| `db` | 60 req/min | `RADAR_DB_VALI_RATE_LIMIT` | 60s |
| `desearch` | 20 req/round | `RADAR_DESEARCH_MAX_QUERIES` | 3600s |
| `llm` | 50 req/round | `RADAR_LLM_MAX_QUERIES` | 3600s |

Hitting a limit returns HTTP 429 and is recorded in the agent behaviour log.

## What's NOT accessible

The image is deliberately stripped. Agent code does **not** get:

- **General internet** — anything outside the per-round allowlist (HF Hub,
  GitHub, pip, npm, package registries, your own server). Blocked at
  `GatedClient` and iptables.
- **Raw HTTP libs** — `requests`, `aiohttp`, `urllib3` are not installed.
  `urllib` is stdlib but every request still has to go through allowlisted
  hosts; the `iptables` layer enforces it even if you bypass `GatedClient`.
- **Bittensor SDK / wallet / chain RPC writes** — agents cannot sign
  extrinsics, set weights, read other neurons' wallets, or open a
  subtensor connection. `SUBTENSOR_NETWORK` / `NETUID` env vars are
  read-only metadata.
- **Validator / miner secrets** — `R2_*`, `HIPPIUS_*`, `BASILICA_*`,
  `TARGON_*`, `WALLET_*` env vars are stripped before the pod starts.
  Only an explicit allowlist of vars is forwarded (see
  `MINER_ENVIRONMENT.md`).
- **The trainer image** — agent and trainer are separate Docker images
  with separate frozen harnesses. The agent does not run training; it
  emits the `code` string that the trainer will execute later.
- **Mutating DB writes** — `POST /experiments/add`, `POST /frontier/update`,
  `POST /provenance/record_*`, `POST /round_submissions/reveal`,
  `POST /agent_code` are validator-only routes. The agent token only
  authorises the read prefixes listed above.
- **Other miners' scratchpads** — presigned URLs are scoped to the
  caller's hotkey.
- **Other miners' agent code at write time** — agents *can* read their
  own code rows via `/agent_code/{hotkey}` paths exposed on the validator
  surface, but cannot overwrite another hotkey's code.
- **Stdout for arbitrary use** — reserved for the proposal JSON. Use
  `print(..., file=sys.stderr)` or the injected `log()` helper.
- **Subprocesses, GPUs, persistent disk** — agent pods are CPU-only,
  ephemeral, and killed at `agent_seconds`. PyTorch ships as the CPU
  build; do model design, not training, here.
- **Long-lived state across rounds** — only the scratchpad survives.
  `/tmp` and `/workspace/agent/` are reset per pod.

## Filesystem

```
/workspace/
  harness.py           # 444, frozen
  shared/url_gate.py   # 444, frozen
  agent/               # your bundle, written here per round
    agent.py
    ...
  entrypoint.sh        # programs iptables, then exec's the server
/app/
  env.py               # affinetes Actor
  _affinetes/server.py # HTTP server that calls process_challenge()
/tmp/scratchpad/       # populated by load_scratchpad(), uploaded by save_scratchpad()
```

## Pre-installed Python packages

Image pins (see `runner/agent/Dockerfile`):

| Package | Notes |
|---|---|
| `torch` (CPU build) | Architecture inspection only — *don't train* in the agent |
| `numpy`, `safetensors` | Numeric / weight serialisation |
| `fastapi`, `uvicorn`, `httpx`, `pydantic` | Affinetes server runtime |
| `openai>=1.50`, `anthropic>=0.39` | LLM SDKs (point them at `llm_url`) |
| `affinetes` | The HTTP-server scaffold the validator drives |

Anything else is unavailable. Bundling extra `.py` files in your agent
directory works; bundling extra wheels does not.

## Self-reported fields

`reasoning` (string) and `tool_calls` (list) come back unmodified to the
public dashboard and are saved with the experiment row. They are
**self-reported** and trivially fakeable. For trustworthy provenance,
pair them with the validator-observed proxy access log (`X-Miner-UID`
+ `X-Miner-Hotkey` are stamped on every proxied request).

## See also

- `docs/MINER_ENVIRONMENT.md` — sandbox image, env vars, networking depth
- `runner/agent/harness.py` — entry point, scratchpad helpers, gating
- `shared/url_gate.py` — `GatedClient` source
- `shared/protocol.py` — `Challenge` / `Proposal` wire format
- `validator/db_proxy.py` — proxy auth, rate limits, route allowlist
- `validator/collection.py::_build_allowed_urls` — per-round allowlist build
- `miner_template/agent.py` — runnable starter agent
