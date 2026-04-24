"""
Configuration for RADAR.

All settings in one place. Override via environment variables prefixed with RADAR_.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # read .env file from project root


class Config:
    """Central configuration — edit here or set RADAR_* env vars."""

    # ── Runner Backend ────────────────────────────────────────────────
    # Where to execute training: "local" | "affinetes"
    RUNNER_BACKEND: str = os.getenv("RADAR_RUNNER_BACKEND", "local")

    # ── LLM Configuration ──────────────────────────────────────────────
    LLM_MODEL_STRONG: str = os.getenv("RADAR_LLM_MODEL_STRONG", "o3")  # for planning

    # ── Desearch Proxy ────────────────────────────────────────────────
    # Desearch API base URL (path /desearch/ai/search/links/web is appended)
    DESEARCH_SN22_URL: str = os.getenv("RADAR_DESEARCH_SN22_URL", "https://api.desearch.ai")
    # Desearch API key (subnet owner provides). Sent as `Authorization: <key>`.
    DESEARCH_API_KEY: str = os.getenv("RADAR_DESEARCH_API_KEY", "")
    # Max queries per miner per tempo
    DESEARCH_MAX_QUERIES: int = int(os.getenv("RADAR_DESEARCH_MAX_QUERIES", "20"))
    # Enable desearch proxy (set to "true" to enable)
    DESEARCH_ENABLED: bool = os.getenv("RADAR_DESEARCH_ENABLED", "false").lower() == "true"

    # ── LLM Proxy (Chutes AI) ────────────────────────────────────────
    # Chutes AI inference endpoint
    CHUTES_API_URL: str = os.getenv("RADAR_CHUTES_API_URL", "https://llm.chutes.ai/v1")
    # Chutes AI API key (subnet owner provides)
    CHUTES_API_KEY: str = os.getenv("RADAR_CHUTES_API_KEY", "")
    # Comma-separated list of allowed model names (empty = all)
    CHUTES_ALLOWED_MODELS: str = os.getenv("RADAR_CHUTES_ALLOWED_MODELS", "")
    # Max LLM queries per miner per tempo
    LLM_MAX_QUERIES: int = int(os.getenv("RADAR_LLM_MAX_QUERIES", "50"))
    # Enable LLM proxy (set to "true" to enable)
    LLM_ENABLED: bool = os.getenv("RADAR_LLM_ENABLED", "false").lower() == "true"
    # Timeout (seconds) for each Chutes AI request
    CHUTES_TIMEOUT: float = float(os.getenv("RADAR_CHUTES_TIMEOUT", "120"))

    # ── Official Training Image ──────────────────────────────────────
    OFFICIAL_TRAINING_IMAGE: str = os.getenv("OFFICIAL_TRAINING_IMAGE", "ghcr.io/tensorlink-ai/radar/radar-runner:latest")
    OFFICIAL_TRAINING_IMAGE_DIGEST: str = os.getenv("OFFICIAL_TRAINING_IMAGE_DIGEST", "")

    # ── Multi-Task ──────────────────────────────────────────────────
    # Comma-separated list of enabled task names. Empty or "all" = all built-in tasks.
    ENABLED_TASKS: str = os.getenv("RADAR_ENABLED_TASKS", "ts_forecasting")

    # ── R2 Audit Log ────────────────────────────────────────────────
    R2_ACCOUNT_ID: str = os.getenv("R2_ACCOUNT_ID", "")
    R2_ACCESS_KEY_ID: str = os.getenv("R2_ACCESS_KEY_ID", "")
    R2_SECRET_ACCESS_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
    R2_BUCKET: str = os.getenv("R2_BUCKET", "")
    R2_PRESIGNED_TTL: int = int(os.getenv("RADAR_PRESIGNED_TTL", "5400"))

    # ── Round Timing ───────────────────────────────────────────────
    #
    # Timing is controlled at three distinct layers. Read this carefully —
    # the names have historically been confusing.
    #
    #   (1) BLOCK WINDOWS (on-chain, ~12s/block)
    #       Rigid phase boundaries agreed via consensus block height.
    #       Validator-global. Env vars below (RADAR_*_WINDOW).
    #
    #   (2) VALIDATOR OPERATIONAL TIMEOUTS (seconds, validator-side)
    #       Soft guardrails on HTTP calls / R2 polling. Validator-global.
    #       e.g. TRAINER_PREPARE_TIMEOUT (wait for TrainerReady).
    #
    #   (3) PER-TASK SECOND BUDGETS (seconds, set in task YAML)
    #       Declared in tasks/<task>/<task>.yaml, so different tasks can
    #       demand different amounts of work:
    #
    #         agent_seconds : Phase A — agent pod wall-clock.
    #                         0/unset ⇒ inherit Config.AGENT_TIMEOUT.
    #         time_budget   : Phase B — trainer *training-loop* wall-clock
    #                         (how long the harness runs training).
    #         kill_timeout  : Phase B — hard subprocess kill for trainer
    #                         run/eval commands. Always >= time_budget.
    #
    # Per phase:
    #   Phase A (Submission):  SUBMISSION_WINDOW_BLOCKS × 12s = outer boundary
    #                          task.agent_seconds (or AGENT_TIMEOUT)  = inner agent cap
    #   Phase B (Training):    TRAINING_WINDOW_BLOCKS × 12s = outer boundary
    #                          task.time_budget             = inner trainer-loop cap
    #   Phase C (Evaluation):  EVAL_WINDOW_BLOCKS × 12s     = outer boundary AND R2 polling timeout
    #   Fallback:              FALLBACK_WINDOW_BLOCKS × 12s = outer boundary AND polling timeout
    #
    # Example with defaults:
    #   Phase A: 50 blocks (600s), agent_seconds=600s     → agent gets ~full window
    #   Phase B: 150 blocks (1800s), time_budget=300s     → training loop 5 min of 30 min window
    #   Phase C: 25 blocks (300s)                         → checkpoint polling for ~5 min
    #   Fallback: 50 blocks (600s)                        → re-dispatch polling for ~10 min
    #   Total: 275 blocks (~55 min)

    # Block windows (on-chain phase boundaries, ~12s per block)
    ROUND_INTERVAL_BLOCKS: int = int(os.getenv("RADAR_ROUND_INTERVAL", "275"))
    SUBMISSION_WINDOW_BLOCKS: int = int(os.getenv("RADAR_SUBMISSION_WINDOW", "50"))
    TRAINING_WINDOW_BLOCKS: int = int(os.getenv("RADAR_TRAINING_WINDOW", "150"))
    EVAL_WINDOW_BLOCKS: int = int(os.getenv("RADAR_EVAL_WINDOW", "25"))
    FALLBACK_ENABLED: bool = os.getenv("RADAR_FALLBACK_ENABLED", "true").lower() == "true"
    FALLBACK_WINDOW_BLOCKS: int = int(os.getenv("RADAR_FALLBACK_WINDOW", "50"))

    # Liveness filter: validators whose on-chain last_update is older than
    # this many blocks are treated as offline and skipped when computing
    # work-splitting / dispatch assignments. Default ~2 hours (600 × 12s).
    VALIDATOR_STALE_BLOCKS: int = int(os.getenv("RADAR_VALIDATOR_STALE_BLOCKS", "600"))

    SKIP_TRAINING_WAIT: bool = os.getenv("RADAR_SKIP_TRAINING_WAIT", "false").lower() == "true"
    SIZE_GATE_TOLERANCE: float = float(os.getenv("RADAR_SIZE_GATE_TOLERANCE", "0.10"))

    # ── Agent (Phase A) ──────────────────────────────────────────
    # Default agent pod wall-clock (seconds). Per-task overrides come from
    # `agent_seconds` in the task YAML; this is the fallback when a task
    # doesn't set one. Must fit within SUBMISSION_WINDOW_BLOCKS × 12s.
    AGENT_TIMEOUT: int = int(os.getenv("RADAR_AGENT_TIMEOUT", "600"))
    AGENT_POD_RETRIES: int = int(os.getenv("RADAR_AGENT_POD_RETRIES", "3"))

    # Max number of agent pods this validator runs concurrently in Phase A.
    # Each pod is on a separate Basilica node so there's no local resource
    # contention; the cap exists to avoid overwhelming the orchestration
    # layer / R2 with many simultaneous starts. Set to 1 for the old
    # sequential behavior.
    AGENT_CONCURRENCY: int = int(os.getenv("RADAR_AGENT_CONCURRENCY", "8"))

    # Official agent image (subnet-owner controlled, locked down)
    OFFICIAL_AGENT_IMAGE: str = os.getenv(
        "RADAR_OFFICIAL_AGENT_IMAGE",
        "ghcr.io/tensorlink-ai/radar/radar-agent:latest",
    )
    # Comma-separated URL prefixes that agent pods are allowed to reach.
    # The validator proxy and presigned R2 URLs are added automatically.
    AGENT_ALLOWED_URLS: str = os.getenv("RADAR_AGENT_ALLOWED_URLS", "")

    # ── Warm-Standby Trainer ────────────────────────────────────
    TRAINER_PREPARE_TIMEOUT: int = int(os.getenv("RADAR_TRAINER_PREPARE_TIMEOUT", "600"))
    TRAINER_READY_POLL_INTERVAL: int = int(os.getenv("RADAR_TRAINER_READY_POLL", "15"))
    TRAINER_RELEASE_SAFETY_MARGIN: float = float(os.getenv("RADAR_TRAINER_SAFETY_MARGIN", "1.1"))

    # Default GPU spec sent in TrainerRequest (validator-controlled)
    TRAINER_GPU_COUNT: int = int(os.getenv("RADAR_TRAINER_GPU_COUNT", "1"))
    TRAINER_MIN_GPU_MEMORY_GB: int = int(os.getenv("RADAR_TRAINER_MIN_GPU_MEMORY_GB", "16"))
    TRAINER_GPU_MODELS: str = os.getenv("RADAR_TRAINER_GPU_MODELS", "")  # comma-separated, empty = any GPU
    TRAINER_MEMORY: str = os.getenv("RADAR_TRAINER_MEMORY", "16Gi")

    # Subnet-owner fallback proxy — handles jobs from non-responsive trainers
    FALLBACK_PROXY_URL: str = os.getenv("RADAR_FALLBACK_PROXY_URL", "")

    # ── Phase C Eval ─────────────────────────────────────────────
    EVAL_DEVICE: str = os.getenv("RADAR_EVAL_DEVICE", "cpu")

    # ── Scoring ──────────────────────────────────────────────────
    EMA_ALPHA: float = float(os.getenv("RADAR_EMA_ALPHA", "0.3"))
    SOFTMAX_TEMPERATURE: float = float(os.getenv("RADAR_SOFTMAX_TEMP", "0.1"))
    # Minimum fractional CRPS improvement over the feasible frontier's best
    # CRPS required to earn a nonzero score. 0.005 = 0.5%. Set to 0 to keep
    # the old behaviour (any positive sigmoid score counts).
    FRONTIER_IMPROVEMENT_THRESHOLD: float = float(
        os.getenv("RADAR_FRONTIER_IMPROVEMENT_THRESHOLD", "0.005"),
    )

    # ── Agent Scratchpad ────────────────────────────────────────
    SCRATCHPAD_ENABLED: bool = os.getenv("RADAR_SCRATCHPAD_ENABLED", "true").lower() == "true"
    SCRATCHPAD_MAX_MB: int = int(os.getenv("RADAR_SCRATCHPAD_MAX_MB", "10"))
    SCRATCHPAD_TTL: int = int(os.getenv("RADAR_SCRATCHPAD_TTL", "1800"))

    # ── Query API ────────────────────────────────────────────────
    QUERY_RATE_LIMIT: int = int(os.getenv("RADAR_QUERY_RATE_LIMIT", "10"))

    # ── GIFT-Eval Benchmark Data (Phase C evaluation) ──────────────
    GIFT_EVAL_R2_BUCKET: str = os.getenv("RADAR_GIFT_EVAL_BUCKET", "gift-eval-benchmark")
    GIFT_EVAL_R2_PREFIX: str = os.getenv("RADAR_GIFT_EVAL_PREFIX", "gift-eval-full")
    GIFT_EVAL_CACHE_DIR: str = os.getenv("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")
    GIFT_EVAL_DATASETS_PER_ROUND: int = int(os.getenv("RADAR_GIFT_EVAL_DATASETS", "0"))  # 0 = all datasets
    GIFT_EVAL_MAX_SERIES_PER_DATASET: int = int(os.getenv("RADAR_GIFT_EVAL_MAX_SERIES", "500"))

    # ── Pretrain Data (Phase B training) ─────────────────────────
    PRETRAIN_R2_BUCKET: str = os.getenv("RADAR_PRETRAIN_BUCKET", "gift-eval-pretrain")
    PRETRAIN_R2_PREFIX: str = os.getenv("RADAR_PRETRAIN_PREFIX", "datasets/radar/v1")
    PRETRAIN_SHARDS_PER_ROUND: int = int(os.getenv("RADAR_PRETRAIN_SHARDS", "8"))
    PRETRAIN_SHUFFLE_BUFFER: int = int(os.getenv("RADAR_PRETRAIN_SHUFFLE_BUFFER", "10000"))

    # ── Postgres ──────────────────────────────────────────────
    PG_DSN: str = os.getenv("RADAR_PG_DSN", "postgresql://radar:radar@localhost:5432/radar")
    # TLS mode for the asyncpg pool:
    #   ""        → no TLS (local Docker)
    #   "require" → TLS without verification. DEPRECATED: preserves the
    #               historical behaviour that skipped hostname/cert checks,
    #               kept for operators who rely on it. Emits a runtime
    #               warning.  Prefer "verify" for any managed Postgres.
    #   "verify"  → TLS with full hostname + certificate verification.
    #               Required for Crunchy Bridge / any production
    #               managed-Postgres deployment.
    PG_SSL: str = os.getenv("RADAR_PG_SSL", "")
    # asyncpg pool sizing. Total max across all DB-server processes
    # should stay under the cluster's max_connections. Validator mode
    # typically wants larger pools (many concurrent writers); dashboard
    # mode wants smaller pools paired with a statement_timeout so a slow
    # public query can't pressure the shared cluster.
    PG_POOL_MIN: int = int(os.getenv("RADAR_PG_POOL_MIN", "2"))
    PG_POOL_MAX: int = int(os.getenv("RADAR_PG_POOL_MAX", "10"))
    # Set to 0 when the DSN points at a transaction-mode pooler
    # (Supabase Supavisor, PgBouncer) that forbids server-side
    # prepared statements. Leave unset for direct connections.
    PG_STATEMENT_CACHE_SIZE: str = os.getenv("RADAR_PG_STATEMENT_CACHE_SIZE", "")
    # `SET statement_timeout` applied on every pool connection. 0 = no
    # timeout. Set to 5000 (5s) on dashboard-mode deploys so a slow
    # public query can't pressure the shared Crunchy cluster.
    PG_STATEMENT_TIMEOUT_MS: int = int(os.getenv("RADAR_PG_STATEMENT_TIMEOUT_MS", "0"))

    # Startup retry budget for the asyncpg bootstrap connect + pool
    # create. Covers containers that come up before Postgres is ready
    # (Railway / k8s sidecar cold start, Crunchy Bridge rolling restart,
    # DNS propagation) so a transient ECONNREFUSED doesn't crash-loop
    # the process. Defaults to 6 retries with 1s initial backoff doubling
    # up to 30s, for roughly one minute of total grace before giving up.
    PG_STARTUP_RETRIES: int = int(os.getenv("RADAR_PG_STARTUP_RETRIES", "6"))
    PG_STARTUP_BACKOFF_INITIAL_S: float = float(
        os.getenv("RADAR_PG_STARTUP_BACKOFF_INITIAL_S", "1.0")
    )
    PG_STARTUP_BACKOFF_MAX_S: float = float(
        os.getenv("RADAR_PG_STARTUP_BACKOFF_MAX_S", "30.0")
    )

    # ── Network / Schema Isolation ────────────────────────────
    # Which Bittensor network this DB process serves: "testnet" or "mainnet".
    # A single Postgres database holds BOTH networks; isolation is enforced by
    # Postgres *schemas* (NOT a `network` tag column, NOT separate DBs).
    #
    # Why schemas, not a tag column:
    #   - zero changes to existing SQL (search_path handles qualification)
    #   - impossible to accidentally leak cross-network rows via a forgotten
    #     WHERE clause — the other schema's tables are simply invisible
    #   - per-network backup / drop / restore is trivial (pg_dump --schema=...)
    #   - per-schema sequences keep experiment IDs independent per network
    #
    # The value is embedded in `SET search_path` at connection time, so the
    # allowlist regex in shared/pg_store.py is the *only* thing preventing
    # SQL injection via this variable. Defaults to "testnet" for safety.
    NETWORK: str = os.getenv("RADAR_NETWORK", "testnet")

    # ── Neuron Mode ───────────────────────────────────────────
    # RADAR_NEURON_MODE controls which surface this process serves.
    #   validator  — Epistula-authed write/read API for validators + miners,
    #                plus desearch/LLM proxies. Runs metagraph sync + round
    #                loop. Requires a Bittensor wallet. Optionally also mounts
    #                the internal Jinja dashboard when RADAR_DASHBOARD_ENABLED=true.
    #   dashboard  — Open public JSON API at /dashboard/api/*. No wallet, no
    #                proxies, no metagraph sync, no Jinja, no Epistula.
    #                This is what gets deployed to Railway behind radarnet.io.
    #   all        — Everything on one process (legacy / dev default).
    NEURON_MODE: str = os.getenv("RADAR_NEURON_MODE", "all").lower()

    # CORS origins for the public JSON API. Comma-separated, empty = no CORS
    # (safe default for validator mode, which doesn't speak to browsers).
    # In production dashboard mode set this to "https://radarnet.io".
    DASHBOARD_CORS_ORIGINS: str = os.getenv("RADAR_DASHBOARD_CORS_ORIGINS", "")

    # ── Database API ──────────────────────────────────────────
    DB_API_URL: str = os.getenv("RADAR_DB_API_URL", "")
    DB_API_PORT: int = int(os.getenv("RADAR_DB_API_PORT", "8090"))

    # ── Miner Listener ───────────────────────────────────────
    MINER_LISTENER_PORT: int = int(os.getenv("RADAR_MINER_LISTENER_PORT", "8091"))

    # ── Validator Proxy ───────────────────────────────────────
    PROXY_PORT: int = int(os.getenv("RADAR_PROXY_PORT", "8080"))
    # External URL for the validator proxy — used in challenge JSON so
    # agent pods on Basilica can reach this validator.  If empty, falls
    # back to http://localhost:{PROXY_PORT} (only works for local Docker).
    VALIDATOR_EXTERNAL_URL: str = os.getenv("RADAR_VALIDATOR_EXTERNAL_URL", "")

    # ── Database Auth ─────────────────────────────────────────
    DB_VALI_RATE_LIMIT: int = int(os.getenv("RADAR_DB_VALI_RATE_LIMIT", "60"))
    # Shared subnet API key — lightweight gate before Epistula verification.
    # Subnet owner generates once, distributes to validators for reverse proxy auth.
    # Empty string = disabled (open access, Epistula-only).
    DB_API_KEY: str = os.getenv("RADAR_DB_API_KEY", "")

    # ── Validator event stream (wandb-style log + metric tail) ────────
    # Validators batch log lines + scalar metrics in-memory and flush to
    # the centralized DB via /events. The dashboard SPA polls
    # /dashboard/api/validators/{hotkey}/events for a live tail.
    EVENTS_ENABLED: bool = os.getenv("RADAR_EVENTS_ENABLED", "true").lower() == "true"
    # Seconds between flushes from the validator's in-memory buffer.
    EVENTS_FLUSH_INTERVAL_S: float = float(os.getenv("RADAR_EVENTS_FLUSH_INTERVAL_S", "5"))
    # Soft cap on how many events the validator buffers between flushes.
    # Above this, the buffer drops the oldest events (so a stuck flush
    # never grows the validator's RAM without bound).
    EVENTS_BUFFER_MAX: int = int(os.getenv("RADAR_EVENTS_BUFFER_MAX", "2000"))
    # Hard cap on a single POST batch — keeps each ingest insert under
    # the DB server's 5 MB body limit even with verbose log lines.
    EVENTS_BATCH_MAX: int = int(os.getenv("RADAR_EVENTS_BATCH_MAX", "500"))
    # Drop events older than this. The pruner runs once per
    # ``EVENTS_PRUNE_INTERVAL_S`` on the DB server. 0 disables pruning.
    EVENTS_RETENTION_DAYS: int = int(os.getenv("RADAR_EVENTS_RETENTION_DAYS", "7"))
    EVENTS_PRUNE_INTERVAL_S: int = int(os.getenv("RADAR_EVENTS_PRUNE_INTERVAL_S", "3600"))
    # Loggers whose records are forwarded into the event stream. Empty
    # string means "everything attached to the root logger". Comma-
    # separated names match Python logging's hierarchy (e.g. "validator").
    EVENTS_LOGGER_NAMES: str = os.getenv("RADAR_EVENTS_LOGGER_NAMES", "")
    # Minimum log level captured by the handler (forwarded levels above
    # this are written; below are dropped). Values: debug, info, warning,
    # error, critical.
    EVENTS_LOG_LEVEL: str = os.getenv("RADAR_EVENTS_LOG_LEVEL", "info").lower()

    # ── Dashboard (read-only web UI for inspecting experiments) ───────
    DASHBOARD_ENABLED: bool = os.getenv("RADAR_DASHBOARD_ENABLED", "false").lower() == "true"
    # Shared key used to log into the dashboard. Empty = dashboard refuses
    # every request (fail-closed) even if ENABLED is true.
    DASHBOARD_KEY: str = os.getenv("RADAR_DASHBOARD_KEY", "")
    # Session cookie TTL in seconds (default 8h)
    DASHBOARD_SESSION_TTL: int = int(os.getenv("RADAR_DASHBOARD_SESSION_TTL", "28800"))
    # Max bytes of a log file streamed inline before falling back to presigned URL
    DASHBOARD_MAX_LOG_BYTES: int = int(os.getenv("RADAR_DASHBOARD_MAX_LOG_BYTES", "10485760"))
    DASHBOARD_PAGE_SIZE: int = int(os.getenv("RADAR_DASHBOARD_PAGE_SIZE", "50"))


VALID_NEURON_MODES = frozenset({"validator", "dashboard", "all"})


def validate_neuron_mode() -> None:
    """Fail-fast check: RADAR_NEURON_MODE must be one of the known modes."""
    mode = Config.NEURON_MODE
    if mode not in VALID_NEURON_MODES:
        raise ValueError(
            f"Invalid RADAR_NEURON_MODE={mode!r}. "
            f"Must be one of: {sorted(VALID_NEURON_MODES)}"
        )
