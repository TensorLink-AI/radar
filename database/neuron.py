"""Subnet owner / dashboard process — centralized Postgres experiment DB.

Two deployable modes, selected by ``RADAR_NEURON_MODE``:

  * ``validator``  — Epistula-authed write/read API for validators + miners,
                     plus desearch/LLM proxies. Runs metagraph sync + round
                     loop. Requires a Bittensor wallet. Optionally also mounts
                     the internal Jinja dashboard when
                     ``RADAR_DASHBOARD_ENABLED=true``.
  * ``dashboard``  — Open public JSON API at ``/dashboard/api/*``. No wallet,
                     no proxies, no metagraph sync, no Jinja, no Epistula.
                     This is what gets deployed to Railway behind radarnet.io.
  * ``all``        — Everything on one process (legacy / dev default).

Both modes share the same Postgres cluster with independent connection
pools, independent auth, and independent route sets.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Awaitable, Callable, Optional, TypeVar

import uvicorn

from config import Config, validate_neuron_mode
from database.server import (
    app, include_validator_routes,
    set_db, set_auth, set_challenge, set_frontier,
    set_access_logger, set_hotkey_map, set_rate_limit, set_ip_rate_limit,
    set_r2, set_hippius, set_pool,
    get_current_challenge, get_current_frontier,
)
from shared.migrations import apply_migrations
from shared.pareto import ParetoFront
from shared.pg_access_logger import PgAccessLogger
from shared.pg_store import (
    PgExperimentStore,
    create_pg_pool,
    ensure_schema_exists,
)
from shared.challenge import SIZE_BUCKETS as _DEFAULT_SIZE_BUCKETS
from shared.task import load_enabled_tasks

logger = logging.getLogger(__name__)


T = TypeVar("T")


# Errors worth retrying on Postgres startup. Everything here is a transient
# "server not reachable yet" failure that goes away once the DB is ready or
# DNS has propagated — OSError covers ECONNREFUSED from asyncpg's socket
# connect, ConnectionError covers asyncpg-wrapped forms, and the named
# asyncpg classes catch auth/timeout races when the server is coming up.
def _is_retryable_pg_startup_error(exc: BaseException) -> bool:
    import asyncpg

    if isinstance(exc, (OSError, ConnectionError, asyncio.TimeoutError)):
        return True
    if isinstance(
        exc,
        (
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.ConnectionFailureError,
        ),
    ):
        return True
    return False


async def _with_startup_retry(
    op: Callable[[], Awaitable[T]],
    *,
    what: str,
    retries: int,
    initial_backoff_s: float,
    max_backoff_s: float,
) -> T:
    """Run ``op`` with exponential backoff on transient connection errors.

    ``retries`` is the number of *retries* after the first attempt, so the
    total number of attempts is ``retries + 1``. The default Config values
    give roughly one minute of startup grace, which is plenty for a managed
    Postgres (Crunchy Bridge / Supabase) to finish a rolling restart or for
    a sidecar pg to come up behind the neuron container.
    """
    attempt = 0
    wait = initial_backoff_s
    while True:
        try:
            return await op()
        except BaseException as exc:  # noqa: BLE001 — retry decision below
            if attempt >= retries or not _is_retryable_pg_startup_error(exc):
                raise
            logger.warning(
                "%s failed (attempt %d/%d): %s — retry in %.1fs",
                what, attempt + 1, retries + 1, exc, wait,
            )
            await asyncio.sleep(wait)
            attempt += 1
            wait = min(wait * 2, max_backoff_s)


def _mode_runs_validator_surface(mode: str) -> bool:
    """True for modes that mount the Epistula-authed validator/miner routes."""
    return mode in ("validator", "all")


def _mode_runs_dashboard_api(mode: str) -> bool:
    """True for modes that expose the public JSON API at /dashboard/api/*."""
    return mode in ("dashboard", "all")


def _mode_runs_chain(mode: str) -> bool:
    """True for modes that read the chain (wallet / subtensor / metagraph)."""
    return mode in ("validator", "all")


class DatabaseNeuron:
    """Subnet owner database server — surface depends on ``Config.NEURON_MODE``."""

    def __init__(self, config):
        self.config = config
        self.mode = Config.NEURON_MODE
        validate_neuron_mode()

        self.netuid = getattr(config, "netuid", 1)
        self.port = int(getattr(config, "port", Config.DB_API_PORT))

        # Bittensor components — only constructed when this process needs to
        # talk to the chain. Dashboard mode never imports bittensor so it can
        # start without a wallet file on disk.
        self.wallet = None
        self.subtensor = None
        self.metagraph = None
        if _mode_runs_chain(self.mode):
            import bittensor as bt  # lazy import
            self.wallet = bt.Wallet(config=config)
            self.subtensor = bt.Subtensor(config=config)
            self.metagraph = self.subtensor.metagraph(self.netuid)

        # Tasks
        self.tasks = load_enabled_tasks(Config.ENABLED_TASKS)

        # Cache the dashboard's bucket resolver so it survives reloads of
        # ``self.tasks`` later. Tasks may declare their own size_buckets in
        # YAML; unknown task names fall back to the global default so the
        # dashboard still shows a sensible bucket layout for historical
        # experiments belonging to tasks no longer enabled here.
        self._get_task_buckets = self._build_task_bucket_resolver()

        # Per-task Pareto fronts (only populated in modes that run chain logic)
        self.pareto_fronts: dict[str, ParetoFront] = {}

        # R2 audit log — used by both validator write path and dashboard
        # log/bundle viewing, so it's shared across modes when configured.
        self.r2 = None
        if Config.R2_BUCKET:
            try:
                from shared.r2_audit import R2AuditLog
                self.r2 = R2AuditLog(bucket=Config.R2_BUCKET)
            except Exception as e:
                logger.warning("R2 audit log unavailable: %s", e)

        # Hippius client — opt-in via Config.HIPPIUS_ENABLED. Powers the
        # /experiments/{id}/verify endpoint on the validator surface; the
        # rest of the DB server doesn't need it. Lazy import + tolerant
        # construction mirrors the validator-side wiring (TEN-244).
        self.hippius = self._init_hippius()

        # Pool, store, logger set in async init
        self.pool = None
        self.store = None
        self.access_logger = None

    def _init_hippius(self):
        """Construct the Hippius client when opted in; return None otherwise.

        Mirrors `Validator._init_hippius`: lazy import so the module loads
        without the TEN-242 wrapper present, tolerant constructor so a flap
        at startup downgrades to disabled instead of crashing the DB server.
        """
        if not Config.HIPPIUS_ENABLED:
            return None
        try:
            from shared.hippius_client import HippiusClient  # type: ignore
        except ImportError:
            logger.warning(
                "HIPPIUS_ENABLED=true but shared.hippius_client is not "
                "available yet (TEN-242). /experiments/{id}/verify will "
                "respond 503 until the wrapper ships."
            )
            return None
        try:
            return HippiusClient(
                ipfs_api_url=Config.HIPPIUS_IPFS_API_URL,
                hippius_key=Config.HIPPIUS_KEY,
                substrate_rpc=Config.HIPPIUS_SUBSTRATE_RPC,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Hippius client init failed: %s", e)
            return None

    def _build_task_bucket_resolver(self) -> Callable[[str], list[tuple[int, int]]]:
        """Closure that maps a task name to its FLOPs-equivalent bucket list.

        Mirrors ``shared.challenge._resolve_task_buckets`` so the dashboard
        chart bins experiments into the same buckets that scoring uses, even
        when a task has overridden the global SIZE_BUCKETS in YAML.
        """
        default_buckets = list(_DEFAULT_SIZE_BUCKETS)

        def resolve(task_name: str) -> list[tuple[int, int]]:
            spec = self.tasks.get(task_name) if task_name else None
            if spec is None or not spec.size_buckets:
                return list(default_buckets)
            return [(int(lo), int(hi)) for lo, hi in spec.size_buckets]

        return resolve

    async def _init_db(self):
        """Create the asyncpg pool + ensure schemas exist.

        Mode-neutral: both validator and dashboard processes need a working
        pool and the full schema so queries either side issues succeed.
        Pool sizing and statement_timeout come from config so deployments
        can tighten dashboard-mode pressure independently.
        """
        import ssl as _ssl
        import warnings

        import asyncpg

        pg_dsn = getattr(self.config, "pg_dsn", None) or Config.PG_DSN
        network = Config.NETWORK

        pool_kwargs: dict = {
            "min_size": Config.PG_POOL_MIN,
            "max_size": Config.PG_POOL_MAX,
        }

        # TLS: "" → off, "verify" → full verification (recommended for
        # managed Postgres / Crunchy Bridge), "require" → TLS without
        # verification (deprecated, kept so operators relying on the
        # legacy skip-verify behaviour aren't silently changed).
        if Config.PG_SSL == "verify":
            pool_kwargs["ssl"] = _ssl.create_default_context()
        elif Config.PG_SSL == "require":
            warnings.warn(
                "RADAR_PG_SSL=require establishes TLS but skips hostname / "
                "certificate verification. This preserves pre-Crunchy "
                "behaviour and will be removed in a future release. Set "
                "RADAR_PG_SSL=verify for managed Postgres.",
                DeprecationWarning,
                stacklevel=2,
            )
            ctx = _ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            pool_kwargs["ssl"] = ctx
        elif Config.PG_SSL:
            raise ValueError(
                f"Unknown RADAR_PG_SSL value: {Config.PG_SSL!r}. "
                "Valid values: '' (off), 'require' (deprecated), 'verify'."
            )

        # Explicit opt-in to disable asyncpg's prepared-statement cache
        # (required by transaction-mode poolers like Supavisor / PgBouncer).
        # Direct Crunchy / local connections should leave this unset so
        # asyncpg's normal cache is used.
        if Config.PG_STATEMENT_CACHE_SIZE != "":
            pool_kwargs["statement_cache_size"] = int(Config.PG_STATEMENT_CACHE_SIZE)

        # Apply statement_timeout per-connection when configured — lets
        # dashboard-mode deployments cap slow public queries without
        # affecting the validator pool. Runs in `init=` (once per physical
        # connection) rather than `setup=` because statement_timeout
        # survives RESET ALL on release.
        if Config.PG_STATEMENT_TIMEOUT_MS > 0:
            timeout_ms = int(Config.PG_STATEMENT_TIMEOUT_MS)

            async def _apply_timeout(conn):
                await conn.execute(f"SET statement_timeout = {timeout_ms}")

            pool_kwargs["init"] = _apply_timeout

        # Bootstrap step: CREATE SCHEMA must run on a connection whose
        # search_path is NOT already set to that schema (otherwise we're
        # asking the schema to create itself). Use a bare asyncpg.connect
        # so no pool init/setup hook runs. Both validator and dashboard
        # modes run this — dashboard mode starting before the validator
        # needs the schema to exist too, and CREATE SCHEMA IF NOT EXISTS
        # is idempotent.
        logger.info(
            "Bootstrapping Postgres schema %r (RADAR_NETWORK)", network,
        )
        bootstrap_kwargs = {
            k: pool_kwargs[k]
            for k in ("ssl", "statement_cache_size")
            if k in pool_kwargs
        }
        retries = Config.PG_STARTUP_RETRIES
        initial_backoff = Config.PG_STARTUP_BACKOFF_INITIAL_S
        max_backoff = Config.PG_STARTUP_BACKOFF_MAX_S

        bootstrap_conn = await _with_startup_retry(
            lambda: asyncpg.connect(pg_dsn, **bootstrap_kwargs),
            what="Postgres bootstrap connect",
            retries=retries,
            initial_backoff_s=initial_backoff,
            max_backoff_s=max_backoff,
        )
        try:
            await ensure_schema_exists(bootstrap_conn, network)
        finally:
            await bootstrap_conn.close()

        # Every connection borrowed from this pool will have its search_path
        # pinned to `"<network>", public`, so all unqualified DDL/DML that
        # follows resolves to tables inside this schema.
        self.pool = await _with_startup_retry(
            lambda: create_pg_pool(pg_dsn, schema=network, **pool_kwargs),
            what="Postgres pool create",
            retries=retries,
            initial_backoff_s=initial_backoff,
            max_backoff_s=max_backoff,
        )
        self.store = PgExperimentStore(self.pool)
        await self.store.init_schema()

        self.access_logger = PgAccessLogger(self.pool)
        await self.access_logger.init_schema()

        # Init agent_submissions + agent_bundles + training_metas tables
        from shared.pg_schema import AGENT_CODE_SCHEMA
        async with self.pool.acquire() as conn:
            await conn.execute(AGENT_CODE_SCHEMA)

        # Init proxy query log table
        from shared.pg_schema import PROXY_QUERY_LOG_SCHEMA
        async with self.pool.acquire() as conn:
            await conn.execute(PROXY_QUERY_LOG_SCHEMA)

        # Apply any pending SQL deltas layered on top of the base schema.
        # Ordering rationale: ``ensure_schema_exists`` created the
        # namespace; ``init_schema`` already ran the idempotent
        # ``CREATE TABLE IF NOT EXISTS`` bootstrap for the tables declared
        # in ``pg_schema.py``; migrations now apply forward-only deltas
        # (ALTER TABLE, backfills, new indexes). On a fresh DB both paths
        # are no-ops for already-declared objects; on a restored DB the
        # bootstrap is the no-op and migrations do the real work. Fail
        # fast — if a migration errors, we let the exception propagate
        # and crash the DB server at startup so Railway keeps the
        # previous deploy live.
        async with self.pool.acquire() as conn:
            applied = await apply_migrations(conn, Config.NETWORK)
        if applied:
            logger.info(
                "Applied %d migrations to schema %r: %s",
                len(applied), Config.NETWORK, applied,
            )
        else:
            logger.info(
                "Applied 0 migrations to schema %r (already up to date)",
                Config.NETWORK,
            )

        # ── Shared wiring (both surfaces need DB access) ──
        set_db(self.store)
        set_pool(self.pool)
        if self.r2:
            set_r2(self.r2)
        if self.hippius:
            set_hippius(self.hippius)
        set_access_logger(self.access_logger)

        # ── Validator-surface wiring (Epistula auth + rate limit) ──
        if _mode_runs_validator_surface(self.mode):
            include_validator_routes(app)
            set_auth(self.metagraph)
            set_rate_limit(Config.DB_VALI_RATE_LIMIT)
            set_ip_rate_limit(Config.DB_IP_RATE_LIMIT)

            # Desearch proxy (SN22 arxiv search)
            if Config.DESEARCH_ENABLED:
                from validator.desearch_proxy import (
                    DesearchProxy, set_proxy, register_routes,
                )
                desearch = DesearchProxy(
                    sn22_url=Config.DESEARCH_SN22_URL,
                    api_key=Config.DESEARCH_API_KEY,
                    max_queries=Config.DESEARCH_MAX_QUERIES,
                    pool=self.pool,
                )
                set_proxy(desearch)
                register_routes(app)
                logger.info("Desearch proxy enabled (SN22: %s)", Config.DESEARCH_SN22_URL)

            # LLM proxy (Chutes AI)
            if Config.LLM_ENABLED:
                from validator.llm_proxy import LLMProxy
                from validator.llm_routes import set_proxy as set_llm_proxy
                from validator.llm_routes import register_routes as register_llm_routes
                allowed_models = [
                    m.strip() for m in Config.CHUTES_ALLOWED_MODELS.split(",")
                    if m.strip()
                ]
                llm = LLMProxy(
                    chutes_url=Config.CHUTES_API_URL,
                    chutes_api_key=Config.CHUTES_API_KEY,
                    allowed_models=allowed_models,
                    max_queries=Config.LLM_MAX_QUERIES,
                    pool=self.pool,
                    timeout=Config.CHUTES_TIMEOUT,
                )
                set_llm_proxy(llm)
                register_llm_routes(app)
                logger.info(
                    "LLM proxy enabled (Chutes: %s, models: %s)",
                    Config.CHUTES_API_URL, allowed_models or "all",
                )

            # Internal Jinja operator UI (still behind cookie auth)
            if Config.DASHBOARD_ENABLED:
                try:
                    from database.dashboard import mount_dashboard
                    mount_dashboard(
                        app,
                        store=self.store,
                        pool=self.pool,
                        r2=self.r2,
                        get_challenge=get_current_challenge,
                        get_frontier=get_current_frontier,
                        get_task_buckets=self._get_task_buckets,
                    )
                except Exception as e:
                    logger.warning("Dashboard mount failed: %s", e)

        # ── Public JSON API (no auth, served in dashboard + all modes) ──
        if _mode_runs_dashboard_api(self.mode):
            try:
                from database.dashboard import mount_public_api
                mount_public_api(
                    app,
                    store=self.store,
                    pool=self.pool,
                    r2=self.r2,
                    get_challenge=get_current_challenge,
                    get_frontier=get_current_frontier,
                    get_task_buckets=self._get_task_buckets,
                )
            except Exception as e:
                logger.warning("Public JSON API mount failed: %s", e)

        # ── CORS (only when origins are configured) ──
        self._maybe_mount_cors()

        logger.info(
            "Database initialized: mode=%s schema=%r pool_min=%d pool_max=%d ssl=%r",
            self.mode, Config.NETWORK,
            Config.PG_POOL_MIN, Config.PG_POOL_MAX, Config.PG_SSL or "off",
        )

    def _maybe_mount_cors(self):
        """Enable CORS for the public JSON API when origins are configured.

        Only applies when the public JSON API is served (dashboard + all
        modes). Validator-only processes never mount it since their
        traffic doesn't come from browsers.
        """
        if not _mode_runs_dashboard_api(self.mode):
            return
        origins_raw = Config.DASHBOARD_CORS_ORIGINS or ""
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
        if not origins:
            return

        # Idempotent: don't double-register if re-called
        from fastapi.middleware.cors import CORSMiddleware
        for m in getattr(app, "user_middleware", []):
            if m.cls is CORSMiddleware:
                return

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
            allow_credentials=False,
            max_age=600,
        )
        logger.info(
            "CORS enabled on /dashboard/api/* for origins: %s",
            ", ".join(origins),
        )

    async def _rebuild_pareto(self):
        """Rebuild Pareto fronts from all DB elements."""
        from shared.task import load_task
        self.pareto_fronts = {}
        all_elements = await self.store.get_pareto_elements()
        skipped = 0
        for elem in all_elements:
            if not elem.task:
                skipped += 1
                continue
            if elem.task not in self.pareto_fronts:
                try:
                    ts = load_task(elem.task) if elem.task not in self.tasks else self.tasks[elem.task]
                    objective_fn = lambda e, _ts=ts: _ts.objective_vector(e.objectives)
                    self.pareto_fronts[elem.task] = ParetoFront(
                        max_size=50, objective_fn=objective_fn,
                    )
                except Exception:
                    logger.warning("Skipping task %r — no YAML definition", elem.task)
                    continue
            try:
                self.pareto_fronts[elem.task].update(elem)
            except Exception:
                pass
        if skipped:
            logger.warning("Pareto rebuild: skipped %d experiments with empty task", skipped)
        # Set initial frontier
        all_candidates = []
        for pf in self.pareto_fronts.values():
            all_candidates.extend([c.element.to_dict() for c in pf.candidates])
        set_frontier(all_candidates)
        logger.info(
            "Pareto rebuilt: %d tasks, %d total frontier candidates",
            len(self.pareto_fronts), len(all_candidates),
        )

    def _sync_metagraph(self):
        """Sync metagraph and update hotkey map. No-op outside chain modes."""
        if not _mode_runs_chain(self.mode) or self.metagraph is None:
            return
        self.metagraph.sync()
        hotkeys = self.metagraph.hotkeys or []
        hotkey_map = {
            hotkeys[uid]: uid
            for uid in range(self.metagraph.n)
            if uid < len(hotkeys)
        }
        set_hotkey_map(hotkey_map)
        set_auth(self.metagraph)

    def _refresh_round_id(self) -> int:
        """Recompute round_id from chain height. Returns -1 outside chain modes."""
        if not _mode_runs_chain(self.mode) or self.subtensor is None:
            return -1
        from shared.challenge import round_id_from_block
        try:
            current_block = self.subtensor.block
        except Exception as e:
            logger.warning("Round refresh: subtensor read failed: %s", e)
            return -1
        round_id = round_id_from_block(
            current_block, Config.ROUND_INTERVAL_BLOCKS,
        )
        if self.access_logger is not None:
            self.access_logger.set_round(round_id)
        return round_id

    async def run(self):
        """Main loop: init DB, start server, periodic loops (chain modes only)."""
        await self._init_db()

        # Chain-driven state only runs in modes with a subtensor connection.
        # Dashboard mode skips Pareto rebuild + metagraph sync + round loop.
        if _mode_runs_chain(self.mode):
            await self._rebuild_pareto()
            self._sync_metagraph()
            self._refresh_round_id()

        db_size = await self.store.get_size()
        logger.info(
            "DB server starting on port %d (mode=%s network=%s). "
            "DB: %d experiments, tasks: %s",
            self.port, self.mode, Config.NETWORK,
            db_size, list(self.tasks.keys()),
        )

        # Start uvicorn in background
        server_config = uvicorn.Config(
            app, host="0.0.0.0", port=self.port, log_level="warning",
            limit_concurrency=200,  # DDoS: cap concurrent connections
            limit_max_requests=50000,  # Recycle worker after N requests
        )
        server = uvicorn.Server(server_config)
        server_task = asyncio.create_task(server.serve())

        round_task: Optional[asyncio.Task] = None

        if _mode_runs_chain(self.mode):
            # Round tracker — rounds are ~55 min (275 blocks × 12s), so a
            # 30s poll keeps miner_access_log.round_id within one block of
            # the actual round boundary.
            async def _round_loop():
                while True:
                    await asyncio.sleep(30)
                    try:
                        self._refresh_round_id()
                    except Exception as e:
                        logger.warning("Round refresh loop: %s", e)

            round_task = asyncio.create_task(_round_loop())

            # Periodic metagraph sync
            try:
                while True:
                    await asyncio.sleep(300)  # 5 minutes
                    try:
                        self._sync_metagraph()
                        logger.info("Metagraph synced: %d neurons", self.metagraph.n)
                    except Exception as e:
                        logger.warning("Metagraph sync failed: %s", e)
            except asyncio.CancelledError:
                if round_task is not None:
                    round_task.cancel()
                server.should_exit = True
                await server_task
        else:
            # Dashboard mode: no chain loops, just serve HTTP until killed.
            try:
                await server_task
            except asyncio.CancelledError:
                server.should_exit = True
                await server_task


def get_config():
    parser = argparse.ArgumentParser(description="Radar Database Server")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--port", type=int, default=Config.DB_API_PORT)
    parser.add_argument("--pg_dsn", type=str, default="")
    parser.add_argument("--task", type=str, default="")

    # Only stitch the wallet / subtensor argparse extensions in when the
    # process actually needs them. Dashboard mode must parse without
    # importing bittensor at all.
    if _mode_runs_chain(Config.NEURON_MODE):
        import bittensor as bt
        bt.Wallet.add_args(parser)
        bt.Subtensor.add_args(parser)
        return bt.Config(parser)

    # Dashboard mode: return a plain namespace that behaves like bt.Config
    # for the attribute access patterns we use.
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    validate_neuron_mode()
    config = get_config()
    neuron = DatabaseNeuron(config)
    asyncio.run(neuron.run())


if __name__ == "__main__":
    main()
