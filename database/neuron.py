"""Subnet owner process — centralized Postgres experiment database.

Runs the FastAPI server that validators write to and miners read from
(via validator proxy). Single source of truth for experiments, provenance,
and Pareto frontier.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import bittensor as bt
import uvicorn

from config import Config
from database.server import (
    app, set_db, set_auth, set_challenge, set_frontier,
    set_access_logger, set_hotkey_map, set_rate_limit,
    set_r2, set_pool,
    get_current_challenge, get_current_frontier,
)
from shared.pareto import ParetoFront
from shared.pg_access_logger import PgAccessLogger
from shared.pg_store import (
    PgExperimentStore,
    create_pg_pool,
    ensure_schema_exists,
)
from shared.task import load_enabled_tasks

logger = logging.getLogger(__name__)


class DatabaseNeuron:
    """Subnet owner database server."""

    def __init__(self, config: bt.Config):
        self.config = config
        self.netuid = config.netuid
        self.port = int(getattr(config, "port", Config.DB_API_PORT))

        # Bittensor components
        self.wallet = bt.Wallet(config=config)
        self.subtensor = bt.Subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(self.netuid)

        # Tasks
        self.tasks = load_enabled_tasks(Config.ENABLED_TASKS)

        # Per-task Pareto fronts
        self.pareto_fronts: dict[str, ParetoFront] = {}

        # R2 audit log
        self.r2 = None
        if Config.R2_BUCKET:
            try:
                from shared.r2_audit import R2AuditLog
                self.r2 = R2AuditLog(bucket=Config.R2_BUCKET)
            except Exception as e:
                logger.warning("R2 audit log unavailable: %s", e)

        # Pool, store, logger set in async init
        self.pool = None
        self.store = None
        self.access_logger = None

    async def _init_db(self):
        """Create asyncpg pool and init schemas."""
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

        # Bootstrap step: CREATE SCHEMA must run on a connection whose
        # search_path is NOT already set to that schema (otherwise we're
        # asking the schema to create itself). Use a bare asyncpg.connect
        # so no pool init/setup hook runs.
        logger.info(
            "Bootstrapping Postgres schema %r (RADAR_NETWORK)", network,
        )
        bootstrap_kwargs = {
            k: pool_kwargs[k]
            for k in ("ssl", "statement_cache_size")
            if k in pool_kwargs
        }
        bootstrap_conn = await asyncpg.connect(pg_dsn, **bootstrap_kwargs)
        try:
            await ensure_schema_exists(bootstrap_conn, network)
        finally:
            await bootstrap_conn.close()

        # Every connection borrowed from this pool will have its search_path
        # pinned to `"<network>", public`, so all unqualified DDL/DML that
        # follows resolves to tables inside this schema.
        self.pool = await create_pg_pool(
            pg_dsn, schema=network, **pool_kwargs,
        )
        self.store = PgExperimentStore(self.pool)
        await self.store.init_schema()

        self.access_logger = PgAccessLogger(self.pool)
        await self.access_logger.init_schema()

        # Init agent_submissions table
        from shared.pg_schema import AGENT_CODE_SCHEMA
        async with self.pool.acquire() as conn:
            await conn.execute(AGENT_CODE_SCHEMA)

        # Init proxy query log table
        from shared.pg_schema import PROXY_QUERY_LOG_SCHEMA
        async with self.pool.acquire() as conn:
            await conn.execute(PROXY_QUERY_LOG_SCHEMA)

        # Wire into server
        set_db(self.store)
        set_pool(self.pool)
        if self.r2:
            set_r2(self.r2)
        set_access_logger(self.access_logger)
        set_auth(self.metagraph)
        set_rate_limit(Config.DB_VALI_RATE_LIMIT)

        # Desearch proxy (SN22 arxiv search)
        if Config.DESEARCH_ENABLED:
            from validator.desearch_proxy import DesearchProxy, set_proxy, register_routes
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

        # Read-only web dashboard (subnet-owner operator UI)
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
                )
            except Exception as e:
                logger.warning("Dashboard mount failed: %s", e)

        logger.info(
            "Database initialized: schema=%r pool_min=%d pool_max=%d ssl=%r",
            network, Config.PG_POOL_MIN, Config.PG_POOL_MAX, Config.PG_SSL or "off",
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
        """Sync metagraph and update hotkey map."""
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
        """Recompute the current round_id from chain height and push it to
        the access logger so miner_access_log rows get the right tag.

        Returns the round_id now in effect, or -1 if the subtensor read
        failed.
        """
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
        """Main loop: init DB, start server, periodic metagraph sync."""
        await self._init_db()
        await self._rebuild_pareto()
        self._sync_metagraph()
        self._refresh_round_id()

        db_size = await self.store.get_size()
        logger.info(
            "Database server starting on port %d (network=%s). "
            "DB: %d experiments, tasks: %s",
            self.port, Config.NETWORK, db_size, list(self.tasks.keys()),
        )

        # Start uvicorn in background
        server_config = uvicorn.Config(
            app, host="0.0.0.0", port=self.port, log_level="warning",
            limit_concurrency=200,  # DDoS: cap concurrent connections
            limit_max_requests=50000,  # Recycle worker after N requests
        )
        server = uvicorn.Server(server_config)
        server_task = asyncio.create_task(server.serve())

        # Round tracker — rounds are ~55 min (275 blocks × 12s), so a 30s
        # poll keeps miner_access_log.round_id within one block of the
        # actual round boundary.
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
            round_task.cancel()
            server.should_exit = True
            await server_task


def get_config() -> bt.Config:
    parser = argparse.ArgumentParser(description="Radar Database Server")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--port", type=int, default=Config.DB_API_PORT)
    parser.add_argument("--pg_dsn", type=str, default="")
    parser.add_argument("--task", type=str, default="")
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    return bt.Config(parser)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    config = get_config()
    neuron = DatabaseNeuron(config)
    asyncio.run(neuron.run())


if __name__ == "__main__":
    main()
