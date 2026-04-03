"""Subnet owner process — centralized Postgres experiment database.

Runs the FastAPI server that validators write to and miners read from
(via validator proxy). Single source of truth for experiments, provenance,
and Pareto frontier.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import asyncpg
import bittensor as bt
import uvicorn

from config import Config
from database.server import (
    app, set_db, set_pool, set_auth, set_challenge, set_frontier,
    set_access_logger, set_hotkey_map, set_rate_limit,
)
from shared.pareto import ParetoFront
from shared.pg_access_logger import PgAccessLogger
from shared.pg_store import PgExperimentStore
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

        pg_dsn = getattr(self.config, "pg_dsn", None) or Config.PG_DSN

        pool_kwargs: dict = {"min_size": 2, "max_size": 10}

        # SSL support (required for Supabase / managed Postgres)
        if Config.PG_SSL:
            ctx = _ssl.create_default_context()
            if Config.PG_SSL == "require":
                ctx.check_hostname = False
                ctx.verify_mode = _ssl.CERT_NONE
            pool_kwargs["ssl"] = ctx

        # Disable prepared-statement cache when using connection poolers
        # (e.g. Supabase Supavisor in transaction mode).
        if Config.PG_SSL or "supabase" in pg_dsn:
            pool_kwargs["statement_cache_size"] = 0

        self.pool = await asyncpg.create_pool(pg_dsn, **pool_kwargs)
        self.store = PgExperimentStore(self.pool)
        await self.store.init_schema()

        self.access_logger = PgAccessLogger(self.pool)
        await self.access_logger.init_schema()

        # Init commitment table
        from shared.pg_schema import COMMITMENT_SCHEMA
        async with self.pool.acquire() as conn:
            await conn.execute(COMMITMENT_SCHEMA)

        # Wire into server
        set_db(self.store)
        set_pool(self.pool)
        set_access_logger(self.access_logger)
        set_auth(self.metagraph)
        set_rate_limit(Config.DB_VALI_RATE_LIMIT)

        logger.info("Database initialized, schemas created")

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

    async def run(self):
        """Main loop: init DB, start server, periodic metagraph sync."""
        await self._init_db()
        await self._rebuild_pareto()
        self._sync_metagraph()

        db_size = await self.store.get_size()
        logger.info(
            "Database server starting on port %d. DB: %d experiments, tasks: %s",
            self.port, db_size, list(self.tasks.keys()),
        )

        # Start uvicorn in background
        server_config = uvicorn.Config(
            app, host="0.0.0.0", port=self.port, log_level="warning",
        )
        server = uvicorn.Server(server_config)
        server_task = asyncio.create_task(server.serve())

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
    config = get_config()
    neuron = DatabaseNeuron(config)
    asyncio.run(neuron.run())


if __name__ == "__main__":
    main()
