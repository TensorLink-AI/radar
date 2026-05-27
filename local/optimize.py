"""Prompt-population optimizer loop for the local stack.

Wires the existing ``miner_template.optimizers`` plugins (gepa,
random_mutate, or any ``pkg.mod:func``) into the local SQLite store.

Usage::

    # one-shot
    python -m local.optimize --agent_dir /path/to/agent

    # daemon — re-run whenever a new experiment lands
    python -m local.optimize --agent_dir /path/to/agent --watch

    # explicit optimizer + config
    CHUTES_API_KEY=cpk_... python -m local.optimize \\
        --agent_dir /path/to/agent --optimizer gepa --population 6

The ``prompts/`` directory lives under ``<agent_dir>`` by default so
the agent (via ``miner_template.prompts.load_active``) finds it
without extra config.

For ``--optimizer gepa`` the adapter needs a reflector LM. We point it
at Chutes whenever ``CHUTES_API_KEY`` is set; ``OPENAI_API_KEY`` is
the fallback. Without either key, GEPA refuses to run — use
``--optimizer random_mutate`` for a no-key dev loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local.providers import CHUTES_BASE_URL, CHUTES_DEFAULT_MODEL, OPENAI_BASE_URL
from local.store import LocalStore
from miner_template import optimizers as opt_mod
from miner_template import prompts as prompts_mod
from miner_template.optimizers import ResultRow
from miner_template.prompts import Prompt

logger = logging.getLogger("local.optimize")


def _experiments_to_rows(store: LocalStore, task: str,
                         limit: int = 500) -> list[ResultRow]:
    """SQLite rows → ``ResultRow``.

    We map the validator-computed ``score`` to ``scores['raw_score']``
    so the GEPA / random_mutate defaults (which key on ``raw_score``)
    work out of the box. The primary objective metric is exposed as
    ``scores['metric']`` for optimizers that want to rank on the
    underlying number rather than the Pareto-scaled score.
    """
    rows = store.recent_experiments(n=limit)
    out: list[ResultRow] = []
    for r in rows:
        if not r.get("success"):
            continue
        scores = {
            "raw_score": float(r.get("score") or 0.0),
        }
        if r.get("metric") is not None:
            scores["metric"] = float(r["metric"])
        for k, v in (r.get("objectives") or {}).items():
            try:
                scores[k] = float(v)
            except (TypeError, ValueError):
                continue
        out.append(ResultRow(
            round_id=int(r.get("round_id", 0) or 0),
            submission_id=str(r.get("id", "") or ""),
            task_name=task or str(r.get("task", "") or ""),
            prompt_id=r.get("prompt_id") or None,
            architecture_code=str(r.get("code", "") or ""),
            scores=scores,
            created_at=str(r.get("timestamp", "") or ""),
        ))
    return out


def _configure_gepa_env() -> None:
    """Point the GEPA reflector at Chutes (or OpenAI). Read by
    ``miner_template.optimizers.gepa._default_lm`` via ``MINER_LLM_*``."""
    if os.environ.get("MINER_LLM_URL"):
        return  # operator already overrode it
    if os.environ.get("CHUTES_API_KEY"):
        os.environ["MINER_LLM_URL"] = CHUTES_BASE_URL
        os.environ["MINER_LLM_API_KEY"] = os.environ["CHUTES_API_KEY"]
        os.environ.setdefault("MINER_LLM_MODEL", CHUTES_DEFAULT_MODEL)
        return
    if os.environ.get("OPENAI_API_KEY"):
        os.environ["MINER_LLM_URL"] = OPENAI_BASE_URL
        os.environ["MINER_LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
        os.environ.setdefault("MINER_LLM_MODEL", "gpt-4o-mini")


def _ensure_seed(prompts_dir: Path) -> list[Prompt]:
    """Make sure ``active.json`` exists; create a starter pop if not."""
    pop = prompts_mod.load_active(prompts_dir)
    if pop:
        return pop
    seeded = prompts_mod.seed_default(prompts_dir)
    logger.info("seeded prompts/active.json at %s with %d variant(s)",
                prompts_dir, len(seeded))
    return seeded


def _run_once(store: LocalStore, prompts_dir: Path, *, optimizer: str,
              population: int, budget: int, task: str,
              score_key: str) -> int:
    """One optimization pass. Returns the new population size."""
    pop = _ensure_seed(prompts_dir)
    results = _experiments_to_rows(store, task=task)
    logger.info(
        "running %s: %d scored row(s), population=%d, target=%d",
        optimizer, len(results), len(pop), population,
    )

    if optimizer == "gepa":
        _configure_gepa_env()

    optimize_fn = opt_mod.resolve(optimizer)
    new_pop = optimize_fn(
        results,
        pop,
        {
            "population": population,
            "budget": budget,
            "score_key": score_key,
            "elite_k": max(1, population // 3),
        },
    )
    if not new_pop:
        logger.warning("optimizer returned empty population; keeping current")
        return len(pop)

    next_gen = max(p.generation for p in new_pop)
    prompts_mod.archive_current(next_gen, prompts_dir)
    prompts_mod.save_active(new_pop, prompts_dir)
    logger.info(
        "wrote generation %d (%d prompt(s)) → %s",
        next_gen, len(new_pop), prompts_dir / "active.json",
    )
    return len(new_pop)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local prompt optimizer")
    parser.add_argument("--db", default="local/radar_local.db")
    parser.add_argument("--agent_dir", default="",
                        help="Agent directory. prompts/ lives under it.")
    parser.add_argument("--prompts_dir", default="",
                        help="Override prompts dir (default: "
                             "<agent_dir>/prompts or ./prompts).")
    parser.add_argument("--optimizer", default="gepa",
                        help="gepa | random_mutate | pkg.mod:func")
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--budget", type=int, default=20,
                        help="GEPA rollout budget (ignored by other optimizers).")
    parser.add_argument("--task", default="synth_regression")
    parser.add_argument("--score_key", default="raw_score")
    parser.add_argument("--watch", action="store_true",
                        help="Re-run whenever a new experiment row appears.")
    parser.add_argument("--poll_seconds", type=float, default=5.0)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [optimize] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.prompts_dir:
        prompts_dir = Path(args.prompts_dir).resolve()
    elif args.agent_dir:
        prompts_dir = Path(args.agent_dir).resolve() / "prompts"
    else:
        prompts_dir = Path("prompts").resolve()
    prompts_dir.mkdir(parents=True, exist_ok=True)
    logger.info("prompts_dir=%s db=%s optimizer=%s",
                prompts_dir, args.db, args.optimizer)

    store = LocalStore(args.db)
    try:
        _run_once(store, prompts_dir, optimizer=args.optimizer,
                  population=args.population, budget=args.budget,
                  task=args.task, score_key=args.score_key)
        if not args.watch:
            return 0

        last_total = store.stats()["total"]
        logger.info("watching (poll=%.1fs, last_total=%d)…",
                    args.poll_seconds, last_total)
        while True:
            time.sleep(args.poll_seconds)
            total = store.stats()["total"]
            if total <= last_total:
                continue
            logger.info("detected %d new experiment row(s)",
                        total - last_total)
            last_total = total
            _run_once(store, prompts_dir, optimizer=args.optimizer,
                      population=args.population, budget=args.budget,
                      task=args.task, score_key=args.score_key)
    except KeyboardInterrupt:
        logger.info("interrupted; bye")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
