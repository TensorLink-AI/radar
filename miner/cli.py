"""Miner CLI subcommands — ``results``, ``optimize``, ``prompts``.

Routed from ``miner/neuron.py``'s ``main()``.  When the first argv is
one of the known subcommands the neuron delegates here; otherwise the
neuron falls through to the legacy ``run`` behavior so existing
deployments keep working without flag churn.

  miner/neuron.py run [existing flags]
    legacy entrypoint — serve agent + trainer-launcher listener

  miner/neuron.py results [--since 1d] [--limit 50] [--json] [--task X]
    one-shot: print recent scored results from the DB

  miner/neuron.py optimize \\
      [--optimizer gepa|random_mutate|pkg.mod:func] \\
      [--rounds-back 20] [--population 8] [--budget 30] \\
      [--score-key raw_score] [--dry-run] [--task ts_forecasting]
    one-shot: pull results, evolve the prompt population, write new
    active.json (archives the old as history/gen_NNN.json)

  miner/neuron.py optimize --watch [--every-seconds 600]
    daemon loop: re-run optimize whenever ``my_summary().last_round_id``
    advances

  miner/neuron.py prompts list
    show the current population

  miner/neuron.py prompts history
    list archived generation numbers

  miner/neuron.py prompts rollback <gen>
    restore an archived generation (with safety-archive of current)

DB URL + API key are read from env (``RADAR_DB_URL``,
``RADAR_MINER_API_KEY``) or flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from miner_template import optimizers, prompts as prompts_mod
from miner_template.optimizers import ResultRow
from miner_template.results_client import MinerResultsClient

logger = logging.getLogger(__name__)

# argv[1] values that route here instead of the legacy `run` entrypoint.
SUBCOMMANDS = frozenset({"results", "optimize", "prompts"})


def is_subcommand(argv: list[str]) -> bool:
    """Return True if argv[1] looks like one of our subcommands."""
    return len(argv) >= 2 and argv[1] in SUBCOMMANDS


def dispatch(argv: list[str]) -> int:
    """Top-level entrypoint.  argv[0] is the program, argv[1] is the
    subcommand.  Returns a process exit code."""
    if len(argv) < 2 or argv[1] not in SUBCOMMANDS:
        print(
            f"Unknown subcommand. Expected one of: {sorted(SUBCOMMANDS)}",
            file=sys.stderr,
        )
        return 2

    parser = _root_parser()
    args = parser.parse_args(argv[1:])

    if args.subcommand == "results":
        return cmd_results(args)
    if args.subcommand == "optimize":
        return cmd_optimize(args)
    if args.subcommand == "prompts":
        return cmd_prompts(args)
    parser.error(f"unhandled subcommand {args.subcommand}")
    return 2


# ── argparse plumbing ────────────────────────────────────────────────


def _root_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="miner",
        description="Radar miner — agent host, results polling, prompt optimizer.",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    # results
    r = sub.add_parser("results", help="Print recent scored results.")
    _add_db_flags(r)
    _add_window_flags(r)
    r.add_argument("--json", action="store_true",
                   help="Emit one JSON line per result instead of a table.")

    # optimize
    o = sub.add_parser("optimize", help="Run a prompt-optimization pass.")
    _add_db_flags(o)
    _add_window_flags(o)
    o.add_argument("--optimizer", default="gepa",
                   help="Optimizer to use: 'gepa', 'random_mutate', or "
                        "'package.module:func'.  Default: gepa.")
    o.add_argument("--population", type=int, default=8,
                   help="Target size of the new prompt population.")
    o.add_argument("--budget", type=int, default=30,
                   help="Optimizer rollout/compile budget (GEPA-only).")
    o.add_argument("--score-key", default="raw_score",
                   help="scores[] field to optimize against.")
    o.add_argument("--elite-k", type=int, default=2,
                   help="Top-K survivors carried forward (random_mutate).")
    o.add_argument("--prompts-dir", default=None,
                   help=f"Where prompts live "
                        f"(default ${prompts_mod.DEFAULT_PROMPTS_DIR}).")
    o.add_argument("--seed", action="store_true",
                   help="Seed a default prompt if active.json is missing.")
    o.add_argument("--dry-run", action="store_true",
                   help="Print the diff but don't write files.")
    o.add_argument("--watch", action="store_true",
                   help="Loop forever, re-running when new rounds finish.")
    o.add_argument("--every-seconds", type=int, default=600,
                   help="--watch poll interval.")

    # prompts
    pr = sub.add_parser("prompts", help="Inspect/manage the prompt population.")
    psub = pr.add_subparsers(dest="prompts_command", required=True)
    pl = psub.add_parser("list", help="Show the current active population.")
    pl.add_argument("--prompts-dir", default=None)
    pl.add_argument("--json", action="store_true")
    psub.add_parser("history", help="List archived generations.").add_argument(
        "--prompts-dir", default=None,
    )
    rb = psub.add_parser("rollback", help="Restore an archived generation.")
    rb.add_argument("generation", type=int)
    rb.add_argument("--prompts-dir", default=None)

    return p


def _add_db_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db-url",
                   default=os.getenv("RADAR_DB_URL", ""),
                   help="DB server URL (env: RADAR_DB_URL).")
    p.add_argument("--api-key",
                   default=os.getenv("RADAR_MINER_API_KEY", ""),
                   help="Miner bearer API key (env: RADAR_MINER_API_KEY).")


def _add_window_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rounds-back", type=int, default=20,
                   help="How many rounds of history to consider.")
    p.add_argument("--since", default="",
                   help="ISO timestamp override; takes precedence over --rounds-back.")
    p.add_argument("--limit", type=int, default=500,
                   help="Max rows pulled from the DB.")
    p.add_argument("--task", default="",
                   help="Filter to one task name (e.g. ts_forecasting).")


# ── results ─────────────────────────────────────────────────────────


def cmd_results(args) -> int:
    if not args.db_url or not args.api_key:
        print(
            "results: missing --db-url or --api-key "
            "(or RADAR_DB_URL / RADAR_MINER_API_KEY).",
            file=sys.stderr,
        )
        return 2
    since = args.since or _approx_since(args.rounds_back)
    with MinerResultsClient(args.db_url, args.api_key) as c:
        rows = c.results(since=since, limit=args.limit, task=args.task)
    if args.json:
        for r in rows:
            print(json.dumps({
                "round_id": r.round_id,
                "submission_id": r.submission_id,
                "task_name": r.task_name,
                "prompt_id": r.prompt_id,
                "scores": r.scores,
                "created_at": r.created_at,
            }))
    else:
        _print_results_table(rows)
    return 0


def _print_results_table(rows: list[ResultRow]) -> None:
    if not rows:
        print("(no results)")
        return
    header = f"{'round':>6}  {'prompt_id':<14}  {'score':>8}  {'task':<16}  submission"
    print(header)
    print("-" * len(header))
    for r in rows:
        pid = (r.prompt_id or "-")[:12]
        score = float(r.scores.get("raw_score", 0.0) or 0.0)
        task = (r.task_name or "-")[:14]
        print(f"{r.round_id:>6}  {pid:<14}  {score:>8.4f}  {task:<16}  {r.submission_id}")


# ── optimize ────────────────────────────────────────────────────────


def cmd_optimize(args) -> int:
    if not args.db_url or not args.api_key:
        print(
            "optimize: missing --db-url or --api-key "
            "(or RADAR_DB_URL / RADAR_MINER_API_KEY).",
            file=sys.stderr,
        )
        return 2
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else None

    if args.watch:
        return _watch_loop(args, prompts_dir)
    return _optimize_once(args, prompts_dir)


def _optimize_once(args, prompts_dir: Optional[Path]) -> int:
    population = prompts_mod.load_active(prompts_dir)
    if not population:
        if args.seed:
            population = prompts_mod.seed_default(prompts_dir)
            print(f"Seeded default prompt ({population[0].id[:8]}…)")
        else:
            print(
                "optimize: no active prompt population yet. "
                "Re-run with --seed to write a default starter prompt.",
                file=sys.stderr,
            )
            return 2

    since = args.since or _approx_since(args.rounds_back)
    with MinerResultsClient(args.db_url, args.api_key) as c:
        results = c.results(since=since, limit=args.limit, task=args.task)
    print(
        f"Loaded {len(results)} scored results since {since or 'beginning'} "
        f"(population size {len(population)}).",
    )

    try:
        fn = optimizers.resolve(args.optimizer)
    except (ImportError, ValueError, AttributeError, TypeError) as e:
        print(f"optimize: cannot resolve optimizer: {e}", file=sys.stderr)
        return 2

    config = {
        "population": args.population,
        "budget": args.budget,
        "score_key": args.score_key,
        "elite_k": args.elite_k,
    }
    try:
        new_pop = fn(results, population, config)
    except Exception as e:
        print(f"optimize: optimizer raised: {e}", file=sys.stderr)
        return 1

    if not new_pop:
        print("optimize: optimizer returned an empty population — no change.",
              file=sys.stderr)
        return 1

    if args.dry_run:
        _print_pop_diff(population, new_pop)
        return 0

    next_gen = (prompts_mod.list_history(prompts_dir) or [0])[-1] + 1
    prompts_mod.archive_current(next_gen, prompts_dir)
    prompts_mod.save_active(new_pop, prompts_dir)
    print(
        f"Wrote generation {next_gen + 1} "
        f"({len(new_pop)} prompts); previous archived as gen_{next_gen:03d}.json",
    )
    return 0


def _watch_loop(args, prompts_dir: Optional[Path]) -> int:
    last_round = -1
    print(f"watch: polling every {args.every_seconds}s …")
    while True:
        try:
            with MinerResultsClient(args.db_url, args.api_key) as c:
                summary = c.summary()
            cur = int(summary.get("last_round_id", -1) or -1)
        except Exception as e:
            logger.warning("watch: summary fetch failed: %s — retrying", e)
            time.sleep(args.every_seconds)
            continue
        if cur > last_round:
            print(f"watch: new round {cur} — running optimizer …")
            code = _optimize_once(args, prompts_dir)
            if code != 0:
                logger.warning("watch: optimizer exited %d", code)
            last_round = cur
        time.sleep(args.every_seconds)


def _print_pop_diff(old: list, new: list) -> None:
    old_ids = {p.id for p in old}
    new_ids = {p.id for p in new}
    survivors = old_ids & new_ids
    fresh = new_ids - old_ids
    dropped = old_ids - new_ids
    print(f"would write population: size={len(new)}")
    print(f"  survivors: {len(survivors)}  new: {len(fresh)}  dropped: {len(dropped)}")
    for p in new:
        marker = "·" if p.id in old_ids else "+"
        snippet = p.template.replace("\n", " ⏎ ")[:80]
        print(f"  {marker} [gen {p.generation}] {p.id[:8]}…  {snippet}")


# ── prompts subcommands ────────────────────────────────────────────


def cmd_prompts(args) -> int:
    if args.prompts_command == "list":
        return _prompts_list(args)
    if args.prompts_command == "history":
        return _prompts_history(args)
    if args.prompts_command == "rollback":
        return _prompts_rollback(args)
    return 2


def _prompts_list(args) -> int:
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else None
    pop = prompts_mod.load_active(prompts_dir)
    if args.json:
        print(json.dumps([p.to_dict() for p in pop], indent=2))
        return 0
    if not pop:
        print("(no active prompt population)")
        return 0
    for p in pop:
        snippet = p.template.replace("\n", " ⏎ ")[:80]
        parent = (p.parent_id or "-")[:8]
        print(f"  [gen {p.generation}] {p.id[:8]}… parent={parent}  {snippet}")
    return 0


def _prompts_history(args) -> int:
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else None
    gens = prompts_mod.list_history(prompts_dir)
    if not gens:
        print("(no archived generations)")
        return 0
    for g in gens:
        print(f"  gen_{g:03d}")
    return 0


def _prompts_rollback(args) -> int:
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else None
    try:
        prompts_mod.rollback_to(args.generation, prompts_dir)
    except FileNotFoundError as e:
        print(f"rollback: {e}", file=sys.stderr)
        return 2
    print(f"Rolled back to generation {args.generation}.")
    return 0


# ── helpers ────────────────────────────────────────────────────────


def _approx_since(rounds_back: int) -> str:
    """Map ``--rounds-back`` to an ISO timestamp.  Heuristic: each round
    is roughly 55 min (see CLAUDE.md).  Good enough for an upper-bound
    filter — the DB will still cap with --limit."""
    if rounds_back <= 0:
        return ""
    delta = timedelta(minutes=55 * rounds_back)
    return (datetime.now(timezone.utc) - delta).isoformat(timespec="seconds")
