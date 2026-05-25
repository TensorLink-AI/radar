"""Local miner process.

Polls the SQLite store for open challenges, runs the agent's
``design_architecture`` to produce a proposal, posts it back. No
hosting, no commitment — the agent runs in-process. The validator's
trainer then trains and evaluates whatever code the agent emitted.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local.store import LocalStore

logger = logging.getLogger("local.miner")


def _load_agent(module_path: str | None):
    """Return a callable ``design_architecture(challenge: dict) -> dict``.

    Default: ``local.agent.design_architecture``.
    """
    if not module_path:
        from local.agent import design_architecture
        return design_architecture
    p = Path(module_path)
    if not p.exists():
        raise FileNotFoundError(f"agent module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("user_agent", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "design_architecture", None)
    if not callable(fn):
        raise AttributeError(
            f"{module_path} must define design_architecture(challenge)"
        )
    return fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local radar miner")
    parser.add_argument("--db", default="local/radar_local.db",
                        help="SQLite path (must match the validator's)")
    parser.add_argument("--miner_id", default=f"miner-{uuid.uuid4().hex[:6]}",
                        help="Stable miner identifier")
    parser.add_argument("--agent_module", default="",
                        help="Optional: path to a .py file overriding the "
                             "default agent")
    parser.add_argument("--poll_seconds", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=0,
                        help="Number of rounds to participate in; 0 = forever")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [miner:" + args.miner_id + "] %(message)s",
        datefmt="%H:%M:%S",
    )

    design = _load_agent(args.agent_module or None)
    store = LocalStore(args.db)
    logger.info("starting; db=%s agent=%s",
                args.db, args.agent_module or "local.agent")

    last_round = -1
    submitted = 0
    try:
        while args.rounds == 0 or submitted < args.rounds:
            open_ch = store.open_challenge()
            if open_ch is None or open_ch["round_id"] == last_round:
                time.sleep(args.poll_seconds)
                continue

            challenge = open_ch["payload"]
            round_id = open_ch["round_id"]
            challenge_id = open_ch["challenge_id"]
            logger.info(
                "got challenge round=%d bucket=[%d, %d] frontier=%d",
                round_id, challenge.get("min_flops_equivalent", 0),
                challenge.get("max_flops_equivalent", 0),
                len(challenge.get("feasible_frontier", [])),
            )

            try:
                proposal = design(challenge)
            except Exception as e:  # noqa: BLE001
                logger.exception("agent crashed: %s", e)
                last_round = round_id
                continue

            store.post_proposal(challenge_id, round_id, args.miner_id, proposal)
            logger.info("submitted proposal '%s'", proposal.get("name", "?"))
            last_round = round_id
            submitted += 1
    except KeyboardInterrupt:
        logger.info("interrupted; bye")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
