"""Local miner process.

Polls the SQLite store for open challenges, runs the agent's
``design_architecture`` to produce a proposal, posts it back.

Two agent-loading modes:

* ``--agent_module path/to/agent.py`` — single file (legacy)
* ``--agent_dir path/to/dir/`` — same as ``miner/neuron.py --agent_dir``
  in real radar; the dir must contain ``agent.py`` and any inter-file
  imports are resolved relative to that dir.

The agent's ``design_architecture`` is called with either
``(challenge, client)`` or just ``(challenge)``, depending on its
signature. ``client`` is a real ``shared.url_gate.GatedClient`` pointed
at the validator's local services URL (db / llm / desearch / wiki).
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local.store import LocalStore
from shared.url_gate import GatedClient, parse_allowed_urls

logger = logging.getLogger("local.miner")


def _load_from_dir(agent_dir: Path) -> Callable:
    """Load ``agent.py`` from a directory, making sibling files
    importable from inside the agent module."""
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"agent dir not found: {agent_dir}")
    entry = agent_dir / "agent.py"
    if not entry.is_file():
        raise FileNotFoundError(f"{agent_dir} must contain agent.py")
    # So ``from helpers import foo`` works inside the agent.
    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location("user_agent", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "design_architecture", None)
    if not callable(fn):
        raise AttributeError(
            f"{entry} must define design_architecture(challenge, client?)"
        )
    return fn


def _load_from_file(path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location("user_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "design_architecture", None)
    if not callable(fn):
        raise AttributeError(
            f"{path} must define design_architecture(challenge, client?)"
        )
    return fn


def _load_agent(agent_dir: str, agent_module: str) -> Callable:
    if agent_dir:
        return _load_from_dir(Path(agent_dir).resolve())
    if agent_module:
        p = Path(agent_module).resolve()
        if not p.exists():
            raise FileNotFoundError(f"agent module not found: {agent_module}")
        return _load_from_file(p)
    from local.agent import design_architecture
    return design_architecture


def _call_agent(design: Callable, challenge: dict,
                client: Optional[GatedClient]) -> dict:
    """Dispatch on the agent's signature: 2-arg (challenge, client) vs
    1-arg (challenge). Matches what ``runner/agent/harness.py`` does."""
    try:
        sig = inspect.signature(design)
        n_positional = sum(
            1 for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    except (TypeError, ValueError):
        n_positional = 1
    if n_positional >= 2 and client is not None:
        return design(challenge, client)
    return design(challenge)


def _build_client(challenge: dict, miner_id: str) -> Optional[GatedClient]:
    """GatedClient pointed at the validator's services URL.

    Returns ``None`` if the challenge has no allowed URLs (older
    challenges from a pre-services validator)."""
    allowed_raw = challenge.get("allowed_urls", "")
    prefixes = parse_allowed_urls(allowed_raw)
    if not prefixes:
        return None
    headers = {"X-Miner-Id": miner_id}
    return GatedClient(prefixes, default_headers=headers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local radar miner")
    parser.add_argument("--db", default="local/radar_local.db",
                        help="SQLite path (must match the validator's)")
    parser.add_argument("--miner_id", default=f"miner-{uuid.uuid4().hex[:6]}",
                        help="Stable miner identifier")
    parser.add_argument("--agent_dir", default="",
                        help="Directory containing agent.py (mirrors "
                             "miner/neuron.py --agent_dir)")
    parser.add_argument("--agent_module", default="",
                        help="Single-file alternative to --agent_dir")
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

    design = _load_agent(args.agent_dir, args.agent_module)
    agent_label = args.agent_dir or args.agent_module or "local.agent"
    store = LocalStore(args.db)
    logger.info("starting; db=%s agent=%s", args.db, agent_label)

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

            client = _build_client(challenge, args.miner_id)
            try:
                proposal = _call_agent(design, challenge, client)
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
