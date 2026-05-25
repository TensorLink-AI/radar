"""Launch a local validator and one or more miners on the same SQLite DB.

Two OS processes (validator + miner) is the default — that's the
'role separation' the user asked for. Multiple miners can be added
via ``--miners N``.

Example::

    python local/run.py                       # 1 validator + 1 miner, forever
    python local/run.py --rounds 5            # stop both after 5 rounds
    python local/run.py --miners 3 --rounds 3 # 1 validator + 3 miners

Both processes inherit stdout/stderr and are line-buffered, so logs
interleave readably in the parent terminal. SIGINT (Ctrl-C) cleanly
terminates both.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _spawn(args: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    # Force unbuffered output so logs from both children interleave live.
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.Popen(
        args, cwd=str(ROOT), env=env,
        stdout=sys.stdout, stderr=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local radar stack")
    parser.add_argument("--db", default="local/radar_local.db")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Number of rounds; 0 = forever")
    parser.add_argument("--miners", type=int, default=1,
                        help="How many miner processes to launch")
    parser.add_argument("--phase_a_seconds", type=float, default=10.0)
    parser.add_argument("--gap_seconds", type=float, default=2.0)
    parser.add_argument("--agent_dir", default="",
                        help="Directory containing agent.py; passed to "
                             "every miner process.")
    parser.add_argument("--agent_module", default="",
                        help="Single-file alternative to --agent_dir.")
    parser.add_argument("--wiki_dir", default="",
                        help="Local markdown directory exposed to the "
                             "agent at GET /wiki.")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    py = sys.executable

    validator_cmd = [
        py, "-m", "local.validator",
        "--db", args.db,
        "--rounds", str(args.rounds),
        "--phase_a_seconds", str(args.phase_a_seconds),
        "--gap_seconds", str(args.gap_seconds),
        "--log_level", args.log_level,
    ]
    if args.wiki_dir:
        validator_cmd += ["--wiki_dir", args.wiki_dir]

    procs: list[subprocess.Popen] = []
    print(f"[run] launching validator: {' '.join(validator_cmd)}", flush=True)
    procs.append(_spawn(validator_cmd))

    # Give the validator a head start so the first challenge is published
    # before miners start polling — keeps the early logs tidy.
    time.sleep(1.0)

    for i in range(args.miners):
        miner_id = f"miner-{i:02d}"
        miner_cmd = [
            py, "-m", "local.miner",
            "--db", args.db,
            "--miner_id", miner_id,
            "--rounds", str(args.rounds),
            "--log_level", args.log_level,
        ]
        if args.agent_dir:
            miner_cmd += ["--agent_dir", args.agent_dir]
        elif args.agent_module:
            miner_cmd += ["--agent_module", args.agent_module]
        print(f"[run] launching miner {miner_id}", flush=True)
        procs.append(_spawn(miner_cmd))

    def _shutdown(*_):
        print("[run] shutting down…", flush=True)
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait for the validator. When it exits (either because --rounds was
    # set or the user hit Ctrl-C), reap the miners.
    exit_code = procs[0].wait()
    for p in procs[1:]:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in procs[1:]:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    print(f"[run] done; validator exit={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
