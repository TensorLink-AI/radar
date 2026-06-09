"""Crash/hang isolation for Phase B+C training.

The trainer ``exec()``s miner code in the validator process; a segfault
in submitted torch code (or an unbounded hang) takes the validator, the
services server and the backup daemon down with it, mid-round. This
wrapper runs ``local.trainer.run_training`` in a spawned subprocess with
a hard timeout: a crash or hang becomes an ordinary failed experiment.

Isolation policy (``RADAR_TRAIN_ISOLATION``):
  * ``auto`` (default) — subprocess for the torch tasks, in-process for
    the numpy synth task (spawn overhead dwarfs its runtime).
  * ``always`` / ``never`` — force either way.

The result dict crosses the process boundary by pickle (it's plain
JSON-able data); workdir paths are under /tmp so the parent can still
mirror artifacts and persist checkpoints from them.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Wallclock allowance for Phase C (the GIFT-Eval pass) + setup slack,
# added on top of the task's training budget to form the kill deadline.
DEFAULT_EVAL_ALLOWANCE_SEC = 3600
DEFAULT_SLACK_SEC = 600


def _failure(error: str, status: str = "subprocess_error") -> dict:
    return {
        "success": False,
        "metric": None,
        "objectives": {"flops_equivalent_size": 0, "num_params": 0,
                       "train_seconds": 0.0},
        "loss_curve": [],
        "val_curve": [],
        "analysis": f"training isolation: {error}",
        "error": error,
        "workdir": "",
        "harness_status": status,
    }


def _child(conn, code: str, kwargs: dict) -> None:
    """Subprocess entry: run training, ship the result back over a pipe."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [trainer-sub] %(message)s",
                        datefmt="%H:%M:%S")
    try:
        from local.trainer import run_training
        result = run_training(code, **kwargs)
    except BaseException as e:  # noqa: BLE001 — must always answer the pipe
        result = _failure(f"{type(e).__name__}: {e}")
    try:
        conn.send(result)
    finally:
        conn.close()


def _isolation_mode(task: Any) -> str:
    mode = os.environ.get("RADAR_TRAIN_ISOLATION", "auto").lower()
    if mode in ("always", "never"):
        return mode
    # auto: only the torch tasks are worth (and safe to) isolating.
    from local.task import (
        SyntheticDataGeneratorSpec, TSDataPipelineSpec, TSForecastingSpec,
    )
    heavy = isinstance(
        task, (TSForecastingSpec, TSDataPipelineSpec,
               SyntheticDataGeneratorSpec),
    )
    return "always" if heavy else "never"


def _timeout_for(task: Any, timeout_seconds: Optional[float]) -> float:
    if timeout_seconds is not None:
        return float(timeout_seconds)
    env = os.environ.get("RADAR_TRAIN_TIMEOUT_SEC", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    budget = float(getattr(task, "time_budget_seconds", 0) or 0)
    return budget * 1.5 + DEFAULT_EVAL_ALLOWANCE_SEC + DEFAULT_SLACK_SEC


def run_training_isolated(code: str, *,
                          timeout_seconds: Optional[float] = None,
                          **kwargs) -> dict:
    """Drop-in for ``local.trainer.run_training`` with subprocess
    isolation and a hard timeout (see module docstring for policy)."""
    task = kwargs.get("task")
    if _isolation_mode(task) == "never":
        from local.trainer import run_training
        return run_training(code, **kwargs)

    timeout = _timeout_for(task, timeout_seconds)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child, args=(child_conn, code, kwargs),
        name="radar-trainer", daemon=False,
    )
    proc.start()
    child_conn.close()

    result: Optional[dict] = None
    try:
        if parent_conn.poll(timeout):
            result = parent_conn.recv()
    except (EOFError, OSError) as e:
        logger.warning("training subprocess pipe broke: %s", e)
    finally:
        parent_conn.close()

    proc.join(timeout=5 if result is not None else 30)
    timed_out = proc.is_alive()
    if timed_out:
        logger.warning(
            "training subprocess still alive after %.0fs — terminating",
            timeout,
        )
        proc.terminate()
        proc.join(timeout=15)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)

    if result is not None:
        return result
    if timed_out:
        return _failure(
            f"training subprocess timed out after {timeout:.0f}s",
            status="subprocess_timeout",
        )
    return _failure(
        f"training subprocess died (exitcode={proc.exitcode}) — "
        "likely a native crash in submitted code",
        status="subprocess_crash",
    )
