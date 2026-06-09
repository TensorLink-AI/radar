"""Cheap screening tier — triage k candidates before one full run.

Without this, every hypothesis pays full price (an hour of training plus
the full GIFT-Eval pass). When screening is enabled the challenge
advertises it, the agent may submit ``candidates: [{name, code}, ...]``
alongside its primary submission, and the validator gives each candidate
a short training budget and a truncated eval
(``RADAR_GIFT_EVAL_MAX_TASKS``) — then promotes only the screening
winner to the full run. Screening results are recorded in the final
experiment's ``objectives['screening']``, never as experiment rows
(subset metrics aren't comparable to full ones).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 8


def screening_card(*, max_candidates: int, budget_seconds: int,
                   eval_max_tasks: int) -> dict:
    """The challenge-payload block advertising screening to the agent."""
    return {
        "enabled": True,
        "max_candidates": int(max_candidates),
        "budget_seconds": int(budget_seconds),
        "eval_max_tasks": int(eval_max_tasks),
        "note": (
            "submit candidates=[{name, code}, ...] alongside your primary "
            "code; each gets budget_seconds of training and a truncated "
            "eval, and only the screening winner runs at full budget"
        ),
    }


def _valid_candidates(raw) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for c in raw[:MAX_CANDIDATES]:
        if isinstance(c, dict) and isinstance(c.get("code"), str) and c["code"].strip():
            out.append({"name": str(c.get("name") or f"candidate-{len(out)}"),
                        "code": c["code"]})
    return out


def screen_candidates(raw_candidates, *, task, seed: int,
                      min_flops: int, max_flops: int,
                      budget_seconds: int, eval_max_tasks: int,
                      train_fn: Callable[..., dict],
                      ) -> tuple[Optional[dict], list[dict]]:
    """Short-budget run per candidate; returns (winner, summaries).

    ``winner`` is ``None`` when no candidate survived screening (caller
    falls back to the primary submission). Screen-run workdirs are
    reclaimed immediately.
    """
    from local.artifacts import cleanup_workdir

    candidates = _valid_candidates(raw_candidates)
    if len(candidates) < 2:
        return None, []

    screen_task = dataclasses.replace(
        task, time_budget_seconds=int(budget_seconds),
    )
    saved = os.environ.get("RADAR_GIFT_EVAL_MAX_TASKS")
    os.environ["RADAR_GIFT_EVAL_MAX_TASKS"] = str(int(eval_max_tasks))
    summaries: list[dict] = []
    try:
        for i, cand in enumerate(candidates):
            logger.info(
                "  screening %d/%d: '%s' (%ds budget, %d eval tasks)",
                i + 1, len(candidates), cand["name"], budget_seconds,
                eval_max_tasks,
            )
            try:
                result = train_fn(
                    cand["code"], seed=seed, task=screen_task,
                    min_flops=min_flops, max_flops=max_flops,
                )
            except Exception as e:  # noqa: BLE001
                result = {"success": False, "metric": None,
                          "error": f"{type(e).__name__}: {e}", "workdir": ""}
            summaries.append({
                "name": cand["name"],
                "success": bool(result.get("success")),
                "metric": result.get("metric"),
                "error": (result.get("error") or "")[:300],
            })
            wd = result.get("workdir") or ""
            if wd:
                cleanup_workdir(Path(wd))
    finally:
        if saved is None:
            os.environ.pop("RADAR_GIFT_EVAL_MAX_TASKS", None)
        else:
            os.environ["RADAR_GIFT_EVAL_MAX_TASKS"] = saved

    survivors = [
        (s, c) for s, c in zip(summaries, candidates)
        if s["success"] and s["metric"] is not None
    ]
    if not survivors:
        return None, summaries
    best_summary, best_cand = min(survivors, key=lambda sc: sc[0]["metric"])
    best_summary["winner"] = True
    logger.info(
        "  screening winner: '%s' (screen metric=%.6f)",
        best_cand["name"], best_summary["metric"],
    )
    return best_cand, summaries
