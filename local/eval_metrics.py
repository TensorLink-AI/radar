"""GIFT-Eval metric finalization shared by every torch task dispatcher.

Three jobs, all downstream of the per-task breakdown ``prepare.validate``
already computes (and which used to be thrown away):

* **Persist** a compact per-dataset table into ``objectives['per_task']``
  so paired comparisons and lab reports have raw material.
* **Canary split** — when ``RADAR_EVAL_CANARY_FRAC`` > 0, a deterministic
  (name-hashed) fraction of GIFT tasks is held out of the scored
  aggregate. The canary aggregate is recorded but never selected on, so
  long-horizon benchmark overfitting shows up as scored/canary divergence.
* **Paired per-dataset Δ** — parent-vs-child comparison per dataset.
  Per-dataset noise is heavily correlated between the two runs, so a
  sign test over ~90 paired tasks has far more power than comparing two
  noisy geomeans.

Pure functions, numpy-free, importable everywhere.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Optional

CANARY_ENV = "RADAR_EVAL_CANARY_FRAC"
# Salt is versioned: changing it re-deals the canary set, which breaks
# scored-metric comparability — bump deliberately, never casually.
CANARY_SALT = "gift-eval-canary-v1"
# Paired sign test: minimum joined tasks before the test gates anything,
# and the one-sided z threshold (95%).
PAIRED_MIN_TASKS = 10
PAIRED_Z_THRESHOLD = 1.645


def geomean(values: list[float]) -> float:
    logs = [math.log(v) for v in values if math.isfinite(v) and v > 0]
    if not logs:
        return float("inf")
    return math.exp(sum(logs) / len(logs))


def task_metric(ncrps: float, nmase: float) -> float:
    """Per-dataset analogue of the leaderboard metric: sqrt(crps*mase)."""
    return math.sqrt(max(float(ncrps), 0.0) * max(float(nmase), 0.0))


def canary_frac() -> float:
    try:
        f = float(os.environ.get(CANARY_ENV, "0") or "0")
    except ValueError:
        return 0.0
    return min(max(f, 0.0), 0.5)


def is_canary(name: str, frac: float) -> bool:
    """Deterministic, order-independent membership by name hash."""
    if frac <= 0.0:
        return False
    h = hashlib.sha256(f"{CANARY_SALT}:{name}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    return u < frac


def compact_per_task(per_task: list[dict]) -> list[dict]:
    """Strip a prepare.py per-task row down to what comparisons need."""
    out: list[dict] = []
    for t in per_task or []:
        name = t.get("config_name") or t.get("name")
        nc, nm = t.get("normalized_crps", t.get("ncrps")), t.get(
            "normalized_mase", t.get("nmase"))
        if name is None or nc is None or nm is None:
            continue
        try:
            nc_f, nm_f = float(nc), float(nm)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(nc_f) and math.isfinite(nm_f)):
            continue
        out.append({"name": str(name), "ncrps": round(nc_f, 6),
                    "nmase": round(nm_f, 6)})
    return out


def finalize_gift_eval(eval_metrics: dict,
                       canary: Optional[float] = None) -> Optional[dict]:
    """Turn a ``_gift_eval_score`` result into the scored numbers + extras.

    Returns ``{"crps", "mase", "metric", "extras": {...}}`` or ``None``
    when no finite aggregate can be produced (caller records a failure).

    With a per-task breakdown the aggregates are recomputed here from the
    non-canary subset; without one (legacy eval path) the top-level
    crps/mase are passed through and the canary split is skipped.
    """
    frac = canary_frac() if canary is None else min(max(canary, 0.0), 0.5)
    per_task = compact_per_task(eval_metrics.get("per_task") or [])
    extras: dict = {}

    if per_task:
        extras["per_task"] = per_task
        scored = [t for t in per_task if not is_canary(t["name"], frac)]
        held = [t for t in per_task if is_canary(t["name"], frac)]
        if not scored:  # pathological frac/hash overlap — score everything
            scored, held = per_task, []
        crps = geomean([t["ncrps"] for t in scored])
        mase = geomean([t["nmase"] for t in scored])
        extras["n_tasks"] = len(scored)
        if frac > 0:
            extras["eval_canary_frac"] = frac
            extras["n_tasks_canary"] = len(held)
            if held:
                c_crps = geomean([t["ncrps"] for t in held])
                c_mase = geomean([t["nmase"] for t in held])
                if math.isfinite(c_crps) and math.isfinite(c_mase):
                    extras["canary_crps"] = c_crps
                    extras["canary_mase"] = c_mase
                    extras["canary_metric"] = math.sqrt(
                        max(c_crps, 0.0) * max(c_mase, 0.0))
    else:
        crps, mase = eval_metrics.get("crps"), eval_metrics.get("mase")
        if crps is None or mase is None:
            return None
        crps, mase = float(crps), float(mase)
        if "n_tasks" in eval_metrics:
            extras["n_tasks"] = int(eval_metrics["n_tasks"])

    if not (math.isfinite(crps) and math.isfinite(mase)):
        return None
    return {
        "crps": float(crps),
        "mase": float(mase),
        "metric": math.sqrt(max(crps, 0.0) * max(mase, 0.0)),
        "extras": extras,
    }


def paired_per_task_delta(parent_per_task: Optional[list[dict]],
                          child_per_task: Optional[list[dict]],
                          ) -> Optional[dict]:
    """Paired parent-vs-child comparison over the shared GIFT datasets.

    Per dataset the compared value is ``task_metric`` (lower=better);
    ``log_ratio = log(parent / child)`` so positive means the child
    improved. The sign test drops ties; ``significant`` is the one-sided
    95% verdict, only ever claimed with ≥ ``PAIRED_MIN_TASKS`` non-tied
    pairs. Returns ``None`` when either side lacks a usable breakdown.
    """
    if not parent_per_task or not child_per_task:
        return None
    parent_by = {t["name"]: t for t in compact_per_task(parent_per_task)}
    ratios: list[float] = []
    for t in compact_per_task(child_per_task):
        p = parent_by.get(t["name"])
        if p is None:
            continue
        pm = task_metric(p["ncrps"], p["nmase"])
        cm = task_metric(t["ncrps"], t["nmase"])
        if pm <= 0 or cm <= 0:
            continue
        ratios.append(math.log(pm / cm))
    if not ratios:
        return None
    non_tied = [r for r in ratios if r != 0.0]
    n_improved = sum(1 for r in non_tied if r > 0)
    n_eff = len(non_tied)
    z = ((n_improved - n_eff / 2.0) / math.sqrt(n_eff / 4.0)
         if n_eff > 0 else 0.0)
    return {
        "n": len(ratios),
        "n_improved": n_improved,
        "frac_improved": (n_improved / n_eff) if n_eff else 0.0,
        "mean_log_ratio": sum(ratios) / len(ratios),
        "z": z,
        "significant": bool(n_eff >= PAIRED_MIN_TASKS
                            and z >= PAIRED_Z_THRESHOLD),
    }
