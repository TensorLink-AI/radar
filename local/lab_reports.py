"""Auto-generated structured post-mortems — one per experiment.

A round burns ~an hour of training plus a full GIFT-Eval pass; without
this module what's retained is a code blob and one scalar. The lab
report converts that spend into a retrievable finding: hypothesis (the
miner's motivation), what was compared against what, the paired
per-dataset outcome, and a noise-aware verdict. Reports accumulate in
SQLite (``lab_reports`` table) and are served to agents at
``GET /lab_reports`` and ``GET /experiments/{id}/report`` so later
rounds can build on validated evidence instead of folklore.

Deterministic by default; an optional one-call LLM narrative is added
when a provider key is configured (``RADAR_LAB_REPORT_LLM=0`` opts out).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

from local.eval_metrics import paired_per_task_delta, task_metric
from local.noise import noise_floor, within_noise

logger = logging.getLogger(__name__)

_HYPOTHESIS_MAX = 600
_TOP_N_DATASETS = 5

# objectives key → human label for the comparison anchor each special
# round type pins. Order matters: the first present key wins.
_BASELINE_KEYS = (
    ("replicate_of", "replicate_of"),
    ("ablation_of", "ablation_target"),
    ("recipe_only_of", "recipe_base"),
    ("transfer_of", "transfer_source"),
)


def _baseline_for(store, exp: dict) -> tuple[Optional[dict], str]:
    """Resolve the experiment this one should be compared against."""
    objs = exp.get("objectives", {}) or {}
    for key, label in _BASELINE_KEYS:
        ref = objs.get(key)
        if ref is not None:
            base = store.get_experiment(int(ref))
            if base is not None:
                return base, label
    pid = exp.get("parent_index")
    if pid is not None:
        base = store.get_experiment(int(pid))
        if base is not None:
            return base, "parent"
    return None, ""


def _dataset_movers(paired_base: Optional[list[dict]],
                    paired_child: Optional[list[dict]]) -> dict:
    """Top improved/regressed datasets by per-task log-ratio."""
    if not paired_base or not paired_child:
        return {}
    base_by = {t["name"]: t for t in paired_base}
    moves: list[tuple[str, float]] = []
    for t in paired_child:
        b = base_by.get(t["name"])
        if b is None:
            continue
        bm = task_metric(b["ncrps"], b["nmase"])
        cm = task_metric(t["ncrps"], t["nmase"])
        if bm <= 0 or cm <= 0:
            continue
        moves.append((t["name"], math.log(bm / cm)))
    if not moves:
        return {}
    moves.sort(key=lambda kv: -kv[1])
    fmt = lambda kv: {"name": kv[0], "log_ratio": round(kv[1], 4)}  # noqa: E731
    return {
        "top_improved": [fmt(m) for m in moves[:_TOP_N_DATASETS] if m[1] > 0],
        "top_regressed": [fmt(m) for m in moves[-_TOP_N_DATASETS:] if m[1] < 0],
    }


def _diff_stats(base_code: str, code: str) -> dict:
    import difflib
    added = removed = 0
    for line in difflib.unified_diff(
        (base_code or "").splitlines(), (code or "").splitlines(), n=0,
    ):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"lines_added": added, "lines_removed": removed}


def _verdict(exp: dict, baseline: Optional[dict], baseline_kind: str,
             delta: Optional[float], paired: Optional[dict],
             floor: Optional[dict]) -> str:
    if not exp.get("success"):
        err = (exp.get("analysis") or "failed").splitlines()[0]
        return f"failed: {err[:200]}"
    if exp.get("mode") == "replicate" and baseline is not None:
        wn = within_noise(delta or 0.0, floor)
        band = ("within" if wn else "beyond") if wn is not None else "unknown vs"
        return (f"noise probe: |Δ|={abs(delta or 0.0):.5f} vs original "
                f"#{baseline['id']} — {band} current noise band")
    if baseline is None:
        return "new design; no baseline to compare against"
    if delta is None:
        return f"no comparable metric vs {baseline_kind} #{baseline['id']}"
    if delta <= 0:
        return (f"no improvement vs {baseline_kind} #{baseline['id']} "
                f"(Δ={delta:.5f})")
    sig = paired.get("significant") if paired else None
    wn = within_noise(delta, floor)
    if sig:
        return (f"improvement vs {baseline_kind} #{baseline['id']} "
                f"(Δ={delta:.5f}, paired-significant "
                f"{paired['n_improved']}/{paired['n']})")
    if sig is False and paired and paired.get("n", 0) >= 10:
        return (f"apparent improvement Δ={delta:.5f} NOT paired-significant "
                f"({paired['n_improved']}/{paired['n']}) — likely noise")
    if wn:
        return f"improvement Δ={delta:.5f} within noise band — uncredited"
    return f"improvement vs {baseline_kind} #{baseline['id']} (Δ={delta:.5f})"


def build_report(store, exp: dict,
                 floor: Optional[dict] = None) -> dict:
    """Deterministic structured report for one experiment row."""
    objs = exp.get("objectives", {}) or {}
    baseline, baseline_kind = _baseline_for(store, exp)
    delta: Optional[float] = None
    if (baseline is not None and exp.get("metric") is not None
            and baseline.get("metric") is not None):
        delta = float(baseline["metric"]) - float(exp["metric"])

    base_objs = (baseline or {}).get("objectives", {}) or {}
    paired = paired_per_task_delta(
        base_objs.get("per_task"), objs.get("per_task"),
    )

    report: dict = {
        "experiment_id": exp["id"],
        "round_id": exp.get("round_id"),
        "task": exp.get("task"),
        "name": exp.get("name"),
        "miner_id": exp.get("miner_id"),
        "mode": exp.get("mode", "new"),
        "round_kind": _round_kind(exp),
        "hypothesis": (exp.get("motivation") or "")[:_HYPOTHESIS_MAX],
        "outcome": {
            "success": bool(exp.get("success")),
            "metric": exp.get("metric"),
            "crps": objs.get("crps"),
            "mase": objs.get("mase"),
            "canary_metric": objs.get("canary_metric"),
            # Exposed score-correlated proxy (held-out GIFT slice) — the
            # feedback signal agents climb between rounds.
            "proxy_metric": objs.get("proxy_metric"),
            "cumulative_compute": objs.get("cumulative_compute"),
        },
    }
    if baseline is not None:
        report["baseline"] = {
            "kind": baseline_kind,
            "experiment_id": baseline["id"],
            "name": baseline.get("name"),
            "metric": baseline.get("metric"),
        }
        report["delta"] = delta
        report["diff_stats"] = _diff_stats(
            baseline.get("code") or "", exp.get("code") or "",
        )
        movers = _dataset_movers(
            base_objs.get("per_task"), objs.get("per_task"),
        )
        if movers:
            report["datasets"] = movers
    if paired:
        report["paired"] = paired
    if floor and floor.get("n_pairs"):
        report["noise_floor"] = {
            "n_pairs": floor["n_pairs"],
            "threshold": floor["threshold"],
        }
    report["verdict"] = _verdict(
        exp, baseline, baseline_kind, delta, paired, floor,
    )
    return report


def _round_kind(exp: dict) -> str:
    objs = exp.get("objectives", {}) or {}
    for key, _ in _BASELINE_KEYS:
        if objs.get(key) is not None:
            return key.replace("_of", "")
    if exp.get("mode") == "continue":
        kind = objs.get("continuation_kind")
        return f"continuation:{kind}" if kind else "continuation"
    return "new"


def _maybe_narrate(report: dict) -> None:
    """Optional one-call LLM narrative; never fatal, never on the stub."""
    if os.environ.get("RADAR_LAB_REPORT_LLM", "1") == "0":
        return
    if not (os.environ.get("CHUTES_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        return
    try:
        import json as _json
        from local.providers import llm_dispatch
        slim = {k: v for k, v in report.items() if k != "narrative"}
        resp = llm_dispatch({
            "messages": [{
                "role": "user",
                "content": (
                    "You are the lab-notebook scribe for an automated ML "
                    "research system. Summarize this experiment report in "
                    "2-3 sentences: what was tried, what happened, and the "
                    "actionable takeaway. Be concrete; no hedging filler.\n\n"
                    + _json.dumps(slim)
                ),
            }],
        })
        text = (resp or {}).get("content", "").strip()
        if text:
            report["narrative"] = text[:1500]
    except Exception as e:  # noqa: BLE001
        logger.debug("lab report narrative skipped: %s", e)


def backfill_reports(store, *, task: Optional[str] = None,
                     limit: Optional[int] = None,
                     narrate: bool = False) -> int:
    """Build + persist reports for experiments that don't have one yet.

    Reports are otherwise only written at round end, so any experiment that
    ran before report generation was wired in (or a round where generation
    was skipped) leaves a permanent hole the read-only dashboard can never
    fill. This walks the experiments table and fills those gaps. Idempotent
    — experiments with a stored report are skipped. The noise floor is
    computed once per distinct task. LLM narration is off by default so a
    bulk backfill doesn't fan out into hundreds of provider calls.
    """
    existing = store.lab_report_experiment_ids()
    exps = store.recent_experiments(n=limit or 1_000_000)
    floors: dict[str, Optional[dict]] = {}
    n = 0
    for exp in exps:
        if exp.get("id") in existing:
            continue
        t = exp.get("task") or ""
        if task is not None and t != task:
            continue
        floor = floors.get(t)
        if floor is None:
            floor = noise_floor(store.recent_experiments(n=10_000), task=t)
            floors[t] = floor
        try:
            report = build_report(store, exp, floor=floor)
            if narrate:
                _maybe_narrate(report)
            store.add_lab_report(
                experiment_id=exp["id"],
                round_id=exp.get("round_id", 0), task=t, report=report,
            )
            n += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("backfill lab report failed for exp=%s: %s",
                           exp.get("id"), e)
    return n


def generate_round_reports(store, round_id: int, task: str) -> int:
    """Build + persist a report for every experiment of this round."""
    exps = [
        e for e in store.recent_experiments(n=500)
        if e.get("round_id") == round_id and e.get("task") == task
    ]
    if not exps:
        return 0
    floor = noise_floor(store.recent_experiments(n=10_000), task=task)
    n = 0
    for exp in exps:
        try:
            report = build_report(store, exp, floor=floor)
            _maybe_narrate(report)
            store.add_lab_report(
                experiment_id=exp["id"], round_id=round_id, task=task,
                report=report,
            )
            n += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("lab report failed for exp=%s: %s", exp.get("id"), e)
    return n


# ── HTTP handlers (routed from local/services.py) ───────────────────


def recent_reports(store, n: int = 20, task: Optional[str] = None) -> dict:
    return {"reports": store.recent_lab_reports(n=n, task=task)}


def report_for(store, experiment_id: int) -> dict:
    """Stored report, or built on demand (backfill for old experiments)."""
    report = store.get_lab_report(experiment_id)
    if report is not None:
        return {"report": report}
    exp = store.get_experiment(experiment_id)
    if exp is None:
        return {"error": "not found"}
    floor = noise_floor(store.recent_experiments(n=10_000),
                        task=exp.get("task"))
    report = build_report(store, exp, floor=floor)
    store.add_lab_report(
        experiment_id=experiment_id, round_id=exp.get("round_id", 0),
        task=exp.get("task") or "", report=report,
    )
    return {"report": report}


# ── CLI: backfill reports for an existing DB ────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m local.lab_reports --backfill`` — populate missing reports.

    Use against a DB whose rounds predate report generation (the dashboard
    is read-only and can't fill the table itself).
    """
    import argparse

    from local.store import LocalStore

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--db", default="local/radar_local.db")
    parser.add_argument("--backfill", action="store_true",
                        help="Build reports for experiments lacking one.")
    parser.add_argument("--task", default=None,
                        help="Restrict to a single task name.")
    parser.add_argument("--narrate", action="store_true",
                        help="Add the optional LLM narrative (needs a key).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    store = LocalStore(args.db)
    try:
        if args.backfill:
            n = backfill_reports(store, task=args.task, narrate=args.narrate)
            print(f"backfilled {n} lab report(s)")
        else:
            parser.error("nothing to do — pass --backfill")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
