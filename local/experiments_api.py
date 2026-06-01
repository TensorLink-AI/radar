"""HTTP handlers for ``/experiments/*``, ``/frontier``, and ``/challenge``.

Pure functions over a ``LocalStore`` — kept out of ``services.py`` so the
HTTP router stays a thin dispatcher and ``services.py`` stays under the
300-line cap.

Endpoint coverage (mirrors the menu in ``miners/*/prompts.py``):

  GET  /frontier?task=                  strict Pareto frontier
  GET  /experiments/recent?n=           most recent N (``limit=`` alias)
  GET  /experiments/pareto?task=        same as /frontier locally
  GET  /experiments/failures?n=         recent failures
  GET  /experiments/families?task=      group by name
  GET  /experiments/stats?task=         aggregate counts
  GET  /experiments/tasks               distinct task names
  GET  /experiments/{idx}               one record
  GET  /experiments/{idx}/diff          diff vs ``parent_index``
  GET  /experiments/lineage/{idx}       root → idx chain
  GET  /experiments/{idx}/lineage_diffs per-step diffs along the chain
  GET  /experiments/diff/{a}/{b}        pair diff
  POST /experiments/search              body ``{"query": "..."}``
  GET  /challenge                       active round metadata
"""

from __future__ import annotations

import difflib
from typing import Optional

from local.continuation import continuation_frontier
from local.scoring import compute_pareto
from local.store import LocalStore


def _filter_task(exps: list[dict], task: Optional[str]) -> list[dict]:
    if not task:
        return exps
    return [e for e in exps if e.get("task") == task]


def frontier(store: LocalStore, task: Optional[str] = None) -> dict:
    exps = _filter_task(store.recent_experiments(n=10_000), task)
    return {
        "frontier": compute_pareto(exps),
        "continuation": continuation_frontier(exps),
    }


def _parent_summary(exp: dict) -> dict:
    """Compact, weights-free view of a continuation-eligible parent."""
    objs = exp.get("objectives", {}) or {}
    curve = exp.get("loss_curve") or []
    return {
        "id": exp["id"],
        "name": exp.get("name"),
        "task": exp.get("task"),
        "metric": exp.get("metric"),
        "n_rounds": exp.get("n_rounds", 1),
        "cumulative_compute": exp.get("cumulative_compute", 0.0),
        "best_val_loss": objs.get("best_val_loss"),
        "checkpoint_available": exp.get("checkpoint_ref") is not None,
        "loss_curve_tail": [float(x) for x in curve[-8:]],
        "num_params": objs.get("num_params"),
        "flops_equivalent_size": objs.get("flops_equivalent_size"),
    }


def parents(
    store: LocalStore,
    *,
    task: Optional[str] = None,
    min_flops: Optional[int] = None,
    max_flops: Optional[int] = None,
) -> dict:
    """Continuation-eligible parents for a bucket.

    When ``min_flops`` / ``max_flops`` are omitted they're taken from the
    active challenge so an agent can just call ``GET /parents``.
    """
    if min_flops is None or max_flops is None:
        ch = store.open_challenge()
        payload = (ch or {}).get("payload", {}) if ch else {}
        if min_flops is None:
            min_flops = int(payload.get("min_flops_equivalent", 0) or 0)
        if max_flops is None:
            max_flops = int(payload.get("max_flops_equivalent", 0) or 0)
        if task is None:
            task = (payload.get("task", {}) or {}).get("name")
    rows = store.eligible_parents(
        task=task, min_flops=int(min_flops), max_flops=int(max_flops),
    )
    return {"parents": [_parent_summary(e) for e in rows]}


def signature(store: LocalStore, idx: int) -> dict:
    """Tensor names + shapes of an experiment's checkpoint (no weights).

    Lets a continuation miner author code whose ``build_model`` produces a
    state_dict that loads cleanly over the parent (warm-start is strict).
    """
    exp = store.get_experiment(idx)
    if exp is None:
        return {"error": "not found"}
    sig = (exp.get("objectives", {}) or {}).get("param_signature")
    return {
        "id": idx,
        "checkpoint_available": exp.get("checkpoint_ref") is not None,
        "signature": sig or {},
    }


def trajectory(store: LocalStore, idx: int) -> dict:
    """Stitch a lineage's loss curve + GIFT-eval points, root → idx.

    Per-segment loss samples are concatenated and tagged with their source
    experiment so a miner can see where each continuation began (and
    whether it actually bent the curve). The GIFT-eval series carries one
    ``(cumulative_compute, metric)`` point per lineage member — the real
    objective trajectory whose deltas are the continuation-frontier Δ's.
    """
    chain = store.lineage(idx)
    loss_curve: list[dict] = []
    gift_eval: list[dict] = []
    boundaries: list[dict] = []
    for exp in chain:
        objs = exp.get("objectives", {}) or {}
        boundaries.append({
            "experiment_id": exp["id"],
            "n_rounds": exp.get("n_rounds", 1),
            "start_index": len(loss_curve),
        })
        for v in (exp.get("loss_curve") or []):
            loss_curve.append({
                "i": len(loss_curve),
                "loss": float(v),
                "experiment_id": exp["id"],
                "n_rounds": exp.get("n_rounds", 1),
            })
        gift_eval.append({
            "experiment_id": exp["id"],
            "n_rounds": exp.get("n_rounds", 1),
            "cumulative_compute": exp.get("cumulative_compute", 0.0),
            "metric": exp.get("metric"),
            "best_val_loss": objs.get("best_val_loss"),
        })
    return {
        "id": idx,
        "lineage": [e["id"] for e in chain],
        "loss_curve": loss_curve,
        "gift_eval": gift_eval,
        "boundaries": boundaries,
    }


def pareto(store: LocalStore, task: Optional[str] = None) -> dict:
    # Local stack doesn't track near-frontier separately; same as frontier.
    return frontier(store, task)


def recent(store: LocalStore, n: int = 10,
           task: Optional[str] = None) -> list[dict]:
    n = max(int(n), 1)
    if task is None:
        return store.recent_experiments(n=n)
    rows = _filter_task(store.recent_experiments(n=10_000), task)
    return rows[:n]


def failures(store: LocalStore, n: int = 5,
             task: Optional[str] = None) -> dict:
    rows = store.recent_experiments(n=10_000)
    rows = [e for e in rows if not e.get("success")]
    rows = _filter_task(rows, task)
    return {"failures": rows[:max(int(n), 1)]}


def stats(store: LocalStore, task: Optional[str] = None) -> dict:
    if task is None:
        return store.stats()
    exps = _filter_task(store.recent_experiments(n=10_000), task)
    total = len(exps)
    successful = sum(1 for e in exps if e.get("success"))
    metrics = [
        e["metric"] for e in exps
        if e.get("success") and e.get("metric") is not None
    ]
    return {
        "task": task,
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "best_metric": min(metrics) if metrics else None,
        "worst_metric": max(metrics) if metrics else None,
        "mean_metric": sum(metrics) / len(metrics) if metrics else None,
    }


def tasks(store: LocalStore) -> dict:
    rows = store.recent_experiments(n=10_000)
    names = sorted({e["task"] for e in rows if e.get("task")})
    return {"tasks": names}


def list_artifacts(
    store: LocalStore,
    *,
    round_id: Optional[int] = None,
    miner_id: Optional[str] = None,
    task: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 200,
) -> dict:
    rows = store.list_artifacts(
        round_id=round_id, miner_id=miner_id, task=task, kind=kind, limit=limit,
    )
    return {"artifacts": rows}


def artifacts_for_experiment(store: LocalStore, exp_id: int) -> dict:
    exp = store.get_experiment(exp_id)
    if exp is None:
        return {"error": "not found", "artifacts": []}
    rows = store.list_artifacts(
        round_id=exp["round_id"], miner_id=exp["miner_id"], limit=500,
    )
    return {
        "experiment_id": exp_id,
        "round_id": exp["round_id"],
        "miner_id": exp["miner_id"],
        "artifacts": rows,
    }


def families(store: LocalStore, task: Optional[str] = None) -> dict:
    """Group successful experiments by ``name`` as a proxy for arch families.

    Real radar clusters on architecture features; locally we settle for
    name-based grouping. Sorted by descending member count.
    """
    exps = _filter_task(store.recent_experiments(n=10_000), task)
    by_name: dict[str, list[dict]] = {}
    for e in exps:
        if not e.get("success"):
            continue
        by_name.setdefault(e.get("name") or "?", []).append(e)
    fams = []
    for name, members in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        metrics = [m["metric"] for m in members if m.get("metric") is not None]
        fams.append({
            "name": name,
            "count": len(members),
            "best_metric": min(metrics) if metrics else None,
            "members": [m["id"] for m in members],
        })
    return {"families": fams}


def search(store: LocalStore, query: str) -> dict:
    if not query:
        return {"query": "", "matches": []}
    needle = query.lower()
    rows = store.recent_experiments(n=10_000)
    hits = []
    for e in rows:
        haystack = " ".join([
            e.get("name") or "", e.get("code") or "",
            e.get("motivation") or "", e.get("reasoning") or "",
        ]).lower()
        if needle in haystack:
            hits.append({
                "id": e["id"], "name": e["name"], "metric": e["metric"],
                "success": e["success"], "task": e["task"],
            })
    return {"query": query, "matches": hits}


def _diff_text(a: dict, b: dict) -> str:
    a_code = (a.get("code") or "").splitlines(keepends=True)
    b_code = (b.get("code") or "").splitlines(keepends=True)
    return "".join(difflib.unified_diff(
        a_code, b_code,
        fromfile=f"exp-{a['id']}", tofile=f"exp-{b['id']}",
    ))


def parent_diff(store: LocalStore, idx: int) -> dict:
    child = store.get_experiment(idx)
    if child is None:
        return {"error": "not found"}
    pid = child.get("parent_index")
    if pid is None:
        return {"id": idx, "parent_index": None, "diff": ""}
    parent = store.get_experiment(pid)
    if parent is None:
        return {"id": idx, "parent_index": pid, "diff": "",
                "error": "parent missing"}
    return {"id": idx, "parent_index": pid, "diff": _diff_text(parent, child)}


def pair_diff(store: LocalStore, a: int, b: int) -> dict:
    ea = store.get_experiment(a)
    eb = store.get_experiment(b)
    if ea is None or eb is None:
        return {"error": "not found"}
    return {"a": a, "b": b, "diff": _diff_text(ea, eb)}


def lineage(store: LocalStore, idx: int) -> dict:
    chain: list[dict] = []
    seen: set[int] = set()
    cur = store.get_experiment(idx)
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        chain.append({
            "id": cur["id"], "name": cur["name"], "metric": cur["metric"],
            "success": cur["success"], "parent_index": cur["parent_index"],
        })
        pid = cur.get("parent_index")
        if pid is None:
            break
        cur = store.get_experiment(pid)
    chain.reverse()
    return {"id": idx, "chain": chain}


def lineage_diffs(store: LocalStore, idx: int) -> dict:
    chain = lineage(store, idx)["chain"]
    diffs = []
    for prev, curr in zip(chain, chain[1:]):
        a = store.get_experiment(prev["id"])
        b = store.get_experiment(curr["id"])
        if a and b:
            diffs.append({
                "from": prev["id"], "to": curr["id"],
                "diff": _diff_text(a, b),
            })
    return {"id": idx, "diffs": diffs}


def active_challenge(store: LocalStore) -> dict:
    ch = store.open_challenge()
    if ch is None:
        return {"active": False}
    return {"active": True, **ch}
