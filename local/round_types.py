"""Validator-owned experiment round types beyond new/continuation.

The continuation machinery proved the pattern: seeded validator-owned
coins stamp a ``round_type`` and the miner only fills in the blank. This
module extends the vocabulary so the experiment record can attribute
*which* change mattered:

* ``replicate``    — re-run a frontier member's exact code with a new
                     seed. No miner. Measures the eval/seed noise floor
                     (``local/noise.py``) and evicts luck-based frontier
                     members from credibility.
* ``ablate``       — the validator picks a frontier member and asks the
                     miner for a *minimal one-component diff* of it.
* ``recipe_only``  — frozen architecture (AST-enforced), the miner may
                     only change optimizer / schedule / training config.
* ``transfer``     — a smaller-bucket frontier winner is handed to the
                     miner to scale into this round's (bigger) bucket.

All four downgrade to a normal round when their prerequisite (a usable
source experiment) is missing, mirroring the continuation downgrade.
"""

from __future__ import annotations

import ast
import random
from typing import Optional

SPECIAL_TYPES = ("replicate", "ablate", "recipe_only", "transfer")


def special_round_type(round_id: int, *, replicate_pct: float = 0.0,
                       ablate_pct: float = 0.0, recipe_pct: float = 0.0,
                       transfer_pct: float = 0.0) -> str:
    """One seeded draw mapped onto cumulative bands. '' = no special round.

    Independent of the continuation schedule's namespace; a special round
    pre-empts the continuation flip for that round.
    """
    bands = (
        ("replicate", max(0.0, replicate_pct)),
        ("ablate", max(0.0, ablate_pct)),
        ("recipe_only", max(0.0, recipe_pct)),
        ("transfer", max(0.0, transfer_pct)),
    )
    total = sum(p for _, p in bands)
    if total <= 0.0:
        return ""
    u = random.Random(f"special-round-{round_id}").random()
    acc = 0.0
    for name, p in bands:
        acc += p
        if u < acc:
            return name
    return ""


def _in_epoch(exp: dict, epoch: Optional[dict]) -> bool:
    if not epoch:
        return True
    objs = exp.get("objectives", {}) or {}
    return all(objs.get(k) == v for k, v in epoch.items())


def _frontier_members(experiments: list[dict], task: str,
                      epoch: Optional[dict]) -> list[dict]:
    """Successful, code-bearing, in-epoch members of this task's
    per-bucket Pareto union — replicates excluded (they're copies)."""
    from local.scoring import compute_pareto_by_bucket

    same_task = [
        e for e in experiments
        if e.get("task") == task and e.get("mode") != "replicate"
        and e.get("code")
    ]
    return [
        e for e in compute_pareto_by_bucket(same_task)
        if _in_epoch(e, epoch)
    ]


def _seeded_pick(members: list[dict], round_id: int, salt: str,
                 ) -> Optional[dict]:
    if not members:
        return None
    rng = random.Random(f"{salt}-{round_id}")
    return members[rng.randrange(len(members))]


def pick_replicate_source(experiments: list[dict], *, task: str,
                          round_id: int,
                          epoch: Optional[dict] = None) -> Optional[dict]:
    return _seeded_pick(
        _frontier_members(experiments, task, epoch), round_id, "replicate",
    )


def pick_ablation_target(experiments: list[dict], *, task: str,
                         round_id: int, min_flops: int, max_flops: int,
                         epoch: Optional[dict] = None) -> Optional[dict]:
    """Frontier member inside this round's bucket (gate-disabled tasks
    match everything)."""
    from local.scoring import passes_size_gate

    members = [
        e for e in _frontier_members(experiments, task, epoch)
        if passes_size_gate(e.get("objectives", {}), min_flops, max_flops)
    ]
    return _seeded_pick(members, round_id, "ablate")


def pick_transfer_source(experiments: list[dict], *, task: str,
                         round_id: int, min_flops: int,
                         buckets: dict[str, tuple[int, int]],
                         epoch: Optional[dict] = None) -> Optional[dict]:
    """Frontier member from any bucket strictly below this round's —
    the design that earned its place small and hasn't been tried big."""
    from local.scoring import passes_size_gate

    smaller = [(lo, hi) for lo, hi in buckets.values() if hi <= min_flops]
    members = [
        e for e in _frontier_members(experiments, task, epoch)
        if any(passes_size_gate(e.get("objectives", {}), lo, hi)
               for lo, hi in smaller)
    ]
    return _seeded_pick(members, round_id, "transfer")


def source_card(exp: dict, buckets: Optional[dict] = None) -> dict:
    """Compact challenge-payload view of a source/target experiment."""
    objs = exp.get("objectives", {}) or {}
    card = {
        "id": exp["id"],
        "name": exp.get("name"),
        "metric": exp.get("metric"),
        "flops_equivalent_size": objs.get("flops_equivalent_size"),
        "code": exp.get("code"),
    }
    if buckets:
        from local.scoring import passes_size_gate
        for bname, (lo, hi) in buckets.items():
            if passes_size_gate(objs, lo, hi):
                card["source_bucket"] = bname
                break
    return card


# ── recipe_only enforcement ─────────────────────────────────────────


def _build_model_dump(code: str) -> Optional[str]:
    """Canonical AST dump of ``build_model`` plus every top-level class —
    the architecture surface a recipe-only round must leave untouched.
    Optimizer/scheduler/config hooks (``build_optimizer`` etc.) are free.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    keep: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            keep.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "build_model":
                keep.append(node)
    if not keep:
        return None
    return "\n".join(
        ast.dump(n, annotate_fields=False, include_attributes=False)
        for n in keep
    )


def same_build_model(code_a: str, code_b: str) -> bool:
    """True when the architecture surface (build_model + classes) of the
    two submissions is AST-identical (formatting/comments ignored)."""
    a, b = _build_model_dump(code_a), _build_model_dump(code_b)
    return a is not None and a == b


def enforce_recipe_only(submission_code: str,
                        base_code: str) -> tuple[bool, str]:
    """Soft gate for recipe_only rounds: returns (ok, note)."""
    if same_build_model(submission_code, base_code):
        return True, ""
    return False, ("recipe_only violated: build_model/classes differ from "
                   "the base — treated as a normal new-design round")
