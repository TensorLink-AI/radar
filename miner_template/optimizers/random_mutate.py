"""Baseline optimizer: rank prompts by mean score and clone+perturb winners.

No LLM, no DSPy dependency — used as a CI-safe default and as a
sanity-check sibling to GEPA.  Useful when a miner wants to bootstrap
a population without an LLM cost on day one.

Strategy::

    1.  group ResultRow by prompt_id; compute mean score per prompt
    2.  promote the top-K to the new population
    3.  for each survivor, generate (population_size - K) // K children
        by appending a deterministic perturbation suffix
    4.  if no scores yet, simply clone the active population
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from miner_template.optimizers import ResultRow
from miner_template.prompts import Prompt

# Cheap deterministic prompt perturbations.  Picked by hashing the
# parent's id + child index so two runs on the same inputs produce the
# same children — important for reproducibility.
#
# Perturbations are deliberately architecture-agnostic: they nudge the
# *process* (exploration, frontier analysis, hypothesis tracking) rather
# than prescribing a specific op family. Architecture-prescriptive
# directives compound across generations and collapse the population
# toward whatever the prompt currently favors — exactly what the
# Pareto-front sampler is supposed to prevent.
_PERTURBATIONS: list[str] = [
    "\n\nThink step by step before answering.",
    "\n\nBefore committing to a design, list at least three structurally "
    "different candidates and pick the one whose inductive bias best "
    "matches the task — do not default to a familiar shortlist.",
    "\n\nSurvey the current frontier for op families that are absent "
    "(state-space, spectral mixing, gated convs, learned routing, etc.) "
    "and consider whether one of those gaps is worth exploring.",
    "\n\nState the hypothesis your architecture is testing in one "
    "sentence in the motivation field, so future rounds can tell "
    "exploration from exploitation.",
    "\n\nReport hyperparameter choices in the motivation field.",
    "\n\nOptimize for sample efficiency over raw capacity.",
    "\n\nDescribe each layer by its operation, shape, and information "
    "flow — name the mechanism, not the brand.",
    "\n\nIf the frontier is narrow, prefer a structurally novel "
    "candidate over a marginal tweak of an existing member.",
]


def _mean_score(rows: Iterable[ResultRow], key: str = "raw_score") -> float:
    vals = [float(r.scores.get(key, 0.0) or 0.0) for r in rows]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _rank_population(
    results: list[ResultRow],
    population: list[Prompt],
    score_key: str,
) -> list[tuple[Prompt, float]]:
    """Return ``(prompt, mean_score)`` sorted best-first.  Prompts with
    no observed rows fall to the bottom but stay in the ranking."""
    by_pid: dict[str, list[ResultRow]] = {}
    for r in results:
        if r.prompt_id:
            by_pid.setdefault(r.prompt_id, []).append(r)
    scored: list[tuple[Prompt, float]] = []
    for p in population:
        scored.append((p, _mean_score(by_pid.get(p.id, []), score_key)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def _perturb(parent: Prompt, child_idx: int, generation: int) -> Prompt:
    seed = hashlib.sha256(
        f"{parent.id}:{child_idx}".encode()
    ).digest()
    idx = seed[0] % len(_PERTURBATIONS)
    # Preserve parent metadata (e.g. ``slot`` for multi-slot
    # populations) so mutated children stay routable.
    child_metadata = dict(parent.metadata or {})
    child_metadata.update({
        "mutation": "random_mutate",
        "perturbation_idx": idx,
    })
    return Prompt.new(
        template=parent.template + _PERTURBATIONS[idx],
        generation=generation,
        parent_id=parent.id,
        metadata=child_metadata,
    )


def optimize(
    results: list[ResultRow],
    population: list[Prompt],
    config: dict,
) -> list[Prompt]:
    """Return the next population.

    Config keys:
      ``population`` (int, default 8): target size of the new population.
      ``elite_k``   (int, default 2): how many top performers to carry.
      ``score_key`` (str, default "raw_score"): scores field to rank on.
    """
    target = int(config.get("population", 8))
    elite_k = int(config.get("elite_k", 2))
    score_key = str(config.get("score_key", "raw_score"))
    if target < 1:
        raise ValueError("population must be >= 1")
    if elite_k < 1:
        elite_k = 1

    if not population:
        return []  # caller should seed_default() before optimizing

    next_gen = max(p.generation for p in population) + 1

    ranked = _rank_population(results, population, score_key)
    elites = [p for p, _ in ranked[:elite_k]]

    new_pop: list[Prompt] = list(elites)
    child_idx = 0
    while len(new_pop) < target:
        parent = elites[child_idx % len(elites)]
        new_pop.append(_perturb(parent, child_idx, next_gen))
        child_idx += 1

    return new_pop[:target]
