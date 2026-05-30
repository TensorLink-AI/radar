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
_PERTURBATIONS: list[str] = [
    "\n\nThink step by step before answering.",
    "\n\nFavor architectures with explicit attention sparsity.",
    "\n\nPrefer designs that reuse parameters across layers.",
    "\n\nConsider hybrid CNN+attention encoders.",
    "\n\nWhen in doubt, fall back to a small transformer baseline.",
    "\n\nReport hyperparameter choices in the motivation field.",
    "\n\nOptimize for sample efficiency over raw capacity.",
    "\n\nAvoid recurrent layers unless justified by data length.",
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
