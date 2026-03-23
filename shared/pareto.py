"""
Pareto front management with diversity-aware candidate sampling.

Now task-agnostic: objectives come from the TaskSpec, not hardcoded.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

from shared.database import DataElement


@dataclass
class ParetoCandidate:
    """A candidate on the Pareto front with selection metadata."""
    element: DataElement
    selection_count: int = 0
    successful_children: int = 0
    ucb_score: float = float("inf")


class ParetoFront:
    """
    Multi-objective Pareto front with UCT-inspired sampling.

    Objectives are defined by a pluggable function. All values normalized
    so that lower = better internally for dominance comparisons.
    """

    def __init__(
        self,
        max_size: int = 50,
        objective_fn: Optional[Callable[[DataElement], tuple[float, ...]]] = None,
    ):
        self.max_size = max_size
        self.candidates: list[ParetoCandidate] = []
        self.total_selections: int = 0
        self._objective_fn = objective_fn or self._default_objectives

    def _dominates(self, a: DataElement, b: DataElement) -> bool:
        objs_a = self._objective_fn(a)
        objs_b = self._objective_fn(b)
        at_least_as_good = all(oa <= ob for oa, ob in zip(objs_a, objs_b))
        strictly_better = any(oa < ob for oa, ob in zip(objs_a, objs_b))
        return at_least_as_good and strictly_better

    @staticmethod
    def _default_objectives(elem: DataElement) -> tuple[float, ...]:
        metric = elem.metric if elem.metric is not None else float("inf")
        exec_time = elem.objectives.get("exec_time", float("inf"))
        memory_mb = elem.objectives.get("memory_mb", float("inf"))
        return (metric, exec_time, memory_mb)

    def update(self, element: DataElement) -> bool:
        """Try to add an element. Returns True if added (non-dominated)."""
        if not element.success or element.metric is None:
            return False

        self.candidates = [
            c for c in self.candidates
            if not self._dominates(element, c.element)
        ]

        for c in self.candidates:
            if self._dominates(c.element, element):
                return False

        self.candidates.append(ParetoCandidate(element=element))

        if len(self.candidates) > self.max_size:
            self.candidates.sort(key=lambda c: -c.selection_count)
            self.candidates.pop(0)

        return True

    def would_add(self, element: DataElement) -> bool:
        """Return True if this element would be added to the front (is non-dominated)."""
        if not element.success or element.metric is None:
            return False
        for c in self.candidates:
            if self._dominates(c.element, element):
                return False
        return True

    def sample_via_uct(self, exploration_weight: float = 1.414) -> Optional[DataElement]:
        """Sample one candidate using UCT scoring."""
        samples = self.sample(n=1, exploration_weight=exploration_weight)
        return samples[0] if samples else None

    def sample(self, n: int = 1, exploration_weight: float = 1.414) -> list[DataElement]:
        """Sample n candidates using UCT scoring."""
        if not self.candidates:
            return []

        n = min(n, len(self.candidates))

        for c in self.candidates:
            if c.selection_count == 0:
                c.ucb_score = float("inf")
            else:
                metrics = sorted(set(
                    self._objective_fn(cand.element)[0]
                    for cand in self.candidates
                ))
                rank = metrics.index(self._objective_fn(c.element)[0])
                exploitation = 1.0 - (rank / max(len(metrics) - 1, 1))
                exploration = exploration_weight * math.sqrt(
                    math.log(self.total_selections + 1) / c.selection_count
                )
                c.ucb_score = exploitation + exploration

        ranked = sorted(self.candidates, key=lambda c: -c.ucb_score)
        selected = ranked[:n]

        for c in selected:
            c.selection_count += 1
        self.total_selections += n

        return [c.element for c in selected]

    def count_dominated_by(self, element: DataElement) -> int:
        """Count how many current front members are dominated by element."""
        count = 0
        for c in self.candidates:
            if self._dominates(element, c.element):
                count += 1
        return count

    def record_child_outcome(self, parent_index: int, success: bool):
        for c in self.candidates:
            if c.element.index == parent_index and success:
                c.successful_children += 1

    @property
    def best(self) -> Optional[DataElement]:
        if not self.candidates:
            return None
        return min(self.candidates, key=lambda c: self._objective_fn(c.element)[0]).element

    @property
    def size(self) -> int:
        return len(self.candidates)

    def get_elements(self) -> list[DataElement]:
        return [c.element for c in self.candidates]

    def get_feasible(
        self, min_flops: int, max_flops: int,
    ) -> list["ParetoCandidate"]:
        """Filter front to candidates within a FLOPs-equivalent range."""
        return [
            c for c in self.candidates
            if min_flops <= c.element.objectives.get("flops_equivalent_size", 0) <= max_flops
        ]

    def summary(self) -> str:
        if not self.candidates:
            return "Pareto front: empty"
        lines = [f"Pareto front: {self.size} candidates"]
        for c in sorted(self.candidates, key=lambda c: self._objective_fn(c.element)[0]):
            objs = self._objective_fn(c.element)
            lines.append(
                f"  [{c.element.index:04d}] {c.element.name} | "
                f"metric={objs[0]:.6f} | selected={c.selection_count}"
            )
        return "\n".join(lines)
