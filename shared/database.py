"""
Experiment database with evolutionary lineage tracking.

JSON-file backed. Each experiment is a DataElement storing code, metrics,
analysis, motivation, parent index, and more.
"""

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class DataElement:
    """One experiment in the database."""

    index: int = -1
    timestamp: float = 0.0
    name: str = ""
    code: str = ""
    motivation: str = ""
    trace: str = ""
    metric: Optional[float] = None
    success: bool = False
    analysis: str = ""
    parent: Optional[int] = None
    generation: int = 0
    objectives: dict = field(default_factory=dict)
    score: float = 0.0
    miner_uid: int = -1
    miner_hotkey: str = ""
    loss_curve: list[float] = field(default_factory=list)
    manifest_sha256: str = ""
    generated_samples: list = field(default_factory=list)
    task: str = ""
    round_id: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataElement":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_api_dict(self) -> dict:
        """JSON shape exposed to miners via the DB API."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "name": self.name,
            "task": self.task,
            "miner_uid": self.miner_uid,
            "miner_hotkey": self.miner_hotkey,
            "generation": self.generation,
            "parent_index": self.parent,
            "code": self.code,
            "motivation": self.motivation,
            "results": {
                "success": self.success,
                "metric": self.metric,
                **{k: v for k, v in self.objectives.items()},
                "loss_curve": self.loss_curve,
            },
            "analysis": self.analysis,
            "score": self.score,
            "round_id": self.round_id,
        }

    def summary(self) -> str:
        status = "OK" if self.success else "FAIL"
        metric_str = f"{self.metric:.6f}" if self.metric is not None else "N/A"
        return (
            f"[{self.index:04d}] {self.name} | {status} | "
            f"metric={metric_str} | gen={self.generation} | parent={self.parent}"
        )


class ExperimentDB:
    """JSON-based experiment database with evolutionary lineage.

    .. deprecated::
        Use :class:`shared.sqlite_store.SQLiteExperimentStore` instead.
        This class is kept for backward compatibility with existing tests.
    """

    def __init__(self, db_dir: str = "./experiments"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_file = self.db_dir / "experiments.json"
        self.elements: list[DataElement] = []
        self._load()

    def _load(self):
        if self.db_file.exists():
            with open(self.db_file, "r") as f:
                data = json.load(f)
            self.elements = [DataElement.from_dict(d) for d in data]

    def _save(self):
        with open(self.db_file, "w") as f:
            json.dump([e.to_dict() for e in self.elements], f, indent=2)

    @property
    def size(self) -> int:
        return len(self.elements)

    def add(self, element: DataElement) -> int:
        element.index = self.size
        element.timestamp = time.time()
        self.elements.append(element)
        self._save()
        return element.index

    def add_batch(self, elements: list[DataElement]) -> list[int]:
        """Add multiple experiments with a single save (avoids N full rewrites)."""
        indices = []
        now = time.time()
        for elem in elements:
            elem.index = self.size
            elem.timestamp = now
            self.elements.append(elem)
            indices.append(elem.index)
        if indices:
            self._save()
        return indices

    def get(self, index: int) -> Optional[DataElement]:
        if 0 <= index < self.size:
            return self.elements[index]
        return None

    def get_successful(self) -> list[DataElement]:
        successful = [e for e in self.elements if e.success and e.metric is not None]
        return sorted(successful, key=lambda e: e.metric)

    def get_best(self, n: int = 1) -> list[DataElement]:
        return self.get_successful()[:n]

    def get_recent(self, n: int = 5) -> list[DataElement]:
        return list(reversed(self.elements[-n:]))

    def get_failures(self, n: int = 10) -> list[DataElement]:
        failed = [e for e in self.elements if not e.success]
        return list(reversed(failed[-n:]))

    def get_children(self, parent_index: int) -> list[DataElement]:
        return [e for e in self.elements if e.parent == parent_index]

    def get_lineage(self, index: int) -> list[DataElement]:
        lineage = []
        current = self.get(index)
        while current is not None:
            lineage.append(current)
            current = self.get(current.parent) if current.parent is not None else None
        return list(reversed(lineage))

    def search(self, query: str, top_k: int = 10) -> list[DataElement]:
        """Keyword search across motivation and analysis fields."""
        query_words = set(query.lower().split())
        scored = []
        for elem in self.elements:
            text = (elem.motivation + " " + elem.analysis).lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((overlap, elem))
        scored.sort(key=lambda x: -x[0])
        return [elem for _, elem in scored[:top_k]]

    def search_failures(self, query: str, top_k: int = 10) -> list[DataElement]:
        """Keyword search filtered to failed experiments."""
        query_words = set(query.lower().split())
        scored = []
        for elem in self.elements:
            if elem.success:
                continue
            text = (elem.motivation + " " + elem.analysis).lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((overlap, elem))
        scored.sort(key=lambda x: -x[0])
        return [elem for _, elem in scored[:top_k]]

    def get_component_stats(self, patterns: Optional[dict[str, str]] = None) -> dict:
        """
        Frequency of code patterns in successful experiments.

        Args:
            patterns: dict mapping category name to regex pattern.
                      If None, returns empty stats (caller should provide
                      task-specific patterns).
        """
        if not patterns:
            return {}
        stats: dict[str, Counter] = {k: Counter() for k in patterns}
        for elem in self.elements:
            if not elem.success:
                continue
            for category, pattern in patterns.items():
                for match in re.findall(pattern, elem.code, re.IGNORECASE):
                    stats[category][match] += 1
        return {k: dict(v.most_common(10)) for k, v in stats.items()}

    def get_pareto_elements(self) -> list[DataElement]:
        """Return all successful elements (for external Pareto computation)."""
        return [e for e in self.elements if e.success and e.metric is not None]

    def count_in_flops_range(self, min_flops: int, max_flops: int) -> int:
        """Count successful experiments within a FLOPs-equivalent range."""
        count = 0
        for e in self.elements:
            if not e.success or e.metric is None:
                continue
            flops = e.objectives.get("flops_equivalent_size", 0)
            if min_flops <= flops <= max_flops:
                count += 1
        return count

    def get_in_flops_range(
        self, min_flops: int, max_flops: int,
    ) -> list[DataElement]:
        """Return successful experiments within a FLOPs-equivalent range."""
        results = []
        for e in self.elements:
            if not e.success or e.metric is None:
                continue
            flops = e.objectives.get("flops_equivalent_size", 0)
            if min_flops <= flops <= max_flops:
                results.append(e)
        return results

    def stats(self) -> dict:
        successful = [e for e in self.elements if e.success]
        metrics = [e.metric for e in successful if e.metric is not None]
        return {
            "total": self.size,
            "successful": len(successful),
            "failed": self.size - len(successful),
            "best_metric": min(metrics) if metrics else None,
            "worst_metric": max(metrics) if metrics else None,
            "mean_metric": sum(metrics) / len(metrics) if metrics else None,
            "max_generation": max((e.generation for e in self.elements), default=0),
        }
