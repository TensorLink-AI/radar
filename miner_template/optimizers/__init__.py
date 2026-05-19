"""Prompt-optimizer plugin registry for miners.

A plugin is any callable with the signature::

    optimize(results: list[ResultRow],
             population: list[Prompt],
             config: dict) -> list[Prompt]

Built-ins:
  ``random_mutate`` — deterministic baseline, no LLM required (CI-safe)
  ``gepa``          — DSPy GEPA adapter (Genetic-Pareto reflective evolution)

Custom optimizers can be referenced as ``"package.module:func"`` —
e.g. ``my_research.optimizers:my_optimizer`` — and ``resolve()`` will
import them on demand.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from miner_template.prompts import Prompt

# ── Public types ─────────────────────────────────────────────────────


@dataclass(slots=True)
class ResultRow:
    """One scored submission, as returned by ``/miners/me/results``."""

    round_id: int
    submission_id: str
    task_name: str
    prompt_id: Optional[str]
    architecture_code: str
    scores: dict = field(default_factory=dict)
    created_at: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ResultRow":
        return cls(
            round_id=int(d.get("round_id", 0) or 0),
            submission_id=str(d.get("submission_id", "") or ""),
            task_name=str(d.get("task_name", "") or ""),
            prompt_id=d.get("prompt_id"),
            architecture_code=str(d.get("architecture_code", "") or ""),
            scores=dict(d.get("scores") or {}),
            created_at=str(d.get("created_at", "") or ""),
        )


Optimizer = Callable[[list[ResultRow], list[Prompt], dict], list[Prompt]]


# ── Resolver ─────────────────────────────────────────────────────────


_BUILTINS = {
    "random_mutate": "miner_template.optimizers.random_mutate:optimize",
    "gepa": "miner_template.optimizers.gepa:optimize",
}


def resolve(name: str) -> Optimizer:
    """Return the optimizer callable for ``name``.

    Accepts a built-in alias (``"gepa"``) or a fully qualified
    ``"package.module:func"`` path.
    """
    target = _BUILTINS.get(name, name)
    if ":" not in target:
        raise ValueError(
            f"unknown optimizer {name!r}; expected one of {sorted(_BUILTINS)} "
            f"or 'package.module:func'"
        )
    module_path, func_name = target.split(":", 1)
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"optimizer {name!r}: cannot import {module_path!r}: {e}"
        ) from e
    try:
        func = getattr(mod, func_name)
    except AttributeError:
        raise AttributeError(
            f"optimizer {name!r}: {module_path}:{func_name} not found"
        )
    if not callable(func):
        raise TypeError(f"optimizer {name!r}: {target} is not callable")
    return func


def list_builtins() -> list[str]:
    return sorted(_BUILTINS)
