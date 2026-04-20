"""Generic eval-result helpers driven by TaskSpec.objectives.

Lets error paths in validator/evaluator.py produce result dicts whose
metric keys come from the task's Objective list (name + default value),
instead of hardcoding CRPS/MASE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.task import TaskSpec


def build_error_result(task: "TaskSpec | None", error: str) -> dict:
    """Build a failure-path eval result dict for the given task.

    Seeds `flops_equivalent_size=0` and every declared objective at its
    `default` value, then attaches the error message. If `task` is None,
    falls back to the ts_forecasting legacy shape so direct unit-test
    callers of `evaluate_checkpoint` keep working.
    """
    if task is None:
        return {
            "crps": float("inf"),
            "mase": float("inf"),
            "flops_equivalent_size": 0,
            "error": error,
        }
    out: dict = {"flops_equivalent_size": 0, "error": error}
    for obj in task.objectives:
        out[obj.name] = obj.default
    return out
