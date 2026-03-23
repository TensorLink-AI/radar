"""
Generates the analysis string stored in the DB after each experiment.

Pure function, no LLM. Template-based analysis comparing result to parent.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from shared.database import DataElement

if TYPE_CHECKING:
    from shared.task import TaskSpec


def analyze(result: dict, parent: DataElement, task: Optional["TaskSpec"] = None) -> str:
    """
    Returns a 2-3 sentence analysis comparing result to parent.
    Covers: metric change, throughput, convergence status.

    Args:
        result: execution result dict
        parent: parent DataElement
        task: TaskSpec for metric direction (defaults to lower-is-better)
    """
    if not result.get("success", False):
        reason = _failure_reason(result)
        return f"Experiment failed. {reason}"

    lower_is_better = task.primary_lower_is_better if task else True
    parts = []

    # Metric comparison
    result_metric = result.get("metric")
    if result_metric is not None and parent.metric is not None:
        if lower_is_better:
            delta = parent.metric - result_metric
        else:
            delta = result_metric - parent.metric
        pct = (delta / max(abs(parent.metric), 1e-8)) * 100
        if abs(pct) < 0.1:
            parts.append(f"No significant change from parent (metric {result_metric:.4f} vs {parent.metric:.4f}).")
        elif pct > 0:
            parts.append(f"Improved {pct:.1f}% over parent ({result_metric:.4f} vs {parent.metric:.4f}).")
        else:
            parts.append(f"Regressed {abs(pct):.1f}% from parent ({result_metric:.4f} vs {parent.metric:.4f}).")
    elif result_metric is not None:
        parts.append(f"Metric: {result_metric:.4f} (no parent metric for comparison).")

    # Throughput comparison (only if objectives include relevant data)
    result_time = result.get("exec_time", 0)
    parent_time = parent.objectives.get("exec_time", 0)
    if result_time > 0 and parent_time > 0:
        # Check for any step-like objective
        result_steps = result.get("objectives", {}).get("num_steps", 0)
        parent_steps = parent.objectives.get("num_steps", 0)
        if result_steps and parent_steps:
            result_throughput = result_steps / result_time
            parent_throughput = parent_steps / parent_time
            if result_throughput > parent_throughput * 1.05:
                parts.append(f"Throughput improved ({result_throughput:.1f} vs {parent_throughput:.1f} steps/s).")
            elif result_throughput < parent_throughput * 0.95:
                parts.append(f"Throughput decreased ({result_throughput:.1f} vs {parent_throughput:.1f} steps/s).")

    # Memory comparison (only if memory_mb is in objectives)
    result_mem = result.get("objectives", {}).get("memory_mb", 0)
    parent_mem = parent.objectives.get("memory_mb", 0)
    if result_mem and parent_mem and abs(result_mem - parent_mem) > 100:
        if result_mem < parent_mem:
            parts.append(f"Memory reduced ({result_mem:.0f} vs {parent_mem:.0f} MB).")
        else:
            parts.append(f"Memory increased ({result_mem:.0f} vs {parent_mem:.0f} MB).")

    # Convergence check (only if loss_curve data exists)
    loss_curve = result.get("loss_curve", [])
    if len(loss_curve) >= 3:
        tail = loss_curve[-3:]
        if all(tail[i] <= tail[i - 1] for i in range(1, len(tail))):
            parts.append("Metric still improving at end — longer budget would help.")
        elif all(tail[i] >= tail[i - 1] for i in range(1, len(tail))):
            parts.append("Metric worsening at end — possible divergence.")

    if not parts:
        parts.append(f"Experiment completed successfully with metric {result_metric}.")

    return " ".join(parts)


def _failure_reason(result: dict) -> str:
    trace = result.get("trace", "")
    rc = result.get("return_code", -1)

    if rc == -9:
        return "Timed out before completing."

    if "OOM" in trace or "OutOfMemory" in trace or "CUDA out of memory" in trace:
        return "Out of memory (OOM)."

    if "SyntaxError" in trace:
        return "Syntax error in code."

    if "ImportError" in trace or "ModuleNotFoundError" in trace:
        return "Missing dependency or import error."

    if "RuntimeError" in trace:
        return "Runtime error during execution."

    return f"Crashed with return code {rc}."
