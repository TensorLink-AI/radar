"""Pipeline-task helpers for v6 — exec-and-probe a miner's
``build_pipeline`` to give the LLM fast local feedback before submitting.

The validator-side trainer in ``local/data_pipeline.py`` performs the
shape check at round time; this module is the in-agent preview so the
designer gets a smoke result in <1s without burning a round.

v6 extends the v4 shape-only probe with cheap *distribution-quality*
statistics over the pulled batches (per-channel variance, constant
fraction, lag-1 autocorrelation, value range, cross-batch variety) plus
degeneracy warnings — see ``core.pipeline_quality``. A collapsed or
zero-variance generator trains fine and then fails the GIFT gate;
surfacing it here lets the designer fix it in-round.
"""

from __future__ import annotations

import time
from typing import Any

from core.pipeline_quality import quality_stats


# v6: pull a few batches (was 2) so cross-batch variety is measurable —
# an iterator that yields the same batch every step is a common degenerate
# mode that 2 draws can miss.
MAX_BATCHES = 4
SMOKE_TIMEOUT_SECONDS = 8.0


def _as_tensor_info(x) -> dict[str, Any]:
    """Describe a tensor-like object without serialising values."""
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.is_tensor(x):
        return {
            "shape": list(x.shape), "dtype": str(x.dtype),
            "kind": "torch.Tensor",
        }
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return {
                "shape": list(x.shape), "dtype": str(x.dtype),
                "kind": "numpy.ndarray",
            }
    except ImportError:
        pass
    try:
        shape = getattr(x, "shape", None)
        return {
            "shape": list(shape) if shape is not None else None,
            "dtype": str(getattr(x, "dtype", type(x).__name__)),
            "kind": type(x).__name__,
        }
    except Exception:  # noqa: BLE001
        return {"kind": type(x).__name__}


def _unwrap_batch(batch) -> tuple[Any, Any]:
    if isinstance(batch, dict):
        inp = batch.get("input", batch.get("context"))
        tgt = batch.get("target")
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        inp, tgt = batch
    else:
        raise TypeError(
            f"batch must be dict {{input|context, target}} or "
            f"(inputs, targets) tuple — got {type(batch).__name__}",
        )
    if inp is None or tgt is None:
        raise TypeError("batch missing 'input' or 'target'")
    return inp, tgt


def smoke_test_pipeline(code: str, challenge: dict) -> dict[str, Any]:
    """Exec ``code``, call ``build_pipeline(...)`` with the task constants,
    pull up to ``MAX_BATCHES`` batches, verify shapes, and (on success)
    compute distribution-quality stats + degeneracy warnings.

    Never raises — every failure mode becomes ``{"ok": False, "error": ...}``
    so the LLM-facing tool wrapper can render it as a result string.
    """
    started = time.perf_counter()
    task = (challenge or {}).get("task") or {}
    tp = task.get("task_params") or {}
    if not tp:
        return {"ok": False, "error": "challenge.task.task_params missing"}

    ns: dict[str, Any] = {"__name__": "data_pipeline_submission"}
    try:
        exec(code, ns)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"exec failed: {type(exc).__name__}: {exc}"}

    build_pipeline = ns.get("build_pipeline")
    if not callable(build_pipeline):
        return {"ok": False, "error": "submission lacks build_pipeline(...)"}

    try:
        iterator = build_pipeline(**tp)
    except TypeError as exc:
        # Most likely signature mismatch against task_params keys.
        return {
            "ok": False,
            "error": (
                f"build_pipeline rejected task_params {list(tp.keys())}: "
                f"{exc}"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"build_pipeline raised: {type(exc).__name__}: {exc}",
        }
    if iterator is None:
        return {"ok": False, "error": "build_pipeline returned None"}

    try:
        it = iter(iterator)
    except TypeError as exc:
        return {"ok": False, "error": f"build_pipeline result not iterable: {exc}"}

    shapes: list[dict] = []
    raw_inputs: list = []
    raw_targets: list = []
    for i in range(MAX_BATCHES):
        if time.perf_counter() - started > SMOKE_TIMEOUT_SECONDS:
            return {
                "ok": False,
                "error": (
                    f"smoke test timed out after {SMOKE_TIMEOUT_SECONDS:.0f}s "
                    f"with {i} batch(es) produced"
                ),
            }
        try:
            batch = next(it)
        except StopIteration:
            return {
                "ok": False,
                "error": (
                    f"iterator exhausted after {i} batch(es) — "
                    "build_pipeline must yield indefinitely"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": f"batch {i} raised: {type(exc).__name__}: {exc}",
            }
        try:
            inp, tgt = _unwrap_batch(batch)
        except TypeError as exc:
            return {"ok": False, "error": f"batch {i}: {exc}"}
        ii = _as_tensor_info(inp)
        ti = _as_tensor_info(tgt)
        shapes.append({"batch_index": i, "input": ii, "target": ti})
        # Light coherence checks (batch dim agreement, rank-3 shape).
        in_shape = ii.get("shape") or []
        tg_shape = ti.get("shape") or []
        if len(in_shape) != 3:
            return {
                "ok": False,
                "error": (
                    f"batch {i} input rank {len(in_shape)}, expected 3 "
                    f"(batch, context_len, num_variates); got shape {in_shape}"
                ),
                "shapes": shapes,
            }
        if len(tg_shape) != 3:
            return {
                "ok": False,
                "error": (
                    f"batch {i} target rank {len(tg_shape)}, expected 3 "
                    f"(batch, prediction_len, num_variates); got shape {tg_shape}"
                ),
                "shapes": shapes,
            }
        if in_shape[0] != tg_shape[0]:
            return {
                "ok": False,
                "error": (
                    f"batch {i} dim mismatch: input batch={in_shape[0]} "
                    f"vs target batch={tg_shape[0]}"
                ),
                "shapes": shapes,
            }
        raw_inputs.append(inp)
        raw_targets.append(tgt)

    stats, warnings = quality_stats(raw_inputs, raw_targets)

    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "ok": True,
        "n_batches": len(shapes),
        "shapes": shapes,
        "elapsed_ms": round(elapsed * 1000.0, 1),
    }
    if stats is not None:
        result["quality"] = stats
    if warnings:
        result["quality_warnings"] = warnings
    return result
