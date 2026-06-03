"""Memory-efficient forward-pass tracer.

Purpose: give the LLM (and tests) a compact view of what actually happens
inside a model during a forward pass — op by op, with input/output shapes
and param counts — so it can debug shape mismatches without holding any
tensor data in memory.

Memory discipline:
  * No tensors are retained. Hooks record only ``tuple[int, ...]`` shapes,
    class-name strings, and integer param counts.
  * Leaf modules only — ``nn.Sequential`` / ``nn.ModuleList`` containers
    are skipped so we don't double-count or bloat the trace.
  * Entries are capped (``max_entries``) so pathological architectures
    can't explode the returned list.
  * Forward pass runs under ``torch.no_grad()`` on a single dummy sample
    (batch=1) inferred from ``task_params``.
"""

import sys

import torch
import torch.nn as nn

from core.input_shape import infer_input


# Same ceiling used by flops_estimator — 512 MB of params. Anything bigger
# would OOM a normal machine during tracing anyway.
_MAX_PARAM_BYTES = 512 * 1024 * 1024

# Container modules we skip so the trace only reflects leaf ops that
# actually touch tensors.
_CONTAINER_TYPES = (nn.Sequential, nn.ModuleList, nn.ModuleDict)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _shape_of(x) -> tuple | None:
    """Extract a shape tuple from a hook input/output without retaining it.

    Handles the common cases: a single tensor, a tuple of tensors (returns
    the first), or ``None``.
    """
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)) and x:
        first = x[0]
        if isinstance(first, torch.Tensor):
            return tuple(first.shape)
    return None


def _count_module_params(module: nn.Module) -> int:
    """Count parameters that belong directly to ``module`` (not its children).

    Using ``parameters(recurse=False)`` keeps the accounting per-op rather
    than double-counting nested modules.
    """
    return sum(p.numel() for p in module.parameters(recurse=False))


def _count_param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def trace_architecture(code: str, challenge: dict,
                       max_entries: int = 200) -> tuple[list[dict], str]:
    """Build the model and trace its forward pass op-by-op.

    Returns ``(entries, error)``. On failure ``entries`` is empty and
    ``error`` is a human-readable string. Each entry is a dict:

        {"idx": int, "name": str, "op": str,
         "input_shape": tuple, "output_shape": tuple, "params": int}
    """
    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    constraints = task.get("constraints", []) or []

    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
    except Exception as exc:
        return [], f"Code execution failed: {exc}"

    build_model_fn = namespace.get("build_model")
    if not callable(build_model_fn):
        return [], "build_model not found or not callable"

    try:
        model = build_model_fn(**tp)
    except Exception as exc:
        return [], f"build_model() raised: {exc}"

    if not isinstance(model, nn.Module):
        return [], "build_model() did not return an nn.Module"

    param_bytes = _count_param_bytes(model)
    if param_bytes > _MAX_PARAM_BYTES:
        del model
        return [], (
            f"Model parameters use {param_bytes / (1024**2):.0f} MB, "
            f"exceeding the {_MAX_PARAM_BYTES // (1024**2)} MB safety limit"
        )

    input_shape, input_dtype = infer_input(tp, constraints)
    try:
        if input_dtype == torch.long:
            vocab = tp.get("vocab_size", tp.get("n_vocab", 1000))
            dummy = torch.randint(0, int(vocab), input_shape)
        else:
            dummy = torch.randn(*input_shape)
    except Exception as exc:
        del model
        return [], f"Failed to create dummy input: {exc}"

    entries: list[dict] = []
    hooks: list = []
    overflow = {"hit": False}

    def _make_hook(qualified_name: str, module: nn.Module):
        def _hook(mod, inp, out):
            # Early-out once we've hit the cap. We still let the forward
            # pass complete — cheaper than detaching hooks mid-flight.
            if len(entries) >= max_entries:
                overflow["hit"] = True
                return
            entries.append({
                "idx": len(entries),
                "name": qualified_name or "<root>",
                "op": type(mod).__name__,
                "input_shape": _shape_of(inp),
                "output_shape": _shape_of(out),
                "params": _count_module_params(mod),
            })
        return _hook

    # Attach hooks to leaf modules only.
    for name, module in model.named_modules():
        if module is model:
            continue  # skip the top-level container (its trace = final output)
        if isinstance(module, _CONTAINER_TYPES):
            continue
        # A module is "leaf" if it has no submodules we'd already hook.
        has_hookable_children = any(
            not isinstance(child, _CONTAINER_TYPES)
            for child in module.children()
        )
        if has_hookable_children:
            continue
        hooks.append(module.register_forward_hook(_make_hook(name, module)))

    try:
        model.eval()
        with torch.no_grad():
            model(dummy)
    except Exception as exc:
        for h in hooks:
            h.remove()
        del model
        return [], f"Forward pass failed during trace: {exc}"
    finally:
        for h in hooks:
            h.remove()

    if overflow["hit"]:
        _log(
            f"[trace] Hit max_entries={max_entries}; truncating. "
            "Increase max_entries if you need more detail."
        )

    del model
    return entries, ""


def format_trace(entries: list[dict], max_rows: int | None = None) -> str:
    """Render a compact table of trace entries for LLM / stderr consumption."""
    if not entries:
        return "(no trace entries recorded)"

    rows = entries if max_rows is None else entries[:max_rows]

    # Compute column widths dynamically so the table stays readable for
    # any task's shapes.
    name_w = max(4, max(len(str(e["name"])) for e in rows))
    op_w = max(2, max(len(str(e["op"])) for e in rows))
    in_w = max(5, max(len(str(e["input_shape"])) for e in rows))
    out_w = max(6, max(len(str(e["output_shape"])) for e in rows))

    header = (
        f"{'Idx':>4}  {'Name':<{name_w}}  {'Op':<{op_w}}  "
        f"{'Input':<{in_w}}  {'Output':<{out_w}}  {'Params':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    total_params = 0
    for e in rows:
        total_params += int(e.get("params") or 0)
        lines.append(
            f"{e['idx']:>4}  {str(e['name']):<{name_w}}  "
            f"{str(e['op']):<{op_w}}  {str(e['input_shape']):<{in_w}}  "
            f"{str(e['output_shape']):<{out_w}}  {e['params']:>10,}"
        )
    if max_rows is not None and len(entries) > max_rows:
        lines.append(f"... ({len(entries) - max_rows} more entries omitted)")
    # Sum params across ALL entries, not just displayed rows.
    full_total = sum(int(e.get("params") or 0) for e in entries)
    lines.append(sep)
    lines.append(f"Total leaf params: {full_total:,}  "
                 f"(ops traced: {len(entries)})")
    return "\n".join(lines)
