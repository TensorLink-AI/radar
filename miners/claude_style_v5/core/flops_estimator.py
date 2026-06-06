"""Estimate FLOPs for generated model code.

Primary method: ``torch.utils.flop_counter.FlopCounterMode`` — the same
analytical counter the validator uses in ``runner/timeseries_forecast/flops.py``.

Fallback 1: ``torch.jit.trace`` + FlopCounterMode (handles dynamic control flow).
Fallback 2: Forward-pass hooks (the previous primary method, kept as last resort).
Fallback 3: Static module-tree walk (approximate but safe).
"""

import math
import sys

import torch
import torch.nn as nn

from core.input_shape import infer_input

# Maximum parameter memory (bytes) we'll allow build_model to allocate.
# 512 MB covers even the largest bucket (125M FLOPs) with headroom.
_MAX_PARAM_BYTES = 512 * 1024 * 1024


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ── FlopCounterMode estimation (primary — matches validator) ──────


def _flopcounter_mode_flops(model: nn.Module, dummy: torch.Tensor,
                            out_shape: list | None = None) -> tuple[int, str]:
    """Estimate FLOPs using torch.utils.flop_counter.FlopCounterMode.

    This is the same approach the validator uses and is the source of truth
    for the size gate.  Returns (total_flops, error_message).

    If ``out_shape`` is a mutable list, the actual output tensor's shape is
    appended to it (as a tuple) so callers can perform additional coherence
    checks without re-running the model.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return 0, "FlopCounterMode not available in this torch version"

    try:
        model.eval()
        with torch.no_grad():
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                output = model(dummy)
            total = flop_counter.get_total_flops()
            if out_shape is not None and isinstance(output, torch.Tensor):
                out_shape.append(tuple(output.shape))
            return int(total), ""
    except Exception as exc:
        return 0, f"FlopCounterMode failed: {exc}"


# ── JIT trace + FlopCounterMode (fallback 1) ─────────────────────


def _jit_trace_flops(model: nn.Module, dummy: torch.Tensor,
                     out_shape: list | None = None) -> tuple[int, str]:
    """Trace the model with torch.jit.trace, then measure with FlopCounterMode."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return 0, "FlopCounterMode not available"

    try:
        model.eval()
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                output = traced(dummy)
            total = flop_counter.get_total_flops()
            if out_shape is not None and isinstance(output, torch.Tensor):
                out_shape.append(tuple(output.shape))
            return int(total), ""
    except Exception as exc:
        return 0, f"JIT trace + FlopCounterMode failed: {exc}"


# ── Forward-pass hook estimation (fallback 2 — previous primary) ──


def _forward_pass_flops(model: nn.Module, input_shape: list[int],
                        dtype: torch.dtype) -> tuple[int, str]:
    """Estimate FLOPs by running a forward pass with hooks on leaf modules.

    Kept as a fallback for torch versions or model structures where
    FlopCounterMode doesn't work.
    """
    flop_counts: list[int] = []
    hooks: list[torch.utils.hooks.RemovableHook] = []

    def _linear_hook(module, inp, out):
        x = inp[0]
        batch_elements = x.numel() // x.shape[-1]
        flop_counts.append(2 * module.in_features * module.out_features * batch_elements)

    def _conv1d_hook(module, inp, out):
        x = inp[0]
        batch_size = x.shape[0]
        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        out_len = out.shape[2]
        flop_counts.append(
            2 * module.in_channels * k * module.out_channels * out_len * batch_size // module.groups
        )

    def _conv2d_hook(module, inp, out):
        x = inp[0]
        batch_size = x.shape[0]
        kh, kw = (module.kernel_size if isinstance(module.kernel_size, tuple)
                   else (module.kernel_size, module.kernel_size))
        oh, ow = out.shape[2], out.shape[3]
        flop_counts.append(
            2 * module.in_channels * kh * kw * module.out_channels * oh * ow * batch_size // module.groups
        )

    def _mha_hook(module, inp, out):
        q = inp[0]
        if module.batch_first:
            batch_size, seq_len = q.shape[0], q.shape[1]
        else:
            seq_len, batch_size = q.shape[0], q.shape[1]
        d = module.embed_dim
        proj_flops = 4 * 2 * seq_len * d * d * batch_size
        attn_flops = 2 * seq_len * seq_len * d * batch_size
        flop_counts.append(proj_flops + attn_flops)

    def _norm_hook(module, inp, out):
        x = inp[0]
        flop_counts.append(2 * x.numel())

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(_conv1d_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv2d_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(_mha_hook))
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.InstanceNorm1d)):
            hooks.append(m.register_forward_hook(_norm_hook))

    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(*input_shape) if dtype == torch.float32 else torch.randint(0, 100, input_shape)
            model(dummy)
        return sum(flop_counts), ""
    except Exception as exc:
        return 0, f"Forward pass hooks failed: {exc}"
    finally:
        for h in hooks:
            h.remove()


# ── Static module-tree walk (fallback 3) ─────────────────────────


def _static_linear_flops(module: nn.Linear, seq_len: int) -> int:
    return 2 * module.in_features * module.out_features * seq_len


def _static_conv1d_flops(module: nn.Conv1d, seq_len: int) -> int:
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
    out_len = (seq_len + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    return 2 * module.in_channels * k * module.out_channels * max(out_len, 1) // module.groups


def _static_conv2d_flops(module: nn.Conv2d, h: int, w: int) -> int:
    ph, pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
    dh, dw = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation)
    sh, sw = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
    kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return 2 * module.in_channels * kh * kw * module.out_channels * max(oh, 1) * max(ow, 1) // module.groups


def _static_mha_flops(module: nn.MultiheadAttention, seq_len: int) -> int:
    d = module.embed_dim
    proj_flops = 4 * 2 * seq_len * d * d
    attn_flops = 2 * seq_len * seq_len * d
    return proj_flops + attn_flops


def _walk_flops(model: nn.Module, seq_len: int) -> int:
    """Walk the module tree and sum FLOPs from leaf layers (last-resort fallback)."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += _static_linear_flops(m, seq_len)
        elif isinstance(m, nn.Conv1d):
            total += _static_conv1d_flops(m, seq_len)
        elif isinstance(m, nn.Conv2d):
            side = max(int(seq_len ** 0.5), 1)
            total += _static_conv2d_flops(m, side, side)
        elif isinstance(m, nn.MultiheadAttention):
            total += _static_mha_flops(m, seq_len)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.InstanceNorm1d)):
            norm_features = getattr(m, 'normalized_shape', None)
            if isinstance(norm_features, (list, tuple)):
                elem = 1
                for s in norm_features:
                    elem *= s
            else:
                elem = getattr(m, 'num_features', seq_len)
            total += 2 * elem * seq_len
    return total


def _count_param_bytes(model: nn.Module) -> int:
    """Total bytes used by model parameters."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


# ── Resize suggestion helper ─────────────────────────────────────


def suggest_resize(estimated: int, gate_min: int, gate_max: int,
                   target: int) -> str:
    """Produce an actionable resize suggestion when FLOPs are off-gate."""
    if estimated <= 0:
        return ""
    if estimated < gate_min:
        factor = math.sqrt(gate_min / estimated)
        return (
            f"SUGGESTION: Your model has ~{estimated:,} FLOPs but needs >= {gate_min:,}. "
            f"Try increasing hidden_dim by factor ~{factor:.2f} "
            f"(e.g. multiply hidden_dim by {factor:.1f}), or add more layers."
        )
    elif estimated > gate_max:
        factor = math.sqrt(estimated / gate_max)
        return (
            f"SUGGESTION: Your model has ~{estimated:,} FLOPs but needs <= {gate_max:,}. "
            f"Try reducing hidden_dim by factor ~{1/factor:.3f} "
            f"(e.g. divide hidden_dim by {factor:.1f}), or remove layers."
        )
    return ""


# ── Public API ───────────────────────────────────────────────────


def estimate_flops(code: str, challenge: dict,
                   out_shape_sink: list | None = None) -> tuple[int | None, str]:
    """Execute build_model, estimate FLOPs using FlopCounterMode (matching validator).

    Falls back through: JIT trace, forward-pass hooks, static walk.

    Returns (estimated_flops, error_message).
    On success error_message is empty. On failure estimated_flops is None.

    If ``out_shape_sink`` is a mutable list, the actual forward-pass output
    shape is appended to it as a tuple of ints — enabling callers like
    ``validation`` to run an output-shape coherence check without paying for
    a second forward pass.
    """
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    constraints = task.get("constraints", [])

    # Execute the code in a restricted namespace
    namespace = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
    except Exception as exc:
        return None, f"Code execution failed: {exc}"

    build_model_fn = namespace.get("build_model")
    if not callable(build_model_fn):
        return None, "build_model not found or not callable"

    # Instantiate the model using task_params from the challenge
    try:
        model = build_model_fn(**tp)
    except Exception as exc:
        return None, f"build_model() raised: {exc}"

    if not isinstance(model, nn.Module):
        return None, "build_model() did not return an nn.Module"

    # Safety check: reject models that are absurdly large
    param_bytes = _count_param_bytes(model)
    if param_bytes > _MAX_PARAM_BYTES:
        del model
        return None, (
            f"Model parameters use {param_bytes / (1024**2):.0f} MB, "
            f"exceeding the {_MAX_PARAM_BYTES // (1024**2)} MB safety limit"
        )

    # Infer input shape and dtype generically from task_params
    input_shape, input_dtype = infer_input(tp, constraints)

    # Build dummy input tensor
    try:
        if input_dtype == torch.long:
            vocab = tp.get("vocab_size", tp.get("n_vocab", 1000))
            dummy = torch.randint(0, int(vocab), input_shape)
        else:
            dummy = torch.randn(*input_shape)
    except Exception as exc:
        del model
        return None, f"Failed to create dummy input: {exc}"

    # ── Primary: FlopCounterMode (same as validator) ─────────────
    total_flops, err = _flopcounter_mode_flops(model, dummy, out_shape_sink)
    if not err and total_flops > 0:
        _log(f"[flops] FlopCounterMode: {total_flops:,}")
        del model
        return total_flops, ""

    if err:
        _log(f"[flops] FlopCounterMode failed, trying fallbacks: {err}")

    # ── Fallback 1: JIT trace + FlopCounterMode ──────────────────
    total_flops, err2 = _jit_trace_flops(model, dummy, out_shape_sink)
    if not err2 and total_flops > 0:
        _log(f"[flops] JIT+FlopCounterMode: {total_flops:,}")
        del model
        return total_flops, ""

    if err2:
        _log(f"[flops] JIT fallback also failed: {err2}")

    # ── Fallback 2: Forward-pass hooks ───────────────────────────
    total_flops, err3 = _forward_pass_flops(model, input_shape, input_dtype)
    if not err3 and total_flops >= 0:
        _log(f"[flops] Hook-based fallback: {total_flops:,} "
             "(WARNING: may diverge from validator's FlopCounterMode)")
        del model
        return total_flops, ""

    # ── Fallback 3: Static module-tree walk ──────────────────────
    int_vals = [v for v in tp.values() if isinstance(v, int)]
    seq_len = int_vals[1] if len(int_vals) > 1 else max(int_vals[0], 1) if int_vals else 1
    total_flops = _walk_flops(model, seq_len)
    _log(f"[flops] Static walk fallback: {total_flops:,} (approximate)")

    del model
    return total_flops, ""
