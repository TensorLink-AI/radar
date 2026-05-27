"""Estimate FLOPs for generated model code.

Primary method: forward-pass hooks that observe actual tensor shapes.
Fallback: static module-tree walk (approximate but safe).
"""

import torch
import torch.nn as nn

# Maximum parameter memory (bytes) we'll allow build_model to allocate.
# 512 MB covers even the largest bucket (125M FLOPs) with headroom.
_MAX_PARAM_BYTES = 512 * 1024 * 1024


# ── Static helpers (fallback path) ──────────────────────────────


def _static_linear_flops(module: nn.Linear, seq_len: int) -> int:
    """FLOPs for a Linear layer: 2 * in * out per sequence element."""
    return 2 * module.in_features * module.out_features * seq_len


def _static_conv1d_flops(module: nn.Conv1d, seq_len: int) -> int:
    """FLOPs for Conv1d: 2 * C_in * K * C_out * L_out / groups."""
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
    out_len = (seq_len + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    return 2 * module.in_channels * k * module.out_channels * max(out_len, 1) // module.groups


def _static_conv2d_flops(module: nn.Conv2d, h: int, w: int) -> int:
    """FLOPs for Conv2d with spatial dims h x w."""
    ph, pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
    dh, dw = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation)
    sh, sw = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
    kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return 2 * module.in_channels * kh * kw * module.out_channels * max(oh, 1) * max(ow, 1) // module.groups


def _static_mha_flops(module: nn.MultiheadAttention, seq_len: int) -> int:
    """FLOPs for MultiheadAttention: QKV projections + attention + output proj."""
    d = module.embed_dim
    # 3 projections (Q, K, V) + output projection: 4 * 2 * seq * d * d
    proj_flops = 4 * 2 * seq_len * d * d
    # Attention scores: 2 * seq^2 * d  (QK^T + softmax·V)
    attn_flops = 2 * seq_len * seq_len * d
    return proj_flops + attn_flops


def _count_param_bytes(model: nn.Module) -> int:
    """Total bytes used by model parameters."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


def _walk_flops(model: nn.Module, seq_len: int) -> int:
    """Walk the module tree and sum FLOPs from leaf layers (fallback).

    Uses seq_len as the token/sequence dimension.  This is approximate —
    it cannot account for internal reshapes that change the effective
    sequence length seen by inner layers.
    """
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


# ── Input shape inference ──────────────────────────────────────


def _infer_input_shape(tp: dict) -> list[int] | None:
    """Try to determine model input shape from task_params."""
    # Known task patterns
    if "context_len" in tp and "num_variates" in tp:
        return [1, int(tp["context_len"]), int(tp["num_variates"])]
    # Generic fallback: use first two integer params
    int_vals = [(k, v) for k, v in tp.items() if isinstance(v, int)]
    if len(int_vals) >= 2:
        return [1, int_vals[0][1], int_vals[1][1]]
    elif len(int_vals) == 1:
        return [1, int_vals[0][1]]
    return None


# ── Forward-pass hook estimation (primary path) ─────────────────


def _forward_pass_flops(model: nn.Module, input_shape: list[int]) -> tuple[int, str]:
    """Estimate FLOPs by running a forward pass with hooks.

    Registers hooks on compute-heavy leaf modules, runs the model with
    the given *input_shape* (including batch dim), and sums the FLOPs
    each hook records from the *actual* input tensor shape it receives.
    This correctly handles internal reshapes (e.g. channel-independent
    models that fold variates into the batch dimension).

    Returns (total_flops, error_message).  error_message is empty on
    success.
    """
    flop_counts: list[int] = []
    hooks: list[torch.utils.hooks.RemovableHook] = []

    # -- hook closures ---------------------------------------------------

    def _linear_hook(module, inp, out):
        x = inp[0]
        # all dims except the feature (last) dim
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

    # -- register hooks --------------------------------------------------

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

    # -- run forward pass ------------------------------------------------

    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(*input_shape)
            model(dummy)
        return sum(flop_counts), ""
    except Exception as exc:
        return 0, f"Forward pass failed: {exc}"
    finally:
        for h in hooks:
            h.remove()


# ── Public API ───────────────────────────────────────────────────


def estimate_flops(code: str, challenge: dict) -> tuple[int | None, str]:
    """Execute build_model, estimate FLOPs via forward-pass hooks.

    Falls back to static module-tree walk if the forward pass fails.

    Returns (estimated_flops, error_message).
    On success error_message is empty. On failure estimated_flops is None.
    """
    task = challenge.get("task", {})
    tp = task.get("task_params", {})

    # Execute the code in a restricted namespace
    namespace = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
    except Exception as exc:
        return None, f"Code execution failed: {exc}"

    build_model_fn = namespace.get("build_model")
    if not callable(build_model_fn):
        return None, "build_model not found or not callable"

    # Instantiate the model using task_params from the challenge (never hardcode)
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

    # Primary: forward-pass hooks (accurate for models that reshape internally)
    input_shape = _infer_input_shape(tp)
    if input_shape is not None:
        total_flops, fwd_err = _forward_pass_flops(model, input_shape)
        if not fwd_err and total_flops >= 0:
            del model
            return total_flops, ""

    # Fallback: static module-tree walk
    int_vals = [v for v in tp.values() if isinstance(v, int)]
    seq_len = int_vals[1] if len(int_vals) > 1 else max(int_vals[0], 1) if int_vals else 1
    total_flops = _walk_flops(model, seq_len)

    del model
    return total_flops, ""
