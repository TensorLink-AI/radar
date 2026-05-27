"""Generic input shape inference from task_params and constraints.

Determines the correct input tensor shape and dtype for any task,
so the FLOPs estimator can run a forward pass without hardcoded
assumptions about specific tasks like ts_forecasting or nanogpt.
"""

import re

import torch


# ── Known task heuristics ────────────────────────────────────────
# These are soft pattern matchers, NOT hardcoded task names.  They
# look at the *keys* present in task_params to infer the role of
# each parameter.  New tasks with different key names will fall
# through to the generic path.

_TOKEN_ID_KEYS = frozenset({
    "vocab_size", "n_vocab", "vocabulary_size",
})

_SEQUENCE_LENGTH_KEYS = frozenset({
    "block_size", "seq_len", "sequence_length", "max_seq_len",
    "context_len", "context_length", "max_length",
})

_CHANNEL_KEYS = frozenset({
    "num_variates", "n_channels", "in_channels", "num_channels",
    "input_channels", "n_features", "num_features",
})

# Vision-style keys — image side lengths and class counts.
_IMAGE_SIZE_KEYS = frozenset({
    "image_size", "img_size", "height", "width", "input_size",
})
_CLASS_COUNT_KEYS = frozenset({
    "n_classes", "num_classes", "num_labels", "n_labels", "n_targets",
})

# Generic feature-size keys.
_FEATURE_KEYS = frozenset({
    "input_dim", "d_input", "feature_size", "feature_dim", "d_model",
    "in_features", "n_inputs",
})

# Graph-style keys.
_GRAPH_NODE_KEYS = frozenset({"num_nodes", "n_nodes"})
_GRAPH_EDGE_KEYS = frozenset({"edge_dim", "num_edges", "n_edges"})


def _looks_like_token_task(tp: dict) -> bool:
    """Does this task take discrete token IDs as input?"""
    return bool(_TOKEN_ID_KEYS & set(tp.keys()))


def _find_key(tp: dict, candidates: frozenset) -> str | None:
    """Find the first key in tp that matches one of the candidates."""
    for k in tp:
        if k in candidates:
            return k
    return None


def _parse_constraints_for_shape(constraints: list[str]) -> list[int] | None:
    """Try to extract input shape from constraint strings.

    Looks for patterns like:
      "Model input: (batch, context_len, num_variates)"
      "Input shape: (B, block_size)"
      "Input tensor shape: (batch, seq_len)"
    """
    for c in constraints:
        m = re.search(
            r"[Ii]nput[^:]*:\s*\(([^)]+)\)", c
        )
        if m:
            dims = [d.strip() for d in m.group(1).split(",")]
            # Skip the batch dimension (first), return the rest as
            # hints (the caller resolves names to values from tp)
            return dims[1:] if len(dims) > 1 else dims
    return None


def infer_input(tp: dict, constraints: list[str] | None = None,
                ) -> tuple[list[int], torch.dtype]:
    """Infer model input shape and dtype from task_params.

    Returns (shape_including_batch, dtype).
    The batch dimension is always 1.

    Strategy (constraints-first — explicit declarations always win):
      1. If constraints declare an input shape that resolves against tp,
         use it (handles vision, graph, or any task that names its dims).
      2. Token-ID tasks (has vocab_size etc.) → (1, seq_len) LongTensor.
      3. Vision tasks with image_size / n_channels → (1, C, H, W).
      4. Sequence + channels (ts_forecasting-style) → (1, seq, channels).
      5. Generic feature-size key → (1, feat).
      6. Graph node count → (1, num_nodes, feat).
      7. Single sequence-length key → (1, seq_len).
      8. Fallback: first two integer task_params as dims.
    """
    if constraints is None:
        constraints = []

    # ── 1. Constraint-declared shape takes precedence ────────────
    # If the task explicitly declares an input shape and every name in
    # it resolves against task_params, trust it — it covers vision,
    # graph, and any task that uses names we don't have heuristics for.
    hint_dims = _parse_constraints_for_shape(constraints)
    if hint_dims:
        resolved: list[int] = []
        for d in hint_dims:
            if d in tp and isinstance(tp[d], int):
                resolved.append(int(tp[d]))
            else:
                try:
                    resolved.append(int(d))
                except ValueError:
                    resolved = []
                    break
        if resolved:
            # Token-ID task? Keep long dtype and only use the sequence dim.
            if _looks_like_token_task(tp):
                # Drop channel/class dims if the shape has them — token
                # models want (B, S) long input.
                seq_key = _find_key(tp, _SEQUENCE_LENGTH_KEYS)
                if seq_key and hint_dims and hint_dims[0] == seq_key:
                    return [1, int(tp[seq_key])], torch.long
            return [1] + resolved, torch.float32

    # ── 2. Token-ID task (nanogpt-style) ─────────────────────────
    if _looks_like_token_task(tp):
        seq_key = _find_key(tp, _SEQUENCE_LENGTH_KEYS)
        seq_len = int(tp[seq_key]) if seq_key else 128
        return [1, seq_len], torch.long

    # ── 3. Vision task (image_size + n_channels / classes) ───────
    img_key = _find_key(tp, _IMAGE_SIZE_KEYS)
    ch_key = _find_key(tp, _CHANNEL_KEYS)
    if img_key:
        side = int(tp[img_key])
        channels = int(tp[ch_key]) if ch_key else 3
        return [1, channels, side, side], torch.float32

    # ── 4. Sequence + channel task (ts_forecasting-style) ────────
    seq_key = _find_key(tp, _SEQUENCE_LENGTH_KEYS)
    if seq_key and ch_key:
        return [1, int(tp[seq_key]), int(tp[ch_key])], torch.float32

    # ── 5. Generic feature-size key ──────────────────────────────
    feat_key = _find_key(tp, _FEATURE_KEYS)
    if feat_key:
        # If there's also a sequence dim, use (B, S, F).
        if seq_key:
            return [1, int(tp[seq_key]), int(tp[feat_key])], torch.float32
        return [1, int(tp[feat_key])], torch.float32

    # ── 6. Graph node count ──────────────────────────────────────
    node_key = _find_key(tp, _GRAPH_NODE_KEYS)
    if node_key:
        # Default node feature dim of 1 when unspecified — enough to run
        # a forward pass.
        return [1, int(tp[node_key]), 1], torch.float32

    # ── 7. Single sequence-length key ────────────────────────────
    if seq_key:
        return [1, int(tp[seq_key])], torch.float32

    # ── 8. Generic fallback: use integer-valued params ───────────
    int_vals = [(k, v) for k, v in tp.items() if isinstance(v, int) and v > 0]
    if len(int_vals) >= 2:
        return [1, int_vals[0][1], int_vals[1][1]], torch.float32
    elif len(int_vals) == 1:
        return [1, int_vals[0][1]], torch.float32

    # Last resort — minimal 2D input
    return [1, 64], torch.float32
