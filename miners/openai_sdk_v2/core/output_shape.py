"""Generic output shape inference and verification.

Purpose: catch "model outputs wrong shape" failures BEFORE submission, so
training doesn't die on tensor-size mismatches during training.

Strategy (task-agnostic):
  1. Parse the task's ``constraints`` strings for a pattern like
     ``Output[...]: (batch, D1, D2, len(some_list))``. The dimension names
     come from whatever the challenge declares — no task-specific names
     are baked in.
  2. Resolve each non-batch dimension against ``task_params``:
       - literal integer        → e.g. ``96``
       - direct key             → ``key_name`` → ``tp["key_name"]``
       - ``len(...)``           → ``len(some_list)`` → ``len(tp["some_list"])``
       - unresolved name        → wildcard (``-1``) — skipped during compare
  3. Fallback when no ``Output:`` constraint exists: if a ``build_model(...)``
     signature constraint is present AND ``task_params`` contains a key that
     names a per-sample output length (``pred*``, ``forecast*``, ``target*``,
     ``out*``), derive the expected non-batch dims as the build_model params
     in declared order, minus any param that looks input-like (sequence/
     context, vocab, image, graph-node keys). The classic ts_forecasting
     mismatch (model emits 96 vs target 64) is caught here without baking a
     task name into the validator.
  4. Compare against the actual tensor shape the model produces on a dummy
     forward pass. Works for 2D, 3D, 4D, or any rank that the constraint
     specifies.

If no constraint string is present (or none parses cleanly) inference returns
None and the verifier is a no-op — we never reject on guesses.
"""

import re
from typing import Iterable


_BATCH_TOKENS = frozenset({"b", "batch", "batch_size", "bs", "n", "nbatch"})

# Soft pattern matchers (substring tests on lowercased key names).
# Used by the build_model-signature fallback to decide which params belong
# on the output side (kept) vs the input side (dropped). New tasks naming
# their dims differently fall through and the fallback simply declines —
# no false rejections, just no extra check.
_INPUT_NAME_HINTS = (
    "context", "ctx_", "input", "in_dim", "in_features",
    "seq_len", "sequence", "block", "max_len", "max_length",
    "vocab", "image", "img", "height", "width",
    "node", "edge", "feature_size", "feature_dim", "d_model",
)
_OUTPUT_LENGTH_HINTS = (
    "pred", "forecast", "target", "horizon", "out_len", "output_len",
)

# Anchors the shape extractor on the first "Output" mention. We then scan
# forward to the next opening paren and extract a balanced paren group so
# nested expressions like ``len(quantiles)`` survive.
_OUTPUT_ANCHOR_RE = re.compile(r"[Oo]utput")


def _extract_shape_group(line: str) -> str | None:
    """Return the text inside the first balanced paren group after 'Output'.

    Handles nested parens (``len(quantiles)``) by tracking depth.
    Returns None when no balanced group is found.
    """
    anchor = _OUTPUT_ANCHOR_RE.search(line)
    if not anchor:
        return None
    start = line.find("(", anchor.end())
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(line)):
        c = line[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return line[start + 1:i]
    return None  # unbalanced


def _split_top_level(s: str) -> list[str]:
    """Split ``s`` on commas that are not nested inside parens.

    Preserves expressions like ``len(x, y)`` as a single token.
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for c in s:
        if c == "(":
            depth += 1
            buf.append(c)
        elif c == ")":
            depth -= 1
            buf.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _resolve_dim(token: str, tp: dict) -> int | None:
    """Resolve a single dimension token to an int.

    Returns None when the token is unresolved (treated as wildcard by the
    caller). Batch tokens return None too — the caller strips the batch dim
    before calling this.
    """
    t = token.strip()
    if not t:
        return None

    # Literal integer?
    try:
        return int(t)
    except ValueError:
        pass

    # len(key)?
    m = re.fullmatch(r"len\(\s*([A-Za-z_][A-Za-z_0-9]*)\s*\)", t)
    if m:
        key = m.group(1)
        v = tp.get(key)
        if isinstance(v, (list, tuple)):
            return len(v)
        return None

    # Direct key lookup.
    v = tp.get(t)
    if isinstance(v, int):
        return v
    if isinstance(v, (list, tuple)):
        # e.g. a shape constraint that references the `quantiles` list
        # as a dim means "one output per quantile" — use its length.
        return len(v)

    # "num_X" / "n_X" → len(tp["X"]) or tp["X"] when X is a collection/int
    # e.g. constraint says ``num_quantiles`` but task_params only has
    # ``quantiles`` (the list). We resolve to ``len(quantiles)``.
    for prefix in ("num_", "n_"):
        if t.startswith(prefix) and len(t) > len(prefix):
            base = t[len(prefix):]
            bv = tp.get(base)
            if isinstance(bv, (list, tuple)):
                return len(bv)
            if isinstance(bv, int):
                return bv

    # "X_len" / "X_length" / "X_size" → len(tp["X"]) when X is a collection.
    for suffix in ("_len", "_length", "_size"):
        if t.endswith(suffix) and len(t) > len(suffix):
            base = t[: -len(suffix)]
            bv = tp.get(base)
            if isinstance(bv, (list, tuple)):
                return len(bv)

    return None


_BUILD_MODEL_SIG_RE = re.compile(r"build_model\s*\(([^)]*)\)")


def _looks_input_like(name: str) -> bool:
    """Soft check: does this task_param name describe an input dimension?

    Substring match on hints — keeps the heuristic generic. False negatives
    (treating a true input dim as output) just produce a verifier wildcard
    when the name doesn't resolve cleanly, so the worst case is no check.
    """
    low = name.lower()
    return any(h in low for h in _INPUT_NAME_HINTS)


def _has_output_length_key(task_params: dict) -> bool:
    """Trigger guard for the build_model-signature fallback.

    Returns True when task_params declares a per-sample output length under
    one of the common name patterns. Without this we'd risk inferring an
    output shape for tasks that don't have one (e.g. classification).
    """
    for k in task_params:
        low = k.lower()
        if any(h in low for h in _OUTPUT_LENGTH_HINTS):
            return True
    return False


def _infer_from_build_model_signature(
    task_params: dict, constraints: Iterable[str],
) -> list[int] | None:
    """Fallback: derive expected output dims from a build_model signature.

    Looks for a ``build_model(p1, p2, ..., pN)`` string in the constraints,
    drops params that look input-like by name, and resolves the rest in
    declared order. List-valued params resolve to their length (e.g.
    ``quantiles`` → ``len(quantiles)``).

    Only fires when ``task_params`` advertises an output-length key — see
    ``_has_output_length_key`` — to avoid synthesising an expected shape for
    tasks that don't conceptually have one.
    """
    if not _has_output_length_key(task_params):
        return None

    for c in constraints:
        if not isinstance(c, str):
            continue
        m = _BUILD_MODEL_SIG_RE.search(c)
        if not m:
            continue
        params = [p.strip() for p in m.group(1).split(",") if p.strip()]
        if not params:
            continue
        kept: list[int] = []
        for p in params:
            # Strip type annotations / defaults defensively (e.g. "ctx=512").
            name = p.split(":")[0].split("=")[0].strip()
            if not name or _looks_input_like(name):
                continue
            v = _resolve_dim(name, task_params)
            if v is None:
                # An output-side param we can't resolve to an int isn't a
                # safe basis for a hard reject — back out entirely.
                return None
            kept.append(v)
        if kept:
            return kept
    return None


def infer_output_shape(task_params: dict,
                       constraints: Iterable[str] | None) -> list[int] | None:
    """Infer the expected output shape (excluding batch) from constraints.

    Returns a list of ints where unresolved dims are ``-1`` (wildcards),
    or ``None`` if no output-shape constraint was found / parseable.
    """
    if not constraints:
        return None

    for c in constraints:
        if not isinstance(c, str):
            continue
        group = _extract_shape_group(c)
        if group is None:
            continue
        raw_dims = [d for d in _split_top_level(group) if d]
        if len(raw_dims) < 2:
            # Only a batch dim (or nothing useful) — keep looking.
            continue

        # Drop batch dimension if present. If the first token doesn't look
        # like a batch token we still drop it — constraints in this codebase
        # always list batch first.
        first = raw_dims[0].lower().strip()
        non_batch = raw_dims[1:] if first in _BATCH_TOKENS or first == "" else raw_dims[1:]

        resolved: list[int] = []
        for tok in non_batch:
            v = _resolve_dim(tok, task_params)
            resolved.append(v if v is not None else -1)
        return resolved

    # No explicit Output: constraint — fall back to deriving expected dims
    # from the build_model signature when the task advertises a per-sample
    # output length. Keeps the check task-agnostic: bails out for tasks
    # whose params don't name an output-length dim.
    return _infer_from_build_model_signature(task_params, constraints)


def verify_output_shape(actual: tuple | list,
                        expected: list[int]) -> str | None:
    """Compare a full actual shape (incl. batch) against expected (excl. batch).

    Returns None on success, else a human-readable error string suitable for
    surfacing to the LLM. ``-1`` entries in ``expected`` are wildcards.
    """
    if actual is None:
        return "Model forward pass produced no tensor output"

    actual = tuple(actual)
    # The expected list excludes batch; the actual shape includes it.
    actual_non_batch = actual[1:]
    expected_rank = len(expected) + 1  # +1 for batch

    def _pretty_expected() -> str:
        parts = ["B"] + [str(e) if e >= 0 else "?" for e in expected]
        return "(" + ", ".join(parts) + ")"

    if len(actual) != expected_rank:
        return (
            f"Output rank mismatch: expected {expected_rank}D "
            f"{_pretty_expected()}, got {len(actual)}D {tuple(actual)}. "
            "Check that forward() returns a tensor with the shape required "
            "by the task constraints."
        )

    for i, (a, e) in enumerate(zip(actual_non_batch, expected)):
        if e >= 0 and a != e:
            return (
                f"Output dim {i + 1} (non-batch dim {i}) mismatch: "
                f"expected {e}, got {a}. "
                f"Full expected {_pretty_expected()}, actual {tuple(actual)}. "
                "Make sure every dimension is derived from task_params — "
                "never hardcode lengths. Length-like params (e.g. context "
                "and output/prediction lengths) are INDEPENDENT: a layer "
                "that bridges them must project explicitly, e.g. "
                "`nn.Linear(context_len // patch, output_len)`."
            )

    return None
