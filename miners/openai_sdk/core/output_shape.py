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
  3. Compare against the actual tensor shape the model produces on a dummy
     forward pass. Works for 2D, 3D, 4D, or any rank that the constraint
     specifies.

If no constraint string is present (or none parses cleanly) inference returns
None and the verifier is a no-op — we never reject on guesses.
"""

import re
from typing import Iterable


_BATCH_TOKENS = frozenset({"b", "batch", "batch_size", "bs", "n", "nbatch"})

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

    return None


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
                "never hardcode lengths."
            )

    return None
