"""Architecture reasoning guidance for the system prompt.

Three sections are built:

1. Operation reference — a compact table of common primitives with FLOPs
   formulas and inductive biases. Framed as a REFERENCE, not a menu — the
   LLM can use any PyTorch operation and measure it with
   ``estimate_layer_flops``.
2. Design questions — seven questions with Q1/Q3/Q4/Q6 pre-filled from
   the challenge so the LLM starts with concrete facts and spends its
   tokens on the questions that actually require reasoning.
3. Frontier gap analysis — what operations appear in frontier code,
   neutral report (never recommends).

``build_arch_guidance`` stays under ~2000 tokens.  The frontier section
is omitted entirely when no frontier exists.
"""

import re

from core.history import extract_flops_budget
from core.input_shape import infer_input
from core.output_shape import infer_output_shape


_OP_REFERENCE = """\
## Common Operation Costs (reference — you are not limited to these)

```
Op                 | FLOPs formula                           | Inductive bias
-------------------|-----------------------------------------|-------------------
nn.Linear(I,O)     | 2*I*O*elements                          | none
nn.Conv1d(Ci,Co,K) | 2*Ci*K*Co*Lout/groups                   | locality, shift-equivariant
nn.Conv2d(Ci,Co,K) | 2*Ci*Kh*Kw*Co*Oh*Ow/groups              | spatial locality
Depthwise-sep conv | ~8-9x cheaper than standard conv        | locality + channel bottleneck
nn.MHA(d,heads)    | 8*S*d^2 + 2*S^2*d                       | global pairwise (S^2 cost)
GLU / gating       | extra linear + elementwise              | feature selection
nn.GRU(h)          | ~8*h^2*S                                | causal, state compression
Residual add       | 0                                       | gradient flow
LayerNorm          | 2*elements                              | activation stability
```

For any operation not listed here, use `estimate_layer_flops` on a
minimal prototype to measure its actual cost."""


# ── Frontier scanning ────────────────────────────────────────────

_OP_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("attention",        re.compile(r"\bnn\.(MultiheadAttention|TransformerEncoder|TransformerDecoder|TransformerEncoderLayer)\b")),
    ("convolution",      re.compile(r"\bnn\.Conv\d[d]?\b")),
    ("recurrence",       re.compile(r"\bnn\.(GRU|LSTM|RNN)\b")),
    ("gating/GLU",       re.compile(r"\b(GLU|SwiGLU|GatedLinear|sigmoid\s*\()", re.IGNORECASE)),
    ("depthwise-sep",    re.compile(r"groups\s*=\s*[A-Za-z_0-9]+\s*\)")),
    ("normalization",    re.compile(r"\bnn\.(LayerNorm|BatchNorm\d[d]?|GroupNorm|InstanceNorm\d[d]?|RMSNorm)\b")),
    ("state-space/SSM",  re.compile(r"\b(S4|Mamba|SSM|state_space)\b", re.IGNORECASE)),
]


def scan_frontier_ops(frontier: list[dict]) -> str:
    """Scan frontier code for known operation patterns and report neutrally.

    Returns a 1-3 line summary of what's present and what's absent. Never
    recommends — the strategy + LLM decide what to do with gaps.
    """
    if not frontier:
        return "No frontier yet — bootstrap territory."

    total = len(frontier)
    counts: dict[str, int] = {name: 0 for name, _ in _OP_PATTERNS}
    linear_only: int = 0
    for member in frontier:
        code = member.get("code", "") or ""
        if not isinstance(code, str):
            continue
        hit_any = False
        for name, pat in _OP_PATTERNS:
            if pat.search(code):
                counts[name] += 1
                if name in ("attention", "convolution", "recurrence",
                            "gating/GLU", "state-space/SSM"):
                    hit_any = True
        if not hit_any and "nn.Linear" in code:
            linear_only += 1

    present = [f"{c}/{total} {name}" for name, c in counts.items() if c]
    if linear_only:
        present.append(f"{linear_only}/{total} MLP-only")
    not_tried = [name for name, c in counts.items() if not c]

    lines = []
    if present:
        lines.append("Frontier uses: " + ", ".join(present) + ".")
    else:
        lines.append("Frontier has code but no familiar op patterns matched.")
    if not_tried:
        lines.append("Not yet explored: " + ", ".join(not_tried) + ".")
    return "\n".join(lines)


# ── Design questions ─────────────────────────────────────────────

def _pretty_shape(shape: list[int] | tuple) -> str:
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _pretty_expected_output(expected: list[int]) -> str:
    parts = ["B"] + [str(e) if e >= 0 else "?" for e in expected]
    return "(" + ", ".join(parts) + ")"


def _check_causal(constraints: list[str]) -> bool:
    text = " ".join(c for c in constraints if isinstance(c, str)).lower()
    return any(k in text for k in ("causal", "autoregressive", "no future", "masked future"))


def _build_design_questions(challenge: dict) -> str:
    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    constraints = task.get("constraints", []) or []
    _, flops_max = extract_flops_budget(challenge)
    target = int(flops_max * 0.6) if flops_max else 0

    # Q1: input
    try:
        input_shape, _ = infer_input(tp, constraints)
        in_pretty = _pretty_shape(input_shape)
        in_elem = 1
        for d in input_shape[1:]:
            in_elem *= d
        q1 = (
            f"Q1. Input: shape {in_pretty}, "
            f"{in_elem:,} elements per sample excluding batch."
        )
    except Exception:
        in_elem = 0
        q1 = "Q1. Input: shape unknown — inspect task_params and constraints."

    # Q3: causality
    causal = _check_causal(constraints)
    q3 = (
        "Q3. Causal constraint: YES — constraints mention causal/autoregressive "
        "behavior. Future tokens must not leak into past predictions."
        if causal else
        "Q3. Causal constraints: not obviously flagged in constraints — "
        "infer from task description if relevant."
    )

    # Q4: input → output
    expected = infer_output_shape(tp, constraints)
    if expected is not None:
        out_pretty = _pretty_expected_output(expected)
        out_elem = 1
        for d in expected:
            if d > 0:
                out_elem *= d
        transform_bits: list[str] = []
        if in_elem and out_elem:
            ratio = out_elem / in_elem
            if ratio < 0.9:
                transform_bits.append(f"compress by ~{1/ratio:.2f}x")
            elif ratio > 1.1:
                transform_bits.append(f"expand by ~{ratio:.2f}x")
            else:
                transform_bits.append("roughly preserve element count")
        try:
            if expected and len(input_shape) - 1 != len(expected):
                transform_bits.append(
                    f"change rank from {len(input_shape) - 1}D to {len(expected)}D"
                )
        except Exception:
            pass
        extra = f" ({'; '.join(transform_bits)})" if transform_bits else ""
        q4 = (
            f"Q4. Input→Output: {in_pretty if in_elem else 'input'} → {out_pretty}"
            f"{extra}."
        )
    else:
        out_elem = 0
        q4 = (
            "Q4. Input→Output: no parseable output-shape constraint — "
            "derive the output structure from the task description/objectives."
        )

    # Q6: budget per element
    if target and in_elem and out_elem:
        per_elem = target / (in_elem * out_elem)
        q6 = (
            f"Q6. Budget ratio: target {target:,} FLOPs / "
            f"({in_elem:,} in × {out_elem:,} out) ≈ {per_elem:.2f} ops per "
            f"input×output element pair."
        )
    elif target and in_elem:
        q6 = (
            f"Q6. Budget ratio: target {target:,} FLOPs / {in_elem:,} input "
            f"elements ≈ {target / in_elem:.2f} ops per input element."
        )
    else:
        q6 = "Q6. Budget ratio: compute once shapes are fixed."

    # Q5: channel independence — flag counts only, never recommend.
    q5_bits: list[str] = []
    for k in ("num_variates", "n_channels", "in_channels", "num_channels",
             "input_channels", "n_features", "num_features"):
        if k in tp and isinstance(tp[k], int):
            q5_bits.append(f"{k}={tp[k]}")
            break
    if q5_bits:
        q5 = (
            f"Q5. Channel / variate count: {', '.join(q5_bits)}. "
            "Decide whether channels share parameters or are modeled independently."
        )
    else:
        q5 = (
            "Q5. Channel / variate count: no explicit channel param — "
            "treat as single-channel unless the task description says otherwise."
        )

    q2 = (
        "Q2. Long-range vs local interactions: look at the domain prompt. "
        "If outputs depend on far-apart input positions, a global op "
        "(attention, state-space, full MLP over positions) may help; if not, "
        "local ops (convs, windowed attention) are cheaper."
    )

    q7 = (
        "Q7. Frontier gaps: see the 'Frontier Ops' section for what's been "
        "tried. An unexplored op family is an opportunity — but only if it "
        "matches the task's inductive biases."
    )

    return (
        "## Design Questions (answer these before writing code)\n"
        + "\n".join([q1, q2, q3, q4, q5, q6, q7])
    )


# ── Public entry point ───────────────────────────────────────────

def build_arch_guidance(challenge: dict, frontier: list[dict] | None = None) -> str:
    """Return the full architecture-reasoning guidance section."""
    sections: list[str] = [_OP_REFERENCE, _build_design_questions(challenge)]
    if frontier:
        sections.append("## Frontier Ops\n" + scan_frontier_ops(frontier))
    return "\n\n".join(sections)
