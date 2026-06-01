"""Architectural-primitive injection pool for v3.

The analyst samples a deterministic per-round subset from this pool
and hands them to the researcher as **required ingredients** — the
researcher's brainstorm phase MUST either propose a candidate that
uses each injected primitive, or explicitly reject it with reason.

The point is to break the LLM's habit of defaulting to a
patch-transformer for every task. Primitives are drawn from
families that are systematically under-represented on existing
miner frontiers (state-space, conv, mixer, recurrent, structured,
classical-signal). Each entry is a short tag the researcher can
expand on.

Sampling is deterministic on ``(round_id, task_name)`` so:
  * parallel miners on the same round see the same injection,
  * GEPA can attribute outcomes to specific injections,
  * reruns of a round produce identical primitives.
"""
from __future__ import annotations

import hashlib

# Curated under-used architectural primitives. Tagged loosely so the
# LLM has both a family hint and a concrete handle. Keep this list
# short and biased toward things absent from typical transformer
# baselines — adding mainstream primitives dilutes the divergence
# signal.
PRIMITIVE_POOL: tuple[str, ...] = (
    "state-space (S4/S5/Mamba-style selective scan)",
    "depthwise-separable conv1d stack",
    "MLP-Mixer (token-mixing + channel-mixing MLPs)",
    "linear-recurrent unit (LRU / RWKV-WKV)",
    "Hyena (long convolution with implicit parameterization)",
    "gMLP / spatial gating unit",
    "Fourier mixing (FNO / AFNO global token mix)",
    "Hopfield-style associative memory layer",
    "Newton-Schulz iteration as a layer",
    "tied / shared weights across depth (universal-transformer style)",
    "structured sparsity (block-sparse linear, butterfly factorization)",
    "graph-conv over a learned token adjacency",
    "Kalman-filter-style recurrence with learned gains",
    "wavelet-transform front-end + small backbone",
    "iterated refinement (small block applied N times with residuals)",
)


def sample_primitives(
    round_id: int, task_name: str = "", n: int = 2,
) -> list[str]:
    """Deterministically sample ``n`` primitives for this round.

    Hashes ``(round_id, task_name)`` into a starting offset, then walks
    the pool with a hash-derived stride to spread picks across the
    list. Both args influence the hash so different tasks on the same
    round get different injections — useful when the validator runs
    multiple task families.
    """
    if n <= 0 or not PRIMITIVE_POOL:
        return []
    pool_n = len(PRIMITIVE_POOL)
    n = min(n, pool_n)

    seed_bytes = f"{round_id}|{task_name}".encode("utf-8")
    digest = hashlib.sha256(seed_bytes).digest()
    start = digest[0] % pool_n
    # Stride coprime with pool_n keeps the walk full-period. We force
    # odd-ness; with PRIMITIVE_POOL length 15 this gives 8 valid
    # strides, which is plenty.
    stride = (digest[1] | 1) % pool_n or 1

    picks: list[str] = []
    seen: set[int] = set()
    idx = start
    while len(picks) < n:
        if idx not in seen:
            picks.append(PRIMITIVE_POOL[idx])
            seen.add(idx)
        idx = (idx + stride) % pool_n
        if len(seen) == pool_n:
            break
    return picks


def format_primitives_for_prompt(primitives: list[str]) -> str:
    """Render the injected primitives as a bullet list for prompts."""
    if not primitives:
        return "(no primitives injected this round)"
    return "\n".join(f"  - {p}" for p in primitives)
