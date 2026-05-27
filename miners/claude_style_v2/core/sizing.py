"""Task-agnostic FLOP sizing via geometric-probe-then-refine.

FLOPs as a function of a scalar size knob is monotonic but piecewise:
head-count divisibility, group counts, kernel floors, and integer
truncation all create tiny flat/jumpy regions. Binary search oscillates
on those boundaries. Geometric probing on a log scale always terminates,
never needs derivative info, and returns a MEASURED point — not an
interpolation — so the caller knows the exact size that produced the
reported FLOPs.

Algorithm:

1. Take ``N_PROBE_GEOMETRIC`` log-spaced probes across
   ``[size_min, size_max]``. Wrap each probe in ``try / except`` —
   failures are treated as unreachable points, not fatal.
2. Find the two successful probes that bracket ``target_flops``.
3. Between the two bracketing sizes, take up to ``N_PROBE_REFINE`` more
   linearly-spaced probes (exclusive of the bracket endpoints — they
   already have measurements).
4. Hard cap ``MAX_PROBES`` total — never exceed.
5. Return the measured ``(size, flops)`` pair whose flops is closest to
   ``target_flops``.
"""
from __future__ import annotations

import math
from typing import Callable

N_PROBE_GEOMETRIC = 8
N_PROBE_REFINE = 5
MAX_PROBES = 15


def _geometric(size_min: int, size_max: int, n: int) -> list[int]:
    """Return up to ``n`` distinct log-spaced integer sizes in
    ``[size_min, size_max]`` (endpoints included)."""
    if n <= 0 or size_min <= 0 or size_max < size_min:
        return []
    if n == 1 or size_min == size_max:
        return [size_min]
    log_min = math.log(size_min)
    log_max = math.log(size_max)
    sizes: list[int] = []
    for i in range(n):
        frac = i / (n - 1)
        s = int(round(math.exp(log_min + (log_max - log_min) * frac)))
        s = max(size_min, min(size_max, s))
        if s not in sizes:
            sizes.append(s)
    return sizes


def _linear_interior(low: int, high: int, n: int) -> list[int]:
    """Return up to ``n`` distinct linearly-spaced integer sizes strictly
    between ``low`` and ``high`` (exclusive)."""
    if n <= 0 or high - low <= 1:
        return []
    out: list[int] = []
    for i in range(1, n + 1):
        frac = i / (n + 1)
        s = int(round(low + (high - low) * frac))
        if low < s < high and s not in out:
            out.append(s)
    return out


def _run_probe(probe: Callable[[int], int | None], size: int,
               measurements: dict[int, int | None]) -> None:
    """Run ``probe(size)`` if we haven't already, recording the outcome.

    Any exception, a ``None`` return, or a non-positive return is
    recorded as ``None`` (unreachable). Successful measurements are
    stored as ints.
    """
    if size in measurements:
        return
    try:
        flops = probe(size)
    except Exception:
        measurements[size] = None
        return
    if isinstance(flops, (int, float)) and flops > 0:
        measurements[size] = int(flops)
    else:
        measurements[size] = None


def sweep_sizes(
    probe: Callable[[int], int | None],
    size_min: int,
    size_max: int,
    target_flops: int,
) -> dict:
    """Geometric-then-refine FLOP sweep.

    ``probe(size)`` must return the measured FLOPs for that size, or
    ``None`` if the size is unreachable (e.g. build_model raised, the
    FlopCounter couldn't run). Any exception is caught and treated as
    unreachable — failures never abort the sweep.

    Returns a dict::

        {
            "probes":   [(size, flops), ...],   # successes, sorted by size
            "failures": [size, ...],            # unreachable sizes
            "best":     (size, flops) | None,   # closest to target
            "n_probes": int,                    # total attempts
        }

    ``best`` ties are broken by preferring the smaller size (conservative
    — a smaller model with the same FLOPs wastes less training compute).
    """
    measurements: dict[int, int | None] = {}

    def _budget_left() -> int:
        return MAX_PROBES - len(measurements)

    # Phase 1 — geometric sweep
    for size in _geometric(size_min, size_max, N_PROBE_GEOMETRIC):
        if _budget_left() <= 0:
            break
        _run_probe(probe, size, measurements)

    def _successes() -> list[tuple[int, int]]:
        return sorted(
            (s, f) for s, f in measurements.items()
            if isinstance(f, int) and f > 0
        )

    successes = _successes()

    # Phase 2 — linear refine between the bracketing pair
    if successes and _budget_left() > 0:
        below = [(s, f) for s, f in successes if f <= target_flops]
        above = [(s, f) for s, f in successes if f >= target_flops]
        if below and above:
            low = max(below, key=lambda sf: sf[0])[0]
            high = min(above, key=lambda sf: sf[0])[0]
        elif below:
            # All probes below target — refine between the top two sizes
            # to squeeze a tiny bit more out of the search.
            sizes = [s for s, _ in successes]
            low = sizes[-2] if len(sizes) >= 2 else sizes[-1]
            high = sizes[-1]
        else:
            # All probes above target — refine between the bottom two.
            sizes = [s for s, _ in successes]
            low = sizes[0]
            high = sizes[1] if len(sizes) >= 2 else sizes[0]

        if high > low + 1:
            refine_n = min(N_PROBE_REFINE, _budget_left())
            for size in _linear_interior(low, high, refine_n):
                if _budget_left() <= 0:
                    break
                _run_probe(probe, size, measurements)

    successes = _successes()
    failures = sorted(s for s, f in measurements.items() if f is None)

    best: tuple[int, int] | None = None
    if successes:
        # Primary key: distance to target. Secondary: smaller size wins
        # (cheap tiebreaker, useful when target is exactly achievable).
        best = min(successes, key=lambda sf: (abs(sf[1] - target_flops),
                                              sf[0]))

    return {
        "probes": successes,
        "failures": failures,
        "best": best,
        "n_probes": len(measurements),
    }
