"""Persistent experiment history across rounds — stored in scratchpad state."""

import json
import os
import time


SIZE_BUCKETS = {
    "tiny":         (100_000,       500_000),
    "small":        (500_000,     2_000_000),
    "medium_small": (2_000_000,  10_000_000),
    "medium":       (10_000_000, 50_000_000),
    "large":        (50_000_000, 125_000_000),
}


def identify_bucket(flops_min: int, flops_max: int) -> str:
    """Identify which size bucket matches the given FLOPs range."""
    for name, (bmin, bmax) in SIZE_BUCKETS.items():
        if bmin == flops_min and bmax == flops_max:
            return name
    # Fuzzy match: find closest bucket by midpoint
    mid = (flops_min + flops_max) / 2
    best = "medium"
    best_dist = float("inf")
    for name, (bmin, bmax) in SIZE_BUCKETS.items():
        bmid = (bmin + bmax) / 2
        dist = abs(mid - bmid)
        if dist < best_dist:
            best_dist = dist
            best = name
    return best


def extract_flops_budget(challenge: dict) -> tuple[int, int]:
    """Extract FLOPs min/max from challenge, supporting both field formats."""
    fb = challenge.get("flops_budget", {})
    if isinstance(fb, dict) and (fb.get("min") or fb.get("max")):
        return int(fb.get("min", 0)), int(fb.get("max", 0))
    fmin = challenge.get("min_flops_equivalent", 0)
    fmax = challenge.get("max_flops_equivalent", 0)
    if fmin or fmax:
        return int(fmin), int(fmax)
    return 0, 0


def get_history(state: dict) -> list[dict]:
    """Get experiment history from scratchpad state."""
    return state.get("history", [])


def get_bucket_history(state: dict, bucket: str) -> list[dict]:
    """Get history entries for a specific size bucket."""
    return [e for e in get_history(state) if e.get("bucket") == bucket]


def format_history(entries: list[dict], max_entries: int = 10) -> str:
    """Format history entries for inclusion in prompts."""
    if not entries:
        return "No previous submissions."
    lines: list[str] = []
    for e in entries[-max_entries:]:
        lines.append(
            f"- {e.get('name', '?')} (bucket={e.get('bucket', '?')}, "
            f"strategy={e.get('strategy', '?')}): {e.get('motivation', '')}"
        )
    return "\n".join(lines)


def add_entry(state: dict, *, name: str, code: str, motivation: str,
              bucket: str = "", flops: int = 0,
              strategy: str = "", metadata: dict | None = None) -> dict:
    """Add a history entry to the scratchpad state. Returns updated state."""
    if "history" not in state:
        state["history"] = []
    entry = {
        "name": name,
        "motivation": motivation,
        "code_hash": hash(code) & 0xFFFFFFFF,
        "code_length": len(code),
        "bucket": bucket,
        "flops_target": flops,
        "strategy": strategy,
        "timestamp": time.time(),
    }
    if metadata:
        entry["metadata"] = metadata
    state["history"].append(entry)

    # Keep last 50 entries to avoid scratchpad bloat
    if len(state["history"]) > 50:
        state["history"] = state["history"][-50:]

    return state


def load_state(scratch_dir: str) -> dict:
    """Load state dict from scratch_dir/state.json."""
    path = os.path.join(scratch_dir, "state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(scratch_dir: str, state: dict) -> None:
    """Save state dict to scratch_dir/state.json."""
    os.makedirs(scratch_dir, exist_ok=True)
    path = os.path.join(scratch_dir, "state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
