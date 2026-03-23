#!/usr/bin/env python3
"""
Failure Analyst Agent — learns from recent failures.

Strategy: Query DB for recent failures, extract common patterns from
failed code, modify parent to avoid those patterns.

Output: a submission module with build_model() and build_optimizer().
"""

import json
import re
import sys

import requests

# Template with placeholders for size-aware model generation
_SUBMISSION_TEMPLATE = '''
import torch
import torch.nn as nn

class PatchForecaster(nn.Module):
    def __init__(self, ctx, pred, variates, n_q, d_model={d_model}, patch_size={patch_size}, depth={depth}):
        super().__init__()
        self.patch_size = patch_size
        self.pred_len = pred
        self.variates = variates
        self.n_q = n_q
        n_patches = ctx // patch_size
        self.patch_embed = nn.Linear(patch_size * variates, d_model)
        self.pos_emb = nn.Embedding(n_patches, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead={nhead}, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(d_model, pred * variates * n_q)

    def forward(self, x):
        B, T, V = x.shape
        x = x.reshape(B, T // self.patch_size, self.patch_size * V)
        h = self.patch_embed(x)
        pos = torch.arange(h.size(1), device=x.device).unsqueeze(0)
        h = h + self.pos_emb(pos)
        h = self.encoder(h)
        out = self.head(h.mean(dim=1))
        return out.view(B, self.pred_len, V, self.n_q)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return PatchForecaster(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

def training_config():
    return {{"batch_size": 64, "eval_interval": 200}}
'''


def _pick_model_dims(min_flops: int, max_flops: int) -> tuple[int, int, int, int]:
    """Pick (d_model, depth, patch_size, nhead) to fit within FLOPs range.

    Targets the midpoint of [min_flops, max_flops]. Uses wider models with
    more layers so wallclock calibration is dominated by actual compute
    rather than Python/BLAS overhead (which causes 2-4x variance for tiny
    models across different CPUs).
    """
    if min_flops > 0 and max_flops > 0:
        target = (min_flops + max_flops) // 2
    else:
        target = 10_000_000

    # (d_model, depth, patch_size, nhead) — larger models measure more
    # reliably via wallclock calibration.
    if target < 500_000:
        return 32, 1, 64, 4
    elif target < 5_000_000:
        return 64, 2, 32, 8
    elif target < 20_000_000:
        return 96, 2, 32, 8
    elif target < 60_000_000:
        return 128, 3, 32, 8
    elif target < 120_000_000:
        return 160, 3, 32, 8
    else:
        return 192, 4, 32, 8


def _make_default_submission(min_flops: int = 0, max_flops: int = 0) -> str:
    d_model, depth, patch_size, nhead = _pick_model_dims(min_flops, max_flops)
    return _SUBMISSION_TEMPLATE.format(
        d_model=d_model, depth=depth, patch_size=patch_size, nhead=nhead,
    ).strip()


def log(msg: str):
    """Write reasoning trace to stderr (captured by validator)."""
    print(msg, file=sys.stderr)


def main():
    challenge = json.loads(sys.stdin.read())
    log("Failure analyst agent starting")

    min_flops = challenge.get("min_flops_equivalent", 0)
    max_flops = challenge.get("max_flops_equivalent", 0)
    log(f"FLOPs range: [{min_flops}, {max_flops}]")

    frontier = challenge.get("feasible_frontier", [])
    db_url = challenge.get("db_url", "")
    # Use best frontier member as starting point, or empty
    if frontier:
        best = min(frontier, key=lambda x: x.get("metric", float("inf")))
        parent_code = best.get("code", "")
        log(f"Using frontier best (metric={best.get('metric', '?')}) as starting point")
    else:
        parent_code = ""
        log("No frontier available, using default submission")

    # Query failures
    failures = []
    if db_url:
        try:
            resp = requests.get(f"{db_url}/experiments/failures?n=10", timeout=10)
            if resp.ok:
                failures = resp.json()
        except Exception:
            pass

    log(f"Fetched {len(failures)} recent failures")

    # Analyze failure patterns
    failure_patterns = {
        "batch_size": [],
        "lr": [],
        "depth": [],
        "width": [],
    }

    for exp in failures:
        code = exp.get("code", "")

        bs_match = re.search(r"batch_size['\"]?\s*[:=]\s*(\d+)", code)
        if bs_match:
            failure_patterns["batch_size"].append(int(bs_match.group(1)))

        lr_match = re.search(r"lr\s*=\s*([\d.e-]+)", code)
        if lr_match:
            try:
                failure_patterns["lr"].append(float(lr_match.group(1)))
            except ValueError:
                pass

        depth_match = re.search(r"(?:depth|n_layers|num_layers)\s*=\s*(\d+)", code)
        if depth_match:
            failure_patterns["depth"].append(int(depth_match.group(1)))

        width_match = re.search(r"(?:width|d_model|hidden_size)\s*=\s*(\d+)", code)
        if width_match:
            failure_patterns["width"].append(int(width_match.group(1)))

    # If parent code was designed for a different size bucket, fall back
    # to a fresh default that targets the current round's FLOPs range.
    has_build_model = "def build_model" in parent_code
    if has_build_model and max_flops > 0:
        log(f"Parent code exists but max_flops={max_flops} set — "
            "falling back to default submission to avoid size-gate overshoot")
        modified_code = _make_default_submission(min_flops, max_flops)
    elif has_build_model:
        modified_code = parent_code
    else:
        modified_code = _make_default_submission(min_flops, max_flops)
    motivation_parts = []

    # Avoid learning rates that commonly fail
    if len(failure_patterns["lr"]) >= 2:
        from collections import Counter
        avg_bad_lr = sum(failure_patterns["lr"]) / len(failure_patterns["lr"])

        lr_match = re.search(r"lr\s*=\s*([\d.e-]+)", modified_code)
        if lr_match:
            current_lr = float(lr_match.group(1))
            if abs(current_lr - avg_bad_lr) / max(avg_bad_lr, 1e-10) < 0.5:
                new_lr = current_lr * 0.3
                modified_code = re.sub(
                    r"lr\s*=\s*[\d.e-]+",
                    f"lr={new_lr:.2e}",
                    modified_code,
                )
                motivation_parts.append(
                    f"Reduced lr from {current_lr:.2e} to {new_lr:.2e} "
                    f"(avg failing lr was {avg_bad_lr:.2e})"
                )

    if modified_code == parent_code or (not has_build_model and not motivation_parts):
        # Fallback: conservative hyperparameter change
        lr_match = re.search(r"lr\s*=\s*([\d.e-]+)", modified_code)
        if lr_match:
            old_lr = float(lr_match.group(1))
            new_lr = old_lr * 0.5
            modified_code = re.sub(
                r"lr\s*=\s*[\d.e-]+",
                f"lr={new_lr:.2e}",
                modified_code,
            )
            motivation_parts.append(
                f"Reduced lr from {old_lr:.2e} to {new_lr:.2e} as safety measure "
                f"(analyzed {len(failures)} failures)"
            )

    if not motivation_parts:
        motivation_parts.append(f"Analyzed {len(failures)} failures, applied default module")

    result = {
        "code": modified_code,
        "name": "failure_analyst_fix",
        "motivation": ". ".join(motivation_parts),
    }

    log(f"Final motivation: {'. '.join(motivation_parts)}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
