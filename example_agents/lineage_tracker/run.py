#!/usr/bin/env python3
"""
Lineage Tracker Agent — follows evolutionary lineages.

Strategy: Find the best experiment on the Pareto front, trace its lineage,
identify what types of changes produced the biggest improvements, and apply
a similar change to the parent.

Output: a submission module with build_model() and build_optimizer().
"""

import json
import re
import sys
from difflib import unified_diff

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
    log("Lineage tracker agent starting")

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

    # Find best experiment
    best_exp = None
    pareto = []
    if db_url:
        try:
            resp = requests.get(f"{db_url}/experiments/pareto", timeout=10)
            if resp.ok:
                pareto = resp.json()
                if pareto:
                    best_exp = min(
                        [e for e in pareto if e.get("results", {}).get("success")],
                        key=lambda e: e.get("results", {}).get("metric", float("inf")),
                        default=None,
                    )
        except Exception:
            pass

    # Get lineage of best experiment
    lineage = []
    if best_exp and db_url:
        try:
            resp = requests.get(
                f"{db_url}/experiments/lineage/{best_exp['index']}", timeout=10,
            )
            if resp.ok:
                lineage = resp.json()
        except Exception:
            pass

    log(f"Best experiment: {best_exp.get('index', '?') if best_exp else 'none'}, lineage length: {len(lineage)}")

    # Analyze changes between generations
    changes = []
    for i in range(1, len(lineage)):
        prev = lineage[i - 1]
        curr = lineage[i]

        prev_code = prev.get("code", "")
        curr_code = curr.get("code", "")
        prev_metric = prev.get("results", {}).get("metric")
        curr_metric = curr.get("results", {}).get("metric")

        if prev_metric is None or curr_metric is None:
            continue

        improvement = prev_metric - curr_metric

        diff_lines = list(unified_diff(
            prev_code.splitlines(), curr_code.splitlines(), lineterm="",
        ))
        added = [l[1:] for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
        removed = [l[1:] for l in diff_lines if l.startswith("-") and not l.startswith("---")]

        change_type = _categorize_change(added, removed)
        changes.append({
            "type": change_type,
            "improvement": improvement,
            "motivation": curr.get("motivation", ""),
            "added_sample": added[:5],
        })

    log(f"Analyzed {len(changes)} lineage changes: {[c['type'] for c in changes]}")

    # Find the most impactful change type
    type_impact = {}
    for c in changes:
        t = c["type"]
        if t not in type_impact:
            type_impact[t] = []
        type_impact[t].append(c["improvement"])

    best_type = None
    best_avg = -float("inf")
    for t, improvements in type_impact.items():
        avg = sum(improvements) / len(improvements)
        if avg > best_avg:
            best_avg = avg
            best_type = t

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

    if best_type == "optimizer" or best_type is None:
        lr_match = re.search(r"lr\s*=\s*([\d.e-]+)", modified_code)
        if lr_match:
            old_lr = float(lr_match.group(1))
            new_lr = old_lr * 0.8
            modified_code = re.sub(r"lr\s*=\s*[\d.e-]+", f"lr={new_lr:.2e}", modified_code)
            motivation_parts.append(
                f"Lineage analysis: '{best_type or 'optimizer'}' changes averaged "
                f"{best_avg:.4f} improvement. Adjusted lr {old_lr:.2e} -> {new_lr:.2e}"
            )

    elif best_type == "architecture":
        if "LayerNorm" in modified_code:
            modified_code = modified_code.replace("LayerNorm", "RMSNorm")
            motivation_parts.append(
                f"Lineage analysis: architecture changes averaged {best_avg:.4f} improvement. "
                f"Applying RMSNorm (lighter normalization)"
            )
        elif re.search(r"depth\s*=\s*(\d+)", modified_code):
            depth_match = re.search(r"depth\s*=\s*(\d+)", modified_code)
            old_depth = int(depth_match.group(1))
            new_depth = max(old_depth - 1, 2)
            modified_code = re.sub(r"depth\s*=\s*\d+", f"depth={new_depth}", modified_code)
            motivation_parts.append(
                f"Lineage analysis: architecture changes averaged {best_avg:.4f} improvement. "
                f"Reduced depth {old_depth} -> {new_depth} for more steps in budget"
            )

    elif best_type == "hyperparameter":
        bs_match = re.search(r"['\"]?batch_size['\"]?\s*[:=]\s*(\d+)", modified_code)
        if bs_match:
            old_bs = int(bs_match.group(1))
            new_bs = old_bs * 2
            modified_code = re.sub(
                r"(['\"]?batch_size['\"]?\s*[:=]\s*)\d+",
                lambda m: m.group(1) + str(new_bs),
                modified_code,
            )
            motivation_parts.append(
                f"Lineage analysis: hyperparameter changes averaged {best_avg:.4f} improvement. "
                f"Doubled batch size {old_bs} -> {new_bs}"
            )

    if not motivation_parts:
        motivation_parts.append(
            f"Analyzed {len(lineage)} ancestors. Applied default submission module."
        )

    result = {
        "code": modified_code,
        "name": f"lineage_{best_type or 'analysis'}",
        "motivation": ". ".join(motivation_parts),
    }

    log(f"Best change type: {best_type}, avg improvement: {best_avg:.4f}")
    log(f"Final motivation: {'. '.join(motivation_parts)}")
    print(json.dumps(result))


def _categorize_change(added: list[str], removed: list[str]) -> str:
    """Categorize a code change by type."""
    all_text = " ".join(added + removed).lower()

    if any(kw in all_text for kw in ["adam", "sgd", "optimizer", "lr=", "learning_rate", "weight_decay"]):
        return "optimizer"
    if any(kw in all_text for kw in ["layer", "attention", "conv", "linear", "norm", "activation", "depth", "width"]):
        return "architecture"
    if any(kw in all_text for kw in ["batch_size", "dropout", "warmup", "schedule", "epochs"]):
        return "hyperparameter"
    if any(kw in all_text for kw in ["compile", "autocast", "amp", "gradient_accumulation"]):
        return "efficiency"

    return "other"


if __name__ == "__main__":
    main()
