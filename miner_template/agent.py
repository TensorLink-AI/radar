"""Starter miner agent — reads Challenge from stdin, writes Proposal to stdout.

Build: docker build -t my-agent:latest .
Validators will pull this image and run it in a sandbox.

The agent receives a Challenge JSON on stdin containing:
  - feasible_frontier: list of {code, metric, objectives} for frontier
    points in this round's size bucket
  - task, db_url, desearch_url, seed
  - min_flops_equivalent, max_flops_equivalent
  - round_id, eval_split_seed

It must print a Proposal JSON to stdout with:
  - code: architecture module with build_model() and build_optimizer()
  - name: human-readable name
  - motivation: why this architecture should work well
"""

import json
import os
import sys
import tarfile
import tempfile
import urllib.request


def log(msg: str):
    """Write reasoning trace to stderr (captured by validator)."""
    print(msg, file=sys.stderr)


def load_scratchpad(challenge: dict, local_dir: str = "/tmp/scratchpad") -> str:
    """Download and extract the agent's persistent scratchpad from R2.

    Returns the local directory path. Creates it empty if no scratchpad exists
    (first round for this miner).
    """
    os.makedirs(local_dir, exist_ok=True)
    url = challenge.get("scratchpad_get_url", "")
    if not url:
        return local_dir

    try:
        archive_path = os.path.join(local_dir, "state.tar.gz")
        # Use explicit timeout to avoid hanging indefinitely on slow/unresponsive URLs
        with urllib.request.urlopen(url, timeout=10) as resp, open(archive_path, "wb") as f:
            f.write(resp.read())
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(local_dir, filter="data")
        os.remove(archive_path)
        log(f"Loaded scratchpad ({len(os.listdir(local_dir))} files)")
    except Exception as e:
        log(f"Scratchpad load: {e} (starting fresh)")
    return local_dir


def save_scratchpad(challenge: dict, local_dir: str = "/tmp/scratchpad") -> bool:
    """Upload the agent's scratchpad to R2 for next round.

    Tars everything in local_dir and uploads via presigned PUT URL.
    Call this before exiting.
    """
    url = challenge.get("scratchpad_put_url", "")
    if not url:
        return False

    try:
        fd, archive_path = tempfile.mkstemp(suffix=".tar.gz", prefix="scratchpad_")
        os.close(fd)
        with tarfile.open(archive_path, "w:gz") as tar:
            for root, dirs, files in os.walk(local_dir):
                for f in files:
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full, local_dir)
                    tar.add(full, arcname=arcname)

        max_mb = challenge.get("scratchpad_max_mb", 10)
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        if size_mb > max_mb:
            log(f"Scratchpad too large ({size_mb:.1f}MB > {max_mb}MB limit). Not saving.")
            os.remove(archive_path)
            return False

        with open(archive_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Content-Type", "application/gzip")
        urllib.request.urlopen(req)
        os.remove(archive_path)
        log(f"Saved scratchpad ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        log(f"Scratchpad save failed: {e}")
        return False


def _pick_hyperparams(min_flops: int, max_flops: int) -> dict:
    """Pick d_model, nhead, num_layers, dim_feedforward to fit the FLOPs range.

    Targets 60% of the bucket max to stay safely within bounds. FLOPs scale
    roughly as: seq_len * num_layers * (4*d^2 + 2*seq*d + 8*d*ff)
    where seq_len=512, so we use a lookup table with measured values.
    """
    target = int(max_flops * 0.6)

    # Preset configs ordered by approximate FLOPs-equivalent
    presets = [
        # (d_model, nhead, num_layers, dim_feedforward, ~flops)
        (16,  2, 1,   32,    150_000),     # tiny-low
        (24,  2, 1,   64,    350_000),     # tiny-mid
        (32,  2, 2,   64,    800_000),     # small-low
        (48,  4, 2,  128,  1_500_000),     # small-mid
        (64,  4, 2,  192,  3_500_000),     # medium-small-low
        (64,  4, 3,  256,  7_000_000),     # medium-small-mid
        (96,  4, 3,  384, 20_000_000),     # medium-low
        (128, 4, 3,  512, 40_000_000),     # medium-mid
        (128, 4, 4,  512, 60_000_000),     # large-low
        (160, 8, 4,  640, 90_000_000),     # large-mid
    ]

    # Pick the largest preset that fits under target
    best = presets[0]
    for preset in presets:
        if preset[4] <= target:
            best = preset

    return {
        "d_model": best[0],
        "nhead": best[1],
        "num_layers": best[2],
        "dim_feedforward": best[3],
    }


def design_architecture(challenge: dict) -> dict:
    """Design a model architecture for this round.

    Override this function with your agent logic. You can:
    - Study challenge["feasible_frontier"] to see what's already working
    - Query the validator DB at challenge["db_url"]
    - Design architecture within the FLOPs range
    - Use an LLM, evolutionary strategies, etc.

    Returns dict with: code, name, motivation
    """
    # Load persistent state from previous rounds (optional)
    scratch_dir = load_scratchpad(challenge)
    # Your agent's state lives here — probe DB, notes, whatever
    # e.g. if os.path.exists(f"{scratch_dir}/probe.db"): ...

    frontier = challenge.get("feasible_frontier", [])
    min_flops = challenge.get("min_flops_equivalent", 0)
    max_flops = challenge.get("max_flops_equivalent", 0)
    log(f"Designing for FLOPs range [{min_flops}, {max_flops}], frontier size: {len(frontier)}")

    hp = _pick_hyperparams(min_flops, max_flops)
    log(f"Selected hyperparams: {hp}")

    if frontier:
        best = min(frontier, key=lambda x: x.get("metric", float("inf")))
        motivation = (
            f"Improving on frontier best (metric={best.get('metric', '?')}). "
            f"Target FLOPs range: [{min_flops}, {max_flops}]"
        )
    else:
        motivation = (
            f"No frontier in size range [{min_flops}, {max_flops}]. "
            f"Exploring with baseline transformer."
        )

    d_model = hp["d_model"]
    nhead = hp["nhead"]
    num_layers = hp["num_layers"]
    dim_feedforward = hp["dim_feedforward"]

    code = f'''
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, quantiles):
        super().__init__()
        d_model = {d_model}
        self.proj_in = nn.Linear(num_variates, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead={nhead}, dim_feedforward={dim_feedforward}, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers={num_layers})
        self.proj_out = nn.Linear(
            d_model, prediction_len * num_variates * len(quantiles),
        )
        self.prediction_len = prediction_len
        self.num_variates = num_variates
        self.num_quantiles = len(quantiles)

    def forward(self, x):
        B = x.shape[0]
        h = self.proj_in(x)
        h = self.encoder(h)
        out = self.proj_out(h[:, -1])
        return out.view(B, self.prediction_len, self.num_variates, self.num_quantiles)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return SimpleTransformer(context_len, prediction_len, num_variates, quantiles)

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
'''
    # Save state for next round
    save_scratchpad(challenge, scratch_dir)

    return {
        "code": code,
        "name": f"transformer_d{d_model}_l{num_layers}_ff{dim_feedforward}",
        "motivation": motivation,
    }


def main():
    challenge = json.loads(sys.stdin.read())
    log("Agent starting")
    proposal = design_architecture(challenge)
    log(f"Proposal: {proposal['name']} — {proposal['motivation']}")
    print(json.dumps(proposal))


if __name__ == "__main__":
    main()
