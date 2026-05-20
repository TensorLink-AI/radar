"""Starter miner agent — submit this file (or a directory of .py files) to the subnet.

Validators inject your code into the official sandboxed agent image and
call ``design_architecture(challenge, client)``.  The ``client`` is a
GatedClient that can only reach validator-approved URLs:
  - challenge["db_url"]      — experiment database (read past results)
  - challenge["desearch_url"] — arxiv search via Desearch (SN22)
  - challenge["llm_url"]      — LLM inference via Chutes AI
  - Scratchpad presigned URLs  — persistent private storage across rounds

You do NOT build a Docker image. Just write .py files and run the miner:

  python miner/neuron.py --agent_dir miner_template/ ...

The miner neuron will POST your code to the DB server, and validators
fetch + run it every round.

Your agent module MUST define:
    design_architecture(challenge: dict, client: GatedClient) -> dict
        Returns {"code": str, "name": str, "motivation": str}

The ``code`` field should be a Python module that defines:
    build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module
    build_optimizer(model) -> torch.optim.Optimizer
"""

import json
import os
import sys

# When a miner needs to make authenticated requests back to the
# validator / database server it should use the HMAC helper from
# ``shared.auth`` (with the ``RADAR_SHARED_SECRET`` env var) instead of
# the removed chain-based wallet signing.
try:
    from shared.auth import sign_request as _hmac_sign_request  # noqa: F401
except Exception:  # pragma: no cover — template runs in sandboxes without the SDK
    _hmac_sign_request = None


def log(msg: str):
    """Write reasoning trace to stderr (captured by validator)."""
    print(msg, file=sys.stderr)


def _pick_hyperparams(min_flops: int, max_flops: int) -> dict:
    """Pick d_model, nhead, num_layers, dim_feedforward to fit the FLOPs range.

    Targets 60% of the bucket max to stay safely within bounds.
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


def _search_arxiv(client, challenge: dict, query: str) -> list[dict]:
    """Search arxiv papers via the desearch proxy.

    Returns a list of paper dicts with title, abstract, arxiv_id, etc.
    """
    desearch_url = challenge.get("desearch_url", "")
    if not desearch_url:
        log("No desearch_url in challenge, skipping arxiv search")
        return []

    try:
        resp = client.post_json(
            f"{desearch_url}/search",
            {"query": query, "max_results": 5},
        )
        return resp.get("results", [])
    except Exception as e:
        log(f"Arxiv search failed: {e}")
        return []


def _query_llm(client, challenge: dict, prompt: str, model: str = "") -> str:
    """Query the LLM proxy for architecture suggestions.

    Returns the LLM response content string.
    """
    llm_url = challenge.get("llm_url", "")
    if not llm_url:
        log("No llm_url in challenge, skipping LLM query")
        return ""

    try:
        # Check available models first
        models_resp = client.get_json(f"{llm_url}/models")
        available = models_resp.get("models", [])
        if available and not model:
            model = available[0]
        elif not model:
            model = "deepseek-ai/DeepSeek-V3-0324"

        resp = client.post_json(
            f"{llm_url}/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 4096,
            },
        )
        return resp.get("content", "")
    except Exception as e:
        log(f"LLM query failed: {e}")
        return ""


def _query_experiments(client, challenge: dict) -> list[dict]:
    """Fetch recent experiment results from the database.

    Returns a list of experiment dicts.
    """
    db_url = challenge.get("db_url", "")
    if not db_url:
        return []

    try:
        resp = client.get_json(f"{db_url}/experiments/recent?limit=10")
        return resp if isinstance(resp, list) else resp.get("experiments", [])
    except Exception as e:
        log(f"DB query failed: {e}")
        return []


def design_architecture(challenge: dict, client) -> dict:
    """Design a model architecture for this round.

    Args:
        challenge: Round challenge dict with frontier, FLOPs range, URLs, etc.
        client: GatedClient — the ONLY way to make HTTP requests.
            Use client.get_json(url), client.post_json(url, payload), etc.

    Returns:
        dict with keys: code, name, motivation
    """
    # Load persistent state from previous rounds
    scratch_dir = load_scratchpad(challenge)

    frontier = challenge.get("feasible_frontier", [])
    min_flops = challenge.get("min_flops_equivalent", 0)
    max_flops = challenge.get("max_flops_equivalent", 0)
    log(f"Designing for FLOPs range [{min_flops}, {max_flops}], frontier size: {len(frontier)}")

    # Example: search arxiv for relevant papers
    # papers = _search_arxiv(client, challenge, "time series forecasting transformer")
    # for p in papers:
    #     log(f"  Paper: {p.get('title', '?')}")

    # Example: ask the LLM for architecture ideas
    # llm_response = _query_llm(client, challenge,
    #     f"Suggest a PyTorch time series model architecture with ~{max_flops} FLOPs"
    # )
    # log(f"LLM suggestion: {llm_response[:200]}")

    # Example: check past experiment results
    # experiments = _query_experiments(client, challenge)

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
