"""Default miner agent for the local stack.

Mirrors ``miner_template/agent.py``'s contract — ``design_architecture``
returns ``{code, name, motivation, reasoning, tool_calls}`` — but emits
a numpy-MLP submission (no torch dependency) sized for the round's
FLOPs bucket.

Miners can override this by passing ``--agent_module
path/to/their.py`` to ``local/miner.py``; that module just needs a
top-level ``design_architecture(challenge: dict) -> dict``.
"""

from __future__ import annotations

import random
from typing import Optional

from local.task import INPUT_DIM, MAX_HIDDEN_WIDTH, estimate_flops_equivalent


def _pick_shape(min_flops: int, max_flops: int) -> tuple[list[int], str, float, int]:
    """Walk presets, take the largest one that fits.

    Returns (hidden_sizes, activation, learning_rate, epochs).
    """
    target = int(max_flops * 0.6)
    presets: list[tuple[list[int], str, float, int]] = [
        ([4],            "relu", 5e-2, 100),
        ([8],            "relu", 5e-2, 100),
        ([16],           "relu", 3e-2, 100),
        ([16, 16],       "tanh", 2e-2, 120),
        ([32, 16],       "tanh", 1e-2, 120),
        ([32, 32],       "tanh", 1e-2, 150),
        ([64, 32],       "tanh", 8e-3, 150),
        ([64, 64, 32],   "tanh", 5e-3, 180),
        ([96, 64, 32],   "tanh", 5e-3, 200),
        ([128, 64, 32],  "tanh", 3e-3, 200),
    ]
    best = presets[0]
    for p in presets:
        flops = estimate_flops_equivalent(p[0])
        if flops <= target:
            best = p
    return best


def _emit_code(hidden_sizes: list[int], activation: str,
               learning_rate: float, epochs: int) -> str:
    return f"""# Numpy MLP submission for the local radar stack.

class Model:
    hidden_sizes = {hidden_sizes!r}
    activation = {activation!r}
    learning_rate = {learning_rate!r}
    epochs = {epochs!r}


def build_model(input_dim, output_dim):
    return Model()
"""


def design_architecture(challenge: dict, client=None) -> dict:
    """Return a proposal dict for the given challenge.

    The signature mirrors the distributed harness — ``client`` is a
    ``shared.url_gate.GatedClient`` when called via ``local/miner.py``
    against a running validator, ``None`` for direct unit calls. We use
    it (when available) only to read the validator-side frontier so the
    code path is exercised; the agent's design logic doesn't depend on
    LLM/arxiv here.
    """
    db_url = challenge.get("db_url", "")
    if client is not None and db_url:
        try:
            client.get_json(f"{db_url}/frontier", timeout=5)
        except Exception:  # noqa: BLE001
            pass  # local services unreachable — fall through
    min_flops = int(challenge.get("min_flops_equivalent", 0))
    max_flops = int(challenge.get("max_flops_equivalent", 0))
    frontier = challenge.get("feasible_frontier", []) or []
    seed = int(challenge.get("seed", 0))
    rng = random.Random(seed)

    hidden, activation, lr, epochs = _pick_shape(min_flops, max_flops)
    # Inject a small amount of exploration so successive rounds don't
    # all submit the same architecture.
    if rng.random() < 0.3 and len(hidden) > 1:
        hidden = hidden[:-1]
    elif rng.random() < 0.3:
        hidden = [*hidden, max(4, hidden[-1] // 2)]
    lr = lr * (0.5 + rng.random())

    reasoning_lines = [
        f"Round target FLOPs range [{min_flops}, {max_flops}]; frontier "
        f"size {len(frontier)}.",
        f"Picked MLP shape {hidden} activation={activation} "
        f"epochs={epochs} lr={lr:.4g}.",
    ]
    if frontier:
        best = min(frontier, key=lambda x: x.get("metric", float("inf")))
        motivation = (
            f"Trying to beat frontier best (metric={best.get('metric')!s}) "
            f"with a {len(hidden)}-layer MLP."
        )
    else:
        motivation = (
            f"Bootstrapping bucket [{min_flops}, {max_flops}] with a "
            f"{len(hidden)}-layer MLP."
        )

    code = _emit_code(hidden, activation, lr, epochs)
    name = f"mlp_{'x'.join(str(h) for h in hidden)}_{activation}"

    return {
        "code": code,
        "name": name,
        "motivation": motivation,
        "reasoning": "\n".join(reasoning_lines),
        "tool_calls": [
            {
                "tool": "internal",
                "name": "_pick_shape",
                "input": {"min_flops": min_flops, "max_flops": max_flops},
                "output": {"hidden": hidden, "activation": activation,
                           "lr": lr, "epochs": epochs},
            }
        ],
        "prompt_id": "",
    }
