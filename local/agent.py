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

import os
import random
from pathlib import Path
from typing import Optional

from local.task import INPUT_DIM, MAX_HIDDEN_WIDTH, estimate_flops_equivalent


def _load_active_prompt(round_id: int) -> dict:
    """Return ``{id, template}`` for the prompt variant this round
    should use. Reads via ``miner_template.prompts`` so the same
    on-disk layout (``prompts/active.json`` + ``history/gen_NNN.json``)
    works for both the local stack and the real miner CLI.

    Returns ``{"id": "", "template": ""}`` if no population exists
    (fresh stack, optimizer hasn't been run yet) so the agent falls
    back to its hardcoded heuristic.
    """
    try:
        from miner_template import prompts as prompts_mod
    except ImportError:
        return {"id": "", "template": ""}
    prompts_dir = os.environ.get("MINER_PROMPTS_DIR")
    pop = prompts_mod.load_active(Path(prompts_dir) if prompts_dir else None)
    if not pop:
        return {"id": "", "template": ""}
    try:
        pick = prompts_mod.pick_for_round(pop, round_id)
    except ValueError:
        return {"id": "", "template": ""}
    return {"id": pick.id, "template": pick.template}


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


def _tail_descending(tail: list[float], min_points: int = 4) -> bool:
    """Heuristic: is a loss curve still meaningfully descending at the end?

    Compares the average over the last third of the tail against the first
    third. Still-descending ⇒ the model is under-trained ⇒ a good
    continuation candidate. A flat/rising tail ⇒ prefer a fresh design.
    """
    pts = [float(x) for x in tail if x is not None]
    if len(pts) < min_points:
        return False
    k = max(1, len(pts) // 3)
    early = sum(pts[:k]) / k
    late = sum(pts[-k:]) / k
    if early <= 0:
        return False
    # Require >2% relative improvement across the window to count as descending.
    return (early - late) / abs(early) > 0.02


def _choose_continuation(challenge: dict, client=None) -> tuple[str, "Optional[int]"]:
    """Pick a warm-start parent by inspecting loss-curve tails.

    Returns ``(mode, parent_index)``. Uses the parents embedded in the
    challenge when present (zero-network), else queries ``/parents``.
    Falls back to ``("new", None)`` whenever nothing is clearly worth
    continuing — which is always the case for the numpy task (no
    checkpoints), so synth runs are unaffected.
    """
    if not challenge.get("continuation_allowed"):
        return "new", None
    parents = challenge.get("eligible_parents") or []
    if not parents and client is not None and challenge.get("db_url"):
        try:
            resp = client.get_json(
                f"{challenge['db_url']}/parents", timeout=5,
            )
            parents = (resp or {}).get("parents", [])
        except Exception:  # noqa: BLE001
            parents = []
    best_id = None
    best_metric = float("inf")
    for par in parents:
        if not par.get("checkpoint_available"):
            continue
        if not _tail_descending(par.get("loss_curve_tail", [])):
            continue
        m = par.get("metric")
        if m is not None and m < best_metric:
            best_metric = m
            best_id = par.get("id")
    if best_id is not None:
        return "continue", int(best_id)
    return "new", None


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

    # Read the active prompt variant for this round. The id round-trips
    # via the proposal so the local optimizer can attribute scores back
    # to the variant that produced this architecture.
    round_id = int(challenge.get("round_id", 0) or 0)
    prompt = _load_active_prompt(round_id)
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

    # Decide whether to warm-start from a still-improving parent. For the
    # numpy task there are never eligible parents, so this is a no-op there;
    # it's the reference path real ts_forecasting miners build on.
    mode, parent_index = _choose_continuation(challenge, client)
    if mode == "continue":
        reasoning_lines.append(
            f"Continuing from parent #{parent_index} (loss still descending)."
        )

    return {
        "code": code,
        "name": name,
        "motivation": motivation,
        "reasoning": "\n".join(reasoning_lines),
        "mode": mode,
        "parent_index": parent_index,
        "tool_calls": [
            {
                "tool": "internal",
                "name": "_pick_shape",
                "input": {"min_flops": min_flops, "max_flops": max_flops},
                "output": {"hidden": hidden, "activation": activation,
                           "lr": lr, "epochs": epochs},
            }
        ],
        "prompt_id": prompt["id"],
    }
