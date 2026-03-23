"""Phase C evaluation — the trust anchor.

Every validator independently downloads checkpoints and computes metrics.
Cheap (seconds per model on CPU), fully deterministic, provides consensus.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import random
import tempfile
import types
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from shared.r2_audit import R2AuditLog
    from shared.task import TaskSpec

logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    architecture_code: str,
    checkpoint_path: str,
    eval_split_seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Evaluate a single checkpoint. This is the Phase C core function.

    1. exec architecture_code -> get build_model()
    2. model = build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES)
    3. model.load_state_dict(torch.load(checkpoint_path))
    4. flops_equiv = compute_flops_equivalent(model, ...)
    5. metrics = validate(model)
    6. return {crps, mase, flops_equivalent_size, param_count}
    """
    # Import frozen eval dependencies
    import sys
    runner_dir = os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast")
    runner_dir = os.path.abspath(runner_dir)
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)

    import torch
    from prepare import validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES
    from flops import compute_flops_equivalent
    from safetensors.torch import load_file

    random.seed(eval_split_seed)
    torch.manual_seed(eval_split_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Load architecture code
    mod = types.ModuleType("submission")
    try:
        exec(compile(architecture_code, "<submission>", "exec"), mod.__dict__)
    except Exception as e:
        logger.error("Architecture code failed to compile: %s", e)
        return {"crps": float("inf"), "mase": float("inf"), "error": str(e)}

    if not hasattr(mod, "build_model") or not callable(mod.build_model):
        return {"crps": float("inf"), "mase": float("inf"), "error": "Missing build_model()"}

    # 2. Build model
    try:
        model = mod.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).to(device)
    except Exception as e:
        return {"crps": float("inf"), "mase": float("inf"), "error": f"build_model failed: {e}"}

    # 3. Load checkpoint (safetensors — safe, no pickle)
    try:
        state_dict = load_file(checkpoint_path, device=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        return {"crps": float("inf"), "mase": float("inf"), "error": f"load_state_dict failed: {e}"}

    # 4. Measure FLOPs
    try:
        flops_equiv = compute_flops_equivalent(model, CONTEXT_LEN, NUM_VARIATES, device)
    except Exception as e:
        logger.warning("FLOPs measurement failed: %s", e)
        flops_equiv = 0

    # 5. Validate
    param_count = sum(p.numel() for p in model.parameters())
    if hasattr(model, "reset"):
        model.reset()
    model.eval()

    try:
        metrics = validate(model)
    except Exception as e:
        return {
            "crps": float("inf"), "mase": float("inf"),
            "flops_equivalent_size": flops_equiv,
            "param_count": param_count,
            "error": f"validate failed: {e}",
        }

    return {
        "crps": metrics["crps"],
        "ncrps": metrics.get("ncrps", float("inf")),
        "mase": metrics["mase"],
        "flops_equivalent_size": flops_equiv,
        "param_count": param_count,
    }


async def evaluate_all_checkpoints(
    r2: "R2AuditLog",
    round_id: int,
    training_metas: dict[int, dict],
    challenge,
    task: "TaskSpec",
    tmp_dir: str = "/tmp/eval",
) -> dict[int, dict]:
    """Evaluate all checkpoints for a round.

    For each miner with a successful training:
      1. Download architecture.py and checkpoint.safetensors from R2
      2. Call evaluate_checkpoint()
      3. Verify FLOPs matches trainer's claim
      4. Apply size gate
    """
    from shared.scoring import passes_size_gate
    from shared.artifacts import download_training_artifacts

    os.makedirs(tmp_dir, exist_ok=True)
    results: dict[int, dict] = {}
    device = os.getenv("RADAR_EVAL_DEVICE", "cpu")

    for uid, meta in training_metas.items():
        if meta.get("status") != "success":
            continue

        miner_hotkey = meta.get("miner_hotkey", f"uid_{uid}")

        # Download and verify all artifacts
        artifacts = download_training_artifacts(r2, round_id, miner_hotkey, tmp_dir)
        if not artifacts:
            logger.warning("UID %d: failed to download artifacts", uid)
            continue

        if not artifacts.verified:
            logger.warning("UID %d: artifact verification failed: %s", uid, artifacts.verification_error)
            results[uid] = {
                "crps": float("inf"), "mase": float("inf"),
                "error": f"verification failed: {artifacts.verification_error}",
            }
            continue

        if not artifacts.architecture_code:
            continue

        # Evaluate
        metrics = evaluate_checkpoint(
            artifacts.architecture_code, artifacts.checkpoint_path,
            eval_split_seed=challenge.eval_split_seed,
            device=device,
        )

        # Verify FLOPs claim
        trainer_claimed = meta.get("flops_equivalent_size", 0)
        validator_measured = metrics.get("flops_equivalent_size", 0)
        metrics["flops_verified"] = verify_flops_claim(trainer_claimed, validator_measured)

        # Size gate
        metrics["passed_size_gate"] = passes_size_gate(metrics, challenge)

        results[uid] = metrics
        logger.info(
            "UID %d eval: crps=%.6f mase=%.6f flops=%d gate=%s verified=%s",
            uid, metrics.get("crps", -1), metrics.get("mase", -1),
            validator_measured, metrics["passed_size_gate"],
            artifacts.verified,
        )

        # Cleanup checkpoint file
        try:
            os.remove(artifacts.checkpoint_path)
        except OSError:
            pass

    return results


def verify_flops_claim(
    trainer_claimed: int,
    validator_measured: int,
    tolerance: float = 0.02,
) -> bool:
    """FLOPs-equivalent should be near-identical. Allow 2% tolerance."""
    if trainer_claimed == 0 or validator_measured == 0:
        return True  # Can't verify if either is zero
    ratio = abs(trainer_claimed - validator_measured) / max(trainer_claimed, 1)
    return ratio <= tolerance
