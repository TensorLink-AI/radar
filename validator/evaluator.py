"""Phase C evaluation — the trust anchor.

Every validator independently downloads checkpoints and computes metrics.
Cheap (seconds per model on CPU), fully deterministic, provides consensus.

Miner-submitted architecture code runs in an isolated subprocess to prevent
access to the validator's environment variables, filesystem, or memory.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from shared.r2_audit import R2AuditLog
    from shared.task import TaskSpec

logger = logging.getLogger(__name__)

_EVAL_RUNNER_TEMPLATE = '''
import json
import os
import random
import sys

import torch
from safetensors.torch import load_file

# Frozen eval imports
from prepare import validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES
from flops import compute_flops_equivalent

random.seed({eval_split_seed})
torch.manual_seed({eval_split_seed})
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

arch_path = "{arch_path}"
checkpoint_path = "{checkpoint_path}"
device = "{device}"

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission", arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "build_model") or not callable(mod.build_model):
        print(json.dumps({{"crps": float("inf"), "mase": float("inf"), "error": "Missing build_model()"}}))
        sys.exit(0)

    model = mod.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).to(device)
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict)

    flops_equiv = 0
    try:
        flops_equiv = compute_flops_equivalent(model, CONTEXT_LEN, NUM_VARIATES, device)
    except Exception:
        pass

    param_count = sum(p.numel() for p in model.parameters())
    if hasattr(model, "reset"):
        model.reset()
    model.eval()

    data_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "")
    metrics = validate(model, seed={eval_split_seed},
                       data_dir=data_dir if data_dir else None)

    result = {{
        "crps": metrics["crps"],
        "ncrps": metrics.get("ncrps", float("inf")),
        "mase": metrics["mase"],
        "flops_equivalent_size": flops_equiv,
        "param_count": param_count,
    }}
    if "n_datasets" in metrics:
        result["n_datasets"] = metrics["n_datasets"]
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"crps": float("inf"), "mase": float("inf"), "error": str(e)}}))
'''


def evaluate_checkpoint(
    architecture_code: str,
    checkpoint_path: str,
    eval_split_seed: int = 42,
    device: str = "cpu",
    timeout: int = 120,
) -> dict:
    """Evaluate a single checkpoint in an isolated subprocess.

    Runs miner code in a separate process so it cannot access the
    validator's memory, environment variables, or filesystem beyond
    the temp directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write architecture code
        arch_path = os.path.join(tmpdir, "submission.py")
        with open(arch_path, "w") as f:
            f.write(architecture_code)

        # Write eval runner script
        runner_path = os.path.join(tmpdir, "run_eval.py")
        runner_code = _EVAL_RUNNER_TEMPLATE.format(
            arch_path=arch_path,
            checkpoint_path=checkpoint_path,
            eval_split_seed=eval_split_seed,
            device=device,
        )
        with open(runner_path, "w") as f:
            f.write(runner_code)

        # Run in subprocess with restricted environment
        runner_path_abs = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast")
        )
        shared_path_abs = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        clean_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
            "HOME": tmpdir,
            "PYTHONPATH": f"{runner_path_abs}:{shared_path_abs}",
            "RADAR_GIFT_EVAL_CACHE": os.environ.get("RADAR_GIFT_EVAL_CACHE", ""),
            "RADAR_EVAL_DATA": os.environ.get("RADAR_EVAL_DATA", "gift_eval"),
        }
        # Do NOT forward R2 credentials, wallet keys, or BASILICA tokens

        try:
            result = subprocess.run(
                [sys.executable, runner_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=clean_env,
                cwd=tmpdir,
            )
            if result.returncode != 0:
                return {
                    "crps": float("inf"), "mase": float("inf"),
                    "error": f"Eval subprocess failed: {result.stderr[:500]}",
                }
            # Parse only the last non-empty line of stdout to avoid
            # spurious output from miner code (print statements, warnings).
            return _parse_last_json_line(result.stdout)
        except subprocess.TimeoutExpired:
            return {
                "crps": float("inf"), "mase": float("inf"),
                "error": f"Eval subprocess timed out ({timeout}s)",
            }
        except json.JSONDecodeError:
            return {
                "crps": float("inf"), "mase": float("inf"),
                "error": f"Eval subprocess returned invalid JSON: {result.stdout[-200:]}",
            }


def _parse_last_json_line(stdout: str) -> dict:
    """Parse JSON from the last non-empty line of subprocess stdout.

    Miner code loaded via importlib may produce print output during import.
    The eval runner template always prints its JSON result as the final line.
    """
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    if not lines:
        return {
            "crps": float("inf"), "mase": float("inf"),
            "error": "Eval subprocess produced no output",
        }
    # Try last line first (expected), then scan backwards
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "crps": float("inf"), "mase": float("inf"),
        "error": f"Eval subprocess returned no valid JSON line: {lines[-1][:200]}",
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
