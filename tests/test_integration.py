"""Integration test — verify the 3-phase round data flow without Bittensor.

Tests the data structures and scoring pipeline end-to-end:
1. Generate a deterministic challenge
2. Create mock submissions
3. Simulate Phase C evaluation results
4. Score the round
5. Update DB and Pareto front
"""

import tempfile

from shared.challenge import generate_challenge, round_start_block, current_phase
from shared.database import DataElement, ExperimentDB
from shared.dedup import code_similarity
from shared.pareto import ParetoFront
from shared.protocol import Challenge, Proposal
from shared.scoring import score_round, scores_to_weights, ema_update, passes_size_gate
from shared.task import ml_training_task, TaskSpec
from validator.analyzer import analyze


def _local_task() -> TaskSpec:
    task = ml_training_task()
    task.target_file = "submission.py"
    return task


def test_end_to_end_phase_c_scoring():
    """Full round: challenge -> eval results -> scoring -> DB update."""
    task = _local_task()

    # Create DB with seed experiments
    tmpdir = tempfile.mkdtemp()
    db = ExperimentDB(db_dir=tmpdir)
    db.add(DataElement(
        name="seed_0", code="seed code v0",
        success=True, metric=0.85,
        objectives={"crps": 0.85, "mase": 1.0, "flops_equivalent_size": 200_000},
        generation=0,
    ))

    # Build Pareto front
    objective_fn = lambda elem: task.objective_vector(elem.objectives)
    pareto = ParetoFront(max_size=50, objective_fn=objective_fn)
    for elem in db.get_pareto_elements():
        pareto.update(elem)

    # Generate challenge
    block_hash = "a" * 64
    challenge = generate_challenge(block_hash, task.to_dict())
    assert challenge.round_id > 0
    assert challenge.min_flops_equivalent > 0

    # Mock submissions
    submissions = {
        0: Proposal(code="import torch\ndef build_model(): pass\ndef build_optimizer(): pass", name="arch_0"),
        1: Proposal(code="import torch\nclass Model: pass\ndef build_model(): pass\ndef build_optimizer(): pass", name="arch_1"),
    }

    # Mock Phase C eval results
    eval_results = {
        0: {
            "crps": 0.80, "mase": 0.95,
            "flops_equivalent_size": challenge.min_flops_equivalent + 1000,
            "param_count": 1_000_000,
            "passed_size_gate": True,
            "flops_verified": True,
        },
        1: {
            "crps": 0.75, "mase": 0.90,
            "flops_equivalent_size": challenge.min_flops_equivalent + 2000,
            "param_count": 2_000_000,
            "passed_size_gate": True,
            "flops_verified": True,
        },
    }

    # Score
    round_scores = score_round(
        eval_results, challenge, pareto, task.objectives, {},
    )
    assert len(round_scores) == 2
    # UID 1 has better CRPS -> should score higher
    assert round_scores[1] >= round_scores[0]

    # Weights
    uids, weights = scores_to_weights(round_scores, temperature=0.1)
    assert abs(sum(weights) - 1.0) < 1e-6

    # Update DB
    for uid, metrics in eval_results.items():
        element = DataElement(
            name=f"round_test_miner_{uid}",
            code=submissions[uid].code,
            metric=metrics["crps"],
            success=True,
            objectives=metrics,
            miner_uid=uid,
            parent=0,
            generation=1,
        )
        db.add(element)
        pareto.update(element)

    assert db.size == 3


def test_analyzer_success():
    parent = DataElement(
        name="parent", metric=0.9, success=True,
        objectives={"exec_time": 200, "memory_mb": 30000, "num_steps": 1000},
    )
    result = {
        "success": True, "metric": 0.85, "exec_time": 190,
        "objectives": {"exec_time": 190, "memory_mb": 29000, "num_steps": 1100},
        "loss_curve": [1.0, 0.9, 0.87, 0.85],
    }
    text = analyze(result, parent)
    assert len(text) > 10
    assert "Improved" in text or "improved" in text.lower()


def test_analyzer_failure():
    parent = DataElement(name="parent", metric=0.9, success=True)
    result = {"success": False, "trace": "SyntaxError: invalid syntax", "return_code": 1}
    text = analyze(result, parent)
    assert "failed" in text.lower() or "Failed" in text


def test_task_config_submission_format():
    task = ml_training_task()
    assert task.target_file == "submission.py"
    assert "harness.py" in task.run_command
    assert task.eval_command != ""
    assert "evaluate.py" in task.eval_command


def test_task_config_has_eval_command():
    task = ml_training_task()
    assert hasattr(task, "eval_command")
    d = task.to_dict()
    assert "eval_command" in d
    restored = TaskSpec.from_dict(d)
    assert restored.eval_command == task.eval_command


def test_seed_in_challenge():
    challenge = Challenge(
        challenge_id="test", seed=12345,
    )
    j = challenge.to_json()
    restored = Challenge.from_json(j)
    assert restored.seed == 12345


def test_database_new_fields():
    elem = DataElement(
        name="test",
        manifest_sha256="abc123",
        generated_samples=[[1, 2, 3]],
    )
    d = elem.to_dict()
    assert d["manifest_sha256"] == "abc123"
    assert d["generated_samples"] == [[1, 2, 3]]
    restored = DataElement.from_dict(d)
    assert restored.manifest_sha256 == "abc123"
    assert restored.generated_samples == [[1, 2, 3]]
