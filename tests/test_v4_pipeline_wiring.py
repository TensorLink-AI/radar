"""Tests for the claude_style_v4 wiring of the ts_data_pipeline task.

Focused on the contract surface — validation branching, smoke-test
contract checks, fallback round-trip, prompt builders. The full
designer subagent + LLM flow isn't exercised here; that's an
integration test the dev runs end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

V4 = Path(__file__).resolve().parent.parent / "miners" / "claude_style_v4"
sys.path.insert(0, str(V4))


@pytest.fixture
def pipeline_challenge():
    return {
        "task": {
            "name": "ts_data_pipeline",
            "task_params": {
                "context_len": 512,
                "prediction_len": 96,
                "num_variates": 1,
                "quantiles": [0.1, 0.5, 0.9],
            },
        },
        "frozen_arch": {
            "version": 3,
            "source_experiment_id": 42,
            "source_name": "patch_tx",
            "source_metric": 0.5,
            "source_flops": 1_500_000,
            "code": "def build_model(*a, **k): pass\n",
        },
    }


def test_validation_branches_on_pipeline_task(pipeline_challenge):
    from core.validation import _required_functions, validate_code
    req = _required_functions(pipeline_challenge)
    # build_pipeline is the only contract; no build_optimizer.
    assert set(req.keys()) == {"build_pipeline"}
    assert req["build_pipeline"] == [
        "context_len", "prediction_len", "num_variates", "quantiles",
    ]
    ok, errs = validate_code(
        "def build_pipeline(context_len, prediction_len, "
        "num_variates, quantiles):\n    return iter([])\n",
        pipeline_challenge,
    )
    assert ok, errs


def test_validation_rejects_arch_contract_on_pipeline(pipeline_challenge):
    from core.validation import validate_code
    ok, errs = validate_code("def build_model(**k): pass\n", pipeline_challenge)
    assert not ok
    assert any("build_pipeline" in e for e in errs)


def test_validation_keeps_arch_contract_for_other_tasks():
    from core.validation import _required_functions
    ch = {"task": {"name": "ts_forecasting",
                   "task_params": {"context_len": 512, "prediction_len": 96}}}
    req = _required_functions(ch)
    assert "build_model" in req and "build_optimizer" in req


def test_pipeline_designer_role_exposes_expected_tools():
    from tools import ROLE_TOOLS
    assert "pipeline_designer" in ROLE_TOOLS
    tools = ROLE_TOOLS["pipeline_designer"]
    # Critical tools the prompt instructs the LLM to call
    for name in ("read_frozen_arch", "pipeline_smoke_test",
                 "validate_code", "submit"):
        assert name in tools
    # Architecture-specific tools should NOT leak in
    for name in ("sketch_architecture", "estimate_layer_flops",
                 "size_to_flops"):
        assert name not in tools


def test_fallback_emits_valid_pipeline(pipeline_challenge):
    from core.fallback_templates import fallback_name_for, generate_fallback
    from core.validation import validate_code

    code = generate_fallback(pipeline_challenge)
    assert "def build_pipeline" in code
    ok, errs = validate_code(code, pipeline_challenge)
    assert ok, errs
    name = fallback_name_for(pipeline_challenge)
    assert name.startswith("pipeline_fallback_v3_")


def test_smoke_test_accepts_correct_pipeline(pipeline_challenge):
    pytest.importorskip("torch")
    from core.fallback_templates import generate_fallback
    from core.pipeline_probe import smoke_test_pipeline
    code = generate_fallback(pipeline_challenge)
    result = smoke_test_pipeline(code, pipeline_challenge)
    assert result["ok"], result
    assert result["n_batches"] == 2
    # Shape (B, context_len, num_variates) for input
    in_shape = result["shapes"][0]["input"]["shape"]
    assert in_shape[1] == 512 and in_shape[2] == 1


def test_smoke_test_rejects_wrong_rank(pipeline_challenge):
    pytest.importorskip("torch")
    from core.pipeline_probe import smoke_test_pipeline
    code = (
        "import torch\n"
        "def build_pipeline(context_len, prediction_len, "
        "num_variates, quantiles):\n"
        "    def _iter():\n"
        "        while True:\n"
        "            yield {\n"
        "                'input': torch.zeros(4, context_len),\n"
        "                'target': torch.zeros(4, prediction_len),\n"
        "            }\n"
        "    return _iter()\n"
    )
    result = smoke_test_pipeline(code, pipeline_challenge)
    assert not result["ok"]
    assert "rank" in result["error"]


def test_smoke_test_catches_exhaustion(pipeline_challenge):
    pytest.importorskip("torch")
    from core.pipeline_probe import smoke_test_pipeline
    code = (
        "import torch\n"
        "def build_pipeline(context_len, prediction_len, "
        "num_variates, quantiles):\n"
        "    yield {\n"
        "        'input': torch.zeros(4, context_len, num_variates),\n"
        "        'target': torch.zeros(4, prediction_len, num_variates),\n"
        "    }\n"
    )
    result = smoke_test_pipeline(code, pipeline_challenge)
    assert not result["ok"]
    assert "exhausted" in result["error"]


def test_smoke_test_rejects_signature_mismatch(pipeline_challenge):
    from core.pipeline_probe import smoke_test_pipeline
    code = (
        "def build_pipeline(a, b):\n"
        "    return iter([])\n"
    )
    result = smoke_test_pipeline(code, pipeline_challenge)
    assert not result["ok"]
    assert "task_params" in result["error"] or "rejected" in result["error"]


def test_pipeline_prompts_compile_and_stay_lean(pipeline_challenge):
    from prompts import (
        build_pipeline_designer_system_prompt,
        build_pipeline_designer_user_prompt,
    )
    sp = build_pipeline_designer_system_prompt(pipeline_challenge)
    up = build_pipeline_designer_user_prompt(pipeline_challenge, None)
    # System prompt explicitly names the scoring formula and the task
    # name; the user prompt advertises the frozen arch version.
    assert "ts_data_pipeline" in sp
    assert "AULC" in sp and "GIFT-Eval" in sp
    assert "v3" in up
    # Stay roughly under the size budget so context bloat is bounded.
    assert len(sp) < 8000, f"system prompt too large: {len(sp)}"
    assert len(up) < 1500, f"user prompt too large: {len(up)}"


def test_user_prompt_does_not_leak_frozen_arch_source(pipeline_challenge):
    """The user prompt must not paste the full frozen arch source —
    that's a tool call (read_frozen_arch include_source=True), and
    leaking it here is the main context-bloat trap we're avoiding.
    """
    from prompts import build_pipeline_designer_user_prompt
    pipeline_challenge["frozen_arch"]["code"] = (
        "# UNIQUE_SOURCE_SENTINEL_DO_NOT_PASTE\n" * 50
    )
    up = build_pipeline_designer_user_prompt(pipeline_challenge, None)
    assert "UNIQUE_SOURCE_SENTINEL_DO_NOT_PASTE" not in up
