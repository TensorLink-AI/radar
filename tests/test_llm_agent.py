"""Tests for the LLM-powered miner agent (miner_template/agent.py)."""

import ast
import json
import os
import sys
import tempfile
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from miner_template.agent import (
    build_system_prompt,
    build_user_prompt,
    extract_code_block,
    gather_db_context,
    load_history,
    save_history,
    validate_code,
)


# ---------------------------------------------------------------------------
# Task fixtures
# ---------------------------------------------------------------------------
TS_TASK = {
    "name": "ts_forecasting",
    "description": "Train a time series model to minimize CRPS.",
    "run_command": "python /workspace/frozen/harness.py {target}",
    "time_budget": 300,
    "objectives": [
        {"name": "crps", "lower_is_better": True, "weight": 1.0, "primary": True},
        {"name": "mase", "lower_is_better": True, "weight": 0.5},
    ],
    "domain_system_prompt": "You are a time-series expert.",
    "constraints": ["Model input: (batch, context_len, num_variates)"],
    "anti_patterns": ["Tweaking a single hyperparameter"],
    "example_hypotheses": ["PatchTST-style patching improves efficiency"],
}

NANOGPT_TASK = {
    "name": "nanogpt",
    "description": "Optimize a nanoGPT-style LM training script.",
    "run_command": "python {target}",
    "time_budget": 300,
    "objectives": [
        {"name": "val_loss", "lower_is_better": True, "weight": 1.0, "primary": True},
        {"name": "iter_per_sec", "lower_is_better": False, "weight": 0.4},
    ],
    "domain_system_prompt": "You are an expert LM researcher.",
    "constraints": ["Code must run on a single GPU"],
    "anti_patterns": ["Breaking the evaluation protocol"],
    "example_hypotheses": ["SwiGLU activation in MLP improves expressiveness"],
}


@pytest.fixture
def ts_challenge():
    return {
        "challenge_id": "test-123",
        "seed": 42,
        "round_id": 7,
        "min_flops_equivalent": 2_000_000,
        "max_flops_equivalent": 10_000_000,
        "eval_split_seed": 99,
        "task": TS_TASK,
        "db_url": "",
        "desearch_url": "",
        "feasible_frontier": [],
        "scratchpad_get_url": "",
        "scratchpad_put_url": "",
        "scratchpad_max_mb": 10,
    }


@pytest.fixture
def nanogpt_challenge():
    return {
        "challenge_id": "test-456",
        "seed": 99,
        "round_id": 12,
        "min_flops_equivalent": 10_000_000,
        "max_flops_equivalent": 50_000_000,
        "eval_split_seed": 77,
        "task": NANOGPT_TASK,
        "db_url": "",
        "desearch_url": "",
        "feasible_frontier": [],
        "scratchpad_get_url": "",
        "scratchpad_put_url": "",
        "scratchpad_max_mb": 10,
    }


@pytest.fixture
def ts_challenge_with_frontier(ts_challenge):
    ts_challenge["feasible_frontier"] = [
        {
            "code": "import torch\nclass Foo(nn.Module): pass",
            "metric": 0.42,
            "objectives": {"crps": 0.42, "mase": 0.55},
        },
        {
            "code": "import torch\nclass Bar(nn.Module): pass",
            "metric": 0.50,
            "objectives": {"crps": 0.50, "mase": 0.60},
        },
    ]
    return ts_challenge


# ---------------------------------------------------------------------------
# validate_code — harness-based tasks (ts_forecasting)
# ---------------------------------------------------------------------------
class TestValidateCodeHarness:
    def test_valid_minimal(self):
        code = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

def build_model(context_len, prediction_len, num_variates, quantiles):
    return M()

def build_optimizer(model):
    return torch.optim.Adam(model.parameters())
"""
        ok, msg = validate_code(code, TS_TASK)
        assert ok, msg

    def test_missing_build_model(self):
        code = "def build_optimizer(model): pass"
        ok, msg = validate_code(code, TS_TASK)
        assert not ok
        assert "build_model" in msg

    def test_missing_build_optimizer(self):
        code = "def build_model(context_len, prediction_len, num_variates, quantiles): pass"
        ok, msg = validate_code(code, TS_TASK)
        assert not ok
        assert "build_optimizer" in msg

    def test_syntax_error(self):
        ok, msg = validate_code("def foo(:\n  pass", TS_TASK)
        assert not ok
        assert "Syntax" in msg

    def test_forbidden_import(self):
        code = """
import subprocess
def build_model(context_len, prediction_len, num_variates, quantiles): pass
def build_optimizer(model): pass
"""
        ok, msg = validate_code(code, TS_TASK)
        assert not ok
        assert "subprocess" in msg

    def test_forbidden_from_import(self):
        code = """
from socket import socket
def build_model(context_len, prediction_len, num_variates, quantiles): pass
def build_optimizer(model): pass
"""
        ok, msg = validate_code(code, TS_TASK)
        assert not ok
        assert "socket" in msg

    def test_with_optional_hooks(self):
        code = """
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

def build_model(context_len, prediction_len, num_variates, quantiles): return M()
def build_optimizer(model): return torch.optim.Adam(model.parameters())
def configure_amp(): return {"enabled": True, "dtype": "bfloat16"}
def training_config(): return {"batch_size": 64}
def init_weights(model): pass
"""
        ok, msg = validate_code(code, TS_TASK)
        assert ok, msg


# ---------------------------------------------------------------------------
# validate_code — standalone tasks (nanogpt)
# ---------------------------------------------------------------------------
class TestValidateCodeStandalone:
    def test_no_build_model_required(self):
        """Standalone tasks don't need build_model/build_optimizer."""
        code = """
import torch
model = torch.nn.Linear(10, 10)
print("val_loss: 1.23")
"""
        ok, msg = validate_code(code, NANOGPT_TASK)
        assert ok, msg

    def test_syntax_error_still_caught(self):
        ok, msg = validate_code("def foo(:", NANOGPT_TASK)
        assert not ok

    def test_forbidden_import_still_caught(self):
        code = "import subprocess\nprint('hello')"
        ok, msg = validate_code(code, NANOGPT_TASK)
        assert not ok


# ---------------------------------------------------------------------------
# extract_code_block
# ---------------------------------------------------------------------------
class TestExtractCodeBlock:
    def test_python_fence(self):
        resp = "Here:\n```python\nimport torch\nprint('hi')\n```\nDone."
        code = extract_code_block(resp)
        assert "import torch" in code
        assert "```" not in code

    def test_plain_fence(self):
        code = extract_code_block("```\nimport torch\n```")
        assert "import torch" in code

    def test_no_fence(self):
        code = extract_code_block("import torch\nprint('hi')")
        assert "import torch" in code

    def test_multiple_blocks_takes_first(self):
        resp = "```python\nfirst\n```\ntext\n```python\nsecond\n```"
        assert "first" in extract_code_block(resp)


# ---------------------------------------------------------------------------
# build_system_prompt — task-aware
# ---------------------------------------------------------------------------
class TestBuildSystemPrompt:
    def test_harness_task_includes_build_model(self):
        prompt = build_system_prompt(TS_TASK)
        assert "build_model" in prompt
        assert "build_optimizer" in prompt
        assert "harness-based" in prompt

    def test_standalone_task_no_build_model(self):
        prompt = build_system_prompt(NANOGPT_TASK)
        assert "standalone" in prompt.lower()
        assert "build_model" not in prompt

    def test_includes_domain_prompt(self):
        prompt = build_system_prompt(TS_TASK)
        assert "time-series expert" in prompt

    def test_includes_objectives(self):
        prompt = build_system_prompt(TS_TASK)
        assert "crps" in prompt
        assert "PRIMARY" in prompt

    def test_includes_constraints(self):
        prompt = build_system_prompt(TS_TASK)
        assert "context_len" in prompt

    def test_includes_anti_patterns(self):
        prompt = build_system_prompt(TS_TASK)
        assert "hyperparameter" in prompt

    def test_includes_hypotheses(self):
        prompt = build_system_prompt(TS_TASK)
        assert "PatchTST" in prompt

    def test_empty_task(self):
        """Empty task still produces a usable prompt."""
        prompt = build_system_prompt({})
        assert "expert" in prompt.lower()
        assert "Python code block" in prompt


# ---------------------------------------------------------------------------
# build_user_prompt — task-aware
# ---------------------------------------------------------------------------
class TestBuildUserPrompt:
    def test_includes_task_name(self, ts_challenge):
        prompt = build_user_prompt(ts_challenge, {}, [])
        assert "ts_forecasting" in prompt

    def test_includes_flops(self, ts_challenge):
        prompt = build_user_prompt(ts_challenge, {}, [])
        assert "2,000,000" in prompt

    def test_no_frontier_message(self, ts_challenge):
        prompt = build_user_prompt(ts_challenge, {}, [])
        assert "No frontier" in prompt

    def test_with_frontier(self, ts_challenge_with_frontier):
        prompt = build_user_prompt(ts_challenge_with_frontier, {}, [])
        assert "Frontier" in prompt
        assert "0.42" in prompt

    def test_nanogpt_task(self, nanogpt_challenge):
        prompt = build_user_prompt(nanogpt_challenge, {}, [])
        assert "nanogpt" in prompt
        assert "50,000,000" in prompt

    def test_with_db_context(self, ts_challenge):
        db_ctx = {
            "recent_experiments": [
                {"name": "exp1", "metric": 0.3, "success": True, "flops": 5_000_000},
            ],
            "dead_ends": ["pure MLP without attention"],
        }
        prompt = build_user_prompt(ts_challenge, db_ctx, [])
        assert "exp1" in prompt
        assert "Dead Ends" in prompt

    def test_with_history(self, ts_challenge):
        history = [{"round": 5, "name": "my_arch", "metric": 0.35}]
        prompt = build_user_prompt(ts_challenge, {}, history)
        assert "my_arch" in prompt


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------
class TestHistory:
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            history = [{"round": 1, "name": "a"}, {"round": 2, "name": "b"}]
            save_history(d, history)
            assert load_history(d) == history

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            assert load_history(d) == []

    def test_truncates_to_50(self):
        with tempfile.TemporaryDirectory() as d:
            history = [{"round": i} for i in range(100)]
            save_history(d, history)
            loaded = load_history(d)
            assert len(loaded) == 50
            assert loaded[0]["round"] == 50


# ---------------------------------------------------------------------------
# gather_db_context
# ---------------------------------------------------------------------------
class TestGatherDbContext:
    def test_no_db_url(self, ts_challenge):
        assert gather_db_context(ts_challenge) == {}

    def test_db_errors_gracefully(self, ts_challenge):
        ts_challenge["db_url"] = "http://localhost:99999"
        ctx = gather_db_context(ts_challenge)
        assert isinstance(ctx, dict)


# ---------------------------------------------------------------------------
# Integration: full agent flow (mocked LLM)
# ---------------------------------------------------------------------------
class TestDesignArchitecture:
    def test_llm_success_harness_task(self, ts_challenge):
        """LLM returns valid harness code for ts_forecasting."""
        from miner_template.agent import design_architecture

        valid_response = '''```python
import torch
import torch.nn as nn

class LLMArch(nn.Module):
    def __init__(self, ctx, pred, var, q):
        super().__init__()
        self.fc = nn.Linear(ctx, pred * var * len(q))
        self.pred = pred
        self.var = var
        self.nq = len(q)

    def forward(self, x):
        B = x.shape[0]
        out = self.fc(x.mean(dim=2))
        return out.view(B, self.pred, self.var, self.nq)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return LLMArch(context_len, prediction_len, num_variates, quantiles)

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
```'''
        with mock.patch("miner_template.agent.CHUTES_API_KEY", "test-key"):
            with mock.patch("miner_template.agent.llm_chat", return_value=valid_response):
                with mock.patch("miner_template.agent.load_scratchpad") as lsp:
                    with mock.patch("miner_template.agent.save_scratchpad"):
                        lsp.return_value = tempfile.mkdtemp()
                        result = design_architecture(ts_challenge)

        assert result["code"]
        assert "LLMArch" in result["name"]
        ok, msg = validate_code(result["code"], TS_TASK)
        assert ok, msg

    def test_llm_success_standalone_task(self, nanogpt_challenge):
        """LLM returns valid standalone code for nanogpt."""
        from miner_template.agent import design_architecture

        valid_response = '''```python
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        return self.fc(x)

model = MiniGPT()
print("val_loss: 1.23")
print("training_seconds: 42.0")
```'''
        with mock.patch("miner_template.agent.CHUTES_API_KEY", "test-key"):
            with mock.patch("miner_template.agent.llm_chat", return_value=valid_response):
                with mock.patch("miner_template.agent.load_scratchpad") as lsp:
                    with mock.patch("miner_template.agent.save_scratchpad"):
                        lsp.return_value = tempfile.mkdtemp()
                        result = design_architecture(nanogpt_challenge)

        assert result["code"]
        assert "MiniGPT" in result["name"]
        ok, msg = validate_code(result["code"], NANOGPT_TASK)
        assert ok, msg

    def test_llm_failure_returns_empty(self, ts_challenge):
        """If LLM fails entirely, return empty code (no fallback)."""
        from miner_template.agent import design_architecture

        with mock.patch("miner_template.agent.CHUTES_API_KEY", "test-key"):
            with mock.patch(
                "miner_template.agent.llm_chat",
                side_effect=RuntimeError("API down"),
            ):
                with mock.patch("miner_template.agent.load_scratchpad") as lsp:
                    with mock.patch("miner_template.agent.save_scratchpad"):
                        lsp.return_value = tempfile.mkdtemp()
                        result = design_architecture(ts_challenge)

        assert result["code"] == ""
        assert result["name"] == "failed"

    def test_llm_fix_on_validation_error(self, ts_challenge):
        """If first LLM response is invalid, agent asks for fix."""
        from miner_template.agent import design_architecture

        bad_response = "```python\nprint('no build_model')\n```"
        good_response = '''```python
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x
def build_model(context_len, prediction_len, num_variates, quantiles): return M()
def build_optimizer(model): return torch.optim.Adam(model.parameters())
```'''
        with mock.patch("miner_template.agent.CHUTES_API_KEY", "test-key"):
            with mock.patch(
                "miner_template.agent.llm_chat",
                side_effect=[bad_response, good_response],
            ):
                with mock.patch("miner_template.agent.load_scratchpad") as lsp:
                    with mock.patch("miner_template.agent.save_scratchpad"):
                        lsp.return_value = tempfile.mkdtemp()
                        result = design_architecture(ts_challenge)

        assert result["code"]
        ok, msg = validate_code(result["code"], TS_TASK)
        assert ok, msg

    def test_no_api_key_returns_empty(self, ts_challenge):
        """Without API key, agent can't do anything — returns empty."""
        from miner_template.agent import design_architecture

        with mock.patch("miner_template.agent.CHUTES_API_KEY", ""):
            with mock.patch("miner_template.agent.load_scratchpad") as lsp:
                with mock.patch("miner_template.agent.save_scratchpad"):
                    lsp.return_value = tempfile.mkdtemp()
                    result = design_architecture(ts_challenge)

        assert result["code"] == ""
        assert result["name"] == "failed"
