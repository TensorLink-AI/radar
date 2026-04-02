"""Tests for the LLM-powered miner agent (miner_template/agent.py)."""

import ast
import json
import os
import sys
import tempfile
from unittest import mock

import pytest

# Add project root to path so we can import the agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from miner_template.agent import (
    build_user_prompt,
    extract_code_block,
    fallback_architecture,
    gather_db_context,
    load_history,
    save_history,
    validate_code,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def challenge():
    return {
        "challenge_id": "test-123",
        "seed": 42,
        "round_id": 7,
        "min_flops_equivalent": 2_000_000,
        "max_flops_equivalent": 10_000_000,
        "eval_split_seed": 99,
        "task": {"name": "ts_forecasting"},
        "db_url": "",
        "desearch_url": "",
        "feasible_frontier": [],
        "scratchpad_get_url": "",
        "scratchpad_put_url": "",
        "scratchpad_max_mb": 10,
    }


@pytest.fixture
def challenge_with_frontier(challenge):
    challenge["feasible_frontier"] = [
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
    return challenge


# ---------------------------------------------------------------------------
# validate_code
# ---------------------------------------------------------------------------
class TestValidateCode:
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
        ok, msg = validate_code(code)
        assert ok, msg

    def test_missing_build_model(self):
        code = """
def build_optimizer(model):
    pass
"""
        ok, msg = validate_code(code)
        assert not ok
        assert "build_model" in msg

    def test_missing_build_optimizer(self):
        code = """
def build_model(context_len, prediction_len, num_variates, quantiles):
    pass
"""
        ok, msg = validate_code(code)
        assert not ok
        assert "build_optimizer" in msg

    def test_syntax_error(self):
        code = "def foo(:\n  pass"
        ok, msg = validate_code(code)
        assert not ok
        assert "Syntax" in msg

    def test_forbidden_import(self):
        code = """
import subprocess
def build_model(context_len, prediction_len, num_variates, quantiles): pass
def build_optimizer(model): pass
"""
        ok, msg = validate_code(code)
        assert not ok
        assert "subprocess" in msg

    def test_forbidden_from_import(self):
        code = """
from socket import socket
def build_model(context_len, prediction_len, num_variates, quantiles): pass
def build_optimizer(model): pass
"""
        ok, msg = validate_code(code)
        assert not ok
        assert "socket" in msg

    def test_missing_param(self):
        code = """
def build_model(context_len, prediction_len):
    pass
def build_optimizer(model):
    pass
"""
        ok, msg = validate_code(code)
        assert not ok
        assert "missing required" in msg.lower() or "num_variates" in msg

    def test_with_optional_hooks(self):
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

def configure_amp():
    return {"enabled": True, "dtype": "bfloat16"}

def training_config():
    return {"batch_size": 64}

def init_weights(model):
    pass
"""
        ok, msg = validate_code(code)
        assert ok, msg


# ---------------------------------------------------------------------------
# extract_code_block
# ---------------------------------------------------------------------------
class TestExtractCodeBlock:
    def test_python_fence(self):
        resp = "Here is the code:\n```python\nimport torch\nprint('hi')\n```\nDone."
        code = extract_code_block(resp)
        assert "import torch" in code
        assert "```" not in code

    def test_plain_fence(self):
        resp = "```\nimport torch\n```"
        code = extract_code_block(resp)
        assert "import torch" in code

    def test_no_fence(self):
        resp = "import torch\nprint('hi')"
        code = extract_code_block(resp)
        assert "import torch" in code

    def test_multiple_blocks_takes_first(self):
        resp = "```python\nfirst_block\n```\ntext\n```python\nsecond_block\n```"
        code = extract_code_block(resp)
        assert "first_block" in code


# ---------------------------------------------------------------------------
# fallback_architecture
# ---------------------------------------------------------------------------
class TestFallbackArchitecture:
    @pytest.mark.parametrize("min_f,max_f", [
        (100_000, 500_000),
        (500_000, 2_000_000),
        (2_000_000, 10_000_000),
        (10_000_000, 50_000_000),
        (50_000_000, 125_000_000),
    ])
    def test_generates_valid_code(self, min_f, max_f):
        challenge = {
            "min_flops_equivalent": min_f,
            "max_flops_equivalent": max_f,
        }
        code = fallback_architecture(challenge)
        ok, msg = validate_code(code)
        assert ok, f"Fallback for [{min_f}, {max_f}] failed: {msg}"

    def test_code_parses_cleanly(self):
        challenge = {
            "min_flops_equivalent": 2_000_000,
            "max_flops_equivalent": 10_000_000,
        }
        code = fallback_architecture(challenge)
        tree = ast.parse(code)
        func_names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        assert "build_model" in func_names
        assert "build_optimizer" in func_names
        assert "build_scheduler" in func_names
        assert "configure_amp" in func_names
        assert "training_config" in func_names
        assert "init_weights" in func_names

    def test_contains_revin(self):
        challenge = {
            "min_flops_equivalent": 2_000_000,
            "max_flops_equivalent": 10_000_000,
        }
        code = fallback_architecture(challenge)
        assert "RevIN" in code


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------
class TestBuildUserPrompt:
    def test_basic_prompt(self, challenge):
        prompt = build_user_prompt(challenge, {}, [])
        assert "FLOPs budget" in prompt
        assert "2,000,000" in prompt
        assert "No frontier" in prompt

    def test_with_frontier(self, challenge_with_frontier):
        prompt = build_user_prompt(challenge_with_frontier, {}, [])
        assert "Frontier" in prompt
        assert "0.42" in prompt

    def test_with_db_context(self, challenge):
        db_ctx = {
            "recent_experiments": [
                {"name": "exp1", "metric": 0.3, "success": True, "flops": 5_000_000},
            ],
            "dead_ends": ["pure MLP without attention"],
        }
        prompt = build_user_prompt(challenge, db_ctx, [])
        assert "exp1" in prompt
        assert "Dead Ends" in prompt

    def test_with_history(self, challenge):
        history = [
            {"round": 5, "name": "my_arch", "metric": 0.35},
        ]
        prompt = build_user_prompt(challenge, {}, history)
        assert "my_arch" in prompt


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------
class TestHistory:
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            history = [{"round": 1, "name": "a"}, {"round": 2, "name": "b"}]
            save_history(d, history)
            loaded = load_history(d)
            assert loaded == history

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
# gather_db_context (with no DB)
# ---------------------------------------------------------------------------
class TestGatherDbContext:
    def test_no_db_url(self, challenge):
        ctx = gather_db_context(challenge)
        assert ctx == {}

    def test_db_errors_gracefully(self, challenge):
        challenge["db_url"] = "http://localhost:99999"
        ctx = gather_db_context(challenge)
        # Should not raise, returns partial results
        assert isinstance(ctx, dict)


# ---------------------------------------------------------------------------
# Integration: full agent flow (mocked LLM)
# ---------------------------------------------------------------------------
class TestDesignArchitecture:
    def test_fallback_when_no_api_key(self, challenge):
        """Without CHUTES_API_KEY, agent falls back to deterministic arch."""
        from miner_template.agent import design_architecture

        with mock.patch("miner_template.agent.CHUTES_API_KEY", ""):
            with mock.patch("miner_template.agent.load_scratchpad") as lsp:
                with mock.patch("miner_template.agent.save_scratchpad"):
                    lsp.return_value = tempfile.mkdtemp()
                    result = design_architecture(challenge)

        assert "code" in result
        assert "name" in result
        assert "motivation" in result
        ok, msg = validate_code(result["code"])
        assert ok, msg
        assert "fallback" in result["name"].lower()

    def test_llm_success(self, challenge):
        """With mocked LLM returning valid code, agent uses it."""
        from miner_template.agent import design_architecture

        valid_code = '''```python
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
            with mock.patch("miner_template.agent.llm_chat", return_value=valid_code):
                with mock.patch("miner_template.agent.load_scratchpad") as lsp:
                    with mock.patch("miner_template.agent.save_scratchpad"):
                        lsp.return_value = tempfile.mkdtemp()
                        result = design_architecture(challenge)

        assert "code" in result
        assert "LLMArch" in result["name"]
        ok, msg = validate_code(result["code"])
        assert ok, msg

    def test_llm_failure_falls_back(self, challenge):
        """If LLM returns invalid code, agent falls back."""
        from miner_template.agent import design_architecture

        with mock.patch("miner_template.agent.CHUTES_API_KEY", "test-key"):
            with mock.patch(
                "miner_template.agent.llm_chat",
                side_effect=RuntimeError("API down"),
            ):
                with mock.patch("miner_template.agent.load_scratchpad") as lsp:
                    with mock.patch("miner_template.agent.save_scratchpad"):
                        lsp.return_value = tempfile.mkdtemp()
                        result = design_architecture(challenge)

        assert "code" in result
        ok, msg = validate_code(result["code"])
        assert ok, msg
        assert "fallback" in result["name"].lower()
