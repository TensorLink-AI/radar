"""Tests for the frozen training harness (runner/timeseries_forecast/harness.py)."""

import ast
import os
import sys


# ── Test that harness.py is valid Python ─────────────
def test_harness_syntax():
    """Harness file parses without syntax errors."""
    harness_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "harness.py"
    )
    with open(harness_path) as f:
        source = f.read()
    ast.parse(source)


# ── Test submission validation logic ─────────────────

MINIMAL_SUBMISSION = '''
import torch
import torch.nn as nn

class TinyForecaster(nn.Module):
    def __init__(self, ctx, pred, variates, n_q):
        super().__init__()
        self.pred_len = pred
        self.n_q = n_q
        self.fc = nn.Linear(ctx, pred * n_q)
    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, T)
        out = self.fc(x).reshape(B, V, self.pred_len, self.n_q)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return TinyForecaster(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)
'''

CUSTOM_LOSS_SUBMISSION = '''
import torch
import torch.nn as nn

class TinyForecaster(nn.Module):
    def __init__(self, ctx, pred, variates, n_q):
        super().__init__()
        self.pred_len = pred
        self.n_q = n_q
        self.fc = nn.Linear(ctx, pred * n_q)
    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, T)
        out = self.fc(x).reshape(B, V, self.pred_len, self.n_q)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return TinyForecaster(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)

def compute_loss(predictions, targets, quantiles):
    """Custom quantile loss."""
    errors = targets.unsqueeze(-1) - predictions
    q = torch.tensor(quantiles, device=predictions.device)
    return torch.mean(torch.max(q * errors, (q - 1) * errors))
'''

CONFIG_SUBMISSION = '''
import torch
import torch.nn as nn

class TinyForecaster(nn.Module):
    def __init__(self, ctx, pred, variates, n_q):
        super().__init__()
        self.pred_len = pred
        self.n_q = n_q
        self.fc = nn.Linear(ctx, pred * n_q)
    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, T)
        out = self.fc(x).reshape(B, V, self.pred_len, self.n_q)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return TinyForecaster(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)

def training_config():
    return {"batch_size": 128, "eval_interval": 100}
'''

COMPILE_SUBMISSION = '''
import torch
import torch.nn as nn

class TinyForecaster(nn.Module):
    def __init__(self, ctx, pred, variates, n_q):
        super().__init__()
        self.pred_len = pred
        self.n_q = n_q
        self.fc = nn.Linear(ctx, pred * n_q)
    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, T)
        out = self.fc(x).reshape(B, V, self.pred_len, self.n_q)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return TinyForecaster(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)

COMPILE = True
'''

BROKEN_SUBMISSION = '''
def build_optimizer(model):
    return None
# Missing build_model!
'''

WRONG_SHAPE_SUBMISSION = '''
import torch
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self, ctx, pred, variates, n_q):
        super().__init__()
        self.fc = nn.Linear(ctx, 10)  # Wrong output dim
    def forward(self, x):
        B, T, V = x.shape
        return self.fc(x.mean(dim=2))  # Wrong shape

def build_model(context_len, prediction_len, num_variates, quantiles):
    return BadModel(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)
'''

STATEFUL_SUBMISSION = '''
import torch
import torch.nn as nn

class StatefulModel(nn.Module):
    def __init__(self, ctx, pred, variates, n_q):
        super().__init__()
        self.pred_len = pred
        self.n_q = n_q
        self.fc = nn.Linear(ctx, pred * n_q)
        self._state = None

    def forward(self, x):
        B, T, V = x.shape
        self._state = x.mean().item()
        x = x.permute(0, 2, 1).reshape(B * V, T)
        out = self.fc(x).reshape(B, V, self.pred_len, self.n_q)
        return out.permute(0, 2, 1, 3)

    def reset(self):
        self._state = None

def build_model(context_len, prediction_len, num_variates, quantiles):
    return StatefulModel(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)
'''

GENERATE_SUBMISSION = '''
import torch
import torch.nn as nn

class TinyForecaster(nn.Module):
    def __init__(self, ctx, pred, variates, n_q):
        super().__init__()
        self.pred_len = pred
        self.n_q = n_q
        self.fc = nn.Linear(ctx, pred * n_q)
    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, T)
        out = self.fc(x).reshape(B, V, self.pred_len, self.n_q)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return TinyForecaster(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)

def build_scheduler(optimizer, total_steps):
    """Custom scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
'''


def test_minimal_submission_is_valid():
    """Minimal submission has required functions."""
    tree = ast.parse(MINIMAL_SUBMISSION)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build_model" in names
    assert "build_optimizer" in names


def test_custom_loss_submission_has_compute_loss():
    tree = ast.parse(CUSTOM_LOSS_SUBMISSION)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "compute_loss" in names


def test_config_submission_has_training_config():
    tree = ast.parse(CONFIG_SUBMISSION)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "training_config" in names


def test_compile_submission_has_flag():
    assert "COMPILE = True" in COMPILE_SUBMISSION


def test_broken_submission_missing_build_model():
    tree = ast.parse(BROKEN_SUBMISSION)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build_model" not in names


def test_stateful_submission_has_reset():
    tree = ast.parse(STATEFUL_SUBMISSION)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "reset" in names
    assert "build_model" in names


def test_scheduler_submission_has_build_scheduler():
    tree = ast.parse(GENERATE_SUBMISSION)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build_scheduler" in names
    assert "build_model" in names


def test_default_scheduler_not_in_minimal():
    """Minimal submission doesn't define build_scheduler — harness uses default."""
    tree = ast.parse(MINIMAL_SUBMISSION)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build_scheduler" not in names


def test_trailing_grad_accum_flush():
    """Harness code handles trailing grad accum."""
    harness_path = os.path.join(
        os.path.dirname(__file__), "..", "runner", "timeseries_forecast", "harness.py"
    )
    with open(harness_path) as f:
        source = f.read()
    # Check for the trailing flush logic
    assert "step % grad_accum != 0" in source
    assert "optimizer.step()" in source
