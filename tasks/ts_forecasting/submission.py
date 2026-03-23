# Experiment: seed_baseline
"""
Minimal time-series forecasting submission (seed).

Implements a simple linear forecaster. The GEPA-Research agent will evolve
this file to improve CRPS on held-out forecasting tasks.

Required interface:
  - build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module
  - build_optimizer(model) -> optimizer

Model input:  (batch, context_len, num_variates) float tensor
Model output: (batch, prediction_len, num_variates, num_quantiles) float tensor
"""

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
        out = self.fc(x.mean(dim=2))
        return out.view(B, self.pred_len, self.n_q).unsqueeze(2).expand(-1, -1, V, -1)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return TinyForecaster(context_len, prediction_len, num_variates, len(quantiles))


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)
