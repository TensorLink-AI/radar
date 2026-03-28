# Experiment: seed_baseline
"""
Minimal causal transformer for graph complexity task (seed).

Handles both direct and teacher-forced prediction modes:
  - Direct: input (B, context_len) -> output (B, prediction_len, vocab_size)
  - Teacher-forced: input (B, context_len + prediction_len - 1) -> output (B, seq_len, vocab_size)
    Harness extracts last prediction_len positions.

prediction_mode is passed from the challenge (env var PREDICTION_MODE).
vocab_size varies per round (256/1024/4096).
Domain losses (CRPS, MASE, MSE, SNR) computed automatically from logits.
"""

import math
import torch
import torch.nn as nn


class CausalTransformer(nn.Module):
    def __init__(self, context_len, prediction_len, vocab_size, prediction_mode):
        super().__init__()
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.vocab_size = vocab_size
        self.prediction_mode = prediction_mode

        d_model = 128
        nhead = 4
        num_layers = 2
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_len + prediction_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

        # Direct mode projection head
        if prediction_mode == "direct":
            self.direct_head = nn.Linear(d_model, prediction_len * vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)

        if self.prediction_mode == "direct" and T == self.context_len:
            # Direct: use last hidden state to predict all horizon steps
            out = self.direct_head(h[:, -1])  # (B, pred*vocab)
            return out.view(B, self.prediction_len, self.vocab_size)
        else:
            # Teacher-forced: produce logits at every position
            return self.head(h)  # (B, T, vocab_size)


def build_model(context_len, prediction_len, vocab_size, prediction_mode="direct"):
    return CausalTransformer(context_len, prediction_len, vocab_size, prediction_mode)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
