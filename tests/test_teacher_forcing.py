"""Tests for teacher forcing — verify correct input construction."""

import torch
import torch.nn as nn


class _MockModel(nn.Module):
    """Records input shapes for verification."""

    def __init__(self, vocab_size, prediction_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.prediction_len = prediction_len
        self.last_input_shape = None
        self.embed = nn.Embedding(vocab_size, 32)
        self.head = nn.Linear(32, vocab_size)

    def forward(self, x):
        self.last_input_shape = tuple(x.shape)
        h = self.embed(x)
        return self.head(h)


def test_teacher_forced_feeds_true_tokens():
    """In TF mode, model input is cat([context, true_targets[:-1]])."""
    context_len = 16
    prediction_len = 8
    vocab_size = 32

    model = _MockModel(vocab_size, prediction_len)
    x = torch.randint(0, vocab_size, (2, context_len))
    y = torch.randint(0, vocab_size, (2, prediction_len))

    # Teacher-forced input construction (same as harness)
    inp = torch.cat([x, y[:, :-1]], dim=1)
    logits = model(inp)

    assert model.last_input_shape == (2, context_len + prediction_len - 1)
    # Verify the input contains actual target tokens, not predictions
    assert torch.equal(inp[:, context_len:], y[:, :-1])


def test_direct_mode_no_target_tokens():
    """In direct mode, model only receives context."""
    context_len = 16
    prediction_len = 8
    vocab_size = 32

    model = _MockModel(vocab_size, prediction_len)
    x = torch.randint(0, vocab_size, (2, context_len))

    logits = model(x)
    assert model.last_input_shape == (2, context_len)


def test_teacher_forced_extracts_last_positions():
    """Harness extracts last prediction_len positions from TF output."""
    context_len = 16
    prediction_len = 8
    vocab_size = 32

    model = _MockModel(vocab_size, prediction_len)
    x = torch.randint(0, vocab_size, (2, context_len))
    y = torch.randint(0, vocab_size, (2, prediction_len))

    inp = torch.cat([x, y[:, :-1]], dim=1)
    logits = model(inp)  # (B, ctx+pred-1, vocab)

    logits_pred = logits[:, -prediction_len:]
    assert logits_pred.shape == (2, prediction_len, vocab_size)
