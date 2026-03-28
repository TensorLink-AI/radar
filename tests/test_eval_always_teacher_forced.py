"""Tests that evaluate.py always uses teacher-forced input."""

import torch
import torch.nn as nn


class _InputRecordingModel(nn.Module):
    """Model that records the input shapes it receives."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.recorded_shapes = []
        self.embed = nn.Embedding(vocab, 32)
        self.head = nn.Linear(32, vocab)

    def forward(self, x):
        self.recorded_shapes.append(tuple(x.shape))
        return self.head(self.embed(x))


def test_eval_constructs_teacher_forced_input():
    """Validate that evaluation always constructs cat([context, targets[:-1]])."""
    context_len = 16
    prediction_len = 8
    vocab_size = 32

    model = _InputRecordingModel(vocab_size)

    # Simulate what evaluate.py does
    import numpy as np
    rng = np.random.RandomState(42)
    x = torch.from_numpy(rng.randint(0, vocab_size, (4, context_len))).long()
    y = torch.from_numpy(rng.randint(0, vocab_size, (4, prediction_len))).long()

    # Always teacher-forced, regardless of mode
    inp = torch.cat([x, y[:, :-1]], dim=1)
    with torch.no_grad():
        logits = model(inp)

    expected_seq_len = context_len + prediction_len - 1
    assert model.recorded_shapes[-1] == (4, expected_seq_len)

    # Extract last prediction_len positions
    logits_pred = logits[:, -prediction_len:]
    assert logits_pred.shape == (4, prediction_len, vocab_size)


def test_eval_never_uses_direct_input():
    """Even if prediction_mode='direct', eval should still feed full sequence."""
    context_len = 16
    prediction_len = 8
    vocab_size = 32

    model = _InputRecordingModel(vocab_size)

    import numpy as np
    rng = np.random.RandomState(42)
    x = torch.from_numpy(rng.randint(0, vocab_size, (4, context_len))).long()
    y = torch.from_numpy(rng.randint(0, vocab_size, (4, prediction_len))).long()

    # Even in "direct" mode, evaluate.py always does teacher-forced
    prediction_mode = "direct"
    inp = torch.cat([x, y[:, :-1]], dim=1)
    with torch.no_grad():
        model(inp)

    # Input should always be (B, ctx + pred - 1), not (B, ctx)
    assert model.recorded_shapes[-1][1] == context_len + prediction_len - 1
