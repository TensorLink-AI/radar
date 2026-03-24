"""Tests for shared.dedup — code similarity for provenance queries."""

from shared.dedup import code_similarity


def test_identical_code():
    code = "import torch\nclass Model(nn.Module):\n    pass"
    assert code_similarity(code, code) == 1.0


def test_different_code():
    a = "import torch\nclass Model(nn.Module):\n    def forward(self, x):\n        return self.linear(x)"
    b = "import numpy as np\ndef compute(data):\n    return np.mean(data) * 2.0"
    sim = code_similarity(a, b)
    assert sim < 0.5


def test_empty_code_similarity():
    assert code_similarity("", "") == 0.0
    assert code_similarity("import torch", "") == 0.0
