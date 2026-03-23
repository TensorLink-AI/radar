"""Tests for shared.dedup — code similarity and parent identity checking."""

from shared.dedup import code_similarity, is_unchanged_from_parent


def test_identical_code():
    code = "import torch\nclass Model(nn.Module):\n    pass"
    assert code_similarity(code, code) == 1.0


def test_different_code():
    a = "import torch\nclass Model(nn.Module):\n    def forward(self, x):\n        return self.linear(x)"
    b = "import numpy as np\ndef compute(data):\n    return np.mean(data) * 2.0"
    sim = code_similarity(a, b)
    assert sim < 0.5


def test_empty_code():
    assert code_similarity("", "") == 0.0
    assert code_similarity("import torch", "") == 0.0


def test_unchanged_from_parent_identical():
    code = "import torch\nmodel = nn.Linear(10, 10)"
    assert is_unchanged_from_parent(code, code) is True


def test_unchanged_from_parent_trivial_change():
    """Whitespace-only or comment-only changes should still be caught."""
    parent = "import torch\nmodel = nn.Linear(10, 10)"
    child = "import torch\nmodel = nn.Linear(10, 10)  # added comment"
    assert is_unchanged_from_parent(child, parent) is True


def test_changed_from_parent():
    """Real code changes should pass."""
    parent = "import torch\nmodel = nn.Linear(10, 10)"
    child = "import torch\nmodel = nn.Linear(10, 20)\noptimizer = torch.optim.Adam(model.parameters())"
    assert is_unchanged_from_parent(child, parent) is False


def test_unchanged_empty():
    """Empty inputs should not be flagged."""
    assert is_unchanged_from_parent("", "") is False
    assert is_unchanged_from_parent("import torch", "") is False
