"""Miner-side continuation helper (claude_style_v2 / v3 share one copy).

The two agents are standalone --agent_dir targets, so their ``core/`` is
duplicated. We load the v2 copy by file path and assert it stays
byte-identical to the v3 copy, then exercise the pure logic + the
``build_context`` fetch/degrade path with a fake gated client.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_V2 = _ROOT / "miners" / "claude_style_v2" / "core" / "continuation.py"
_V3 = _ROOT / "miners" / "claude_style_v3" / "core" / "continuation.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_cont_{path.parent.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cont():
    return _load(_V2)


# Deliberately uses a top-level helper class + imports that build_model
# depends on — the case a naive "extract build_model only" approach drops.
PARENT_CODE = '''
import torch
import torch.nn as nn

COMPILE = True

HIDDEN = 64


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return nn.Sequential(Block(HIDDEN), nn.Linear(HIDDEN, prediction_len))


def init_weights(model):
    for p in model.parameters():
        nn.init.normal_(p)


def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def training_config():
    return {"batch_size": 32}
'''


class FakeClient:
    """Stand-in for the agent's GatedClient (only get_json is used)."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []

    def get_json(self, url: str):
        self.calls.append(url)
        for suffix, body in self.routes.items():
            if url.endswith(suffix):
                if isinstance(body, Exception):
                    raise body
                return body
        raise AssertionError(f"unexpected url {url}")


def test_v2_v3_identical():
    assert _V2.read_text() == _V3.read_text()


def test_is_continuation_round(cont):
    assert cont.is_continuation_round(
        {"round_type": "continuation", "eligible_parents": [{"id": 1}]}
    )
    assert not cont.is_continuation_round(
        {"round_type": "new", "eligible_parents": [{"id": 1}]}
    )
    assert not cont.is_continuation_round(
        {"round_type": "continuation", "eligible_parents": []}
    )


def test_choose_parent_prefers_best_with_checkpoint(cont):
    parents = [
        {"id": 1, "metric": 0.5, "checkpoint_available": True, "cumulative_compute": 10},
        {"id": 2, "metric": 0.3, "checkpoint_available": True, "cumulative_compute": 20},
        {"id": 3, "metric": 0.1, "checkpoint_available": False},
    ]
    assert cont.choose_parent(parents)["id"] == 2
    assert cont.choose_parent([{"id": 9, "metric": 0.1, "checkpoint_available": False}]) is None
    assert cont.choose_parent([]) is None


def test_choose_parent_tie_breaks_on_compute(cont):
    parents = [
        {"id": 1, "metric": 0.3, "checkpoint_available": True, "cumulative_compute": 99},
        {"id": 2, "metric": 0.3, "checkpoint_available": True, "cumulative_compute": 5},
    ]
    # equal metric → prefer the one with less accumulated compute
    assert cont.choose_parent(parents)["id"] == 2


def test_extract_arch_source(cont):
    arch = cont.extract_arch_source(PARENT_CODE)
    assert "def build_model" in arch
    assert "def init_weights" in arch
    assert "COMPILE = True" in arch
    # helper class + imports + constants build_model depends on come along
    assert "class Block" in arch
    assert "import torch.nn as nn" in arch
    assert "HIDDEN = 64" in arch
    # recipe hooks are NOT frozen
    assert "build_optimizer" not in arch
    assert "training_config" not in arch


def test_extract_arch_source_requires_build_model(cont):
    assert cont.extract_arch_source("def build_optimizer(m):\n    return None") is None
    assert cont.extract_arch_source("def (((bad syntax") is None


def test_arch_matches_recipe_only_change(cont):
    arch = cont.extract_arch_source(PARENT_CODE)
    submission = arch + (
        "\n\ndef build_optimizer(model):\n"
        "    import torch\n"
        "    return torch.optim.AdamW(model.parameters(), lr=5e-4)\n"
    )
    assert cont.arch_matches(submission, arch)


def test_arch_matches_rejects_reshape(cont):
    arch = cont.extract_arch_source(PARENT_CODE)
    changed = PARENT_CODE.replace(
        "nn.Linear(HIDDEN, prediction_len)",
        "nn.Linear(HIDDEN, prediction_len * 2)",
    )
    assert changed != PARENT_CODE  # guard against a stale substring
    assert not cont.arch_matches(changed, arch)


def test_arch_matches_rejects_missing_symbol(cont):
    arch = cont.extract_arch_source(PARENT_CODE)
    only_build = (
        "def build_model(context_len, prediction_len, num_variates, quantiles):\n"
        "    import torch.nn as nn\n"
        "    return nn.Linear(context_len, prediction_len)\n"
    )
    # missing helper class / imports / init_weights → frozen node unmatched
    assert not cont.arch_matches(only_build, arch)


def test_arch_matches_rejects_modified_helper(cont):
    """Changing a helper class build_model depends on must be caught even
    though build_model itself is untouched."""
    arch = cont.extract_arch_source(PARENT_CODE)
    changed = PARENT_CODE.replace("self.lin = nn.Linear(d, d)",
                                  "self.lin = nn.Linear(d, d * 2)")
    assert not cont.arch_matches(changed, arch)


def test_arch_matches_allows_extra_additions(cont):
    """The designer may add extra imports/helpers for its recipe without
    breaking the match — only the frozen nodes must be reproduced."""
    arch = cont.extract_arch_source(PARENT_CODE)
    submission = (
        "import math\n\n"  # extra import the recipe might use
        + arch
        + "\n\ndef build_optimizer(model):\n"
        "    import torch\n"
        "    return torch.optim.AdamW(model.parameters(), lr=math.exp(-7))\n"
    )
    assert cont.arch_matches(submission, arch)


def test_arch_matches_ignores_formatting(cont):
    arch = cont.extract_arch_source(PARENT_CODE)
    # reformat the parent (extra blank lines / comments) — structurally same
    reformatted = PARENT_CODE.replace("COMPILE = True", "COMPILE = True  # opt in")
    assert cont.arch_matches(reformatted, arch)


def test_build_context_happy_path(cont):
    client = FakeClient({"/experiments/42": {"code": PARENT_CODE}})
    challenge = {
        "round_type": "continuation",
        "eligible_parents": [
            {"id": 42, "metric": 0.2, "checkpoint_available": True, "n_rounds": 3},
        ],
    }
    ctx = cont.build_context(challenge, client, "http://db")
    assert ctx is not None
    assert ctx["parent_id"] == 42
    assert "def build_model" in ctx["frozen_arch"]
    assert "class Block" in ctx["frozen_arch"]
    # only the one experiment fetch — no extra /signature round-trip
    assert client.calls == ["http://db/experiments/42"]


def test_build_context_new_round_is_none(cont):
    assert cont.build_context({"round_type": "new"}, FakeClient({}), "http://db") is None


def test_build_context_degrades_on_fetch_failure(cont):
    client = FakeClient({"/experiments/42": RuntimeError("boom")})
    challenge = {
        "round_type": "continuation",
        "eligible_parents": [
            {"id": 42, "metric": 0.2, "checkpoint_available": True},
        ],
    }
    # code fetch fails → design fresh, no crash
    assert cont.build_context(challenge, client, "http://db") is None


def test_build_context_none_without_checkpoint(cont):
    challenge = {
        "round_type": "continuation",
        "eligible_parents": [
            {"id": 7, "metric": 0.1, "checkpoint_available": False},
        ],
    }
    assert cont.build_context(challenge, FakeClient({}), "http://db") is None


def test_recipe_brief_shape(cont):
    brief = cont.recipe_brief({"parent_id": 9, "parent": {"metric": 0.3}})
    assert brief["_continuation"] is True
    assert "#9" in brief["summary"]
    assert isinstance(brief["plan"], list) and brief["plan"]
