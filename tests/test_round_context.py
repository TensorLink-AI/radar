"""Tests for the agent-side special-round/screening helpers
(miners/claude_style_v{4,5}/core/round_context.py). The module is
dependency-free, so it's loaded standalone — no torch, no package
imports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MINERS = Path(__file__).resolve().parent.parent / "miners"


def _load(version: str):
    path = _MINERS / f"claude_style_{version}" / "core" / "round_context.py"
    spec = importlib.util.spec_from_file_location(
        f"round_context_{version}", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=["v4", "v5"])
def rc(request):
    return _load(request.param)


def _card(code="def build_model(): pass"):
    return {"id": 7, "name": "tgt", "metric": 0.5, "code": code,
            "source_bucket": "tiny"}


def test_build_detects_each_kind(rc):
    assert rc.build({"round_type": "new"}) is None
    assert rc.build({"round_type": "continuation"}) is None
    for kind, key in [("ablate", "ablation_target"),
                      ("recipe_only", "recipe_base"),
                      ("transfer", "transfer_source")]:
        ctx = rc.build({"round_type": kind, key: _card()})
        assert ctx == {"kind": kind, "target": _card()}
        # Card without code → not actionable → normal round.
        assert rc.build({"round_type": kind, key: {"id": 7}}) is None


def test_briefs_are_focused(rc):
    abl = rc.brief({"kind": "ablate", "target": _card()})
    assert abl["_special"] == "ablate"
    assert "EXACTLY ONE" in abl["summary"]
    rec = rc.brief({"kind": "recipe_only", "target": _card()})
    assert "frozen" in rec["summary"]
    xfer = rc.brief({"kind": "transfer", "target": _card()}, bucket="large")
    assert "tiny" in xfer["summary"] and "large" in xfer["summary"]


def test_preamble_carries_target_code(rc):
    for kind in ("ablate", "transfer"):
        text = rc.preamble({"kind": kind, "target": _card("CODE_MARKER")})
        assert "CODE_MARKER" in text
        assert "#7" in text


def _state(n_validated=3):
    cands = {}
    for i in range(n_validated):
        cands[f"cand_{i:08x}"] = {
            "code": f"code_{i}", "validated": True, "created_at": float(i),
        }
    cands["cand_unvalid"] = {
        "code": "code_bad", "validated": False, "created_at": 99.0,
    }
    return {"candidates": cands}


def test_screening_candidates_packaging(rc):
    challenge = {"screening": {"enabled": True, "max_candidates": 3}}
    out = rc.screening_candidates(
        challenge, _state(), primary_name="main", primary_code="primary",
    )
    assert out[0] == {"name": "main", "code": "primary"}
    assert len(out) == 3  # capped
    # Unvalidated candidate never ships; newest validated first.
    codes = [c["code"] for c in out]
    assert "code_bad" not in codes
    assert codes[1] == "code_2"


def test_screening_requires_advert_and_alternates(rc):
    # No screening card → None.
    assert rc.screening_candidates(
        {}, _state(), primary_name="m", primary_code="p",
    ) is None
    # Advertised but no validated alternates → None (primary alone).
    challenge = {"screening": {"enabled": True, "max_candidates": 4}}
    assert rc.screening_candidates(
        challenge, {"candidates": {}}, primary_name="m", primary_code="p",
    ) is None
    # Duplicate of the primary doesn't count as an alternate.
    state = {"candidates": {"cand_x": {
        "code": "p", "validated": True, "created_at": 1.0,
    }}}
    assert rc.screening_candidates(
        challenge, state, primary_name="m", primary_code="p",
    ) is None
