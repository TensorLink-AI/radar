"""Tests for the synthetic_data_generator task: spec wiring, the fixed
reference architecture, challenge contract, and continuation epoch pinning.

The trainer dispatch path (``run_synth_generator_training``) needs torch +
GIFT-Eval data and is exercised in integration runs only; here we exercise the
parts that stay numpy-only, plus a torch-gated check of the reference model.
"""

from __future__ import annotations

import pytest

from local.synthetic_arch import (
    REFERENCE_ARCH_CODE, REFERENCE_ARCH_VERSION, reference_arch,
)
from local.task import (
    SyntheticDataGeneratorSpec, buckets_for, make_spec,
)


def test_make_spec_synthetic_data_generator():
    spec = make_spec("synthetic_data_generator")
    assert isinstance(spec, SyntheticDataGeneratorSpec)
    assert spec.context_len == 512
    assert spec.prediction_len == 96
    assert spec.num_variates == 1
    assert len(spec.quantiles) == 9


def test_buckets_for_synth_matches_ts():
    sdg = make_spec("synthetic_data_generator")
    ts = make_spec("ts_forecasting")
    assert buckets_for(sdg) == buckets_for(ts)


def test_reference_arch_card_is_fixed():
    arch = reference_arch()
    assert arch.version == REFERENCE_ARCH_VERSION == 1
    card = arch.to_card()
    assert card["version"] == 1
    assert card["fixed"] is True
    assert card["code"] == REFERENCE_ARCH_CODE
    assert "build_model" in card["code"]


# ── challenge contract ──────────────────────────────────────────────


def test_task_dict_advertises_pipeline_contract_and_gift_scoring():
    from local.validator import _task_dict

    d = _task_dict(make_spec("synthetic_data_generator"))
    assert d["name"] == "synthetic_data_generator"
    # Shares the ts_data_pipeline runner / build_pipeline surface.
    assert d["runner_dir"] == "ts_data_pipeline"
    assert any("build_pipeline" in c for c in d["constraints"])
    # Scored GIFT-only (crps/mase), not the AULC composite.
    obj_names = {o["name"] for o in d["objectives"]}
    assert obj_names == {"crps", "mase"}


def test_build_challenge_injects_fixed_reference_arch(tmp_path):
    from local.store import LocalStore
    from local.validator import _build_challenge

    store = LocalStore(str(tmp_path / "t.db"))
    try:
        challenge = _build_challenge(
            0, store, make_spec("synthetic_data_generator"),
            services_url="http://127.0.0.1:0",
        )
    finally:
        store.close()
    # The fixed reference arch rides on the challenge as a frozen_arch card so
    # the existing miner pipeline tools work unchanged.
    assert "frozen_arch" in challenge
    assert challenge["frozen_arch"]["fixed"] is True
    assert "build_model" in challenge["frozen_arch"]["code"]
    # Size buckets are meaningless (fixed arch) → gate disabled.
    assert challenge["min_flops_equivalent"] == 0
    assert challenge["max_flops_equivalent"] == 0


def test_current_epoch_pins_synth_arch_version():
    from local.validator import _current_epoch

    sdg = make_spec("synthetic_data_generator")
    assert _current_epoch(sdg) == {"synth_arch_version": 1}


# ── reference model (torch-gated) ───────────────────────────────────


def test_reference_model_param_count_and_forward_shape():
    torch = pytest.importorskip("torch")

    ns: dict = {"__name__": "ref_arch"}
    exec(REFERENCE_ARCH_CODE, ns)
    build = ns["build_model"]

    context_len, prediction_len, num_variates = 512, 96, 1
    quantiles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    model = build(context_len, prediction_len, num_variates, quantiles)

    n_params = sum(p.numel() for p in model.parameters())
    # "Miniature ~10M": keep it honestly in that ballpark.
    assert 8_000_000 < n_params < 13_000_000, n_params

    x = torch.randn(4, context_len, num_variates)
    out = model(x)
    assert out.shape == (4, prediction_len, num_variates, len(quantiles))
    assert torch.isfinite(out).all()


def test_reference_model_hooks_present():
    pytest.importorskip("torch")
    ns: dict = {"__name__": "ref_arch"}
    exec(REFERENCE_ARCH_CODE, ns)
    for hook in ("build_model", "init_weights", "build_optimizer",
                 "training_config", "build_scheduler"):
        assert callable(ns.get(hook)), hook
    cfg = ns["training_config"]()
    assert cfg["batch_size"] > 0
