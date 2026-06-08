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
    assert arch.version == REFERENCE_ARCH_VERSION
    card = arch.to_card()
    assert card["version"] == REFERENCE_ARCH_VERSION
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
    assert _current_epoch(sdg) == {"synth_arch_version": REFERENCE_ARCH_VERSION}


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


def _exec_arch():
    ns: dict = {"__name__": "ref_arch"}
    exec(REFERENCE_ARCH_CODE, ns)
    return ns


def test_reference_model_handles_missing_values():
    torch = pytest.importorskip("torch")
    ns = _exec_arch()
    Q = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    model = ns["build_model"](512, 96, 1, Q)
    x = torch.randn(2, 512, 1)
    x[:, :40, :] = float("nan")          # leading gap
    x[0, 200:260, :] = float("nan")      # interior gap
    out = model(x)
    assert out.shape == (2, 96, 1, len(Q))
    assert torch.isfinite(out).all()     # robust scaler + mask stay finite


def test_reference_model_clamp_blocks_sinh_overflow():
    torch = pytest.importorskip("torch")
    ns = _exec_arch()
    Q = (0.1, 0.5, 0.9)
    model = ns["build_model"](64, 32, 1, Q)
    with torch.no_grad():               # force absurd head outputs
        model.out_head.weight.mul_(50.0)
        model.out_head.bias.add_(25.0)
    out = model(torch.randn(2, 64, 1))
    assert torch.isfinite(out).all()


def test_normuon_optimizer_trains():
    torch = pytest.importorskip("torch")
    ns = _exec_arch()
    Q = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    qs = torch.tensor(Q)

    def pinball(pred, tgt):
        e = tgt.unsqueeze(-1) - pred
        return torch.max(qs * e, (qs - 1) * e).mean()

    torch.manual_seed(0)
    model = ns["build_model"](128, 32, 1, Q)
    ns["init_weights"](model)
    opt = ns["build_optimizer"](model)
    assert type(opt).__name__ == "NorMuon"
    sched = ns["build_scheduler"](opt, 200)
    # A learnable target (deterministic function of the context) so a real
    # optimizer must drive the loss down meaningfully.
    ctx = torch.randn(8, 128, 1)
    tgt = ctx[:, -32:, :] * 0.8 + 0.1
    model.train()
    first = None
    for i in range(120):
        opt.zero_grad()
        loss = pinball(model(ctx), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        assert torch.isfinite(loss), f"NaN loss at step {i}"
        if i == 0:
            first = loss.item()
    # A clear, machine-robust margin — proves NorMuon actually optimizes
    # (a broken optimizer plateaus or diverges) without tuning to a seed.
    assert loss.item() < first * 0.85
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_reference_model_hooks_present():
    pytest.importorskip("torch")
    ns: dict = {"__name__": "ref_arch"}
    exec(REFERENCE_ARCH_CODE, ns)
    for hook in ("build_model", "init_weights", "build_optimizer",
                 "training_config", "build_scheduler"):
        assert callable(ns.get(hook)), hook
    cfg = ns["training_config"]()
    assert cfg["batch_size"] > 0


# ── dashboard surfacing ─────────────────────────────────────────────


def _add_sdg_exp(store, *, name, crps, mase, metric, parent_index=None,
                 mode="new", n_rounds=1, cumc=0.0):
    return store.add_experiment(
        round_id=0, miner_id="m", name=name, code="c", motivation="",
        reasoning="", tool_calls=[], metric=metric, success=True,
        objectives={
            "flops_equivalent_size": 9_800_000, "num_params": 9_817_824,
            "crps": crps, "mase": mase, "synth_arch_version": 1,
            "cumulative_compute": cumc,
        },
        score=0.0, loss_curve=[], task="synthetic_data_generator",
        parent_index=parent_index, mode=mode, n_rounds=n_rounds,
        cumulative_compute=cumc,
    )


def test_dashboard_crps_mase_frontier_includes_synth(tmp_path):
    import sqlite3

    from local.dashboard import _frontier_crps_mase
    from local.store import LocalStore

    db_path = tmp_path / "t.db"
    store = LocalStore(str(db_path))
    _add_sdg_exp(store, name="sdg_a", crps=0.4, mase=0.6, metric=0.49)
    store.close()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        front = _frontier_crps_mase(conn)
    finally:
        conn.close()
    assert "sdg_a" in {p["name"] for p in front}


def test_dashboard_lineage_includes_synth_continuation(tmp_path):
    import sqlite3

    from local.dashboard import _lineage
    from local.store import LocalStore

    db_path = tmp_path / "t.db"
    store = LocalStore(str(db_path))
    root = _add_sdg_exp(store, name="sdg_root", crps=0.5, mase=0.7, metric=0.59)
    _add_sdg_exp(store, name="sdg_child", crps=0.4, mase=0.6, metric=0.49,
                 parent_index=root, mode="continue", n_rounds=2, cumc=1e9)
    store.close()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        lin = _lineage(conn, "")
    finally:
        conn.close()
    tasks = {n["task"] for n in lin["nodes"]}
    assert "synthetic_data_generator" in tasks
    # The continuation chain produces a within-task lineage edge.
    assert any(
        e["kind"] == "lineage" and e["from"] == root for e in lin["edges"]
    )
