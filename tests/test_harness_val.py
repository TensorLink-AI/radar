"""Tests for val loss tracking + best-checkpoint selection in runner/harness.py.

Covers:
  - _next_val_step schedule generator
  - val skipped when get_val_dataloader returns None
  - val runs and selects best checkpoint by lowest val loss
  - val failure does not kill training
  - val uses TaskRunner.default_loss, NEVER the submission's compute_loss
  - on_step_end never receives val_loss in its kwargs
  - train_loss_history is populated on log_every_n_steps cadence
  - TrainingMeta serialization round-trip with the new fields
  - Old-format meta dicts (no val fields) still parse via from_dict
"""

from __future__ import annotations

import json
import os
import sys
import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from runner.harness import (  # noqa: E402
    TrainingConfig,
    _next_val_step,
    _read_config,
    _training_loop,
    run_training,
)
from shared.artifacts import TrainingMeta  # noqa: E402


# ── _next_val_step ────────────────────────────────────────────────

def test_next_val_step_doubling_sequence():
    """base=10, growth=2.0 → 10, 20, 40, 80, 160, 320, 640, 1280."""
    seq = []
    cur = 0
    for _ in range(8):
        nxt = _next_val_step(cur, base=10, growth=2.0)
        seq.append(nxt)
        cur = nxt
    assert seq == [10, 20, 40, 80, 160, 320, 640, 1280]


def test_next_val_step_monotonically_increasing():
    cur = 0
    for _ in range(30):
        nxt = _next_val_step(cur, base=5, growth=1.5)
        assert nxt > cur, f"Expected next > {cur}, got {nxt}"
        cur = nxt


def test_next_val_step_handles_non_integer_growth():
    """Non-integer growth must still strictly advance (no infinite loops)."""
    seq = []
    cur = 0
    for _ in range(8):
        nxt = _next_val_step(cur, base=10, growth=1.3)
        seq.append(nxt)
        cur = nxt
    for a, b in zip(seq, seq[1:]):
        assert b > a, f"Sequence not strictly increasing: {seq}"


def test_next_val_step_below_base():
    """If current_step is below base, the next val happens at base."""
    assert _next_val_step(0, base=10, growth=2.0) == 10
    assert _next_val_step(5, base=10, growth=2.0) == 10
    assert _next_val_step(9, base=10, growth=2.0) == 10


# ── Test fixtures: tiny model, fake runner, fake submission ────────

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


def _make_batch(scale: float = 1.0):
    return {
        "input": torch.randn(2, 4) * scale,
        "target": torch.randn(2, 4) * scale,
    }


class _FakeRunner:
    """Minimal TaskRunner for tests.

    - `train_batches` controls the training data.
    - `val_batches` controls the val data (or None to disable val).
    - `default_loss_fn(predictions, targets, val_call_idx) -> tensor` lets tests
      vary the loss returned per val invocation by inspecting `_val_call_count`.
    """

    def __init__(
        self, train_batches=None, val_batches=None,
        default_loss_fn=None, raise_on_val_fetch=False,
    ):
        self.train_batches = train_batches if train_batches is not None else [
            _make_batch() for _ in range(40)
        ]
        self.val_batches = val_batches
        self._default_loss_fn = default_loss_fn or (
            lambda preds, targets, _idx: ((preds - targets) ** 2).mean()
        )
        self._raise_on_val_fetch = raise_on_val_fetch
        self._val_call_count = 0  # incremented per _run_val invocation

    def build_model(self, sub, device):
        return _TinyModel().to(device)

    def get_dataloader(self, batch_size):
        for b in self.train_batches:
            yield b

    def get_val_dataloader(self, batch_size):
        if self._raise_on_val_fetch:
            raise RuntimeError("simulated val_dataloader failure")
        if self.val_batches is None:
            return None
        return iter(list(self.val_batches))

    def default_loss(self, predictions, targets):
        # Detect val mode by absence of grad. The harness wraps val in
        # torch.no_grad().
        if not torch.is_grad_enabled():
            idx = self._val_call_count
            return self._default_loss_fn(predictions, targets, idx)
        return ((predictions - targets) ** 2).mean()

    def measure_flops(self, model, device):
        return 1_000_000

    def wrap_loss(self, sub_loss_fn):
        return sub_loss_fn


def _make_submission(
    *, training_config=None, compute_loss=None, on_step_end=None,
    configure_amp=None,
):
    sub = types.ModuleType("fake_submission")

    def build_optimizer(model):
        return torch.optim.SGD(model.parameters(), lr=1e-2)

    # build_model presence is checked in run_training (sanity gate); the
    # actual model construction lives on the runner. Stub for the gate.
    sub.build_model = lambda *args, **kw: _TinyModel()
    sub.build_optimizer = build_optimizer
    if training_config is not None:
        sub.training_config = lambda: training_config
    if compute_loss is not None:
        sub.compute_loss = compute_loss
    if on_step_end is not None:
        sub.on_step_end = on_step_end
    if configure_amp is not None:
        sub.configure_amp = configure_amp
    return sub


# A wrapper around _run_val (called via _training_loop) that bumps the
# runner's val call counter — installed by tests that want per-call control.
def _wrap_run_val_to_increment(runner):
    import runner.harness as harness_mod  # noqa: F401
    from runner import harness as _h
    original = _h._run_val

    def wrapped(rnr, *args, **kwargs):
        result = original(rnr, *args, **kwargs)
        rnr._val_call_count += 1
        return result

    return mock.patch.object(_h, "_run_val", wrapped)


# ── Val skipped when runner returns None ──────────────────────────

def test_val_skipped_when_get_val_dataloader_returns_none(monkeypatch):
    """Training should run normally; result has empty val_loss_history; checkpoint is current state."""
    runner = _FakeRunner(val_batches=None)
    sub = _make_submission(training_config={
        "batch_size": 2,
        "log_every_n_steps": 5,
        "val_base_step": 5,
    })

    saved = {}

    def fake_save_file(state_dict, path):
        saved["state"] = {k: v.clone() for k, v in state_dict.items()}
        saved["path"] = path

    def fake_load_submission(_arch_code):
        return sub

    monkeypatch.setattr("safetensors.torch.save_file", fake_save_file)
    monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
    monkeypatch.setattr("runner.harness._load_submission", fake_load_submission)

    cfg = TrainingConfig(time_budget=10, miner_hotkey="5T", round_id=1)
    result = run_training(runner, "ignored", cfg)

    assert result["status"] == "success"
    assert result["val_loss_history"] == []
    assert result["best_val_loss"] is None
    assert result["best_val_step"] == -1
    # Saved checkpoint should match the current model state (no best-val captured).
    assert "state" in saved


# ── Val runs and selects lowest-loss checkpoint ───────────────────

def test_val_selects_best_checkpoint_minimum():
    """With val losses controlled to dip then rise, best_val_step is the dip."""
    val_losses = [3.0, 2.0, 1.0, 1.5, 4.0]  # min at index 2
    seq = iter(val_losses)

    def loss_fn(preds, targets, _idx):
        try:
            v = next(seq)
        except StopIteration:
            v = 4.0
        return torch.tensor(v)

    runner = _FakeRunner(
        train_batches=[_make_batch() for _ in range(60)],
        val_batches=[_make_batch() for _ in range(2)],
        default_loss_fn=loss_fn,
    )
    sub = _make_submission(training_config={
        "batch_size": 2,
        "log_every_n_steps": 100,  # don't pollute train history
        "val_schedule": "fixed",
        "val_base_step": 1,
    })

    model = runner.build_model(sub, "cpu")
    import time
    out = _training_loop(runner, sub, model, "cpu", time_budget=60, start=time.time())

    # Val should have been called at least 4 times (5th is the synthetic final-step val).
    assert len(out["val_history"]) >= 3
    # The minimum across observed val_history should match best_val_loss.
    losses = [h["loss"] for h in out["val_history"]]
    assert out["best_val_loss"] == pytest.approx(min(losses))
    # best_val_step is the step where the minimum was observed.
    min_idx = losses.index(min(losses))
    assert out["best_val_step"] == out["val_history"][min_idx]["step"]
    # best_state must be a CPU state_dict snapshot.
    assert out["best_state"] is not None
    for k, v in out["best_state"].items():
        assert v.device.type == "cpu"


# ── Val failure modes ─────────────────────────────────────────────

def test_val_dataloader_exception_does_not_kill_training():
    """If get_val_dataloader raises, training proceeds with empty val history."""
    runner = _FakeRunner(raise_on_val_fetch=True)
    sub = _make_submission(training_config={
        "batch_size": 2, "log_every_n_steps": 100, "val_base_step": 1,
    })

    import time
    model = runner.build_model(sub, "cpu")
    out = _training_loop(runner, sub, model, "cpu", time_budget=10, start=time.time())

    assert out["val_history"] == []
    assert out["best_state"] is None
    assert out["best_val_step"] == -1


def test_val_nan_losses_yield_no_history():
    """If val loss is always non-finite, val_history stays empty."""
    def loss_fn(preds, targets, _idx):
        return torch.tensor(float("nan"))

    runner = _FakeRunner(
        val_batches=[_make_batch() for _ in range(2)],
        default_loss_fn=loss_fn,
    )
    sub = _make_submission(training_config={
        "batch_size": 2, "log_every_n_steps": 100,
        "val_schedule": "fixed", "val_base_step": 1,
    })

    import time
    model = runner.build_model(sub, "cpu")
    out = _training_loop(runner, sub, model, "cpu", time_budget=10, start=time.time())

    assert out["val_history"] == []
    assert out["best_state"] is None


# ── Val uses default_loss, NOT submission compute_loss ────────────

def test_val_uses_default_loss_not_submission_compute_loss():
    """Submission's compute_loss returns 0 — val_history should be nonzero."""
    def degenerate_compute_loss(preds, targets):
        return torch.tensor(0.0, requires_grad=True)

    # default_loss returns 7.0 so it's distinguishable.
    def loss_fn(preds, targets, _idx):
        return torch.tensor(7.0)

    runner = _FakeRunner(
        val_batches=[_make_batch() for _ in range(2)],
        default_loss_fn=loss_fn,
    )
    sub = _make_submission(
        training_config={
            "batch_size": 2, "log_every_n_steps": 100,
            "val_schedule": "fixed", "val_base_step": 1,
        },
        compute_loss=degenerate_compute_loss,
    )

    import time
    model = runner.build_model(sub, "cpu")
    out = _training_loop(runner, sub, model, "cpu", time_budget=10, start=time.time())

    assert len(out["val_history"]) >= 1
    for entry in out["val_history"]:
        assert entry["loss"] == pytest.approx(7.0), \
            f"Val loss must come from default_loss=7.0, got {entry}"


# ── on_step_end signature is unchanged ────────────────────────────

def test_on_step_end_does_not_receive_val_loss():
    """Submission's on_step_end must receive only the existing kwargs."""
    received_kwargs: list[set] = []

    def on_step_end(**kwargs):
        received_kwargs.append(set(kwargs.keys()))

    runner = _FakeRunner(
        val_batches=[_make_batch() for _ in range(2)],
        default_loss_fn=lambda p, t, _i: torch.tensor(1.0),
    )
    sub = _make_submission(
        training_config={
            "batch_size": 2, "log_every_n_steps": 100,
            "val_schedule": "fixed", "val_base_step": 1,
        },
        on_step_end=on_step_end,
    )

    import time
    model = runner.build_model(sub, "cpu")
    _training_loop(runner, sub, model, "cpu", time_budget=10, start=time.time())

    # on_step_end must have been called at least once
    assert received_kwargs, "on_step_end was never called"
    expected = {"model", "optimizer", "step", "total_steps", "loss_value"}
    for kw in received_kwargs:
        assert kw == expected, (
            f"on_step_end kwargs must be exactly {expected}, got {kw}"
        )


# ── Train loss history cadence ────────────────────────────────────

def test_train_loss_history_logged_at_log_every_n_steps():
    """Train history should sample at optim_step % log_every_n_steps == 0."""
    runner = _FakeRunner(
        train_batches=[_make_batch() for _ in range(35)],
    )
    sub = _make_submission(training_config={
        "batch_size": 2,
        "log_every_n_steps": 5,
        "val_schedule": "none",
    })

    import time
    model = runner.build_model(sub, "cpu")
    out = _training_loop(runner, sub, model, "cpu", time_budget=10, start=time.time())

    steps = [e["step"] for e in out["train_history"]]
    # All sampled steps are multiples of 5.
    for s in steps:
        assert s % 5 == 0, f"Train history step {s} not a multiple of 5"
    # Strictly increasing.
    assert steps == sorted(steps)


# ── TrainingMeta round-trip with new fields ───────────────────────

def test_training_meta_roundtrip_with_loss_histories():
    meta = TrainingMeta(
        round_id=42, miner_hotkey="5Hk", status="success",
        train_loss_history=[
            {"step": 10, "loss": 1.5},
            {"step": 20, "loss": 1.0},
        ],
        val_loss_history=[{"step": 10, "loss": 0.9}, {"step": 20, "loss": 0.7}],
        best_val_loss=0.7, best_val_step=20,
    )

    text = meta.to_json()
    blob = json.loads(text)
    restored = TrainingMeta.from_dict(blob)

    assert restored.round_id == 42
    assert restored.train_loss_history == [
        {"step": 10, "loss": 1.5}, {"step": 20, "loss": 1.0},
    ]
    assert restored.val_loss_history == [
        {"step": 10, "loss": 0.9}, {"step": 20, "loss": 0.7},
    ]
    assert restored.best_val_loss == 0.7
    assert restored.best_val_step == 20


def test_training_meta_old_format_parses_with_defaults():
    """Old metas with `loss_curve` and missing val fields must still load."""
    old = {
        "round_id": 9, "miner_hotkey": "5Old", "status": "success",
        "num_steps": 200, "loss_curve": [1.0, 0.8, 0.6],  # legacy
    }
    meta = TrainingMeta.from_dict(old)

    assert meta.round_id == 9
    assert meta.num_steps == 200
    assert meta.train_loss_history == []
    assert meta.val_loss_history == []
    assert meta.best_val_loss is None
    assert meta.best_val_step == -1


# ── _read_config picks up new keys ────────────────────────────────

def test_read_config_picks_up_new_val_keys():
    sub = types.ModuleType("s")
    sub.training_config = lambda: {
        "log_every_n_steps": 25,
        "val_schedule": "fixed",
        "val_base_step": 50,
        "val_growth": 3.0,
    }
    cfg = _read_config(sub)
    assert cfg["log_every_n_steps"] == 25
    assert cfg["val_schedule"] == "fixed"
    assert cfg["val_base_step"] == 50
    assert cfg["val_growth"] == 3.0


def test_read_config_clamps_out_of_range():
    sub = types.ModuleType("s")
    sub.training_config = lambda: {
        "log_every_n_steps": 99999,
        "val_growth": 0.1,  # below min
        "val_base_step": -5,  # below min
    }
    cfg = _read_config(sub)
    assert cfg["log_every_n_steps"] == 1000
    assert cfg["val_growth"] == pytest.approx(1.1)
    assert cfg["val_base_step"] == 1


def test_read_config_invalid_val_schedule_falls_back():
    sub = types.ModuleType("s")
    sub.training_config = lambda: {"val_schedule": "made_up"}
    cfg = _read_config(sub)
    assert cfg["val_schedule"] == "logarithmic"
