"""Tests for the three harness loosenings:

  1. ``model._aux_losses`` is summed and added to the main loss before backprop.
  2. ``on_step_begin`` hook fires after backward() and before optimizer.step(),
     symmetric to the existing on_step_end.
  3. ``time_budget`` is sqrt-scaled by ``max_flops`` (anchor 10M FLOPs, clamped
     to ``[base, 4×base]``) and surfaced as ``effective_time_budget``.

All three additions are opt-in: a submission that uses none of them must train
identically to the previous harness.
"""

from __future__ import annotations

import os
import sys
import time
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from runner.harness import (  # noqa: E402
    REFERENCE_FLOPS,
    TrainingConfig,
    _scaled_time_budget,
    _training_loop,
    run_training,
)


# ── Shared fixtures ───────────────────────────────────────────────────

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self._aux_losses: list = []

    def forward(self, x):
        return self.fc(x)


def _make_batch():
    return {"input": torch.randn(2, 4), "target": torch.randn(2, 4)}


class _FakeRunner:
    def __init__(self, model_factory=None, n_batches: int = 8):
        self._model_factory = model_factory or (lambda: _TinyModel())
        self._train_batches = [_make_batch() for _ in range(n_batches)]

    def build_model(self, sub, device):
        return self._model_factory().to(device)

    def get_dataloader(self, batch_size):
        for b in self._train_batches:
            yield b

    def get_val_dataloader(self, batch_size):
        return None

    def default_loss(self, predictions, targets):
        return ((predictions - targets) ** 2).mean()

    def measure_flops(self, model, device):
        return 0  # disable FLOPs cadence in these tests

    def wrap_loss(self, sub_loss_fn):
        return sub_loss_fn


def _make_submission(**hooks):
    sub = types.ModuleType("fake_submission")
    sub.build_model = lambda *a, **kw: _TinyModel()
    sub.build_optimizer = lambda model: torch.optim.SGD(model.parameters(), lr=1e-2)
    if "training_config" in hooks:
        sub.training_config = lambda: hooks["training_config"]
    if "on_step_begin" in hooks:
        sub.on_step_begin = hooks["on_step_begin"]
    if "on_step_end" in hooks:
        sub.on_step_end = hooks["on_step_end"]
    return sub


# ── Change 3: _scaled_time_budget ─────────────────────────────────────

class TestScaledTimeBudget:
    """Sqrt scaling, anchored at REFERENCE_FLOPS, clamped to [base, 4×base]."""

    def test_anchor_returns_base(self):
        """max_flops == REFERENCE_FLOPS → unchanged."""
        assert _scaled_time_budget(300, REFERENCE_FLOPS) == 300

    def test_10x_flops_yields_sqrt10x_budget(self):
        """100M flops → 300 * sqrt(10) ≈ 948s."""
        assert _scaled_time_budget(300, 100_000_000) == 948

    def test_below_anchor_clamped_to_base(self):
        """1M flops would give 300 * sqrt(0.1) ≈ 95 — must clamp up to base."""
        assert _scaled_time_budget(300, 1_000_000) == 300

    def test_above_ceiling_clamped_to_4x(self):
        """10B flops would give 300 * sqrt(1000) ≈ 9486 — must clamp to 1200."""
        assert _scaled_time_budget(300, 10_000_000_000) == 1200

    def test_zero_max_flops_returns_base(self):
        """When max_flops is unset (size gate disabled), no scaling."""
        assert _scaled_time_budget(300, 0) == 300

    def test_negative_max_flops_returns_base(self):
        assert _scaled_time_budget(300, -1) == 300


def test_run_training_surfaces_effective_time_budget(monkeypatch):
    """run_training result includes effective_time_budget computed from max_flops."""
    runner = _FakeRunner()
    sub = _make_submission(training_config={"batch_size": 2})

    monkeypatch.setattr("safetensors.torch.save_file", lambda *a, **kw: None)
    monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
    monkeypatch.setattr("runner.harness._load_submission", lambda _: sub)

    cfg = TrainingConfig(
        time_budget=300, max_flops=100_000_000, min_flops=1,
        miner_hotkey="hk", round_id=1,
    )
    result = run_training(runner, "ignored", cfg)

    assert result["status"] == "success"
    assert result["effective_time_budget"] == 948
    # Env var sees the effective budget too — downstream task harnesses
    # honour the scaled value.
    assert os.environ["TIME_BUDGET"] == "948"


# ── Change 1: _aux_losses convention ──────────────────────────────────

class _AuxModel(nn.Module):
    """Toy model that pushes a fixed-magnitude aux term per forward()."""

    def __init__(self, aux_scale: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        # learnable scalar — gradient must reach it via the aux path
        self.aux_param = nn.Parameter(torch.tensor(1.0))
        self._aux_losses: list = []
        self._aux_scale = aux_scale

    def forward(self, x):
        # Append a differentiable aux term that depends on aux_param.
        # The main loss path does NOT touch aux_param, so any gradient on
        # aux_param can only have come through ``_aux_losses``.
        self._aux_losses.append(self._aux_scale * self.aux_param.pow(2))
        return self.fc(x)


class _NoAuxModel(nn.Module):
    """Same shape as _AuxModel but without populating _aux_losses."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.aux_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.fc(x)


def _run_short_training(model_factory, *, batches=None, seed=0):
    torch.manual_seed(seed)
    runner = _FakeRunner(model_factory=model_factory)
    if batches is not None:
        runner._train_batches = batches
    sub = _make_submission(training_config={
        "batch_size": 2, "log_every_n_steps": 1,
    })
    model = runner.build_model(sub, "cpu")
    out = _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())
    return model, out


class TestAuxLosses:
    def test_aux_loss_gradient_flows_to_aux_only_param(self):
        """aux_param has no path through main loss — gradient must come from aux."""
        torch.manual_seed(0)
        model = _AuxModel(aux_scale=0.01)
        runner = _FakeRunner()
        sub = _make_submission(training_config={"batch_size": 2})
        out = _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())

        # After training, aux_param should have moved away from 1.0 because
        # ∂(0.01 * aux_param²)/∂aux_param = 0.02 * aux_param drives it toward 0.
        assert out["step"] > 0
        assert model.aux_param.item() < 1.0, (
            f"aux_param did not move: {model.aux_param.item()} (gradient never flowed)"
        )

    def test_aux_list_cleared_each_step(self):
        """The harness must clear ``_aux_losses`` after consuming it."""
        torch.manual_seed(0)
        model = _AuxModel(aux_scale=0.01)
        runner = _FakeRunner()
        sub = _make_submission(training_config={"batch_size": 2})
        _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())
        # If the list weren't cleared, it'd hold one entry per forward call.
        assert model._aux_losses == []

    def test_no_aux_losses_attribute_is_safe(self):
        """A model without ``_aux_losses`` trains normally."""
        torch.manual_seed(0)
        runner = _FakeRunner(model_factory=lambda: nn.Linear(4, 4))
        sub = _make_submission(training_config={"batch_size": 2})
        model = runner.build_model(sub, "cpu")
        out = _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())
        assert out["step"] > 0

    def test_no_aux_path_is_bit_identical_to_baseline(self):
        """A model that never populates _aux_losses produces the same trajectory
        as a model without the attribute at all (i.e. presence of the attribute
        but empty list must not perturb training)."""

        # Use the same fixed batches so randomness is identical.
        torch.manual_seed(123)
        batches = [_make_batch() for _ in range(8)]

        # Run A: model with empty _aux_losses list (always empty)
        torch.manual_seed(0)
        model_a, out_a = _run_short_training(
            lambda: _NoAuxModel(), batches=batches, seed=0,
        )

        # Run B: identical model but with `_aux_losses = []` set.
        class _EmptyAuxModel(_NoAuxModel):
            def __init__(self):
                super().__init__()
                self._aux_losses: list = []

        torch.manual_seed(0)
        model_b, out_b = _run_short_training(
            lambda: _EmptyAuxModel(), batches=batches, seed=0,
        )

        # Both runs must produce identical aux_param values.
        assert model_a.aux_param.item() == pytest.approx(model_b.aux_param.item(), abs=1e-7)
        # Loss histories must match.
        assert len(out_a["train_history"]) == len(out_b["train_history"])
        for ha, hb in zip(out_a["train_history"], out_b["train_history"]):
            assert ha["loss"] == pytest.approx(hb["loss"], abs=1e-6)

    def test_aux_path_diverges_from_baseline(self):
        """Two short trainings on identical seeds — one with aux, one without —
        must produce different parameters (proving aux actually moves the model)."""
        batches = [_make_batch() for _ in range(8)]

        torch.manual_seed(0)
        model_baseline, _ = _run_short_training(
            lambda: _NoAuxModel(), batches=batches, seed=0,
        )

        torch.manual_seed(0)
        model_aux, _ = _run_short_training(
            lambda: _AuxModel(aux_scale=0.5), batches=batches, seed=0,
        )

        # fc.weight is shared in both runs and should diverge because the aux
        # term doesn't touch fc, but it changes the optimizer step magnitude
        # via grad accumulation only when aux contributes — and aux_param
        # itself is what diverges.
        assert model_baseline.aux_param.item() == pytest.approx(1.0, abs=1e-6)
        assert model_aux.aux_param.item() < 0.99

    def test_non_finite_aux_does_not_crash(self):
        """A non-finite aux total must be silently dropped, not propagated."""

        class _NaNAuxModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
                self._aux_losses: list = []

            def forward(self, x):
                self._aux_losses.append(
                    torch.tensor(float("nan"), requires_grad=True)
                )
                return self.fc(x)

        runner = _FakeRunner()
        sub = _make_submission(training_config={"batch_size": 2})
        model = _NaNAuxModel()
        out = _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())
        # Training continued (step counter advanced); list cleared.
        assert out["step"] > 0
        assert model._aux_losses == []

    def test_non_tensor_aux_entries_ignored(self):
        """Non-tensor / non-grad entries in _aux_losses must not crash."""

        class _BadAuxModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
                self._aux_losses: list = []

            def forward(self, x):
                # Scalar floats / detached tensors should be silently filtered.
                self._aux_losses.append(0.5)
                self._aux_losses.append(torch.tensor(1.0))  # no requires_grad
                return self.fc(x)

        runner = _FakeRunner()
        sub = _make_submission(training_config={"batch_size": 2})
        model = _BadAuxModel()
        out = _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())
        assert out["step"] > 0
        assert model._aux_losses == []


# ── Change 2: on_step_begin hook ──────────────────────────────────────

class TestOnStepBegin:
    def test_on_step_begin_called(self):
        calls = []

        def on_step_begin(**kw):
            calls.append(("begin", kw["step"]))

        runner = _FakeRunner()
        sub = _make_submission(
            training_config={"batch_size": 2},
            on_step_begin=on_step_begin,
        )
        model = runner.build_model(sub, "cpu")
        out = _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())
        assert len(calls) == out["step"]
        assert calls[0] == ("begin", 1)

    def test_on_step_begin_failure_does_not_kill_training(self):
        def boom(**kw):
            raise RuntimeError("hook failed")

        runner = _FakeRunner()
        sub = _make_submission(
            training_config={"batch_size": 2},
            on_step_begin=boom,
        )
        model = runner.build_model(sub, "cpu")
        out = _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())
        assert out["step"] > 0  # training survived the hook failures

    def test_ordering_backward_begin_step_end(self):
        """Strict ordering: backward → on_step_begin → step → on_step_end.

        We assert this by snapshotting:
          • param.grad existence at on_step_begin (proves backward already ran)
          • param value vs. snapshot at on_step_end (proves step happened
            between the two hooks)
        and checking call order across the run.
        """
        events: list[str] = []
        seen = {"begin_grad_was_set": [], "param_changed_between_hooks": []}
        last_param_at_begin: dict = {}

        def on_step_begin(*, model, **kw):
            events.append("begin")
            grad = next(model.parameters()).grad
            seen["begin_grad_was_set"].append(grad is not None and torch.any(grad != 0).item())
            last_param_at_begin["w"] = next(model.parameters()).detach().clone()

        def on_step_end(*, model, **kw):
            events.append("end")
            now = next(model.parameters()).detach()
            if "w" in last_param_at_begin:
                # optimizer.step() ran between begin and end → params changed.
                changed = not torch.equal(now, last_param_at_begin["w"])
                seen["param_changed_between_hooks"].append(changed)

        runner = _FakeRunner()
        sub = _make_submission(
            training_config={"batch_size": 2},
            on_step_begin=on_step_begin,
            on_step_end=on_step_end,
        )
        model = runner.build_model(sub, "cpu")
        _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())

        # Strict alternation: begin, end, begin, end, ...
        assert events[0::2] == ["begin"] * (len(events) // 2)
        assert events[1::2] == ["end"] * (len(events) // 2)
        # Every begin saw a populated gradient (backward had already run).
        assert all(seen["begin_grad_was_set"])
        # Every (begin, end) pair surrounded an optimizer.step() that changed params.
        assert all(seen["param_changed_between_hooks"])

    def test_on_step_begin_step_index_matches_on_step_end(self):
        """on_step_begin and on_step_end of the same step share the step index."""
        seen = []

        def on_step_begin(**kw):
            seen.append(("begin", kw["step"]))

        def on_step_end(**kw):
            seen.append(("end", kw["step"]))

        runner = _FakeRunner()
        sub = _make_submission(
            training_config={"batch_size": 2},
            on_step_begin=on_step_begin,
            on_step_end=on_step_end,
        )
        model = runner.build_model(sub, "cpu")
        _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())

        # Walk pairs: ("begin", n), ("end", n), ("begin", n+1), ("end", n+1), ...
        for i in range(0, len(seen), 2):
            kind_b, step_b = seen[i]
            kind_e, step_e = seen[i + 1]
            assert (kind_b, kind_e) == ("begin", "end")
            assert step_b == step_e

    def test_on_step_begin_can_modify_grads_before_step(self):
        """Hook zeroing gradients before optimizer.step() must prevent param updates."""

        def zero_grads(*, model, **kw):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        runner = _FakeRunner()
        sub = _make_submission(
            training_config={"batch_size": 2},
            on_step_begin=zero_grads,
        )
        model = runner.build_model(sub, "cpu")
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _training_loop(runner, sub, model, "cpu", time_budget=30, start=time.time())
        after = model.state_dict()
        # All params unchanged because gradients were zeroed before each step.
        for k in before:
            assert torch.equal(before[k], after[k]), f"{k} moved despite zeroed grads"
