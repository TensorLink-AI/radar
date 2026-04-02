"""Tests for validator.evaluator — Phase C evaluation (trust anchor)."""

from validator.evaluator import verify_flops_claim, evaluate_checkpoint


def test_verify_flops_claim_within_tolerance():
    assert verify_flops_claim(1_000_000, 1_010_000, tolerance=0.02) is True


def test_verify_flops_claim_outside_tolerance():
    assert verify_flops_claim(1_000_000, 1_050_000, tolerance=0.02) is False


def test_verify_flops_claim_zero_values():
    """Zero values can't be verified — return True."""
    assert verify_flops_claim(0, 1_000_000) is True
    assert verify_flops_claim(1_000_000, 0) is True
    assert verify_flops_claim(0, 0) is True


def test_verify_flops_claim_exact_match():
    assert verify_flops_claim(1_000_000, 1_000_000) is True


def test_evaluate_checkpoint_accepts_runner_dir():
    """evaluate_checkpoint() accepts runner_dir parameter."""
    import tempfile
    import os

    # Create a dummy checkpoint file so shutil.copy2 doesn't fail
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(b"\x00" * 16)
        dummy_ckpt = f.name

    try:
        result = evaluate_checkpoint(
            architecture_code="invalid",
            checkpoint_path=dummy_ckpt,
            runner_dir="runner/timeseries_forecast",
        )
        # Should return error dict (no torch), not crash
        assert "error" in result or "crps" in result
    finally:
        os.unlink(dummy_ckpt)
