"""Tests for validator.evaluator — Phase C evaluation (trust anchor)."""

from validator.evaluator import verify_flops_claim


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
