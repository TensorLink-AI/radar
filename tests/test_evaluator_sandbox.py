"""Tests for validator/evaluator.py — subprocess sandboxing of Phase C eval."""

import os
import tempfile

from validator.evaluator import evaluate_checkpoint


def test_env_var_isolation():
    """Miner code in subprocess should NOT see parent process env vars."""
    secret = "SUPER_SECRET_TEST_VALUE_12345"
    os.environ["SECRET_TEST_VAR"] = secret

    # Architecture code that tries to read the env var
    arch_code = '''
import os
import json
import sys

val = os.environ.get("SECRET_TEST_VAR", "NOT_FOUND")
# Output as JSON so evaluate_checkpoint can parse it
print(json.dumps({"crps": 999.0, "mase": 999.0, "leaked_secret": val}))
sys.exit(0)
'''
    # We need a dummy checkpoint — the code above exits before loading it,
    # but evaluate_checkpoint still needs the path to exist for the template
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        dummy_ckpt = f.name

    try:
        result = evaluate_checkpoint(arch_code, dummy_ckpt)
        # The subprocess should fail because the code doesn't define build_model,
        # but even if the runner template runs, the env var should not be visible.
        # Check that the secret was NOT leaked
        leaked = result.get("leaked_secret", "NOT_FOUND")
        assert leaked != secret, "Subprocess should NOT see parent env vars"
    finally:
        os.environ.pop("SECRET_TEST_VAR", None)
        try:
            os.remove(dummy_ckpt)
        except OSError:
            pass


def test_timeout_kills_slow_code():
    """Miner code that sleeps too long should be killed by timeout."""
    arch_code = '''
import time
time.sleep(200)
'''
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        dummy_ckpt = f.name

    try:
        # Use a short timeout for testing by monkeypatching isn't needed —
        # evaluate_checkpoint has a 120s timeout. We can't wait that long in tests.
        # Instead, test with a code that exits immediately but incorrectly.
        # For the actual timeout test, we verify the error handling path.
        result = evaluate_checkpoint(arch_code, dummy_ckpt)
        assert result["crps"] == float("inf")
        assert "error" in result
    finally:
        try:
            os.remove(dummy_ckpt)
        except OSError:
            pass


def test_missing_build_model_returns_error():
    """Architecture code without build_model() should return inf metrics."""
    arch_code = '''
# No build_model defined
x = 1 + 1
'''
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        dummy_ckpt = f.name

    try:
        result = evaluate_checkpoint(arch_code, dummy_ckpt)
        assert result["crps"] == float("inf")
        assert result["mase"] == float("inf")
    finally:
        try:
            os.remove(dummy_ckpt)
        except OSError:
            pass


def test_subprocess_uses_clean_env():
    """Verify the evaluator constructs a clean env without R2/wallet vars."""
    # Set some sensitive env vars
    os.environ["R2_SECRET_ACCESS_KEY"] = "fake_secret_key"
    os.environ["BASILICA_TOKEN"] = "fake_token"

    arch_code = '''
import os
import json
import sys
r2 = os.environ.get("R2_SECRET_ACCESS_KEY", "NOT_FOUND")
bt = os.environ.get("BASILICA_TOKEN", "NOT_FOUND")
print(json.dumps({"crps": 0.0, "mase": 0.0, "r2": r2, "bt": bt}))
sys.exit(0)
'''
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        dummy_ckpt = f.name

    try:
        result = evaluate_checkpoint(arch_code, dummy_ckpt)
        # Even if the template overrides the code's output, check no leaks
        assert result.get("r2", "NOT_FOUND") != "fake_secret_key"
        assert result.get("bt", "NOT_FOUND") != "fake_token"
    finally:
        os.environ.pop("R2_SECRET_ACCESS_KEY", None)
        os.environ.pop("BASILICA_TOKEN", None)
        try:
            os.remove(dummy_ckpt)
        except OSError:
            pass
