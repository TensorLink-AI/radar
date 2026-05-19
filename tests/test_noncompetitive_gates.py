"""Regression tests: NONCOMPETITIVE early-returns + optional bittensor import.

Confirms the chain-touching code paths are properly gated so a
non-competitive deployment can run without ever invoking bittensor in
the leaf modules (runner/server, miner/neuron, validator/neuron init,
DatabaseClient).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


def test_runner_server_has_no_bittensor_reference():
    """runner/server.py was fully stripped of metagraph/bittensor."""
    import runner.server as srv
    src = open(srv.__file__).read()
    assert "import bittensor" not in src
    assert not hasattr(srv, "_metagraph_cache")
    assert not hasattr(srv, "_load_metagraph")
    assert not hasattr(srv, "_metagraph_lock")


def test_miner_neuron_bittensor_import_is_optional():
    """``import bittensor`` is wrapped in try/except so the module
    loads in non-competitive deployments that don't have bittensor
    installed."""
    src = open("/home/user/radar/miner/neuron.py").read()
    assert "try:\n    import bittensor as bt" in src
    assert "bt = None" in src


def test_validator_neuron_bittensor_import_is_optional():
    src = open("/home/user/radar/validator/neuron.py").read()
    assert "try:\n    import bittensor as bt" in src
    assert "bt = None" in src


def test_validator_get_config_uses_argparse_in_noncompetitive(monkeypatch):
    """get_config() returns a plain Namespace (no bittensor) when
    NONCOMPETITIVE=true."""
    monkeypatch.setenv("RADAR_NONCOMPETITIVE", "true")
    monkeypatch.setenv("RADAR_SERVICE_KEY", "a" * 64)
    import importlib
    import config
    importlib.reload(config)
    # Read the function source instead of importing the whole module
    # (which has heavy transitive deps).  We just need to confirm the
    # branch shape.
    src = open("/home/user/radar/validator/neuron.py").read()
    assert "Config.NONCOMPETITIVE or bt is None" in src
    assert "parser.parse_args()" in src


def test_miner_get_config_uses_argparse_in_noncompetitive():
    src = open("/home/user/radar/miner/neuron.py").read()
    assert "Config.NONCOMPETITIVE or bt is None" in src


def test_set_weights_skipped_in_noncompetitive():
    """_set_weights() has an early-return on Config.NONCOMPETITIVE."""
    src = open("/home/user/radar/validator/neuron.py").read()
    # Look for the gate right at the top of _set_weights.
    assert (
        'def _set_weights(self):\n        """Set weights on chain'
        in src
    )
    assert 'NONCOMPETITIVE: skipping weight set' in src


def test_run_round_uses_walltime_in_noncompetitive():
    src = open("/home/user/radar/validator/neuron.py").read()
    # Walltime path uses time.time() // 12 as a stand-in for block height.
    assert "int(time.time() // 12)" in src


def test_validator_wallet_is_none_in_noncompetitive():
    """The validator init skips wallet/subtensor/metagraph creation
    when NONCOMPETITIVE is set."""
    src = open("/home/user/radar/validator/neuron.py").read()
    assert "self.wallet = None" in src
    assert "self.subtensor = None" in src
    assert "self.metagraph = None" in src


def test_miner_wallet_is_none_in_noncompetitive():
    src = open("/home/user/radar/miner/neuron.py").read()
    assert "self.wallet = None" in src
    assert "self.subtensor = None" in src
    assert "self.metagraph = None" in src


def test_validator_db_client_uses_service_secret_in_noncompetitive():
    src = open("/home/user/radar/validator/neuron.py").read()
    assert "service_secret=Config.SERVICE_KEY.encode()" in src
    assert "key_id=Config.SERVICE_KEY_ID" in src


def test_database_client_constructor_rejects_no_auth():
    """At least one of wallet / service_secret / api_key required."""
    from shared.db_client import DatabaseClient
    with pytest.raises(ValueError):
        DatabaseClient("http://x")


def test_database_client_accepts_service_secret_only():
    from shared.db_client import DatabaseClient
    c = DatabaseClient("http://x", service_secret=b"k" * 32)
    assert c.service_secret == b"k" * 32
    assert c.wallet is None


def test_database_client_accepts_api_key_only():
    from shared.db_client import DatabaseClient
    c = DatabaseClient("http://x", api_key="rdrk_abc")
    assert c.api_key == "rdrk_abc"
    assert c.wallet is None
    assert c.service_secret is None


def test_database_client_signs_with_hmac_when_secret_set():
    from shared.db_client import DatabaseClient
    c = DatabaseClient("http://x", service_secret=b"a" * 32, key_id="op")
    headers = c._sign(b"body")
    assert "X-Radar-Signature" in headers
    assert "X-Radar-Timestamp" in headers
    assert headers["X-Radar-Key-Id"] == "op"


def test_database_client_signs_with_bearer_when_key_set():
    from shared.db_client import DatabaseClient
    c = DatabaseClient("http://x", api_key="rdrk_abc")
    headers = c._sign(b"")
    assert headers == {"Authorization": "Bearer rdrk_abc"}


def test_substrate_module_loads_without_bittensor():
    """shared/substrate.py uses TYPE_CHECKING + lazy _bt() so it loads
    even when bittensor is missing from the environment."""
    src = open("/home/user/radar/shared/substrate.py").read()
    assert "if TYPE_CHECKING" in src
    assert "import bittensor as bt" in src  # in the TYPE_CHECKING block
    # The runtime path goes through _bt() — verify the helper exists.
    assert "def _bt():" in src
    # No bare top-level import outside TYPE_CHECKING.
    lines = src.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("import bittensor") or stripped.startswith(
            "from bittensor"
        ):
            # Walk up to see if we're inside a TYPE_CHECKING block.
            context = "\n".join(lines[max(0, i - 5):i + 1])
            assert (
                "TYPE_CHECKING" in context
                or "def " in context  # inside a function = lazy
            ), f"bare bittensor import at line {i + 1}: {line}"


def test_validator_neuron_imports_without_bittensor(monkeypatch):
    """validator/neuron.py + its transitive graph load when bittensor
    is missing.  Smoke test — confirms the chain-free deployment path
    doesn't blow up at import time."""
    monkeypatch.setenv("RADAR_NONCOMPETITIVE", "true")
    monkeypatch.setenv("RADAR_SERVICE_KEY", "a" * 64)
    monkeypatch.setenv("RADAR_DB_API_URL", "http://localhost:8090")
    import importlib
    import config
    importlib.reload(config)
    import validator.neuron
    importlib.reload(validator.neuron)
    # bt is None when bittensor isn't installed; ImportError was caught
    # at top-level by the try/except.  In this container it IS None
    # (bittensor not in pip set).
    # The important assertion: the module loaded.
    assert validator.neuron is not None
    assert hasattr(validator.neuron, "Validator")


def test_no_top_level_bittensor_imports_in_main_code():
    """Sweep guard: no top-level ``import bittensor`` in code/, only
    inside functions or under ``if TYPE_CHECKING:``."""
    import pathlib

    root = pathlib.Path("/home/user/radar")
    bad: list[str] = []
    for path in root.rglob("*.py"):
        if any(part in {"tests", "scripts"} for part in path.parts):
            continue
        text = path.read_text()
        lines = text.splitlines()
        for i, raw in enumerate(lines):
            stripped = raw.lstrip()
            if not (
                stripped.startswith("import bittensor")
                or stripped.startswith("from bittensor")
            ):
                continue
            indent = len(raw) - len(stripped)
            # Anything indented is inside a function / try block → lazy.
            if indent > 0:
                continue
            # Top-level — must be guarded by TYPE_CHECKING or try/except.
            ctx = "\n".join(lines[max(0, i - 4):i])
            if "TYPE_CHECKING" in ctx or "try:" in ctx:
                continue
            bad.append(f"{path}:{i + 1}: {raw}")
    assert not bad, "unguarded top-level bittensor imports:\n" + "\n".join(bad)
