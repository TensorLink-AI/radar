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
