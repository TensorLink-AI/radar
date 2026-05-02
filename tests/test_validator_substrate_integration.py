"""Integration tests for Phase 4: substrate publishing wired into validator/neuron.py.

The full `Validator.run_round` pulls in bittensor, subtensor, agent pods, and
trainer dispatch — far too much to drive end-to-end in a unit test. We test
the integration in two layers instead:

  * `Validator._init_hippius`: instantiate `Validator` via ``__new__`` (the
    pattern other validator tests already use) and exercise the helper that
    Phase 4 added to ``__init__``.
  * `run_substrate_publish_step`: the helper Phase 4 inserts after
    ``score_round``. Driving it with mocked inputs proves the round-loop
    contract: disabled → no client call, enabled → exactly one upload per
    round, failure → empty dict + no exception.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import bittensor as bt
import pytest

from validator.substrate_publisher import (
    UploadResult,
    run_substrate_publish_step,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _seed_hex(label: str) -> str:
    return "0x" + hashlib.sha256(label.encode()).hexdigest()


def _keypair(label: str) -> bt.Keypair:
    return bt.Keypair.create_from_seed(_seed_hex(label))


class _FakeWallet:
    def __init__(self, kp: bt.Keypair):
        self.hotkey = kp


@dataclass
class _FakeChallenge:
    round_id: int = 7


@dataclass
class _FakeMetagraph:
    hotkeys: list


@dataclass
class _FakeCommitment:
    code_hash: str


def _round_inputs(num_miners: int = 3) -> dict:
    """Build a realistic-shaped set of run_round locals for the helper."""
    wallet = _FakeWallet(_keypair("validator"))
    metagraph = _FakeMetagraph(hotkeys=[
        _keypair(f"miner-{i}").ss58_address for i in range(num_miners + 1)
    ])
    eval_results = {
        uid: {
            "crps": 0.1 + 0.01 * uid,
            "mase": 0.5,
            "flops_equivalent_size": 1_500_000,
            "passed_size_gate": True,
            "flops_verified": True,
        }
        for uid in range(1, num_miners + 1)
    }
    training_metas = {
        uid: {
            "architecture_sha256": f"arch-{uid}",
            "checkpoint_sha256": f"ckpt-{uid}",
            "status": "success",
        }
        for uid in range(1, num_miners + 1)
    }
    commitments = {
        uid: _FakeCommitment(code_hash=f"sha256:code-{uid}")
        for uid in range(1, num_miners + 1)
    }
    return dict(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results=eval_results, training_metas=training_metas,
        commitments=commitments, metagraph=metagraph,
        my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )


# ── run_substrate_publish_step (the round-loop hook) ─────────────────


@pytest.mark.asyncio
async def test_round_with_hippius_disabled_unchanged():
    """hippius=None mirrors HIPPIUS_ENABLED=false — no client work, empty dict."""
    inputs = _round_inputs()
    cids = await run_substrate_publish_step(hippius=None, **inputs)
    assert cids == {}


@pytest.mark.asyncio
async def test_round_with_hippius_enabled_publishes():
    """A configured client gets called exactly once with the structured args,
    and every miner's UID maps to the returned bundle key."""
    inputs = _round_inputs(num_miners=3)
    upload = UploadResult(
        key="radar/0/phase_c/7/abcdef0123456789.tar",
        metadata={"app_tag": "radar", "phase": "phase_c", "run_id": "7"},
        tags={"app_tag": "radar", "phase": "phase_c", "netuid": "0"},
        size=1024, etag="deadbeef",
    )
    client = type("C", (), {})()
    client.upload_bundle = AsyncMock(return_value=upload)

    cids = await run_substrate_publish_step(hippius=client, **inputs)

    client.upload_bundle.assert_awaited_once()
    call = client.upload_bundle.call_args
    bundle_bytes = call.args[0]
    assert isinstance(bundle_bytes, (bytes, bytearray))
    assert call.kwargs["app_tag"] == "radar"
    assert call.kwargs["phase"] == "phase_c"
    assert call.kwargs["run_id"] == "7"
    assert call.kwargs["extra_metadata"]["record_count"] == "3"
    # All miners share the same bundle key (one bundle per round).
    assert cids == {1: upload.key, 2: upload.key, 3: upload.key}


@pytest.mark.asyncio
async def test_round_with_hippius_enabled_publish_failure_doesnt_crash(caplog):
    """A raising client must not propagate — round must keep running."""
    inputs = _round_inputs()
    client = type("C", (), {})()
    client.upload_bundle = AsyncMock(side_effect=RuntimeError("network down"))

    with caplog.at_level("WARNING"):
        cids = await run_substrate_publish_step(hippius=client, **inputs)

    assert cids == {}
    client.upload_bundle.assert_awaited_once()
    # The publish layer logs the failure rather than letting it propagate.
    assert any("Substrate publish failed" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_round_with_no_eval_results_skips_upload():
    """A round that produced zero eval results should not burn an upload."""
    inputs = _round_inputs()
    inputs["eval_results"] = {}
    client = type("C", (), {})()
    client.upload_bundle = AsyncMock()

    cids = await run_substrate_publish_step(hippius=client, **inputs)

    assert cids == {}
    client.upload_bundle.assert_not_called()


@pytest.mark.asyncio
async def test_round_publish_returning_none_yields_empty_cids():
    """When the client signals failure (returns None), no CIDs get propagated."""
    inputs = _round_inputs()
    client = type("C", (), {})()
    client.upload_bundle = AsyncMock(return_value=None)

    cids = await run_substrate_publish_step(hippius=client, **inputs)

    assert cids == {}
    client.upload_bundle.assert_awaited_once()


# ── Validator._init_hippius wiring ───────────────────────────────────


def _make_validator_for_init_test():
    """Skip __init__ (which constructs bittensor) and return a bare instance."""
    from validator.neuron import Validator
    return Validator.__new__(Validator)


def test_init_hippius_returns_none_when_disabled():
    """Default config (HIPPIUS_ENABLED=false) yields self.hippius = None."""
    validator = _make_validator_for_init_test()
    with patch("validator.neuron.Config") as cfg:
        cfg.HIPPIUS_ENABLED = False
        assert validator._init_hippius() is None


def test_init_hippius_returns_none_when_sdk_missing(caplog):
    """HIPPIUS_ENABLED=true with the wrapper not yet shipped: warn + None."""
    validator = _make_validator_for_init_test()
    import builtins

    real_import = builtins.__import__

    def _missing_hippius_client(name, *args, **kwargs):
        if name == "shared.hippius_client":
            raise ImportError("no such module")
        return real_import(name, *args, **kwargs)

    with patch("validator.neuron.Config") as cfg, \
         patch("builtins.__import__", side_effect=_missing_hippius_client), \
         caplog.at_level("WARNING"):
        cfg.HIPPIUS_ENABLED = True
        cfg.HIPPIUS_IPFS_API_URL = "https://ipfs"
        cfg.HIPPIUS_KEY = "k"
        cfg.HIPPIUS_SUBSTRATE_RPC = "rpc"
        result = validator._init_hippius()

    assert result is None
    assert any("not available yet" in m for m in caplog.messages)


def test_init_hippius_constructs_when_enabled():
    """With a stub HippiusClient available, init returns the constructed client."""
    validator = _make_validator_for_init_test()

    class _StubHippiusClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Inject a fake `shared.hippius_client` module so the lazy import succeeds.
    import sys, types
    stub_mod = types.ModuleType("shared.hippius_client")
    stub_mod.HippiusClient = _StubHippiusClient
    with patch("validator.neuron.Config") as cfg, \
         patch.dict(sys.modules, {"shared.hippius_client": stub_mod}):
        cfg.HIPPIUS_ENABLED = True
        cfg.HIPPIUS_IPFS_API_URL = "https://ipfs.example"
        cfg.HIPPIUS_KEY = "key123"
        cfg.HIPPIUS_SUBSTRATE_RPC = "wss://rpc"
        client = validator._init_hippius()

    assert isinstance(client, _StubHippiusClient)
    assert client.kwargs == {
        "ipfs_api_url": "https://ipfs.example",
        "hippius_key": "key123",
        "substrate_rpc": "wss://rpc",
    }


def test_init_hippius_swallows_constructor_errors(caplog):
    """Hippius client construction failure must downgrade to None, not crash."""
    validator = _make_validator_for_init_test()

    class _BoomClient:
        def __init__(self, **kwargs):
            raise RuntimeError("ipfs unreachable")

    import sys, types
    stub_mod = types.ModuleType("shared.hippius_client")
    stub_mod.HippiusClient = _BoomClient
    with patch("validator.neuron.Config") as cfg, \
         patch.dict(sys.modules, {"shared.hippius_client": stub_mod}), \
         caplog.at_level("WARNING"):
        cfg.HIPPIUS_ENABLED = True
        cfg.HIPPIUS_IPFS_API_URL = ""
        cfg.HIPPIUS_KEY = ""
        cfg.HIPPIUS_SUBSTRATE_RPC = ""
        result = validator._init_hippius()

    assert result is None
    assert any("Hippius client init failed" in m for m in caplog.messages)
