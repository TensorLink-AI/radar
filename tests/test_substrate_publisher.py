"""Tests for validator/substrate_publisher.py."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from unittest.mock import AsyncMock

import bittensor as bt
import pytest

from shared.substrate import (
    SCHEMA_VERSION,
    bundle_sha256,
    records_from_bundle,
    verify_record,
)
from validator.substrate_publisher import (
    UploadResult,
    build_phase_c_records,
    publish_phase_c_records,
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
    round_id: int = 42


@dataclass
class _FakeMetagraph:
    hotkeys: list  # list[str]


@dataclass
class _FakeCommitment:
    code_hash: str


@pytest.fixture
def wallet():
    return _FakeWallet(_keypair("validator"))


@pytest.fixture
def metagraph():
    # 4 hotkeys for UIDs 0..3
    return _FakeMetagraph(hotkeys=[
        _keypair(f"miner-{i}").ss58_address for i in range(4)
    ])


@pytest.fixture
def commitments():
    return {
        1: _FakeCommitment(code_hash="sha256:miner-one-code"),
        2: _FakeCommitment(code_hash="sha256:miner-two-code"),
        3: _FakeCommitment(code_hash="sha256:miner-three-code"),
    }


@pytest.fixture
def training_metas():
    return {
        1: {
            "architecture_sha256": "arch-hash-1",
            "checkpoint_sha256": "ckpt-hash-1",
            "status": "success",
        },
        2: {
            "architecture_sha256": "arch-hash-2",
            "checkpoint_sha256": "ckpt-hash-2",
            "status": "success",
        },
        3: {
            "architecture_sha256": "arch-hash-3",
            "checkpoint_sha256": "ckpt-hash-3",
            "status": "success",
        },
    }


def _eval_ok(crps: float = 0.123) -> dict:
    return {
        "crps": crps,
        "mase": 0.456,
        "flops_equivalent_size": 1_500_000,
        "passed_size_gate": True,
        "flops_verified": True,
    }


# ── build_phase_c_records ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_records_produces_one_per_miner(
    wallet, metagraph, commitments, training_metas,
):
    eval_results = {1: _eval_ok(), 2: _eval_ok(0.1), 3: _eval_ok(0.2)}
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results=eval_results, training_metas=training_metas,
        commitments=commitments, metagraph=metagraph,
        my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xdeadbeef",
    )
    assert len(records) == 3
    assert {r.miner_uid for r in records} == {1, 2, 3}


@pytest.mark.asyncio
async def test_build_records_eval_status_size_gate_failed(
    wallet, metagraph, commitments, training_metas,
):
    eval_results = {1: {**_eval_ok(), "passed_size_gate": False}}
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results=eval_results, training_metas=training_metas,
        commitments=commitments, metagraph=metagraph,
        my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    assert records[0].eval_status == "size_gate_failed"
    assert records[0].passed_size_gate is False


@pytest.mark.asyncio
async def test_build_records_eval_status_ok(
    wallet, metagraph, commitments, training_metas,
):
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results={1: _eval_ok()},
        training_metas=training_metas, commitments=commitments,
        metagraph=metagraph, my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    assert records[0].eval_status == "ok"
    assert records[0].passed_size_gate is True
    assert records[0].flops_verified is True


@pytest.mark.asyncio
async def test_build_records_eval_status_eval_failed(
    wallet, metagraph, commitments, training_metas,
):
    eval_results = {1: {
        "passed_size_gate": True,
        "flops_verified": True,
        "error": "runner crashed",
    }}
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results=eval_results, training_metas=training_metas,
        commitments=commitments, metagraph=metagraph,
        my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    assert records[0].eval_status == "eval_failed"


@pytest.mark.asyncio
async def test_build_records_eval_status_no_metric(
    wallet, metagraph, commitments, training_metas,
):
    """Gate passed, no error, but no numeric metric in dict → no_metric."""
    eval_results = {1: {"passed_size_gate": True, "flops_verified": True}}
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results=eval_results, training_metas=training_metas,
        commitments=commitments, metagraph=metagraph,
        my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    assert records[0].eval_status == "no_metric"


@pytest.mark.asyncio
async def test_build_records_signatures_valid(
    wallet, metagraph, commitments, training_metas,
):
    eval_results = {1: _eval_ok(), 2: _eval_ok(0.1), 3: _eval_ok(0.2)}
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results=eval_results, training_metas=training_metas,
        commitments=commitments, metagraph=metagraph,
        my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    for r in records:
        ok, err = verify_record(r, expected_hotkey=wallet.hotkey.ss58_address)
        assert ok, err


@pytest.mark.asyncio
async def test_build_records_includes_miner_hotkey(
    wallet, metagraph, commitments, training_metas,
):
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results={2: _eval_ok()},
        training_metas=training_metas, commitments=commitments,
        metagraph=metagraph, my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    assert records[0].miner_hotkey == metagraph.hotkeys[2]
    assert records[0].miner_hotkey  # non-empty


@pytest.mark.asyncio
async def test_build_records_includes_code_hash(
    wallet, metagraph, commitments, training_metas,
):
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results={2: _eval_ok()},
        training_metas=training_metas, commitments=commitments,
        metagraph=metagraph, my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    assert records[0].code_hash == "sha256:miner-two-code"


@pytest.mark.asyncio
async def test_build_records_tolerates_missing_commitment(
    wallet, metagraph, training_metas,
):
    """A miner that has an eval result but no commitment yields code_hash=''."""
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results={1: _eval_ok()},
        training_metas=training_metas, commitments={},
        metagraph=metagraph, my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    assert records[0].code_hash == ""


@pytest.mark.asyncio
async def test_build_records_filters_non_serialisable_metrics(
    wallet, metagraph, commitments, training_metas,
):
    """List/dict/NaN values must be dropped before signing."""
    eval_results = {1: {
        "crps": 0.1,
        "passed_size_gate": True,
        "flops_verified": True,
        "nan_metric": float("nan"),
        "inf_metric": float("inf"),
        "list_metric": [1, 2, 3],
        "dict_metric": {"nested": True},
        "note": "fine string",
    }}
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results=eval_results, training_metas=training_metas,
        commitments=commitments, metagraph=metagraph,
        my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    metrics = records[0].metrics
    assert metrics["crps"] == 0.1
    assert metrics["note"] == "fine string"
    assert "nan_metric" not in metrics
    assert "inf_metric" not in metrics
    assert "list_metric" not in metrics
    assert "dict_metric" not in metrics


@pytest.mark.asyncio
async def test_build_records_includes_sha256_fields(
    wallet, metagraph, commitments, training_metas,
):
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results={2: _eval_ok()},
        training_metas=training_metas, commitments=commitments,
        metagraph=metagraph, my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    assert records[0].architecture_sha256 == "arch-hash-2"
    assert records[0].checkpoint_sha256 == "ckpt-hash-2"


# ── publish_phase_c_records ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_success_returns_result(
    wallet, metagraph, commitments, training_metas,
):
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results={1: _eval_ok(), 2: _eval_ok(0.2)},
        training_metas=training_metas, commitments=commitments,
        metagraph=metagraph, my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    expected = UploadResult(cid="bafyfake123", size_bytes=512)
    client = type("C", (), {})()
    client.upload_bundle = AsyncMock(return_value=expected)

    result = await publish_phase_c_records(
        client, records, round_id=42,
        validator_hotkey=wallet.hotkey.ss58_address,
    )

    assert result is expected
    client.upload_bundle.assert_awaited_once()
    args, _ = client.upload_bundle.call_args
    bundle_bytes, metadata = args
    # Metadata carries the spec'd tags
    assert metadata["app"] == "radar"
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["round_id"] == "42"
    assert metadata["validator_hotkey"] == wallet.hotkey.ss58_address
    assert metadata["record_count"] == "2"
    assert metadata["bundle_sha256"] == bundle_sha256(bundle_bytes)
    # Bundle round-trips back to the same records (cryptographically intact).
    parsed = records_from_bundle(bundle_bytes)
    assert parsed == records


@pytest.mark.asyncio
async def test_publish_failure_returns_none(
    wallet, metagraph, commitments, training_metas, caplog,
):
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results={1: _eval_ok()},
        training_metas=training_metas, commitments=commitments,
        metagraph=metagraph, my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    client = type("C", (), {})()
    client.upload_bundle = AsyncMock(side_effect=RuntimeError("hippius down"))
    with caplog.at_level("WARNING"):
        result = await publish_phase_c_records(
            client, records, round_id=42, validator_hotkey="5xyz",
        )
    assert result is None
    assert any("Substrate publish failed" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_publish_empty_records_no_call():
    client = type("C", (), {})()
    client.upload_bundle = AsyncMock()
    result = await publish_phase_c_records(
        client, records=[], round_id=42, validator_hotkey="5xyz",
    )
    assert result is None
    client.upload_bundle.assert_not_called()


@pytest.mark.asyncio
async def test_publish_no_client_returns_none(
    wallet, metagraph, commitments, training_metas, caplog,
):
    """A None client (operator hasn't opted in to substrate yet) must not raise."""
    records = await build_phase_c_records(
        wallet=wallet, challenge=_FakeChallenge(),
        eval_results={1: _eval_ok()},
        training_metas=training_metas, commitments=commitments,
        metagraph=metagraph, my_uid=0, current_block=100_000,
        task_name="ts_forecasting", block_hash="0xabc",
    )
    with caplog.at_level("WARNING"):
        result = await publish_phase_c_records(
            None, records, round_id=42, validator_hotkey="5xyz",
        )
    assert result is None
    assert any("no Hippius client" in m for m in caplog.messages)
