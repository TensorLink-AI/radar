"""Tests for tools/verify_radar_experiment.py.

The CLI talks to two external systems (a RADAR API and a Hippius IPFS
gateway). We mock both ends — the API via httpx.AsyncClient, the Hippius
client by patching ``HippiusClient`` inside the tool module.
"""

from __future__ import annotations

import hashlib
import io
from unittest.mock import AsyncMock, MagicMock, patch

import bittensor as bt
import httpx
import pytest

from shared.substrate import (
    SCHEMA_VERSION,
    PhaseCRecord,
    records_to_bundle,
    sign_record,
)
from tools.verify_radar_experiment import main


# ── Fixtures ──────────────────────────────────────────────────────────


def _kp(label: str) -> bt.Keypair:
    return bt.Keypair.create_from_seed(
        "0x" + hashlib.sha256(label.encode()).hexdigest(),
    )


def _make_signed_bundle(
    *,
    keypair: bt.Keypair,
    round_id: int = 42,
    miner_uid: int = 5,
    miner_hotkey: str = "5MinerHK",
    metrics: dict | None = None,
) -> bytes:
    """Build a single-record gzip bundle, signed by ``keypair``."""
    rec = PhaseCRecord(
        schema_version=SCHEMA_VERSION,
        round_id=round_id, block_hash="0xabc", task="ts_forecasting",
        miner_uid=miner_uid, miner_hotkey=miner_hotkey,
        code_hash="sha256:c", architecture_sha256="a", checkpoint_sha256="c",
        metrics=metrics if metrics is not None else {"crps": 0.123, "mase": 0.4},
        passed_size_gate=True, flops_verified=True, eval_status="ok",
        validator_uid=0, validator_hotkey=keypair.ss58_address,
        validator_block_height=100_000, timestamp=1.0,
    )

    class _W:
        hotkey = keypair

    return records_to_bundle([sign_record(rec, _W())])


def _api_response(
    *,
    cids: list[dict] | None = None,
    metrics: dict | None = None,
    miner_uid: int = 5,
    miner_hotkey: str = "5MinerHK",
    round_id: int = 42,
) -> dict:
    return {
        "index": 7,
        "round_id": round_id,
        "miner_uid": miner_uid,
        "miner_hotkey": miner_hotkey,
        "results": metrics or {"crps": 0.123, "mase": 0.4},
        "substrate_cids": cids if cids is not None else [],
    }


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in that returns a preset response."""

    def __init__(self, *, status_code: int = 200, json_body=None, raise_exc=None):
        self.status_code = status_code
        self.json_body = json_body
        self.raise_exc = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        if self.raise_exc:
            raise self.raise_exc
        resp = MagicMock()
        resp.status_code = self.status_code
        resp.json = MagicMock(return_value=self.json_body)
        if self.status_code >= 400:
            resp.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    f"{self.status_code}", request=MagicMock(), response=resp,
                ),
            )
        else:
            resp.raise_for_status = MagicMock()
        return resp


def _patch_api(json_body=None, raise_exc=None):
    """Patch httpx.AsyncClient to return ``json_body`` (or raise ``raise_exc``)."""
    fake = _FakeAsyncClient(json_body=json_body, raise_exc=raise_exc)
    return patch("tools.verify_radar_experiment.httpx.AsyncClient",
                 lambda *a, **kw: fake)


def _patch_hippius(bundle: bytes | None = None, raise_exc: Exception | None = None):
    """Replace the lazily-imported HippiusClient with one that yields ``bundle``."""
    fake = MagicMock()
    if raise_exc is not None:
        fake.download_bundle = AsyncMock(side_effect=raise_exc)
    else:
        fake.download_bundle = AsyncMock(return_value=bundle)
    fake.close = AsyncMock()
    # The CLI does ``from shared.hippius_client import HippiusClient`` inside
    # _run, so we patch it on the source module.
    return patch("shared.hippius_client.HippiusClient",
                 lambda **kw: fake), fake


def _argv(extra: dict | None = None) -> list[str]:
    base = {
        "--api-url": "http://radar.example.com",
        "--experiment-id": "7",
    }
    if extra:
        base.update(extra)
    return [a for k, v in base.items() for a in (k, v)]


# ── Tests ─────────────────────────────────────────────────────────────


def test_verify_tool_all_match_exit_0(capsys):
    """Bundle on Hippius matches the API record → exit 0."""
    kp = _kp("validator-1")
    bundle = _make_signed_bundle(keypair=kp)
    api_body = _api_response(cids=[{
        "kind": "phase_c_record",
        "validator_hotkey": kp.ss58_address,
        "cid": "bafyok",
        "round_id": 42,
    }])

    api_patch = _patch_api(json_body=api_body)
    hip_patch, fake = _patch_hippius(bundle=bundle)
    with api_patch, hip_patch:
        rc = main(_argv())
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "All substrate records verified" in out
    fake.download_bundle.assert_awaited_once_with("bafyok")


def test_verify_tool_metric_mismatch_exit_1(capsys):
    """Signed metric disagrees with API → exit 1, mismatch shown."""
    kp = _kp("validator-2")
    bundle = _make_signed_bundle(keypair=kp, metrics={"crps": 0.123})
    api_body = _api_response(
        cids=[{"validator_hotkey": kp.ss58_address, "cid": "bafymis", "round_id": 42}],
        metrics={"crps": 0.999},  # different
    )

    api_patch = _patch_api(json_body=api_body)
    hip_patch, _ = _patch_hippius(bundle=bundle)
    with api_patch, hip_patch:
        rc = main(_argv())
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "crps" in out  # the mismatched field name is in the report
    assert "Verification failed" in out


def test_verify_tool_no_cids_exit_2(capsys):
    """API returns an experiment with empty substrate_cids → exit 2."""
    api_body = _api_response(cids=[])

    api_patch = _patch_api(json_body=api_body)
    with api_patch:
        rc = main(_argv())
    assert rc == 2
    assert "no substrate_cids" in capsys.readouterr().out


def test_verify_tool_api_unreachable_exit_3(capsys):
    """Transport error from httpx → exit 3."""
    api_patch = _patch_api(raise_exc=httpx.ConnectError("connection refused"))
    with api_patch:
        rc = main(_argv())
    assert rc == 3
    err = capsys.readouterr().err
    assert "API unreachable" in err


def test_verify_tool_signature_failure_exit_1(capsys):
    """A bundle signed by validator A but the API claims B → exit 1."""
    kp_a = _kp("validator-a")
    kp_b = _kp("validator-b")
    bundle = _make_signed_bundle(keypair=kp_a)
    # API claims the CID came from kp_b → expected_hotkey mismatch in verify.
    api_body = _api_response(cids=[{
        "validator_hotkey": kp_b.ss58_address, "cid": "bafyforged", "round_id": 42,
    }])

    api_patch = _patch_api(json_body=api_body)
    hip_patch, _ = _patch_hippius(bundle=bundle)
    with api_patch, hip_patch:
        rc = main(_argv())
    assert rc == 1
    out = capsys.readouterr().out
    assert "validator_hotkey mismatch" in out


def test_verify_tool_fetch_failure_exit_1(capsys):
    """IPFS gateway flap surfaces in the per-CID report and exits 1."""
    api_body = _api_response(cids=[
        {"validator_hotkey": "5G", "cid": "bafyflap", "round_id": 42},
    ])

    api_patch = _patch_api(json_body=api_body)
    hip_patch, _ = _patch_hippius(raise_exc=RuntimeError("ipfs down"))
    with api_patch, hip_patch:
        rc = main(_argv())
    assert rc == 1
    out = capsys.readouterr().out
    assert "fetch failed" in out


def test_verify_tool_help_renders_clean(capsys):
    """--help should print our argparse output (regression: bittensor used
    to hijack argparse via its module-load-time logging config)."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--api-url" in out
    assert "--experiment-id" in out
    assert "Exit codes" in out
