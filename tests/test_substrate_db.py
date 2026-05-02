"""Tests for TEN-240 Phase 5: substrate_cids schema, DB client, and API endpoints.

Combines:
  * `DataElement` field roundtrip
  * `pg_schema` column threading (DDL string + INSERT_SQL + serialisation
    helpers) without standing up a real Postgres
  * `DatabaseClient.add_experiment` payload shape (with and without CID)
  * `database/server.py` add + GET /verify endpoints
  * `database/verify.verify_experiment` mocked-Hippius unit tests
  * `database/neuron.DatabaseNeuron._init_hippius` wiring

Each test is intentionally tight and uses no real Postgres / Hippius — the
DB-level invariants are covered structurally (DDL inspection + INSERT_SQL
arity match) so the suite stays fast on CI.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import bittensor as bt
import pytest
from fastapi.testclient import TestClient

from shared.database import DataElement
from shared.db_client import DatabaseClient
from shared.pg_schema import (
    INSERT_SQL,
    SCHEMA_INDEX_DDL,
    SCHEMA_TABLE_DDL,
    element_to_params,
)
from shared.substrate import (
    SCHEMA_VERSION,
    PhaseCRecord,
    records_to_bundle,
    sign_record,
)


# ── DataElement ───────────────────────────────────────────────────────


def test_dataelement_substrate_cids_default_empty():
    elem = DataElement()
    assert elem.substrate_cids == []
    # Default factory must not be a shared list across instances.
    elem.substrate_cids.append({"cid": "x"})
    assert DataElement().substrate_cids == []


def test_dataelement_to_api_dict_includes_substrate_cids():
    entries = [{
        "kind": "phase_c_record", "validator_hotkey": "5G", "cid": "bafy",
        "round_id": 7,
    }]
    elem = DataElement(substrate_cids=entries)
    api = elem.to_api_dict()
    assert api["substrate_cids"] == entries


def test_dataelement_to_api_dict_tolerates_jsonb_string():
    """Defense-in-depth: if a row arrives with substrate_cids as a JSON
    string (no asyncpg JSONB codec), to_api_dict must still emit a list."""
    entries = [{"cid": "bafy"}]
    elem = DataElement(substrate_cids=json.dumps(entries))  # type: ignore[arg-type]
    api = elem.to_api_dict()
    assert api["substrate_cids"] == entries


def test_dataelement_from_dict_roundtrip_carries_substrate_cids():
    entries = [{"cid": "bafy", "validator_hotkey": "5G"}]
    elem = DataElement(substrate_cids=entries)
    parsed = DataElement.from_dict(elem.to_dict())
    assert parsed.substrate_cids == entries


# ── pg_schema DDL & serialisation ────────────────────────────────────


def test_schema_table_declares_substrate_cids():
    assert "substrate_cids JSONB NOT NULL DEFAULT '[]'" in SCHEMA_TABLE_DDL


def test_schema_index_declares_gin_on_substrate_cids():
    assert "idx_substrate_cids" in SCHEMA_INDEX_DDL
    assert "GIN(substrate_cids)" in SCHEMA_INDEX_DDL


def test_insert_sql_param_count_includes_substrate_cids():
    """The INSERT statement now binds 24 params; element_to_params produces
    a 24-tuple. Mismatched arity here would silently misorder columns at
    runtime against a real DB."""
    elem = DataElement(
        name="t", code="x", success=True,
        substrate_cids=[{"cid": "bafy"}],
    )
    params = element_to_params(elem, next_id=1)
    assert len(params) == 24
    # Last param is the substrate_cids JSONB string, not raw list.
    assert isinstance(params[-1], str)
    assert json.loads(params[-1]) == [{"cid": "bafy"}]
    # INSERT_SQL declares $24 and 24 columns — basic structural check.
    assert "$24" in INSERT_SQL
    assert "substrate_cids" in INSERT_SQL


# ── DatabaseClient.add_experiment payload shape ──────────────────────


class _MockWallet:
    class hotkey:
        ss58_address = "5FakeValidator"

        @staticmethod
        def sign(msg):
            return b"\x00" * 64


def _patched_post(client, return_index: int = 7):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"index": return_index}
    mock_resp.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    return mock_resp, mock_http


@pytest.mark.asyncio
async def test_add_experiment_without_substrate_cid_unchanged():
    """No substrate_cid → payload shape is exactly the legacy shape."""
    client = DatabaseClient(db_url="http://fake", wallet=_MockWallet())
    _, mock_http = _patched_post(client)
    with patch.object(client, "_get_client", return_value=mock_http):
        idx = await client.add_experiment({"name": "x"})
    assert idx == 7
    sent = json.loads(mock_http.post.call_args.kwargs["content"])
    assert sent == {"data": {"name": "x"}}
    assert "substrate_cid" not in sent
    assert "validator_hotkey" not in sent


@pytest.mark.asyncio
async def test_add_experiment_with_substrate_cid_includes_fields():
    client = DatabaseClient(db_url="http://fake", wallet=_MockWallet())
    _, mock_http = _patched_post(client)
    with patch.object(client, "_get_client", return_value=mock_http):
        idx = await client.add_experiment(
            {"name": "x", "round_id": 99},
            substrate_cid="bafycid",
            validator_hotkey="5Gxxxxx",
        )
    assert idx == 7
    sent = json.loads(mock_http.post.call_args.kwargs["content"])
    assert sent["data"] == {"name": "x", "round_id": 99}
    assert sent["substrate_cid"] == "bafycid"
    assert sent["validator_hotkey"] == "5Gxxxxx"


# ── server.py: add_experiment + GET /verify endpoints ────────────────


class _MockStore:
    def __init__(self):
        self._elements: list[DataElement] = []
        self.provenance = None

    async def get_size(self):
        return len(self._elements)

    async def add(self, element: DataElement) -> int:
        element.index = len(self._elements)
        self._elements.append(element)
        return element.index

    async def get(self, index: int):
        for e in self._elements:
            if e.index == index:
                return e
        return None


def _server_with_store():
    """Reset module-level injection slots and return a TestClient + store."""
    from database import server
    store = _MockStore()
    server.set_db(store)
    server.set_hippius(None)  # default state for the verify tests
    return TestClient(server.app), store, server


def test_add_experiment_without_substrate_cid_legacy_shape():
    client, store, _ = _server_with_store()
    r = client.post("/experiments/add", json={"data": {"name": "exp", "round_id": 1}})
    assert r.status_code == 200
    assert store._elements[0].substrate_cids == []


def test_add_experiment_with_substrate_cid_appends_audit_entry():
    client, store, _ = _server_with_store()
    r = client.post(
        "/experiments/add",
        json={
            "data": {"name": "exp", "round_id": 42, "miner_uid": 5},
            "substrate_cid": "bafyfakecid",
            "validator_hotkey": "5Gabc",
        },
    )
    assert r.status_code == 200
    elem = store._elements[0]
    assert elem.substrate_cids == [{
        "kind": "phase_c_record",
        "validator_hotkey": "5Gabc",
        "cid": "bafyfakecid",
        "round_id": 42,
    }]


def test_add_experiment_with_artifact_cids_appends_each():
    """Phase 7: artifact_cids land in the same substrate_cids list with
    their declared kinds (checkpoint/architecture/training_meta).
    Entries without a ``cid`` are dropped server-side so a half-built
    dual-write doesn't poison the audit list."""
    client, store, _ = _server_with_store()
    r = client.post(
        "/experiments/add",
        json={
            "data": {"name": "exp", "round_id": 42, "miner_uid": 5},
            "substrate_cid": "bafyrec",
            "validator_hotkey": "5Gabc",
            "artifact_cids": [
                {"kind": "checkpoint", "cid": "bafyckpt"},
                {"kind": "architecture", "cid": "bafyarch"},
                {"kind": "training_meta", "cid": "bafymeta"},
                {"kind": "noop"},  # missing cid — dropped
            ],
        },
    )
    assert r.status_code == 200
    cids = store._elements[0].substrate_cids
    kinds = [c["kind"] for c in cids]
    assert kinds == [
        "phase_c_record", "checkpoint", "architecture", "training_meta",
    ]
    assert {c["cid"] for c in cids} == {
        "bafyrec", "bafyckpt", "bafyarch", "bafymeta",
    }


def test_add_experiment_artifact_cids_pydantic_rejects_non_dicts():
    """Strict input validation: a string in artifact_cids → 422 Unprocessable.

    Catches validator-side bugs early rather than silently filtering them
    out and producing partial audit lists."""
    client, _, _ = _server_with_store()
    r = client.post(
        "/experiments/add",
        json={
            "data": {"name": "exp", "round_id": 42},
            "artifact_cids": ["not a dict"],
        },
    )
    assert r.status_code == 422


def test_get_experiment_response_includes_substrate_cids():
    client, store, _ = _server_with_store()
    store._elements.append(DataElement(
        index=0, name="exp", round_id=42, miner_uid=5,
        substrate_cids=[{"cid": "bafy", "validator_hotkey": "5G", "round_id": 42}],
    ))
    r = client.get("/experiments/0")
    assert r.status_code == 200
    body = r.json()
    assert body["substrate_cids"] == [
        {"cid": "bafy", "validator_hotkey": "5G", "round_id": 42}
    ]


def test_verify_endpoint_503_when_hippius_unconfigured():
    client, store, _ = _server_with_store()
    store._elements.append(DataElement(index=0, round_id=42, miner_uid=5))
    r = client.get("/experiments/0/verify")
    assert r.status_code == 503
    assert "Hippius" in r.json()["detail"]


def test_verify_endpoint_404_for_missing_experiment():
    client, _, _ = _server_with_store()
    r = client.get("/experiments/99/verify")
    assert r.status_code == 404


def test_verify_endpoint_empty_when_no_substrate_cids():
    """An experiment with no CIDs returns an empty verifications list."""
    client, store, server = _server_with_store()
    store._elements.append(DataElement(index=0, round_id=42, miner_uid=5))
    server.set_hippius(MagicMock())  # any non-None client
    r = client.get("/experiments/0/verify")
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_id"] == 0
    assert body["substrate_cids"] == []
    assert body["verifications"] == []


def test_verify_endpoint_returns_per_cid_results():
    """End-to-end: signed bundle on mocked Hippius matches a stored experiment."""
    client, store, server = _server_with_store()

    kp = bt.Keypair.create_from_seed(
        "0x" + hashlib.sha256(b"validator-1").hexdigest(),
    )
    record = PhaseCRecord(
        schema_version=SCHEMA_VERSION,
        round_id=42, block_hash="0xabc", task="ts_forecasting",
        miner_uid=5, miner_hotkey="5MinerKey",
        code_hash="sha256:c", architecture_sha256="a", checkpoint_sha256="c",
        metrics={"crps": 0.123, "mase": 0.4, "passed_size_gate": True},
        passed_size_gate=True, flops_verified=True, eval_status="ok",
        validator_uid=0, validator_hotkey=kp.ss58_address,
        validator_block_height=100_000, timestamp=1.0,
    )

    class _W:
        hotkey = kp

    signed = sign_record(record, _W())
    bundle = records_to_bundle([signed])

    elem = DataElement(
        index=0, name="exp", round_id=42, miner_uid=5, miner_hotkey="5MinerKey",
        objectives={"crps": 0.123, "mase": 0.4, "passed_size_gate": True},
        substrate_cids=[{
            "kind": "phase_c_record", "validator_hotkey": kp.ss58_address,
            "cid": "bafycid", "round_id": 42,
        }],
    )
    store._elements.append(elem)

    fake_hippius = MagicMock()
    fake_hippius.download_bundle = AsyncMock(return_value=bundle)
    server.set_hippius(fake_hippius)

    r = client.get("/experiments/0/verify")
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_id"] == 0
    assert len(body["verifications"]) == 1
    v = body["verifications"][0]
    assert v["fetchable"] is True
    assert v["signature_valid"] is True
    assert v["matches_db"] is True
    assert v["discrepancies"] == []
    fake_hippius.download_bundle.assert_awaited_once_with("bafycid")


def test_verify_endpoint_flags_metric_mismatch():
    """When DB and bundle disagree on a metric, matches_db=False with a diff."""
    client, store, server = _server_with_store()
    kp = bt.Keypair.create_from_seed(
        "0x" + hashlib.sha256(b"validator-2").hexdigest(),
    )
    record = PhaseCRecord(
        schema_version=SCHEMA_VERSION,
        round_id=42, block_hash="0xabc", task="ts_forecasting",
        miner_uid=5, miner_hotkey="5MinerKey",
        code_hash="sha256:c", architecture_sha256="a", checkpoint_sha256="c",
        metrics={"crps": 0.123},
        passed_size_gate=True, flops_verified=True, eval_status="ok",
        validator_uid=0, validator_hotkey=kp.ss58_address,
        validator_block_height=100_000, timestamp=1.0,
    )

    class _W:
        hotkey = kp

    signed = sign_record(record, _W())
    bundle = records_to_bundle([signed])

    elem = DataElement(
        index=0, round_id=42, miner_uid=5, miner_hotkey="5MinerKey",
        objectives={"crps": 0.999},  # DB disagrees
        substrate_cids=[{
            "kind": "phase_c_record", "validator_hotkey": kp.ss58_address,
            "cid": "bafycid", "round_id": 42,
        }],
    )
    store._elements.append(elem)

    fake_hippius = MagicMock()
    fake_hippius.download_bundle = AsyncMock(return_value=bundle)
    server.set_hippius(fake_hippius)

    body = client.get("/experiments/0/verify").json()
    v = body["verifications"][0]
    assert v["signature_valid"] is True
    assert v["matches_db"] is False
    assert any("crps" in d for d in v["discrepancies"])


def test_verify_endpoint_fetch_failure_surfaces_in_discrepancies():
    client, store, server = _server_with_store()
    elem = DataElement(
        index=0, round_id=42, miner_uid=5,
        substrate_cids=[{
            "kind": "phase_c_record", "validator_hotkey": "5G",
            "cid": "bafycid", "round_id": 42,
        }],
    )
    store._elements.append(elem)

    fake_hippius = MagicMock()
    fake_hippius.download_bundle = AsyncMock(side_effect=RuntimeError("ipfs flap"))
    server.set_hippius(fake_hippius)

    body = client.get("/experiments/0/verify").json()
    v = body["verifications"][0]
    assert v["fetchable"] is False
    assert v["signature_valid"] is False
    assert any("fetch failed" in d for d in v["discrepancies"])


def test_verify_endpoint_corrupt_bundle_flagged():
    client, store, server = _server_with_store()
    elem = DataElement(
        index=0, round_id=42, miner_uid=5,
        substrate_cids=[{
            "kind": "phase_c_record", "validator_hotkey": "5G",
            "cid": "bafycid", "round_id": 42,
        }],
    )
    store._elements.append(elem)

    fake_hippius = MagicMock()
    fake_hippius.download_bundle = AsyncMock(return_value=b"not a bundle")
    server.set_hippius(fake_hippius)

    body = client.get("/experiments/0/verify").json()
    v = body["verifications"][0]
    assert v["fetchable"] is True
    assert v["signature_valid"] is False
    assert any("parse failed" in d for d in v["discrepancies"])


# ── DatabaseNeuron._init_hippius wiring ──────────────────────────────


def test_database_neuron_init_hippius_disabled_returns_none():
    from database.neuron import DatabaseNeuron
    neuron = DatabaseNeuron.__new__(DatabaseNeuron)
    with patch("database.neuron.Config") as cfg:
        cfg.HIPPIUS_ENABLED = False
        assert neuron._init_hippius() is None


def test_database_neuron_init_hippius_missing_sdk_returns_none(caplog):
    from database.neuron import DatabaseNeuron
    neuron = DatabaseNeuron.__new__(DatabaseNeuron)
    import builtins
    real_import = builtins.__import__

    def _missing(name, *args, **kwargs):
        if name == "shared.hippius_client":
            raise ImportError("not yet")
        return real_import(name, *args, **kwargs)

    with patch("database.neuron.Config") as cfg, \
         patch("builtins.__import__", side_effect=_missing), \
         caplog.at_level("WARNING"):
        cfg.HIPPIUS_ENABLED = True
        cfg.HIPPIUS_IPFS_API_URL = ""
        cfg.HIPPIUS_KEY = ""
        cfg.HIPPIUS_SUBSTRATE_RPC = ""
        result = neuron._init_hippius()

    assert result is None
    assert any("not available yet" in m for m in caplog.messages)


# ── Sanity: bundle helper round-trip used by /verify ─────────────────


def test_records_from_bundle_handles_gzip_and_plain():
    """Reaffirm the publisher↔server contract that /verify relies on."""
    kp = bt.Keypair.create_from_seed(
        "0x" + hashlib.sha256(b"sanity").hexdigest(),
    )

    class _W:
        hotkey = kp

    rec = PhaseCRecord(
        schema_version=SCHEMA_VERSION,
        round_id=1, block_hash="b", task="t",
        miner_uid=1, miner_hotkey="h",
        code_hash="c", architecture_sha256="a", checkpoint_sha256="c",
        metrics={"x": 1}, passed_size_gate=True, flops_verified=True,
        eval_status="ok",
        validator_uid=0, validator_hotkey=kp.ss58_address,
        validator_block_height=1, timestamp=1.0,
    )
    signed = sign_record(rec, _W())
    gz = records_to_bundle([signed])
    assert gz[:2] == b"\x1f\x8b"
    plain = gzip.decompress(gz)
    from shared.substrate import records_from_bundle
    assert records_from_bundle(gz) == [signed]
    assert records_from_bundle(plain) == [signed]
