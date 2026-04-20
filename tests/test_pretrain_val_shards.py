"""Val-shard plumbing tests: manifest → validator → coordinator payload → trainer env.

Covers the end-to-end wiring so the val pretrain split stays fixed across rounds
and is actually delivered to the pod as RADAR_PRETRAIN_VAL_SHARD_URLS.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.pretrain_data import PretrainBenchmark
from shared.protocol import Proposal
from validator.coordinator import Job, TrainingCoordinator


class _ManifestR2:
    """Minimal R2 stub that hands back a preset manifest + presigned URLs."""

    bucket = "test-bucket"

    def __init__(self, manifest: dict):
        self._manifest = manifest
        self.presign_calls: list[str] = []

    def download_json(self, key):
        return self._manifest

    def generate_presigned_get_url(self, key, ttl=900):
        self.presign_calls.append(key)
        return f"https://example.com/{key}?ttl={ttl}"


# ── PretrainBenchmark.get_val_shard_keys ──────────────────────────


def test_get_val_shard_keys_reads_manifest():
    manifest = {
        "shards": [
            {"s3_key": f"datasets/radar/v1/shard_{i:05d}.parquet"} for i in range(10)
        ],
        "val_shard_keys": [
            "datasets/radar/v1/val_000.parquet",
            "datasets/radar/v1/val_001.parquet",
        ],
    }
    bench = PretrainBenchmark(r2=_ManifestR2(manifest), r2_prefix="datasets/radar/v1")
    keys = bench.get_val_shard_keys()
    assert keys == [
        "datasets/radar/v1/val_000.parquet",
        "datasets/radar/v1/val_001.parquet",
    ]


def test_get_val_shard_keys_empty_for_old_manifest():
    """Manifests without val_shard_keys return [] (backwards compatible)."""
    manifest = {
        "shards": [{"s3_key": "datasets/radar/v1/shard_0.parquet"}],
    }
    bench = PretrainBenchmark(r2=_ManifestR2(manifest), r2_prefix="datasets/radar/v1")
    assert bench.get_val_shard_keys() == []


def test_select_shards_excludes_val_keys():
    """Val keys declared in the manifest must never land in the training pool."""
    val_keys = [
        "datasets/radar/v1/shard_00000.parquet",
        "datasets/radar/v1/shard_00001.parquet",
    ]
    manifest = {
        "shards": [
            {"s3_key": f"datasets/radar/v1/shard_{i:05d}.parquet"} for i in range(20)
        ],
        "val_shard_keys": val_keys,
    }
    bench = PretrainBenchmark(r2=_ManifestR2(manifest), r2_prefix="datasets/radar/v1")

    for seed in (0, 1, 42, 999, 123456):
        chosen = bench.select_shards(seed=seed, n=5)
        assert len(chosen) == 5
        for k in val_keys:
            assert k not in chosen


def test_select_shards_warns_on_overlap(caplog):
    """If the manifest accidentally lists a val key under shards too, we warn."""
    val_keys = ["datasets/radar/v1/shard_00000.parquet"]
    manifest = {
        "shards": [
            {"s3_key": f"datasets/radar/v1/shard_{i:05d}.parquet"} for i in range(5)
        ],
        "val_shard_keys": val_keys,
    }
    bench = PretrainBenchmark(r2=_ManifestR2(manifest), r2_prefix="datasets/radar/v1")
    with caplog.at_level(logging.WARNING, logger="shared.pretrain_data"):
        chosen = bench.select_shards(seed=7, n=3)
    assert val_keys[0] not in chosen
    assert any("overlapping" in rec.message for rec in caplog.records)


def test_val_shard_keys_stable_across_seeds():
    """Val keys are fixed across rounds — they don't depend on the seed."""
    manifest = {
        "shards": [
            {"s3_key": f"datasets/radar/v1/shard_{i:05d}.parquet"} for i in range(10)
        ],
        "val_shard_keys": [
            "datasets/radar/v1/val_a.parquet",
            "datasets/radar/v1/val_b.parquet",
        ],
    }
    bench = PretrainBenchmark(r2=_ManifestR2(manifest), r2_prefix="datasets/radar/v1")
    a = bench.get_val_shard_keys()
    _ = bench.select_shards(seed=1, n=3)
    b = bench.get_val_shard_keys()
    _ = bench.select_shards(seed=12345, n=3)
    c = bench.get_val_shard_keys()
    assert a == b == c


# ── Coordinator payload threading ─────────────────────────────────


def _make_coordinator():
    return TrainingCoordinator(
        wallet=MagicMock(),
        metagraph=MagicMock(hotkeys=["hk0", "hk1"]),
        r2=MagicMock(),
        my_uid=10,
    )


def _make_challenge():
    return MagicMock(
        seed=42, round_id=1, min_flops_equivalent=0,
        max_flops_equivalent=1_000_000,
        task={"time_budget": 300, "name": "ts", "runner_dir": "runner/timeseries_forecast"},
    )


@pytest.mark.asyncio
async def test_dispatch_includes_val_urls_when_provided():
    coord = _make_coordinator()
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}
    val_urls = [
        "https://example.com/val_0?sig=x",
        "https://example.com/val_1?sig=y",
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 202
    mock_resp.json.return_value = {"status": "accepted"}
    mock_post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient") as mock_client, \
         patch("shared.artifacts.generate_upload_urls", return_value={"checkpoint": "http://fake/ckpt"}), \
         patch("validator.coordinator.sign_request", return_value={"X-Epistula-Signed-By": "hk0"}):
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=mock_post))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        await coord.dispatch_jobs(
            jobs, _make_challenge(), submissions, endpoints,
            extras={"pretrain_val_shard_urls": val_urls},
        )

    assert mock_post.await_count == 1
    sent_payload = json.loads(mock_post.await_args.kwargs["content"])
    assert sent_payload["pretrain_val_shard_urls"] == val_urls


@pytest.mark.asyncio
async def test_dispatch_omits_val_key_when_empty():
    """If no val URLs are passed, the payload must not carry the key at all."""
    coord = _make_coordinator()
    jobs = [Job(arch_owner=0, trainer_uid=1, dispatcher=10, round_id=1)]
    submissions = {0: Proposal(code="code_a")}
    endpoints = {1: "http://trainer:8080"}

    mock_resp = MagicMock()
    mock_resp.status_code = 202
    mock_resp.json.return_value = {"status": "accepted"}
    mock_post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient") as mock_client, \
         patch("shared.artifacts.generate_upload_urls", return_value={"checkpoint": "http://fake/ckpt"}), \
         patch("validator.coordinator.sign_request", return_value={"X-Epistula-Signed-By": "hk0"}):
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=mock_post))
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        await coord.dispatch_jobs(
            jobs, _make_challenge(), submissions, endpoints,
        )

    sent_payload = json.loads(mock_post.await_args.kwargs["content"])
    assert "pretrain_val_shard_urls" not in sent_payload


# ── Trainer-side env var wiring (runner/server.py) ────────────────


@pytest.mark.asyncio
async def test_trainer_server_sets_val_env_var():
    """runner/server.py::_train_and_upload publishes val URLs via env var."""
    from runner import server as rs

    val_urls = ["https://example.com/val_a", "https://example.com/val_b"]
    captured_env: dict[str, str | None] = {}

    def fake_runner(arch_code, training_config):
        captured_env["RADAR_PRETRAIN_VAL_SHARD_URLS"] = os.environ.get(
            "RADAR_PRETRAIN_VAL_SHARD_URLS",
        )
        return {"status": "success", "flops_equivalent_size": 0}

    os.environ.pop("RADAR_PRETRAIN_VAL_SHARD_URLS", None)
    try:
        with patch.object(rs, "_upload_artifacts") as mock_upload:
            await rs._train_and_upload(
                fake_runner,
                "architecture_code",
                {"round_id": 1, "miner_hotkey": "5Test"},
                upload_urls={},
                gift_eval_urls={},
                pretrain_shard_urls=None,
                pretrain_val_shard_urls=val_urls,
            )
            assert mock_upload.called

        assert captured_env["RADAR_PRETRAIN_VAL_SHARD_URLS"] == json.dumps(val_urls)
    finally:
        os.environ.pop("RADAR_PRETRAIN_VAL_SHARD_URLS", None)


@pytest.mark.asyncio
async def test_trainer_server_clears_val_env_var_when_unset():
    """When no val URLs are provided, the env var must be cleared."""
    from runner import server as rs

    os.environ["RADAR_PRETRAIN_VAL_SHARD_URLS"] = "stale"

    def fake_runner(arch_code, training_config):
        # Env var must be absent inside the runner
        return {"status": "success", "flops_equivalent_size": 0}

    captured: dict[str, str | None] = {}

    def fake_runner_capture(arch_code, training_config):
        captured["val"] = os.environ.get("RADAR_PRETRAIN_VAL_SHARD_URLS")
        return {"status": "success", "flops_equivalent_size": 0}

    try:
        with patch.object(rs, "_upload_artifacts"):
            await rs._train_and_upload(
                fake_runner_capture,
                "architecture_code",
                {"round_id": 1, "miner_hotkey": "5Test"},
                upload_urls={},
                gift_eval_urls={},
                pretrain_shard_urls=None,
                pretrain_val_shard_urls=None,
            )

        assert captured["val"] is None
    finally:
        os.environ.pop("RADAR_PRETRAIN_VAL_SHARD_URLS", None)
