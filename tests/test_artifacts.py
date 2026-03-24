"""Tests for shared.artifacts — upload, download, verify, path helpers."""

import os
import tempfile

from shared.artifacts import (
    TrainingMeta,
    checkpoint_key,
    architecture_key,
    meta_key,
    stdout_key,
    sha256_file,
    sha256_text,
    upload_training_artifacts,
    download_training_artifacts,
    list_round_artifacts,
    verify_uploaded_artifacts,
)


def test_path_helpers():
    assert checkpoint_key(42, "5Foo") == "round_42/miner_5Foo/checkpoint.safetensors"
    assert architecture_key(42, "5Foo") == "round_42/miner_5Foo/architecture.py"
    assert meta_key(42, "5Foo") == "round_42/miner_5Foo/training_meta.json"
    assert stdout_key(42, "5Foo") == "round_42/miner_5Foo/stdout.log"


def test_training_meta_roundtrip():
    meta = TrainingMeta(
        round_id=1, miner_hotkey="5Bar", status="success",
        flops_equivalent_size=500_000, training_time_seconds=120.5,
        num_steps=1000, num_params_M=2.5,
        loss_curve=[1.0, 0.8, 0.6], peak_vram_mb=1024.0,
        checkpoint_sha256="abc123", architecture_sha256="def456",
        stdout_sha256="ghi789",
    )
    text = meta.to_json()
    restored = TrainingMeta.from_json(text)
    assert restored.round_id == 1
    assert restored.miner_hotkey == "5Bar"
    assert restored.loss_curve == [1.0, 0.8, 0.6]
    assert restored.checkpoint_sha256 == "abc123"


def test_training_meta_from_dict_ignores_extra_keys():
    d = {"round_id": 5, "status": "success", "unknown_field": "ignored"}
    meta = TrainingMeta.from_dict(d)
    assert meta.round_id == 5
    assert meta.status == "success"


def test_sha256_text():
    h = sha256_text("hello world")
    assert len(h) == 64
    assert h == sha256_text("hello world")  # deterministic
    assert h != sha256_text("hello world!")


def test_sha256_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(b"test data for hashing")
        path = f.name
    try:
        h = sha256_file(path)
        assert len(h) == 64
        # Same content = same hash
        assert h == sha256_file(path)
    finally:
        os.unlink(path)


def test_upload_and_download(mock_r2):
    """Full upload → download → verify cycle."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"fake checkpoint data")
        ckpt_path = f.name

    try:
        meta = TrainingMeta(
            round_id=1, miner_hotkey="5Test", status="success",
            flops_equivalent_size=200_000, num_steps=500,
        )

        ok = upload_training_artifacts(
            r2=mock_r2,
            round_id=1,
            miner_hotkey="5Test",
            checkpoint_path=ckpt_path,
            architecture_code="def build_model(): pass",
            stdout_log="step 1 | loss: 0.5\nstep 2 | loss: 0.4\n",
            meta=meta,
        )
        assert ok
        assert meta.checkpoint_sha256  # hashes populated
        assert meta.architecture_sha256
        assert meta.stdout_sha256

        # Download
        with tempfile.TemporaryDirectory() as dl_dir:
            result = download_training_artifacts(mock_r2, 1, "5Test", dl_dir)
            assert result is not None
            assert result.verified
            assert result.verification_error == ""
            assert result.architecture_code == "def build_model(): pass"
            assert "loss: 0.5" in result.stdout_log
            assert os.path.exists(result.checkpoint_path)
    finally:
        os.unlink(ckpt_path)


def test_download_missing_meta(mock_r2):
    result = download_training_artifacts(mock_r2, 99, "nonexistent", "/tmp")
    assert result is None


def test_download_hash_mismatch(mock_r2):
    """Tampered checkpoint should fail verification."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"original data")
        ckpt_path = f.name

    try:
        meta = TrainingMeta(round_id=2, miner_hotkey="5Tamper", status="success")
        upload_training_artifacts(
            mock_r2, 2, "5Tamper", ckpt_path,
            "code", "stdout", meta,
        )

        # Tamper with checkpoint in R2
        ck = checkpoint_key(2, "5Tamper")
        mock_r2._s3.put_object(Bucket=mock_r2.bucket, Key=ck, Body=b"tampered data")

        with tempfile.TemporaryDirectory() as dl_dir:
            result = download_training_artifacts(mock_r2, 2, "5Tamper", dl_dir)
            assert result is not None
            assert not result.verified
            assert "checkpoint hash mismatch" in result.verification_error
    finally:
        os.unlink(ckpt_path)


def test_list_round_artifacts(mock_r2):
    # Upload artifacts for two miners
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"data")
        ckpt_path = f.name

    try:
        for hotkey in ["5Alice", "5Bob"]:
            meta = TrainingMeta(round_id=10, miner_hotkey=hotkey, status="success")
            upload_training_artifacts(
                mock_r2, 10, hotkey, ckpt_path,
                "code", "stdout", meta,
            )

        hotkeys = list_round_artifacts(mock_r2, 10)
        assert len(hotkeys) == 2
        assert "5Alice" in hotkeys
        assert "5Bob" in hotkeys

        # Different round should be empty
        assert list_round_artifacts(mock_r2, 99) == []
    finally:
        os.unlink(ckpt_path)


def test_list_round_artifacts_empty(mock_r2):
    assert list_round_artifacts(mock_r2, 1) == []


# ── verify_uploaded_artifacts tests ──────────────────────────────


def test_verify_uploaded_artifacts_success(mock_r2):
    """Valid upload passes verification."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"checkpoint data")
        ckpt_path = f.name

    try:
        meta = TrainingMeta(round_id=5, miner_hotkey="5Valid", status="success")
        upload_training_artifacts(
            mock_r2, 5, "5Valid", ckpt_path, "code", "stdout", meta,
        )

        ok, err = verify_uploaded_artifacts(mock_r2, 5, "5Valid")
        assert ok
        assert err == ""
    finally:
        os.unlink(ckpt_path)


def test_verify_uploaded_artifacts_missing_meta(mock_r2):
    """Missing meta file fails verification."""
    ok, err = verify_uploaded_artifacts(mock_r2, 99, "5Missing")
    assert not ok
    assert "training_meta.json missing" in err


def test_verify_uploaded_artifacts_missing_checkpoint(mock_r2):
    """Meta exists but checkpoint missing fails verification."""
    meta = TrainingMeta(round_id=7, miner_hotkey="5NoCkpt", status="success")
    mock_r2.upload_json(meta_key(7, "5NoCkpt"), meta.to_dict())

    ok, err = verify_uploaded_artifacts(mock_r2, 7, "5NoCkpt")
    assert not ok
    assert "checkpoint missing" in err


def test_verify_uploaded_artifacts_round_id_mismatch(mock_r2):
    """Meta with wrong round_id fails verification."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"data")
        ckpt_path = f.name

    try:
        # Upload with round_id=3 but meta claims round_id=999
        meta = TrainingMeta(round_id=999, miner_hotkey="5Wrong", status="success")
        upload_training_artifacts(
            mock_r2, 3, "5Wrong", ckpt_path, "code", "stdout", meta,
        )
        # Meta was uploaded with round_id=999 inside the JSON

        ok, err = verify_uploaded_artifacts(mock_r2, 3, "5Wrong")
        assert not ok
        assert "round_id mismatch" in err
    finally:
        os.unlink(ckpt_path)


def test_verify_uploaded_artifacts_hotkey_mismatch(mock_r2):
    """Meta with wrong miner_hotkey fails verification."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"data")
        ckpt_path = f.name

    try:
        meta = TrainingMeta(round_id=4, miner_hotkey="5Impostor", status="success")
        # Upload to 5Real's path but meta says 5Impostor
        ck = checkpoint_key(4, "5Real")
        mk = meta_key(4, "5Real")
        mock_r2.upload_file_from_disk(ckpt_path, ck)
        mock_r2.upload_json(mk, meta.to_dict())

        ok, err = verify_uploaded_artifacts(mock_r2, 4, "5Real")
        assert not ok
        assert "miner_hotkey mismatch" in err
    finally:
        os.unlink(ckpt_path)
