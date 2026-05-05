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
    """Bucket paths use the opaque submission_id, not miner_hotkey."""
    sid = "abc123"
    assert checkpoint_key(42, sid) == "round_42/submission_abc123/checkpoint.safetensors"
    assert architecture_key(42, sid) == "round_42/submission_abc123/architecture.py"
    assert meta_key(42, sid) == "round_42/submission_abc123/training_meta.json"
    assert stdout_key(42, sid) == "round_42/submission_abc123/stdout.log"


def test_path_helpers_reject_path_traversal():
    """Submission IDs with unsafe characters must be rejected."""
    import pytest
    for bad in ("../oops", "a/b", "abc def", ""):
        with pytest.raises(ValueError):
            checkpoint_key(1, bad)


def test_training_meta_roundtrip():
    meta = TrainingMeta(
        round_id=1, submission_id="sid_bar", status="success",
        flops_equivalent_size=500_000, training_time_seconds=120.5,
        num_steps=1000, num_params_M=2.5,
        peak_vram_mb=1024.0,
        train_loss_history=[{"step": 10, "loss": 1.0}, {"step": 20, "loss": 0.8}],
        val_loss_history=[{"step": 10, "loss": 0.9}],
        best_val_loss=0.9, best_val_step=10,
        checkpoint_sha256="abc123", architecture_sha256="def456",
        stdout_sha256="ghi789",
    )
    text = meta.to_json()
    restored = TrainingMeta.from_json(text)
    assert restored.round_id == 1
    assert restored.submission_id == "sid_bar"
    assert restored.train_loss_history == [{"step": 10, "loss": 1.0}, {"step": 20, "loss": 0.8}]
    assert restored.val_loss_history == [{"step": 10, "loss": 0.9}]
    assert restored.best_val_loss == 0.9
    assert restored.best_val_step == 10
    assert restored.checkpoint_sha256 == "abc123"


def test_training_meta_old_format_without_new_fields():
    """Old metas (pre-loss-tracking) must still parse via from_dict."""
    old = {
        "round_id": 7, "submission_id": "sid_old", "status": "success",
        "loss_curve": [1.0, 0.5],  # legacy field — should be silently dropped
        "num_steps": 100,
    }
    meta = TrainingMeta.from_dict(old)
    assert meta.round_id == 7
    assert meta.num_steps == 100
    assert meta.train_loss_history == []
    assert meta.val_loss_history == []
    assert meta.best_val_loss is None
    assert meta.best_val_step == -1
    # New schedule-policy fields default empty so legacy metas deserialize cleanly.
    assert meta.val_cadence_unit == "step"
    assert meta.val_base == 0.0
    assert meta.val_growth == 0.0
    assert meta.val_eval_tokens == 0
    assert meta.flops_per_step_estimate == 0.0
    assert meta.reference_eval_loss_history == []


def test_training_meta_roundtrip_with_schedule_policy_fields():
    """All new policy fields round-trip through to_dict/from_dict/to_json/from_json."""
    train_hist = [
        {"step": 100, "flops": 6_000_000, "loss": 1.5},
        {"step": 200, "flops": 12_000_000, "loss": 1.0},
    ]
    val_hist = [
        {"step": 100, "flops": 6_000_000, "loss": 1.7},
        {"step": 200, "flops": 12_000_000, "loss": 1.2},
    ]
    ref_hist = [
        {"step": 100, "flops": 6_000_000, "loss": 1.9},
    ]
    meta = TrainingMeta(
        round_id=11, submission_id="sid_sched", status="success",
        train_loss_history=train_hist,
        val_loss_history=val_hist,
        best_val_loss=1.2, best_val_step=200,
        val_cadence_unit="flops",
        val_base=1e15,
        val_growth=2.0,
        val_eval_tokens=4096,
        flops_per_step_estimate=6e8,
        reference_eval_loss_history=ref_hist,
    )

    restored = TrainingMeta.from_dict(meta.to_dict())
    assert restored.val_cadence_unit == "flops"
    assert restored.val_base == 1e15
    assert restored.val_growth == 2.0
    assert restored.val_eval_tokens == 4096
    assert restored.flops_per_step_estimate == 6e8
    assert restored.reference_eval_loss_history == ref_hist
    assert restored.train_loss_history == train_hist
    assert restored.val_loss_history == val_hist

    restored2 = TrainingMeta.from_json(meta.to_json())
    assert restored2.val_cadence_unit == "flops"
    assert restored2.val_base == 1e15
    assert restored2.val_growth == 2.0
    assert restored2.val_eval_tokens == 4096
    assert restored2.flops_per_step_estimate == 6e8
    assert restored2.reference_eval_loss_history == ref_hist
    assert restored2.train_loss_history == train_hist
    assert restored2.val_loss_history == val_hist


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
    """Full upload → download → verify cycle, keyed by submission_id."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"fake checkpoint data")
        ckpt_path = f.name

    try:
        sid = "sid_test"
        meta = TrainingMeta(
            round_id=1, submission_id=sid, status="success",
            flops_equivalent_size=200_000, num_steps=500,
        )

        ok = upload_training_artifacts(
            r2=mock_r2,
            round_id=1,
            submission_id=sid,
            checkpoint_path=ckpt_path,
            architecture_code="def build_model(): pass",
            stdout_log="step 1 | loss: 0.5\nstep 2 | loss: 0.4\n",
            meta=meta,
        )
        assert ok
        assert meta.checkpoint_sha256
        assert meta.architecture_sha256
        assert meta.stdout_sha256

        with tempfile.TemporaryDirectory() as dl_dir:
            result = download_training_artifacts(mock_r2, 1, sid, dl_dir)
            assert result is not None
            assert result.verified
            assert result.verification_error == ""
            assert result.architecture_code == "def build_model(): pass"
            assert "loss: 0.5" in result.stdout_log
            assert os.path.exists(result.checkpoint_path)
    finally:
        os.unlink(ckpt_path)


def test_download_missing_meta(mock_r2):
    result = download_training_artifacts(mock_r2, 99, "sid_nonexistent", "/tmp")
    assert result is None


def test_download_hash_mismatch(mock_r2):
    """Tampered checkpoint should fail verification."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"original data")
        ckpt_path = f.name

    try:
        sid = "sid_tamper"
        meta = TrainingMeta(round_id=2, submission_id=sid, status="success")
        upload_training_artifacts(
            mock_r2, 2, sid, ckpt_path,
            "code", "stdout", meta,
        )

        ck = checkpoint_key(2, sid)
        mock_r2._s3.put_object(Bucket=mock_r2.bucket, Key=ck, Body=b"tampered data")

        with tempfile.TemporaryDirectory() as dl_dir:
            result = download_training_artifacts(mock_r2, 2, sid, dl_dir)
            assert result is not None
            assert not result.verified
            assert "checkpoint hash mismatch" in result.verification_error
    finally:
        os.unlink(ckpt_path)


def test_list_round_artifacts(mock_r2):
    """list_round_artifacts returns the submission_ids present for a round."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"data")
        ckpt_path = f.name

    try:
        for sid in ["sid_alice", "sid_bob"]:
            meta = TrainingMeta(round_id=10, submission_id=sid, status="success")
            upload_training_artifacts(
                mock_r2, 10, sid, ckpt_path,
                "code", "stdout", meta,
            )

        sids = list_round_artifacts(mock_r2, 10)
        assert len(sids) == 2
        assert "sid_alice" in sids
        assert "sid_bob" in sids

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
        sid = "sid_valid"
        meta = TrainingMeta(round_id=5, submission_id=sid, status="success")
        upload_training_artifacts(
            mock_r2, 5, sid, ckpt_path, "code", "stdout", meta,
        )

        ok, err = verify_uploaded_artifacts(mock_r2, 5, sid)
        assert ok
        assert err == ""
    finally:
        os.unlink(ckpt_path)


def test_verify_uploaded_artifacts_missing_meta(mock_r2):
    """Missing meta file fails verification."""
    ok, err = verify_uploaded_artifacts(mock_r2, 99, "sid_missing")
    assert not ok
    assert "training_meta.json missing" in err


def test_verify_uploaded_artifacts_missing_checkpoint(mock_r2):
    """Meta exists but checkpoint missing fails verification."""
    sid = "sid_no_ckpt"
    meta = TrainingMeta(round_id=7, submission_id=sid, status="success")
    mock_r2.upload_json(meta_key(7, sid), meta.to_dict())

    ok, err = verify_uploaded_artifacts(mock_r2, 7, sid)
    assert not ok
    assert "checkpoint missing" in err


def test_verify_uploaded_artifacts_round_id_mismatch(mock_r2):
    """Meta with wrong round_id fails verification."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"data")
        ckpt_path = f.name

    try:
        sid = "sid_wrong"
        meta = TrainingMeta(round_id=999, submission_id=sid, status="success")
        upload_training_artifacts(
            mock_r2, 3, sid, ckpt_path, "code", "stdout", meta,
        )

        ok, err = verify_uploaded_artifacts(mock_r2, 3, sid)
        assert not ok
        assert "round_id mismatch" in err
    finally:
        os.unlink(ckpt_path)


def test_verify_uploaded_artifacts_submission_id_mismatch(mock_r2):
    """Meta with wrong submission_id fails verification."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"data")
        ckpt_path = f.name

    try:
        # Meta claims sid_impostor, but we upload to sid_real's path
        meta = TrainingMeta(round_id=4, submission_id="sid_impostor", status="success")
        ck = checkpoint_key(4, "sid_real")
        mk = meta_key(4, "sid_real")
        mock_r2.upload_file_from_disk(ckpt_path, ck)
        mock_r2.upload_json(mk, meta.to_dict())

        ok, err = verify_uploaded_artifacts(mock_r2, 4, "sid_real")
        assert not ok
        assert "submission_id mismatch" in err
    finally:
        os.unlink(ckpt_path)


def test_verify_uploaded_artifacts_legacy_meta_without_submission_id(mock_r2):
    """Older trainer images upload metas without ``submission_id``.

    The path is path-locked by the presigned PUT URL, so the meta is
    already authenticated by its location. Accept it instead of failing
    the round on a missing field.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        f.write(b"data")
        ckpt_path = f.name

    try:
        sid = "sid_legacy"
        # Legacy meta dict — has miner_hotkey, no submission_id field
        legacy_meta = {
            "round_id": 9,
            "miner_hotkey": "5HpLegacyHotkey",
            "status": "success",
            "flops_equivalent_size": 500_000,
        }
        ck = checkpoint_key(9, sid)
        mk = meta_key(9, sid)
        mock_r2.upload_file_from_disk(ckpt_path, ck)
        mock_r2.upload_json(mk, legacy_meta)

        ok, err = verify_uploaded_artifacts(mock_r2, 9, sid)
        assert ok, f"expected legacy meta to verify, got error: {err}"
        assert err == ""
    finally:
        os.unlink(ckpt_path)
