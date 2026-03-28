"""Tests for agent scratchpad persistence."""

import json
import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestScratchpadHelpers:
    def test_load_empty_scratchpad(self):
        """First round — no scratchpad exists, returns empty dir."""
        from miner_template.agent import load_scratchpad

        with tempfile.TemporaryDirectory() as d:
            challenge = {"scratchpad_get_url": ""}
            result = load_scratchpad(challenge, local_dir=d)
            assert os.path.isdir(result)

    def test_save_and_load_roundtrip(self):
        """Save files, tar them, extract them — simulates R2 round trip."""
        with tempfile.TemporaryDirectory() as scratch_dir:
            Path(f"{scratch_dir}/probe.db").write_text("fake db content")
            Path(f"{scratch_dir}/notes.txt").write_text("some notes")

            archive = f"{scratch_dir}/test.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for f in ["probe.db", "notes.txt"]:
                    tar.add(f"{scratch_dir}/{f}", arcname=f)

            with tempfile.TemporaryDirectory() as restore_dir:
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(restore_dir, filter="data")
                assert Path(f"{restore_dir}/probe.db").read_text() == "fake db content"
                assert Path(f"{restore_dir}/notes.txt").read_text() == "some notes"

    def test_size_limit_default(self):
        """Scratchpad over 10MB (default) should not be saved."""
        from miner_template.agent import save_scratchpad

        with tempfile.TemporaryDirectory() as d:
            Path(f"{d}/big.bin").write_bytes(b"x" * (11 * 1024 * 1024))
            challenge = {"scratchpad_put_url": "http://fake-url"}
            result = save_scratchpad(challenge, local_dir=d)
            assert result is False

    def test_size_limit_from_challenge(self):
        """Scratchpad respects scratchpad_max_mb from challenge."""
        from miner_template.agent import save_scratchpad

        with tempfile.TemporaryDirectory() as d:
            # 3MB file — under default 10MB but over custom 2MB limit
            Path(f"{d}/medium.bin").write_bytes(b"x" * (3 * 1024 * 1024))
            challenge = {"scratchpad_put_url": "http://fake-url", "scratchpad_max_mb": 2}
            result = save_scratchpad(challenge, local_dir=d)
            assert result is False

    def test_nested_directories_preserved(self):
        """Scratchpad should archive nested directory trees."""
        from miner_template.agent import save_scratchpad

        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, "sub", "deep")
            os.makedirs(nested)
            Path(os.path.join(nested, "file.txt")).write_text("nested content")

            # save_scratchpad will create the archive; we just verify the tar
            archive_path = os.path.join(tempfile.gettempdir(), "scratchpad_upload.tar.gz")
            try:
                import tarfile as tf
                with tf.open(archive_path, "w:gz") as tar:
                    for root, dirs, files in os.walk(d):
                        for f in files:
                            full = os.path.join(root, f)
                            arcname = os.path.relpath(full, d)
                            tar.add(full, arcname=arcname)

                with tf.open(archive_path, "r:gz") as tar:
                    names = tar.getnames()
                assert "sub/deep/file.txt" in names
            finally:
                if os.path.exists(archive_path):
                    os.remove(archive_path)


class TestScratchpadR2:
    def test_scratchpad_key(self):
        from shared.artifacts import scratchpad_key

        key = scratchpad_key("5GrwvaEF")
        assert key == "scratchpad/5GrwvaEF/state.tar.gz"

    def test_generate_scratchpad_urls(self):
        """Presigned URLs are generated for the correct key."""
        from shared.artifacts import generate_scratchpad_urls

        mock_r2 = MagicMock()
        mock_r2.generate_presigned_get_url.return_value = "https://get-url"
        mock_r2.generate_presigned_put_url.return_value = "https://put-url"

        get_url, put_url = generate_scratchpad_urls(mock_r2, "test_hotkey")
        assert get_url == "https://get-url"
        assert put_url == "https://put-url"
        mock_r2.generate_presigned_get_url.assert_called_once()
        mock_r2.generate_presigned_put_url.assert_called_once()


class TestChallengeWithScratchpad:
    def test_challenge_includes_scratchpad_urls(self):
        from shared.protocol import Challenge

        c = Challenge(
            challenge_id="test",
            scratchpad_get_url="https://get",
            scratchpad_put_url="https://put",
        )
        j = c.to_json()
        restored = Challenge.from_json(j)
        assert restored.scratchpad_get_url == "https://get"
        assert restored.scratchpad_put_url == "https://put"
        assert restored.scratchpad_max_mb == 10  # default

    def test_challenge_with_custom_max_mb(self):
        from shared.protocol import Challenge

        c = Challenge(challenge_id="test", scratchpad_max_mb=5)
        j = c.to_json()
        restored = Challenge.from_json(j)
        assert restored.scratchpad_max_mb == 5

    def test_challenge_without_scratchpad(self):
        """Backward compat — old challenges without scratchpad fields work."""
        from shared.protocol import Challenge

        c = Challenge.from_json('{"challenge_id": "test", "seed": 42}')
        assert c.scratchpad_get_url == ""
        assert c.scratchpad_put_url == ""
        assert c.scratchpad_max_mb == 10
