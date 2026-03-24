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

    def test_size_limit(self):
        """Scratchpad over 10MB should not be saved."""
        from miner_template.agent import save_scratchpad

        with tempfile.TemporaryDirectory() as d:
            Path(f"{d}/big.bin").write_bytes(b"x" * (11 * 1024 * 1024))
            challenge = {"scratchpad_put_url": "http://fake-url"}
            result = save_scratchpad(challenge, local_dir=d)
            assert result is False


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

    def test_challenge_without_scratchpad(self):
        """Backward compat — old challenges without scratchpad fields work."""
        from shared.protocol import Challenge

        c = Challenge.from_json('{"challenge_id": "test", "seed": 42}')
        assert c.scratchpad_get_url == ""
        assert c.scratchpad_put_url == ""
