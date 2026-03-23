"""Tests for on-chain image commitment."""

import json
from unittest.mock import patch, MagicMock
import pytest
from shared.commitment import (
    ImageCommitment, verify_subnet_version,
    pull_and_verify_image, _image_exists_locally,
)


class TestImageCommitment:
    def test_roundtrip(self):
        c = ImageCommitment(image_url="docker.io/test:v1", image_digest="sha256:abc")
        j = c.to_json()
        c2 = ImageCommitment.from_json(j, miner_uid=5)
        assert c2.image_url == "docker.io/test:v1"
        assert c2.image_digest == "sha256:abc"
        assert c2.miner_uid == 5

    def test_is_valid(self):
        assert ImageCommitment(image_url="x", image_digest="y").is_valid
        assert not ImageCommitment(image_url="", image_digest="y").is_valid
        assert not ImageCommitment(image_url="x", image_digest="").is_valid
        assert not ImageCommitment().is_valid

    def test_version_check(self):
        c = ImageCommitment(subnet_version="0.1.0")
        assert verify_subnet_version(c, "0.1.0")
        assert not verify_subnet_version(c, "0.2.0")
        # Empty version = skip check
        assert verify_subnet_version(ImageCommitment(), "0.1.0")

    def test_from_json_missing_fields(self):
        """Gracefully handle missing fields in JSON."""
        c = ImageCommitment.from_json('{"image_url": "test"}')
        assert c.image_url == "test"
        assert c.image_digest == ""
        assert c.subnet_version == ""

    def test_to_json_excludes_internal_fields(self):
        """to_json should not include miner_uid or hotkey."""
        c = ImageCommitment(
            image_url="x", image_digest="y",
            miner_uid=5, hotkey="abc",
        )
        d = json.loads(c.to_json())
        assert "miner_uid" not in d
        assert "hotkey" not in d
        assert d["image_url"] == "x"


class TestPullAndVerifyImage:
    """Tests for pull_and_verify_image with local-first logic."""

    @patch("shared.commitment.subprocess.run")
    def test_skips_pull_when_image_exists_locally(self, mock_run):
        """Should not docker pull if image already exists in local daemon."""
        # First call: docker image inspect (exists locally)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        c = ImageCommitment(image_url="agent-failure:latest", image_digest="")
        result = pull_and_verify_image(c)
        assert result is True
        # Should only call inspect, never pull
        calls = [call.args[0] for call in mock_run.call_args_list]
        assert any("image" in cmd and "inspect" in cmd for cmd in calls)
        assert not any("pull" in cmd for cmd in calls)

    @patch("shared.commitment.subprocess.run")
    def test_pulls_when_image_not_local(self, mock_run):
        """Should docker pull if image is not available locally."""
        def side_effect(cmd, **kwargs):
            m = MagicMock()
            if "image" in cmd and "inspect" in cmd:
                m.returncode = 1  # not local
            elif "pull" in cmd:
                m.returncode = 0
                m.check_returncode = MagicMock()
            else:
                m.returncode = 0
                m.stdout = ""
            return m

        mock_run.side_effect = side_effect
        c = ImageCommitment(
            image_url="docker.io/user/agent:v1", image_digest="",
        )
        result = pull_and_verify_image(c)
        assert result is True

    @patch("shared.commitment.subprocess.run")
    def test_local_image_no_digest_skips_verify(self, mock_run):
        """Local images with no committed digest should pass without verify."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        c = ImageCommitment(
            image_url="agent-systematic:latest", image_digest="",
        )
        result = pull_and_verify_image(c)
        assert result is True
