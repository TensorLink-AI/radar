"""Tests for on-chain image commitment."""

import json
from unittest.mock import patch, MagicMock
import pytest
from shared.commitment import (
    ImageCommitment, verify_subnet_version,
    pull_and_verify_image, _image_exists_locally,
    _decode_commitment_raw, read_miner_commitments,
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
        assert ImageCommitment(image_url="x", listener_url="http://localhost:8090").is_valid
        assert not ImageCommitment(image_url="", listener_url="http://localhost:8090").is_valid
        assert not ImageCommitment(image_url="x", listener_url="").is_valid
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


class TestDecodeCommitmentRaw:
    """Tests for raw metadata decoding when SDK fails on Raw* variants."""

    def test_standard_raw_dict(self):
        """Decode metadata with Raw* key containing byte list."""
        commitment_json = '{"image_url": "test:latest", "image_digest": ""}'
        byte_list = list(commitment_json.encode("utf-8"))
        metadata = {"info": {"fields": [{"Raw279": [byte_list]}]}}
        result = _decode_commitment_raw(metadata)
        assert result == commitment_json

    def test_nested_tuple_format(self):
        """Decode metadata where fields[0] is a list/tuple."""
        commitment_json = '{"image_url": "x"}'
        byte_list = list(commitment_json.encode("utf-8"))
        metadata = {"info": {"fields": [[{"Raw18": [byte_list]}]]}}
        result = _decode_commitment_raw(metadata)
        assert result == commitment_json

    def test_flat_byte_list(self):
        """Decode metadata where Raw* value is a flat byte list (no nesting)."""
        commitment_json = '{"image_url": "x"}'
        byte_list = list(commitment_json.encode("utf-8"))
        metadata = {"info": {"fields": [{"Raw18": byte_list}]}}
        result = _decode_commitment_raw(metadata)
        assert result == commitment_json

    def test_empty_metadata(self):
        assert _decode_commitment_raw({}) == ""
        assert _decode_commitment_raw({"info": {}}) == ""
        assert _decode_commitment_raw({"info": {"fields": []}}) == ""

    def test_read_miner_commitments_chain_with_raw_fallback(self):
        """Validator reads commitments even when SDK decode_metadata fails."""
        commitment_json = ImageCommitment(
            image_url="ghcr.io/test:latest",
            listener_url="http://1.2.3.4:8090",
        ).to_json()
        byte_list = list(commitment_json.encode("utf-8"))
        raw_metadata = {"info": {"fields": [{"Raw200": [byte_list]}]}}

        mock_sub = MagicMock()
        mock_sub.get_commitment.return_value = ""  # SDK decode fails
        mock_sub.get_commitment_metadata.return_value = raw_metadata

        mock_mg = MagicMock()
        mock_mg.n = 1
        mock_mg.hotkeys = ["5FakeHotkey"]

        result = read_miner_commitments(mock_sub, 279, mock_mg)
        assert len(result) == 1
        assert result[0].image_url == "ghcr.io/test:latest"
        assert result[0].listener_url == "http://1.2.3.4:8090"


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
