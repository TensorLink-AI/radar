"""Tests for on-chain image commitment."""

import json
from unittest.mock import patch, MagicMock
import pytest
from shared.commitment import (
    ImageCommitment, verify_subnet_version,
    pull_and_verify_image, _image_exists_locally,
    read_miner_commitments, _read_all_commitments, _read_per_uid_commitments,
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
        assert ImageCommitment(code_hash="sha256:abc", listener_url="http://localhost:8090").is_valid
        assert ImageCommitment(image_url="x", listener_url="http://localhost:8090").is_valid
        assert not ImageCommitment(image_url="", listener_url="http://localhost:8090").is_valid
        assert not ImageCommitment(code_hash="sha256:abc", listener_url="").is_valid
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
        # Compact format uses short key "i" for image_url
        assert d["i"] == "x"


    def test_to_chain_json_under_128_bytes(self):
        """Chain JSON must fit within Raw128 (128 bytes)."""
        c = ImageCommitment(
            code_hash="sha256:abc123def456",
            subnet_version="0.3.0",
            listener_url="http://123.456.789.012:8090",
            trainer_image="ghcr.io/tensorlink-ai/radar/radar-runner:latest",
        )
        chain = c.to_chain_json()
        assert len(chain) <= 128
        d = json.loads(chain)
        # Essential fields present
        assert d["ch"] == c.code_hash
        assert d["l"] == c.listener_url
        # Non-essential fields excluded from chain JSON
        assert "i" not in d   # image_url (deprecated)
        assert "d" not in d   # image_digest
        assert "t" not in d   # trainer_image

    def test_to_chain_json_drops_version_if_over_limit(self):
        """Version should be dropped if needed to stay under 128 bytes."""
        c = ImageCommitment(
            code_hash="sha256:abcdef0123456789abcdef0123456789abcdef01234567",
            subnet_version="0.3.0",
            listener_url="http://very-long-hostname.example.com:8090",
        )
        chain = c.to_chain_json()
        assert len(chain) <= 128
        d = json.loads(chain)
        assert d["ch"] == c.code_hash
        assert d["l"] == c.listener_url

    def test_to_chain_json_roundtrips_via_from_json(self):
        """Chain JSON should be parseable by from_json."""
        c = ImageCommitment(
            code_hash="sha256:abc123",
            listener_url="http://host:8090",
            subnet_version="0.3.0",
        )
        c2 = ImageCommitment.from_json(c.to_chain_json(), miner_uid=5)
        assert c2.code_hash == c.code_hash
        assert c2.listener_url == c.listener_url
        assert c2.miner_uid == 5


class TestReadMinerCommitments:
    """Tests for reading commitments from chain."""

    def _make_metagraph(self, n=3, hotkeys=None, validator_permits=None):
        mg = MagicMock()
        mg.n = n
        mg.hotkeys = hotkeys or [f"hotkey_{i}" for i in range(n)]
        mg.validator_permit = validator_permits or [False] * n
        return mg

    def test_read_all_commitments_parses_sdk_output(self):
        """get_all_commitments returns {hotkey: json_str}, should parse correctly."""
        commitment = ImageCommitment(
            image_url="docker.io/test:v1", listener_url="http://host:8090",
        )
        subtensor = MagicMock()
        subtensor.get_all_commitments.return_value = {
            "hotkey_1": commitment.to_json(),
        }
        result = _read_all_commitments(
            subtensor, netuid=1,
            hotkey_to_uid={"hotkey_0": 0, "hotkey_1": 1},
            miner_uids={0, 1},
        )
        assert 1 in result
        assert result[1].image_url == "docker.io/test:v1"
        assert result[1].listener_url == "http://host:8090"
        assert result[1].miner_uid == 1
        assert result[1].hotkey == "hotkey_1"

    def test_read_all_commitments_skips_validators(self):
        """Commitments from validator UIDs should be skipped."""
        subtensor = MagicMock()
        subtensor.get_all_commitments.return_value = {
            "hotkey_0": ImageCommitment(image_url="x", listener_url="y").to_json(),
        }
        result = _read_all_commitments(
            subtensor, netuid=1,
            hotkey_to_uid={"hotkey_0": 0},
            miner_uids={1, 2},  # UID 0 is not a miner
        )
        assert len(result) == 0

    def test_read_all_commitments_handles_exception(self):
        """Should return empty dict if get_all_commitments raises."""
        subtensor = MagicMock()
        subtensor.get_all_commitments.side_effect = Exception("RPC error")
        result = _read_all_commitments(
            subtensor, netuid=1, hotkey_to_uid={}, miner_uids=set(),
        )
        assert result == {}

    def test_read_all_commitments_skips_bad_json(self):
        """Should skip entries with invalid JSON."""
        subtensor = MagicMock()
        subtensor.get_all_commitments.return_value = {
            "hotkey_0": "not-json",
            "hotkey_1": ImageCommitment(image_url="ok", listener_url="y").to_json(),
        }
        result = _read_all_commitments(
            subtensor, netuid=1,
            hotkey_to_uid={"hotkey_0": 0, "hotkey_1": 1},
            miner_uids={0, 1},
        )
        assert len(result) == 1
        assert 1 in result

    def test_falls_back_to_per_uid_when_all_empty(self):
        """If get_all_commitments returns empty, fall back to per-UID."""
        mg = self._make_metagraph(n=2, validator_permits=[False, False])
        subtensor = MagicMock()
        subtensor.get_all_commitments.return_value = {}
        # Per-UID fallback returns metadata
        commitment_json = ImageCommitment(
            image_url="img:v1", listener_url="http://h:8090",
        ).to_json()
        json_bytes = list(commitment_json.encode("utf-8"))

        subtensor.get_commitment_metadata.return_value = {
            "info": {"fields": [[{"Raw100": [json_bytes]}]]},
        }

        result = read_miner_commitments(subtensor, netuid=1, metagraph=mg)
        assert len(result) == 2


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
