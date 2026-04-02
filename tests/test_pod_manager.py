"""Tests for pod manager."""

import sys
import pytest
from types import ModuleType
from unittest.mock import MagicMock, patch
from validator.pod_manager import get_mode, pre_validate_code, verify_miner_pod, _parse_image_ref


class TestGetMode:
    def test_default(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("RADAR_AFFINETES_MODE", None)
            assert get_mode() == "docker"

    def test_basilica(self):
        with patch.dict("os.environ", {"RADAR_AFFINETES_MODE": "basilica"}):
            assert get_mode() == "basilica"

    def test_invalid_fallback(self):
        with patch.dict("os.environ", {"RADAR_AFFINETES_MODE": "invalid"}):
            assert get_mode() == "docker"


class TestPreValidateCode:
    def test_valid(self):
        code = "def build_model(): pass\ndef build_optimizer(): pass"
        ok, reason = pre_validate_code(code)
        assert ok
        assert reason == ""

    def test_missing_build_model(self):
        ok, reason = pre_validate_code("def build_optimizer(): pass")
        assert not ok
        assert "build_model" in reason

    def test_missing_build_optimizer(self):
        ok, reason = pre_validate_code("def build_model(): pass")
        assert not ok
        assert "build_optimizer" in reason

    def test_syntax_error(self):
        ok, reason = pre_validate_code("def ???")
        assert not ok
        assert "Syntax error" in reason


class TestParseImageRef:
    def test_image_with_tag(self):
        img, tag = _parse_image_ref("ghcr.io/tensorlink-ai/radar/ts-runner:latest")
        assert img == "ghcr.io/tensorlink-ai/radar/ts-runner"
        assert tag == "latest"

    def test_image_without_tag(self):
        img, tag = _parse_image_ref("ghcr.io/tensorlink-ai/radar/ts-runner")
        assert img == "ghcr.io/tensorlink-ai/radar/ts-runner"
        assert tag == ""

    def test_image_with_version_tag(self):
        img, tag = _parse_image_ref("myregistry.io/repo:v1.2.3")
        assert img == "myregistry.io/repo"
        assert tag == "v1.2.3"


def _make_meta(image="ghcr.io/tensorlink-ai/radar/ts-runner", image_tag="latest",
               state="running", replicas=1):
    """Create a mock PublicDeploymentMetadataResponse."""
    meta = MagicMock()
    meta.image = image
    meta.image_tag = image_tag
    meta.state = state
    meta.replicas = replicas
    meta.instance_name = "test-pod"
    meta.id = "deploy-uuid"
    meta.uptime_seconds = 3600
    return meta


def _patch_basilica(meta=None, side_effect=None):
    """Create a fake basilica module with a mock BasilicaClient."""
    fake_mod = ModuleType("basilica")
    mock_client = MagicMock()
    if side_effect:
        mock_client.get_public_deployment_metadata.side_effect = side_effect
    elif meta is not None:
        mock_client.get_public_deployment_metadata.return_value = meta
    mock_cls = MagicMock(return_value=mock_client)
    fake_mod.BasilicaClient = mock_cls
    return fake_mod, mock_client


class TestVerifyMinerPod:
    @pytest.mark.asyncio
    async def test_matching_image_and_tag(self):
        """Matching image + tag -> passes."""
        meta = _make_meta()
        fake_mod, mock_client = _patch_basilica(meta=meta)
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod("my-trainer-pod")
            assert ok
            assert reason == "ok"
            mock_client.get_public_deployment_metadata.assert_called_once_with(
                instance_name="my-trainer-pod",
            )

    @pytest.mark.asyncio
    async def test_wrong_image(self):
        """Wrong image name -> fails with descriptive message."""
        meta = _make_meta(image="ghcr.io/evil/backdoor")
        fake_mod, _ = _patch_basilica(meta=meta)
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod("my-pod")
            assert not ok
            assert "Wrong image" in reason
            assert "ghcr.io/evil/backdoor" in reason

    @pytest.mark.asyncio
    async def test_wrong_tag(self):
        """Wrong tag -> fails."""
        meta = _make_meta(image_tag="malicious")
        fake_mod, _ = _patch_basilica(meta=meta)
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod("my-pod")
            assert not ok
            assert "Wrong tag" in reason
            assert "malicious" in reason

    @pytest.mark.asyncio
    async def test_state_not_running(self):
        """Pod state is not running -> fails."""
        meta = _make_meta(state="failed")
        fake_mod, _ = _patch_basilica(meta=meta)
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod("my-pod")
            assert not ok
            assert "not running" in reason
            assert "failed" in reason

    @pytest.mark.asyncio
    async def test_zero_replicas(self):
        """Replicas == 0 -> fails."""
        meta = _make_meta(replicas=0)
        fake_mod, _ = _patch_basilica(meta=meta)
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod("my-pod")
            assert not ok
            assert "No ready replicas" in reason

    @pytest.mark.asyncio
    async def test_no_expected_image_configured(self):
        """No expected image configured -> passes gracefully."""
        with patch("config.Config.OFFICIAL_TRAINING_IMAGE", ""):
            ok, reason = await verify_miner_pod("my-pod", expected_image="")
            assert ok
            assert "no expected image" in reason

    @pytest.mark.asyncio
    async def test_basilica_sdk_not_installed(self):
        """basilica SDK missing -> passes with warning."""
        with patch.dict(sys.modules, {"basilica": None}):
            ok, reason = await verify_miner_pod("my-pod")
            assert ok
            assert "basilica not available" in reason

    @pytest.mark.asyncio
    async def test_api_exception(self):
        """API call raises exception -> fails with error message."""
        fake_mod, _ = _patch_basilica(side_effect=RuntimeError("connection refused"))
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod("my-pod")
            assert not ok
            assert "Attestation failed" in reason
            assert "connection refused" in reason

    @pytest.mark.asyncio
    async def test_empty_instance_name(self):
        """Empty instance_name -> still calls API (caller handles skip)."""
        fake_mod, _ = _patch_basilica(side_effect=RuntimeError("invalid instance name"))
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod("")
            assert not ok
            assert "Attestation failed" in reason

    @pytest.mark.asyncio
    async def test_image_without_tag_only_compares_image(self):
        """Expected image has no tag -> only compare image portion."""
        meta = _make_meta(image="myregistry.io/repo", image_tag="anything")
        fake_mod, _ = _patch_basilica(meta=meta)
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod(
                "my-pod", expected_image="myregistry.io/repo"
            )
            assert ok
            assert reason == "ok"

    @pytest.mark.asyncio
    async def test_active_state_passes(self):
        """State 'active' should also pass."""
        meta = _make_meta(state="active")
        fake_mod, _ = _patch_basilica(meta=meta)
        with patch.dict(sys.modules, {"basilica": fake_mod}):
            ok, reason = await verify_miner_pod("my-pod")
            assert ok
