"""Tests for pod manager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from validator.pod_manager import get_mode, pre_validate_code, verify_miner_pod


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


class TestVerifyMinerPod:
    @pytest.mark.asyncio
    async def test_no_digest_configured(self):
        """No expected digest -> passes."""
        with patch("config.Config.OFFICIAL_TRAINING_IMAGE_DIGEST", ""):
            ok, reason = await verify_miner_pod("http://pod:8080", "attest-123")
            assert ok
            assert "no digest" in reason

    @pytest.mark.asyncio
    async def test_basilica_not_installed(self):
        """basilica SDK missing -> passes with warning."""
        with patch("config.Config.OFFICIAL_TRAINING_IMAGE_DIGEST", "sha256:abc"):
            with patch.dict("sys.modules", {"basilica": None}):
                ok, reason = await verify_miner_pod("http://pod:8080", "attest-123")
                # ImportError path
                assert ok
