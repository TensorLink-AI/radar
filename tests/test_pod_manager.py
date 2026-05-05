"""Tests for pod manager."""

import sys
import pytest
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch
from validator.pod_manager import (
    get_mode, pre_validate_code, verify_miner_pod, _parse_image_ref,
    _normalise_agent_code, _write_agent_code, launch_agent_pod,
    run_agent_on_pod, cleanup_agent_env, reap_orphan_agent_pods,
    _record_deployment_name,
)


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


def _make_meta(image="ghcr.io/tensorlink-ai/radar/radar-runner", image_tag="latest",
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


class TestNormaliseAgentCode:
    def test_bundle_dict_passthrough(self):
        bundle = {"files": {"agent.py": "code"}, "entry_point": "agent.py"}
        assert _normalise_agent_code(bundle) is bundle

    def test_string_wraps_as_bundle(self):
        result = _normalise_agent_code("print('hi')")
        assert result == {
            "files": {"agent.py": "print('hi')"},
            "entry_point": "agent.py",
        }

    def test_dict_without_files_wraps(self):
        result = _normalise_agent_code({"code": "x"})
        assert "files" in result
        assert "agent.py" in result["files"]


class TestWriteAgentCode:
    def test_string_creates_agent_py(self):
        import os
        tmpdir = _write_agent_code("def design_architecture(): pass")
        agent_file = os.path.join(tmpdir, "agent", "agent.py")
        assert os.path.exists(agent_file)
        with open(agent_file) as f:
            assert "design_architecture" in f.read()

    def test_bundle_creates_multiple_files(self):
        import os
        bundle = {"files": {"agent.py": "main", "utils.py": "helper"}}
        tmpdir = _write_agent_code(bundle)
        assert os.path.exists(os.path.join(tmpdir, "agent", "agent.py"))
        assert os.path.exists(os.path.join(tmpdir, "agent", "utils.py"))


class TestLaunchAgentPod:
    @pytest.mark.asyncio
    async def test_stashes_code_for_inline_delivery(self):
        """Both modes stash agent code for inline delivery via process_challenge."""
        mock_env = MagicMock()
        mock_af = MagicMock()
        mock_af.load_env.return_value = mock_env

        bundle = {"files": {"agent.py": "code"}, "entry_point": "agent.py"}
        for mode in ("docker", "basilica"):
            with patch("validator.pod_manager._af", return_value=mock_af):
                env = await launch_agent_pod(
                    image_url="test:latest",
                    mode=mode,
                    agent_code=bundle,
                )

            call_kwargs = mock_af.load_env.call_args
            assert call_kwargs.kwargs["mode"] == mode
            assert env._agent_code == bundle


class TestRunAgentOnPod:
    @pytest.mark.asyncio
    async def test_passes_inline_code(self):
        """Stashed agent code is always passed to process_challenge."""
        mock_env = MagicMock()
        mock_env._agent_code = {"files": {"agent.py": "code"}}
        mock_env.process_challenge = AsyncMock(
            return_value={"code": "x", "name": "n", "motivation": "m"},
        )

        result = await run_agent_on_pod(mock_env, '{"seed": 1}', timeout=60)
        call_kwargs = mock_env.process_challenge.call_args.kwargs
        assert "agent_code" in call_kwargs
        assert call_kwargs["agent_code"]["files"]["agent.py"] == "code"
        assert result["code"] == "x"

    @pytest.mark.asyncio
    async def test_no_code_stashed_skips_kwarg(self):
        """When no agent code is stashed, agent_code kwarg is omitted."""
        mock_env = MagicMock()
        mock_env._agent_code = None
        mock_env.process_challenge = AsyncMock(
            return_value={"code": "x", "name": "n", "motivation": "m"},
        )

        result = await run_agent_on_pod(mock_env, '{"seed": 1}', timeout=60)
        call_kwargs = mock_env.process_challenge.call_args.kwargs
        assert "agent_code" not in call_kwargs
        assert result["code"] == "x"

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """Exception from process_challenge returns None after retries."""
        mock_env = MagicMock()
        mock_env._agent_code = None
        mock_env.process_challenge = AsyncMock(
            side_effect=RuntimeError("deployment failed"),
        )
        with patch("config.Config.AGENT_POD_RETRIES", 0):
            result = await run_agent_on_pod(mock_env, '{}', timeout=60)
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_timeout_kwarg(self):
        """timeout is passed as regular kwarg for BasilicaBackend TTL."""
        mock_env = MagicMock()
        mock_env._agent_code = None
        mock_env.process_challenge = AsyncMock(return_value={"code": "x"})

        await run_agent_on_pod(mock_env, '{}', timeout=600)
        call_kwargs = mock_env.process_challenge.call_args.kwargs
        assert call_kwargs["timeout"] == 600
        assert call_kwargs["_timeout"] == 600

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self):
        """Retries on first failure, succeeds on second attempt."""
        mock_env = MagicMock()
        mock_env._agent_code = None
        mock_env.process_challenge = AsyncMock(
            side_effect=[RuntimeError("transient"), {"code": "ok"}],
        )
        with patch("config.Config.AGENT_POD_RETRIES", 2), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await run_agent_on_pod(mock_env, '{}', timeout=60)
        assert result == {"code": "ok"}
        assert mock_env.process_challenge.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_exhausted_returns_none(self):
        """All retries exhausted returns None."""
        mock_env = MagicMock()
        mock_env._agent_code = None
        mock_env.process_challenge = AsyncMock(
            side_effect=RuntimeError("always fails"),
        )
        with patch("config.Config.AGENT_POD_RETRIES", 1), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await run_agent_on_pod(mock_env, '{}', timeout=60)
        assert result is None
        assert mock_env.process_challenge.call_count == 2

    @pytest.mark.asyncio
    async def test_task_id_disambiguates_concurrent_calls(self):
        """task_id is forwarded to process_challenge so affinetes can
        generate unique Basilica deployment names for concurrent calls."""
        mock_env = MagicMock()
        mock_env._agent_code = None
        mock_env.process_challenge = AsyncMock(return_value={"code": "x"})

        await run_agent_on_pod(mock_env, '{}', timeout=60, task_id=7)
        call_kwargs = mock_env.process_challenge.call_args.kwargs
        assert call_kwargs["task_id"] == 7

    @pytest.mark.asyncio
    async def test_task_id_omitted_when_not_provided(self):
        """Without task_id, the kwarg is not injected (backwards compat)."""
        mock_env = MagicMock()
        mock_env._agent_code = None
        mock_env.process_challenge = AsyncMock(return_value={"code": "x"})

        await run_agent_on_pod(mock_env, '{}', timeout=60)
        call_kwargs = mock_env.process_challenge.call_args.kwargs
        assert "task_id" not in call_kwargs

    @pytest.mark.asyncio
    async def test_records_deployment_names_across_retries(self):
        """Each retry's deployment name is captured for cleanup."""
        env = MagicMock(spec=["_agent_code", "_deployment_name", "process_challenge"])
        env._agent_code = None
        # Simulate Basilica re-naming the deployment on every retry
        names = ["radar-agent-pc-1001", "radar-agent-pc-1002", "radar-agent-pc-1003"]

        async def _proc(**kw):
            env._deployment_name = names.pop(0)
            if names:
                raise RuntimeError("502 Bad Gateway")
            return {"code": "x"}

        env.process_challenge = _proc
        with patch("config.Config.AGENT_POD_RETRIES", 2), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await run_agent_on_pod(env, '{}', timeout=60)
        assert result == {"code": "x"}
        assert env._radar_deployment_names == [
            "radar-agent-pc-1001",
            "radar-agent-pc-1002",
            "radar-agent-pc-1003",
        ]


class TestRecordDeploymentName:
    def test_picks_up_string_attr(self):
        env = MagicMock(spec=["_deployment_name"])
        env._deployment_name = "radar-agent-1"
        _record_deployment_name(env)
        assert env._radar_deployment_names == ["radar-agent-1"]

    def test_dedupes_repeated_names(self):
        env = MagicMock(spec=["deployment_name"])
        env.deployment_name = "radar-agent-1"
        _record_deployment_name(env)
        _record_deployment_name(env)
        assert env._radar_deployment_names == ["radar-agent-1"]

    def test_falls_back_to_deployment_object(self):
        dep = MagicMock()
        dep.name = "radar-agent-deep"
        env = MagicMock(spec=["_deployment"])
        env._deployment = dep
        _record_deployment_name(env)
        assert env._radar_deployment_names == ["radar-agent-deep"]

    def test_silent_when_no_name_available(self):
        env = MagicMock(spec=[])
        _record_deployment_name(env)
        assert not hasattr(env, "_radar_deployment_names")


class TestCleanupAgentEnv:
    @pytest.mark.asyncio
    async def test_none_env_is_noop(self):
        await cleanup_agent_env(None)  # must not raise

    @pytest.mark.asyncio
    async def test_calls_env_cleanup(self):
        env = MagicMock()
        env.cleanup = AsyncMock()
        env._radar_deployment_names = []
        await cleanup_agent_env(env)
        env.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_swallows_cleanup_exception(self):
        env = MagicMock()
        env.cleanup = AsyncMock(side_effect=RuntimeError("boom"))
        env._radar_deployment_names = []
        await cleanup_agent_env(env)  # must not raise

    @pytest.mark.asyncio
    async def test_force_deletes_recorded_names(self):
        """Every recorded deployment is force-deleted via BasilicaClient."""
        env = MagicMock()
        env.cleanup = AsyncMock()
        env._radar_deployment_names = ["a", "b", "c"]

        fake_client = MagicMock()
        fake_client.delete_deployment = MagicMock()
        fake_module = ModuleType("basilica")
        fake_module.BasilicaClient = MagicMock(return_value=fake_client)
        with patch.dict(sys.modules, {"basilica": fake_module}):
            await cleanup_agent_env(env)

        deleted = [c.args[0] for c in fake_client.delete_deployment.call_args_list]
        assert deleted == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_force_delete_failures_swallowed(self):
        env = MagicMock()
        env.cleanup = AsyncMock()
        env._radar_deployment_names = ["a"]

        fake_client = MagicMock()
        fake_client.delete_deployment = MagicMock(side_effect=RuntimeError("404"))
        fake_module = ModuleType("basilica")
        fake_module.BasilicaClient = MagicMock(return_value=fake_client)
        with patch.dict(sys.modules, {"basilica": fake_module}):
            await cleanup_agent_env(env)  # must not raise


class TestReapOrphanAgentPods:
    @pytest.mark.asyncio
    async def test_no_op_when_not_basilica_mode(self):
        with patch("validator.pod_manager.get_mode", return_value="docker"):
            n = await reap_orphan_agent_pods()
        assert n == 0

    @pytest.mark.asyncio
    async def test_deletes_old_matching_deployments(self):
        old = MagicMock()
        old.name = "radar-agent-pc-old"
        old.uptime_seconds = 4000  # > 1800

        young = MagicMock()
        young.name = "radar-agent-pc-young"
        young.uptime_seconds = 60  # fresh

        other = MagicMock()
        other.name = "radar-trainer-x"
        other.uptime_seconds = 9999  # wrong prefix

        no_age = MagicMock()
        no_age.name = "radar-agent-pc-noage"
        no_age.uptime_seconds = None
        no_age.created_at = None  # unparseable → skip

        fake_client = MagicMock()
        fake_client.list_deployments = MagicMock(
            return_value=[old, young, other, no_age],
        )
        fake_client.delete_deployment = MagicMock()
        fake_module = ModuleType("basilica")
        fake_module.BasilicaClient = MagicMock(return_value=fake_client)

        with patch("validator.pod_manager.get_mode", return_value="basilica"), \
             patch.dict(sys.modules, {"basilica": fake_module}):
            n = await reap_orphan_agent_pods(
                prefix="radar-agent", max_age_seconds=1800,
            )

        assert n == 1
        deleted = [c.args[0] for c in fake_client.delete_deployment.call_args_list]
        assert deleted == ["radar-agent-pc-old"]

    @pytest.mark.asyncio
    async def test_basilica_sdk_missing_returns_zero(self):
        with patch("validator.pod_manager.get_mode", return_value="basilica"), \
             patch.dict(sys.modules, {"basilica": None}):
            # Importing a module set to None raises ImportError
            n = await reap_orphan_agent_pods()
        assert n == 0
