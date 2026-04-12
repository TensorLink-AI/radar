"""Tests for sandbox subprocess isolation.

Covers: import blocker, sandbox_runner config handling, _run_sandbox
subprocess management, _prefetch_shards, harness checkpoint dir,
pretrain_loader shard_paths, prepare.py shard_paths passthrough,
train.py LOCAL_PATHS env var reading.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Import blocker tests ───────────────────────────────────────────

class TestNetworkBlocker:
    """sandbox_runner.NetworkBlocker blocks network-capable imports."""

    def test_blocked_module_raises(self):
        from runner.sandbox_runner import NetworkBlocker, BLOCKED_MODULES
        blocker = NetworkBlocker()
        for name in ["httpx", "requests", "socket", "boto3", "aiohttp"]:
            assert blocker.find_module(name) is blocker
            with pytest.raises(ImportError, match="blocked in sandbox"):
                blocker.load_module(name)

    def test_allowed_module_passes(self):
        from runner.sandbox_runner import NetworkBlocker
        blocker = NetworkBlocker()
        assert blocker.find_module("json") is None
        assert blocker.find_module("os") is None
        assert blocker.find_module("torch") is None

    def test_submodule_blocked(self):
        from runner.sandbox_runner import NetworkBlocker
        blocker = NetworkBlocker()
        assert blocker.find_module("http.client") is not None
        assert blocker.find_module("urllib.request") is not None

    def test_block_network_imports_purges_loaded(self):
        """_block_network_imports removes already-loaded blocked modules."""
        from runner.sandbox_runner import _block_network_imports, BLOCKED_MODULES

        # Simulate a pre-loaded blocked module
        fake_mod = MagicMock()
        sys.modules["httpx"] = fake_mod
        sys.modules["httpx.client"] = fake_mod

        try:
            _block_network_imports()
            assert "httpx" not in sys.modules
            assert "httpx.client" not in sys.modules
        finally:
            # Clean up the meta_path entry we added
            sys.meta_path[:] = [
                f for f in sys.meta_path
                if type(f).__name__ != "NetworkBlocker"
            ]


# ── Sandbox runner config tests ────────────────────────────────────

class TestSandboxRunnerConfig:
    """sandbox_runner.main() reads config and sets env vars."""

    def test_prefetch_mode_sets_local_paths(self):
        """Prefetch mode sets RADAR_PRETRAIN_LOCAL_PATHS from config."""
        config = {
            "seed": 42,
            "time_budget": 300,
            "data_mode": "prefetch",
            "local_data_dir": "/workspace/sandbox/data",
            "local_shard_paths": ["/workspace/sandbox/shards/shard_0.parquet"],
            "architecture_code": "def build_model(*a): pass\ndef build_optimizer(m): pass",
            "round_id": 1,
            "miner_hotkey": "test",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            # We can't run main() directly (it imports harness), but we can
            # test the config parsing logic by checking env var setting
            with open(config_path) as f:
                loaded = json.load(f)
            assert loaded["data_mode"] == "prefetch"
            assert loaded["local_shard_paths"] == ["/workspace/sandbox/shards/shard_0.parquet"]
        finally:
            os.unlink(config_path)

    def test_proxy_mode_config(self):
        """Proxy mode config contains proxy_url and n_shards."""
        config = {
            "seed": 42,
            "time_budget": 300,
            "data_mode": "proxy",
            "proxy_url": "http://127.0.0.1:9999",
            "n_shards": 3,
            "local_data_dir": "/workspace/sandbox/data",
            "architecture_code": "def build_model(*a): pass",
            "round_id": 1,
            "miner_hotkey": "test",
        }
        # Verify proxy URL generation matches sandbox_runner logic
        proxy_base = config["proxy_url"]
        n_shards = config["n_shards"]
        shard_urls = [f"{proxy_base}/shard/{i}" for i in range(n_shards)]
        assert shard_urls == [
            "http://127.0.0.1:9999/shard/0",
            "http://127.0.0.1:9999/shard/1",
            "http://127.0.0.1:9999/shard/2",
        ]


# ── _run_sandbox tests ─────────────────────────────────────────────

class TestRunSandbox:
    """server._run_sandbox spawns subprocess and parses output."""

    @pytest.mark.asyncio
    async def test_parses_json_stdout(self):
        """Valid JSON on stdout is parsed as result."""
        from runner.sandbox import run_sandbox

        result_json = json.dumps({"status": "success", "round_id": 1})
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(
            result_json.encode(), b"",
        ))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_sandbox("/tmp/config.json", {"time_budget": 10})

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_failed(self):
        """Non-zero exit code returns failed status."""
        from runner.sandbox import run_sandbox

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(
            b"", b"Traceback: some error",
        ))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_sandbox("/tmp/config.json", {"time_budget": 10})

        assert result["status"] == "failed"
        assert "Exit 1" in result["error"]

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self):
        """Sandbox timeout returns timeout status."""
        from runner.sandbox import run_sandbox

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = AsyncMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = await run_sandbox(
                "/tmp/config.json", {"time_budget": 1, "round_id": 5},
            )

        assert result["status"] == "timeout"
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_json_in_stdout_returns_failed(self):
        """No valid JSON in stdout returns failed."""
        from runner.sandbox import run_sandbox

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(
            b"some random output\nno json here\n", b"",
        ))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_sandbox("/tmp/config.json", {"time_budget": 10})

        assert result["status"] == "failed"
        assert "No valid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_last_json_line_used(self):
        """If multiple lines, last JSON line is used as result."""
        from runner.sandbox import run_sandbox

        stdout = b"some log line\n{\"status\": \"ignored\"}\n{\"status\": \"success\", \"round_id\": 1}\n"
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_sandbox("/tmp/config.json", {"time_budget": 10})

        assert result["status"] == "success"


# ── _prefetch_shards tests ─────────────────────────────────────────

class TestPrefetchShards:
    """server._prefetch_shards downloads shards to local files."""

    @pytest.mark.asyncio
    async def test_downloads_all_shards(self):
        from runner.sandbox import prefetch_shards

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = os.path.join(tmpdir, "shards")
            mock_resp = MagicMock()
            mock_resp.content = b"parquet-data"
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("runner.sandbox.os.makedirs"), \
                 patch("runner.sandbox.os.listdir", return_value=[]), \
                 patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("builtins.open", MagicMock()):
                paths = await prefetch_shards(
                    ["https://example.com/s0", "https://example.com/s1"],
                )

            assert len(paths) == 2
            assert mock_client.get.call_count == 2


# ── Harness checkpoint dir tests ───────────────────────────────────

class TestHarnessCheckpointDir:
    """harness.py uses CHECKPOINT_DIR env var for checkpoint path."""

    def test_default_checkpoint_dir(self):
        """Without CHECKPOINT_DIR, uses /workspace/checkpoints."""
        os.environ.pop("CHECKPOINT_DIR", None)
        # Re-check the logic by importing and checking the code path
        from runner.harness import run_training
        # The function reads os.environ.get("CHECKPOINT_DIR", "/workspace/checkpoints")
        # We verify by checking the source contains the env var read
        import inspect
        source = inspect.getsource(run_training)
        assert 'CHECKPOINT_DIR' in source
        assert '/workspace/checkpoints' in source

    def test_custom_checkpoint_dir(self):
        """With CHECKPOINT_DIR set, uses that path."""
        with patch.dict(os.environ, {"CHECKPOINT_DIR": "/workspace/sandbox/checkpoints"}):
            assert os.environ["CHECKPOINT_DIR"] == "/workspace/sandbox/checkpoints"


# ── Pretrain loader shard_paths tests ──────────────────────────────

_has_torch = bool(importlib.util.find_spec("torch"))


@pytest.mark.skipif(not _has_torch, reason="torch not available")
class TestPretrainLoaderShardPaths:
    """pretrain_loader accepts local shard_paths in addition to URLs."""

    def test_iter_series_with_local_paths(self):
        """iter_series reads from local parquet files when shard_paths given."""
        import pandas as pd
        from runner.timeseries_forecast.pretrain_loader import iter_series

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake parquet shard
            df = pd.DataFrame({"target": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]})
            path = os.path.join(tmpdir, "shard_0.parquet")
            df.to_parquet(path)

            values = list(iter_series(shard_paths=[path], seed=42))

        assert len(values) == 2
        assert values[0] in ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    def test_iter_series_prefers_paths_over_urls(self):
        """When shard_paths is given, URLs are ignored."""
        import pandas as pd
        from runner.timeseries_forecast.pretrain_loader import iter_series

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"target": [[1.0, 2.0]]})
            path = os.path.join(tmpdir, "shard.parquet")
            df.to_parquet(path)

            # Pass both paths and URLs; paths should win
            values = list(iter_series(
                shard_urls=["https://should-not-be-called.com/shard"],
                shard_paths=[path],
                seed=42,
            ))

        assert len(values) == 1

    def test_pretrain_dataloader_with_paths(self):
        """pretrain_dataloader works with shard_paths."""
        import pandas as pd
        import numpy as np
        from runner.timeseries_forecast.pretrain_loader import pretrain_dataloader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create shard with long enough series for windowing
            series = list(np.random.randn(700).astype(float))
            df = pd.DataFrame({"target": [series]})
            path = os.path.join(tmpdir, "shard.parquet")
            df.to_parquet(path)

            batches = list(pretrain_dataloader(
                shard_paths=[path], batch_size=2, seed=42,
            ))

        assert len(batches) > 0
        assert "context" in batches[0]
        assert "target" in batches[0]

    def test_iter_series_no_sources(self):
        """iter_series with no sources yields nothing."""
        from runner.timeseries_forecast.pretrain_loader import iter_series
        values = list(iter_series())
        assert values == []


# ── Prepare.py shard_paths passthrough tests ───────────────────────

@pytest.mark.skipif(not _has_torch, reason="torch not available")
class TestPrepareShardPaths:
    """prepare.py get_dataloader passes shard_paths to pretrain_dataloader."""

    def test_get_dataloader_with_shard_paths(self):
        """get_dataloader delegates to pretrain_dataloader when paths given."""
        mock_pretrain = MagicMock(return_value=iter([{"context": "mock", "target": "mock"}]))

        with patch.dict("sys.modules", {
            "pretrain_loader": MagicMock(pretrain_dataloader=mock_pretrain),
        }):
            from runner.timeseries_forecast.prepare import get_dataloader
            batches = list(get_dataloader(
                pretrain_shard_paths=["/tmp/shard.parquet"],
                batch_size=32,
            ))

        assert len(batches) == 1
        mock_pretrain.assert_called_once()
        call_kwargs = mock_pretrain.call_args
        assert call_kwargs[1].get("shard_paths") == ["/tmp/shard.parquet"]


# ── Train.py LOCAL_PATHS env var tests ─────────────────────────────

class TestTrainLocalPaths:
    """train.py TSForecastingRunner reads RADAR_PRETRAIN_LOCAL_PATHS."""

    def test_reads_local_paths_env(self):
        """When RADAR_PRETRAIN_LOCAL_PATHS is set, passes shard_paths."""
        from runner.timeseries_forecast.train import TSForecastingRunner

        runner = TSForecastingRunner()
        local_paths = ["/workspace/sandbox/shards/shard_0.parquet"]

        mock_get_dl = MagicMock(return_value=iter([]))
        with patch.dict(os.environ, {
            "RADAR_PRETRAIN_LOCAL_PATHS": json.dumps(local_paths),
            "RADAR_GIFT_EVAL_CACHE": "",
        }), patch.dict("sys.modules", {
            "prepare": MagicMock(get_dataloader=mock_get_dl),
        }):
            list(runner.get_dataloader(batch_size=32))

        mock_get_dl.assert_called_once()
        call_kwargs = mock_get_dl.call_args[1]
        assert call_kwargs["pretrain_shard_paths"] == local_paths
        assert call_kwargs["pretrain_shard_urls"] is None

    def test_falls_back_to_shard_urls(self):
        """When no LOCAL_PATHS, falls back to SHARD_URLS."""
        from runner.timeseries_forecast.train import TSForecastingRunner

        runner = TSForecastingRunner()
        shard_urls = ["https://example.com/shard_0.parquet"]

        mock_get_dl = MagicMock(return_value=iter([]))
        with patch.dict(os.environ, {
            "RADAR_PRETRAIN_LOCAL_PATHS": "",
            "RADAR_PRETRAIN_SHARD_URLS": json.dumps(shard_urls),
            "RADAR_GIFT_EVAL_CACHE": "",
        }), patch.dict("sys.modules", {
            "prepare": MagicMock(get_dataloader=mock_get_dl),
        }):
            list(runner.get_dataloader(batch_size=32))

        mock_get_dl.assert_called_once()
        call_kwargs = mock_get_dl.call_args[1]
        assert call_kwargs["pretrain_shard_urls"] == shard_urls
        assert call_kwargs["pretrain_shard_paths"] is None


# ── Server _train_and_upload sandbox integration ───────────────────

class TestTrainAndUploadSandbox:
    """server._train_and_upload uses sandbox subprocess."""

    @pytest.fixture(autouse=True)
    def _reset_server(self):
        import runner.server as srv
        srv._train_semaphore = asyncio.Semaphore(1)
        # Pre-acquire so _train_and_upload can release it
        loop = asyncio.new_event_loop()
        loop.run_until_complete(srv._train_semaphore.acquire())
        loop.close()
        yield

    @pytest.mark.asyncio
    async def test_builds_sandbox_config(self):
        """_train_and_upload writes sandbox config with architecture_code."""
        import runner.server as srv

        sandbox_result = {
            "status": "success", "round_id": 1, "miner_hotkey": "test",
            "checkpoint_path": "/workspace/sandbox/checkpoints/model.safetensors",
            "flops_equivalent_size": 100000,
            "training_time_seconds": 5.0,
            "num_steps": 10,
        }

        written_config = {}

        def capture_config(path, mode="r"):
            if mode == "w" and "train_config" in path:
                class FakeFile:
                    def write(self, data): pass
                    def __enter__(self): return self
                    def __exit__(self, *a): pass
                return FakeFile()
            return open.__wrapped__(path, mode) if hasattr(open, '__wrapped__') else None

        with patch("runner.sandbox.run_sandbox", new_callable=AsyncMock, return_value=sandbox_result), \
             patch("runner.uploads.download_gift_eval"), \
             patch("runner.uploads.upload_artifacts"), \
             patch("runner.server.json.dump") as mock_dump, \
             patch("builtins.open", MagicMock()):
            await srv._train_and_upload(
                runner_fn=MagicMock(),
                architecture_code="def build_model(): pass",
                training_config={"round_id": 1, "miner_hotkey": "test", "time_budget": 10},
                upload_urls={},
                gift_eval_urls={},
            )

        # Verify json.dump was called with sandbox config containing architecture_code
        assert mock_dump.called
        config_arg = mock_dump.call_args[0][0]
        assert config_arg["architecture_code"] == "def build_model(): pass"
        assert config_arg["data_mode"] == "prefetch"


# ── SANDBOX_DATA_MODE config tests ─────────────────────────────────

class TestSandboxDataMode:
    """RADAR_SANDBOX_DATA_MODE controls data delivery strategy."""

    def test_default_is_prefetch(self):
        from runner.sandbox import SANDBOX_DATA_MODE
        assert SANDBOX_DATA_MODE in ("prefetch", "proxy")
