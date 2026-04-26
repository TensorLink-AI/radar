"""Tests for runner/sandbox.py and runner/sandbox_runner.py.

Covers env scrubbing, network-import blocking, command construction,
JSON envelope parsing, and prefetch helpers.  End-to-end subprocess
execution is exercised via a small architecture stub that doesn't need
torch / GPU.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _restore_meta_path():
    """Strip sandbox NetworkBlockers from sys.meta_path after the test.

    sandbox_runner.main() installs a global blocker; without cleanup the
    rest of the pytest session can't import http.cookies / urllib / etc.
    """
    saved = list(sys.meta_path)
    yield
    from runner.sandbox_runner import _NetworkBlocker
    sys.meta_path[:] = [m for m in saved if not isinstance(m, _NetworkBlocker)]
    # Also evict any blocked modules that survived as cached errors.
    sys.modules.pop("http.cookies", None)


# ── Network-import blocker ─────────────────────────────────────────

def test_blocker_rejects_high_level_clients():
    from runner.sandbox_runner import _NetworkBlocker, _BLOCKED_MODULES

    blocker = _NetworkBlocker()
    for name in ("httpx", "requests", "boto3", "aiohttp", "urllib.request"):
        spec = blocker.find_spec(name)
        assert spec is not None, f"{name} should be blocked"
        assert spec.loader is blocker
    # Stdlib primitives stay importable so torch / pandas / asyncio work.
    for name in ("socket", "ssl", "json", "time"):
        assert blocker.find_spec(name) is None, f"{name} must NOT be blocked"
    # Sanity: blocker keyed by frozenset.
    assert "httpx" in _BLOCKED_MODULES
    assert "socket" not in _BLOCKED_MODULES


def test_blocker_load_raises_importerror():
    from runner.sandbox_runner import _NetworkBlocker
    import importlib.machinery

    blocker = _NetworkBlocker()
    spec = importlib.machinery.ModuleSpec("httpx", blocker)
    with pytest.raises(ImportError, match="blocked"):
        blocker.create_module(spec)


# ── Env scrubbing ──────────────────────────────────────────────────

def test_scrub_env_drops_secret_prefixes():
    from runner.sandbox import _scrub_env

    secrets = {
        "R2_ACCESS_KEY_ID": "abc",
        "R2_SECRET_ACCESS_KEY": "xyz",
        "AWS_PROFILE": "dev",
        "BASILICA_TOKEN": "tok",
        "WALLET_PASS": "pw",
        "OPENAI_API_KEY": "sk-...",
        "DESEARCH_API_KEY": "ds-...",
        "BITTENSOR_HOTKEY_PHRASE": "twelve words",
        "GH_TOKEN": "gh_...",
    }
    fwd = {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0"}

    with patch.dict(os.environ, {**secrets, **fwd}, clear=True):
        env = _scrub_env()

    for key in secrets:
        assert key not in env, f"{key} leaked into sandbox env"
    assert env["PATH"] == "/usr/bin"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_scrub_env_extra_overrides_drop_secrets_too():
    from runner.sandbox import _scrub_env

    extra = {"RADAR_OK": "yes", "R2_LEAK": "no"}
    with patch.dict(os.environ, {"PATH": "/p"}, clear=True):
        env = _scrub_env(extra=extra)
    assert env["RADAR_OK"] == "yes"
    assert "R2_LEAK" not in env


# ── Command construction ───────────────────────────────────────────

def test_build_command_uses_wrapper_when_present(tmp_path):
    from runner import sandbox as sbx

    fake_wrap = tmp_path / "wrap.sh"
    fake_wrap.write_text("#!/bin/sh\nexec \"$@\"\n")
    fake_wrap.chmod(0o755)
    with patch.object(sbx, "WRAPPER_SCRIPT", str(fake_wrap)):
        cmd = sbx._build_command("/tmp/cfg.json")
    assert cmd[0] == str(fake_wrap)
    assert cmd[-1] == "/tmp/cfg.json"


def test_build_command_falls_back_when_wrapper_missing():
    from runner import sandbox as sbx

    with patch.object(sbx, "WRAPPER_SCRIPT", "/nonexistent/path.sh"):
        cmd = sbx._build_command("/tmp/cfg.json")
    assert "wrap" not in cmd[0]
    assert cmd[-1] == "/tmp/cfg.json"


# ── JSON envelope parsing ──────────────────────────────────────────

def test_last_json_line_picks_final_object():
    from runner.sandbox import _last_json_line

    out = "noise line\nWARNING: stuff\n{\"status\": \"success\", \"x\": 1}\n"
    assert _last_json_line(out) == {"status": "success", "x": 1}


def test_last_json_line_skips_unparseable_tail():
    from runner.sandbox import _last_json_line

    out = '{"status": "success"}\nGarbage trailing text\n'
    assert _last_json_line(out) == {"status": "success"}


def test_last_json_line_returns_none_on_no_object():
    from runner.sandbox import _last_json_line

    assert _last_json_line("just logs\nnothing parseable\n") is None
    assert _last_json_line("") is None


# ── Prefetch ───────────────────────────────────────────────────────

def test_prefetch_shards_writes_files(tmp_path):
    from runner import sandbox as sbx

    dest = tmp_path / "shards"
    payloads = [b"alpha-bytes", b"beta-bytes"]

    class _Resp:
        def __init__(self, body):
            self.content = body
        def raise_for_status(self):
            pass

    class _FakeClient:
        def __init__(self, *_, **__):
            self._idx = 0
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            return False
        async def get(self, url, follow_redirects=True):
            body = payloads[self._idx]
            self._idx += 1
            return _Resp(body)

    with patch("httpx.AsyncClient", _FakeClient):
        paths = asyncio.run(sbx.prefetch_shards(
            ["https://x/a", "https://x/b"], dest_dir=str(dest),
        ))
    assert len(paths) == 2
    assert open(paths[0], "rb").read() == b"alpha-bytes"
    assert open(paths[1], "rb").read() == b"beta-bytes"


def test_prefetch_shards_resets_existing_dir(tmp_path):
    from runner import sandbox as sbx

    dest = tmp_path / "shards"
    dest.mkdir()
    (dest / "stale.parquet").write_bytes(b"stale")

    class _Resp:
        content = b"fresh"
        def raise_for_status(self):
            pass

    class _C:
        def __init__(self, *_, **__):
            pass
        async def __aenter__(self_inner):
            return self_inner
        async def __aexit__(self_inner, *_):
            return False
        async def get(self_inner, url, follow_redirects=True):
            return _Resp()

    with patch("httpx.AsyncClient", _C):
        asyncio.run(sbx.prefetch_shards(["https://x/q"], dest_dir=str(dest)))

    assert not (dest / "stale.parquet").exists()


# ── End-to-end subprocess ──────────────────────────────────────────

def _fake_harness_module():
    """Build a stand-in for runner.harness that the sandbox can import."""
    mod = type(sys)("runner.harness")

    class TC:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.round_id = kw.get("round_id", 0)
            self.miner_hotkey = kw.get("miner_hotkey", "x")
        @classmethod
        def from_dict(cls, d):
            return cls(**d)

    def run_training(runner, code, tc):
        # Echo the env state the sandbox set up so the test can assert on it.
        return {
            "status": "success",
            "round_id": tc.round_id,
            "miner_hotkey": tc.miner_hotkey,
            "checkpoint_dir_env": os.environ.get("CHECKPOINT_DIR"),
            "pretrain_paths_env": os.environ.get("RADAR_PRETRAIN_LOCAL_PATHS"),
            "val_paths_env": os.environ.get("RADAR_PRETRAIN_VAL_LOCAL_PATHS"),
            "shard_url_env": os.environ.get("RADAR_PRETRAIN_SHARD_URLS"),
        }

    mod.TrainingConfig = TC
    mod.run_training = run_training
    return mod


def _fake_task_module():
    mod = type(sys)("runner.timeseries_forecast.train")
    mod._runner = object()
    return mod


def _run_sandbox_main_with_pipe(monkeypatch, sandbox_runner, fake_modules):
    """Run sandbox_runner.main() with the result channel rerouted through a pipe.

    Skips the real ``_seal_stdout`` so we don't dup2 /dev/null over the
    pytest stdout fd.  Returns the bytes written to the private fd.
    """
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sandbox_runner, "_seal_stdout", lambda: None)
    monkeypatch.setattr(sandbox_runner, "_RESULT_FD", write_fd, raising=False)

    with patch.dict(sys.modules, fake_modules):
        try:
            rc = sandbox_runner.main()
        finally:
            os.close(write_fd)

    payload = b""
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        payload += chunk
    os.close(read_fd)
    return rc, payload


def test_sandbox_runner_emits_json_and_strips_inherited_url(
    tmp_path, monkeypatch, _restore_meta_path,
):
    """Result envelope reaches the parent via the private fd, not stdout."""
    from runner import sandbox_runner

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "task_name": "ts_forecasting",
        "round_id": 7,
        "miner_hotkey": "abc",
        "seed": 11,
        "time_budget": 5,
        "architecture_code": "x = 1",
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "pretrain_shard_paths": [str(tmp_path / "p1.parquet")],
        "pretrain_val_shard_paths": [],
        "gift_eval_dir": str(tmp_path / "gift"),
    }))

    fake_modules = {
        "runner.harness": _fake_harness_module(),
        "runner.timeseries_forecast": type(sys)("runner.timeseries_forecast"),
        "runner.timeseries_forecast.train": _fake_task_module(),
    }

    monkeypatch.setattr(sys, "argv", ["sandbox_runner", str(cfg_path)])
    monkeypatch.setenv("RADAR_PRETRAIN_SHARD_URLS", '["https://leak.example/a"]')

    rc, payload = _run_sandbox_main_with_pipe(monkeypatch, sandbox_runner, fake_modules)

    assert rc == 0
    lines = payload.decode().strip().splitlines()
    result = json.loads(lines[-1])
    assert result["status"] == "success"
    assert result["round_id"] == 7
    assert result["checkpoint_dir_env"] == str(tmp_path / "ckpt")
    assert json.loads(result["pretrain_paths_env"]) == [str(tmp_path / "p1.parquet")]
    assert result["val_paths_env"] is None
    # The leaked URL env from the parent must NOT survive into the sandbox.
    assert result["shard_url_env"] is None


def test_sandbox_runner_unknown_task_fails(tmp_path, monkeypatch, _restore_meta_path):
    from runner import sandbox_runner

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "task_name": "totally_made_up",
        "architecture_code": "pass",
        "round_id": 0,
        "miner_hotkey": "x",
        "time_budget": 5,
    }))

    monkeypatch.setattr(sys, "argv", ["sandbox_runner", str(cfg_path)])
    fake_modules = {"runner.harness": _fake_harness_module()}
    rc, payload = _run_sandbox_main_with_pipe(monkeypatch, sandbox_runner, fake_modules)
    assert rc == 2
    result = json.loads(payload.decode().strip().splitlines()[-1])
    assert result["status"] == "failed"
    assert "unknown task" in result["error"]


def test_sandbox_subprocess_blocks_httpx_import(tmp_path):
    """Spawn an actual subprocess and confirm a miner can't ``import httpx``.

    This exercises the import blocker for real, not just the unit test.
    """
    arch = textwrap.dedent('''
        try:
            import httpx
            ok = False
            err = "httpx imported"
        except ImportError as e:
            ok = True
            err = str(e)

        def build_model(c, p, n, q):
            return None

        def build_optimizer(m):
            return None

        # Smuggle the result out via the harness return path: the harness
        # prints whatever build_model returns indirectly, but we use a
        # global that the harness inspects.  The fake harness used in
        # this test echoes env state, so we instead write a marker file.
        import os
        with open(os.environ["RADAR_TEST_MARKER"], "w") as f:
            f.write("ok=%s err=%s" % (ok, err))
    ''').strip()

    cfg = {
        "task_name": "ts_forecasting",
        "architecture_code": arch,
        "round_id": 1,
        "miner_hotkey": "m",
        "time_budget": 5,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "pretrain_shard_paths": [],
        "pretrain_val_shard_paths": [],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    marker = tmp_path / "marker.txt"

    # Monkey-patch the runner package layout: install fake harness +
    # task that EXEC the architecture so the import attempt actually
    # runs in the subprocess.
    helper = tmp_path / "fake_runner.py"
    helper.write_text(textwrap.dedent('''
        import os
        import sys
        import types

        harness = types.ModuleType("runner.harness")

        class TC:
            def __init__(self, **kw):
                self.round_id = kw.get("round_id", 0)
                self.miner_hotkey = kw.get("miner_hotkey", "x")
            @classmethod
            def from_dict(cls, d):
                return cls(**d)

        def run_training(_runner, code, tc):
            ns = {}
            exec(code, ns)
            return {"status": "success", "round_id": tc.round_id,
                    "miner_hotkey": tc.miner_hotkey}

        harness.TrainingConfig = TC
        harness.run_training = run_training
        sys.modules["runner.harness"] = harness

        pkg = types.ModuleType("runner.timeseries_forecast")
        sys.modules["runner.timeseries_forecast"] = pkg

        train_mod = types.ModuleType("runner.timeseries_forecast.train")
        train_mod._runner = object()
        sys.modules["runner.timeseries_forecast.train"] = train_mod
    ''').strip())

    runner_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "runner")
    )

    code = textwrap.dedent(f'''
        import sys, os
        sys.path.insert(0, {runner_path!r})
        sys.path.insert(0, {str(tmp_path)!r})
        import fake_runner  # registers the stubs into sys.modules
        from sandbox_runner import main
        sys.argv = ["sandbox_runner", {str(cfg_path)!r}]
        sys.exit(main())
    ''').strip()

    import subprocess
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
        "PYTHONPATH": runner_path,
        "RADAR_TEST_MARKER": str(marker),
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    assert marker.exists(), f"marker not written; stderr={proc.stderr}"
    contents = marker.read_text()
    assert "ok=True" in contents
    assert "blocked" in contents


# ── Result envelope is unspoofable from miner code ──────────────────

def _spawn_sandbox_with_arch(tmp_path, arch_code: str, *, runner_path: str):
    """Spawn sandbox_runner.py in a real subprocess executing ``arch_code``.

    Returns the captured (stdout, stderr, returncode).  Uses the same
    fake-runner trick as the network-blocker test: the in-test fake
    harness exec()'s the architecture string so the miner's code really
    runs through the sealed-stdout path.
    """
    cfg = {
        "task_name": "ts_forecasting",
        "architecture_code": arch_code,
        "round_id": 1,
        "miner_hotkey": "m",
        "time_budget": 5,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "pretrain_shard_paths": [],
        "pretrain_val_shard_paths": [],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    helper = tmp_path / "fake_runner.py"
    helper.write_text(textwrap.dedent('''
        import sys, types
        harness = types.ModuleType("runner.harness")
        class TC:
            def __init__(self, **kw):
                self.round_id = kw.get("round_id", 0)
                self.miner_hotkey = kw.get("miner_hotkey", "x")
            @classmethod
            def from_dict(cls, d):
                return cls(**d)
        def run_training(_runner, code, tc):
            ns = {}
            exec(code, ns)
            return {"status": "success", "round_id": tc.round_id,
                    "miner_hotkey": tc.miner_hotkey, "honest": True}
        harness.TrainingConfig = TC
        harness.run_training = run_training
        sys.modules["runner.harness"] = harness
        pkg = types.ModuleType("runner.timeseries_forecast")
        sys.modules["runner.timeseries_forecast"] = pkg
        train_mod = types.ModuleType("runner.timeseries_forecast.train")
        train_mod._runner = object()
        sys.modules["runner.timeseries_forecast.train"] = train_mod
    ''').strip())

    code = textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {runner_path!r})
        sys.path.insert(0, {str(tmp_path)!r})
        import fake_runner
        from sandbox_runner import main
        sys.argv = ["sandbox_runner", {str(cfg_path)!r}]
        sys.exit(main())
    ''').strip()

    import subprocess
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
        "PYTHONPATH": runner_path,
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=False, env=env, timeout=30,
    )
    return proc.stdout, proc.stderr, proc.returncode


def test_miner_print_cannot_spoof_result(tmp_path):
    """A miner that prints fake JSON must not corrupt the result envelope."""
    runner_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "runner")
    )
    arch = textwrap.dedent('''
        # Try every known way a miner might reach the parent's stdout.
        import sys, os
        try:
            print('{"status": "success", "honest": false, "spoofed": true}')
        except Exception:
            pass
        try:
            sys.__stdout__.write('{"status": "success", "spoofed": true}\\n')
        except Exception:
            pass
        try:
            os.write(1, b'{"status": "success", "spoofed": true}\\n')
        except Exception:
            pass

        def build_model(c, p, n, q):
            return None
        def build_optimizer(m):
            return None
    ''').strip()

    stdout, stderr, rc = _spawn_sandbox_with_arch(
        tmp_path, arch, runner_path=runner_path,
    )
    assert rc == 0, f"stderr={stderr.decode('utf-8', 'replace')[-1000:]}"

    # Parse the *last* JSON line on stdout — that's what runner/sandbox.py
    # treats as authoritative.
    lines = [
        line for line in stdout.decode("utf-8", "replace").splitlines()
        if line.strip().startswith("{")
    ]
    assert lines, "sandbox produced no JSON line"
    final = json.loads(lines[-1])

    # The honest envelope from the harness wins; the miner's print() didn't.
    assert final.get("honest") is True
    assert "spoofed" not in final


# ── stdout cap kills runaway miners ────────────────────────────────

def test_run_sandbox_kills_on_stdout_overflow(tmp_path, monkeypatch):
    """A sandbox that floods stdout past the cap is killed and reported."""
    from runner import sandbox as sbx

    # Tiny script that just spews bytes forever.
    flood = tmp_path / "flood.py"
    flood.write_text(textwrap.dedent('''
        import os, sys
        chunk = b"X" * 65536
        while True:
            try:
                os.write(1, chunk)
            except BrokenPipeError:
                break
    ''').strip())

    cfg = {
        "round_id": 99,
        "miner_hotkey": "flooder",
        "time_budget": 30,
        "task_name": "ts_forecasting",
        "architecture_code": "",
    }

    monkeypatch.setattr(sbx, "SANDBOX_ROOT", str(tmp_path))
    monkeypatch.setattr(sbx, "CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(sbx, "WRAPPER_SCRIPT", "/nonexistent/wrap.sh")
    monkeypatch.setattr(sbx, "SANDBOX_RUNNER", str(flood))
    # Skip env-scrub for the test runner.
    monkeypatch.setattr(sbx, "_scrub_env", lambda extra=None: dict(os.environ))

    result, stderr = asyncio.run(
        sbx.run_sandbox(
            cfg,
            timeout_buffer=10,
            stdout_cap_bytes=256 * 1024,
            stderr_cap_bytes=64 * 1024,
        ),
    )
    assert result["status"] == "failed"
    assert "exceeded" in result["error"]
    assert "stdout" in result["error"]
