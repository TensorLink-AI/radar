"""Image hardening tests — entrypoint refusal + bootstrap integrity + /boot_proof.

These run without Docker: the entrypoint script is just POSIX sh and
the bootstrap is stdlib-only Python that takes a parameterized root
path. The /boot_proof endpoint is exercised against the real FastAPI
app via TestClient, with a stub wallet injected.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ENTRYPOINT = REPO / "runner" / "entrypoint.sh"


# ── Entrypoint shell-script tests ─────────────────────────────────


def _run_entrypoint(args: list[str], env: dict[str, str] | None = None):
    """Run entrypoint.sh with a stub `python3` on PATH so `exec python3` succeeds.

    Returns the CompletedProcess. We swap PATH so the only python3 is
    a script that prints a marker and exits 0; the entrypoint refusal
    paths exit before reaching exec, so they don't need it.
    """
    base_env = {} if env is None else dict(env)
    if "PATH" not in base_env:
        # POSIX commands that the script needs (sh, env, sed, grep, unset).
        base_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    return subprocess.run(
        ["/bin/sh", str(ENTRYPOINT), *args],
        env=base_env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_entrypoint_refuses_args():
    proc = _run_entrypoint(["bash", "-c", "echo pwned"])
    assert proc.returncode == 100, proc.stderr
    assert "unexpected positional args" in proc.stderr


@pytest.mark.parametrize("var", [
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "PYTHONUSERBASE",
])
def test_entrypoint_refuses_dangerous_env(var):
    proc = _run_entrypoint([], env={var: "/tmp/evil"})
    assert proc.returncode == 101, f"{var}: rc={proc.returncode} stderr={proc.stderr}"
    assert var in proc.stderr
    assert "banned env var" in proc.stderr


def test_entrypoint_strips_unknown_env(tmp_path):
    """Unknown env vars get unset before exec'ing python3.

    We replace `python3` on PATH with a stub that dumps os.environ to a
    temp file, then check that the unknown var is absent from the dump.
    """
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    dump_path = tmp_path / "env.json"

    # Resolve the real python3 once so the stub doesn't recurse through
    # its own PATH (the entrypoint's `exec python3` would otherwise loop).
    real_python = sys.executable

    stub = stub_dir / "python3"
    stub.write_text(
        "#!/bin/sh\n"
        f"exec {real_python} -c 'import json,os; "
        f'open({json.dumps(str(dump_path))}, "w").write(json.dumps(dict(os.environ)))\'\n'
    )
    stub.chmod(0o755)

    env = {
        # Real PATH for sed / grep, with stub_dir winning for python3.
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        # An unknown variable that should be stripped.
        "BOGUS_ATTACKER_VAR": "value",
        # An allowlisted variable that must survive.
        "RADAR_LOCALNET": "1",
    }
    proc = _run_entrypoint([], env=env)
    assert proc.returncode == 0, proc.stderr
    assert dump_path.exists(), proc.stderr
    captured = json.loads(dump_path.read_text())
    assert "BOGUS_ATTACKER_VAR" not in captured
    assert captured.get("RADAR_LOCALNET") == "1"
    # PYTHONNOUSERSITE always pinned.
    assert captured.get("PYTHONNOUSERSITE") == "1"


# ── Bootstrap integrity tests ──────────────────────────────────────


def _stage_image(root: Path) -> dict:
    """Materialise a minimal `image` tree under `root` matching what the bootstrap expects."""
    files = {
        "/usr/local/bin/radar-entrypoint.sh": "#!/bin/sh\nexec python3 /workspace/_bootstrap.py\n",
        "/workspace/_bootstrap.py": "# bootstrap stub\n",
        "/workspace/server.py": "# server stub\n",
        "/workspace/runner/__init__.py": "",
        "/workspace/runner/harness.py": "# harness\n",
        "/workspace/shared/__init__.py": "",
        "/workspace/shared/auth.py": "# auth\n",
        "/workspace/frozen/harness.py": "# frozen harness\n",
    }
    dirs = [
        "/workspace",
        "/workspace/runner",
        "/workspace/shared",
        "/workspace/frozen",
    ]
    for rel, contents in files.items():
        p = root / rel.lstrip("/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)
    # Compute hashes table.
    import hashlib
    table_files = {}
    for rel in files:
        p = root / rel.lstrip("/")
        table_files[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    table = {"version": "1", "files": table_files, "dirs": dirs}
    hashes_path = root / "workspace" / "_bootstrap_hashes.json"
    hashes_path.write_text(json.dumps(table, sort_keys=True, separators=(",", ":")))
    return {"files": table_files, "dirs": dirs, "hashes_path": str(hashes_path)}


def _import_bootstrap():
    """Import runner._bootstrap freshly so each test gets a clean module."""
    sys.path.insert(0, str(REPO))
    if "runner._bootstrap" in sys.modules:
        del sys.modules["runner._bootstrap"]
    from runner import _bootstrap as bs
    return bs


def test_bootstrap_writes_boot_proof_on_success(tmp_path):
    _stage_image(tmp_path)
    bs = _import_bootstrap()
    proof = bs.run(root=str(tmp_path), exec_target=False)
    assert proof["bootstrap_version"] == bs.BOOTSTRAP_VERSION
    assert proof["file_count"] == len(proof["files_hashed"])
    assert len(proof["hashes_root_sha256"]) == 64
    proof_file = tmp_path / "tmp" / "boot_proof.json"
    assert proof_file.exists()
    on_disk = json.loads(proof_file.read_text())
    assert on_disk == proof


def _run_bootstrap_subprocess(root: Path) -> subprocess.CompletedProcess:
    """Run the bootstrap in a subprocess so os._exit is observable."""
    code = (
        "import sys; sys.path.insert(0, " + repr(str(REPO)) + ");"
        "from runner import _bootstrap;"
        "_bootstrap.run(root=" + repr(str(root)) + ", exec_target=False)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_bootstrap_detects_modified_file(tmp_path):
    _stage_image(tmp_path)
    (tmp_path / "workspace" / "server.py").write_text("# tampered!\n")
    proc = _run_bootstrap_subprocess(tmp_path)
    assert proc.returncode == 1, proc.stderr
    assert "hash mismatch" in proc.stderr
    assert "/workspace/server.py" in proc.stderr


def test_bootstrap_detects_extra_file(tmp_path):
    _stage_image(tmp_path)
    extra = tmp_path / "workspace" / "shared" / "evil.py"
    extra.write_text("# malicious\n")
    proc = _run_bootstrap_subprocess(tmp_path)
    assert proc.returncode == 1, proc.stderr
    assert "unexpected entries in protected dirs" in proc.stderr
    assert "/workspace/shared/evil.py" in proc.stderr


def test_bootstrap_detects_missing_file(tmp_path):
    _stage_image(tmp_path)
    (tmp_path / "workspace" / "shared" / "auth.py").unlink()
    proc = _run_bootstrap_subprocess(tmp_path)
    assert proc.returncode == 1, proc.stderr
    assert "missing required files" in proc.stderr
    assert "/workspace/shared/auth.py" in proc.stderr


# ── /boot_proof endpoint tests ──────────────────────────────────────


@pytest.fixture
def boot_proof_module():
    sys.path.insert(0, str(REPO))
    from runner import boot_proof as bp
    return bp


class _StubKeypair:
    def __init__(self, ss58_address: str = "stub_hotkey"):
        self.ss58_address = ss58_address
        self._signed: bytes | None = None

    def sign(self, payload: bytes) -> bytes:
        # Deterministic stub signature so tests can verify the input.
        self._signed = payload
        return b"\xab" * 16 + payload[:4]


class _StubWallet:
    def __init__(self):
        self.hotkey = _StubKeypair()


def _write_proof(tmp_path: Path) -> Path:
    proof = {
        "boot_time": 1700000000,
        "files_hashed": ["/workspace/server.py"],
        "file_count": 1,
        "hashes_root_sha256": "0" * 64,
        "bootstrap_version": "1",
    }
    p = tmp_path / "boot_proof.json"
    p.write_text(json.dumps(proof))
    return p


def test_boot_proof_endpoint_signs_response(boot_proof_module, tmp_path):
    proof_file = _write_proof(tmp_path)
    wallet = _StubWallet()
    status, body = boot_proof_module.build_boot_proof_response(
        proof_path=str(proof_file), wallet=wallet,
    )
    assert status == 200
    assert body["proof"]["bootstrap_version"] == "1"
    assert body["signer_hotkey"] == "stub_hotkey"
    assert body["signature"]  # non-empty
    # Stub sign() captured exactly the canonical-JSON we hash.
    canonical = boot_proof_module._canonical_json(body["proof"])
    assert wallet.hotkey._signed == canonical
    # Returned canonical hash matches what we'd compute locally.
    import hashlib
    assert body["canonical_payload_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_boot_proof_endpoint_503_when_missing(boot_proof_module, tmp_path):
    missing = tmp_path / "no_such_file.json"
    status, body = boot_proof_module.build_boot_proof_response(
        proof_path=str(missing), wallet=_StubWallet(),
    )
    assert status == 503
    assert body["reason"] == "missing_boot_proof"


def test_boot_proof_via_fastapi(boot_proof_module, tmp_path, monkeypatch):
    """End-to-end through TestClient — verifies the route + JSONResponse plumbing."""
    proof_file = _write_proof(tmp_path)

    monkeypatch.setattr(boot_proof_module, "BOOT_PROOF_PATH", str(proof_file))
    monkeypatch.setattr(boot_proof_module, "_load_trainer_wallet", lambda: _StubWallet())

    from runner import server as srv
    from fastapi.testclient import TestClient
    client = TestClient(srv.app)
    resp = client.get("/boot_proof")
    assert resp.status_code == 200
    body = resp.json()
    assert body["signer_hotkey"] == "stub_hotkey"
    assert body["signature"]


def test_boot_proof_via_fastapi_503(boot_proof_module, tmp_path, monkeypatch):
    monkeypatch.setattr(boot_proof_module, "BOOT_PROOF_PATH", str(tmp_path / "missing.json"))
    if "runner.server" in sys.modules:
        del sys.modules["runner.server"]
    from runner import server as srv
    from fastapi.testclient import TestClient
    client = TestClient(srv.app)
    resp = client.get("/boot_proof")
    assert resp.status_code == 503
    assert resp.json()["reason"] == "missing_boot_proof"
