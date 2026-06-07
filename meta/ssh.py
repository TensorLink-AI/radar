"""Thin SSH wrapper over paramiko.

Only the verbs the orchestrator needs: ``run`` (capture stdout+rc),
``put_file`` (push a local string/blob), ``start_tmux`` (launch a
detached session so the validator survives SSH drops), and
``http_get_local`` (curl over the SSH connection to read the per-unit
dashboard's JSON without exposing it publicly).

Falls back to a subprocess ``ssh``/``scp`` shell-out if paramiko isn't
installed, so the orchestrator can still operate from a vanilla pod
image. Both paths produce the same return shape.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SSHTarget:
    host: str
    port: int = 22
    user: str = "root"
    key_path: str = ""


@dataclass
class CmdResult:
    rc: int
    stdout: str
    stderr: str


def _have_paramiko() -> bool:
    try:
        import paramiko  # noqa: F401
        return True
    except ImportError:
        return False


def _client(target: SSHTarget):
    import paramiko  # type: ignore
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict = {}
    if target.key_path:
        kwargs["key_filename"] = target.key_path
    c.connect(
        hostname=target.host,
        port=target.port,
        username=target.user,
        timeout=15,
        **kwargs,
    )
    return c


def run(target: SSHTarget, cmd: str, *, timeout: float = 60.0) -> CmdResult:
    """Run a single command. Returns rc + captured stdout/stderr."""
    if _have_paramiko():
        c = _client(target)
        try:
            stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            rc = stdout.channel.recv_exit_status()
            return CmdResult(rc=rc, stdout=out, stderr=err)
        finally:
            c.close()
    # Subprocess fallback.
    argv = _ssh_argv(target) + [cmd]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return CmdResult(rc=p.returncode, stdout=p.stdout, stderr=p.stderr)


def put_file(target: SSHTarget, remote_path: str, content: str) -> None:
    """Atomically push ``content`` to ``remote_path`` on the pod."""
    if _have_paramiko():
        c = _client(target)
        try:
            sftp = c.open_sftp()
            try:
                tmp = remote_path + ".tmp"
                with sftp.file(tmp, "w") as f:
                    f.write(content)
                sftp.posix_rename(tmp, remote_path)
            finally:
                sftp.close()
        finally:
            c.close()
        return
    # Subprocess fallback via scp.
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(content)
        local = f.name
    try:
        scp_argv = (
            ["scp", "-P", str(target.port), "-o", "StrictHostKeyChecking=no"]
            + (["-i", target.key_path] if target.key_path else [])
            + [local, f"{target.user}@{target.host}:{remote_path}"]
        )
        subprocess.run(scp_argv, check=True, timeout=120)
    finally:
        Path(local).unlink(missing_ok=True)


def start_tmux(
    target: SSHTarget,
    *,
    session: str,
    cwd: str,
    argv: Iterable[str],
    env: dict[str, str] | None = None,
) -> CmdResult:
    """Launch ``argv`` inside a detached tmux session.

    The launched command survives SSH disconnects; the orchestrator
    later re-attaches by running ``tmux capture-pane -p -t <session>``
    over a fresh ``run()`` call to tail logs.
    """
    env_str = ""
    if env:
        env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    quoted = " ".join(shlex.quote(a) for a in argv)
    inner = f"cd {shlex.quote(cwd)} && {env_str} {quoted}".strip()
    cmd = (
        f"tmux kill-session -t {shlex.quote(session)} 2>/dev/null; "
        f"tmux new-session -d -s {shlex.quote(session)} "
        f"{shlex.quote(inner)}"
    )
    return run(target, cmd, timeout=30)


def tail_tmux(
    target: SSHTarget, *, session: str, lines: int = 200
) -> CmdResult:
    """Capture the last ``lines`` of the tmux session's pane."""
    cmd = (
        f"tmux capture-pane -p -S -{int(lines)} -t {shlex.quote(session)}"
        " 2>/dev/null || echo '<no session>'"
    )
    return run(target, cmd, timeout=15)


def http_get_local(
    target: SSHTarget, *, path: str, port: int = 8765, timeout: float = 10.0
) -> CmdResult:
    """``curl`` a local-bound HTTP endpoint on the pod (e.g. the
    per-unit ``local/dashboard.py`` JSON API)."""
    url = f"http://127.0.0.1:{port}{path}"
    cmd = f"curl -fsS --max-time {int(timeout)} {shlex.quote(url)}"
    return run(target, cmd, timeout=timeout + 5)


def _ssh_argv(target: SSHTarget) -> list[str]:
    argv = [
        "ssh",
        "-p", str(target.port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
    ]
    if target.key_path:
        argv += ["-i", target.key_path]
    argv.append(f"{target.user}@{target.host}")
    return argv
