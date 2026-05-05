"""Container bootstrap — the trainer image's load-bearing integrity check.

Runs as the immediate child of /usr/local/bin/radar-entrypoint.sh. Reads
the build-time hash table at /workspace/_bootstrap_hashes.json, verifies
every load-bearing file matches, refuses to start on any mismatch or any
unexpected file in a protected directory, writes /tmp/boot_proof.json,
then exec's /workspace/server.py.

Stdlib-only by design — must not import torch / fastapi / shared.* /
anything else that could be redirected by a miner-supplied PYTHONPATH.
The entrypoint script already refuses to start with PYTHONPATH set, but
the bootstrap is the second line of defence and must not depend on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Iterable


BOOTSTRAP_VERSION = "1"
HASHES_PATH = "/workspace/_bootstrap_hashes.json"
BOOT_PROOF_PATH = "/tmp/boot_proof.json"
SERVER_PATH = "/workspace/server.py"


def _die(msg: str, code: int = 1) -> None:
    sys.stderr.write("radar-bootstrap: " + msg + "\n")
    sys.stderr.flush()
    os._exit(code)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj) -> bytes:
    """Stable JSON encoding for hashing — sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _resolve(root: str, abs_in_image: str) -> str:
    """Resolve an absolute-in-image path against an arbitrary root.

    The bootstrap normally runs with root='/', but tests parameterize
    root to a tmpdir mirroring the image layout.
    """
    rel = abs_in_image.lstrip("/")
    return os.path.join(root, rel)


def _load_hash_table(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read())
    except FileNotFoundError:
        _die("hash table missing at " + path)
    except Exception as e:
        _die("hash table unreadable at " + path + ": " + repr(e))
    if not isinstance(data, dict):
        _die("hash table malformed: not a JSON object")
    files = data.get("files")
    dirs = data.get("dirs")
    if not isinstance(files, dict) or not isinstance(dirs, list):
        _die("hash table malformed: missing 'files' object or 'dirs' array")
    return data


def _verify_files(expected: dict, root: str) -> None:
    missing = []
    bad = []
    for rel, want in expected.items():
        abs_path = _resolve(root, rel)
        if not os.path.isfile(abs_path):
            missing.append(rel)
            continue
        try:
            got = _sha256_file(abs_path)
        except Exception as e:
            bad.append(rel + " (" + repr(e) + ")")
            continue
        if got != want:
            bad.append(rel)
    if missing:
        _die("missing required files: " + ", ".join(sorted(missing)))
    if bad:
        _die("hash mismatch on: " + ", ".join(sorted(bad)))


def _verify_dirs(expected_dirs: Iterable[str], expected_files: dict, root: str) -> None:
    """Check each protected directory contains only expected entries.

    A protected directory listing is allowed only:
      - files whose absolute-in-image path is a key of ``expected_files``
      - subdirectories whose absolute-in-image path is in ``expected_dirs``
        (those are walked separately on their own iteration)

    Anything else is treated as an extra file and aborts the boot.
    """
    expected_dirs_set = set(expected_dirs)
    extras = []
    for d in expected_dirs_set:
        abs_dir = _resolve(root, d)
        if not os.path.isdir(abs_dir):
            _die("missing required directory: " + d)
        for entry in os.listdir(abs_dir):
            full = os.path.join(abs_dir, entry)
            # Reconstruct the absolute-in-image path for comparison.
            rel = "/" + os.path.relpath(full, root).replace(os.sep, "/")
            # The hash table itself is expected by file path but not
            # listed in `expected_files` (it can't hash itself). Its
            # integrity comes from hashes_root_sha256 instead.
            if rel == HASHES_PATH:
                continue
            if os.path.isdir(full):
                if rel not in expected_dirs_set:
                    extras.append(rel + "/")
            else:
                if rel not in expected_files:
                    extras.append(rel)
    if extras:
        _die("unexpected entries in protected dirs: " + ", ".join(sorted(extras)))


def run(root: str = "/", *, exec_target: bool = True) -> dict:
    """Verify image integrity, write the boot proof, optionally exec the server.

    ``root`` is parameterized for tests — production uses '/'. When
    ``exec_target`` is False we skip the os.execvp so callers can
    introspect the proof.
    """
    hashes_file = _resolve(root, HASHES_PATH)
    table = _load_hash_table(hashes_file)
    files = table["files"]
    dirs = table["dirs"]

    _verify_files(files, root)
    _verify_dirs(dirs, files, root)

    hashes_root_sha256 = hashlib.sha256(_canonical_json(files)).hexdigest()

    proof = {
        "boot_time": int(time.time()),
        "files_hashed": sorted(files.keys()),
        "file_count": len(files),
        "hashes_root_sha256": hashes_root_sha256,
        "bootstrap_version": BOOTSTRAP_VERSION,
    }
    proof_path = _resolve(root, BOOT_PROOF_PATH)
    proof_dir = os.path.dirname(proof_path)
    if proof_dir:
        os.makedirs(proof_dir, exist_ok=True)
    with open(proof_path, "w") as f:
        f.write(json.dumps(proof))

    if exec_target:
        # Replace the bootstrap process with the real server. The server
        # inherits the cleansed environment and PID 1's process tree.
        os.execvp("python3", ["python3", _resolve(root, SERVER_PATH)])
    return proof


if __name__ == "__main__":
    run()
