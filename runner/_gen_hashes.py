"""Build-time hash table generator for the trainer image.

Run by the Dockerfile after every COPY is done. Walks the explicit file
list below, computes sha256 for each, and writes
/workspace/_bootstrap_hashes.json. The bootstrap reads that JSON at
container start and refuses to run on any mismatch or unexpected file
in a protected directory.

Single source of truth — _bootstrap.py does NOT keep its own copy of
this list. If you add a COPY to the Dockerfile, add the destination
here. If you remove one, remove it here. Build fails loudly if the list
mentions a file that's missing on disk, or if a protected directory
contains an entry that isn't on the list.

The generator does NOT include _bootstrap_hashes.json in the table
(that would be circular). Its integrity is covered by:
  - chmod 444 + the image's content-addressed digest, and
  - the bootstrap's hashes_root_sha256 field, which validators compare
    against the expected canonical-encoded hash table.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys


# Absolute-in-image paths of every load-bearing file COPYed into the
# image by either runner/Dockerfile or runner/timeseries_forecast/
# Dockerfile. The build-time check below drops anything that's not on
# disk in the current build, so per-Dockerfile flavors stay correct.
EXPECTED_FILES = [
    # Container-level scripts
    "/usr/local/bin/radar-entrypoint.sh",
    "/usr/local/bin/sandbox_wrap.sh",
    # Workspace top-level
    "/workspace/_bootstrap.py",
    "/workspace/_gen_hashes.py",
    "/workspace/server.py",
    "/workspace/launcher.py",
    "/workspace/sandbox_runner.py",
    "/workspace/harness.py",
    "/workspace/prepare.py",
    "/workspace/evaluate.py",
    "/workspace/flops.py",
    "/workspace/pretrain_loader.py",
    "/workspace/env.py",
    # Generic harness package
    "/workspace/runner/__init__.py",
    "/workspace/runner/harness.py",
    "/workspace/runner/server.py",
    "/workspace/runner/launcher.py",
    "/workspace/runner/handler.py",
    "/workspace/runner/sandbox.py",
    "/workspace/runner/uploads.py",
    "/workspace/runner/boot_proof.py",
    # Timeseries task package
    "/workspace/runner/timeseries_forecast/__init__.py",
    "/workspace/runner/timeseries_forecast/train.py",
    "/workspace/runner/timeseries_forecast/eval_template.py",
    "/workspace/runner/timeseries_forecast/harness.py",
    "/workspace/runner/timeseries_forecast/prepare.py",
    "/workspace/runner/timeseries_forecast/evaluate.py",
    "/workspace/runner/timeseries_forecast/flops.py",
    "/workspace/runner/timeseries_forecast/pretrain_loader.py",
    # Frozen reference copies
    "/workspace/frozen/harness.py",
    "/workspace/frozen/prepare.py",
    "/workspace/frozen/evaluate.py",
    "/workspace/frozen/flops.py",
    "/workspace/frozen/auth.py",
    "/workspace/frozen/server.py",
    # Shared modules
    "/workspace/shared/__init__.py",
    "/workspace/shared/auth.py",
    "/workspace/shared/artifacts.py",
    "/workspace/shared/r2_audit.py",
    "/workspace/shared/pretrain_data.py",
    "/workspace/shared/gift_eval.py",
]

# Protected directories — bootstrap rejects any entry inside these that
# isn't in EXPECTED_FILES (or itself a protected directory). Per-job
# /var/radar/sandbox is intentionally excluded so harness writes survive.
EXPECTED_DIRS = [
    "/workspace",
    "/workspace/runner",
    "/workspace/runner/timeseries_forecast",
    "/workspace/shared",
    "/workspace/frozen",
]

OUTPUT_PATH = "/workspace/_bootstrap_hashes.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    files = {}
    missing = []
    for path in EXPECTED_FILES:
        if not os.path.isfile(path):
            missing.append(path)
            continue
        files[path] = _sha256(path)
    if missing:
        sys.stderr.write(
            "radar-gen-hashes: expected files missing from the image: "
            + ", ".join(missing) + "\n"
        )
        return 1

    # Cross-check: every entry in a protected dir must be on the list,
    # otherwise an extra file slipped in via COPY without us noticing.
    expected_set = set(files.keys())
    expected_dirs_set = set(EXPECTED_DIRS)
    extras = []
    for d in EXPECTED_DIRS:
        if not os.path.isdir(d):
            sys.stderr.write("radar-gen-hashes: protected dir missing: " + d + "\n")
            return 1
        for entry in os.listdir(d):
            full = os.path.join(d, entry)
            rel = "/" + os.path.relpath(full, "/").replace(os.sep, "/")
            # Skip the output file we're about to write — covered by
            # hashes_root_sha256, not by self-hashing.
            if rel == OUTPUT_PATH:
                continue
            if os.path.isdir(full):
                if rel not in expected_dirs_set:
                    extras.append(rel + "/")
            else:
                if rel not in expected_set:
                    extras.append(rel)
    if extras:
        sys.stderr.write(
            "radar-gen-hashes: unexpected entries in protected dirs: "
            + ", ".join(sorted(extras)) + "\n"
        )
        return 1

    table = {
        "version": "1",
        "files": files,
        "dirs": EXPECTED_DIRS,
    }
    payload = _canonical_json(table)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(payload)
    sys.stdout.write(
        "radar-gen-hashes: wrote " + OUTPUT_PATH
        + " (" + str(len(files)) + " files, "
        + hashlib.sha256(_canonical_json(files)).hexdigest()[:16] + " root)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
