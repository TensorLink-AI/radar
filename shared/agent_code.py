"""Agent code bundle — the wire format for miner agent submissions.

Miners provide one or more .py files.  The bundle is a JSON object::

    {
        "files": {
            "agent.py": "def design_architecture(challenge, client): ...",
            "helpers.py": "def my_util(): ..."
        },
        "entry_point": "agent.py",
        "code_hash": "sha256:abc123..."
    }

- ``files``:  filename → source code.  All files are written to
  ``/workspace/agent/`` inside the container.
- ``entry_point``:  which file contains ``design_architecture()``.
  Defaults to ``agent.py``.
- ``code_hash``:  SHA-256 of the canonical bundle (sorted filenames,
  concatenated contents).  Committed on-chain for integrity.
"""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def compute_code_hash(files: dict[str, str]) -> str:
    """Deterministic SHA-256 of an agent code bundle.

    Sorts filenames, concatenates ``name + content`` for each, hashes.
    This is the value committed on-chain.
    """
    h = hashlib.sha256()
    for name in sorted(files.keys()):
        h.update(name.encode())
        h.update(files[name].encode())
    return f"sha256:{h.hexdigest()}"


def validate_bundle(bundle: dict) -> tuple[bool, str]:
    """Validate an agent code bundle dict.

    Checks:
    - ``files`` key exists and is a non-empty dict
    - ``entry_point`` file exists in ``files``
    - Entry point contains ``design_architecture``
    - No path traversal in filenames
    - All values are strings
    """
    files = bundle.get("files")
    if not isinstance(files, dict) or not files:
        return False, "Bundle must have a non-empty 'files' dict"

    entry = bundle.get("entry_point", "agent.py")
    if entry not in files:
        return False, f"Entry point '{entry}' not in files: {sorted(files.keys())}"

    for name, code in files.items():
        if not isinstance(name, str) or not isinstance(code, str):
            return False, f"File names and contents must be strings, got {type(name)}/{type(code)}"
        if "/" in name or "\\" in name or ".." in name:
            return False, f"Invalid filename (no paths allowed): {name!r}"
        if not name.endswith(".py"):
            return False, f"Only .py files allowed: {name!r}"

    # Check entry point has design_architecture
    import ast
    try:
        tree = ast.parse(files[entry])
    except SyntaxError as e:
        return False, f"Syntax error in {entry}: {e}"

    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "design_architecture" not in func_names:
        return False, f"Entry point {entry} missing design_architecture()"

    return True, ""


def bundle_from_directory(directory: str, entry_point: str = "agent.py") -> dict:
    """Load all .py files from a directory into a bundle dict.

    Used by miners to build their submission from a local directory.
    """
    import os

    files: dict[str, str] = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path) as f:
                files[name] = f.read()

    if not files:
        raise ValueError(f"No .py files found in {directory}")
    if entry_point not in files:
        raise ValueError(f"Entry point {entry_point} not found in {directory}")

    return {
        "files": files,
        "entry_point": entry_point,
        "code_hash": compute_code_hash(files),
    }


def bundle_to_json(bundle: dict) -> str:
    """Serialize a bundle to JSON."""
    return json.dumps(bundle, sort_keys=True)


def bundle_from_json(raw: str) -> dict:
    """Deserialize a bundle from JSON."""
    return json.loads(raw)
