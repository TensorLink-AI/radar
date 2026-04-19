"""Tests for shared/agent_code.py — bundle format, validation, hashing."""

import os
import tempfile

import pytest

from shared.agent_code import (
    bundle_from_directory,
    bundle_from_json,
    bundle_to_json,
    compute_code_hash,
    validate_bundle,
)


# ── compute_code_hash ────────────────────────────────────────────

class TestComputeCodeHash:
    def test_deterministic(self):
        files = {"agent.py": "print('hello')", "helpers.py": "x = 1"}
        h1 = compute_code_hash(files)
        h2 = compute_code_hash(files)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_order_independent(self):
        files_a = {"b.py": "b", "a.py": "a"}
        files_b = {"a.py": "a", "b.py": "b"}
        assert compute_code_hash(files_a) == compute_code_hash(files_b)

    def test_content_change_changes_hash(self):
        h1 = compute_code_hash({"agent.py": "v1"})
        h2 = compute_code_hash({"agent.py": "v2"})
        assert h1 != h2


# ── validate_bundle ──────────────────────────────────────────────

class TestValidateBundle:
    def test_valid_single_file(self):
        bundle = {
            "files": {"agent.py": "def design_architecture(c, cl): pass"},
            "entry_point": "agent.py",
        }
        ok, err = validate_bundle(bundle)
        assert ok, err

    def test_valid_multi_file(self):
        bundle = {
            "files": {
                "agent.py": "from helpers import foo\ndef design_architecture(c, cl): pass",
                "helpers.py": "def foo(): pass",
            },
            "entry_point": "agent.py",
        }
        ok, err = validate_bundle(bundle)
        assert ok, err

    def test_missing_files(self):
        ok, err = validate_bundle({"entry_point": "agent.py"})
        assert not ok
        assert "files" in err

    def test_empty_files(self):
        ok, err = validate_bundle({"files": {}, "entry_point": "agent.py"})
        assert not ok

    def test_missing_entry_point(self):
        bundle = {
            "files": {"other.py": "def design_architecture(c, cl): pass"},
            "entry_point": "agent.py",
        }
        ok, err = validate_bundle(bundle)
        assert not ok
        assert "Entry point" in err

    def test_missing_design_architecture(self):
        bundle = {
            "files": {"agent.py": "def my_func(): pass"},
            "entry_point": "agent.py",
        }
        ok, err = validate_bundle(bundle)
        assert not ok
        assert "design_architecture" in err

    def test_syntax_error(self):
        bundle = {
            "files": {"agent.py": "def ???"},
            "entry_point": "agent.py",
        }
        ok, err = validate_bundle(bundle)
        assert not ok
        assert "Syntax error" in err

    def test_path_traversal_blocked(self):
        bundle = {
            "files": {"../evil.py": "x = 1", "agent.py": "def design_architecture(c, cl): pass"},
            "entry_point": "agent.py",
        }
        ok, err = validate_bundle(bundle)
        assert not ok
        assert "path traversal" in err.lower() or "Invalid" in err

    def test_subdirectory_files_allowed(self):
        bundle = {
            "files": {
                "agent.py": "from core.llm import query\ndef design_architecture(c, cl): pass",
                "core/__init__.py": "",
                "core/llm.py": "def query(): pass",
            },
            "entry_point": "agent.py",
        }
        ok, err = validate_bundle(bundle)
        assert ok, err

    def test_non_py_blocked(self):
        bundle = {
            "files": {"agent.py": "def design_architecture(c, cl): pass", "data.txt": "hi"},
            "entry_point": "agent.py",
        }
        ok, err = validate_bundle(bundle)
        assert not ok
        assert ".py" in err

    def test_default_entry_point(self):
        bundle = {"files": {"agent.py": "def design_architecture(c, cl): pass"}}
        ok, err = validate_bundle(bundle)
        assert ok


# ── bundle_from_directory ────────────────────────────────────────

class TestBundleFromDirectory:
    def test_loads_py_files(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "agent.py"), "w") as f:
                f.write("def design_architecture(c, cl): pass")
            with open(os.path.join(d, "utils.py"), "w") as f:
                f.write("def helper(): pass")
            # Non-py file should be ignored
            with open(os.path.join(d, "notes.txt"), "w") as f:
                f.write("ignore me")

            bundle = bundle_from_directory(d)
            assert "agent.py" in bundle["files"]
            assert "utils.py" in bundle["files"]
            assert "notes.txt" not in bundle["files"]
            assert bundle["entry_point"] == "agent.py"
            assert bundle["code_hash"].startswith("sha256:")

    def test_no_py_files_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "notes.txt"), "w") as f:
                f.write("no python")
            with pytest.raises(ValueError, match="No .py files"):
                bundle_from_directory(d)

    def test_missing_entry_point_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "other.py"), "w") as f:
                f.write("x = 1")
            with pytest.raises(ValueError, match="Entry point"):
                bundle_from_directory(d)

    def test_loads_subdirectory_files(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "agent.py"), "w") as f:
                f.write("from core.llm import query\ndef design_architecture(c, cl): pass")
            core_dir = os.path.join(d, "core")
            os.makedirs(core_dir)
            with open(os.path.join(core_dir, "__init__.py"), "w") as f:
                f.write("")
            with open(os.path.join(core_dir, "llm.py"), "w") as f:
                f.write("def query(): pass")

            bundle = bundle_from_directory(d)
            assert "agent.py" in bundle["files"]
            assert "core/__init__.py" in bundle["files"]
            assert "core/llm.py" in bundle["files"]
            assert bundle["entry_point"] == "agent.py"


# ── JSON round-trip ──────────────────────────────────────────────

class TestJsonRoundTrip:
    def test_round_trip(self):
        bundle = {
            "files": {"agent.py": "def design_architecture(c, cl): pass"},
            "entry_point": "agent.py",
            "code_hash": "sha256:abc",
        }
        raw = bundle_to_json(bundle)
        restored = bundle_from_json(raw)
        assert restored["files"] == bundle["files"]
        assert restored["code_hash"] == bundle["code_hash"]
