"""Tests for runner/agent/harness.py and pod_manager agent code injection."""

import json
import os
import shutil
import tempfile

import pytest
from unittest.mock import patch

from validator.pod_manager import (
    pre_validate_agent_code,
    _build_agent_env_vars,
    _write_agent_code,
    _inject_allowed_urls_into_challenge,
)


# ── pre_validate_agent_code ──────────────────────────────────────

class TestPreValidateAgentCode:
    def test_valid(self):
        code = "def design_architecture(challenge, client): pass"
        ok, err = pre_validate_agent_code(code)
        assert ok
        assert err == ""

    def test_missing_design_architecture(self):
        code = "def my_agent(): pass"
        ok, err = pre_validate_agent_code(code)
        assert not ok
        assert "design_architecture" in err

    def test_syntax_error(self):
        code = "def ???"
        ok, err = pre_validate_agent_code(code)
        assert not ok
        assert "Syntax error" in err

    def test_with_extra_functions(self):
        code = (
            "def helper(): pass\n"
            "def design_architecture(challenge, client): pass\n"
            "def another_helper(): pass"
        )
        ok, err = pre_validate_agent_code(code)
        assert ok


# ── _build_agent_env_vars ────────────────────────────────────────

class TestBuildAgentEnvVars:
    def test_includes_allowed_urls(self):
        env = _build_agent_env_vars("http://a.com,http://b.com")
        assert env["AGENT_ALLOWED_URLS"] == "http://a.com,http://b.com"

    def test_no_basilica_token(self):
        with patch.dict(os.environ, {"BASILICA_API_TOKEN": "secret"}):
            env = _build_agent_env_vars()
            assert "BASILICA_API_TOKEN" not in env

    def test_no_r2_credentials(self):
        with patch.dict(os.environ, {
            "R2_ACCESS_KEY_ID": "key",
            "R2_SECRET_ACCESS_KEY": "secret",
        }):
            env = _build_agent_env_vars()
            assert "R2_ACCESS_KEY_ID" not in env
            assert "R2_SECRET_ACCESS_KEY" not in env

    def test_forwards_subtensor(self):
        with patch.dict(os.environ, {
            "SUBTENSOR_NETWORK": "finney",
            "NETUID": "42",
        }):
            env = _build_agent_env_vars()
            assert env["SUBTENSOR_NETWORK"] == "finney"
            assert env["NETUID"] == "42"


# ── _write_agent_code ────────────────────────────────────────────

class TestWriteAgentCode:
    def test_writes_string(self):
        tmpdir = _write_agent_code("print('hello')")
        agent_path = os.path.join(tmpdir, "agent", "agent.py")
        assert os.path.exists(agent_path)
        with open(agent_path) as f:
            assert f.read() == "print('hello')"
        shutil.rmtree(tmpdir)

    def test_writes_bundle(self):
        bundle = {
            "files": {
                "agent.py": "def design_architecture(c, cl): pass",
                "helpers.py": "def foo(): pass",
            },
            "entry_point": "agent.py",
        }
        tmpdir = _write_agent_code(bundle)
        assert os.path.exists(os.path.join(tmpdir, "agent", "agent.py"))
        assert os.path.exists(os.path.join(tmpdir, "agent", "helpers.py"))
        with open(os.path.join(tmpdir, "agent", "helpers.py")) as f:
            assert f.read() == "def foo(): pass"
        shutil.rmtree(tmpdir)


# ── _inject_allowed_urls_into_challenge ──────────────────────────

class TestInjectAllowedUrls:
    def test_injects(self):
        challenge = json.dumps({"seed": 42})
        result = _inject_allowed_urls_into_challenge(challenge, "http://a.com")
        data = json.loads(result)
        assert data["allowed_urls"] == "http://a.com"
        assert data["seed"] == 42

    def test_invalid_json_passthrough(self):
        result = _inject_allowed_urls_into_challenge("not json", "http://a.com")
        assert result == "not json"
