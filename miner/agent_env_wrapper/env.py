"""
Affinetes Actor wrapper for the official sandboxed agent image.

Bridges the affinetes HTTP Actor pattern with the frozen harness that
reads Challenge JSON from stdin and writes Proposal JSON to stdout.

Two code-delivery modes:
  - Volume mount (Docker): agent .py files pre-mounted at /workspace/agent/
  - Inline (Basilica):     agent_code dict passed in process_challenge(),
                            written to /workspace/agent/ before running harness
"""

import json
import os
import subprocess


AGENT_DIR = "/workspace/agent"
HARNESS = os.getenv("AGENT_HARNESS", "python /workspace/harness.py")


class Actor:
    def __init__(self):
        os.makedirs(AGENT_DIR, exist_ok=True)

    def _write_inline_code(self, agent_code: dict) -> None:
        """Write inline agent code files to the agent directory."""
        files = agent_code.get("files", {})
        for filename, content in files.items():
            path = os.path.join(AGENT_DIR, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)

        # Set AGENT_MODULE if entry_point specified
        entry = agent_code.get("entry_point", "")
        if entry:
            os.environ["AGENT_MODULE"] = os.path.join(AGENT_DIR, entry)

    async def process_challenge(
        self,
        challenge_json: str,
        timeout: int = 600,
        agent_code: dict | None = None,
        **kwargs,
    ) -> dict:
        """Run the frozen harness with a Challenge, return Proposal.

        Args:
            challenge_json: Serialised Challenge JSON (passed to harness stdin).
            timeout: Max seconds for the agent to run.
            agent_code: Optional dict ``{"files": {...}, "entry_point": "..."}``
                for inline code delivery (Basilica mode).  When provided the
                files are written to /workspace/agent/ before the harness runs.

        Returns:
            dict with code/name/motivation keys, or an error dict.
        """
        # Inline code delivery (Basilica mode)
        if agent_code:
            self._write_inline_code(agent_code)

        try:
            proc = subprocess.run(
                HARNESS.split(),
                input=challenge_json,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                return {
                    "error": f"Harness exit code {proc.returncode}",
                    "stderr": proc.stderr[:2000],
                }
            result = json.loads(proc.stdout.strip())
            # Capture agent reasoning trace from stderr
            result["agent_log"] = proc.stderr[:10000]
            return result
        except subprocess.TimeoutExpired:
            return {"error": f"Agent timed out after {timeout}s"}
        except json.JSONDecodeError as e:
            stderr = proc.stderr[:2000] if "proc" in dir() else ""
            return {"error": f"Invalid JSON from harness: {e}", "stderr": stderr}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    async def health(self) -> dict:
        return {"status": "ok", "harness": HARNESS}
