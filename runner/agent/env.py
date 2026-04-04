"""
Affinetes Actor for the official agent sandbox.

Runs inside the radar-agent Docker image on a Basilica pod.
Receives miner agent code + challenge JSON via process_challenge(),
writes code to disk, runs the frozen harness, returns the proposal.
"""

import json
import os
import subprocess
import traceback


class Actor:
    def __init__(self):
        self.workspace = "/workspace"
        self.agent_dir = os.path.join(self.workspace, "agent")

    async def process_challenge(
        self,
        challenge_json: str,
        agent_code: dict | str | None = None,
        timeout: int = 600,
        **kwargs,
    ) -> dict:
        """Run a miner's agent code against a challenge.

        Args:
            challenge_json: Challenge JSON string.
            agent_code: Bundle dict ``{"files": {...}, "entry_point": "..."}``
                or a single code string.
            timeout: Max seconds for the harness subprocess.

        Returns:
            dict with code/name/motivation keys, or dict with "error" key.
        """
        os.makedirs(self.agent_dir, exist_ok=True)

        # Write miner code to /workspace/agent/
        if agent_code:
            if isinstance(agent_code, dict) and "files" in agent_code:
                for filename, code in agent_code["files"].items():
                    path = os.path.join(self.agent_dir, filename)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w") as f:
                        f.write(code)
                entry = agent_code.get("entry_point", "agent.py")
                os.environ["AGENT_MODULE"] = os.path.join(
                    self.agent_dir, entry,
                )
            else:
                with open(os.path.join(self.agent_dir, "agent.py"), "w") as f:
                    f.write(str(agent_code))

        # Run the frozen harness, passing challenge on stdin
        try:
            proc = subprocess.run(
                ["python", os.path.join(self.workspace, "harness.py")],
                input=challenge_json,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace,
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if proc.returncode != 0:
                return {"error": f"Harness exited {proc.returncode}: {stderr}"}

            result = json.loads(stdout)
            # Attach agent log (reasoning trace from stderr)
            result["agent_log"] = stderr
            return result

        except subprocess.TimeoutExpired:
            return {"error": f"Agent timed out after {timeout}s"}
        except json.JSONDecodeError:
            return {"error": f"Harness returned invalid JSON: {stdout[:500]}"}
        except Exception as e:
            return {"error": f"Agent runner error: {e}\n{traceback.format_exc()}"}

    async def health(self) -> dict:
        """Health check."""
        return {"status": "ok", "workspace": self.workspace}
