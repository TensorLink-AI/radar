"""
Generic affinetes Actor wrapper for stdin/stdout miner agents.

Bridges existing agents that read Challenge JSON from stdin and
write Proposal JSON to stdout into affinetes' HTTP Actor pattern.
Existing agents (systematic, failure_analyst, lineage_tracker) work
without any modification.
"""

import json
import os
import subprocess


class Actor:
    def __init__(self):
        self.entrypoint = os.getenv("AGENT_ENTRYPOINT", "python /app/run.py")

    async def process_challenge(self, challenge_json: str, timeout: int = 120) -> dict:
        """Pass challenge to agent stdin, return Proposal from stdout.

        Always captures stderr as agent_log — the agent's reasoning trace.
        Convention: stdout = structured Proposal JSON, stderr = reasoning.
        """
        try:
            proc = subprocess.run(
                self.entrypoint.split(),
                input=challenge_json, capture_output=True, text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                return {"error": f"Exit code {proc.returncode}",
                        "stderr": proc.stderr[:2000]}
            result = json.loads(proc.stdout.strip())
            result["agent_log"] = proc.stderr[:10000]
            return result
        except subprocess.TimeoutExpired:
            return {"error": f"Agent timed out after {timeout}s"}
        except json.JSONDecodeError as e:
            stderr = proc.stderr[:2000] if "proc" in dir() else ""
            return {"error": f"Invalid JSON: {e}", "stderr": stderr}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    async def health(self) -> dict:
        return {"status": "ok", "entrypoint": self.entrypoint}
