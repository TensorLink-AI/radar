"""LLM-powered miner agent using Chutes for inference.

Reads Challenge JSON from stdin, uses an LLM to design competitive
ML architectures for any task, writes Proposal JSON to stdout.

The agent is fully task-agnostic — it reads the task spec from the challenge
and adapts its prompts, validation, and output accordingly.

Requires:
  - CHUTES_API_KEY env var for Chutes inference
  - CHUTES_MODEL env var (default: deepseek-ai/DeepSeek-V3-0324)

Build: docker build -t my-agent:latest .
"""

import ast
import json
import os
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "")
CHUTES_BASE_URL = os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
CHUTES_MODEL = os.getenv("CHUTES_MODEL", "deepseek-ai/DeepSeek-V3-0324")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", "10"))


def log(msg: str):
    """Write reasoning trace to stderr (captured by validator)."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Scratchpad persistence
# ---------------------------------------------------------------------------
def load_scratchpad(challenge: dict, local_dir: str = "/tmp/scratchpad") -> str:
    os.makedirs(local_dir, exist_ok=True)
    url = challenge.get("scratchpad_get_url", "")
    if not url:
        return local_dir
    try:
        archive_path = os.path.join(local_dir, "state.tar.gz")
        with urllib.request.urlopen(url, timeout=10) as resp:
            with open(archive_path, "wb") as f:
                f.write(resp.read())
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(local_dir, filter="data")
        os.remove(archive_path)
        log(f"Loaded scratchpad ({len(os.listdir(local_dir))} files)")
    except Exception as e:
        log(f"Scratchpad load: {e} (starting fresh)")
    return local_dir


def save_scratchpad(challenge: dict, local_dir: str = "/tmp/scratchpad") -> bool:
    url = challenge.get("scratchpad_put_url", "")
    if not url:
        return False
    try:
        fd, archive_path = tempfile.mkstemp(suffix=".tar.gz", prefix="scratchpad_")
        os.close(fd)
        with tarfile.open(archive_path, "w:gz") as tar:
            for root, _dirs, files in os.walk(local_dir):
                for fname in files:
                    full = os.path.join(root, fname)
                    arcname = os.path.relpath(full, local_dir)
                    tar.add(full, arcname=arcname)
        max_mb = challenge.get("scratchpad_max_mb", 10)
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        if size_mb > max_mb:
            log(f"Scratchpad too large ({size_mb:.1f}MB > {max_mb}MB). Skipping.")
            os.remove(archive_path)
            return False
        with open(archive_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Content-Type", "application/gzip")
        urllib.request.urlopen(req, timeout=15)
        os.remove(archive_path)
        log(f"Saved scratchpad ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        log(f"Scratchpad save failed: {e}")
        return False


# ---------------------------------------------------------------------------
# DB queries (validator experiment database)
# ---------------------------------------------------------------------------
def _db_get(base_url: str, path: str) -> dict | list | None:
    """GET from the validator DB. Returns parsed JSON or None on error."""
    if not base_url:
        return None
    url = base_url.rstrip("/") + path
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=DB_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log(f"DB query {path}: {e}")
        return None


def gather_db_context(challenge: dict) -> dict:
    """Query the validator DB for useful context about past experiments."""
    db_url = challenge.get("db_url", "")
    if not db_url:
        return {}

    context = {}

    recent = _db_get(db_url, "/experiments/recent?n=15")
    if recent:
        context["recent_experiments"] = [
            {
                "name": e.get("name", "?"),
                "metric": e.get("metric"),
                "success": e.get("success", False),
                "analysis": e.get("analysis", ""),
                "flops": e.get("flops_equivalent_size"),
            }
            for e in recent[:15]
        ]

    pareto = _db_get(db_url, "/experiments/pareto")
    if pareto:
        context["pareto_front"] = [
            {
                "name": e.get("name", "?"),
                "metric": e.get("metric"),
                "objectives": e.get("objectives", {}),
                "code_snippet": (e.get("code", "") or "")[:500],
            }
            for e in pareto[:10]
        ]

    comp_stats = _db_get(db_url, "/provenance/component_stats")
    if comp_stats:
        context["component_stats"] = comp_stats

    dead_ends = _db_get(db_url, "/provenance/dead_ends")
    if dead_ends:
        context["dead_ends"] = dead_ends.get("dead_ends", [])[:10]

    failures = _db_get(db_url, "/experiments/failures?n=5")
    if failures:
        context["recent_failures"] = [
            {
                "name": e.get("name", "?"),
                "analysis": e.get("analysis", ""),
            }
            for e in failures[:5]
        ]

    return context


# ---------------------------------------------------------------------------
# LLM inference via Chutes (OpenAI-compatible API)
# ---------------------------------------------------------------------------
def llm_chat(messages: list[dict], temperature: float = None) -> str:
    """Call Chutes LLM API (OpenAI-compatible chat completions)."""
    if not CHUTES_API_KEY:
        raise RuntimeError("CHUTES_API_KEY not set")

    url = CHUTES_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": CHUTES_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {CHUTES_API_KEY}")
    req.add_header("Content-Type", "application/json")

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            log(f"LLM call attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM failed after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Code validation
# ---------------------------------------------------------------------------
FORBIDDEN_MODULES = ("subprocess", "socket", "ftplib")


def validate_code(code: str, task: dict) -> tuple[bool, str]:
    """Validate generated code. Checks are task-aware."""
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # 2. Collect defined function names
    func_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    # 3. For harness-based tasks (run_command contains "harness.py"),
    #    build_model and build_optimizer are required.
    #    For standalone tasks (run_command is "python {target}"),
    #    the code just needs to be valid Python — no required functions.
    run_cmd = task.get("run_command", "python {target}")
    uses_harness = "harness.py" in run_cmd

    if uses_harness:
        if "build_model" not in func_names:
            return False, "Missing build_model() function"
        if "build_optimizer" not in func_names:
            return False, "Missing build_optimizer() function"

    # 4. Check no dangerous imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_MODULES:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_MODULES:
                return False, f"Forbidden import: {node.module}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Prompt construction (task-agnostic)
# ---------------------------------------------------------------------------
def build_system_prompt(task: dict) -> str:
    """Build system prompt dynamically from the task spec."""
    task_name = task.get("name", "unknown")
    domain_prompt = task.get("domain_system_prompt", "")
    constraints = task.get("constraints", [])
    anti_patterns = task.get("anti_patterns", [])
    hypotheses = task.get("example_hypotheses", [])
    objectives = task.get("objectives", [])
    run_cmd = task.get("run_command", "python {target}")
    uses_harness = "harness.py" in run_cmd

    parts = []

    # Domain expertise
    if domain_prompt:
        parts.append(domain_prompt.strip())
    else:
        parts.append(
            "You are an expert autonomous ML researcher. You design PyTorch "
            "code that achieves state-of-the-art results under compute constraints."
        )

    parts.append("")
    parts.append("## Output Format")
    parts.append(
        "You MUST output a single Python code block (```python ... ```) "
        "containing a complete, runnable module. No explanations outside "
        "the code block."
    )

    # Task-specific interface requirements
    if uses_harness:
        parts.append("")
        parts.append("## Required Interface (harness-based task)")
        parts.append("The module MUST define:")
        parts.append("  - build_model(...) -> nn.Module")
        parts.append("  - build_optimizer(model) -> torch.optim.Optimizer")
        parts.append("")
        parts.append("Optional hooks you can define:")
        parts.append("  - training_config() -> dict (batch_size, grad_accum_steps, "
                      "grad_clip, eval_interval)")
        parts.append("  - init_weights(model) -> None")
        parts.append("  - configure_amp() -> dict (enabled, dtype)")
        parts.append("  - transform_batch(batch, step, total_steps) -> dict")
        parts.append("  - on_step_end(model, optimizer, step, total_steps, "
                      "loss_value) -> None")
        parts.append("  - build_scheduler(optimizer, total_steps) -> LRScheduler")
        parts.append("  - compute_loss(predictions, targets, quantiles_list) "
                      "-> Tensor")
        parts.append("  - COMPILE = True/False module attribute")
    else:
        parts.append("")
        parts.append("## Required Interface (standalone task)")
        parts.append(
            "The module IS the full training script. It must run end-to-end "
            "when executed with `python submission.py` and print metrics to "
            "stdout in the format the runner expects."
        )

    # Objectives
    if objectives:
        parts.append("")
        parts.append("## Scoring Objectives")
        for obj in objectives:
            direction = "lower is better" if obj.get("lower_is_better", True) else "higher is better"
            primary = " [PRIMARY]" if obj.get("primary") else ""
            parts.append(
                f"  - {obj['name']} (weight={obj.get('weight', 1.0)}, "
                f"{direction}){primary}"
            )
        parts.append("  - Pareto dominance bonus: 1.5x if you beat ALL metrics")

    # Constraints
    if constraints:
        parts.append("")
        parts.append("## Constraints")
        for c in constraints:
            parts.append(f"  - {c}")

    # Rules
    parts.append("")
    parts.append("## Rules")
    parts.append("  - Only use torch and standard library. No external packages.")
    parts.append("  - FLOPs-equivalent MUST be within the specified [min, max] range.")
    parts.append("  - Target ~60% of max FLOPs for safety margin.")
    parts.append("  - Code must be syntactically valid and runnable.")
    parts.append("  - Use modern PyTorch best practices.")

    # Anti-patterns
    if anti_patterns:
        parts.append("")
        parts.append("## Anti-patterns (AVOID)")
        for ap in anti_patterns:
            parts.append(f"  - {ap}")

    # Hypotheses for inspiration
    if hypotheses:
        parts.append("")
        parts.append("## Example Hypotheses for Inspiration")
        for h in hypotheses:
            parts.append(f"  - {h}")

    return "\n".join(parts)


def build_user_prompt(challenge: dict, db_context: dict, history: list) -> str:
    """Build the user message with all available context."""
    min_flops = challenge.get("min_flops_equivalent", 0)
    max_flops = challenge.get("max_flops_equivalent", 0)
    target_flops = int(max_flops * 0.6)
    frontier = challenge.get("feasible_frontier", [])
    seed = challenge.get("seed", 0)
    task = challenge.get("task", {})
    task_name = task.get("name", "unknown")
    task_desc = task.get("description", "")

    parts = []
    parts.append(f"## Task: {task_name}")
    if task_desc:
        parts.append(task_desc.strip())
    parts.append("")

    parts.append(f"## Round Context")
    parts.append(f"- FLOPs budget: [{min_flops:,} — {max_flops:,}]")
    parts.append(f"- Target FLOPs (60% of max): ~{target_flops:,}")
    parts.append(f"- Seed: {seed}")
    parts.append(f"- Time budget: {task.get('time_budget', 300)}s")
    parts.append("")

    # Frontier
    if frontier:
        parts.append(f"## Current Frontier ({len(frontier)} members)")
        primary_obj = None
        for obj in task.get("objectives", []):
            if obj.get("primary"):
                primary_obj = obj
                break
        metric_name = primary_obj["name"] if primary_obj else "metric"
        best = min(frontier, key=lambda x: x.get("metric", float("inf")))
        parts.append(f"Best {metric_name}: {best.get('metric', '?')}")
        for i, f_entry in enumerate(frontier[:5]):
            parts.append(
                f"\n### Frontier #{i+1} "
                f"({metric_name}={f_entry.get('metric', '?')})"
            )
            code = f_entry.get("code", "")
            if code:
                parts.append(f"```python\n{code[:2000]}\n```")
        parts.append("")
    else:
        parts.append("## No frontier exists yet for this size bucket.")
        parts.append("Design a strong baseline architecture.\n")

    # DB context
    if db_context.get("component_stats"):
        parts.append("## Component Performance Stats")
        stats = db_context["component_stats"]
        if isinstance(stats, dict):
            for comp, info in list(stats.items())[:15]:
                parts.append(f"- {comp}: {info}")
        elif isinstance(stats, list):
            for item in stats[:15]:
                parts.append(f"- {item}")
        parts.append("")

    if db_context.get("recent_experiments"):
        parts.append("## Recent Experiments")
        for exp in db_context["recent_experiments"][:10]:
            status = "OK" if exp.get("success") else "FAIL"
            parts.append(
                f"- [{status}] {exp['name']}: metric={exp.get('metric', '?')}, "
                f"flops={exp.get('flops', '?')}"
            )
            if exp.get("analysis"):
                parts.append(f"  Analysis: {exp['analysis'][:200]}")
        parts.append("")

    if db_context.get("dead_ends"):
        parts.append("## Dead Ends (avoid these patterns)")
        for de in db_context["dead_ends"][:5]:
            parts.append(f"- {de}")
        parts.append("")

    if db_context.get("recent_failures"):
        parts.append("## Recent Failures")
        for fail in db_context["recent_failures"][:5]:
            parts.append(f"- {fail['name']}: {fail.get('analysis', '?')}")
        parts.append("")

    # Scratchpad history
    if history:
        parts.append("## Your Previous Submissions This Session")
        for h in history[-5:]:
            parts.append(
                f"- Round {h.get('round', '?')}: {h.get('name', '?')} "
                f"(metric={h.get('metric', '?')})"
            )
            if h.get("notes"):
                parts.append(f"  Notes: {h['notes']}")
        parts.append("")

    parts.append("## Instruction")
    if frontier:
        parts.append(
            "Design code that IMPROVES on the current frontier. "
            "Study the frontier code carefully and identify weaknesses or "
            "missed opportunities. Be creative but stay within FLOPs budget."
        )
    else:
        parts.append(
            "Design a strong baseline for this size bucket. "
            "Focus on proven techniques. Stay well within the FLOPs budget."
        )
    parts.append(
        "\nOutput ONLY a Python code block (```python ... ```) with the "
        "complete module. No explanations outside the code block."
    )

    return "\n".join(parts)


def extract_code_block(response: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.index("```", start)
        return response[start:end].strip()

    if "```" in response:
        start = response.index("```") + 3
        end = response.index("```", start)
        return response[start:end].strip()

    return response.strip()


# ---------------------------------------------------------------------------
# History tracking (scratchpad)
# ---------------------------------------------------------------------------
def load_history(scratch_dir: str) -> list:
    path = os.path.join(scratch_dir, "history.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_history(scratch_dir: str, history: list):
    path = os.path.join(scratch_dir, "history.json")
    try:
        with open(path, "w") as f:
            json.dump(history[-50:], f)
    except Exception as e:
        log(f"History save failed: {e}")


# ---------------------------------------------------------------------------
# Main agent logic
# ---------------------------------------------------------------------------
def design_architecture(challenge: dict) -> dict:
    """Use LLM to design a competitive architecture for any task."""
    scratch_dir = load_scratchpad(challenge)
    history = load_history(scratch_dir)

    min_flops = challenge.get("min_flops_equivalent", 0)
    max_flops = challenge.get("max_flops_equivalent", 0)
    round_id = challenge.get("round_id", 0)
    task = challenge.get("task", {})
    task_name = task.get("name", "unknown")
    log(f"Round {round_id}: task={task_name}, FLOPs [{min_flops:,} — {max_flops:,}]")

    # Gather context from validator DB
    db_context = gather_db_context(challenge)
    log(f"DB context: {list(db_context.keys())}")

    # Build task-aware prompts
    system_prompt = build_system_prompt(task)
    user_prompt = build_user_prompt(challenge, db_context, history)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    code = None
    for attempt in range(3):
        try:
            log(f"LLM attempt {attempt + 1}...")
            response = llm_chat(messages)
            candidate = extract_code_block(response)

            ok, err = validate_code(candidate, task)
            if ok:
                code = candidate
                log("LLM code validated OK")
                break
            else:
                log(f"Validation failed: {err}")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The code has an error: {err}\n"
                        f"Fix it and output the corrected complete Python "
                        f"module in a single ```python code block."
                    ),
                })
        except Exception as e:
            log(f"LLM attempt {attempt + 1} error: {e}")
            log(traceback.format_exc())

    if code is None:
        log("ERROR: All LLM attempts failed. No submission.")
        # Save history even on failure
        history.append({
            "round": round_id, "name": "FAILED", "task": task_name,
            "min_flops": min_flops, "max_flops": max_flops,
            "used_llm": True, "success": False, "timestamp": time.time(),
        })
        save_history(scratch_dir, history)
        save_scratchpad(challenge, scratch_dir)
        return {"code": "", "name": "failed", "motivation": "LLM failed"}

    # Extract a name from the code
    try:
        tree = ast.parse(code)
        classes = [
            n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        ]
        name = classes[0] if classes else f"llm_{task_name}_r{round_id}"
    except Exception:
        name = f"llm_{task_name}_r{round_id}"

    motivation = (
        f"LLM-designed for {task_name} "
        f"FLOPs [{min_flops:,}—{max_flops:,}]. "
        f"Model: {CHUTES_MODEL}"
    )

    # Record in history
    history.append({
        "round": round_id, "name": name, "task": task_name,
        "min_flops": min_flops, "max_flops": max_flops,
        "used_llm": True, "success": True, "timestamp": time.time(),
    })
    save_history(scratch_dir, history)
    save_scratchpad(challenge, scratch_dir)

    return {"code": code, "name": name, "motivation": motivation}


def main():
    challenge = json.loads(sys.stdin.read())
    task_name = challenge.get("task", {}).get("name", "unknown")
    log(f"Agent starting (Chutes LLM backend, task={task_name})")
    proposal = design_architecture(challenge)
    log(f"Proposal: {proposal['name']} — {proposal['motivation']}")
    print(json.dumps(proposal))


if __name__ == "__main__":
    main()
