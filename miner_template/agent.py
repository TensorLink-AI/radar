"""LLM-powered miner agent using Chutes for inference.

Reads Challenge JSON from stdin, uses an LLM to design competitive
time-series forecasting architectures, writes Proposal JSON to stdout.

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
import textwrap
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

    # Recent experiments (what's been tried)
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

    # Pareto front (best experiments)
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

    # Component stats (which building blocks correlate with success)
    comp_stats = _db_get(db_url, "/provenance/component_stats")
    if comp_stats:
        context["component_stats"] = comp_stats

    # Dead ends (patterns to avoid)
    dead_ends = _db_get(db_url, "/provenance/dead_ends")
    if dead_ends:
        context["dead_ends"] = dead_ends.get("dead_ends", [])[:10]

    # Failure patterns
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
def validate_code(code: str) -> tuple[bool, str]:
    """Validate generated architecture code for basic correctness."""
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # 2. Check required functions exist
    func_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    if "build_model" not in func_names:
        return False, "Missing build_model() function"
    if "build_optimizer" not in func_names:
        return False, "Missing build_optimizer() function"

    # 3. Check no dangerous imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("subprocess", "socket", "http", "ftplib"):
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in (
                "subprocess", "socket", "http", "ftplib",
            ):
                return False, f"Forbidden import: {node.module}"

    # 4. Check build_model signature has correct params
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_model":
            params = [arg.arg for arg in node.args.args]
            expected = {"context_len", "prediction_len", "num_variates", "quantiles"}
            if not expected.issubset(set(params)):
                return False, (
                    f"build_model params {params} missing required: "
                    f"{expected - set(params)}"
                )
            break

    return True, "OK"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert autonomous ML researcher specializing in time-series
    forecasting models. You design PyTorch architectures that minimize CRPS
    (Continuous Ranked Probability Score) on diverse forecasting benchmarks.

    You MUST output a single Python code block containing a complete module.
    The module MUST define:
      - build_model(context_len, prediction_len, num_variates, quantiles)
        -> nn.Module
      - build_optimizer(model) -> torch.optim.Optimizer

    Model contract:
      Input:  (batch, context_len, num_variates) float tensor
      Output: (batch, prediction_len, num_variates, len(quantiles)) float tensor

    Available optional hooks (define them if useful):
      - training_config() -> dict with batch_size, grad_accum_steps,
        grad_clip, eval_interval
      - init_weights(model) -> None (custom init, must NOT change param count)
      - configure_amp() -> dict with enabled, dtype
      - transform_batch(batch, step, total_steps) -> dict with "context",
        "target" keys
      - on_step_end(model, optimizer, step, total_steps, loss_value) -> None
      - build_scheduler(optimizer, total_steps) -> LRScheduler
      - compute_loss(predictions, targets, quantiles_list) -> Tensor
      - COMPILE = True/False module attribute for torch.compile

    Rules:
      - Only use torch and standard library. No external packages.
      - FLOPs-equivalent MUST be within the specified [min, max] range.
        Target ~60% of max for safety margin.
      - Code must be syntactically valid and runnable.
      - Output ONLY the Python code block. No explanations outside the block.
      - Use modern PyTorch best practices.

    Scoring priorities (weights):
      1. CRPS (1.0) — primary metric, lower is better
      2. MASE (0.5) — secondary, lower is better
      3. Training time (0.2) — faster is better
      4. Memory (0.1) — less is better
      5. Pareto dominance bonus: 1.5x if you beat ALL metrics

    Known effective techniques for time-series:
      - PatchTST-style patching (patch_size=16 or 32)
      - Reversible Instance Normalization (RevIN)
      - Channel-independent processing
      - Relative/rotary positional encoding
      - SwiGLU / GeGLU feed-forward layers
      - Cosine annealing with warmup
      - Mixture of quantile heads
""")


def build_user_prompt(challenge: dict, db_context: dict, history: list) -> str:
    """Build the user message with all available context."""
    min_flops = challenge.get("min_flops_equivalent", 0)
    max_flops = challenge.get("max_flops_equivalent", 0)
    target_flops = int(max_flops * 0.6)
    frontier = challenge.get("feasible_frontier", [])
    seed = challenge.get("seed", 0)

    parts = []
    parts.append(f"## Round Context")
    parts.append(f"- FLOPs budget: [{min_flops:,} — {max_flops:,}]")
    parts.append(f"- Target FLOPs (60% of max): ~{target_flops:,}")
    parts.append(f"- Seed: {seed}")
    parts.append(f"- Context length: 512, Prediction length: 96")
    parts.append(f"- Num variates: 1 (univariate)")
    parts.append(f"- Quantiles: [0.1, 0.2, ..., 0.9] (9 quantiles)")
    parts.append("")

    # Frontier
    if frontier:
        parts.append(f"## Current Frontier ({len(frontier)} members)")
        best = min(frontier, key=lambda x: x.get("metric", float("inf")))
        parts.append(f"Best CRPS: {best.get('metric', '?')}")
        for i, f_entry in enumerate(frontier[:5]):
            parts.append(f"\n### Frontier #{i+1} (CRPS={f_entry.get('metric', '?')})")
            code = f_entry.get("code", "")
            if code:
                # Show key parts of the code
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
                f"(CRPS={h.get('metric', '?')})"
            )
            if h.get("notes"):
                parts.append(f"  Notes: {h['notes']}")
        parts.append("")

    parts.append("## Task")
    if frontier:
        parts.append(
            "Design an architecture that IMPROVES on the current frontier. "
            "Study the frontier code carefully and identify weaknesses or "
            "missed opportunities. Be creative but stay within FLOPs budget."
        )
    else:
        parts.append(
            "Design a strong baseline architecture for this size bucket. "
            "Focus on proven techniques: patching, RevIN normalization, "
            "efficient attention. Stay well within the FLOPs budget."
        )
    parts.append(
        "\nOutput ONLY a Python code block (```python ... ```) with the "
        "complete module. No explanations outside the code block."
    )

    return "\n".join(parts)


def extract_code_block(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try ```python ... ``` first
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.index("```", start)
        return response[start:end].strip()

    # Try ``` ... ```
    if "```" in response:
        start = response.index("```") + 3
        end = response.index("```", start)
        return response[start:end].strip()

    # Assume the whole response is code
    return response.strip()


# ---------------------------------------------------------------------------
# Fallback: deterministic architecture (no LLM needed)
# ---------------------------------------------------------------------------
def _pick_hyperparams(min_flops: int, max_flops: int) -> dict:
    """Pick hyperparams to fit the FLOPs range. Targets 60% of max."""
    target = int(max_flops * 0.6)
    presets = [
        (16,  2, 1,   32,    150_000),
        (24,  2, 1,   64,    350_000),
        (32,  2, 2,   64,    800_000),
        (48,  4, 2,  128,  1_500_000),
        (64,  4, 2,  192,  3_500_000),
        (64,  4, 3,  256,  7_000_000),
        (96,  4, 3,  384, 20_000_000),
        (128, 4, 3,  512, 40_000_000),
        (128, 4, 4,  512, 60_000_000),
        (160, 8, 4,  640, 90_000_000),
    ]
    best = presets[0]
    for preset in presets:
        if preset[4] <= target:
            best = preset
    return {
        "d_model": best[0], "nhead": best[1],
        "num_layers": best[2], "dim_feedforward": best[3],
    }


def fallback_architecture(challenge: dict) -> str:
    """Generate a solid PatchTST-style architecture without LLM."""
    min_flops = challenge.get("min_flops_equivalent", 0)
    max_flops = challenge.get("max_flops_equivalent", 0)
    hp = _pick_hyperparams(min_flops, max_flops)
    d = hp["d_model"]
    nh = hp["nhead"]
    nl = hp["num_layers"]
    ff = hp["dim_feedforward"]
    ps = 16 if d >= 32 else 8

    return textwrap.dedent(f"""\
        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class RevIN(nn.Module):
            \"\"\"Reversible Instance Normalization for cross-domain robustness.\"\"\"
            def __init__(self, num_features, eps=1e-5):
                super().__init__()
                self.eps = eps
                self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
                self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

            def forward(self, x, mode):
                if mode == "norm":
                    self._mean = x.mean(dim=1, keepdim=True).detach()
                    self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
                    x = (x - self._mean) / self._std
                    x = x * self.affine_weight + self.affine_bias
                    return x
                else:
                    x = (x - self.affine_bias) / self.affine_weight
                    x = x * self._std + self._mean
                    return x


        class PatchTSForecaster(nn.Module):
            def __init__(self, context_len, prediction_len, num_variates, quantiles):
                super().__init__()
                self.context_len = context_len
                self.prediction_len = prediction_len
                self.num_variates = num_variates
                self.num_quantiles = len(quantiles)

                patch_size = {ps}
                d_model = {d}
                num_patches = context_len // patch_size
                self.patch_size = patch_size
                self.num_patches = num_patches

                self.revin = RevIN(num_variates)
                self.patch_embed = nn.Linear(patch_size * num_variates, d_model)
                self.pos_embed = nn.Parameter(
                    torch.randn(1, num_patches, d_model) * 0.02
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead={nh},
                    dim_feedforward={ff}, dropout=0.1,
                    batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers={nl})
                self.head = nn.Linear(
                    d_model * num_patches,
                    prediction_len * num_variates * self.num_quantiles,
                )

            def forward(self, x):
                B, T, V = x.shape
                x = self.revin(x, "norm")
                patches = x.reshape(B, self.num_patches, self.patch_size * V)
                h = self.patch_embed(patches) + self.pos_embed
                h = self.encoder(h)
                h = h.reshape(B, -1)
                out = self.head(h)
                out = out.view(B, self.prediction_len, self.num_variates,
                               self.num_quantiles)
                # Denorm the median prediction concept via RevIN
                # (simplified: denorm all quantiles equally)
                out_flat = out.permute(0, 3, 1, 2).reshape(
                    B * self.num_quantiles, self.prediction_len, self.num_variates
                )
                out_denorm = self.revin(out_flat, "denorm")
                out = out_denorm.reshape(
                    B, self.num_quantiles, self.prediction_len, self.num_variates
                ).permute(0, 2, 3, 1)
                return out


        def build_model(context_len, prediction_len, num_variates, quantiles):
            return PatchTSForecaster(
                context_len, prediction_len, num_variates, quantiles
            )


        def build_optimizer(model):
            return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


        def build_scheduler(optimizer, total_steps):
            warmup = min(total_steps // 10, 200)
            def lr_lambda(step):
                if step < warmup:
                    return step / max(warmup, 1)
                progress = (step - warmup) / max(total_steps - warmup, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


        def configure_amp():
            return {{"enabled": True, "dtype": "bfloat16"}}


        def training_config():
            return {{
                "batch_size": 128,
                "grad_accum_steps": 1,
                "grad_clip": 1.0,
                "eval_interval": 100,
            }}


        def init_weights(model):
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    """)


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
            json.dump(history[-50:], f)  # keep last 50
    except Exception as e:
        log(f"History save failed: {e}")


# ---------------------------------------------------------------------------
# Main agent logic
# ---------------------------------------------------------------------------
def design_architecture(challenge: dict) -> dict:
    """Use LLM to design a competitive architecture."""
    scratch_dir = load_scratchpad(challenge)
    history = load_history(scratch_dir)

    min_flops = challenge.get("min_flops_equivalent", 0)
    max_flops = challenge.get("max_flops_equivalent", 0)
    round_id = challenge.get("round_id", 0)
    log(f"Round {round_id}: FLOPs [{min_flops:,} — {max_flops:,}]")

    # Gather context from validator DB
    db_context = gather_db_context(challenge)
    log(f"DB context: {list(db_context.keys())}")

    # Try LLM-powered design
    code = None
    name = "unknown"
    motivation = ""

    if CHUTES_API_KEY:
        user_prompt = build_user_prompt(challenge, db_context, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(2):
            try:
                log(f"LLM attempt {attempt + 1}...")
                response = llm_chat(messages)
                candidate = extract_code_block(response)

                ok, err = validate_code(candidate)
                if ok:
                    code = candidate
                    log("LLM code validated OK")
                    break
                else:
                    log(f"Validation failed: {err}")
                    # Ask LLM to fix
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

    # Fallback if LLM failed
    if code is None:
        log("Using fallback architecture (no LLM or LLM failed)")
        code = fallback_architecture(challenge)
        name = f"fallback_patch_r{round_id}"
        motivation = "Fallback PatchTST-style architecture with RevIN"
    else:
        # Extract a name from the code (look for class definitions)
        try:
            tree = ast.parse(code)
            classes = [
                n.name for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef)
            ]
            name = classes[0] if classes else f"llm_arch_r{round_id}"
        except Exception:
            name = f"llm_arch_r{round_id}"
        motivation = (
            f"LLM-designed architecture for FLOPs [{min_flops:,}—{max_flops:,}]. "
            f"Model: {CHUTES_MODEL}"
        )

    # Record in history
    history.append({
        "round": round_id,
        "name": name,
        "min_flops": min_flops,
        "max_flops": max_flops,
        "used_llm": code is not None and CHUTES_API_KEY != "",
        "timestamp": time.time(),
    })
    save_history(scratch_dir, history)
    save_scratchpad(challenge, scratch_dir)

    return {"code": code, "name": name, "motivation": motivation}


def main():
    challenge = json.loads(sys.stdin.read())
    log("Agent starting (Chutes LLM backend)")
    proposal = design_architecture(challenge)
    log(f"Proposal: {proposal['name']} — {proposal['motivation']}")
    print(json.dumps(proposal))


if __name__ == "__main__":
    main()
