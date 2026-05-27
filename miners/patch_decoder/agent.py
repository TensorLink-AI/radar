"""Patch Decoder — deterministic agent that emits a RevIN patch-MLP decoder
scaled dynamically to fit the requested FLOPs budget.

No LLM calls.  The architecture is fully parameterised by a scaling function
that derives dimensions from the challenge, so it adapts to ANY budget and
ANY harness parameters automatically.

Architecture:
  1. RevIN normalisation (learnable affine)
  2. Non-overlapping patch embedding (Linear: patch_size -> d_model)
  3. Learnable positional embedding
  4. N stacked MLP blocks (Linear -> LayerNorm -> GELU -> Linear residual)
  5. Flatten patches -> projection head -> (pred_len, num_variates, n_quantiles)
  6. RevIN denormalisation (applied per-quantile to preserve last-dim = num_variates)
"""

import json
import os
import sys
import tempfile
import textwrap

from core import validation, history


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_active_prompt(round_id: int) -> dict:
    """Return ``{id, template}`` for the prompt variant this round.

    patch_decoder is deterministic — there's no LLM to steer with the
    template — but we still surface ``prompt_id`` on the return dict so
    Phase C scores attribute to the active variant when the miner runs
    ``miner/neuron.py optimize`` against this agent. Empty when the
    optimizer hasn't been run.
    """
    prompts_dir = os.getenv("MINER_PROMPTS_DIR", "prompts")
    path = os.path.join(prompts_dir, "active.json")
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"id": "", "template": ""}
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        return {"id": "", "template": ""}
    pick = rows[round_id % len(rows)]
    if not isinstance(pick, dict):
        return {"id": "", "template": ""}
    return {
        "id": str(pick.get("id", "")),
        "template": str(pick.get("template", "")),
    }


# ── Dynamic scaling ─────────────────────────────────────────────


def _analytical_flops(V, n_patches, patch_size, d_model, ff_mult,
                      n_layers, prediction_len, nq):
    """Analytical FLOPs estimate for the patch-decoder architecture.

    Matches what the forward-pass hook estimator would measure at batch=1
    with V folded into the batch dimension.
    """
    ff_dim = d_model * ff_mult
    # Patch embedding: Linear(patch_size, d_model) on (V, n_patches, patch_size)
    patch_embed = V * n_patches * 2 * patch_size * d_model
    # MLP blocks: fc1 + fc2 per block
    mlp_total = n_layers * V * n_patches * 2 * 2 * d_model * ff_dim
    # Output norm
    norms = (n_layers + 1) * 2 * d_model * V * n_patches
    # Head: Linear(d_model * n_patches, prediction_len * nq) on (V, flat)
    head = V * 2 * (d_model * n_patches) * (prediction_len * nq)
    return patch_embed + mlp_total + norms + head


def _compute_scaling(challenge: dict) -> dict | None:
    """Derive (d_model, n_layers, patch_size) from the challenge budget.

    Grid-searches combinations and picks the one closest to 60% of
    flops_max.  Falls back to a minimal direct-linear config if no
    patch-based config fits.

    Returns None if the task is not ts_forecasting (i.e. required
    task_params keys are missing), signalling that this agent cannot
    handle the task.
    """
    task = challenge.get("task", {})
    tp = task.get("task_params", {})

    # This agent only supports ts_forecasting.  Bail out early when the
    # required task_params keys are absent.
    required_keys = {"context_len", "prediction_len", "num_variates", "quantiles"}
    if not required_keys.issubset(tp.keys()):
        missing = required_keys - tp.keys()
        _log(f"[patch_decoder] Unsupported task — missing task_params keys: {missing}")
        return None

    context_len = tp["context_len"]
    prediction_len = tp["prediction_len"]
    num_variates = tp["num_variates"]
    quantiles = tp["quantiles"]

    flops_min, flops_max = history.extract_flops_budget(challenge)
    target = int(flops_max * 0.6)

    V = num_variates
    nq = len(quantiles)
    ff_mult = 2

    best = None
    best_diff = float("inf")

    for patch_size in [64, 32, 16, 8, 4]:
        if patch_size > context_len:
            continue
        n_patches = context_len // patch_size
        if n_patches < 1:
            continue

        for n_layers in [1, 2, 3, 4, 6]:
            # Scan d_model in steps of 4
            for d_model in range(4, 513, 4):
                total = _analytical_flops(
                    V, n_patches, patch_size, d_model, ff_mult,
                    n_layers, prediction_len, nq,
                )
                diff = abs(total - target)
                if diff < best_diff:
                    best_diff = diff
                    best = {
                        "d_model": d_model,
                        "n_layers": n_layers,
                        "ff_mult": ff_mult,
                        "patch_size": patch_size,
                    }
                if total > target * 1.5:
                    break  # d_model too large, stop scanning

    if best is None or best["d_model"] < 4:
        # Fallback: minimal direct-linear config
        best = {
            "d_model": 4,
            "n_layers": 1,
            "ff_mult": 1,
            "patch_size": min(context_len, 16),
        }

    # Training hyperparameters scaled by model size
    d = best["d_model"]
    if d <= 32:
        best.update(lr="5e-4", batch_size=64, grad_accum=1)
    elif d <= 64:
        best.update(lr="3e-4", batch_size=64, grad_accum=1)
    elif d <= 128:
        best.update(lr="3e-4", batch_size=32, grad_accum=2)
    elif d <= 256:
        best.update(lr="2e-4", batch_size=32, grad_accum=2)
    else:
        best.update(lr="1e-4", batch_size=16, grad_accum=4)

    return best


def _generate_code(cfg: dict, task_params: dict | None = None) -> str:
    """Render a complete Python module string from scaling config."""
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]
    ff = d * cfg["ff_mult"]
    ps = cfg["patch_size"]
    lr = cfg["lr"]
    bs = cfg["batch_size"]
    ga = cfg["grad_accum"]

    param_str = ", ".join(task_params.keys()) if task_params else "**task_params"

    return textwrap.dedent(f"""\
        import torch
        import torch.nn as nn
        import math


        class RevIN(nn.Module):
            \"\"\"Reversible Instance Normalization.\"\"\"
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
                    x = (x - self.affine_bias) / (self.affine_weight + self.eps)
                    x = x * self._std + self._mean
                    return x


        class MLPBlock(nn.Module):
            \"\"\"Pre-norm MLP block with residual connection.\"\"\"
            def __init__(self, d_model, ff_dim, dropout=0.1):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.fc1 = nn.Linear(d_model, ff_dim)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(ff_dim, d_model)
                self.drop = nn.Dropout(dropout)

            def forward(self, x):
                residual = x
                x = self.norm(x)
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = self.drop(x)
                return x + residual


        class PatchDecoder(nn.Module):
            \"\"\"Patch-based MLP decoder with RevIN.\"\"\"
            def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
                super().__init__()
                self.prediction_len = prediction_len
                self.num_variates = num_variates
                self.n_quantiles = n_quantiles

                self.revin = RevIN(num_variates)

                patch_size = {ps}
                d_model = {d}
                self.patch_size = patch_size
                n_patches = context_len // patch_size
                self.n_patches = n_patches

                # Patch embedding
                self.patch_embed = nn.Linear(patch_size, d_model)
                self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

                # MLP blocks
                self.blocks = nn.Sequential(
                    *[MLPBlock(d_model, {ff}, dropout=0.1) for _ in range({n_layers})]
                )
                self.out_norm = nn.LayerNorm(d_model)

                # Projection head
                self.head = nn.Linear(d_model * n_patches, prediction_len * n_quantiles)

            def forward(self, x):
                # x: (batch, context_len, num_variates)
                x = self.revin(x, "norm")
                b, L, V = x.shape

                # Channel-independent: process each variate separately
                x = x.permute(0, 2, 1).reshape(b * V, L)
                x = x.reshape(b * V, self.n_patches, self.patch_size)
                x = self.patch_embed(x) + self.pos_embed
                x = self.blocks(x)
                x = self.out_norm(x)
                x = x.reshape(b * V, -1)
                x = self.head(x)

                # Reshape: (b*V, pred*nq) -> (b, V, pred, nq) -> (b, pred, V, nq)
                x = x.reshape(b, V, self.prediction_len, self.n_quantiles)
                x = x.permute(0, 2, 1, 3)  # (b, pred, V, nq)

                # RevIN denorm per-quantile slice (last dim must be num_variates)
                denormed = []
                for q_idx in range(self.n_quantiles):
                    slice_q = x[:, :, :, q_idx]  # (b, pred, V)
                    denormed.append(self.revin(slice_q, "denorm"))
                return torch.stack(denormed, dim=3)  # (b, pred, V, nq)


        def build_model({param_str}):
            return PatchDecoder(context_len, prediction_len, num_variates, len(quantiles))


        def build_optimizer(model):
            return torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay=0.01)


        def training_config():
            return {{"batch_size": {bs}, "grad_accum_steps": {ga}, "grad_clip": 1.0, "eval_interval": 200}}


        def build_scheduler(optimizer, total_steps):
            warmup = total_steps // 10
            def lr_lambda(step):
                if step < warmup:
                    return step / max(warmup, 1)
                progress = (step - warmup) / max(total_steps - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


        def init_weights(model):
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    """)


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness.  Deterministic — no LLM calls."""

    # ── Active prompt variant (attribution-only, no template use) ─
    round_id = int(challenge.get("round_id", 0) or 0)
    active_prompt = _load_active_prompt(round_id)

    # ── Identify bucket ──────────────────────────────────────────
    flops_min, flops_max = history.extract_flops_budget(challenge)
    bucket = history.identify_bucket(flops_min, flops_max)
    target_flops = int(flops_max * 0.6)

    _log(f"[patch_decoder] Bucket: {bucket}, FLOPs range: {flops_min:,}-{flops_max:,}, "
         f"target: {target_flops:,}")

    # ── Compute scaling dynamically from budget ──────────────────
    cfg = _compute_scaling(challenge)

    if cfg is None:
        task_name = challenge.get("task", {}).get("name", "<unknown>")
        _log(f"[patch_decoder] Skipping submission — patch_decoder only supports "
             f"ts_forecasting tasks (got {task_name!r})")
        return {
            "code": "",
            "name": "patch_decoder_skip",
            "motivation":
                f"patch_decoder only supports ts_forecasting tasks, "
                f"not {task_name!r}",
            "prompt_id": active_prompt["id"],
        }

    _log(f"[patch_decoder] Dynamic config: d_model={cfg['d_model']}, "
         f"layers={cfg['n_layers']}, patch_size={cfg['patch_size']}")

    # ── Generate code ────────────────────────────────────────────
    tp = challenge.get("task", {}).get("task_params", {})
    code = _generate_code(cfg, tp)

    ok, errors = validation.validate_code(code, challenge)
    if not ok:
        _log(f"[patch_decoder] Validation failed: {errors}")
        # Try adjusting d_model +-4 a few times
        for delta in [-4, 4, -8, 8, -12, 12]:
            adj = dict(cfg)
            adj["d_model"] = max(4, cfg["d_model"] + delta)
            code = _generate_code(adj, tp)
            ok, errors = validation.validate_code(code, challenge)
            if ok:
                cfg = adj
                _log(f"[patch_decoder] Adjusted d_model to {cfg['d_model']}, now valid")
                break
        else:
            _log(f"[patch_decoder] BUG: generated code failed validation: {errors}")

    name = f"patch_decoder_{bucket}"
    motivation = (
        f"Deterministic RevIN patch-MLP decoder for {bucket} bucket "
        f"(d_model={cfg['d_model']}, layers={cfg['n_layers']}, "
        f"patch_size={cfg['patch_size']}, target ~{target_flops:,} FLOPs)"
    )

    # ── Update scratchpad ────────────────────────────────────────
    scratch_dir = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821 — injected global
    except Exception:
        pass

    state = history.load_state(scratch_dir) if scratch_dir else {}
    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=target_flops, strategy="deterministic",
    )
    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    try:
        save_scratchpad(challenge, scratch_dir)  # noqa: F821 — injected global
    except Exception:
        pass

    return {
        "code": code,
        "name": name,
        "motivation": motivation,
        "prompt_id": active_prompt["id"],
    }
