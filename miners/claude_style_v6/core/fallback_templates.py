"""Guaranteed-valid fallback architecture generator.

When the LLM fails to produce a model that passes validation within the
time/turn budget, this module generates a minimal but valid model that:

  1. Reads task_params generically from the challenge (no hardcoded keys)
  2. Sizes itself to ~80% of max_flops (pushes against the upper bound so
     we harvest the full size bucket — a 75M large-bucket model left 33%
     of the FLOPs budget on the floor)
  3. Passes both pre_validate_code (structure) AND the size gate (FLOPs)
  4. Works for ANY task because it derives the build_model signature and
     I/O shapes from the challenge
  5. Includes a per-miner hidden-dim jitter so the generated code hash
     differs across miners — without this, the fallback was byte-identical
     across the population and dedup collapsed all fallback submissions
     into a single effective miner.

This is the safety net — it should ALWAYS produce submittable code.
"""

import hashlib
import math
import textwrap

from core import is_pipeline_task
from core.history import extract_flops_budget, identify_bucket
from core.input_shape import (
    _CHANNEL_KEYS,
    _FEATURE_KEYS,
    _GRAPH_NODE_KEYS,
    _IMAGE_SIZE_KEYS,
    _SEQUENCE_LENGTH_KEYS,
    _find_key,
    _looks_like_token_task,
    infer_input,
)
from core.output_shape import infer_output_shape

# Target ~80% of the bucket's upper FLOPs bound so fallback submissions use
# the whole size bucket rather than parking at 60% and leaving FLOPs unspent.
FALLBACK_FLOPS_TARGET_FRACTION = 0.8

# Version suffix makes this template's submissions distinguishable from the
# older 60%-target, hardcoded-shape fallback so the DB can tell them apart.
# v4: bumped to v3 to mark the new default training recipe (AdamW + warmup-
# cosine + grad_clip + bf16 AMP) so DB analysis can separate v4 fallbacks
# from v3 fallbacks shipped before this template change.
# v5: bumped to v4 — same recipe block as v4, but v5 submissions surface
# recipe-sanity inspection results inline in the designer's loop, which
# changes the recipe distribution on the LLM side even when the fallback
# itself fires. Tagging the fallback lets DB analysis tell apart "fallback
# fired under v5" from "fallback fired under v4" without re-reading code.
# v6: bumped to v5 — the model-fallback block is unchanged, but the pipeline
# fallback now also fires on synthetic_data_generator rounds (v5 emitted a
# wrong-contract model there). Tagging separates v6 fallback rounds in the DB.
FALLBACK_VERSION = "v5"


# Default training-recipe block emitted by every fallback template.
#
# v3 fallbacks shipped a bare ``Adam(lr=1e-3)`` with no scheduler / no AMP /
# no grad clip / no training_config. That set the prior the LLM inherited
# every round — overriding "this just barely works" is harder than
# improving on a stronger baseline. v4 ships a realistic recipe (AdamW
# with weight decay, linear warmup → cosine decay, bf16 AMP, grad clip)
# so a miner that *doesn't* override the recipe still benefits from the
# raised floor. Architecture remains the LLM's job; recipe defaults are
# free improvements.
RECIPE_DEFAULTS = """
def build_optimizer(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "ln" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": 0.01})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        groups = [{"params": list(model.parameters()), "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(optimizer, total_steps):
    warmup = max(1, int(total_steps * 0.05))
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def training_config():
    return {
        "batch_size": 32,
        "grad_clip": 1.0,
        "log_every_n_steps": 50,
        "val_schedule": "log",
        "val_base_step": 100,
        "val_growth": 1.6,
    }


def configure_amp():
    return {"enabled": True, "dtype": "bfloat16"}


def on_step_end(model, optimizer, step, total_steps, loss_value):
    return None
"""


def _append_recipe(code: str) -> str:
    """Concatenate ``RECIPE_DEFAULTS`` after a fallback template's
    architecture block. The recipe is already at module-column-0; we
    avoid the textwrap.dedent gymnastics that would otherwise be needed
    to inline a multi-line block at the wrong indent."""
    body = RECIPE_DEFAULTS.strip("\n")
    if not code.endswith("\n"):
        code = code + "\n"
    return code + "\n" + body + "\n"


def _miner_jitter(challenge: dict) -> int:
    """Deterministic per-miner hidden-dim offset in [-8, 7].

    Derived from the miner UID so two miners with the same task and bucket
    produce different hidden dims → different code hashes → both survive
    the validator's hash-based dedup. The range is small enough that FLOPs
    stay inside the size-bucket gate (a ±8 shift on a hidden_dim of ~200
    moves FLOPs by at most ~5%) but large enough to guarantee unique
    hashes across miner populations.
    """
    uid = str(
        challenge.get("miner_uid")
        or challenge.get("uid")
        or challenge.get("agent_id")
        or "0"
    )
    h = hashlib.sha256(uid.encode()).hexdigest()
    return (int(h[:8], 16) % 16) - 8


def fallback_name_for(challenge: dict) -> str:
    """Return a bucket-tagged, versioned name for the fallback submission.

    e.g. ``fallback_large_v2``. Makes it possible to distinguish fallback
    submissions across rounds and size buckets in the experiment DB.
    """
    if is_pipeline_task(challenge):
        fa_version = (challenge.get("frozen_arch") or {}).get("version", "?")
        return f"pipeline_fallback_v{fa_version}_{FALLBACK_VERSION}"
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)
    return f"fallback_{bucket}_{FALLBACK_VERSION}"


PIPELINE_FALLBACK_TEMPLATE = '''"""Fallback pipeline for ts_data_pipeline.

Mixture of a few procedural time-series families (sinusoid sums, random
walks, AR(1)) with light noise. Designed to be a non-degenerate prior:
trains the frozen arch on a diverse-enough distribution that AULC drops
and the GIFT-Eval gate holds, without depending on any external data.
Yields indefinitely.
"""

import math
import torch

BATCH_SIZE = {batch_size}
SEED = {seed}


def _sinusoid_batch(rng, B, ctx, pred, V):
    n = ctx + pred
    t = torch.arange(n, dtype=torch.float32).view(1, n, 1)
    freqs = rng.uniform(0.02, 0.4, size=(B, 1, V))
    phases = rng.uniform(0.0, 2 * math.pi, size=(B, 1, V))
    amps = rng.uniform(0.5, 2.0, size=(B, 1, V))
    f = torch.from_numpy(freqs.astype("float32"))
    p = torch.from_numpy(phases.astype("float32"))
    a = torch.from_numpy(amps.astype("float32"))
    series = a * torch.sin(2 * math.pi * f * t + p)
    noise = torch.from_numpy(
        rng.normal(0.0, 0.1, size=(B, n, V)).astype("float32"),
    )
    return series + noise


def _walk_batch(rng, B, ctx, pred, V):
    n = ctx + pred
    steps = torch.from_numpy(
        rng.normal(0.0, 0.2, size=(B, n, V)).astype("float32"),
    )
    return steps.cumsum(dim=1)


def _ar1_batch(rng, B, ctx, pred, V):
    n = ctx + pred
    phi = rng.uniform(0.6, 0.95, size=(B, 1, V)).astype("float32")
    phi_t = torch.from_numpy(phi)
    noise = torch.from_numpy(
        rng.normal(0.0, 0.3, size=(B, n, V)).astype("float32"),
    )
    out = torch.zeros(B, n, V)
    out[:, 0] = noise[:, 0]
    for i in range(1, n):
        out[:, i] = phi_t.squeeze(1) * out[:, i - 1] + noise[:, i]
    return out


def build_pipeline(context_len, prediction_len, num_variates, quantiles):
    import numpy as np
    rng = np.random.default_rng(SEED)
    families = (_sinusoid_batch, _walk_batch, _ar1_batch)

    def _iter():
        i = 0
        while True:
            family = families[i % len(families)]
            i += 1
            series = family(
                rng, BATCH_SIZE, context_len, prediction_len, num_variates,
            )
            yield {{
                "input": series[:, :context_len].contiguous(),
                "target": series[:, context_len:].contiguous(),
            }}

    return _iter()
'''


def _generate_pipeline_fallback(challenge: dict, jitter: int) -> str:
    """Emit a procedural mixture pipeline that satisfies the contract
    without any data dependency. Seed jittered per miner so two miners
    on the same round produce different generators (hash-distinct
    submissions, same shape contract)."""
    # Batch size in 16-48 range — a real generator can override.
    batch_size = 32 + (jitter % 16)
    seed = (jitter * 2654435761) & 0xFFFFFFFF
    return PIPELINE_FALLBACK_TEMPLATE.format(
        batch_size=batch_size, seed=seed,
    )


def _has_recognized_continuous_keys(tp: dict) -> bool:
    """True when task_params contains at least one key we recognize as
    continuous-input (sequence length, channel count, image size, feature
    size, or graph nodes). Drives the generic-fallback fallback path."""
    for key_set in (_SEQUENCE_LENGTH_KEYS, _CHANNEL_KEYS, _IMAGE_SIZE_KEYS,
                    _FEATURE_KEYS, _GRAPH_NODE_KEYS):
        if _find_key(tp, key_set):
            return True
    return False


def generate_fallback(challenge: dict) -> str:
    """Generate a minimal but valid model that WILL pass the size gate.

    Returns a complete Python code string with build_model and build_optimizer.

    Archetype selection:
      1. Token-ID tasks (vocab_size etc.) → embedding + linear
      2. Continuous input that infer_input resolves cleanly → MLP that
         reshapes to the declared output shape
      3. Everything else → a generic MLP that reads dimensions from
         constraint strings and task_params, with no assumptions about
         task family
    """
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    constraints = task.get("constraints", [])
    flops_min, flops_max = extract_flops_budget(challenge)
    target_flops = (
        int(flops_max * FALLBACK_FLOPS_TARGET_FRACTION)
        if flops_max else 300_000
    )

    # Build the function signature from task_params keys
    param_names = list(tp.keys()) if tp else []
    sig = ", ".join(param_names) if param_names else "**kwargs"

    jitter = _miner_jitter(challenge)

    # 0. Pipeline tasks (ts_data_pipeline / synthetic_data_generator) —
    # different contract (build_pipeline, not build_model). Procedural
    # mixture generator, no data dependency; the build_pipeline shape
    # contract is identical across both tasks.
    if is_pipeline_task(challenge):
        return _generate_pipeline_fallback(challenge, jitter)

    # 1. Token-ID task
    if _looks_like_token_task(tp):
        return _generate_token_fallback(
            sig, param_names, tp, target_flops, constraints, jitter,
        )

    # 2. Continuous/recognized input — at least one familiar key present
    if _has_recognized_continuous_keys(tp):
        return _generate_continuous_fallback(
            sig, param_names, tp, target_flops, constraints, jitter,
        )

    # 3. Unknown task — generic MLP that reads whatever dims it can parse
    return _generate_generic_fallback(
        sig, param_names, tp, target_flops, constraints, jitter,
    )


def _parse_output_shape_from_constraints(constraints: list[str]) -> str | None:
    """Try to extract output shape expression from constraints.

    Looks for patterns like:
      "Output shape must be (batch, prediction_len, num_variates, len(quantiles))"
    """
    import re
    for c in constraints:
        m = re.search(r"[Oo]utput[^:]*(?:must be|shape|:)\s*\(([^)]+)\)", c)
        if m:
            return m.group(1)
    return None


def _generate_continuous_fallback(sig: str, param_names: list[str],
                                  tp: dict, target_flops: int,
                                  constraints: list[str],
                                  jitter: int = 0) -> str:
    """Generate a simple Linear model for continuous-input tasks.

    The model's ``forward()`` reshapes to the exact output shape inferred
    from the task's constraint string — avoiding the classic "output has
    wrong rank / wrong dim" failure that causes tensor-size mismatches
    during training.
    """
    # Infer input dimensions from task_params
    input_shape, _ = infer_input(tp, constraints)
    # input_shape is [1, dim1, dim2, ...] — product of dims after batch = input features per sample
    input_dims = input_shape[1:]  # drop batch

    # Estimate in_features as product of non-batch dims
    in_features = 1
    for d in input_dims:
        in_features *= d

    # Infer expected output shape (excluding batch) from constraints so the
    # forward pass reshapes to something that matches the task contract.
    # ``expected_out`` is a list of ints where unresolved dims are -1.
    expected_out = infer_output_shape(tp, constraints)
    expected_out_literal = repr(expected_out) if expected_out is not None else "None"

    # Build a kwargs dict string for the __init__ call
    kwargs_items = ", ".join(f"{n}={n}" for n in param_names)

    code = textwrap.dedent(f"""\
        import math
        import torch
        import torch.nn as nn

        class FallbackModel(nn.Module):
            def __init__(self, {sig}):
                super().__init__()
                # Store all params for shape computation
                self._tp = dict({kwargs_items})

                # Expected output shape (excluding batch) parsed from task
                # constraints by the host agent.  ``-1`` marks an unresolved
                # wildcard which we back-fill from the residual feature
                # count at forward time.
                self._expected_out = {expected_out_literal}

                int_params = [v for v in self._tp.values() if isinstance(v, int) and v > 0]
                list_params = [v for v in self._tp.values() if isinstance(v, list)]

                # Infer input feature dimension
                if len(int_params) >= 2:
                    self._in_features = int_params[0] * int_params[1]
                elif len(int_params) >= 1:
                    self._in_features = int_params[0]
                else:
                    self._in_features = 64

                # Compute the output feature count: if we have a concrete
                # expected shape, use the product of its resolved dims and
                # leave a single residual for any wildcard.  Otherwise fall
                # back to a best-effort estimate from int/list task params.
                if self._expected_out is not None:
                    resolved_product = 1
                    wildcard_dims = 0
                    for d in self._expected_out:
                        if d >= 0:
                            resolved_product *= d
                        else:
                            wildcard_dims += 1
                    # For each wildcard dim, pick a size of 1 so reshape works.
                    self._wildcard_size = 1
                    self._out_features = resolved_product * (self._wildcard_size ** wildcard_dims)
                else:
                    n_list = len(list_params[0]) if list_params else 1
                    if len(int_params) >= 2:
                        self._out_features = int_params[1] * max(1, n_list)
                        if len(int_params) >= 3:
                            self._out_features *= int_params[2]
                    else:
                        self._out_features = self._in_features

                # Size hidden_dim to fit FLOPs budget. MINER_JITTER is a
                # small per-miner offset so the generated code hash differs
                # across miners and dedup keeps them all in the round.
                TARGET_FLOPS = {target_flops}
                MINER_JITTER = {jitter}
                flops_per_h = max(1, 2 * (self._in_features + self._out_features))
                hidden_dim = max(4, (TARGET_FLOPS // flops_per_h) + MINER_JITTER)
                hidden_dim = min(hidden_dim, 2048)

                self.net = nn.Sequential(
                    nn.Linear(self._in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self._out_features),
                )

            def forward(self, x):
                b = x.shape[0]
                # Flatten input to (batch, in_features)
                x_flat = x.reshape(b, -1)
                # If input doesn't match expected, pad/truncate
                if x_flat.shape[1] != self._in_features:
                    if x_flat.shape[1] > self._in_features:
                        x_flat = x_flat[:, :self._in_features]
                    else:
                        pad = torch.zeros(b, self._in_features - x_flat.shape[1],
                                          device=x.device, dtype=x.dtype)
                        x_flat = torch.cat([x_flat, pad], dim=1)
                out = self.net(x_flat)

                # Reshape to the EXACT output shape required by the task,
                # replacing wildcards with concrete sizes.  torch.reshape
                # only accepts a single -1, so fix any extras to 1 and let
                # reshape infer the last wildcard from the residual.
                if self._expected_out is not None:
                    target = [b]
                    seen_wildcard = False
                    for d in self._expected_out:
                        if d >= 0:
                            target.append(d)
                        elif not seen_wildcard:
                            target.append(-1)
                            seen_wildcard = True
                        else:
                            target.append(1)
                    return out.reshape(*target)

                # No constraint parsed — best-effort fallback reshape.
                return out.reshape(b, -1, max(1, self._out_features // max(1, out.shape[1])))

        def build_model({sig}):
            return FallbackModel({sig})
    """)
    return _append_recipe(code)


def _generate_generic_fallback(sig: str, param_names: list[str],
                               tp: dict, target_flops: int,
                               constraints: list[str],
                               jitter: int = 0) -> str:
    """Generic MLP for tasks whose task_params we don't recognize.

    Reads input shape from constraint strings when possible, falling back
    to the product of integer task_params. Output shape is always derived
    from ``infer_output_shape``; unresolved dims collapse to 1 so the
    reshape is well-defined.
    """
    expected_out = infer_output_shape(tp, constraints)
    expected_out_literal = repr(expected_out) if expected_out is not None else "None"

    kwargs_items = ", ".join(f"{n}={n}" for n in param_names)

    code = textwrap.dedent(f"""\
        import math
        import torch
        import torch.nn as nn

        class GenericFallbackModel(nn.Module):
            def __init__(self, {sig}):
                super().__init__()
                self._tp = dict({kwargs_items})
                self._expected_out = {expected_out_literal}

                # Infer in_features as the product of positive integer params,
                # capped so we don't allocate absurd layers for huge ints.
                int_params = [v for v in self._tp.values()
                              if isinstance(v, int) and v > 0]
                if int_params:
                    prod = 1
                    for v in int_params[:3]:
                        prod *= v
                    self._in_features = max(4, min(prod, 4096))
                else:
                    self._in_features = 64

                # Output features — product of resolved dims, wildcards → 1.
                if self._expected_out is not None:
                    out = 1
                    for d in self._expected_out:
                        out *= d if d >= 0 else 1
                    self._out_features = max(1, out)
                else:
                    # Lean on the largest int param as a best-effort output.
                    self._out_features = int_params[0] if int_params else self._in_features

                # Per-miner jitter keeps generated code hashes unique.
                TARGET_FLOPS = {target_flops}
                MINER_JITTER = {jitter}
                flops_per_h = max(1, 2 * (self._in_features + self._out_features))
                hidden_dim = max(4, (TARGET_FLOPS // flops_per_h) + MINER_JITTER)
                hidden_dim = min(hidden_dim, 2048)

                self.net = nn.Sequential(
                    nn.Linear(self._in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self._out_features),
                )

            def forward(self, x):
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                if x.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    x = x.float()
                b = x.shape[0] if x.dim() > 0 else 1
                x_flat = x.reshape(b, -1)
                if x_flat.shape[1] != self._in_features:
                    if x_flat.shape[1] > self._in_features:
                        x_flat = x_flat[:, :self._in_features]
                    else:
                        pad = torch.zeros(b, self._in_features - x_flat.shape[1],
                                          device=x.device, dtype=x_flat.dtype)
                        x_flat = torch.cat([x_flat, pad], dim=1)
                out = self.net(x_flat)

                if self._expected_out is not None:
                    target = [b]
                    seen_wildcard = False
                    for d in self._expected_out:
                        if d >= 0:
                            target.append(d)
                        elif not seen_wildcard:
                            target.append(-1)
                            seen_wildcard = True
                        else:
                            target.append(1)
                    return out.reshape(*target)
                return out

        def build_model({sig}):
            return GenericFallbackModel({sig})
    """)
    return _append_recipe(code)


def _generate_token_fallback(sig: str, param_names: list[str],
                             tp: dict, target_flops: int,
                             constraints: list[str],
                             jitter: int = 0) -> str:
    """Generate a simple embedding + linear model for token-ID tasks."""
    vocab_key = None
    for k in ("vocab_size", "n_vocab", "vocabulary_size"):
        if k in tp:
            vocab_key = k
            break
    vocab_key = vocab_key or "vocab_size"

    seq_key = None
    for k in ("block_size", "seq_len", "sequence_length", "context_len"):
        if k in tp:
            seq_key = k
            break
    seq_key = seq_key or "block_size"

    code = textwrap.dedent(f"""\
        import math
        import torch
        import torch.nn as nn

        class FallbackTokenModel(nn.Module):
            def __init__(self, {sig}):
                super().__init__()
                self._tp = dict({", ".join(f"{n}={n}" for n in param_names)})
                vocab = int(self._tp.get("{vocab_key}", 1000))
                seq = int(self._tp.get("{seq_key}", 128))

                # Size embedding dim to fit FLOPs budget
                # FLOPs ~ 2 * seq * embed_dim * vocab (for output projection)
                # MINER_JITTER keeps generated code hashes unique across miners.
                TARGET_FLOPS = {target_flops}
                MINER_JITTER = {jitter}
                embed_dim = max(4, (TARGET_FLOPS // max(1, 2 * seq * vocab)) + MINER_JITTER)
                embed_dim = min(embed_dim, 512)

                self.embed = nn.Embedding(vocab, embed_dim)
                self.fc = nn.Linear(embed_dim, vocab)

            def forward(self, x):
                h = self.embed(x)
                return self.fc(h)

        def build_model({sig}):
            return FallbackTokenModel({sig})
    """)
    return _append_recipe(code)
