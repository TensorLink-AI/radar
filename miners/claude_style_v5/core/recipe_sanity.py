"""Lightweight static inspection of generated training-recipe code.

Complements ``core.validation``'s runtime smoke test: that path catches
crashes, this one catches *plausible-but-poor* recipes that run fine but
produce bad models. We can't actually train to find out, so we look for
high-confidence anti-patterns the validator wouldn't reject:

  * lr outside [1e-5, 5e-2] — almost always a typo or bad prior
  * no weight decay AND optimizer is AdamW (AdamW with wd=0 is just Adam)
  * no grad clipping AND optimizer is Adam/AdamW on >1M params (instability)
  * scheduler missing and lr ≥ 5e-4 (won't decay — late-training instability)
  * COMPILE=True without ``configure_amp`` opting into bfloat16 (slow path)
  * batch_size ≥ 256 on a long-context task (memory blow-up risk)

The inspector returns a list of ``Warning`` records. Callers decide what
to do with them — the designer subagent surfaces them inline as a critic-
style message so the LLM can react, the orchestrator logs them, and the
submit handler attaches them to the candidate's motivation.

Design constraints:
  * No torch import — runs in <1ms regardless of whether torch is present.
  * AST-only — we walk the parsed module, not the executed namespace,
    so we can inspect partially-broken code (the runtime smoke test
    bails out on first exception).
  * Conservative — false positives cost the LLM a confused round, so
    every check is gated on a known-bad signature rather than a fuzzy
    heuristic. ``ok=True`` is the strong claim, not ``ok=False``.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Hyperparameter sanity bounds. These are deliberately loose — we only
# flag values that would be *clearly* wrong on a research codebase, not
# values that are merely unusual.
LR_MIN = 1e-5
LR_MAX = 5e-2
# Below this many parameters, missing grad clip is fine — small models
# don't blow up. Above it, the lack of clip is a stability hazard.
GRAD_CLIP_PARAM_THRESHOLD = 1_000_000
# A non-trivial lr that has no scheduler attached is suspicious — the
# model will plateau or oscillate late in training.
SCHED_REQUIRED_LR = 5e-4
# Long-context tasks (any task_params key containing "len" / "ctx") with
# big batches are likely to OOM on the small validator GPU. Used as a
# soft signal, not a hard reject.
LARGE_BATCH_THRESHOLD = 256


@dataclass(frozen=True)
class Warning:
    """One recipe-sanity finding.

    ``severity`` is one of ``info`` / ``warn`` / ``critical``. Critical
    warnings should be surfaced loudly to the LLM (or auto-fixed); warn
    is advisory; info is for telemetry only.
    """
    code: str            # short stable id, e.g. "lr_out_of_band"
    severity: str        # info | warn | critical
    message: str         # human-readable, ≤ 120 chars
    suggestion: str = ""  # short fix hint the LLM can act on


@dataclass
class Report:
    warnings: list[Warning] = field(default_factory=list)
    inspected: bool = True  # False when AST parsing failed

    @property
    def ok(self) -> bool:
        return all(w.severity != "critical" for w in self.warnings)

    def format_for_prompt(self) -> str:
        """Render the report as a short structured block for an LLM.

        Returns an empty string when there's nothing to say so the
        caller can skip the section entirely.
        """
        if not self.warnings:
            return ""
        lines = ["Recipe sanity inspection:"]
        for w in self.warnings:
            tag = {"critical": "!!!", "warn": "!!", "info": "."}[w.severity]
            line = f"{tag} [{w.code}] {w.message}"
            if w.suggestion:
                line += f" → {w.suggestion}"
            lines.append(line)
        return "\n".join(lines)


# ── AST helpers ───────────────────────────────────────────────────────


def _find_function(tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def _optimizer_calls(fn: ast.FunctionDef) -> list[ast.Call]:
    """Collect every ``torch.optim.Xxx(...)`` style call inside fn body."""
    found: list[ast.Call] = []
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        target = sub.func
        # ``Adam(...)`` — direct name reference
        if isinstance(target, ast.Name):
            if target.id in _OPTIMIZER_NAMES:
                found.append(sub)
            continue
        # ``torch.optim.Adam(...)`` — attribute chain
        if isinstance(target, ast.Attribute):
            if target.attr in _OPTIMIZER_NAMES:
                found.append(sub)
    return found


def _kwarg(call: ast.Call, name: str) -> Optional[ast.expr]:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _literal(node: Optional[ast.expr]):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def _optimizer_name(call: ast.Call) -> str:
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


_OPTIMIZER_NAMES = frozenset({
    "SGD", "Adam", "AdamW", "Adamax", "Adagrad", "RMSprop",
    "Adadelta", "NAdam", "RAdam", "Lion", "Lamb",
})


# ── Checks ────────────────────────────────────────────────────────────


def _check_optimizer(tree: ast.Module) -> list[Warning]:
    """Inspect ``build_optimizer`` for lr / weight_decay / family signals."""
    warnings: list[Warning] = []
    fn = _find_function(tree, "build_optimizer")
    if fn is None:
        return warnings
    calls = _optimizer_calls(fn)
    if not calls:
        warnings.append(Warning(
            code="no_optimizer_call",
            severity="warn",
            message="build_optimizer body has no recognizable optimizer call",
            suggestion="instantiate torch.optim.AdamW(...) or similar",
        ))
        return warnings

    for call in calls:
        opt_name = _optimizer_name(call)
        lr = _literal(_kwarg(call, "lr"))
        if lr is None and len(call.args) >= 2:
            # ``Adam(params, 1e-3)`` positional form
            lr = _literal(call.args[1])
        if isinstance(lr, (int, float)):
            f = float(lr)
            if f <= 0 or f != f:  # zero or NaN
                warnings.append(Warning(
                    code="lr_non_positive",
                    severity="critical",
                    message=f"{opt_name} lr={lr!r} is non-positive",
                    suggestion="set lr in [1e-5, 5e-2]; AdamW default 3e-4",
                ))
            elif f < LR_MIN:
                warnings.append(Warning(
                    code="lr_too_small",
                    severity="warn",
                    message=f"{opt_name} lr={lr!r} below {LR_MIN:.0e}",
                    suggestion="model will under-train — try 1e-4 or 3e-4",
                ))
            elif f > LR_MAX:
                warnings.append(Warning(
                    code="lr_too_large",
                    severity="critical",
                    message=f"{opt_name} lr={lr!r} above {LR_MAX:.0e}",
                    suggestion="will diverge — try AdamW(lr=3e-4) instead",
                ))

        if opt_name == "AdamW":
            wd = _literal(_kwarg(call, "weight_decay"))
            if isinstance(wd, (int, float)) and float(wd) == 0.0:
                warnings.append(Warning(
                    code="adamw_no_weight_decay",
                    severity="warn",
                    message="AdamW with weight_decay=0 is just Adam — frontier baseline",
                    suggestion="set weight_decay≈0.01 on non-bias/norm tensors",
                ))

    return warnings


def _check_grad_clip(tree: ast.Module, code: str) -> list[Warning]:
    """``training_config`` should set a grad_clip for any Adam-family run.

    We look at the literal returned by ``training_config`` if present.
    Missing config is a 'warn'; explicit ``None`` / 0 is a 'warn'.
    """
    fn = _find_function(tree, "training_config")
    if fn is None:
        # No training_config means harness defaults apply. The harness
        # default is not aggressive enough to clip — flag it as info.
        return [Warning(
            code="no_training_config",
            severity="info",
            message="no training_config() — harness default has no grad_clip",
            suggestion="add training_config returning {'grad_clip': 1.0, ...}",
        )]
    # Walk the function body for a Return whose value is a dict literal
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        # Pull grad_clip from the dict if present
        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(k, ast.Constant) and k.value == "grad_clip":
                val = _literal(v)
                if val is None:
                    return [Warning(
                        code="grad_clip_null",
                        severity="warn",
                        message="training_config grad_clip is None — Adam-family runs are unstable without it",
                        suggestion="set grad_clip≈1.0",
                    )]
                if isinstance(val, (int, float)) and float(val) == 0.0:
                    return [Warning(
                        code="grad_clip_zero",
                        severity="warn",
                        message="training_config grad_clip=0 disables clipping",
                        suggestion="set grad_clip≈1.0",
                    )]
                return []
        # dict returned, no grad_clip key
        return [Warning(
            code="grad_clip_missing",
            severity="warn",
            message="training_config has no grad_clip key — Adam-family runs are unstable without it",
            suggestion="add 'grad_clip': 1.0",
        )]
    return []


def _check_scheduler(tree: ast.Module) -> list[Warning]:
    """If lr is non-trivial and there's no ``build_scheduler``, warn."""
    if _find_function(tree, "build_scheduler") is not None:
        return []
    fn = _find_function(tree, "build_optimizer")
    if fn is None:
        return []
    max_lr = 0.0
    for call in _optimizer_calls(fn):
        lr = _literal(_kwarg(call, "lr"))
        if lr is None and len(call.args) >= 2:
            lr = _literal(call.args[1])
        if isinstance(lr, (int, float)):
            max_lr = max(max_lr, float(lr))
    if max_lr >= SCHED_REQUIRED_LR:
        return [Warning(
            code="no_scheduler",
            severity="warn",
            message=f"lr={max_lr:.0e} without build_scheduler — constant LR is the frontier baseline",
            suggestion="add LambdaLR with linear warmup + cosine decay",
        )]
    return []


def _check_compile_amp(tree: ast.Module, code: str) -> list[Warning]:
    """``COMPILE=True`` without bf16 AMP leaves perf on the floor."""
    has_compile = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "COMPILE":
                    val = _literal(node.value)
                    has_compile = bool(val)
    if not has_compile:
        return []
    fn = _find_function(tree, "configure_amp")
    if fn is None:
        return [Warning(
            code="compile_no_amp",
            severity="info",
            message="COMPILE=True without configure_amp — slow path on bf16-capable GPUs",
            suggestion="add configure_amp returning {'enabled': True, 'dtype': 'bfloat16'}",
        )]
    return []


def _check_batch_size(tree: ast.Module, challenge: dict) -> list[Warning]:
    """Big batches on long-context tasks bite at training time."""
    if not challenge:
        return []
    tp = (challenge.get("task") or {}).get("task_params") or {}
    if not any("len" in str(k).lower() or "ctx" in str(k).lower() for k in tp):
        return []
    fn = _find_function(tree, "training_config")
    if fn is None:
        return []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(k, ast.Constant) and k.value == "batch_size":
                val = _literal(v)
                if isinstance(val, int) and val >= LARGE_BATCH_THRESHOLD:
                    return [Warning(
                        code="batch_too_large",
                        severity="warn",
                        message=f"batch_size={val} on a long-context task — likely OOM on the validator GPU",
                        suggestion="drop batch_size to ≤128 and use grad_accum_steps for effective size",
                    )]
    return []


# ── Entry point ───────────────────────────────────────────────────────


def inspect_recipe(code: str, challenge: Optional[dict] = None) -> Report:
    """Run every static check against ``code`` and return a Report.

    Never raises — parsing failures degrade to an empty report with
    ``inspected=False`` so the caller can distinguish "no warnings" from
    "couldn't inspect." The runtime smoke test in ``core.validation``
    will already have reported parse errors as a hard failure.
    """
    if not code or not code.strip():
        return Report(inspected=False)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Report(inspected=False)

    warnings: list[Warning] = []
    try:
        warnings.extend(_check_optimizer(tree))
        warnings.extend(_check_grad_clip(tree, code))
        warnings.extend(_check_scheduler(tree))
        warnings.extend(_check_compile_amp(tree, code))
        warnings.extend(_check_batch_size(tree, challenge or {}))
    except Exception as exc:
        logger.warning("recipe_sanity check raised: %s", exc)
        return Report(inspected=False)
    return Report(warnings=warnings)
