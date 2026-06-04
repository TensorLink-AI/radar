"""AST-based pre-flight validation — mirrors the validator's checks exactly.

Uses ``torch.utils.flop_counter.FlopCounterMode`` as the primary FLOPs
measurement (same as the validator), with actionable resize suggestions
when models land outside the budget.
"""

import ast
import math
import os
import tempfile

from core.flops_estimator import estimate_flops, suggest_resize
from core.history import extract_flops_budget
from core.input_shape import infer_input
from core.output_shape import infer_output_shape, verify_output_shape

FORBIDDEN_IMPORTS = {"subprocess", "socket", "ftplib"}

DATA_PIPELINE_TASK = "ts_data_pipeline"

# Pre-flight stepping points — chosen so warmup (5% of total = step 5)
# is exercised in the first sample and the post-warmup cosine branch is
# exercised in the rest. The harness's actual ``scheduler.step()`` calls
# every optim step, so any pattern that produces a tensor lr after warmup
# (the classic ``torch.cos(torch.tensor(x))`` gotcha) is caught here
# before training starts.
_RECIPE_TOTAL_STEPS = 100
_RECIPE_STEP_PROBES = (1, 6, 30, 60, 99)


def _task_name(challenge: dict | None) -> str:
    if not challenge:
        return ""
    return (challenge.get("task") or {}).get("name") or ""


def _required_functions(challenge: dict | None) -> dict[str, list[str]]:
    """Derive required function signatures from the challenge task_params.

    Reads required-function signatures from challenge['task']['task_params']
    keys instead of hardcoding them — the CLAUDE.md spec requires this.

    For ``ts_data_pipeline`` the contract is a ``build_pipeline(...)``
    iterator factory; there is no ``build_optimizer`` because the harness
    uses the frozen architecture's hooks.
    """
    task_params: dict = {}
    if challenge:
        task_params = challenge.get("task", {}).get("task_params", {}) or {}
    param_names = list(task_params.keys())
    if _task_name(challenge) == DATA_PIPELINE_TASK:
        return {"build_pipeline": param_names}
    return {
        "build_model": param_names,
        "build_optimizer": ["model"],
    }


def _build_dummy_input(tp: dict, constraints: list, torch_mod):
    """Construct a [B=1, ...] dummy matching the harness's contract.

    Returns ``(tensor, None)`` on success or ``(None, error_str)`` if the
    shape can't be inferred. Mirrors ``flops_estimator``'s logic so this
    smoke test sees the same tensor the FLOPs path does.
    """
    try:
        input_shape, input_dtype = infer_input(tp, constraints)
    except Exception as exc:
        return None, f"could not infer input shape: {exc}"
    try:
        if input_dtype == torch_mod.long:
            vocab = tp.get("vocab_size", tp.get("n_vocab", 1000))
            return torch_mod.randint(0, int(vocab), input_shape), None
        return torch_mod.randn(*input_shape), None
    except Exception as exc:
        return None, f"could not build dummy input: {exc}"


def _scalar_loss(namespace: dict, predictions, torch_mod):
    """Reduce an arbitrary model output to a scalar we can backward through.

    Prefers the submission's ``compute_loss`` (with a same-shape target) so
    bf16-unsupported ops inside the loss surface here; falls back to
    ``predictions.mean()`` when the submission doesn't define one or the
    custom loss rejects synthetic targets.
    """
    compute_loss = namespace.get("compute_loss")
    if callable(compute_loss) and isinstance(predictions, torch_mod.Tensor):
        try:
            targets = torch_mod.zeros_like(predictions)
            loss = compute_loss(predictions, targets)
            if isinstance(loss, torch_mod.Tensor) and loss.ndim == 0:
                return loss
        except Exception:
            pass
    if isinstance(predictions, torch_mod.Tensor):
        return predictions.float().mean()
    # Tuple/dict outputs — sum every tensor leaf we can find.
    leaves = []

    def _walk(obj):
        if isinstance(obj, torch_mod.Tensor):
            leaves.append(obj.float().mean())
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                _walk(o)
        elif isinstance(obj, dict):
            for o in obj.values():
                _walk(o)
    _walk(predictions)
    if not leaves:
        return None
    return sum(leaves)


def _check_training_recipe(code: str, challenge: dict) -> list[str]:
    """Full runtime smoke test before submitting.

    Catches the four largest classes of mid-training failure that the
    pure-FLOPs / AST validation can't see:
      - forward pass that crashes only in ``model.train()`` mode (the
        classic ``.view()`` after a non-contiguous transpose),
      - bf16-unsupported ops triggered only inside ``torch.amp.autocast``
        (FFT, ``view_as_complex``, custom kernels that hard-require fp32),
      - state_dict that gets ``_orig_mod.`` prefixes when ``COMPILE = True``
        is set but the Phase C reload uses a bare model,
      - scheduler/lambda bugs that turn ``param_group["lr"]`` into a tensor
        or a non-finite value.

    Steps share an exec'd namespace so the cost is one model build, one
    optimizer build, one optional compile + state_dict roundtrip, plus the
    existing scheduler probe.
    """
    # Defer heavy imports so callers without torch (the AST-only checks
    # above) still work.
    try:
        import torch
    except ImportError:
        return []

    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
    except Exception:
        # FLOPs estimator already reported this — don't duplicate.
        return []

    build_model_fn = namespace.get("build_model")
    build_optimizer_fn = namespace.get("build_optimizer")
    if not callable(build_model_fn) or not callable(build_optimizer_fn):
        return []

    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    constraints = task.get("constraints", []) or []
    try:
        model = build_model_fn(**tp)
    except Exception:
        return []

    try:
        optimizer = build_optimizer_fn(model)
    except Exception as exc:
        return [f"build_optimizer() raised: {exc}"]

    # ── Forward + backward smoke test (train mode) ────────────────
    dummy, dummy_err = _build_dummy_input(tp, constraints, torch)
    if dummy is None:
        # Couldn't synthesize a probe input — skip runtime checks and let
        # the existing FLOPs path's diagnostics carry. Don't fail.
        return _scheduler_probe(namespace, model, optimizer, torch)

    model.train()
    try:
        predictions = model(dummy)
    except Exception as exc:
        return [
            f"model.forward() raised in train mode: "
            f"{type(exc).__name__}: {exc}. The training harness runs the "
            "forward pass in train mode on shape "
            f"{tuple(dummy.shape)} — fix this before submitting."
        ]

    loss = _scalar_loss(namespace, predictions, torch)
    if loss is None:
        return _scheduler_probe(namespace, model, optimizer, torch)

    try:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except Exception as exc:
        return [
            f"backward()/optimizer.step() raised: "
            f"{type(exc).__name__}: {exc}. This is the same path the "
            "training harness takes — the run would crash on the first "
            "real batch."
        ]

    # ── AMP smoke test (when configure_amp() opts in) ─────────────
    configure_amp = namespace.get("configure_amp")
    amp_enabled = False
    amp_dtype_str = "bfloat16"
    if callable(configure_amp):
        try:
            cfg = configure_amp() or {}
            amp_enabled = bool(cfg.get("enabled", False))
            amp_dtype_str = str(cfg.get("dtype", "bfloat16"))
        except Exception as exc:
            return [f"configure_amp() raised: {exc}"]
    if amp_enabled:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        amp_dtype = dtype_map.get(amp_dtype_str, torch.bfloat16)
        # autocast on "cpu" rejects the same op set as cuda for bf16/fp16
        # — view_as_complex, fft.*, and most custom kernels — which is
        # exactly what we need to catch here.
        try:
            with torch.amp.autocast("cpu", dtype=amp_dtype, enabled=True):
                predictions = model(dummy)
                loss = _scalar_loss(namespace, predictions, torch)
            if loss is not None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        except Exception as exc:
            return [
                f"forward/backward under AMP ({amp_dtype_str}) raised: "
                f"{type(exc).__name__}: {exc}. configure_amp() opts into "
                f"{amp_dtype_str} but an op rejects it (commonly "
                "view_as_complex, FFT, or a custom kernel). Either drop "
                "AMP, switch dtype to float32, or wrap the offending "
                "block in torch.amp.autocast(..., enabled=False)."
            ]

    # ── COMPILE state_dict round-trip ─────────────────────────────
    if bool(namespace.get("COMPILE", False)):
        try:
            fresh = build_model_fn(**tp)
            compiled = torch.compile(fresh)
            state = compiled.state_dict()
            with tempfile.TemporaryDirectory() as td:
                from safetensors.torch import save_file, load_file
                path = os.path.join(td, "probe.safetensors")
                save_file(state, path)
                reload_target = build_model_fn(**tp)
                reload_target.load_state_dict(load_file(path), strict=True)
        except Exception as exc:
            return [
                f"COMPILE=True checkpoint round-trip failed: "
                f"{type(exc).__name__}: {exc}. torch.compile wraps the "
                "model so saved keys gain a `_orig_mod.` prefix that the "
                "Phase C eval loader (which builds a bare model) can't "
                "match. Either set COMPILE=False, or save "
                "`model._orig_mod.state_dict()` explicitly."
            ]

    # ── Existing scheduler probe ──────────────────────────────────
    return _scheduler_probe(namespace, model, optimizer, torch)


def _scheduler_probe(namespace: dict, model, optimizer, torch) -> list[str]:
    """LR-schedule probe extracted from the legacy ``_check_training_recipe``.

    Walks the user's scheduler across warmup + post-warmup steps and
    asserts ``param_group['lr']`` stays a finite Python float at every
    probe. Catches the ``torch.cos(torch.tensor(x))`` -> tensor-LR gotcha.
    """
    build_scheduler_fn = namespace.get("build_scheduler")
    if not callable(build_scheduler_fn):
        return []

    try:
        scheduler = build_scheduler_fn(optimizer, _RECIPE_TOTAL_STEPS)
    except Exception as exc:
        return [f"build_scheduler() raised: {exc}"]
    if scheduler is None:
        return []

    import warnings
    last_step = 0
    for probe in _RECIPE_STEP_PROBES:
        for _ in range(probe - last_step):
            try:
                with warnings.catch_warnings():
                    # We're stepping the scheduler without a paired
                    # optimizer.step() — that's intentional for the probe
                    # but PyTorch warns about it.
                    warnings.simplefilter("ignore", category=UserWarning)
                    scheduler.step()
            except Exception as exc:
                return [
                    f"scheduler.step() raised at step {probe}: "
                    f"{type(exc).__name__}: {exc}. Make sure lambdas "
                    "return Python floats — use `math.cos(math.pi * x)`, "
                    "NOT `torch.cos(torch.tensor(x))`."
                ]
        last_step = probe
        for pg in optimizer.param_groups:
            lr = pg.get("lr")
            if torch.is_tensor(lr):
                return [
                    f"scheduler made param_group['lr'] a tensor at step "
                    f"{probe} (lr={lr!r}). Lambdas must return Python "
                    "floats — use `math.cos(math.pi * x)`, NOT "
                    "`torch.cos(torch.tensor(x))`. The training loop "
                    "calls optimizer.step() with the resulting lr and "
                    "would crash mid-run."
                ]
            if not isinstance(lr, (int, float)):
                return [
                    f"param_group['lr'] is {type(lr).__name__} at step "
                    f"{probe} (expected int or float)."
                ]
            if not math.isfinite(float(lr)):
                return [
                    f"param_group['lr'] is {lr!r} at step {probe} "
                    "(non-finite)."
                ]
    return []


def validate_code(code: str, challenge: dict | None = None) -> tuple[bool, list[str]]:
    """Validate generated code against the validator's requirements.

    Returns (ok, list_of_errors).  Checks:
      1. Non-empty, non-whitespace code
      2. ast.parse() succeeds (no syntax errors)
      3. Top-level ``def build_model`` exists
      4. Top-level ``def build_optimizer`` exists
      5. build_model has params derived from challenge task_params
      6. build_optimizer has param: model
      7. No forbidden imports (subprocess, socket, ftplib)
      8. FLOPs within budget bounds (if challenge provided)
         — includes actionable resize suggestions on failure
      9. Output shape matches the expected shape parsed from the task
         ``constraints`` (when such a constraint is present). Handles any
         tensor rank — the comparison is driven by the constraint string.
     10. Training-recipe pre-flight — instantiates optimizer + scheduler
         and steps the scheduler across warmup + post-warmup, asserting
         ``param_group['lr']`` stays a finite Python float. Catches the
         classic ``torch.cos(torch.tensor(x))`` -> tensor-LR gotcha at
         submit time so the round isn't wasted on a guaranteed crash.
    """
    errors: list[str] = []

    # 1. Reject empty / whitespace-only code
    if not code or not code.strip():
        return False, ["Empty code — no source provided"]

    # 2. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"SyntaxError: {exc}"]

    # 3-6. Check for required top-level functions — params from challenge task_params
    top_level_funcs: dict[str, list[str]] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in node.args.args if a.arg != "self"]
            top_level_funcs[node.name] = params

    required = _required_functions(challenge)
    for fname, required_params in required.items():
        if fname not in top_level_funcs:
            errors.append(f"Missing required top-level function: {fname}")
        else:
            actual = top_level_funcs[fname]
            for rp in required_params:
                if rp not in actual:
                    errors.append(f"{fname} missing parameter: {rp}")

    # 7. Forbidden imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    errors.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    errors.append(f"Forbidden import: {node.module}")

    # ts_data_pipeline ships a data generator, not a model. FLOPs/output-
    # shape checks would try to forward-pass something that isn't there.
    # Stop after structural + import checks.
    if _task_name(challenge) == DATA_PIPELINE_TASK:
        return len(errors) == 0, errors

    # 8. FLOPs bounds + 9. output shape check — only if no structural errors.
    #    Both share a single forward pass: the estimator captures the
    #    output tensor shape via ``out_shape_sink`` so we don't re-run the
    #    model to verify coherence.
    if not errors and challenge is not None:
        flops_min, flops_max = extract_flops_budget(challenge)
        task = challenge.get("task", {}) or {}
        task_params = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        out_shape_sink: list = []

        run_estimator = bool(flops_min or flops_max) or bool(
            infer_output_shape(task_params, constraints)
        )

        if run_estimator:
            estimated, err = estimate_flops(code, challenge, out_shape_sink)
            if err:
                errors.append(f"FLOPs estimation failed: {err}")
            elif estimated is not None and (flops_min or flops_max):
                gate_min = int(flops_min * 0.9)
                gate_max = int(flops_max * 1.1)
                target = int(flops_max * 0.6)
                if estimated < gate_min:
                    hint = suggest_resize(estimated, gate_min, gate_max, target)
                    errors.append(
                        f"Estimated FLOPs ({estimated:,}) below hard gate "
                        f"minimum ({gate_min:,}). Increase model capacity."
                        + (f"\n{hint}" if hint else "")
                    )
                elif estimated > gate_max:
                    hint = suggest_resize(estimated, gate_min, gate_max, target)
                    errors.append(
                        f"Estimated FLOPs ({estimated:,}) above hard gate "
                        f"maximum ({gate_max:,}). Reduce model capacity."
                        + (f"\n{hint}" if hint else "")
                    )

            # 9. Output shape coherence — must be a HARD gate whenever the
            # task declares a parseable output-shape constraint. The classic
            # "tensor a (96) vs tensor b (64)" training failure happens when
            # build_model() returns a module whose forward() projects to the
            # wrong dim; it would train for an hour before crashing. Better
            # to reject at validate time.
            expected = infer_output_shape(task_params, constraints)
            if expected is not None:
                if not out_shape_sink:
                    # FLOPs path bailed out (hook-based / static-walk fallback,
                    # or we skipped FLOPs entirely because no budget was set).
                    # Run a dedicated forward pass just to capture the shape.
                    _, shape_only_err = estimate_flops(
                        code, challenge, out_shape_sink,
                    )
                    if shape_only_err and not out_shape_sink:
                        # Couldn't even instantiate/run the model — surface it
                        # as a validation failure so the agent retries rather
                        # than submitting something that won't run.
                        errors.append(
                            f"Output shape check failed: {shape_only_err}"
                        )
                if out_shape_sink:
                    shape_err = verify_output_shape(out_shape_sink[0], expected)
                    if shape_err:
                        errors.append(shape_err)

    # 10. Training-recipe pre-flight. Only when structural + shape checks
    # pass — recipe errors are confusing when the model itself doesn't
    # instantiate. Skipped for ts_data_pipeline above.
    if not errors and challenge is not None:
        errors.extend(_check_training_recipe(code, challenge))

    return len(errors) == 0, errors
