"""AST-based pre-flight validation — mirrors the validator's checks exactly.

Uses ``torch.utils.flop_counter.FlopCounterMode`` as the primary FLOPs
measurement (same as the validator), with actionable resize suggestions
when models land outside the budget.
"""

import ast

from core.flops_estimator import estimate_flops, suggest_resize
from core.history import extract_flops_budget
from core.output_shape import infer_output_shape, verify_output_shape

FORBIDDEN_IMPORTS = {"subprocess", "socket", "ftplib"}

def _required_functions(challenge: dict | None) -> dict[str, list[str]]:
    """Derive required function signatures from the challenge task_params.

    Reads build_model params from challenge['task']['task_params'] keys
    instead of hardcoding them — the CLAUDE.md spec requires this.
    """
    if challenge:
        task_params = challenge.get("task", {}).get("task_params", {})
        build_model_params = list(task_params.keys()) if task_params else []
    else:
        build_model_params = []
    return {
        "build_model": build_model_params,
        "build_optimizer": ["model"],
    }


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

    return len(errors) == 0, errors
