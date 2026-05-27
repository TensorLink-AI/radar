"""AST-based pre-flight validation — mirrors the validator's checks exactly."""

import ast

from core.flops_estimator import estimate_flops
from core.history import extract_flops_budget

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

    Returns (ok, list_of_errors).
    """
    errors: list[str] = []

    if not code or not code.strip():
        return False, ["Empty code — no source provided"]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"SyntaxError: {exc}"]

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

    # FLOPs bounds check — only if challenge provided and no structural errors
    if not errors and challenge is not None:
        flops_min, flops_max = extract_flops_budget(challenge)
        if flops_min or flops_max:
            gate_min = int(flops_min * 0.9)
            gate_max = int(flops_max * 1.1)
            estimated, err = estimate_flops(code, challenge)
            if err:
                errors.append(f"FLOPs estimation failed: {err}")
            elif estimated is not None:
                if estimated < gate_min:
                    errors.append(
                        f"Estimated FLOPs ({estimated:,}) below hard gate "
                        f"minimum ({gate_min:,}). Increase model capacity."
                    )
                elif estimated > gate_max:
                    errors.append(
                        f"Estimated FLOPs ({estimated:,}) above hard gate "
                        f"maximum ({gate_max:,}). Reduce model capacity."
                    )

    return len(errors) == 0, errors
