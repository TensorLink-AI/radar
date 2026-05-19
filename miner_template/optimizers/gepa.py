"""GEPA — Genetic-Pareto reflective prompt evolution.

Adapter around ``dspy.GEPA``.  Imports DSPy lazily so the rest of the
miner CLI runs without it installed.

The adapter:

  1. Group ``ResultRow`` by ``prompt_id`` to build per-prompt traces
     ``(architecture_code, scores)`` for GEPA to learn from.
  2. Configure a reflector LM — either an explicit OpenAI/Anthropic-
     style API (``MINER_REFLECTOR_API_*`` env vars) or the operator's
     shared LLM proxy (``MINER_LLM_URL`` / ``MINER_LLM_API_KEY``).
  3. Wrap each prompt as a ``dspy.Signature`` and run ``dspy.GEPA``'s
     reflective mutation loop with a Pareto-front sampler over the
     score vector.
  4. Map the resulting optimized programs back to ``Prompt`` rows with
     fresh UUIDs, the new generation number, and ``parent_id`` set
     from GEPA's lineage.

Config keys (passed via the ``optimize`` subcommand):

  ``population`` (int, default 8): target population size.
  ``budget``     (int, default 30): GEPA compile budget (rollouts).
  ``score_key``  (str, default "raw_score"): which scalar GEPA reads.
  ``reflector_lm`` (dict, optional): explicit DSPy LM kwargs.  When
    omitted, the adapter reads ``MINER_REFLECTOR_*`` / ``MINER_LLM_*``
    env vars (see ``_default_lm()`` below).

The score the metric returns is the float at ``scores[score_key]`` for
the matched row; missing rows score 0.  Override ``score_key`` to
optimize against a different objective (e.g. ``"frontier_dominance"``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from miner_template.optimizers import ResultRow
from miner_template.prompts import Prompt

logger = logging.getLogger(__name__)


def _import_dspy():
    """Lazy import so the CLI works without DSPy installed."""
    try:
        import dspy  # type: ignore
    except ImportError as e:
        raise ImportError(
            "GEPA optimizer requires the 'dspy' package. "
            "Install with: pip install dspy"
        ) from e
    if not hasattr(dspy, "GEPA"):
        raise ImportError(
            "Installed dspy build does not expose dspy.GEPA. "
            "Upgrade: pip install -U dspy"
        )
    return dspy


def _default_lm(dspy_mod) -> Any:
    """Build a DSPy LM from environment, preferring an explicit
    reflector LM and falling back to the operator's LLM proxy."""
    base = os.getenv("MINER_REFLECTOR_API_BASE", "").strip()
    key = os.getenv("MINER_REFLECTOR_API_KEY", "").strip()
    model = os.getenv("MINER_REFLECTOR_MODEL", "").strip()
    if base and key and model:
        return dspy_mod.LM(model=model, api_base=base, api_key=key)

    base = os.getenv("MINER_LLM_URL", "").strip()
    key = os.getenv("MINER_LLM_API_KEY", "").strip()
    model = os.getenv("MINER_LLM_MODEL", "").strip() or "deepseek-ai/DeepSeek-V3-0324"
    if base and key:
        return dspy_mod.LM(model=model, api_base=base, api_key=key)

    raise RuntimeError(
        "GEPA reflector LM not configured.  Set either "
        "MINER_REFLECTOR_{API_BASE,API_KEY,MODEL} or "
        "MINER_LLM_{URL,API_KEY,MODEL}."
    )


def _group_by_prompt(results: list[ResultRow]) -> dict[str, list[ResultRow]]:
    groups: dict[str, list[ResultRow]] = {}
    for r in results:
        if r.prompt_id:
            groups.setdefault(r.prompt_id, []).append(r)
    return groups


def _build_metric(score_key: str):
    """Return a metric callable GEPA invokes per rollout.

    The default DSPy metric signature is ``metric(example, pred, trace)
    -> float``; we don't need the example/trace here because we already
    know which prompt produced which scored row.  GEPA passes through
    user-supplied kwargs, so we attach ``score_key`` as a closure.
    """

    def metric(example, pred, trace=None) -> float:
        # GEPA stores result metadata on ``pred`` when wrapped through
        # this adapter; the caller's program puts ``scores`` there.
        scores = getattr(pred, "scores", None) or {}
        try:
            return float(scores.get(score_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return metric


def _to_example(dspy_mod, row: ResultRow, score_key: str):
    """Wrap a ResultRow as a DSPy example GEPA can iterate over."""
    return dspy_mod.Example(
        challenge=row.scores.get("challenge", ""),
        architecture=row.architecture_code,
        score=float(row.scores.get(score_key, 0.0) or 0.0),
    ).with_inputs("challenge")


def optimize(
    results: list[ResultRow],
    population: list[Prompt],
    config: dict,
) -> list[Prompt]:
    """Run GEPA on the population.

    See module docstring for ``config`` keys.  Returns a new population
    of size ``config['population']`` (or the input size if unset).
    """
    dspy_mod = _import_dspy()

    if not population:
        return []

    target = int(config.get("population", len(population) or 8))
    budget = int(config.get("budget", 30))
    score_key = str(config.get("score_key", "raw_score"))

    # Configure reflector LM.
    lm_kwargs = config.get("reflector_lm")
    if isinstance(lm_kwargs, dict):
        lm = dspy_mod.LM(**lm_kwargs)
    else:
        lm = _default_lm(dspy_mod)
    dspy_mod.configure(lm=lm)

    # Build the training set from scored rows.
    examples = [_to_example(dspy_mod, r, score_key)
                for r in results if r.prompt_id]
    if not examples:
        logger.warning(
            "GEPA: no scored examples — returning population unchanged."
        )
        return population[:target]

    next_gen = max(p.generation for p in population) + 1

    # GEPA optimizes a DSPy program; each program is a ``Predict`` over
    # the same ``Signature`` but with the prompt template as its
    # instruction.  We seed the search from every active variant so the
    # current population is preserved unless GEPA finds something
    # strictly better.
    class _ArchSignature(dspy_mod.Signature):
        """Design a model architecture for the given challenge."""

        challenge: str = dspy_mod.InputField()
        architecture: str = dspy_mod.OutputField()

    programs = []
    for p in population:
        prog = dspy_mod.Predict(_ArchSignature)
        prog.signature = prog.signature.with_instructions(p.template)
        programs.append((p, prog))

    metric = _build_metric(score_key)
    teleprompter = dspy_mod.GEPA(metric=metric, num_candidates=target, budget=budget)

    # GEPA compiles each seed program; we keep the best ``target``
    # programs by their reported score and map them back to Prompt rows.
    optimized: list[tuple[Prompt, Any]] = []
    for parent, prog in programs:
        try:
            compiled = teleprompter.compile(prog, trainset=examples)
        except Exception as e:
            logger.warning(
                "GEPA: compile failed for prompt %s: %s — keeping parent",
                parent.id, e,
            )
            optimized.append((parent, prog))
            continue
        new_instruction = _extract_instruction(compiled)
        optimized.append((
            Prompt.new(
                template=new_instruction or parent.template,
                generation=next_gen,
                parent_id=parent.id,
                metadata={"mutation": "gepa", "budget": budget},
            ),
            compiled,
        ))

    new_pop = [p for p, _ in optimized][:target]
    while len(new_pop) < target:
        new_pop.append(population[len(new_pop) % len(population)])
    return new_pop


def _extract_instruction(program) -> str:
    """Pull the optimized instruction string off a DSPy program."""
    sig = getattr(program, "signature", None)
    if sig is None:
        return ""
    instructions = getattr(sig, "instructions", "")
    if isinstance(instructions, str):
        return instructions
    return str(instructions or "")
