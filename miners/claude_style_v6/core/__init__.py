"""Shared helpers for the openai_sdk agent (FLOPs, validation, history, etc.).

Copied verbatim from ``agents/autonomous/core/`` so this agent can be
deployed as a self-contained package — the miner's harness copies this
directory to ``/workspace/agent/`` and loads ``agent.py`` without any
package context, so there is no way to reach a sibling directory at
runtime.
"""


# ── Pipeline-task identity ─────────────────────────────────────────────
# Task names whose deliverable is a ``build_pipeline(...)`` data generator
# rather than a ``build_model`` / ``build_optimizer`` architecture. Both ride
# the same pipeline-designer flow, frozen-arch card, and build_pipeline
# contract; they differ only validator-side (``synthetic_data_generator``
# pins a FIXED architecture and scores GIFT-only). Centralised here so every
# module — orchestrator routing, tool gating, validation, and the fallback
# generator — agrees on the set. Keeping these out of sync is what let a
# valid ``synthetic_data_generator`` submission get rejected by the
# model-only validation path.
DATA_PIPELINE_TASK = "ts_data_pipeline"
SYNTHETIC_DATA_GENERATOR_TASK = "synthetic_data_generator"
PIPELINE_TASKS = frozenset({DATA_PIPELINE_TASK, SYNTHETIC_DATA_GENERATOR_TASK})


def is_pipeline_task(challenge) -> bool:
    """True when the challenge's task delivers a ``build_pipeline`` generator
    (``ts_data_pipeline`` or ``synthetic_data_generator``)."""
    if not challenge:
        return False
    name = (challenge.get("task") or {}).get("name") or ""
    return name in PIPELINE_TASKS


def call_with_timeout(fn, args=(), kwargs=None, timeout=15):
    """Call ``fn`` forwarding a ``timeout=`` kwarg to the blocking callee.

    ``GatedClient.post_json`` / ``get_json`` / ``get`` / ``put`` all accept
    a ``timeout`` kwarg that reaches ``urllib.urlopen``; a ThreadPoolExecutor
    wrapper can't actually interrupt a socket read, so passing the timeout
    through is the only way to bound wall-clock at the level we claim.
    """
    kwargs = dict(kwargs or {})
    kwargs.setdefault("timeout", timeout)
    return fn(*args, **kwargs)
