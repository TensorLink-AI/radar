"""Shared helpers for the openai_sdk agent (FLOPs, validation, history, etc.).

Copied verbatim from ``agents/autonomous/core/`` so this agent can be
deployed as a self-contained package — the miner's harness copies this
directory to ``/workspace/agent/`` and loads ``agent.py`` without any
package context, so there is no way to reach a sibling directory at
runtime.
"""


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
