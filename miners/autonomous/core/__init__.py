# Autonomous Agent Core Modules


def call_with_timeout(fn, args=(), kwargs=None, timeout=15):
    """Call fn with a timeout passed through to the underlying blocking call.

    Previously used ThreadPoolExecutor, but `with ThreadPoolExecutor` calls
    `shutdown(wait=True)` on exit, which blocks until the worker thread
    completes even when `future.result(timeout=...)` raises TimeoutError.
    Python threads cannot be killed, so the 'timeout' was a lie — the real
    wall-clock was whatever the OS socket timeout was (~127s).

    Instead, pass timeout through to the callee. GatedClient.post_json,
    get, etc. all accept a `timeout=` kwarg that reaches urllib.urlopen.
    """
    kwargs = dict(kwargs or {})
    kwargs.setdefault("timeout", timeout)
    return fn(*args, **kwargs)
