"""Miner-side RunPod Serverless deploy / teardown.

The RunPod backend is structurally similar to Basilica's — the miner
ensures a long-lived endpoint exists, the validator dispatches into
it, and per-round artifacts land in R2. The notable differences:

* **No SSH/exec into the running container**. RunPod workers are
  ephemeral and miner-inaccessible; this is the whole reason for
  switching off Basilica. Mid-round attestation isn't useful here
  because the worker that picked up the job is sealed for its
  lifetime.
* **The validator can't submit to the miner's endpoint directly** —
  RunPod API keys are account-scoped. The miner listener relays
  /train into RunPod with the miner's API key (see
  ``miner/neuron.py``).
* **Endpoints persist across rounds** rather than being
  redeployed per round. ``deploy_runpod`` therefore mostly verifies
  the configured endpoint is healthy and returns a ``Deployment``
  pointing at the miner listener URL (not at RunPod's URL — the
  validator dispatches to the miner listener, which in turn submits
  to RunPod).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from miner.hosting import Deployment

logger = logging.getLogger(__name__)


class RunpodReadinessTimeout(Exception):
    """Endpoint exists but never reported a ready worker in the budget window."""


async def deploy_runpod(
    *,
    runpod_client,
    endpoint_id: str,
    listener_url: str,
    deployed_image_digest: str,
    gpu_class: str,
    request,
    readiness_timeout_s: float = 180.0,
    poll_interval_s: float = 5.0,
) -> Deployment:
    """Confirm the configured RunPod endpoint is live and return a handle.

    The miner pre-provisions the endpoint once; this call validates it
    on every round so a misconfigured endpoint surfaces as a fast
    DEPLOY_FAILED rather than a silent never-trains. The returned
    ``Deployment.url`` points at the miner *listener* — that's what the
    validator dispatches /train to.

    Raises ``RunpodReadinessTimeout`` if no worker is ever reported
    available in the readiness window. Caller tears down nothing on
    failure (RunPod manages the worker pool); the next round retries.
    """
    if not endpoint_id:
        raise ValueError("deploy_runpod: endpoint_id is required")
    if not listener_url:
        raise ValueError("deploy_runpod: listener_url is required")

    info = await _wait_for_endpoint_ready(
        runpod_client=runpod_client,
        endpoint_id=endpoint_id,
        timeout_s=readiness_timeout_s,
        poll_interval_s=poll_interval_s,
    )
    deploy_name = f"radar-runpod-{endpoint_id[-12:]}-{request.round_id}"
    logger.info(
        "RunPod endpoint %s ready: workers=%d/%d image=%s",
        endpoint_id, info.workers_running, info.workers_max, info.image_name or "?",
    )

    return Deployment(
        name=deploy_name,
        url=listener_url,
        gpu_class=gpu_class,
        deployed_image_digest=deployed_image_digest or info.image_digest,
        runpod_endpoint_id=endpoint_id,
        runpod_template_id=info.template_id,
        raw=info,
    )


async def _wait_for_endpoint_ready(
    *,
    runpod_client,
    endpoint_id: str,
    timeout_s: float,
    poll_interval_s: float,
):
    """Poll ``get_endpoint`` until ``workers_max`` is positive.

    A freshly created endpoint can report ``workers_max=0`` for a few
    seconds while RunPod allocates capacity. We accept any positive
    workers_max — workers_running can legitimately be zero if the
    endpoint is in scale-to-zero mode, and the first /train will warm
    one up.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    last_err = ""
    while asyncio.get_event_loop().time() < deadline:
        try:
            info = await runpod_client.get_endpoint(endpoint_id)
        except Exception as e:
            last_err = repr(e)
            await asyncio.sleep(poll_interval_s)
            continue
        if info.workers_max > 0:
            return info
        last_err = (
            f"endpoint {endpoint_id} reports workers_max=0 — capacity not yet allocated"
        )
        await asyncio.sleep(poll_interval_s)
    raise RunpodReadinessTimeout(
        f"RunPod endpoint {endpoint_id} not ready in {timeout_s:.0f}s "
        f"(last_err={last_err})"
    )


async def cancel_active_jobs(
    runpod_client,
    endpoint_id: str,
    job_ids: list[str],
    *,
    attempts: int = 3,
) -> bool:
    """Cancel the per-round jobs we know about; return True iff all succeeded.

    RunPod jobs cancel independently of the endpoint — the endpoint
    persists across rounds. Best-effort: a cancel race against a job
    that just completed is harmless (RunPod returns 4xx, which we
    swallow inside ``cancel_job``).
    """
    if not job_ids:
        return True
    ok = True
    for job_id in job_ids:
        for i in range(attempts):
            try:
                await runpod_client.cancel_job(endpoint_id, job_id)
                logger.info("RunPod cancel ok for %s (attempt %d)", job_id, i + 1)
                break
            except Exception as e:
                wait = 2 ** i
                logger.warning(
                    "RunPod cancel failed for %s (attempt %d/%d): %s — retrying in %ds",
                    job_id, i + 1, attempts, e, wait,
                )
                if i + 1 < attempts:
                    await asyncio.sleep(wait)
                else:
                    ok = False
    return ok


async def submit_dispatch_to_runpod(
    *,
    runpod_client,
    endpoint_id: str,
    payload: dict,
) -> Optional[str]:
    """Submit a dispatch payload to the RunPod endpoint.

    Returns the RunPod job_id on success, or None on RunPod outage
    (caller falls back to the unavailable-multiplier scoring path).
    """
    from shared.runpod_breaker import RunpodUnavailable
    try:
        handle = await runpod_client.submit_job(endpoint_id, payload)
    except RunpodUnavailable as e:
        logger.warning("RunPod unavailable on submit: %s", e)
        return None
    if not handle.job_id:
        logger.error("RunPod submit returned empty job_id (status=%s)", handle.status)
        return None
    logger.info(
        "RunPod job submitted: endpoint=%s job=%s status=%s",
        endpoint_id, handle.job_id, handle.status or "(none)",
    )
    return handle.job_id
