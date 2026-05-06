"""Trainer image launcher — branches between FastAPI server and RunPod handler.

Bootstrap (``_bootstrap.py``) execs into this script after the
integrity check passes. We pick the runtime mode based on env vars
the platform injects:

* ``RP_HANDLER_NAME`` is set by RunPod's serverless runtime when the
  container is invoked as a worker. Its presence is the
  unambiguous signal that we should run as a job handler rather
  than an HTTP server.
* ``RADAR_TRAINER_MODE=runpod`` lets operators force handler mode for
  local testing (e.g. ``runpodctl test`` or a synthetic harness).

Default: FastAPI server (Basilica / Targon). One image, one digest,
two modes — keeps the bootstrap chain unchanged.
"""

from __future__ import annotations

import logging
import os


def _is_runpod_mode() -> bool:
    if os.environ.get("RP_HANDLER_NAME"):
        return True
    if os.environ.get("RADAR_TRAINER_MODE", "").lower() == "runpod":
        return True
    return False


def _start_runpod() -> None:
    # Imported here so the FastAPI path doesn't pay for the runpod
    # SDK at startup (and Basilica/Targon images that don't ship the
    # runpod package don't fail to import).
    from runner.handler import handle
    import runpod
    runpod.serverless.start({"handler": handle})


def _start_server() -> None:
    import uvicorn
    from runner.server import app
    port = int(os.environ.get("TRAINER_PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if _is_runpod_mode():
        _start_runpod()
    else:
        _start_server()


if __name__ == "__main__":
    main()
