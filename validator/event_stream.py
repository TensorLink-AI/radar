"""Validator-side event capture — wandb-style log/metric streaming.

Two pieces:

* :class:`EventBuffer` — an in-memory, bounded ring of events with an
  async background flusher that POSTs batches to the central DB via
  :class:`shared.db_client.DatabaseClient`. Safe to call ``log_*`` from
  any thread.
* :class:`EventLoggingHandler` — a :class:`logging.Handler` that pipes
  every captured log record into the buffer as ``kind='log'``.

Wire it up once during validator init; everything else is automatic.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


# Loggers we never forward — they're either internal noise, or would
# create infinite loops if their own log lines were forwarded as events.
_BLOCKED_LOGGER_PREFIXES = (
    "shared.db_client",       # POSTing the batch logs failures → loop
    "validator.event_stream", # this module
    "httpx",
    "httpcore",
    "asyncio",
    "urllib3",
)


def _level_to_int(name: str) -> int:
    return {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }.get(name.lower(), logging.INFO)


class EventBuffer:
    """Bounded in-memory event buffer with a background flusher task."""

    def __init__(
        self,
        db_client,
        hotkey: str,
        flush_interval_s: float = 5.0,
        buffer_max: int = 2000,
        batch_max: int = 500,
    ):
        self._db_client = db_client
        self._hotkey = hotkey
        self._flush_interval = max(0.5, float(flush_interval_s))
        self._batch_max = max(1, int(batch_max))
        self._buffer: deque[dict] = deque(maxlen=max(self._batch_max, int(buffer_max)))
        self._lock = threading.Lock()
        self._round_id: int = -1
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # Counters for visibility — accessed from tests.
        self.dropped_full = 0
        self.flushed_total = 0
        self.failed_flushes = 0

    # ── Public producers (thread-safe) ─────────────────────────────

    def set_round(self, round_id: int) -> None:
        """Stamp subsequent events with this round id."""
        with self._lock:
            self._round_id = int(round_id)

    def log(self, level: str, message: str, **payload) -> None:
        """Record a log-line event."""
        self._enqueue({
            "kind": "log",
            "level": (level or "info").lower(),
            "payload": {"message": message, **payload},
        })

    def metric(self, name: str, value, **extra) -> None:
        """Record a scalar metric event."""
        self._enqueue({
            "kind": "metric",
            "payload": {"name": name, "value": value, **extra},
        })

    def phase(self, name: str, **payload) -> None:
        """Record a round/phase marker event (e.g. ``phase_a_start``)."""
        self._enqueue({
            "kind": "phase",
            "payload": {"name": name, **payload},
        })

    def _enqueue(self, ev: dict) -> None:
        ev.setdefault("ts", time.time())
        with self._lock:
            ev["round_id"] = self._round_id
            if len(self._buffer) >= self._buffer.maxlen:
                self.dropped_full += 1
            self._buffer.append(ev)

    # ── Flusher lifecycle ─────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background flush task. Must be called from an event loop."""
        if self._task is not None and not self._task.done():
            return
        self._stop = asyncio.Event()
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        """Signal the flusher to stop and drain pending events one last time."""
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._flush_once()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._flush_interval,
                )
            except asyncio.TimeoutError:
                pass
            try:
                await self._flush_once()
            except Exception as e:  # noqa: BLE001 — never crash the loop
                self.failed_flushes += 1
                # Use the module logger directly with stacklevel so the
                # "flush failed" line itself isn't captured as an event
                # (the handler skips this logger by name).
                logger.warning("Event flush failed: %s", e)

    async def _flush_once(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = []
            while self._buffer and len(batch) < self._batch_max:
                batch.append(self._buffer.popleft())
        if not batch:
            return
        ok = await self._db_client.post_events(batch)
        if ok:
            self.flushed_total += len(batch)
        else:
            self.failed_flushes += 1
            # Re-queue at the front so we retry — but only up to the
            # buffer cap so a permanently-down DB doesn't leak memory.
            with self._lock:
                room = self._buffer.maxlen - len(self._buffer)
                requeue = batch[:room]
                # appendleft preserves order
                for ev in reversed(requeue):
                    self._buffer.appendleft(ev)
                if room < len(batch):
                    self.dropped_full += len(batch) - room


class EventLoggingHandler(logging.Handler):
    """``logging.Handler`` that forwards records into an :class:`EventBuffer`.

    Filters out the buffer/flusher's own log lines to avoid feedback
    loops. Formats records lazily via ``record.getMessage()`` so format
    args are interpolated exactly once.
    """

    def __init__(self, buffer: EventBuffer, level: int = logging.INFO):
        super().__init__(level=level)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            name = record.name or ""
            for prefix in _BLOCKED_LOGGER_PREFIXES:
                if name == prefix or name.startswith(prefix + "."):
                    return
            message = record.getMessage()
            payload = {
                "logger": name,
                "module": record.module,
            }
            # Tracebacks are dropped — the public dashboard already has
            # the file/line in `message` after redaction. Keeping the
            # raw traceback here would pull paths into the row that
            # later need scrubbing.
            self._buffer.log(record.levelname, message, **payload)
        except Exception:
            # Never let logging take down the validator.
            self.handleError(record)


def attach_to_loggers(
    handler: logging.Handler, names: str = "", level: int = logging.INFO,
) -> list[logging.Logger]:
    """Attach ``handler`` to the named loggers (or root if ``names`` empty).

    Returns the list of loggers it was attached to so callers can detach
    on shutdown.
    """
    targets: list[logging.Logger] = []
    if not names.strip():
        targets.append(logging.getLogger())
    else:
        for n in (s.strip() for s in names.split(",")):
            if n:
                targets.append(logging.getLogger(n))
    for lg in targets:
        lg.addHandler(handler)
        # Make sure the logger's effective level lets our records through.
        if lg.level == logging.NOTSET or lg.level > level:
            lg.setLevel(level)
    return targets


def install_event_capture(
    db_client,
    hotkey: str,
    *,
    flush_interval_s: Optional[float] = None,
    buffer_max: Optional[int] = None,
    batch_max: Optional[int] = None,
    logger_names: Optional[str] = None,
    log_level: Optional[str] = None,
) -> Optional[tuple[EventBuffer, EventLoggingHandler, list[logging.Logger]]]:
    """Wire up the buffer + handler from validator init in one call.

    Returns ``(buffer, handler, attached_loggers)`` so the caller can
    cleanly tear down at shutdown. Returns ``None`` (and is a no-op) when
    ``Config.EVENTS_ENABLED`` is false.
    """
    if not Config.EVENTS_ENABLED:
        return None

    buffer = EventBuffer(
        db_client=db_client,
        hotkey=hotkey,
        flush_interval_s=flush_interval_s if flush_interval_s is not None
        else Config.EVENTS_FLUSH_INTERVAL_S,
        buffer_max=buffer_max if buffer_max is not None
        else Config.EVENTS_BUFFER_MAX,
        batch_max=batch_max if batch_max is not None
        else Config.EVENTS_BATCH_MAX,
    )
    level = _level_to_int(log_level if log_level is not None
                          else Config.EVENTS_LOG_LEVEL)
    handler = EventLoggingHandler(buffer, level=level)
    attached = attach_to_loggers(
        handler,
        names=logger_names if logger_names is not None
        else Config.EVENTS_LOGGER_NAMES,
        level=level,
    )
    return buffer, handler, attached
