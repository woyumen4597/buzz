"""Phase logging with a heartbeat for long-running highlight steps.

Media analysis and rendering can take minutes on long videos. Without any
output the GUI and the log file look identical to a hang, so every phase logs
when it starts, periodically while it runs, and when it finishes.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator

DEFAULT_HEARTBEAT_SECONDS = 15.0


@contextmanager
def phase(
    label: str,
    logger: logging.Logger,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    level: int = logging.INFO,
) -> Iterator[None]:
    """Log the start, periodic heartbeats and completion of a work phase."""
    started = time.monotonic()
    logger.log(level, "%s: started", label)
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(heartbeat_seconds):
            logger.log(level, "%s: still running (%.0fs elapsed)", label, time.monotonic() - started)

    beat = threading.Thread(target=_beat, name="highlight-heartbeat", daemon=True)
    beat.start()
    try:
        yield
    finally:
        stop.set()
        logger.log(level, "%s: finished in %.1fs", label, time.monotonic() - started)
