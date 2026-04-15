"""Ring buffer + logging handler for the Streamlit «Infolog» tab (session-global in-process)."""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

_MAX_LINES = 8000

_lines: deque[str] = deque(maxlen=_MAX_LINES)
_lock = threading.Lock()

VYVAR_LOGGERS = ("pipeline", "importer")

_handler: InfologHandler | None = None


def _handler_already_attached() -> bool:
    lg = logging.getLogger("pipeline")
    return any(isinstance(h, InfologHandler) for h in lg.handlers)


def log_event(message: str) -> None:
    """Append a user-facing or milestone line (shown in Infolog)."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"{ts}  {message}"
    with _lock:
        _lines.append(line)


def log_gaia_query(ra: float, dec: float, calculated_radius: float) -> None:
    """Structured Gaia query debug line."""
    log_event(f"GAIA QUERY: Center={ra},{dec} | Radius={calculated_radius:.2f} deg")


def log_exception(prefix: str, exc: BaseException) -> None:
    """Log exception message and full traceback into Infolog (for debugging worker/pool failures)."""
    import traceback

    log_event(f"{prefix}: {exc!s}")
    log_event(traceback.format_exc())


def get_lines() -> list[str]:
    with _lock:
        return list(_lines)


def clear_log() -> None:
    with _lock:
        _lines.clear()


def last_job_snapshot(obj: Any) -> None:
    """Pretty-print last job dict into Infolog (truncated if huge)."""
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(obj)
    if len(text) > 120_000:
        text = text[:120_000] + "\n… [truncated]"
    log_event("— Posledný výstup (JSON) —\n" + text)


class InfologHandler(logging.Handler):
    """Sends log records into the in-memory Infolog buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _lock:
                _lines.append(msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def ensure_infolog_logging() -> None:
    """Attach InfologHandler to VYVAR module loggers once (safe across Streamlit reruns)."""
    global _handler
    if _handler_already_attached():
        return

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  [%(name)s]  %(message)s", datefmt="%H:%M:%S")
    h = InfologHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(fmt)
    _handler = h

    for name in VYVAR_LOGGERS:
        lg = logging.getLogger(name)
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
