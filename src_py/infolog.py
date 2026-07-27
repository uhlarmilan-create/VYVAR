"""Ring buffer + logging handler for the Streamlit <<Infolog>> tab (session-global in-process)."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
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
    try:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"{ts}  {message}"
        with _lock:
            _lines.append(line)
    except Exception as exc:  # noqa: BLE001 - Infolog must never crash the pipeline
        logging.getLogger(__name__).debug("log_event append failed: %s", exc)


def log_milestone(message: str) -> None:
    """Infolog buffer plus pipeline logger (headless night-run and UI)."""
    log_event(message)
    logging.getLogger("pipeline").info(message)


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


def save_infolog_to_disk(draft_dir: str | Path, entries: list[str] | None = None) -> str | None:
    """
    Save current Infolog entries to ``draft_dir/infolog_YYYYMMDD_HHMMSS.txt``.

    Returns path if successful, ``None`` on failure (best-effort; never raises).
    """
    _log = logging.getLogger(__name__)
    try:
        root = Path(str(draft_dir)).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / f"infolog_{ts}.txt"
        lines = list(entries) if entries is not None else get_lines()
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# VYVAR Infolog - {ts}\n")
            f.write(f"# Draft: {root}\n")
            f.write("# timestamps: UTC\n")
            f.write("#" + "=" * 60 + "\n\n")
            for entry in lines:
                if isinstance(entry, dict):
                    ts_e = entry.get("time", "")
                    lvl = entry.get("level", "INFO")
                    msg = entry.get("message", str(entry))
                    f.write(f"[{ts_e}] [{lvl}] {msg}\n")
                else:
                    f.write(f"{entry}\n")
        return str(path)
    except Exception as exc:  # noqa: BLE001
        _log.warning("[INFOLOG] Save to disk failed: %s", exc)
        return None


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
        text = text[:120_000] + "\n... [truncated]"
    log_event("- Posledny vystup (JSON) -\n" + text)


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
        for name in VYVAR_LOGGERS:
            logging.getLogger(name).propagate = False
        return

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  [%(name)s]  %(message)s", datefmt="%H:%M:%S")
    fmt.converter = time.gmtime
    h = InfologHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(fmt)
    _handler = h

    for name in VYVAR_LOGGERS:
        lg = logging.getLogger(name)
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        # Avoid propagating to the root logger: hosted Streamlit / runners sometimes
        # attach a StreamHandler to stdout that is later closed; LOGGER.info would then
        # raise ValueError("I/O operation on closed file.").
        lg.propagate = False
