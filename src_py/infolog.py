"""Draft infolog: one append-only ``infolog_<ts>.txt`` per run in the draft directory.

INFOLOG-ONE-2026-10: the run process (night_run) owns the file; there is no UI tab and no
second snapshot file (Milan: the log is read outside VYVAR)."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_session_log_path: Path | None = None
_session_log_file: Any = None
# milestones logged before the run's draft directory exists; written when the file opens
_milestones: list[str] = []
# SPEED-2026-10: a worker process has no infolog file; its messages are collected here and the
# run process writes them (``capture_events`` / ``replay_events``)
_captured: list[str] | None = None

VYVAR_LOGGERS = ("pipeline", "importer")

_handler: InfologHandler | None = None


def _handler_already_attached() -> bool:
    lg = logging.getLogger("pipeline")
    return any(isinstance(h, InfologHandler) for h in lg.handlers)


def log_event(message: str) -> None:
    """Append one line to the run infolog (no-op outside a run; collected inside ``capture_events``)."""
    try:
        if _captured is not None:
            _captured.append(str(message))
            return
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"{ts}  {message}"
        with _lock:
            _append_session_line(line)
    except Exception as exc:  # noqa: BLE001 - Infolog must never crash the pipeline
        logging.getLogger(__name__).debug("log_event append failed: %s", exc)


class capture_events:
    """Collect ``log_event`` messages (in a worker process) instead of writing them."""

    def __enter__(self) -> list[str]:
        global _captured
        self._prev = _captured
        _captured = []
        return _captured

    def __exit__(self, *_exc: Any) -> None:
        global _captured
        _captured = self._prev


def replay_events(messages: list[str] | None) -> None:
    """Write messages a worker collected with ``capture_events``."""
    for m in messages or []:
        log_event(m)


def log_milestone(message: str) -> None:
    """Milestone line; kept until the run infolog opens if logged before it."""
    try:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"{ts}  {message}"
        with _lock:
            if _session_log_file is None:
                _milestones.append(line)
            _append_session_line(line)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("log_milestone append failed: %s", exc)


def _append_session_line(line: str) -> None:
    if _session_log_file is None:
        return
    try:
        _session_log_file.write(f"{line}\n")
        _session_log_file.flush()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("session infolog append failed: %s", exc)


def start_infolog_session(draft_dir: str | Path) -> str | None:
    """Open the run infolog ``draft_dir/infolog_<ts>.txt`` (append, flushed per line)."""
    global _session_log_path, _session_log_file
    try:
        root = Path(str(draft_dir)).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / f"infolog_{ts}.txt"
        fh = path.open("a", encoding="utf-8")
        fh.write(f"# VYVAR Infolog - {ts}\n")
        fh.write(f"# Draft: {root}\n")
        fh.write("# timestamps: UTC\n")
        fh.write("#" + "=" * 60 + "\n\n")
        with _lock:
            for m in _milestones:
                fh.write(f"{m}\n")
            _milestones.clear()
        fh.flush()
        _session_log_path = path
        _session_log_file = fh
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("[INFOLOG] start_infolog_session failed: %s", exc)
        return None


def end_infolog_session() -> None:
    """Close the run infolog (best-effort)."""
    global _session_log_file, _session_log_path
    try:
        if _session_log_file is not None:
            _session_log_file.close()
    except Exception:  # noqa: BLE001
        pass
    _session_log_file = None
    _session_log_path = None


def log_gaia_query(ra: float, dec: float, calculated_radius: float) -> None:
    """Structured Gaia query debug line."""
    log_event(f"GAIA QUERY: Center={ra},{dec} | Radius={calculated_radius:.2f} deg")


def log_exception(prefix: str, exc: BaseException) -> None:
    """Log exception message and full traceback into Infolog (for debugging worker/pool failures)."""
    import traceback

    log_event(f"{prefix}: {exc!s}")
    log_event(traceback.format_exc())


def log_phase_boundary(phase: str, *, status: str = "start") -> None:
    """Timestamped phase milestone."""
    log_milestone(f"[PHASE] {phase} {status}")


class InfologHandler(logging.Handler):
    """Writes VYVAR logger records into the run infolog."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _lock:
                _append_session_line(msg)
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
