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
_session_log_file: Any = None
# milestones logged before the run's draft directory exists; written when the file opens
_milestones: list[str] = []
# SPEED-2026-10: a worker process has no infolog file; its messages are collected here and the
# run process writes them (``capture_events`` / ``replay_events``)
_captured: list[str] | None = None

VYVAR_LOGGERS = ("pipeline", "importer")


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
    global _session_log_file
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
        _session_log_file = fh
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("[INFOLOG] start_infolog_session failed: %s", exc)
        return None


def end_infolog_session() -> None:
    """Close the run infolog (best-effort)."""
    global _session_log_file
    try:
        if _session_log_file is not None:
            _session_log_file.close()
    except Exception:  # noqa: BLE001
        pass
    _session_log_file = None


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
    if _handler_already_attached():
        for name in VYVAR_LOGGERS:
            logging.getLogger(name).propagate = False
        return

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  [%(name)s]  %(message)s", datefmt="%H:%M:%S")
    fmt.converter = time.gmtime
    h = InfologHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(fmt)

    for name in VYVAR_LOGGERS:
        lg = logging.getLogger(name)
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        # Avoid propagating to the root logger: hosted Streamlit / runners sometimes
        # attach a StreamHandler to stdout that is later closed; LOGGER.info would then
        # raise ValueError("I/O operation on closed file.").
        lg.propagate = False


# LOG-NOISE-2026-10: library warnings that repeat for every frame or star (astropy FITSFixedWarning
# for the camera's RADECSYS / MJD-OBS, photutils "star excluded from ePSF fitting", ...) are written
# once per process with the first example; the run reports how many repeats were left out.
_NUMBER = None


def _warning_template(message: str) -> str:
    import re  # noqa: PLC0415

    global _NUMBER
    if _NUMBER is None:
        _NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    return _NUMBER.sub("#", message)


class RepeatedWarningFilter(logging.Filter):
    """Pass the first record of every message template, count the rest."""

    def __init__(self) -> None:
        super().__init__()
        self.repeats: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        key = f"{record.name}: {_warning_template(record.getMessage())}"
        if key in self.repeats:
            self.repeats[key] += 1
            return False
        self.repeats[key] = 0
        return True


_repeat_filter: RepeatedWarningFilter | None = None


def _library_loggers() -> list[logging.Logger]:
    """The libraries' own logger objects. ``logging.getLogger("astroquery")`` before astroquery
    is imported would create a plain Logger under that name and break astroquery's logger setup
    (it calls ``log._set_defaults``), so each logger is taken from its library."""
    out: list[logging.Logger] = [logging.getLogger("py.warnings")]
    try:
        from astropy import log as _astropy_log  # noqa: PLC0415

        out.append(_astropy_log)
    except ImportError:
        pass
    try:
        from astroquery import log as _astroquery_log  # noqa: PLC0415

        out.append(_astroquery_log)
    except ImportError:
        pass
    return out


def quiet_repeated_library_warnings() -> RepeatedWarningFilter:
    """Install the repeat filter on the library loggers once per process."""
    global _repeat_filter
    if _repeat_filter is None:
        _repeat_filter = RepeatedWarningFilter()
        for lg in _library_loggers():
            lg.addFilter(_repeat_filter)
    return _repeat_filter


def log_repeated_warnings_summary() -> None:
    """One INFO line per warning that repeated in this process (how many repeats were left out)."""
    if _repeat_filter is None:
        return
    for key, n in sorted(_repeat_filter.repeats.items()):
        if n:
            logging.getLogger("pipeline").info("[LOG] warning repeated %d more time(s), shown once: %s", n, key[:200])
    _repeat_filter.repeats.clear()
