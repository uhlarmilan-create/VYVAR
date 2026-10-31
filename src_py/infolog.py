"""Draft infolog: one append-only ``infolog_<ts>.txt`` per run in the draft directory.

INFOLOG-ONE-2026-10: the run process (night_run) owns the file; there is no UI tab and no
second snapshot file (Milan: the log is read outside VYVAR)."""

from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_lock = threading.Lock()
_session_log_file: Any = None
# G12-08: every line (log_event, log_milestone, logger records) written before the run's draft
# directory exists; written when the file opens, cleared when a run starts or ends (G12-09).
_pending: list[str] = []
_PENDING_MAX = 50000
_pending_dropped = 0
# SPEED-2026-10: a worker process has no infolog file; its messages are collected here and the
# run process writes them (``capture_events`` / ``replay_events``)
_captured: list[str] | None = None

#: the folder of the VYVAR modules: records logged there go to the run log from INFO, the rest
#: (libraries) from WARNING (G12-07, Milan 2026-10-31)
_SRC_DIR = Path(__file__).resolve().parent


def safe_progress(progress_cb: Any) -> Any:
    """Progress reporter ``msg -> None`` around an optional callback (None: no-op). G06-22: the one copy.

    Some Windows consoles use legacy encodings (cp1252) and raise UnicodeEncodeError on diacritics;
    the message is then passed with ASCII backslash escapes so reporting never raises."""

    def _p(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    return _p


def _write_line(line: str) -> None:
    """Write one line into the open run infolog, else keep it until the file opens (call with _lock)."""
    global _pending_dropped
    if _session_log_file is None:
        if len(_pending) < _PENDING_MAX:
            _pending.append(line)
        else:
            _pending_dropped += 1
        return
    try:
        _session_log_file.write(f"{line}\n")
        _session_log_file.flush()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("session infolog append failed: %s", exc)


def _clear_pending() -> None:
    global _pending_dropped
    with _lock:
        _pending.clear()
        _pending_dropped = 0


def log_event(message: str) -> None:
    """Append one line to the run infolog; before the file opens the line is kept and written when it
    opens (G12-08). Collected inside ``capture_events`` (worker processes)."""
    try:
        if _captured is not None:
            _captured.append(str(message))
            return
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"{ts}  {message}"
        with _lock:
            _write_line(line)
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
    """Milestone line (same path as ``log_event``, G12-08: one buffer for every line)."""
    log_event(message)


def start_infolog_session(draft_dir: str | Path) -> str | None:
    """Open the run infolog ``draft_dir/infolog_<ts>.txt`` (append, flushed per line); the lines
    logged before it opened are written first."""
    global _session_log_file, _pending_dropped
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
            if _pending_dropped:
                fh.write(f"[INFOLOG] {_pending_dropped} line(s) logged before this file opened were not kept "
                         f"(limit {_PENDING_MAX})\n")
            for m in _pending:
                fh.write(f"{m}\n")
            _pending.clear()
            _pending_dropped = 0
            fh.flush()
            _session_log_file = fh
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("[INFOLOG] start_infolog_session failed: %s", exc)
        return None


def infolog_session_open() -> bool:
    """True while a run infolog file is open."""
    return _session_log_file is not None


def end_infolog_session() -> None:
    """Close the run infolog (best-effort); lines kept for a file that never opened are dropped (G12-09)."""
    global _session_log_file
    try:
        if _session_log_file is not None:
            _session_log_file.close()
    except Exception:  # noqa: BLE001
        pass
    _session_log_file = None
    _clear_pending()


def log_phase_boundary(phase: str, *, status: str = "start") -> None:
    """Timestamped phase milestone."""
    log_milestone(f"[PHASE] {phase} {status}")


class InfologHandler(logging.Handler):
    """Writes logger records into the run infolog (kept until the file opens)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _lock:
                _write_line(msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)


class _StdoutHandler(logging.StreamHandler):
    """Console handler bound to the current ``sys.stdout`` (never to a stream that was replaced)."""

    @property
    def stream(self) -> Any:  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, _value: Any) -> None:
        pass


class VyvarRecordFilter(logging.Filter):
    """G12-07: INFO and above of every VYVAR module, WARNING and above of the libraries. A record is a
    VYVAR record when its logger is named after a ``src_py`` module (``getLogger(__name__)``,
    ``"pipeline"``) or it was logged from a ``src_py`` file (``logging.info``)."""

    def __init__(self) -> None:
        super().__init__()
        self._modules = frozenset(p.stem for p in _SRC_DIR.glob("*.py"))
        self._is_vyvar: dict[tuple[str, str], bool] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno < logging.INFO:
            return False
        key = (str(record.name), str(record.pathname or ""))
        hit = self._is_vyvar.get(key)
        if hit is None:
            hit = key[0].split(".", 1)[0] in self._modules
            if not hit:
                try:
                    hit = Path(key[1]).resolve().parent == _SRC_DIR
                except (OSError, ValueError):
                    hit = False
            self._is_vyvar[key] = hit
        return hit


def _run_log_formatter() -> logging.Formatter:
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  [%(name)s]  %(message)s", datefmt="%H:%M:%S")
    fmt.converter = time.gmtime
    return fmt


def setup_run_logging() -> Callable[[], None]:
    """G12-07 (Milan 2026-10-31): the one logging setup of a run, the same for the CLI and the GUI's job
    child. INFO and above of every VYVAR module and WARNING and above of the libraries go to the run
    infolog (kept until it opens, G12-08) and to stdout; library loggers keep their own levels (the
    run adds the LOG-NOISE repeat filter, ``quiet_repeated_library_warnings``). Also clears lines kept from before this run. Returns the teardown
    (a no-op when the setup was already active)."""
    root = logging.getLogger()
    if any(isinstance(h, InfologHandler) for h in root.handlers):
        return lambda: None
    _clear_pending()
    flt = VyvarRecordFilter()
    fmt = _run_log_formatter()
    handlers: list[logging.Handler] = [InfologHandler(), _StdoutHandler()]
    for h in handlers:
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        h.addFilter(flt)
        root.addHandler(h)
    old_level = root.level
    if root.getEffectiveLevel() > logging.INFO:
        root.setLevel(logging.INFO)

    def _teardown() -> None:
        for h in handlers:
            root.removeHandler(h)
        root.setLevel(old_level)

    return _teardown


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
