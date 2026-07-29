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
_session_log_path: Path | None = None
_session_log_file: Any = None
_milestones: list[str] = []

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
            _append_session_line(line)
    except Exception as exc:  # noqa: BLE001 - Infolog must never crash the pipeline
        logging.getLogger(__name__).debug("log_event append failed: %s", exc)


def log_milestone(message: str) -> None:
    """Permanent milestone (also written to session disk log if active)."""
    try:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"{ts}  {message}"
        with _lock:
            _milestones.append(line)
            _lines.append(line)
            _append_session_line(line)
        logging.getLogger("pipeline").info(message)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("log_milestone append failed: %s", exc)


def _append_session_line(line: str) -> None:
    global _session_log_file
    if _session_log_file is None:
        return
    try:
        _session_log_file.write(f"{line}\n")
        _session_log_file.flush()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("session infolog append failed: %s", exc)


def start_infolog_session(draft_dir: str | Path) -> str | None:
    """Open a durable on-disk infolog for this run (survives ring-buffer eviction)."""
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
        fh.write("# session: durable append log (full run)\n")
        fh.write("#" + "=" * 60 + "\n\n")
        for m in _milestones:
            fh.write(f"{m}\n")
        fh.flush()
        _session_log_path = path
        _session_log_file = fh
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("[INFOLOG] start_infolog_session failed: %s", exc)
        return None


def end_infolog_session() -> None:
    """Close durable session log handle (best-effort)."""
    global _session_log_file, _session_log_path
    try:
        if _session_log_file is not None:
            _session_log_file.close()
    except Exception:  # noqa: BLE001
        pass
    _session_log_file = None
    _session_log_path = None


def get_milestones() -> list[str]:
    with _lock:
        return list(_milestones)


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


def write_run_infolog(draft_dir: str | Path, entries: list[str] | None = None) -> str | None:
    """Single entry point to persist the operator infolog for a completed run."""
    return _finalize_or_save_infolog(draft_dir, entries)


def _finalize_or_save_infolog(draft_dir: str | Path, entries: list[str] | None = None) -> str | None:
    """Prefer the durable session log as the authoritative operator artefact."""
    global _session_log_path, _session_log_file
    session_path = _session_log_path
    if session_path is not None:
        path = Path(session_path)
        try:
            if _session_log_file is not None:
                _session_log_file.flush()
                _session_log_file.close()
                _session_log_file = None
            if path.is_file():
                _mark_authoritative_session_infolog(path)
                _session_log_path = None
                return str(path)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("[INFOLOG] finalize session log failed: %s", exc)
        finally:
            _session_log_path = None
            _session_log_file = None
    return save_infolog_to_disk(draft_dir, entries)


def _mark_authoritative_session_infolog(path: Path) -> None:
    """Tag durable session file as the complete operator record."""
    text = path.read_text(encoding="utf-8")
    marker = "# authoritative: durable session log (complete operator record)"
    if marker in text:
        return
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.startswith("#") and "====" not in ln:
            insert_at = i + 1
        else:
            break
    header = marker + "\n"
    milestones = get_milestones()
    if milestones and "milestones (never evicted)" not in text:
        header += "# --- milestones (never evicted) ---\n"
        for m in milestones:
            header += f"{m}\n"
        header += "\n"
    lines.insert(insert_at, header)
    path.write_text("".join(lines), encoding="utf-8")


def get_active_session_log_path() -> Path | None:
    """Path to the durable session log for the current run, if any."""
    return _session_log_path


def log_phase_boundary(phase: str, *, status: str = "start") -> None:
    """Timestamped phase milestone (durable; survives ring-buffer eviction)."""
    log_milestone(f"[PHASE] {phase} {status}")


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
        milestones = get_milestones()
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# VYVAR Infolog - {ts}\n")
            f.write(f"# Draft: {root}\n")
            f.write("# timestamps: UTC\n")
            f.write("# partial: ring-buffer tail only (no active session log)\n")
            f.write("#" + "=" * 60 + "\n\n")
            if milestones:
                f.write("# --- milestones (never evicted) ---\n")
                for entry in milestones:
                    f.write(f"{entry}\n")
                f.write("\n")
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
        _milestones.clear()
    end_infolog_session()


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
