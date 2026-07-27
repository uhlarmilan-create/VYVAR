"""Durable infolog session + milestones (pure Python; complements compiled ``infolog``).

The release build compiles ``infolog.py`` to a .pyd that shadows the source tree.
Session/milestone APIs live here so development and tests work before the next Cython rebuild.
When ``infolog.py`` is recompiled, these helpers should be merged back into it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infolog import get_lines, log_event

_milestones: list[str] = []
_session_log_path: Path | None = None
_session_log_file: Any = None


def log_milestone(message: str) -> None:
    """Permanent milestone (also written to session disk log if active)."""
    try:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"{ts}  {message}"
        _milestones.append(line)
        log_event(message)
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
    return list(_milestones)


def save_infolog_to_disk(draft_dir: str | Path, entries: list[str] | None = None) -> str | None:
    """
    Save Infolog entries with milestones prepended (never evicted from ring buffer).

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
