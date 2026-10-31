"""File stamps as Streamlit cache keys (G14-26 / G14-27).

A view that reads files caches its result with ``st.cache_data`` keyed by the path and a stamp of the
files it reads: ``(path, mtime_ns, size)`` per file, so a rewritten file is read again. The stamp is an
ordinary (hashed) argument of the cached function - never a ``_``-prefixed one, which Streamlit leaves
out of the key.
"""
from __future__ import annotations

from pathlib import Path


def file_stamp(*paths: Path | str | None) -> tuple[tuple[str, int, int], ...]:
    """``(path, mtime_ns, size)`` of every path; ``(path, -1, -1)`` for a missing file or ``None``."""
    out: list[tuple[str, int, int]] = []
    for p in paths:
        if p is None:
            out.append(("", -1, -1))
            continue
        try:
            st = Path(p).stat()
            out.append((str(p), int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            out.append((str(p), -1, -1))
    return tuple(out)


def dir_stamp(directory: Path | str, pattern: str = "*") -> tuple[tuple[str, int, int], ...]:
    """``file_stamp`` of the files matching ``pattern`` in ``directory`` (sorted; empty when it is missing)."""
    d = Path(directory)
    if not d.is_dir():
        return ()
    return file_stamp(*sorted(p for p in d.glob(pattern) if p.is_file()))


def config_fingerprint(cfg: object | None) -> str:
    """Hash of every config value (``AppConfig.to_json``) - a hashed cache argument, so a Settings save is
    part of the key of a view whose loader reads the config (NEW-2 / NEW-3)."""
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    if cfg is None:
        return "none"
    data = cfg.to_json() if hasattr(cfg, "to_json") else vars(cfg)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]
