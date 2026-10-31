"""Shared FITS filename suffix rules (case-insensitive on disk).

G02-05 / G13-31 (step 5): the ONE rule for "is this a FITS frame" and "list the FITS frames of a folder".
Every place that lists light / calibrated / aligned frames goes through :func:`list_fits_files` (or
:func:`path_suffix_is_fits` for a single path). The suffix test is case-insensitive (``.fit``, ``.fts``,
``.FITS``, ``.Fits`` ...); file NAMES are never case-folded, so ``X.fits`` and ``x.fits`` are two frames on a
case-sensitive file system. Names starting with ``.`` (hidden files / directories, macOS ``._x.fits``
AppleDouble files) are skipped, as ``glob`` does.
"""

from __future__ import annotations

import os
from pathlib import Path

# Compare with path.suffix.casefold()
FITS_SUFFIXES_LOWER = frozenset({".fits", ".fit", ".fts"})


def path_suffix_is_fits(path: Path | str) -> bool:
    return Path(path).suffix.casefold() in FITS_SUFFIXES_LOWER


def list_fits_files(directory: Path | str, *, recursive: bool = False) -> list[Path]:
    """Sorted FITS files in ``directory`` (``directory / relative`` paths, not resolved).

    ``recursive=False`` lists the folder itself (replaces ``glob("*.fits")``); ``recursive=True`` walks the
    tree, following directory symlinks like ``glob("**")`` (replaces ``rglob("*.fits")``). A missing folder
    gives ``[]``.
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    out: list[Path] = []
    if not recursive:
        for p in root.iterdir():
            if not p.name.startswith(".") and path_suffix_is_fits(p) and p.is_file():
                out.append(p)
        return sorted(out)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        base = Path(dirpath)
        for name in filenames:
            if name.startswith("."):
                continue
            p = base / name
            if path_suffix_is_fits(p) and p.is_file():
                out.append(p)
    return sorted(out)
