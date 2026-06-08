"""Shared FITS filename suffix rules (case-insensitive on disk)."""

from __future__ import annotations

from pathlib import Path

# Compare with path.suffix.casefold()
FITS_SUFFIXES_LOWER = frozenset({".fits", ".fit", ".fts"})


def path_suffix_is_fits(path: Path) -> bool:
    return path.suffix.casefold() in FITS_SUFFIXES_LOWER
