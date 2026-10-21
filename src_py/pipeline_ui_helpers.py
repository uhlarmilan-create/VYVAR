"""Moved from pipeline.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path

from pipeline import (
    _archive_raw_to_calibrated_light,
)


def _resolve_light_fits_for_quality_inspection(archive: Path, raw_fp: Path | str) -> Path | None:
    """Prefer calibrated counterpart; else existing raw path under archive."""
    m = _archive_raw_to_calibrated_light(archive, raw_fp)
    if m is not None:
        return m[0]
    p = Path(raw_fp)
    if p.is_file():
        return p
    ap = Path(archive).expanduser()
    p2 = ap / p
    if p2.is_file():
        return p2
    return None


def resolve_masterstars_metadata_csv(platesolve_dir: Path | str) -> Path | None:
    """Return masterstars metadata CSV in a platesolve setup dir (full_match preferred)."""
    ps = Path(platesolve_dir)
    for name in ("masterstars_full_match.csv", "masterstars.csv"):
        p = ps / name
        if p.is_file():
            return p
    return None

