"""Per-run outputs of a setup's photometry folder: what a new run replaces.

OUTPUT-REVIEW-2026-10 step 7 (Milan 2026-10-21). Aperture photometry writes the per-target products of every
target it measures; a target it skips writes none. Without a clean start the files of an earlier
run stayed (a skipped target kept its light curve and was exported again from the old data; a
target without a check star now kept its old check-star series). A new aperture-photometry run therefore removes
the per-target products and everything derived from them (ePSF light curves and exports, AAVSO /
VarAstro files) before it writes; the report clears its image cache, the export its folders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

#: lightcurves/: per-target products of aperture photometry and the stages after it (variability, ePSF);
#: trust_*.json, cutout_*.png: written by earlier versions only.
LIGHTCURVE_PRODUCT_PATTERNS: tuple[str, ...] = (
    "lightcurve_*.csv",
    "lightcurve_*.png",
    "comp_quality_*.json",
    "check_kmag_*.csv",
    "field_map_*.jpg",
    "field_map_*.png",
    "varcand_*.csv",
    "trust_*.json",
    "cutout_*.png",
)
#: lightcurves_reports/<sub>/: exports made from the light curves.
EXPORT_SUBDIRS: tuple[str, ...] = ("aavso", "varastro", "epsf")
REPORT_CACHE_DIRNAME = "_report_cache"


def remove_files(directory: Path | str, patterns: Iterable[str]) -> int:
    """Delete the files of ``directory`` (not recursive) matching ``patterns``; returns the count."""
    d = Path(directory)
    if not d.is_dir():
        return 0
    n = 0
    for pat in patterns:
        for p in d.glob(pat):
            if p.is_file():
                p.unlink()
                n += 1
    return n


def clear_export_dirs(photometry_dir: Path | str, subdirs: Iterable[str] = EXPORT_SUBDIRS) -> int:
    """Delete the export files of ``lightcurves_reports/<sub>/`` and of its ``_dontsubmit/`` (written
    again from this run)."""
    rep = Path(photometry_dir) / "lightcurves_reports"
    # G11-03: the export manifest describes these files - it goes with them
    n_man = remove_files(rep, ("export_manifest.json",)) if "aavso" in tuple(subdirs) else 0
    return n_man + sum(remove_files(rep / s, ("*",)) + remove_files(rep / s / "_dontsubmit", ("*",)) for s in subdirs)


def clear_aperture_products(photometry_dir: Path | str) -> dict[str, int]:
    """Start of aperture photometry: remove the per-target products and the exports of the previous run."""
    phot = Path(photometry_dir)
    out = {
        "lightcurves": remove_files(phot / "lightcurves", LIGHTCURVE_PRODUCT_PATTERNS),
        "exports": clear_export_dirs(phot),
    }
    if out["lightcurves"] or out["exports"]:
        LOGGER.info("[APERTURE] previous run removed: %d light-curve products, %d export files",
                    out["lightcurves"], out["exports"])
    return out


def clear_report_cache(photometry_dir: Path | str) -> int:
    """Start of a report: everything in ``_report_cache`` is drawn again."""
    cache = Path(photometry_dir) / REPORT_CACHE_DIRNAME
    if not cache.is_dir():
        return 0
    return remove_files(cache, ("*",))
