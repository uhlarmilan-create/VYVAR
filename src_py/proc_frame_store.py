"""ProcFrameStore - unified in-memory store for proc_*.csv frames.

Replaces shared_csv_cache (Phase 1) and _phase2a_csv_cache (Phase 2A)
with a single object built once per pipeline run.

Design:
- One disk read per frame (union of all consumer columns)
- Column projection views - consumers get a sub-DataFrame without copying
- Single parse pass via usecols callable (no separate nrows=0 sniff)
- dict-compatible interface for backward compat (csv_cache.get() pattern)

Proc schema note (2026-06-25): ``sky_adu_per_px_annulus`` holds per-star annulus sky (ADU/px)
for the Howell err model; ``noise_floor_adu`` remains the DAO detection floor (MASTERSTAR / SNR table).
F-BINGAIN-1 (2026-07-10): ``sigma_bkg_ap`` / ``err_bkg_source`` - empirical background noise provenance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id_series

# Union of all columns needed by any consumer.
# Superset of Phase 1 _needed_cols + Phase 2A _needed_cols_2a.
# Add new columns here if a consumer needs them - no re-read required.
PROC_STORE_COLS = [
    # Identity
    "catalog_id",
    "name",
    # Times
    "bjd_tdb_mid",
    "hjd_mid",
    "jd_mid",
    # Photometry
    "dao_flux",
    "flux",
    "noise_floor_adu",
    "sky_adu_per_px_annulus",
    "sigma_bkg_ap",
    "err_bkg_source",
    "aperture_r_px",
    "aperture_factor_applied",
    "fwhm_px_for_aperture",
    "fwhm_px_scope",
    "snr_aperture_mode",
    "flux_small",
    "flux_large",
    # Quality flags
    "peak_max_adu",
    "saturate_limit_adu_85pct",
    "fwhm_estimate_px",
    "psf_chi2",
    "is_usable",
    "is_saturated",
    "is_noisy",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
    "photometry_ok",
    "edge_safe_10px",
    "edge_fail",
    # Astrometry / position
    "x",
    "y",
    "ra_deg",
    "dec_deg",
    "sky_annulus_r_out_px",
    # Atmosphere
    "airmass",
    # Catalog / color
    "mag",
    "bp_rp",
    "b_v",
    "zone",
    "source_type",
    "gaia_dr3_variable_catalog",
    # PSF photometry (gated - Phase 2A adaptive / PSF selector)
    "psf_flux",
    "psf_flux_err",
    "psf_fit_ok",
    "psf_chi2",
    "psf_quality",
    "psf_quality_fallback",
    "psf_snr",
    "psf_ac_factor",
    "psf_ac_n_used",
    "psf_ac_applied",
    # G2-F002b trust (per-frame proc CSV; Phase 2A frame_time_lookup)
    "catalog_match_mode",
    # wcs_untrusted is NOT stored here - derived in Phase 2A LC export from
    # catalog_match_mode via catalog_match_trust.is_wcs_untrusted_catalog_match_mode.
]

PROC_CSV_GLOB = "proc_*.csv"
"""Canonical glob for per-frame proc CSV. Matches both naming conventions:
calibrated (proc_<obj>_Light_*.csv) and pre-cal (proc_<obj>_*.csv)."""

_MASTERSTAR_PROC_STEMS = frozenset({"masterstar"})


def is_masterstar_proc_name(name: str | Path) -> bool:
    """True when ``name`` is the stacked-reference proc sidecar, not a science epoch.

    Matches ``proc_MASTERSTAR.csv`` and case variants from ``proc_csv_path_for_aligned_fits``.
    """
    stem = Path(name).stem.casefold()
    if not stem.startswith("proc_"):
        return False
    ref = stem[5:]
    return ref in _MASTERSTAR_PROC_STEMS


def proc_csv_path_for_aligned_fits(base_path: Path | str) -> Path:
    """Map aligned FITS path to canonical per-frame proc CSV: ``proc_<stem>.csv``.

    Idempotent when the FITS stem already starts with ``proc_`` (raw calibrated aligned frames).
    Mirrors ``pipeline._safe_proc_name`` stem logic for pre-cal ``Chi_H_*.fits`` basenames.
    """
    p = Path(base_path)
    stem = p.stem
    if not stem.casefold().startswith("proc_"):
        stem = f"proc_{stem}"
    return p.with_name(f"{stem}.csv")


def list_proc_csvs(proc_dir: Path | str, *, recursive: bool = False) -> list[Path]:
    """Single entry point for listing per-frame proc CSV paths (sorted).

    recursive=False - flat glob (typical per_frame_csv_dir).
    recursive=True  - rglob tree (ProcFrameStore.build).
    """
    root = Path(proc_dir).expanduser()
    it = root.rglob(PROC_CSV_GLOB) if recursive else root.glob(PROC_CSV_GLOB)
    return sorted(p for p in it if not is_masterstar_proc_name(p))


_NUMERIC_COLS = (
    "flux",
    "dao_flux",
    "noise_floor_adu",
    "sky_adu_per_px_annulus",
    "sigma_bkg_ap",
    "aperture_r_px",
    "aperture_factor_applied",
    "fwhm_px_for_aperture",
    "fwhm_px_scope",
    "snr_aperture_mode",
    "peak_max_adu",
    "saturate_limit_adu_85pct",
    "psf_chi2",
    "fwhm_estimate_px",
    "flux_small",
    "flux_large",
    "psf_flux",
    "psf_flux_err",
    "psf_snr",
    "x",
    "y",
    "ra_deg",
    "dec_deg",
    "sky_annulus_r_out_px",
    "airmass",
    "mag",
    "bp_rp",
    "b_v",
)


class ProcFrameStore:
    """Unified in-memory store for proc_*.csv frames.

    Usage:
        store = ProcFrameStore.build(proc_csv_dir)
        df = store.get_frame("proc_frame001.csv")        # full union cols
        df = store.get_frame("proc_frame001.csv",
                             cols=["catalog_id","dao_flux"])  # projection
        # dict-compatible (for legacy csv_cache.get() callers):
        df = store.get(str(path))   # returns None on miss (not KeyError)
    """

    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}
        self._columns_on_disk: dict[str, list[str]] = {}
        self.n_frames: int = 0
        self.n_stars_median: float = 0.0

    @classmethod
    def build(
        cls,
        proc_csv_dir: Path,
        *,
        glob_pattern: str = PROC_CSV_GLOB,
        extra_cols: list[str] | None = None,
        dtype_overrides: dict | None = None,
    ) -> ProcFrameStore:
        """Build store by reading all proc_*.csv files once."""
        store = cls()
        if glob_pattern == PROC_CSV_GLOB:
            paths = list_proc_csvs(proc_csv_dir, recursive=True)
        else:
            paths = sorted(
                p
                for p in Path(proc_csv_dir).rglob(glob_pattern)
                if not is_masterstar_proc_name(p)
            )
        if not paths:
            logging.warning(
                "[ProcFrameStore] No files matching %s in %s",
                glob_pattern,
                proc_csv_dir,
            )
            return store

        want_cols = set(PROC_STORE_COLS)
        if extra_cols:
            want_cols.update(extra_cols)

        _dtypes: dict[str, type] = {"catalog_id": str, "name": str}
        if dtype_overrides:
            _dtypes.update(dtype_overrides)

        row_counts: list[int] = []
        n_loaded = 0
        n_failed = 0

        for path in paths:
            key = str(path)
            try:
                df = pd.read_csv(
                    path,
                    usecols=lambda c: c in want_cols,
                    dtype={k: v for k, v in _dtypes.items()},
                    low_memory=False,
                )
                store._columns_on_disk[key] = list(df.columns)

                for id_col in ("catalog_id", "name"):
                    if id_col in df.columns:
                        df[id_col] = normalize_gaia_source_id_series(df[id_col])

                for num_col in _NUMERIC_COLS:
                    if num_col in df.columns:
                        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

                store._store[key] = df
                row_counts.append(len(df))
                n_loaded += 1

            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "[ProcFrameStore] Cannot read %s: %s",
                    path.name,
                    exc,
                )
                n_failed += 1

        store.n_frames = n_loaded
        store.n_stars_median = float(np.median(row_counts)) if row_counts else 0.0

        ram_mb = sum(df.memory_usage(deep=True).sum() for df in store._store.values()) / 1e6

        logging.info(
            "[ProcFrameStore] Built: %d frames loaded, %d failed | "
            "median %d rows/frame | RAM ~%.1f MB",
            n_loaded,
            n_failed,
            int(store.n_stars_median),
            ram_mb,
        )
        if n_failed > 0:
            logging.warning(
                "[PERF-5] %d frame(s) failed to load - "
                "pipeline will fall back to disk for these",
                n_failed,
            )

        return store

    def get_frame(
        self,
        path: str | Path,
        cols: list[str] | None = None,
    ) -> pd.DataFrame | None:
        """Return frame DataFrame, optionally projected to cols."""
        key = str(path)
        df = self._store.get(key)
        if df is None:
            return None
        if cols is None:
            return df
        available = [c for c in cols if c in df.columns]
        return df[available]

    def get(
        self,
        key: str,
        default: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        """dict-compatible get() for legacy csv_cache.get() callers."""
        return self._store.get(key, default)

    def __getitem__(self, key: str) -> pd.DataFrame:
        result = self._store.get(str(key))
        if result is None:
            raise KeyError(key)
        return result

    def keys(self) -> Iterator[str]:
        return iter(self._store.keys())

    def values(self) -> Iterator[pd.DataFrame]:
        return iter(self._store.values())

    def items(self) -> Iterator[tuple[str, pd.DataFrame]]:
        return iter(self._store.items())

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def paths(self) -> list[str]:
        """All loaded frame paths (string keys)."""
        return list(self._store.keys())
