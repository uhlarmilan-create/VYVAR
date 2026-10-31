"""ProcFrameStore - unified in-memory store for proc_*.csv frames.

Replaces shared_csv_cache (comparison-star selection) and _aperture_csv_cache (aperture photometry)
with a single object built once per pipeline run.

Design:
- One disk read per frame (union of all consumer columns)
- Column projection views - consumers get a sub-DataFrame without copying
- Single parse pass via usecols callable (no separate nrows=0 sniff)
- dict-compatible interface for backward compat (csv_cache.get() pattern)

Proc schema note (2026-06-25): ``sky_adu_per_px_annulus`` holds per-star annulus sky (ADU/px)
for the Howell err model. OUTPUT-REVIEW-2026-10: ``name`` / ``flux`` / ``noise_floor_adu`` (copies of
``catalog_id`` / ``dao_flux`` / ``sky_adu_per_px_annulus``) are no longer written.
F-BINGAIN-1 (2026-07-10): ``sigma_bkg_ap`` / ``err_bkg_source`` - empirical background noise provenance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from photometry_defs import FLUX_K_COLUMNS

from gaia_catalog_id import catalog_id_series_for_masterstars_export, normalize_gaia_source_id_series

# Union of all columns needed by any consumer.
# Superset of comparison-star selection _needed_cols + aperture photometry _needed_cols_2a.
# Add new columns here if a consumer needs them - no re-read required.
PROC_STORE_COLS = [
    # Identity
    "catalog_id",
    # Times
    "bjd_tdb_mid",
    "hjd_mid",
    "jd_mid",
    # Photometry
    "dao_flux",
    "sky_adu_per_px_annulus",
    "sky_n_px",  # CCD-ONE-2026-10: n_sky of the CCD equation
    "sigma_bkg_ap",
    "err_bkg_source",
    "aperture_r_px",
    "fwhm_px_for_aperture",
    # APERTURE-K-2026-10: light-curve aperture grid r = k x FWHM
    *FLUX_K_COLUMNS,
    # Quality flags
    "peak_max_adu",
    "saturate_limit_adu_85pct",
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
    "gaia_dr3_variable_catalog",
    # G2-F002b trust (per-frame proc CSV; aperture photometry frame_time_lookup)
    "catalog_match_mode",
    # wcs_untrusted is NOT stored here - derived in the aperture-photometry LC export from
    # catalog_match_mode via catalog_match_trust.is_wcs_untrusted_catalog_match_mode.
]

# OUTPUT-REVIEW-2026-10 step 4b (Milan 2026-10-21: "round, but scientifically acceptable"): proc values
# are written with the precision the measurement has, not with the 17 digits of a float64.
#  - x / y: 1e-4 px (centroids are good to ~1e-2 px);
#  - ra / dec: 1e-7 deg = 0.36 mas (Gaia DR3 positions, epoch-propagated, ~0.02-1 mas);
#  - catalogue magnitudes and colour: 1e-6 mag (Gaia G errors >= 2e-4 mag);
#  - fluxes, sky, noise, peaks, radii, chi2, airmass: 8 significant digits, i.e. a relative
#    error <= 5e-8 (<= 5.4e-8 mag), more than three orders of magnitude below the best
#    photometric errors (>= 1e-4 mag). Times keep their 6 decimals (1e-6 d = 0.09 s, _vyvar_df_to_csv).
PROC_ROUND_DECIMALS: dict[str, int] = {
    "x": 4,
    "y": 4,
    "ra_deg": 7,
    "dec_deg": 7,
    "mag": 6,
    "bp_rp": 6,
    "phot_g_mean_mag": 6,
}
PROC_ROUND_SIGNIFICANT = 8
PROC_ROUND_SIG_COLS: tuple[str, ...] = (
    "dao_flux",
    *FLUX_K_COLUMNS,
    "peak_max_adu",
    "peak_max_adu_aligned",
    "saturate_limit_adu_85pct",
    "sky_adu_per_px_annulus",
    "sigma_bkg_ap",
    "aperture_r_px",
    "fwhm_px_for_aperture",
    "sky_annulus_r_out_px",
    "airmass",
    "psf_flux",
    "psf_flux_err",
    "psf_chi2",
    "psf_chi2_locus_resid_sigma",
)


def _round_significant(values: pd.Series, digits: int) -> pd.Series:
    """Round to ``digits`` significant digits (shortest decimal form: float(f"{v:.8g}"))."""
    num = pd.to_numeric(values, errors="coerce")
    fmt = f"{{:.{int(digits)}g}}"
    out = num.copy()
    ok = np.isfinite(num.to_numpy(dtype=np.float64, na_value=np.nan))
    out[ok] = [float(fmt.format(v)) for v in num[ok].to_numpy(dtype=np.float64)]
    return out


def round_proc_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Copy of ``df`` with the proc precision applied (all known columns, or only ``columns``)."""
    out = df.copy()
    want = set(columns) if columns is not None else None
    for col, nd in PROC_ROUND_DECIMALS.items():
        if col in out.columns and (want is None or col in want):
            num = pd.to_numeric(out[col], errors="coerce")
            if num.notna().any() or out[col].isna().all():
                out[col] = num.round(nd)
    for col in PROC_ROUND_SIG_COLS:
        if col in out.columns and (want is None or col in want):
            if pd.to_numeric(out[col], errors="coerce").notna().any():
                out[col] = _round_significant(out[col], PROC_ROUND_SIGNIFICANT)
    return out


def write_proc_csv(df: pd.DataFrame, path: Path | str) -> None:
    """The one writer of proc CSVs: proc precision, frame values left to ``frames.csv`` when it
    holds them for this file, then the VYVAR CSV writer."""
    _vyvar_df_to_csv(round_proc_columns(strip_frame_columns(df, path)), path)


# OUTPUT-REVIEW-2026-10 step 4b (Milan 2026-10-21): values that are the same for every row of a
# frame live once per frame in ``frames.csv`` next to the proc files (one row per proc file).
# ``read_proc_csv`` puts them back as columns, so readers see the same table as before;
# drafts made earlier carry them inline in every proc row and are read as they are.
FRAME_TABLE_NAME = "frames.csv"
FRAME_TABLE_KEY = "proc_csv"
PROC_FRAME_COLS: tuple[str, ...] = (
    "jd_mid",
    "hjd_mid",
    "bjd_tdb_mid",
    "airmass",
    "aperture_r_px",
    "fwhm_px_for_aperture",
    "sky_annulus_r_out_px",
    "catalog_match_mode",
)

_FRAME_TABLE_CACHE: dict[str, tuple[tuple[int, int], dict[str, dict[str, object]]]] = {}


def frame_table_path(proc_dir: Path | str) -> Path:
    return Path(proc_dir) / FRAME_TABLE_NAME


def frame_values_of(df: pd.DataFrame) -> dict[str, object]:
    """The per-frame values of one proc table (first non-missing value of each frame column)."""
    out: dict[str, object] = {}
    for col in PROC_FRAME_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if col == "catalog_match_mode":
            s = s.astype(str).str.strip()
            s = s[s != ""]
        out[col] = s.iloc[0] if len(s) else None
    return out


def write_frame_table(proc_dir: Path | str, rows: dict[str, dict[str, object]]) -> Path | None:
    """Write / update ``frames.csv`` with ``{proc_csv_name: {column: value}}`` (merged by name)."""
    if not rows:
        return None
    path = frame_table_path(proc_dir)
    table = dict(load_frame_table(proc_dir))
    for name, vals in rows.items():
        table[str(Path(name).name)] = dict(vals)
    recs = [{FRAME_TABLE_KEY: k, **{c: v.get(c) for c in PROC_FRAME_COLS}} for k, v in sorted(table.items())]
    df = pd.DataFrame(recs, columns=[FRAME_TABLE_KEY, *PROC_FRAME_COLS])
    _vyvar_df_to_csv(round_proc_columns(df), path)
    _FRAME_TABLE_CACHE.pop(str(path), None)
    return path


def write_frame_tables(rows_by_proc_path: dict[str, dict[str, object]]) -> list[Path]:
    """``frames.csv`` in every directory of the given proc files (``{proc_path: values}``)."""
    by_dir: dict[str, dict[str, dict[str, object]]] = {}
    for proc_path, vals in rows_by_proc_path.items():
        p = Path(proc_path)
        by_dir.setdefault(str(p.parent), {})[p.name] = vals
    out: list[Path] = []
    for d, rows in by_dir.items():
        w = write_frame_table(d, rows)
        if w is not None:
            out.append(w)
    return out


def load_frame_table(proc_dir: Path | str) -> dict[str, dict[str, object]]:
    """``{proc_csv_name: {column: value}}``; empty when the directory has no frame table."""
    path = frame_table_path(proc_dir)
    try:
        st = path.stat()
    except OSError:
        return {}
    mt = (int(st.st_mtime_ns), int(st.st_size))  # G05-47: a rewrite within the mtime granularity changes the size
    hit = _FRAME_TABLE_CACHE.get(str(path))
    if hit is not None and hit[0] == mt:
        return hit[1]
    df = pd.read_csv(path, dtype={FRAME_TABLE_KEY: str, "catalog_match_mode": str}, low_memory=False)
    table: dict[str, dict[str, object]] = {}
    for rec in df.to_dict("records"):
        name = str(rec.pop(FRAME_TABLE_KEY))
        table[name] = {k: v for k, v in rec.items() if k in PROC_FRAME_COLS and not _is_missing(v)}
    _FRAME_TABLE_CACHE[str(path)] = (mt, table)
    return table


def _is_missing(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return isinstance(v, str) and v.strip() == ""


def attach_frame_columns(df: pd.DataFrame, proc_path: Path | str, want=None) -> pd.DataFrame:
    """Add the frame values of ``proc_path`` from ``frames.csv`` as columns ``df`` does not have.

    ``want`` (iterable or callable) limits the columns, as ``usecols`` does for the file columns.
    """
    p = Path(proc_path)
    vals = load_frame_table(p.parent).get(p.name)
    if not vals:
        return df
    for col, v in vals.items():
        if col in df.columns:
            continue
        if want is not None and not (want(col) if callable(want) else col in want):
            continue
        df[col] = v
    return df


def read_proc_csv(path: Path | str, *, usecols=None, reader=None, **kwargs) -> pd.DataFrame:
    """The one reader of proc CSVs: the file plus its frame values from ``frames.csv``.

    ``usecols`` works as in pandas (a list raises ValueError for a column neither the file nor the
    frame table has; a callable selects). ``reader`` defaults to ``pandas.read_csv``. Id columns are always
    read as text (``VYVAR_CSV_DTYPE``).
    """
    from gaia_catalog_id import VYVAR_CSV_DTYPE  # noqa: PLC0415

    rd = reader or pd.read_csv
    p = Path(path)
    kwargs["dtype"] = {**VYVAR_CSV_DTYPE, **(kwargs.get("dtype") or {})}  # ids always as text (D06)
    if usecols is None:
        df = rd(p, **kwargs)
        return attach_frame_columns(df, p)
    sel = usecols if callable(usecols) else set(usecols).__contains__
    # The row count comes from the file: read catalog_id too when only frame values are asked for.
    df = rd(p, usecols=lambda c: bool(sel(c)) or c == "catalog_id", **kwargs)
    df = attach_frame_columns(df, p, want=sel)
    if "catalog_id" in df.columns and not sel("catalog_id"):
        df = df.drop(columns=["catalog_id"])
    if not callable(usecols):
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise ValueError(f"Usecols do not match columns, columns expected but not found: {missing}")
    return df


def proc_csv_columns(path: Path | str) -> list[str]:
    """Header of a proc CSV including the frame-table columns (``nrows=0`` sniff replacement)."""
    return list(read_proc_csv(path, nrows=0).columns)


def strip_frame_columns(df: pd.DataFrame, proc_path: Path | str) -> pd.DataFrame:
    """``df`` without the frame columns when ``frames.csv`` holds them for this file."""
    p = Path(proc_path)
    if not load_frame_table(p.parent).get(p.name):
        return df
    return df.drop(columns=[c for c in PROC_FRAME_COLS if c in df.columns])


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
    "dao_flux",
    *FLUX_K_COLUMNS,
    "sky_adu_per_px_annulus",
    "sigma_bkg_ap",
    "aperture_r_px",
    "fwhm_px_for_aperture",
    "peak_max_adu",
    "saturate_limit_adu_85pct",
    "x",
    "y",
    "ra_deg",
    "dec_deg",
    "sky_annulus_r_out_px",
    "airmass",
    "mag",
    "bp_rp",
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
        self.n_frames: int = 0
        self.n_stars_median: float = 0.0

    @classmethod
    def build(
        cls,
        proc_csv_dir: Path,
        *,
        glob_pattern: str = PROC_CSV_GLOB,
        extra_cols: list[str] | None = None,
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

        row_counts: list[int] = []
        n_loaded = 0
        n_failed = 0

        for path in paths:
            key = str(path)
            try:
                df = read_proc_csv(
                    path,
                    usecols=lambda c: c in want_cols,
                    dtype={k: v for k, v in _dtypes.items()},
                    low_memory=False,
                )

                if "catalog_id" in df.columns:
                    df["catalog_id"] = normalize_gaia_source_id_series(df["catalog_id"])

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


# --- moved from pipeline_astrometry (LAYER-2026-10) ---


_VYVAR_TIME_JD_CSV_COLS = ("jd_mid", "hjd_mid", "bjd_tdb_mid")


def _vyvar_df_round_time_jd_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Round geocentric/heliocentric/barycentric JD columns to six decimals for stable CSV / spreadsheet display."""
    cols = [c for c in _VYVAR_TIME_JD_CSV_COLS if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def _vyvar_df_to_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Write sidecar / per-frame proc CSV.

    When ``mag`` is present it holds the Gaia catalog G magnitude (``phot_g_mean_mag`` at
    cross-match time) - constant per star for the night, not the frame instrumental magnitude.
    Science paths must use ``dao_flux`` (see ``docs/VYVAR_PROCESS.md``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    export_df = df
    if "catalog_id" in df.columns:
        export_df = df.copy()
        export_df["catalog_id"] = catalog_id_series_for_masterstars_export(df)
    export_df = _vyvar_df_round_time_jd_for_csv(export_df)
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.csv as pacsv  # type: ignore[import-not-found]

        pacsv.write_csv(pa.Table.from_pandas(export_df, preserve_index=False), str(p))
    except Exception:  # noqa: BLE001
        export_df.to_csv(p, index=False, lineterminator="\n", na_rep="")


# --- moved from pipeline_astrometry (LAYER-2026-10) ---


def _safe_proc_name(original_name: str) -> str:
    name = (original_name or "").strip()
    if not name:
        return "proc.fits"
    if name.lower().startswith("proc_"):
        return name
    return "proc_" + name


# --- moved from epsf_psf_merge (LAYER-2026-10) ---


