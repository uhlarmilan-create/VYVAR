"""Moved from photometry_core.py (CONSOLIDATE-01E2). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import math
import numpy as np
import pandas as pd
from aperture_pertarget import flux_k_column  # noqa: F401  (re-exported: one definition)
from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event
from utils import plate_scale_arcsec_per_pixel

from photometry_core import (
    BKG_SCALE_R_CLAMP_HI,
    BKG_SCALE_R_CLAMP_LO,
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    ERR_BKG_SOURCE_HOWELL_FALLBACK,
    ERR_BKG_SOURCE_HOWELL_SCALED,
    LOGGER,
    SIGMA_BKG_AP_COL,
    SKY_ADU_PER_PX_ANNULUS_COL,
    SKY_SURFACE_BG_MEDIAN_ADU_COL,
)


def _intersection_bbox_from_frame_bboxes(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if len(bboxes) < 2:
        return None
    x0_i, y0_i = 0.0, 0.0
    x1_i, y1_i = float("inf"), float("inf")
    for x0, y0, x1, y1 in bboxes:
        x0_i = max(x0_i, x0)
        y0_i = max(y0_i, y0)
        x1_i = min(x1_i, x1)
        y1_i = min(y1_i, y1)
    if not (math.isfinite(x0_i) and math.isfinite(y0_i) and math.isfinite(x1_i) and math.isfinite(y1_i)):
        return None
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return (x0_i, y0_i, x1_i, y1_i)


def _aperture_flux_sky_batch(
    d: np.ndarray,
    pos: np.ndarray,
    r_ap: float | np.ndarray,
    r_in: float | np.ndarray,
    r_out: float | np.ndarray,
    *,
    sky: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sky-subtracted circular aperture sums (photutils ``method='exact'``).

    Shared by production per-star photometry and the scatter/CoG radius ladder
    (IMPL-04). When all stars share one radius, uses one batch aperture call.
    ``sky``: the image for the annulus medians (``sky_estimation.sky_image``: NaN on detected
    sources, SKY-SRC-MASK-2026-10); default ``d``.
    """
    from photutils.aperture import CircularAperture
    from photutils.aperture import aperture_photometry as _aphot
    from photometry_gate_helpers import _aperture_flux_sky_per_star  # noqa: PLC0415

    d_sky = d if sky is None else sky

    pos = np.asarray(pos, dtype=np.float64)
    n = int(pos.shape[0])
    flux_arr = np.full(n, np.nan, dtype=np.float64)
    sky_pp_arr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return flux_arr, sky_pp_arr

    r_ap_arr = np.full(n, float(r_ap), dtype=np.float64) if np.isscalar(r_ap) else np.asarray(r_ap, dtype=np.float64)
    r_in_arr = np.full(n, float(r_in), dtype=np.float64) if np.isscalar(r_in) else np.asarray(r_in, dtype=np.float64)
    r_out_arr = np.full(n, float(r_out), dtype=np.float64) if np.isscalar(r_out) else np.asarray(r_out, dtype=np.float64)

    # Uniform radius: one batch exact sum.
    if (
        np.all(np.isfinite(r_ap_arr))
        and np.all(r_ap_arr > 0)
        and float(np.nanmax(r_ap_arr) - np.nanmin(r_ap_arr)) < 1e-9
        and np.all(np.isfinite(r_in_arr))
        and np.all(np.isfinite(r_out_arr))
        and float(np.nanmax(r_in_arr) - np.nanmin(r_in_arr)) < 1e-9
        and float(np.nanmax(r_out_arr) - np.nanmin(r_out_arr)) < 1e-9
        and float(r_out_arr[0]) > float(r_in_arr[0]) > 0
    ):
        r0 = float(r_ap_arr[0])
        rin = float(r_in_arr[0])
        rout = float(r_out_arr[0])
        try:
            ap = CircularAperture(pos, r=r0)
            phot = _aphot(d, ap, method="exact")
            sums = np.asarray(phot["aperture_sum"], dtype=np.float64)
            area = float(ap.area)
            from sky_estimation import annulus_sky_medians  # noqa: PLC0415

            sky_pp_arr = annulus_sky_medians(d_sky, pos, rin, rout)
            flux_arr = sums - sky_pp_arr * area
            return flux_arr, sky_pp_arr
        except Exception as exc:  # noqa: BLE001
            logging.debug("[PHOT] batch exact aperture failed, falling back per-star: %s", exc)

    # Heterogeneous radii: per-star exact (photutils 2.3 scalar-r constraint).
    return _aperture_flux_sky_per_star(d, pos, r_ap_arr, r_in_arr, r_out_arr, sky=d_sky)


def _safe_polyfit(
    x: np.ndarray,
    y: np.ndarray,
    deg: int,
    *,
    cov: bool = False,
) -> np.ndarray | tuple[np.ndarray, Any] | None:
    """``np.polyfit`` that returns ``None`` when the fit is underdetermined or degenerate."""
    if deg < 0:
        return None
    x_a = np.asarray(x, dtype=np.float64)
    y_a = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x_a) & np.isfinite(y_a)
    x_a = x_a[ok]
    y_a = y_a[ok]
    if x_a.size < deg + 1 or y_a.size < deg + 1:
        return None
    if float(np.ptp(x_a)) == 0.0:
        return None
    try:
        if cov:
            return np.polyfit(x_a, y_a, int(deg), cov=True)
        return np.polyfit(x_a, y_a, int(deg))
    except Exception:  # noqa: BLE001
        # EXC-0120: T4 -- Polyfit failure returns None - callers skip that detrend model branch (EXCEPT-BULK 2026-07-08)
        return None

def _normalize_gaia_id(x: Any) -> str:
    """Gaia ``source_id`` key for joins; delegates to ``normalize_gaia_source_id`` (legacy bool/``none`` guards)."""
    if isinstance(x, (bool, np.bool_)):
        return ""
    out = normalize_gaia_source_id(x)
    if out.lower() == "none":
        return ""
    return out

def _col_values(df: pd.DataFrame, col: str, missing: Any) -> list[Any]:
    """``row.get(col, missing)`` for every row, as a list."""
    return df[col].tolist() if col in df.columns else [missing] * len(df)


def _col_floats(df: pd.DataFrame, col: str, n: int) -> list[float]:
    """``float(pd.to_numeric(row.get(col), errors="coerce"))`` for every row (same per-value result)."""
    if col not in df.columns:
        return [float(pd.to_numeric(None, errors="coerce"))] * n
    c = df[col]
    if pd.api.types.is_float_dtype(c.dtype) or pd.api.types.is_integer_dtype(c.dtype):
        return [float(v) for v in c.tolist()]
    return [float(pd.to_numeric(v, errors="coerce")) for v in c.tolist()]


def _sky_pp_from_values(pre_sub: float, ann: float) -> float:
    """``_sky_pp_for_photometric_error`` from the two column values."""
    if math.isfinite(pre_sub) and pre_sub >= 0:
        return pre_sub
    if math.isfinite(ann) and ann >= 0:
        return ann
    return 0.0


def finalize_hybrid_bkg_fallback_proc_dir(
    proc_dir: Path,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
    setup_label: str = "",
) -> dict[str, Any]:
    """Post-pass: replace raw Howell fallback rows with setup-calibrated ``howell_scaled``.

    ``r_setup`` = median over rows with empirical ``sigma_bkg_ap`` of
    ``sigma_bkg_ap^2 / (A.sky/g + A.(RN/g)^2)``.  Raw ``howell_fallback`` remains only when
    no empirical frames exist in the setup (Casertano et al. 2000 transferred correction).
    """
    from infolog import log_event

    proc_dir = Path(proc_dir)
    ratios: list[float] = []
    files = sorted(proc_dir.glob("proc_*.csv"))
    # SPEED-2026-10: column lists instead of iterrows (same per-value conversions); tables that hold
    # fallback rows are kept for the second pass instead of being read again.
    _kept: dict[Path, pd.DataFrame] = {}
    for proc_path in files:
        try:
            from proc_frame_store import read_proc_csv  # noqa: PLC0415

            df = read_proc_csv(proc_path, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        n = len(df)
        _src = [str(v).strip() for v in _col_values(df, ERR_BKG_SOURCE_COL, "")]
        if any(v == ERR_BKG_SOURCE_HOWELL_FALLBACK for v in _src):
            _kept[proc_path] = df
        _emp = [i for i, v in enumerate(_src) if v == ERR_BKG_SOURCE_EMPIRICAL]
        if not _emp:
            continue
        _sig = _col_floats(df, SIGMA_BKG_AP_COL, n)
        _pre = _col_floats(df, SKY_SURFACE_BG_MEDIAN_ADU_COL, n)
        _ann = _col_floats(df, SKY_ADU_PER_PX_ANNULUS_COL, n)
        _area = _col_floats(df, "aperture_area_px", n)
        _rap = _col_floats(df, "aperture_r_px", n)
        for i in _emp:
            sky = _sky_pp_from_values(_pre[i], _ann[i])
            area = _area[i]
            if not math.isfinite(area) or area <= 0:
                r_ap = _rap[i]
                area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
            rv = bkg_scale_ratio_empirical_over_howell(_sig[i], sky, area, gain=gain, read_noise=read_noise)
            if math.isfinite(rv) and rv > 0:
                ratios.append(rv)

    r_setup, n_ratios = compute_setup_bkg_scale_r(ratios)
    stats: dict[str, Any] = {
        "setup": setup_label or str(proc_dir),
        "n_ratio_samples": n_ratios,
        "r_setup": float(r_setup) if math.isfinite(r_setup) else None,
        "n_files": len(files),
        "n_scaled_rows": 0,
        "n_raw_fallback_rows": 0,
    }
    if not math.isfinite(r_setup):
        return stats

    if not hasattr(finalize_hybrid_bkg_fallback_proc_dir, "_logged_setups"):
        finalize_hybrid_bkg_fallback_proc_dir._logged_setups = set()  # type: ignore[attr-defined]
    _logged: set[str] = finalize_hybrid_bkg_fallback_proc_dir._logged_setups  # type: ignore[attr-defined]
    _key = setup_label or str(proc_dir.resolve())
    if _key not in _logged:
        log_event(
            f"[PHOT] err_bkg howell_scaled setup={_key} r_setup={r_setup:.4f} "
            f"(n_empirical_ratios={n_ratios}, clamp=[{BKG_SCALE_R_CLAMP_LO},{BKG_SCALE_R_CLAMP_HI}])"
        )
        _logged.add(_key)

    for proc_path in files:
        df = _kept.get(proc_path)
        if df is None or df.empty or ERR_BKG_SOURCE_COL not in df.columns:
            continue
        changed = False
        src_col = df[ERR_BKG_SOURCE_COL].astype(str)
        for i in range(len(df)):
            if src_col.iloc[i] != ERR_BKG_SOURCE_HOWELL_FALLBACK:
                continue
            row = df.iloc[i]
            sky = _sky_pp_for_photometric_error(row)
            area = float(pd.to_numeric(row.get("aperture_area_px"), errors="coerce"))
            if not math.isfinite(area) or area <= 0:
                r_ap = float(pd.to_numeric(row.get("aperture_r_px"), errors="coerce"))
                area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
            sig_scaled = scaled_sigma_bkg_ap_from_howell(
                sky, area, gain=gain, read_noise=read_noise, r_setup=r_setup
            )
            if math.isfinite(sig_scaled) and sig_scaled >= 0:
                df.at[df.index[i], SIGMA_BKG_AP_COL] = sig_scaled
                df.at[df.index[i], ERR_BKG_SOURCE_COL] = ERR_BKG_SOURCE_HOWELL_SCALED
                stats["n_scaled_rows"] += 1
                changed = True
            else:
                stats["n_raw_fallback_rows"] += 1
        if changed:
            from proc_frame_store import write_proc_csv  # noqa: PLC0415

            write_proc_csv(df, proc_path)
    return stats

def stamp_masterstar_snr_columns(
    df: pd.DataFrame,
    *,
    image: np.ndarray | None,
    fwhm_dao_px: float,
    bg_sigma_adu: float,
    gain: float = 1.0,
    aperture_fwhm_factor: float = 1.9,
    annulus_inner_fwhm: float = 4.75,
    annulus_outer_fwhm: float = 9.0,
) -> pd.DataFrame:
    """Stamp MASTERSTAR ``snr_ap_pixscaled`` = flux_ap/err_ap_pixscaled and ``snr_peak``.

    Columns used:
    - ``flux_ap``: CircularAperture+annulus on ``image`` when provided and (x,y)
      are finite; otherwise the existing ``flux`` column (DAO/aperture flux
      already in the table).
    - ``err_ap``: pixel-scaled estimate ``sqrt(F/g + (sigma_pix * sqrt(pi r^2))^2)``
      (``_photometric_error_with_bkg_mode``). This is **not** the production
      empty-aperture ``sigma_bkg_ap`` measurement. D3 gates on this estimate
      (floor 10); do not quote it as the empirical error.
    - ``snr_ap_pixscaled``: ``flux_ap / err_ap`` (gated by D3).
    - ``snr_peak``: ``peak_dao / bg_sigma_adu`` (diagnostic; not gated).
    """
    out = df.copy()
    n = int(len(out))
    sig = float(bg_sigma_adu) if math.isfinite(float(bg_sigma_adu)) else 1.0
    sig = max(sig, 1e-6)
    if "peak_dao" in out.columns:
        peak = pd.to_numeric(out["peak_dao"], errors="coerce")
    else:
        peak = pd.Series(np.full(n, np.nan, dtype=np.float64), index=out.index)
    out["snr_peak"] = peak / sig

    from aperture_policy import resolve_aperture_geometry

    fw = max(0.5, float(fwhm_dao_px)) if math.isfinite(float(fwhm_dao_px)) else 2.5
    fac = float(aperture_fwhm_factor) if math.isfinite(float(aperture_fwhm_factor)) and aperture_fwhm_factor > 0 else 1.9
    r_ap, r_in, r_out = resolve_aperture_geometry(
        f=fac,
        fwhm_px=fw,
        annulus_inner_fwhm=float(annulus_inner_fwhm),
        annulus_outer_fwhm=float(annulus_outer_fwhm),
    )
    area = math.pi * r_ap * r_ap
    sigma_bkg_ap = sig * math.sqrt(area)
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0

    flux_ap = np.full(n, np.nan, dtype=np.float64)
    if "flux" in out.columns:
        flux_ap = pd.to_numeric(out["flux"], errors="coerce").to_numpy(dtype=np.float64)

    if image is not None and n > 0 and "x" in out.columns and "y" in out.columns:
        try:
            xx = pd.to_numeric(out["x"], errors="coerce").to_numpy(dtype=np.float64)
            yy = pd.to_numeric(out["y"], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(xx) & np.isfinite(yy)
            if int(np.count_nonzero(ok)) > 0:
                d = np.asarray(image, dtype=np.float64)
                pos = np.column_stack([xx[ok], yy[ok]])
                from sky_estimation import sky_image  # noqa: PLC0415

                flux_m, _sky = _aperture_flux_sky_batch(d, pos, r_ap, r_in, r_out, sky=sky_image(d, fw))
                flux_ap[ok] = flux_m
        except Exception:  # noqa: BLE001
            if "flux" in out.columns:
                flux_ap = pd.to_numeric(out["flux"], errors="coerce").to_numpy(dtype=np.float64)

    f = np.asarray(flux_ap, dtype=np.float64)
    var = np.full(n, np.nan, dtype=np.float64)
    good = np.isfinite(f) & (f > 0)
    var[good] = f[good] / g + sigma_bkg_ap * sigma_bkg_ap
    err_ap = np.sqrt(var)
    snr = np.full(n, np.nan, dtype=np.float64)
    ok_err = good & np.isfinite(err_ap) & (err_ap > 0)
    snr[ok_err] = f[ok_err] / err_ap[ok_err]
    out["snr_ap_pixscaled"] = snr
    out["flux_ap"] = flux_ap
    out["err_ap"] = err_ap
    return out

def _target_display_name(row: Any, *, fallback_cid: str = "") -> str:
    """VSX name when present, else Gaia ``catalog_id`` - never the literal ``nan``."""
    if row is None:
        return str(fallback_cid or "").strip() or "unknown"
    for key in ("vsx_name", "name"):
        try:
            v = row.get(key, "")
        except Exception:  # noqa: BLE001
            v = ""
        if v is None:
            continue
        if isinstance(v, float) and not math.isfinite(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none"):
            return s
    cid = str(fallback_cid or "").strip() or _normalize_gaia_id(
        row.get("catalog_id", "") if hasattr(row, "get") else ""
    )
    return cid or "unknown"

def stamp_vsx_known_variable_on_masterstars(
    ms_df: pd.DataFrame,
    variable_targets_df: pd.DataFrame | None,
    *,
    log_fn: Any | None = None,
    positional_fallback_arcsec: float = 8.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set ``vsx_known_variable`` on masterstars by catalog_id join (primary).

    Positional matching is used only for variable-target rows without a Gaia ``catalog_id``.
    """
    from gaia_catalog_id import normalize_gaia_source_id_series

    out = ms_df.copy()
    if "vsx_known_variable" not in out.columns:
        out["vsx_known_variable"] = False
    else:
        out["vsx_known_variable"] = (
            pd.to_numeric(out["vsx_known_variable"], errors="coerce").fillna(0).astype(bool)
        )

    stats = {"id_join": 0, "positional_fallback": 0}
    if variable_targets_df is None or getattr(variable_targets_df, "empty", True):
        return out, stats

    vt = variable_targets_df.copy()
    vt_ids: set[str] = set()
    if "catalog_id" in vt.columns:
        vt_ids = {
            str(x).strip()
            for x in normalize_gaia_source_id_series(vt["catalog_id"]).tolist()
            if str(x).strip()
        }

    if vt_ids:
        ms_cid = normalize_gaia_source_id_series(out.get("catalog_id", pd.Series([""] * len(out))))
        id_hit = ms_cid.isin(vt_ids)
        stats["id_join"] = int(id_hit.sum())
        out.loc[id_hit, "vsx_known_variable"] = True

    vt_no_id = vt
    if "catalog_id" in vt.columns:
        vt_no_id = vt[normalize_gaia_source_id_series(vt["catalog_id"]).eq("")]
    if (
        not vt_no_id.empty
        and "ra_deg" in vt_no_id.columns
        and "dec_deg" in vt_no_id.columns
        and "ra_deg" in out.columns
        and "dec_deg" in out.columns
    ):
        try:
            from astropy.coordinates import SkyCoord  # noqa: PLC0415
            import astropy.units as u  # noqa: PLC0415

            v_ra = pd.to_numeric(vt_no_id["ra_deg"], errors="coerce")
            v_de = pd.to_numeric(vt_no_id["dec_deg"], errors="coerce")
            ok_v = v_ra.notna() & v_de.notna()
            if bool(ok_v.any()):
                ms_ra = pd.to_numeric(out["ra_deg"], errors="coerce")
                ms_de = pd.to_numeric(out["dec_deg"], errors="coerce")
                ok_m = ms_ra.notna() & ms_de.notna()
                if bool(ok_m.any()):
                    ms_coo = SkyCoord(
                        ra=ms_ra.loc[ok_m].astype(float).to_numpy() * u.deg,
                        dec=ms_de.loc[ok_m].astype(float).to_numpy() * u.deg,
                        frame="icrs",
                    )
                    vt_coo = SkyCoord(
                        ra=v_ra.loc[ok_v].astype(float).to_numpy() * u.deg,
                        dec=v_de.loc[ok_v].astype(float).to_numpy() * u.deg,
                        frame="icrs",
                    )
                    _idx, sep2d, _ = ms_coo.match_to_catalog_sky(vt_coo)
                    near = sep2d <= (float(positional_fallback_arcsec) * u.arcsec)
                    pos_idx = out.index[ok_m][np.asarray(near, dtype=bool)]
                    stats["positional_fallback"] = int(len(pos_idx))
                    if len(pos_idx):
                        out.loc[pos_idx, "vsx_known_variable"] = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[PHOT] VSX positional fallback stamp skipped: %s", exc)

    if log_fn is not None:
        log_fn(
            f"MASTERSTAR vsx_known_variable stamp: catalog_id join={stats['id_join']} "
            f"positional_fallback={stats['positional_fallback']}"
        )
    return out, stats

def build_gs11_summary(
    summary_rows: list[dict[str, Any]],
    cfg: Any,
    *,
    plate_scale_arcsec: float = 1.3,
) -> dict[str, Any]:
    """Aggregate GS11 dilution stats for ``pipeline_meta.json``."""
    enabled = bool(cfg.gs11_dilution_enabled)
    from dilution import GS11_TARGET_MIN_DILUTION as min_d  # noqa: PLC0415
    aperture_arcsec = float("nan")
    corrections_mmag: list[float] = []
    targets_corrected = 0
    targets_skipped_low_d = 0
    if enabled:
        ap_samples_gs11: list[float] = []
        for row in summary_rows:
            try:
                d = float(row.get("dilution_factor", 1.0))
            except (TypeError, ValueError):
                d = 1.0
            try:
                dm = float(row.get("dilution_delta_mag", 0.0))
            except (TypeError, ValueError):
                dm = 0.0
            if not math.isfinite(d):
                continue
            try:
                gs11_ap = float(row.get("gs11_aperture_arcsec", float("nan")))
            except (TypeError, ValueError):
                gs11_ap = float("nan")
            if math.isfinite(gs11_ap) and gs11_ap > 0:
                ap_samples_gs11.append(gs11_ap)
            if d < 1.0 and d < min_d:
                targets_skipped_low_d += 1
            elif d < 1.0 and d >= min_d and math.isfinite(dm) and dm > 0:
                targets_corrected += 1
                corrections_mmag.append(float(dm) * 1000.0)
        if not (math.isfinite(aperture_arcsec) and aperture_arcsec > 0) and ap_samples_gs11:
            aperture_arcsec = float(np.median(np.asarray(ap_samples_gs11, dtype=np.float64)))
        if not (math.isfinite(aperture_arcsec) and aperture_arcsec > 0):
            aperture_arcsec = float(plate_scale_arcsec)
    med_mmag = float(np.median(corrections_mmag)) if corrections_mmag else 0.0
    max_mmag = float(np.max(corrections_mmag)) if corrections_mmag else 0.0
    return {
        "enabled": enabled,
        "aperture_arcsec": float(aperture_arcsec) if math.isfinite(aperture_arcsec) else float("nan"),
        "targets_corrected": int(targets_corrected),
        "targets_skipped_low_d": int(targets_skipped_low_d),
        "median_correction_mmag": med_mmag,
        "max_correction_mmag": max_mmag,
    }


def _get_plate_scale_from_cfg(
    cfg: Any,
    db: Any = None,
    draft_id: int | None = None,
    *,
    fits_path: Path | None = None,
    ms_header: Any | None = None,
) -> float | None:
    """
    Plate scale (arcsec/px) for FOV / max_dist_deg.
    Priority:
    1. Solved WCS/CD matrix from the FITS (authoritative)
    2. DB EQUIPMENTS+TELESCOPE+FITS binning (pixel/focal/binning)
    None if unavailable (no config fallback, PLATE-SCALE-2026-10)
    """
    result: float | None = None

    # 1. Authoritative: solved WCS/CD from the frame's FITS.
    if fits_path is not None or ms_header is not None:
        try:
            _fp = Path(fits_path) if fits_path is not None else Path(".")
            _wcs_ps = _read_plate_scale_from_fits_path(_fp, ms_header=ms_header)
        except Exception:  # noqa: BLE001
            _wcs_ps = None
        if _wcs_ps is not None and math.isfinite(float(_wcs_ps)) and float(_wcs_ps) > 0:
            logging.info(
                "[FOV] _get_plate_scale_from_cfg -> %.4f arcsec/px (solved WCS/CD)",
                float(_wcs_ps),
            )
            return float(_wcs_ps)

    # 2. DB: derive plate scale from EQUIPMENTS + TELESCOPE + FITS binning (if available).
    if db is not None and draft_id is not None:
        try:
            did = int(draft_id)
        except Exception:  # noqa: BLE001
            did = 0
        if did > 0:
            try:
                dr = None
                try:
                    dr = db.fetch_obs_draft_by_id(did) if hasattr(db, "fetch_obs_draft_by_id") else None
                except Exception:  # noqa: BLE001
                    dr = None
                id_eq = int(dr.get("ID_EQUIPMENTS") or 0) if isinstance(dr, dict) else 0
                id_tel = int(dr.get("ID_TELESCOPE") or 0) if isinstance(dr, dict) else 0

                binning = 1
                try:
                    light_rows = (
                        db.fetch_draft_light_rows_for_quality(did)
                        if hasattr(db, "fetch_draft_light_rows_for_quality")
                        else []
                    )
                    fp0 = None
                    for lr in light_rows:
                        fp0 = lr.get("FILE_PATH")
                        if fp0:
                            break
                    if fp0:
                        from astropy.io import fits as _fits_bin

                        with _fits_bin.open(str(fp0), memmap=False) as _hdul:
                            _hdr = _hdul[0].header
                            _xb = _hdr.get("XBINNING") or _hdr.get("BINNING")
                            if _xb is not None:
                                b0 = int(float(_xb))
                                if 1 <= b0 <= 16:
                                    binning = b0
                except Exception:  # noqa: BLE001
                    # EXC-0181: T4 -- DB plate-scale lookup failure returns None (EXCEPT-BULK-2 2026-07-08)
                    binning = 1

                pix_um = None
                foc_mm = None
                try:
                    pix_um = (
                        float(db.get_equipment_pixel_size_um(id_eq))
                        if (hasattr(db, "get_equipment_pixel_size_um") and id_eq > 0)
                        else None
                    )
                except Exception:  # noqa: BLE001
                    pix_um = None
                try:
                    foc_mm = (
                        float(db.get_telescope_focal_mm(id_tel if id_tel > 0 else None))
                        if hasattr(db, "get_telescope_focal_mm")
                        else None
                    )
                except Exception:  # noqa: BLE001
                    foc_mm = None

                if (
                    pix_um is not None
                    and foc_mm is not None
                    and math.isfinite(float(pix_um))
                    and float(pix_um) > 0
                    and math.isfinite(float(foc_mm))
                    and float(foc_mm) > 0
                ):
                    eff_um = float(pix_um) * float(max(1, int(binning)))
                    sc = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(eff_um), focal_length_mm=float(foc_mm))
                    if sc is not None and math.isfinite(float(sc)) and float(sc) > 0:
                        result = float(sc)
                        logging.info(
                            "[FOV] _get_plate_scale_from_cfg -> %.4f arcsec/px (DB: eq/tel/bin=%s)",
                            float(result),
                            int(binning),
                        )
                        return result
            except Exception:  # noqa: BLE001
                pass
    # PLATE-SCALE-2026-10: no config value - a fixed 1.3"/px was silently wrong on every rig
    logging.info(
        "[FOV] _get_plate_scale_from_cfg -> %.4f arcsec/px (None = fallback na max_dist_deg)",
        -1.0,
    )
    return None

def _resolve_plate_scale_arcsec_per_px(
    cfg: Any,
    fits_path: Path | None = None,
    *,
    ms_header: Any | None = None,
) -> float | None:
    """Plate scale (arcsec/px) for GS11 + aperture arcsec conversion.

    Solved WCS/CD matrix of the FITS (no config fallback, PLATE-SCALE-2026-10).
    Returns None when nothing derivable (derive-or-None; no magic default).
    Clamp [0.1, 30.0] when a value is returned.
    """
    _lo, _hi = 0.1, 30.0
    _fits_ps: float | None = None
    if fits_path is not None or ms_header is not None:
        try:
            _fp = Path(fits_path) if fits_path is not None else Path(".")
            _fits_ps = _read_plate_scale_from_fits_path(_fp, ms_header=ms_header)
        except Exception:  # noqa: BLE001
            # EXC-0183: T4 -- WCS pixel_scale_from_wcs path fails - CD-matrix fallback attempted next (EXCEPT-BULK 2026-07-08)
            _fits_ps = None
    if _fits_ps is not None and math.isfinite(_fits_ps) and _lo <= float(_fits_ps) <= _hi:
        return float(_fits_ps)
    logging.warning(
        "[PLATE SCALE] plate scale not derivable from the WCS - returning None"
    )
    return None

def _cd_matrix_scale_arcsec_per_px(hdr: Any) -> float | None:
    """Plate scale (arcsec/px) from the SOLVED astrometric WCS (CD/PC matrix -> CDELT).

    This is the authoritative source: it reflects the actual sky-to-pixel solution,
    independent of stale VY_PLTS / config values. Returns None if no usable WCS.
    """
    if hdr is None:
        return None
    # Full WCS handles CD, PC+CDELT, SIP, etc.
    try:
        import warnings  # noqa: PLC0415

        import numpy as _np  # noqa: PLC0415
        from astropy.wcs import WCS as _WCS  # noqa: PLC0415
        from astropy.wcs import FITSFixedWarning as _FFW  # noqa: PLC0415
        from astropy.wcs.utils import proj_plane_pixel_scales as _pps  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _FFW)
            _w = _WCS(hdr)
        if _w.has_celestial:
            sc = float(_np.mean(_pps(_w))) * 3600.0
            if math.isfinite(sc) and 0.01 < sc < 200.0:
                return float(sc)
    except Exception:  # noqa: BLE001
        pass
    # Raw CD matrix fallback.
    try:
        cd11 = hdr.get("CD1_1")
        cd12 = hdr.get("CD1_2", 0.0)
        if cd11 is not None:
            sc = math.sqrt(float(cd11) ** 2 + float(cd12) ** 2) * 3600.0
            if math.isfinite(sc) and 0.01 < sc < 200.0:
                return float(sc)
    except (TypeError, ValueError):
        pass
    # CDELT1 fallback.
    try:
        cdelt1 = hdr.get("CDELT1")
        if cdelt1 is not None:
            sc = abs(float(cdelt1)) * 3600.0
            if math.isfinite(sc) and 0.01 < sc < 200.0:
                return float(sc)
    except (TypeError, ValueError):
        pass
    return None

def _read_plate_scale_from_fits_path(
    fits_path: Path,
    *,
    ms_header: Any | None = None,
) -> float | None:
    """Plate scale (arcsec/px) from FITS, CD/WCS-FIRST.

    Priority: (1) solved WCS/CD matrix; (2) VY_PLTS header - only if it agrees with
    the CD value within 5% (else ignored, logged); (3) other header keywords, only
    when no usable CD/WCS exists. Clamp [0.1, 30.0] (covers fine ~0.3 and wide-field ~10).
    """
    _MIN, _MAX = 0.1, 30.0
    try:
        from astropy.io import fits as astrofits  # noqa: PLC0415

        if ms_header is not None:
            hdr = ms_header
        else:
            fp = Path(fits_path)
            if not fp.is_file():
                return None
            with astrofits.open(fp, memmap=False) as hdul:
                hdr = hdul[0].header
    except Exception as exc:  # noqa: BLE001
        # EXC-0185: T4 -- Header keyword plate-scale scan fails - returns None after trying CD/WCS paths (EXCEPT-BULK-2 2026-07-08)
        logging.error('[EXC-0184] FITS open/header read for plate scale fails - returns None, caller uses config/default ...: %s', exc)
        return None

    # (1) Authoritative: solved WCS / CD matrix.
    cd_scale = _cd_matrix_scale_arcsec_per_px(hdr)
    if cd_scale is not None and _MIN <= cd_scale <= _MAX:
        # (2) Cross-check VY_PLTS: warn and ignore if it disagrees > 5%.
        vy = hdr.get("VY_PLTS")
        try:
            vyf = float(vy) if vy is not None else None
        except (TypeError, ValueError):
            vyf = None
        if vyf is not None and vyf > 0 and abs(vyf - cd_scale) / cd_scale > 0.05:
            logging.warning(
                "[PLATE SCALE] VY_PLTS=%.3f disagrees with CD-derived %.3f arcsec/px (>5%%) - using CD.",
                vyf,
                cd_scale,
            )
        return float(cd_scale)

    # (3) No usable WCS/CD - fall back to header keywords (still header, above config).
    try:
        for key in ("VY_PLTS", "PIXSCALE", "SECPIX", "SECPIX1", "SCALE", "CDELT1"):
            v = hdr.get(key)
            if v is None:
                continue
            try:
                f = float(v)
                if key == "CDELT1":
                    f = abs(f) * 3600.0
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and _MIN <= f <= _MAX:
                return float(f)
    except Exception:  # noqa: BLE001
        # EXC-0186: T4 -- Non-numeric catalog_id string returned unchanged instead of int-normalized form (EXCEPT-BULK 2026-07-08)
        return None
    return None

def _angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Angular distance in degrees (haversine). Same formula as comparison-star selection ``_dist_deg``."""
    from sky_separation import angular_distance_deg  # noqa: PLC0415

    return angular_distance_deg(ra1, dec1, ra2, dec2)


def common_field_intersection_bbox_px_from_headers(
    *,
    headers: list[tuple[Any, tuple[int, int]]],
) -> tuple[float, float, float, float] | None:
    """Overlap (x0, y0, x1, y1) of the data footprints of aligned frames given as
    ``(header, (ny, nx))`` - the RAM-handoff form (BORDER-FOOTPRINT-2026-10)."""
    from frame_footprint import footprint_bbox  # noqa: PLC0415

    return _intersection_bbox_from_frame_bboxes([footprint_bbox(h, shp) for h, shp in headers])


def common_field_intersection_bbox_px(
    *,
    frame_paths: list[Path],
) -> tuple[float, float, float, float] | None:
    """Overlap (x0, y0, x1, y1) of the data footprints of the aligned frames on disk
    (BORDER-FOOTPRINT-2026-10): each frame's footprint from its frame->reference transform
    ``VY_AM0..8`` (headers only). The aligner fills uncovered pixels with a constant, so the pixel
    values cannot show the footprint. Raises on a frame without the transform."""
    from astropy.io import fits  # noqa: PLC0415

    headers = []
    for fp in (Path(p) for p in frame_paths):
        if not fp.is_file():
            continue
        hdr = fits.getheader(fp)
        headers.append((hdr, (int(hdr.get("NAXIS2", 0)), int(hdr.get("NAXIS1", 0)))))
    return common_field_intersection_bbox_px_from_headers(headers=headers)


def _fwhm_moment_at(arr: np.ndarray, xc: float, yc: float, *, half: int = 6) -> float:
    """2D Gaussian moment FWHM estimate (same recipe as pipeline MASTERSTAR block)."""
    if not (math.isfinite(xc) and math.isfinite(yc)):
        return float("nan")
    xi = int(round(float(xc)))
    yi = int(round(float(yc)))
    h, w = int(arr.shape[0]), int(arr.shape[1])
    x0 = max(0, xi - half)
    x1 = min(w - 1, xi + half)
    y0 = max(0, yi - half)
    y1 = min(h - 1, yi + half)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    patch = arr[y0 : y1 + 1, x0 : x1 + 1].astype(np.float64, copy=False)
    if patch.size < 9:
        return float("nan")
    medp = float(np.nanmedian(patch))
    wgt = patch - medp
    wgt[~np.isfinite(wgt)] = 0.0
    wgt[wgt < 0] = 0.0
    s = float(wgt.sum())
    if not math.isfinite(s) or s <= 0:
        return float("nan")
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    mx = float((wgt * xx).sum() / s)
    my = float((wgt * yy).sum() / s)
    vx = float((wgt * (xx - mx) ** 2).sum() / s)
    vy = float((wgt * (yy - my) ** 2).sum() / s)
    if not (vx > 0 and vy > 0 and math.isfinite(vx) and math.isfinite(vy)):
        return float("nan")
    sigx = math.sqrt(vx)
    sigy = math.sqrt(vy)
    fwhm = 2.355 * 0.5 * (sigx + sigy)
    return float(fwhm) if math.isfinite(fwhm) else float("nan")

def compute_fwhm_gaussian_for_aperture_catalog(
    df: pd.DataFrame,
    data: np.ndarray,
    hdr: Any,
    *,
    gaussian_fwhm_px_override: float | None,
    aperture_fwhm_factor: float,
) -> tuple[np.ndarray, float, float]:
    """Return (fwhm_per_row, fwhm_moment_med, fwhm_gaussian).

    Same computation as in ``enhance_catalog_dataframe_aperture``.

    Used in ``pipeline._apply_aperture_catalog_enhancements_from_st`` for the multi-aperture (r_small / r_large),
    so that the radii match the main aperture.
    """
    arr = np.asarray(data, dtype=np.float32)
    x = pd.to_numeric(df.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    n = len(df)
    if n == 0:
        return np.array([], dtype=np.float64), float("nan"), float("nan")

    fwhm_per = np.array(
        [_fwhm_moment_at(arr, float(x[i]), float(y[i])) for i in range(n)],
        dtype=np.float64,
    )

    fwhm_moment_med = float(np.nanmedian(fwhm_per[np.isfinite(fwhm_per) & (fwhm_per > 0)]))
    if not math.isfinite(fwhm_moment_med) or fwhm_moment_med <= 0:
        fwhm_moment_med = float("nan")

    DAO_TO_GAUSSIAN = 1.0 / 1.5  # 0.667 - physically derived, setup-independent
    fwhm_gaussian: float | None = None

    if gaussian_fwhm_px_override is not None:
        try:
            _ov = float(gaussian_fwhm_px_override)
            if math.isfinite(_ov) and 0.5 < _ov < 30.0:
                fwhm_gaussian = _ov
        except (TypeError, ValueError):
            pass

    if fwhm_gaussian is None and hdr is not None:
        try:
            _vy = hdr.get("VY_FWHM", None)
            if _vy is not None:
                _vy_f = float(_vy)
                if math.isfinite(_vy_f) and 0.5 < _vy_f < 30.0:
                    fwhm_gaussian = _vy_f * DAO_TO_GAUSSIAN
                    if not bool(getattr(enhance_catalog_dataframe_aperture, "_did_log_fwhm", False)):
                        logging.info(
                            f"[PHOT] FWHM from VY_FWHM (DAO): {_vy_f:.3f}px x {DAO_TO_GAUSSIAN:.3f} = "
                            f"{float(fwhm_gaussian):.3f}px -> aperture = "
                            f"{float(fwhm_gaussian) * float(aperture_fwhm_factor):.3f}px"
                        )
                        enhance_catalog_dataframe_aperture._did_log_fwhm = True
        except (TypeError, ValueError):
            pass

    if fwhm_gaussian is None:
        if math.isfinite(fwhm_moment_med) and fwhm_moment_med > 0:
            fwhm_gaussian = fwhm_moment_med * 0.619
            if not bool(getattr(enhance_catalog_dataframe_aperture, "_did_log_fwhm", False)):
                logging.info(
                    f"[PHOT] FWHM fallback momentx0.619: {fwhm_gaussian:.3f}px -> "
                    f"aperture = {float(fwhm_gaussian) * float(aperture_fwhm_factor):.3f}px"
                )
                enhance_catalog_dataframe_aperture._did_log_fwhm = True
        else:
            fwhm_gaussian = float("nan")

    r_ap_test = float(aperture_fwhm_factor) * float(fwhm_gaussian) if math.isfinite(float(fwhm_gaussian)) else float("nan")
    if not math.isfinite(r_ap_test) or r_ap_test < 3.0 or r_ap_test > 20.0:
        fwhm_gaussian = float(fwhm_moment_med)
        logging.warning(
            f"[PHOT] Gaussian FWHM fallback to moment: {fwhm_gaussian:.2f}px "
            f"(r_ap={r_ap_test:.2f}px out of range)"
        )

    return fwhm_per, fwhm_moment_med, float(fwhm_gaussian) if math.isfinite(float(fwhm_gaussian)) else float("nan")



def enhance_catalog_dataframe_aperture(
    df: pd.DataFrame,
    data: np.ndarray,
    hdr: Any,
    *,
    aperture_fwhm_factor: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    nonlinearity_peak_percentile: float | None = None,
    nonlinearity_fwhm_ratio: float | None = None,
    gaussian_fwhm_px_override: float | None = None,
    err_empty_apertures_n: int = 64,
    err_empty_apertures_min: int = 16,
    fwhm_frame_px: float | None = None,
    fwhm_night_median_px: float | None = None,
    qc_fwhm_by_name: dict[str, float] | None = None,
    frame_name: str | None = None,
) -> pd.DataFrame:
    """Replace DAO ``flux`` with aperture photometry (APERTURE-01, per-frame QC FWHM); add linearity flags."""
    out = df.copy()
    arr = np.asarray(data, dtype=np.float32)

    x = pd.to_numeric(out.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    n = len(out)
    if n == 0:
        return out

    # Original DAO flux from detect_stars_and_match_catalog (historically in column ``flux``).
    # ``dao_flux``: sky-subtracted flux (after aperture photometry, if enabled).
    if "flux" in out.columns:
        flux_dao = pd.to_numeric(out["flux"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        flux_dao = np.full(n, np.nan, dtype=np.float64)
    if "dao_flux" not in out.columns:
        out["dao_flux"] = flux_dao

    from aperture_policy import (  # noqa: PLC0415
        fwhm_for_radius,
        resolve_aperture_geometry,
        resolve_frame_fwhm_px,
    )

    _ap01_r_ap = _ap01_r_in = _ap01_r_out = None
    _ap01_fwhm_frame = resolve_frame_fwhm_px(
        hdr=hdr,
        frame_name=frame_name,
        qc_fwhm_by_name=qc_fwhm_by_name,
        fwhm_night_median_px=fwhm_night_median_px,
    )
    if fwhm_frame_px is not None:
        try:
            _ff = float(fwhm_frame_px)
            if math.isfinite(_ff) and 0.5 < _ff < 30.0:
                _ap01_fwhm_frame = _ff
        except (TypeError, ValueError):
            pass
    _ap01_fwhm_used = fwhm_for_radius(
        fwhm_frame_px=_ap01_fwhm_frame,
        fwhm_night_median_px=fwhm_night_median_px,
    )
    if _ap01_fwhm_used is None:
        logging.warning("[APERTURE-01] QC/VY_FWHM missing; skip gauss fallback (FWHM-AUTH-01)")
    if _ap01_fwhm_used is not None:
        _ap01_r_ap, _ap01_r_in, _ap01_r_out = resolve_aperture_geometry(
            f=float(aperture_fwhm_factor),
            fwhm_px=float(_ap01_fwhm_used),
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
        )

    _ap01_ok = (
        _ap01_fwhm_used is not None
        and math.isfinite(float(_ap01_fwhm_used))
        and float(_ap01_fwhm_used) > 0
    )
    # SPEED-2026-10: the per-star moment FWHM (4-5k stars per frame) only feeds the fallback
    # radius when the frame has no QC / VY_FWHM; with one it was computed and discarded.
    fwhm_gaussian_f = float("nan")
    if not _ap01_ok:
        _fwhm_per, _fwhm_moment_med, fwhm_gaussian_f = compute_fwhm_gaussian_for_aperture_catalog(
            out,
            arr,
            hdr,
            gaussian_fwhm_px_override=gaussian_fwhm_px_override,
            aperture_fwhm_factor=aperture_fwhm_factor,
        )
    if (
        _ap01_ok
        or (math.isfinite(float(fwhm_gaussian_f)) and float(fwhm_gaussian_f) > 0)
    ):
        try:
            # Local implementation: sky-subtracted flux via CircularAperture + annulus sky.
            from photutils.aperture import CircularAperture
            from photutils.aperture import aperture_photometry as _aphot
            from aperture_policy import resolve_aperture_geometry  # noqa: PLC0415

            fw = float(fwhm_gaussian_f)
            global_aperture_r_px = max(0.5, float(aperture_fwhm_factor) * fw)
            if _ap01_r_ap is not None:
                global_aperture_r_px = float(_ap01_r_ap)
                fw = float(_ap01_fwhm_used) if _ap01_fwhm_used is not None else fw

            pos = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])

            r_ap = global_aperture_r_px
            if _ap01_r_in is not None and _ap01_r_out is not None:
                r_in = float(_ap01_r_in)
                r_out = float(_ap01_r_out)
            else:
                r_ap, r_in, r_out = resolve_aperture_geometry(
                    f=float(aperture_fwhm_factor),
                    fwhm_px=fw,
                    annulus_inner_fwhm=float(annulus_inner_fwhm),
                    annulus_outer_fwhm=float(annulus_outer_fwhm),
                )

            d = np.asarray(arr, dtype=np.float64)
            if np.any(~np.isfinite(d)):
                fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
                d = np.where(np.isfinite(d), d, fill)

            # AP-FOOTPRINT-2026-10: the annulus sky uses only pixels the aligned frame covers (the
            # aligner fills the rest of the reference grid with a constant); a frame without the
            # frame->reference transform was not resampled.
            from frame_footprint import aligned_footprint, has_alignment_transform  # noqa: PLC0415

            # SKY-SRC-MASK-2026-10: and without the pixels of detected sources (crowded-field sky bias).
            from sky_estimation import sky_image  # noqa: PLC0415

            d_sky = sky_image(d, fw, covered=aligned_footprint(hdr, d.shape) if has_alignment_transform(hdr) else None)

            # A row without a finite position is not measured (NaN), it does not abort the frame:
            # one such row used to throw away the aperture photometry of every star in the frame.
            _fin = np.isfinite(pos).all(axis=1)
            _ix = np.nonzero(_fin)[0]
            sum_ap = np.full(n, np.nan, dtype=np.float64)
            sky_pp_arr = np.full(n, np.nan, dtype=np.float64)
            area_ap_per = float(math.pi * float(r_ap) ** 2)
            if _ix.size:
                ap = CircularAperture(pos[_ix], r=r_ap)
                phot_ap = _aphot(d, ap, method="exact")
                sum_ap[_ix] = np.asarray(phot_ap["aperture_sum"], dtype=np.float64)
                from sky_estimation import annulus_sky_medians  # noqa: PLC0415

                sky_pp_arr[_ix] = annulus_sky_medians(d_sky, pos[_ix], r_in, r_out)
            flux_arr = sum_ap - sky_pp_arr * area_ap_per
            # APERTURE-K-2026-10: the same photometry on the aperture grid r = k x FWHM
            # (same positions, same annulus sky), so comparison-star RMS in comparison-star selection is
            # measured on the apertures the light curves use.
            try:
                from aperture_pertarget import DEFAULT_APERTURE_F_GRID  # noqa: PLC0415

                for _k in DEFAULT_APERTURE_F_GRID:
                    _r_k = max(0.5, float(_k) * float(fw))
                    _sum_k = np.full(n, np.nan, dtype=np.float64)
                    if _ix.size:
                        _sum_k[_ix] = np.asarray(
                            _aphot(d, CircularAperture(pos[_ix], r=_r_k), method="exact")["aperture_sum"],
                            dtype=np.float64,
                        )
                    out[flux_k_column(float(_k))] = _sum_k - sky_pp_arr * float(math.pi * _r_k ** 2)
            except Exception as _fk_exc:  # noqa: BLE001
                logging.warning("[APERTURE-K] grid fluxes not written: %s", _fk_exc)
            out["flux"] = flux_arr.astype(np.float64)
            out["dao_flux"] = out["flux"]
            out["aperture_r_px"] = float(r_ap)
            out["fwhm_px_for_aperture"] = float(fw)
            out["sky_annulus_r_out_px"] = float(r_out)
            out[SKY_ADU_PER_PX_ANNULUS_COL] = sky_pp_arr.astype(np.float64)

            _stamp_fwhm_frame = (
                float(_ap01_fwhm_frame)
                if _ap01_fwhm_frame is not None
                else float("nan")
            )
            out["fwhm_px_for_aperture"] = _stamp_fwhm_frame
            if _ap01_r_out is not None:
                out["sky_annulus_r_out_px"] = float(_ap01_r_out)

            # F-BINGAIN-1: per-frame empirical background noise at production aperture radii.
            # CONSOLIDATE-01D: howell-only skip (no empty-aperture measurement) deleted.
            _n_empty = _clamp_err_empty_apertures_n(err_empty_apertures_n)
            _n_empty_min = _clamp_err_empty_apertures_min(err_empty_apertures_min)
            _sigma_by_r: dict[float, tuple[float, str]] = {}
            _unique_r = np.array([float(r_ap)], dtype=np.float64)
            for _r_u in _unique_r:
                if not math.isfinite(float(_r_u)) or float(_r_u) <= 0:
                    continue
                _ri = float(r_in)
                _ro = float(r_out)
                _seed = _labbe_content_seed_from_header(hdr, r_ap=float(_r_u))
                _frame_id = str(
                    hdr.get("DATE-OBS")
                    or hdr.get("FILENAME")
                    or hdr.get("FRAME")
                    or ""
                )
                _sig, _nv, _reason = measure_empty_aperture_sigma_bkg(
                    d,
                    np.asarray(x, dtype=np.float64),
                    np.asarray(y, dtype=np.float64),
                    float(_r_u),
                    float(_ri),
                    float(_ro),
                    n_apertures=_n_empty,
                    min_valid=_n_empty_min,
                    seed=int(_seed),
                    frame_id=_frame_id,
                    star_list_source="catalog_df_in_memory",
                )
                if not hasattr(enhance_catalog_dataframe_aperture, "_labbe_seeds"):
                    enhance_catalog_dataframe_aperture._labbe_seeds = []
                enhance_catalog_dataframe_aperture._labbe_seeds.append(
                    {"r_ap": float(_r_u), "seed": int(_seed), "n_valid": int(_nv)}
                )
                if math.isfinite(_sig) and _sig >= 0:
                    _sigma_by_r[_sigma_bkg_r_key(_r_u)] = (float(_sig), ERR_BKG_SOURCE_EMPIRICAL)
                else:
                    _sigma_by_r[_sigma_bkg_r_key(_r_u)] = (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK)
                    if not hasattr(enhance_catalog_dataframe_aperture, "_err_bkg_logged"):
                        enhance_catalog_dataframe_aperture._err_bkg_logged = set()
                    _log_id = str(hdr.get("FRAME") or hdr.get("VY_FRAME") or id(hdr))
                    if _log_id not in enhance_catalog_dataframe_aperture._err_bkg_logged:
                        log_event(
                            f"[PHOT] err_bkg empirical fallback (howell): r_ap={float(_r_u):.2f}px "
                            f"n_valid={_nv} reason={_reason or 'unknown'}"
                        )
                        enhance_catalog_dataframe_aperture._err_bkg_logged.add(_log_id)

            _sigma_col = np.full(n, np.nan, dtype=np.float64)
            _src_col = np.full(n, ERR_BKG_SOURCE_HOWELL_FALLBACK, dtype=object)
            _sig_v, _src_v = _sigma_by_r.get(
                _sigma_bkg_r_key(float(r_ap)),
                (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK),
            )
            _sigma_col[:] = _sig_v
            _src_col[:] = _src_v
            _assert_inv_err_sigma_acct_01(
                _sigma_by_r,
                _src_col,
                n=n,
                r_ap_arr=None,
                r_ap=float(r_ap),
            )
            out[SIGMA_BKG_AP_COL] = _sigma_col
            out[ERR_BKG_SOURCE_COL] = _src_col

        except Exception as _ap_exc:  # noqa: BLE001
            logging.warning(
                "[APERTURE] Aperture photometry failed - restoring pre-aperture flux: %s",
                _ap_exc,
                exc_info=True,
            )
            out["dao_flux"] = flux_dao
            out["flux"] = flux_dao
    else:
        out["dao_flux"] = flux_dao
        out["flux"] = flux_dao

    # SAT-LIN-2026-10: non-linear == above LINEARITY_FRAC x saturation (likely_saturated);
    # the former "top-20 % peak and FWHM > 1.25 x median" heuristic is removed.
    if "likely_saturated" in out.columns:
        likely_nl = out["likely_saturated"].fillna(False).astype(bool).to_numpy()
    else:
        likely_nl = np.zeros(n, dtype=bool)


    if "photometry_ok" in out.columns:
        base_ok = out["photometry_ok"].fillna(True).astype(bool).to_numpy()
        out["photometry_ok"] = base_ok & (~likely_nl)
    else:
        out["photometry_ok"] = ~likely_nl

    if "source_type" in out.columns and "dao_flux" in out.columns:
        _forced_mask = (
            out["source_type"].fillna("").astype(str).str.strip().eq("FORCED_APERTURE")
        )
        _dao_num = pd.to_numeric(out["dao_flux"], errors="coerce")
        _has_flux = _dao_num.notna() & (_dao_num != 0)
        out.loc[_forced_mask & _has_flux, "photometry_ok"] = True

    return out


from photometry_gate_helpers import (  # noqa: E402
    _assert_inv_err_sigma_acct_01,
    _clamp_err_empty_apertures_min,
    _clamp_err_empty_apertures_n,
    _labbe_content_seed_from_header,
    _sigma_bkg_r_key,
    _sky_pp_from_annulus_mask,
    bkg_scale_ratio_empirical_over_howell,
    compute_setup_bkg_scale_r,
    measure_empty_aperture_sigma_bkg,
    scaled_sigma_bkg_ap_from_howell,
)
from aperture_photometry import _sky_pp_for_photometric_error  # noqa: E402
