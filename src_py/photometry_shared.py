"""Moved from photometry_core.py (CONSOLIDATE-01E2). Facade re-exports these names."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging
import math
import random
import numpy as np
import pandas as pd
from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event
from proc_frame_store import proc_csv_path_for_aligned_fits
from unit_resolver import resolve_px_from_fwhm_factor
from utils import (
    iter_fits_paths_recursive as _iter_fits_recursive,
    fits_binning_xy_from_header,
    plate_scale_arcsec_per_pixel,
)

from photometry_core import (
    BKG_SCALE_R_CLAMP_HI,
    BKG_SCALE_R_CLAMP_LO,
    ERR_BKG_MODE_EMPIRICAL,
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    ERR_BKG_SOURCE_HOWELL_FALLBACK,
    ERR_BKG_SOURCE_HOWELL_SCALED,
    LOGGER,
    SIGMA_BKG_AP_COL,
    SKY_ADU_PER_PX_ANNULUS_COL,
    _GAIA_ID_DTYPE,
    _aperture_flux_sky_batch,
    _clamp_err_empty_apertures_min,
    _coerce_bool_cell,
    _finite_pixel_bbox_from_array,
    _intersection_bbox_from_frame_bboxes,
    _sky_pp_for_photometric_error,
    compute_per_frame_cog_correction,
)

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
    for proc_path in files:
        try:
            df = pd.read_csv(proc_path, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            src = str(row.get(ERR_BKG_SOURCE_COL, "")).strip()
            if src != ERR_BKG_SOURCE_EMPIRICAL:
                continue
            sig = float(pd.to_numeric(row.get(SIGMA_BKG_AP_COL), errors="coerce"))
            sky = _sky_pp_for_photometric_error(row)
            area = float(pd.to_numeric(row.get("aperture_area_px"), errors="coerce"))
            if not math.isfinite(area) or area <= 0:
                r_ap = float(pd.to_numeric(row.get("aperture_r_px"), errors="coerce"))
                area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
            rv = bkg_scale_ratio_empirical_over_howell(sig, sky, area, gain=gain, read_noise=read_noise)
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
        try:
            df = pd.read_csv(proc_path, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or ERR_BKG_SOURCE_COL not in df.columns:
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
            df.to_csv(proc_path, index=False)
    return stats

def stamp_masterstar_snr_columns(
    df: pd.DataFrame,
    *,
    image: np.ndarray | None,
    fwhm_dao_px: float,
    bg_sigma_adu: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
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
                flux_m, _sky = _aperture_flux_sky_batch(d, pos, r_ap, r_in, r_out)
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
    comps_gs11_rejected: int = 0,
    plate_scale_arcsec: float = 1.3,
) -> dict[str, Any]:
    """Aggregate GS11 dilution stats for ``pipeline_meta.json``."""
    enabled = bool(cfg.gs11_dilution_enabled)
    min_d = float(cfg.gs11_target_min_dilution)
    ap_cfg = float(cfg.gs11_dilution_aperture_arcsec)
    aperture_arcsec = ap_cfg if math.isfinite(ap_cfg) and ap_cfg > 0 else float("nan")
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
        "comps_gs11_rejected": int(comps_gs11_rejected),
        "targets_corrected": int(targets_corrected),
        "targets_skipped_low_d": int(targets_skipped_low_d),
        "median_correction_mmag": med_mmag,
        "max_correction_mmag": max_mmag,
    }

def _get_lc_adaptive(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    """LC mag_inst series using the per-frame ``lc_flux_method`` column (b.4 adaptive).

    Requires ``compute_lc_flux_method`` to have populated ``lc_flux_method``. Falls back to
    aperture (``mag_inst``) for any frame not selected as ``psf`` or with non-finite psf_flux.
    """
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    mag_inst = sub["mag_inst"].to_numpy(dtype=float)
    if "lc_flux_method" not in sub.columns or "psf_flux" not in sub.columns:
        return mag_inst
    psf_flux = pd.to_numeric(sub["psf_flux"], errors="coerce").to_numpy(dtype=float)
    if "psf_ac_applied" in sub.columns:
        ac_ok = sub["psf_ac_applied"].map(_coerce_bool_cell).to_numpy(dtype=bool)
    else:
        ac_ok = np.zeros(len(sub), dtype=bool)
    use_psf = (
        (sub["lc_flux_method"].astype(str).to_numpy() == "psf")
        & ac_ok
        & np.isfinite(psf_flux)
        & (psf_flux > 0)
    )
    psf_mag = np.where(use_psf, -2.5 * np.log10(np.where(psf_flux > 0, psf_flux, np.nan)), np.nan)
    return np.where(np.isfinite(psf_mag), psf_mag, mag_inst)

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
    3. cfg.phase01_plate_scale_arcsec_per_px (last resort)
    4. plate_scale_arcsec_per_pixel(cfg); None if unavailable
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
                    # EXC-0181: T4 -- DB plate-scale lookup failure falls through to config phase01_plate_scale_arcsec_per_px (EXCEPT-BULK-2 2026-07-08)
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
    # 3. Config phase01_plate_scale_arcsec_per_px (last resort).
    try:
        val = float(cfg.phase01_plate_scale_arcsec_per_px)
        if val > 0:
            result = val
            logging.warning(
                "[FOV] _get_plate_scale_from_cfg -> %.4f arcsec/px (config last-resort; no WCS/DB)",
                float(result),
            )
            return result
    except (TypeError, ValueError):
        pass

    try:
        val = plate_scale_arcsec_per_pixel(cfg)
        if val and float(val) > 0:
            result = float(val)
            logging.info(
                "[FOV] _get_plate_scale_from_cfg -> %.4f arcsec/px (None = fallback na max_dist_deg)",
                float(result),
            )
            return result
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PHASE 2A] Plate scale from config failed (non-critical): %s", exc)
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

    Priority: (1) solved WCS/CD matrix from the FITS; (2) config
    ``phase01_plate_scale_arcsec_per_px`` (last resort with warning).
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
    # Config - last resort only (no usable WCS/CD in the FITS).
    try:
        cfg_ps = float(cfg.phase01_plate_scale_arcsec_per_px)
    except (TypeError, ValueError):
        cfg_ps = 0.0
    if math.isfinite(cfg_ps) and _lo <= cfg_ps <= _hi:
        logging.warning(
            "[PLATE SCALE] no usable WCS/CD scale - falling back to config %.3f arcsec/px",
            float(cfg_ps),
        )
        return float(cfg_ps)
    logging.warning(
        "[PLATE SCALE] plate scale not derivable (WCS/CD + config exhausted) - returning None"
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
        if vy is None:
            vy = hdr.get("VY_PLATESCALE")
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
        for key in ("VY_PLTS", "VY_PLATESCALE", "PIXSCALE", "SECPIX", "SECPIX1", "SCALE", "CDELT1"):
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
    """Uhlova vzdialenost v stupnoch (haversine). Same formula as Phase-1 ``_dist_deg``."""
    from sky_separation import angular_distance_deg  # noqa: PLC0415

    return angular_distance_deg(ra1, dec1, ra2, dec2)

@dataclass(frozen=True)
class StressTestResult:
    per_source_rms: dict[str, float]
    frames_sampled: int
    frames_used: int

def stress_test_relative_rms_from_sidecars(
    *,
    frames_root: Path,
    source_ids: list[str],
    sample_frac: float = 0.10,
    seed: int = 42,
    flux_col: str = "flux",
    name_col: str = "name",
    min_stars_per_frame: int = 3,
) -> StressTestResult:
    """Compute relative RMS for many sources on a random frame sample.

    For each sampled frame with a sidecar CSV, compute per-frame ensemble median among present sources,
    then record relative flux for each star: f_i / median(f_all). Returns RMS over time for each star.
    """
    root = Path(frames_root)
    files = [
        fp
        for fp in _iter_fits_recursive(root)
        if proc_csv_path_for_aligned_fits(fp).is_file()
    ]
    if not files or not source_ids:
        return StressTestResult(per_source_rms={}, frames_sampled=0, frames_used=0)

    frac = float(sample_frac)
    frac = 0.10 if not math.isfinite(frac) else max(0.01, min(1.0, frac))
    k = max(1, int(round(len(files) * frac)))
    rnd = random.Random(int(seed))
    sample = rnd.sample(files, k=min(k, len(files)))

    want = [str(x).strip() for x in source_ids if str(x).strip()]
    want_set = set(want)
    rel_lists: dict[str, list[float]] = {nm: [] for nm in want}

    frames_used = 0
    _sidecar_cache: dict[str, pd.DataFrame] = {}
    for fp in sample:
        sidecar = proc_csv_path_for_aligned_fits(fp)
        _sidecar_key = str(sidecar)
        if _sidecar_key not in _sidecar_cache:
            if Path(sidecar).is_file():
                try:
                    # Sidecar per-frame catalogs often carry Gaia IDs; preserve as strings when present.
                    _sidecar_cache[_sidecar_key] = pd.read_csv(
                        sidecar, low_memory=False, dtype=_GAIA_ID_DTYPE
                    )
                except Exception as exc:  # noqa: BLE001
                    # EXC-0187: T4 -- astroquery/Vizier import failure returns empty VSX-neighbor set - VSX comp exclusion sk... (EXCEPT-BULK-2 2026-07-08)
                    LOGGER.debug("[CSV] Skipping row due to parse error: %s", exc)
                    _sidecar_cache[_sidecar_key] = pd.DataFrame()
            else:
                _sidecar_cache[_sidecar_key] = pd.DataFrame()
        dff = _sidecar_cache[_sidecar_key]
        if dff.empty:
            continue
        if name_col not in dff.columns or flux_col not in dff.columns:
            continue
        names = dff[name_col].astype(str).str.strip()
        flux = pd.to_numeric(dff[flux_col], errors="coerce")
        mask = names.isin(want_set) & flux.notna() & (flux.astype(float) > 0)
        if not bool(mask.any()):
            continue
        sub = dff.loc[mask, [name_col, flux_col]].copy()
        sub[name_col] = sub[name_col].astype(str).str.strip()
        sub[flux_col] = pd.to_numeric(sub[flux_col], errors="coerce").astype(float)
        sub = sub.dropna()
        if len(sub) < int(min_stars_per_frame):
            continue
        med = float(sub[flux_col].median())
        if not math.isfinite(med) or med <= 0:
            continue
        frames_used += 1
        for _, row in sub.iterrows():
            nm = str(row[name_col]).strip()
            if nm in rel_lists:
                rel_lists[nm].append(float(row[flux_col]) / med)

    out: dict[str, float] = {}
    for nm, arr in rel_lists.items():
        if len(arr) < 3:
            continue
        mu = 1.0
        rms = math.sqrt(sum((x - mu) ** 2 for x in arr) / float(len(arr)))
        if math.isfinite(rms):
            out[nm] = float(rms)
    return StressTestResult(per_source_rms=out, frames_sampled=int(len(sample)), frames_used=int(frames_used))

def vsx_is_known_variable_top3_per_bin(
    *,
    rows: list[dict[str, Any]],
    phot_category_key: str = "phot_category",
    rms_key: str = "stress_rms",
    ra_key: str = "ra",
    dec_key: str = "dec",
    max_per_bin: int = 3,
    radius_arcsec: float = 2.0,
) -> set[str]:
    """Return set of Gaia source_id strings that are present in VSX near the best (lowest RMS) stars per bin."""
    try:
        from astroquery.vizier import Vizier  # type: ignore
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except Exception:  # noqa: BLE001
        # EXC-0188: T4 -- numpy/fits import failure returns None intersection bbox - alignment crop not applied (EXCEPT-BULK-2 2026-07-08)
        return set()

    by_bin: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        b = str(r.get(phot_category_key) or "").strip()
        sid = str(r.get("source_id_gaia") or "").strip()
        if not b or not sid:
            continue
        v = r.get(rms_key)
        try:
            rms = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rms):
            continue
        by_bin.setdefault(b, []).append(r)

    viz = Vizier(row_limit=50)
    flagged: set[str] = set()
    for _, items in by_bin.items():
        items_sorted = sorted(items, key=lambda x: float(x.get(rms_key)))
        for r in items_sorted[: int(max_per_bin)]:
            sid = str(r.get("source_id_gaia") or "").strip()
            try:
                ra = float(r.get(ra_key))
                de = float(r.get(dec_key))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(ra) and math.isfinite(de)):
                continue
            c = SkyCoord(ra=ra * u.deg, dec=de * u.deg, frame="icrs")
            try:
                t = viz.query_region(c, radius=float(radius_arcsec) * u.arcsec, catalog="B/vsx")
            except Exception as exc:  # noqa: BLE001
                # EXC-0189: T4 -- One aligned frame skipped in common-field bbox - intersection computed from remaining f... (EXCEPT-BULK-2 2026-07-08)
                LOGGER.debug("[CSV] Skipping row due to parse error: %s", exc)
                continue
            if t and len(t) > 0 and len(t[0]) > 0:
                flagged.add(sid)
    return flagged

def common_field_intersection_bbox_px_from_arrays(
    *,
    frame_arrays: list["np.ndarray"],
    finite_stride: int = 16,
) -> tuple[float, float, float, float] | None:
    """Compute intersection bbox of finite pixels across in-memory frames (x0,y0,x1,y1)."""
    try:
        import numpy as np  # noqa: F401
    except Exception:  # noqa: BLE001
        return None

    bboxes: list[tuple[float, float, float, float]] = []
    for arr in frame_arrays:
        bb = _finite_pixel_bbox_from_array(arr, finite_stride=finite_stride)
        if bb is not None:
            bboxes.append(bb)
    return _intersection_bbox_from_frame_bboxes(bboxes)

def common_field_intersection_bbox_px(
    *,
    frame_paths: list[Path],
    finite_stride: int = 16,
) -> tuple[float, float, float, float] | None:
    """Compute intersection bbox of finite pixels across frames (x0,y0,x1,y1).

    Intended for WCS-reprojected aligned frames where uncovered regions are NaN.
    Uses strided sampling for speed.
    """
    try:
        import numpy as np
        from astropy.io import fits
    except Exception:  # noqa: BLE001
        return None

    fps = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(fps) < 2:
        return None

    bboxes: list[tuple[float, float, float, float]] = []
    for fp in fps:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
        except Exception:  # noqa: BLE001
            continue
        bb = _finite_pixel_bbox_from_array(data, finite_stride=finite_stride)
        if bb is not None:
            bboxes.append(bb)
    return _intersection_bbox_from_frame_bboxes(bboxes)

def recommended_aperture_by_color(
    *,
    bp_rp: float | None,
    median_fwhm_blue: float | None,
    median_fwhm_neutral: float | None,
    median_fwhm_red: float | None,
) -> float | None:
    """Return 2.5x median FWHM for the star's coarse color category."""
    if bp_rp is None:
        return None
    try:
        c = float(bp_rp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(c):
        return None
    if c < 0.5:
        f = median_fwhm_blue
    elif c <= 1.5:
        f = median_fwhm_neutral
    else:
        f = median_fwhm_red
    if f is None:
        return None
    try:
        fv = float(f)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv) or fv <= 0:
        return None
    return 2.5 * fv

def bad_columns_for_light_frame(
    bpm: dict[str, Any] | None,
    *,
    light_header: Any,
) -> set[int]:
    """Map native ``bad_x`` from BPM JSON to 0-based integer column indices in the light frame."""
    if not bpm or not isinstance(bpm, dict):
        return set()
    raw = bpm.get("bad_x")
    if not raw:
        return set()
    try:
        lb_x, _ = fits_binning_xy_from_header(light_header)
    except Exception:  # noqa: BLE001
        lb_x = 1
    lb_x = max(1, int(lb_x))
    mb = int(bpm.get("native_binning") or 1)
    mb = max(1, mb)
    factor = max(1, lb_x // mb)
    out: set[int] = set()
    for x in raw:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        out.add(int(xi // factor))
    return out

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
    """Vrati (fwhm_per_row, fwhm_moment_med, fwhm_gaussian) - rovnaky vypocet ako v ``enhance_catalog_dataframe_aperture_bpm``.

    Pouziva sa v ``pipeline._apply_aperture_catalog_enhancements_from_st`` pre multi-aperturu (r_small / r_large),
    aby polomery zodpovedali hlavnej aperture.
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

    DAO_TO_GAUSSIAN = 1.0 / 1.5  # 0.667 - fyzikalne odvodene, setup-nezavisle
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
                    if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_did_log_fwhm", False)):
                        logging.info(
                            f"[PHOT] FWHM z VY_FWHM (DAO): {_vy_f:.3f}px x {DAO_TO_GAUSSIAN:.3f} = "
                            f"{float(fwhm_gaussian):.3f}px -> apertura = "
                            f"{float(fwhm_gaussian) * float(aperture_fwhm_factor):.3f}px"
                        )
                        enhance_catalog_dataframe_aperture_bpm._did_log_fwhm = True
        except (TypeError, ValueError):
            pass

    if fwhm_gaussian is None:
        if math.isfinite(fwhm_moment_med) and fwhm_moment_med > 0:
            fwhm_gaussian = fwhm_moment_med * 0.619
            if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_did_log_fwhm", False)):
                logging.info(
                    f"[PHOT] FWHM fallback momentx0.619: {fwhm_gaussian:.3f}px -> "
                    f"apertura = {float(fwhm_gaussian) * float(aperture_fwhm_factor):.3f}px"
                )
                enhance_catalog_dataframe_aperture_bpm._did_log_fwhm = True
        else:
            fwhm_gaussian = float("nan")

    r_ap_test = float(aperture_fwhm_factor) * float(fwhm_gaussian) if math.isfinite(float(fwhm_gaussian)) else float("nan")
    if not math.isfinite(r_ap_test) or r_ap_test < 3.0 or r_ap_test > 20.0:
        fwhm_gaussian = float(fwhm_moment_med)
        logging.warning(
            f"[PHOT] Gaussian FWHM fallback na moment: {fwhm_gaussian:.2f}px "
            f"(r_ap={r_ap_test:.2f}px mimo rozsahu)"
        )

    return fwhm_per, fwhm_moment_med, float(fwhm_gaussian) if math.isfinite(float(fwhm_gaussian)) else float("nan")

def enhance_catalog_dataframe_aperture_bpm(
    df: pd.DataFrame,
    data: np.ndarray,
    hdr: Any,
    *,
    aperture_enabled: bool,
    aperture_fwhm_factor: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    nonlinearity_peak_percentile: float,
    nonlinearity_fwhm_ratio: float,
    master_dark_path: Path | str | None,
    gaussian_fwhm_px_override: float | None = None,
    r_small_px: float | None = None,
    r_large_px: float | None = None,
    cog_params: dict[str, Any] | None = None,
    err_background_mode: str = ERR_BKG_MODE_EMPIRICAL,
    err_empty_apertures_n: int = 64,
    err_empty_apertures_min: int = 16,
    aperture_variable_factor: float = 1.0,
    aperture_comp_factor: float = 1.0,
    variable_target_catalog_ids: frozenset[str] | None = None,
    aperture_policy_mode: str | None = None,
    fwhm_frame_px: float | None = None,
    fwhm_night_median_px: float | None = None,
    qc_fwhm_by_name: dict[str, float] | None = None,
    frame_name: str | None = None,
) -> pd.DataFrame:
    """Replace DAO ``flux`` with aperture photometry when enabled; add linearity/BPM flags.

    When ``cog_params`` is given (curve-of-growth aperture correction enabled), also
    emits ``dao_flux_apcorr`` / ``ac_factor`` / ``cog_ok`` without overwriting ``dao_flux``.
    """
    out = df.copy()
    arr = np.asarray(data, dtype=np.float32)

    x = pd.to_numeric(out.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    n = len(out)
    if n == 0:
        return out

    # Povodny DAO flux z detect_stars_and_match_catalog (historicky v stlpci ``flux``).
    # ``dao_flux``: sky-subtrahovany flux (po aperturnej fotometrii, ak je zapnuta).
    if "flux" in out.columns:
        flux_dao = pd.to_numeric(out["flux"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        flux_dao = np.full(n, np.nan, dtype=np.float64)
    if "dao_flux" not in out.columns:
        out["dao_flux"] = flux_dao

    fwhm_per, fwhm_moment_med, fwhm_gaussian_f = compute_fwhm_gaussian_for_aperture_catalog(
        out,
        arr,
        hdr,
        gaussian_fwhm_px_override=gaussian_fwhm_px_override,
        aperture_fwhm_factor=aperture_fwhm_factor,
    )
    out["fwhm_estimate_px"] = fwhm_per

    if gaussian_fwhm_px_override is not None:
        try:
            _ov_ok = math.isfinite(float(gaussian_fwhm_px_override)) and 0.5 < float(gaussian_fwhm_px_override) < 30.0
        except (TypeError, ValueError):
            _ov_ok = False
        _fwhm_scope = "per_draft_gaussian_override" if _ov_ok else "per_frame_moment_median"
    elif hdr is not None and hdr.get("VY_FWHM") is not None:
        _fwhm_scope = "per_frame_header_vy_fwhm_dao_scaled"
    elif math.isfinite(float(fwhm_moment_med)) and float(fwhm_moment_med) > 0:
        _fwhm_scope = "per_frame_moment_median"
    else:
        _fwhm_scope = "unknown"
    _snr_mode = "global_fixed"
    _target_cids = variable_target_catalog_ids or frozenset()
    _ap01_mode = None
    if aperture_policy_mode is not None and str(aperture_policy_mode).strip():
        from aperture_policy import (  # noqa: PLC0415
            FWHM_AUTHORITY,
            fwhm_for_radius,
            normalize_aperture_policy_mode,
            resolve_aperture_geometry,
            resolve_frame_fwhm_px,
        )

        _ap01_mode = normalize_aperture_policy_mode(aperture_policy_mode)
    _ap01_fwhm_frame = None
    _ap01_fwhm_used = None
    _ap01_r_ap = _ap01_r_in = _ap01_r_out = None
    if _ap01_mode is not None:
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
            _ap01_mode,
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
            _snr_mode = "aperture_01"
            _fwhm_scope = FWHM_AUTHORITY

    _ap01_ok = (
        _ap01_mode is not None
        and _ap01_fwhm_used is not None
        and math.isfinite(float(_ap01_fwhm_used))
        and float(_ap01_fwhm_used) > 0
    )
    if aperture_enabled and (
        _ap01_ok
        or (math.isfinite(float(fwhm_gaussian_f)) and float(fwhm_gaussian_f) > 0)
    ):
        try:
            # Lokalna implementacia: sky-subtracted flux cez CircularAperture + CircularAnnulus.
            from photutils.aperture import CircularAnnulus, CircularAperture
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

            ap = CircularAperture(pos, r=r_ap)
            an = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
            phot_ap = _aphot(d, ap, method="exact")
            sum_ap = np.asarray(phot_ap["aperture_sum"], dtype=np.float64)
            area_ap_per = float(ap.area)
            sky_pp_arr = np.zeros(n, dtype=np.float64)
            ann_masks = an.to_mask(method="center")
            if not isinstance(ann_masks, (list, tuple)):
                ann_masks = [ann_masks]
            for i, amask in enumerate(ann_masks):
                try:
                    ann_img = amask.to_image(d.shape)
                    sky_pp_arr[i] = _sky_pp_from_annulus_image(d, ann_img)
                except Exception:  # noqa: BLE001
                    sky_pp_arr[i] = float(np.median(d))
            flux_arr = sum_ap - sky_pp_arr * area_ap_per
            out["flux"] = flux_arr.astype(np.float64)
            out["dao_flux"] = out["flux"]
            out["aperture_r_px"] = float(r_ap)
            out["aperture_factor_applied"] = f"global_{float(aperture_fwhm_factor):.3f}x"
            out["fwhm_px_for_aperture"] = float(fw)
            out["fwhm_px_scope"] = _fwhm_scope
            out["snr_aperture_mode"] = _snr_mode
            out["sky_annulus_r_out_px"] = float(r_out)
            out["noise_floor_adu"] = sky_pp_arr.astype(np.float64)
            out[SKY_ADU_PER_PX_ANNULUS_COL] = sky_pp_arr.astype(np.float64)

            if _ap01_mode is not None:
                _stamp_fwhm_frame = (
                    float(_ap01_fwhm_frame)
                    if _ap01_fwhm_frame is not None
                    else float("nan")
                )
                out["aperture_policy_mode"] = str(_ap01_mode)
                out["aperture_f"] = float(aperture_fwhm_factor)
                out["fwhm_px_for_aperture"] = _stamp_fwhm_frame
                out["fwhm_px_scope"] = _fwhm_scope
                out["snr_aperture_mode"] = "aperture_01"
                if _ap01_r_in is not None:
                    out["sky_annulus_r_in_px"] = float(_ap01_r_in)
                if _ap01_r_out is not None:
                    out["sky_annulus_r_out_px"] = float(_ap01_r_out)
                out["aperture_factor_applied"] = (
                    f"aperture_01_{_ap01_mode}_{float(aperture_fwhm_factor):.3f}x"
                )

            # Multi-apertura: rovnaky sky_pp_arr (ADU/px^2) x plocha apertury ako sky odcitanie.
            if r_small_px is not None and r_large_px is not None:
                try:
                    _rs = float(r_small_px)
                    _rl = float(r_large_px)
                except (TypeError, ValueError):
                    _rs, _rl = float("nan"), float("nan")
                if (
                    math.isfinite(_rs)
                    and math.isfinite(_rl)
                    and _rs > 0
                    and _rl > 0
                    and int(sky_pp_arr.shape[0]) == n
                ):
                    try:
                        ap_sm = CircularAperture(pos, r=_rs)
                        ap_lg = CircularAperture(pos, r=_rl)
                        phot_sm = _aphot(d, ap_sm, method="exact")
                        phot_lg = _aphot(d, ap_lg, method="exact")
                        sum_sm = np.asarray(phot_sm["aperture_sum"], dtype=np.float64).ravel()
                        sum_lg = np.asarray(phot_lg["aperture_sum"], dtype=np.float64).ravel()
                        if sum_sm.size != n or sum_lg.size != n:
                            raise ValueError(
                                f"multi-aperture sum size mismatch: n={n} small={sum_sm.size} large={sum_lg.size}"
                            )
                        area_sm = math.pi * _rs * _rs
                        area_lg = math.pi * _rl * _rl
                        flux_sm = sum_sm - sky_pp_arr * area_sm
                        flux_lg = sum_lg - sky_pp_arr * area_lg
                        flux_sm = np.where(np.isfinite(flux_sm), flux_sm, np.nan)
                        flux_lg = np.where(np.isfinite(flux_lg), flux_lg, np.nan)
                        out["flux_small"] = flux_sm.astype(np.float64)
                        out["flux_large"] = flux_lg.astype(np.float64)
                    except (ValueError, TypeError) as _ma_exc:
                        logging.debug("[PHOT] multi-aperture flux_small/flux_large skipped: %s", _ma_exc)

            # F-BINGAIN-1: per-frame empirical background noise at production aperture radii.
            # CONSOLIDATE-01D: howell-only skip (no empty-aperture measurement) deleted.
            _ = err_background_mode  # call-site kw retained; policy is always empirical
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
                if not hasattr(enhance_catalog_dataframe_aperture_bpm, "_labbe_seeds"):
                    enhance_catalog_dataframe_aperture_bpm._labbe_seeds = []
                enhance_catalog_dataframe_aperture_bpm._labbe_seeds.append(
                    {"r_ap": float(_r_u), "seed": int(_seed), "n_valid": int(_nv)}
                )
                if math.isfinite(_sig) and _sig >= 0:
                    _sigma_by_r[_sigma_bkg_r_key(_r_u)] = (float(_sig), ERR_BKG_SOURCE_EMPIRICAL)
                else:
                    _sigma_by_r[_sigma_bkg_r_key(_r_u)] = (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK)
                    if not hasattr(enhance_catalog_dataframe_aperture_bpm, "_err_bkg_logged"):
                        enhance_catalog_dataframe_aperture_bpm._err_bkg_logged = set()
                    _log_id = str(hdr.get("FRAME") or hdr.get("VY_FRAME") or id(hdr))
                    if _log_id not in enhance_catalog_dataframe_aperture_bpm._err_bkg_logged:
                        log_event(
                            f"[PHOT] err_bkg empirical fallback (howell): r_ap={float(_r_u):.2f}px "
                            f"n_valid={_nv} reason={_reason or 'unknown'}"
                        )
                        enhance_catalog_dataframe_aperture_bpm._err_bkg_logged.add(_log_id)

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

            # Per-frame curve-of-growth aperture correction (gated; never overwrites dao_flux).
            if cog_params is not None:
                try:
                    _peak = pd.to_numeric(out.get("peak_max_adu"), errors="coerce").to_numpy(dtype=np.float64) \
                        if "peak_max_adu" in out.columns else None
                    _sat = pd.to_numeric(out.get("saturate_limit_adu"), errors="coerce").to_numpy(dtype=np.float64) \
                        if "saturate_limit_adu" in out.columns else None
                    _rap_for_cog = np.full(n, float(r_ap), dtype=np.float64)
                    _cog = compute_per_frame_cog_correction(
                        d,
                        np.asarray(x, dtype=np.float64),
                        np.asarray(y, dtype=np.float64),
                        pd.to_numeric(out["dao_flux"], errors="coerce").to_numpy(dtype=np.float64),
                        np.asarray(_rap_for_cog, dtype=np.float64),
                        sky_pp_arr,
                        fwhm_px=fw,
                        peak_max_adu=_peak,
                        sat_limit_adu=_sat,
                        ref_fwhm=float(cog_params.get("ref_fwhm", 4.5)),
                        ladder_step_px=float(
                            resolve_px_from_fwhm_factor(
                                cog_params.get("ladder_step_fwhm"),
                                float(cog_params.get("ladder_step_px", 0.5)),
                                fw,
                                param_name="cog_ladder_step_px",
                            )
                        ),
                        min_stars=int(cog_params.get("min_stars", 8)),
                        isolation_fwhm=float(cog_params.get("isolation_fwhm", 6.0)),
                        snr_min=float(cog_params.get("snr_min", 50.0)),
                        sat_frac=float(cog_params.get("sat_frac", 0.85)),
                        gain=float(cog_params.get("gain", 1.0)),
                        read_noise=float(cog_params.get("read_noise", 10.0)),
                        ac_factor_max=float(cog_params.get("ac_factor_max", 5.0)),
                        fallback_ee=cog_params.get("fallback_ee"),
                    )
                    _acf = np.asarray(_cog["ac_factor"], dtype=np.float64)
                    _dao = pd.to_numeric(out["dao_flux"], errors="coerce").to_numpy(dtype=np.float64)
                    out["ac_factor"] = _acf
                    out["dao_flux_apcorr"] = (_dao * _acf).astype(np.float64)
                    out["cog_ok"] = bool(_cog["cog_ok"])
                    if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_cog_logged", False)):
                        logging.info(
                            "[COG] per-frame aperture correction: n_cog=%d cog_ok=%s ref_r=%.2fpx ac_factor median=%.4f",
                            int(_cog["n_cog"]),
                            bool(_cog["cog_ok"]),
                            float(_cog["ref_r_px"]),
                            float(np.nanmedian(_acf)),
                        )
                        enhance_catalog_dataframe_aperture_bpm._cog_logged = True
                except Exception as _cog_exc:  # noqa: BLE001
                    logging.warning("[COG] per-frame aperture correction skipped: %s", _cog_exc)
                    out["ac_factor"] = np.ones(n, dtype=np.float64)
                    out["dao_flux_apcorr"] = pd.to_numeric(out["dao_flux"], errors="coerce").to_numpy(dtype=np.float64)
                    out["cog_ok"] = False
        except Exception as _ap_exc:  # noqa: BLE001
            logging.warning(
                "[FAZA 2A] Aperture photometry failed - restoring pre-aperture flux: %s",
                _ap_exc,
                exc_info=True,
            )
            out["dao_flux"] = flux_dao
            out["flux"] = flux_dao
    else:
        out["dao_flux"] = flux_dao
        out["flux"] = flux_dao

    if "peak_max_adu" in out.columns:
        peak = pd.to_numeric(out["peak_max_adu"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        peak = np.full(n, np.nan, dtype=np.float64)
    finite_pk = peak[np.isfinite(peak)]
    thr_pk = float("nan")
    if finite_pk.size > 0:
        pct = min(100.0, max(0.0, 100.0 - float(nonlinearity_peak_percentile)))
        thr_pk = float(np.percentile(finite_pk, pct))

    ratio = float(nonlinearity_fwhm_ratio)
    likely_nl = np.zeros(n, dtype=bool)
    for i in range(n):
        if not (math.isfinite(fwhm_per[i]) and math.isfinite(fwhm_moment_med) and fwhm_moment_med > 0):
            continue
        if not (math.isfinite(peak[i]) and math.isfinite(thr_pk) and peak[i] >= thr_pk):
            continue
        if fwhm_per[i] > ratio * fwhm_moment_med:
            likely_nl[i] = True
    out["likely_nonlinear"] = likely_nl

    bpm_path = None
    bpm: dict[str, Any] | None = None
    if master_dark_path:
        mp = Path(str(master_dark_path))
        bpm_path = mp.parent / f"{mp.stem}_dark_bpm.json"
        if bpm_path.is_file():
            try:
                bpm = json.loads(bpm_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                bpm = None

    bad_x = bad_columns_for_light_frame(bpm, light_header=hdr)
    on_bad = np.zeros(n, dtype=bool)
    if bad_x:
        for i in range(n):
            if not np.isfinite(x[i]):
                continue
            xi = int(round(float(x[i])))
            if xi in bad_x:
                on_bad[i] = True
    out["on_bad_column"] = on_bad

    if "photometry_ok" in out.columns:
        base_ok = out["photometry_ok"].fillna(True).astype(bool).to_numpy()
        out["photometry_ok"] = base_ok & (~likely_nl) & (~on_bad)
    else:
        out["photometry_ok"] = ~(likely_nl | on_bad)

    if "source_type" in out.columns and "dao_flux" in out.columns:
        _forced_mask = (
            out["source_type"].fillna("").astype(str).str.strip().eq("FORCED_APERTURE")
        )
        _dao_num = pd.to_numeric(out["dao_flux"], errors="coerce")
        _has_flux = _dao_num.notna() & (_dao_num != 0)
        out.loc[_forced_mask & _has_flux, "photometry_ok"] = True

    # Multi-apertura: stlpce vzdy existuju (NaN ak meranie neprebehlo alebo bolo vypnute).
    if n > 0:
        _nan_vec = np.full(n, np.nan, dtype=np.float64)
        if "flux_small" not in out.columns:
            out["flux_small"] = _nan_vec
        if "flux_large" not in out.columns:
            out["flux_large"] = _nan_vec.copy()

    return out
