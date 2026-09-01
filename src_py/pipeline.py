"""Core processing pipeline for FITS observations."""

from __future__ import annotations

import contextlib
import json
import logging
import pickle
import math
import multiprocessing
import os
import shutil
import subprocess
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd

from config import AppConfig, load_config_json
from catalog_match_trust import export_catalog_match_mode_from_internal
import vyvar_alignment_frame  # A-durable: fresh-attr MP func lookup at dispatch (reload-safe)
from vyvar_alignment_frame import (
    _alignment_compute_one_frame,
    _alignment_detect_xy,
    _as_fits_float32_image,
    _astrometry_align_mp_init,
    _astrometry_align_mp_task,
)
from database import (
    DraftTechnicalMetadataError,
    VyvarDatabase,
    _db_header_pixel_native_um_mean,
    _db_to_float as _to_float_db,
    query_local_gaia,
    query_local_gaia_by_source_ids,
    query_local_exoplanet,
    query_local_vsx,
)
from time_utils import _header_float as _header_float_tu
from photometry import (
    common_field_intersection_bbox_px,
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture_bpm,
    recommended_aperture_by_color,
    stress_test_relative_rms_from_sidecars,
    vsx_is_known_variable_top3_per_bin,
)
from fits_suffixes import FITS_SUFFIXES_LOWER, path_suffix_is_fits
from gaia_catalog_id import (
    catalog_id_series_for_masterstars_export,
    normalize_gaia_source_id,
    read_vyvar_csv,
)
from infolog import log_event, log_exception, log_milestone
from optics_selection import resolve_optics_ids_for_platesolve
from calibration import (
    CALIBRATION_LIBRARY_NATIVE_BINNING,
    filter_light_paths_for_calibration_db,
    get_processed_master,
)
from cal_diag import (
    CalDiagGateResult,
    CalDiagSession,
    apply_cal_diag_headers,
    convention_to_dark_mode,
    dark_np_for_cal_diag,
    gate_result_for_frame,
    is_obs_group_aborted,
    passthrough_cal_diag_headers,
    run_cal_diag_pregate,
    write_cal_diag_json,
)
from photometry_core import (
    _fwhm_moment_at,
    merge_photometry_pipeline_meta,
    stamp_masterstar_snr_columns,
)
from proc_frame_store import proc_csv_path_for_aligned_fits

from dao_reconcile import compute_gaia_dao_reconcile, reconcile_to_pipeline_meta, resolve_effective_match_depth
from masterstar_context import header_core_fwhm_px
from plain_stats import plain_mean_med_std, sky_mad_sigma_adu

from utils import (
    ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC,
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    MIN_GAIA_CONE_RADIUS_DEG,
    astrometry_net_scale_bounds_arcsec_per_pix,
    catalog_cone_radius_deg_from_optics,
    catalog_cone_radius_from_fov_diameter_deg,
    dao_detection_fwhm_pixels,
    effective_astrometry_net_tweak_order,
    effective_binned_pixel_pitch_um,
    fits_binning_xy_from_header,
    fits_header_has_celestial_wcs,
    iter_fits_paths_recursive as _iter_fits_recursive,
    masterstar_wcs_quality,
    maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel,
    normalize_telescope_focal_mm_for_plate_scale,
    per_frame_catalog_match_sep_arcsec_for_scale,
    plate_scale_arcsec_per_pixel,
    plate_solve_fov_deg_diagonal_from_scale,
    strip_celestial_wcs_keys,
    strip_vendor_platesolve_metadata,
    wcs_distortion_log_suffix,
    wcs_rotation_angle_deg,
)
from vyvar_platesolver import (
    _fits_header_parse_dec_deg,
    _fits_header_parse_ra_deg,
    pointing_hint_from_header as _pointing_hint_from_header,
)
import itertools

# Aperturna fotometria Faz 0-2A (active_targets.csv, zone_flag, skip_photometry) je v ``photometry_core``
# - ``run_phase0_and_phase1`` / ``run_phase2a``, nie v tomto subore.

_EPSF_SKIP_LOGGED: set[str] = set()


def _photometry_mode_run_flags(
    cfg: AppConfig | None = None,
    *,
    platesolve_dir: Path | str | None = None,
) -> tuple[bool, bool]:
    """Return ``(_run_aperture, _run_epsf)`` from ``photometry_mode`` (and ePSF file presence)."""
    _cfg = cfg if cfg is not None else AppConfig()
    _phot_mode = str(getattr(_cfg, "photometry_mode", "both")).lower().strip()
    if _phot_mode not in ("aperture", "epsf", "both"):
        _phot_mode = "both"
    _run_aperture = _phot_mode in ("aperture", "both")
    _run_epsf = _phot_mode in ("epsf", "both")
    if _run_epsf and platesolve_dir is not None:
        _epsf_fits = Path(platesolve_dir) / "masterstar_epsf.fits"
        if not _epsf_fits.is_file():
            _ps_key = str(Path(platesolve_dir).resolve())
            if _ps_key not in _EPSF_SKIP_LOGGED:
                _EPSF_SKIP_LOGGED.add(_ps_key)
                logging.getLogger("pipeline").info(
                    "photometry_mode includes epsf but no ePSF model found at %s - skipping PSF photometry",
                    _epsf_fits,
                )
            _run_epsf = False
    return _run_aperture, _run_epsf


def _apply_aperture_catalog_enhancements_from_st(
    df: pd.DataFrame,
    data: Any,
    hdr: fits.Header,
    st: dict[str, Any],
) -> pd.DataFrame:
    """Aperture photometry + linearity/BPM flags for per-frame catalog DataFrames."""
    if not bool(st.get("_run_aperture", st.get("aperture_photometry_enabled", True))):
        return df
    mdp = st.get("master_dark_path") or ""
    mdp = str(mdp).strip() or None

    r_small_px: float | None = None
    r_large_px: float | None = None
    if bool(st.get("aperture_correction_enabled", False)):
        try:
            data_arr = np.asarray(data, dtype=np.float32)
            _apt_fw = float(st.get("aperture_fwhm_factor", 1.7))
            _fper, _fmed, fw_g = compute_fwhm_gaussian_for_aperture_catalog(
                df,
                data_arr,
                hdr,
                gaussian_fwhm_px_override=st.get("gaussian_fwhm_px_override"),
                aperture_fwhm_factor=_apt_fw,
            )
            if math.isfinite(fw_g) and fw_g > 0:
                r_small_px = float(st.get("aperture_fwhm_factor_small", 1.5)) * float(fw_g)
                r_large_px = float(st.get("aperture_fwhm_factor_large", 4.0)) * float(fw_g)
        except Exception:  # noqa: BLE001
            r_small_px, r_large_px = None, None

    cog_params: dict[str, Any] | None = None
    if bool(st.get("cog_aperture_correction_enabled", False)):
        cog_params = {
            "ref_fwhm": float(st.get("cog_ref_fwhm", 4.5)),
            "ladder_step_px": float(st.get("cog_ladder_step_px", 0.5)),
            "ladder_step_fwhm": st.get("cog_ladder_step_fwhm"),
            "min_stars": int(st.get("cog_min_stars", 8)),
            "isolation_fwhm": float(st.get("cog_isolation_fwhm", 6.0)),
            "snr_min": float(st.get("cog_snr_min", 50.0)),
            "sat_frac": float(st.get("cog_sat_frac", 0.85)),
            "gain": float(st.get("gain", 1.0)),
            "read_noise": float(st.get("read_noise", 10.0)),
            "ac_factor_max": float(st.get("cog_ac_factor_max", 5.0)),
        }

    try:
        _go = st.get("gaussian_fwhm_px_override")
        try:
            _go_f = float(_go) if _go is not None else None
        except (TypeError, ValueError):
            _go_f = None
        return enhance_catalog_dataframe_aperture_bpm(
            df,
            data,
            hdr,
            aperture_enabled=True,
            aperture_fwhm_factor=float(st.get("aperture_fwhm_factor", 1.7)),
            annulus_inner_fwhm=float(st.get("annulus_inner_fwhm", 4.0)),
            annulus_outer_fwhm=float(st.get("annulus_outer_fwhm", 6.0)),
            nonlinearity_peak_percentile=float(st.get("nonlinearity_peak_percentile", 20.0)),
            nonlinearity_fwhm_ratio=float(st.get("nonlinearity_fwhm_ratio", 1.25)),
            master_dark_path=mdp,
            gaussian_fwhm_px_override=_go_f,
            r_small_px=r_small_px,
            r_large_px=r_large_px,
            cog_params=cog_params,
            err_background_mode="empirical",
            err_empty_apertures_n=int(st.get("err_empty_apertures_n", 64)),
            err_empty_apertures_min=int(st.get("err_empty_apertures_min", 16)),
            aperture_variable_factor=float(st.get("aperture_variable_factor", 1.0)),
            aperture_comp_factor=float(st.get("aperture_comp_factor", 1.0)),
            variable_target_catalog_ids=frozenset(st.get("variable_target_catalog_ids") or []),
            aperture_policy_mode=st.get("aperture_policy_mode"),
            fwhm_night_median_px=st.get("fwhm_night_median_px"),
            qc_fwhm_by_name=st.get("qc_fwhm_by_name"),
            frame_name=st.get("current_frame_name"),
        )
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().catalog_bpm_enhance_fail += 1
        LOGGER.error("[CATALOG] aperture/BPM/COG enhancement failed: %s", exc)
        return df



def _fits_header_vy_algn_aligned(hdr: fits.Header) -> bool:
    """True when frame pixels are on the MASTERSTAR reference alignment grid (VY_ALGN)."""
    try:
        val = hdr.get("VY_ALGN")
        if val is None:
            return True  # legacy frames without tag - keep pixel-fallback behaviour
        if isinstance(val, tuple):
            val = val[0]
        if isinstance(val, bool):
            return bool(val)
        s = str(val).strip().lower()
        if s in ("true", "1", "t", "yes"):
            return True
        if s in ("false", "0", "f", "no"):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _frame_gain_readnoise_for_error_map(
    hdr: fits.Header,
    *,
    db: VyvarDatabase | None,
    equipment_id: int | None,
) -> tuple[float, float]:
    """Gain / read-noise for the per-frame error map.

    Unified resolution (param_resolver): gain header-first (e-/ADU or index-mapped) ->
    DB -> config; read noise DB-first. Matches Phase 2A photometric-error path.
    """
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

    g_res = resolve_gain(hdr, db=db, equipment_id=equipment_id)
    rn_res = resolve_read_noise(hdr, db=db, equipment_id=equipment_id)
    gain = float(g_res.value) if g_res.ok else 1.0
    rn = float(rn_res.value) if rn_res.ok else 10.0
    return gain, rn


def _per_frame_noise_error_map(data: Any, hdr: fits.Header, *, db: VyvarDatabase | None, equipment_id: int | None):
    """Per-pixel noise sigma for CCD-like error: sqrt(max(data,0)/gain + readnoise^2)."""
    import numpy as np

    gain, rn = _frame_gain_readnoise_for_error_map(hdr, db=db, equipment_id=equipment_id)
    d = np.asarray(data, dtype=np.float64)
    sig = np.where(np.isfinite(d), np.maximum(d, 0.0), 0.0)
    var = sig / float(gain) + float(rn) ** 2
    return np.sqrt(np.maximum(var, 1e-24))


def _export_catalog_psf_st_fields(cfg: AppConfig, platesolve_dir: Path) -> dict[str, Any]:
    """ePSF model path and PSF toggles for per-frame catalog workers (TODO-8 Phase 2B)."""
    _epsf_fits = Path(platesolve_dir) / "masterstar_epsf.fits"
    _run_aperture, _run_epsf = _photometry_mode_run_flags(cfg, platesolve_dir=platesolve_dir)
    return {
        "epsf_model_path": str(_epsf_fits) if _epsf_fits.is_file() else "",
        "psf_photometry_enabled": bool(cfg.psf_photometry_enabled),
        "photometry_mode": str(getattr(cfg, "photometry_mode", "both")),
        "_run_aperture": bool(_run_aperture),
        "_run_epsf": bool(_run_epsf),
        "gain": float(cfg.gain),
        "read_noise": float(cfg.read_noise),
        "psf_ac_policy": str(getattr(cfg, "psf_ac_policy", "p4_none") or "p4_none"),
    }


def _epsf_target_catalog_ids(platesolve_dir: Path, *, top_comps: int = 40) -> set[str] | None:
    """Catalog IDs for targeted ePSF: active targets + best comparison stars by ``comp_rms``."""
    phot = Path(platesolve_dir) / "photometry"
    ids: set[str] = set()
    at_p = phot / "active_targets.csv"
    if at_p.is_file():
        try:
            at = read_vyvar_csv(at_p, low_memory=False)
            if "catalog_id" in at.columns:
                for _, row in at.iterrows():
                    z = str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower()
                    if z == "catalog_only":
                        continue
                    raw = row.get("catalog_id")
                    if raw is None:
                        continue
                    s = str(raw).strip()
                    if not s or s.lower() in ("nan", "none"):
                        continue
                    try:
                        ids.add(str(normalize_gaia_source_id(s)).strip())
                    except Exception:  # noqa: BLE001
                        ids.add(s)
        except Exception:  # noqa: BLE001
            pass
    for comp_p in (phot / "comparison_stars.csv", Path(platesolve_dir) / "comparison_stars.csv"):
        if not comp_p.is_file():
            continue
        try:
            cdf = read_vyvar_csv(comp_p, low_memory=False)
            if "catalog_id" in cdf.columns and not cdf.empty:
                for raw in cdf["catalog_id"].fillna("").astype(str).str.strip():
                    if not raw or raw.lower() in ("nan", "none"):
                        continue
                    try:
                        ids.add(str(normalize_gaia_source_id(raw)).strip())
                    except Exception:  # noqa: BLE001
                        ids.add(raw)
            break
        except Exception as _comp_exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] comparison_stars load failed: %s", _comp_exc)
    comp_p = phot / "comparison_stars_per_target.csv"
    if comp_p.is_file() and len(ids) < 5:
        try:
            cdf = read_vyvar_csv(comp_p, low_memory=False)
            if "catalog_id" in cdf.columns and not cdf.empty:
                work = cdf.copy()
                if "comp_rms" in work.columns:
                    work["_cr"] = pd.to_numeric(work["comp_rms"], errors="coerce")
                    work = work.sort_values("_cr", ascending=True, na_position="last")
                seen: list[str] = []
                for raw in work["catalog_id"].fillna("").astype(str).str.strip():
                    if not raw or raw.lower() in ("nan", "none"):
                        continue
                    try:
                        nk = str(normalize_gaia_source_id(raw)).strip()
                    except Exception:  # noqa: BLE001
                        nk = raw
                    if nk in seen:
                        continue
                    seen.append(nk)
                    ids.add(nk)
                    if len(seen) >= int(top_comps):
                        break
        except Exception as _comp_exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] comparison_stars_per_target load failed: %s", _comp_exc)
    if ids:
        LOGGER.debug("[ePSF] target catalog_ids loaded: %d (active+top comps)", len(ids))
    return ids if ids else None


def _add_catalog_ids_from_csv(ids: set[str], comp_p: Path) -> None:
    """Merge ``catalog_id`` values from a comparison-star CSV into *ids*."""
    if not comp_p.is_file():
        return
    try:
        cdf = read_vyvar_csv(comp_p, low_memory=False)
        if "catalog_id" not in cdf.columns or cdf.empty:
            return
        for raw in cdf["catalog_id"].fillna("").astype(str).str.strip():
            if not raw or raw.lower() in ("nan", "none"):
                continue
            try:
                ids.add(str(normalize_gaia_source_id(raw)).strip())
            except Exception:  # noqa: BLE001
                ids.add(raw)
    except Exception as _comp_exc:  # noqa: BLE001
        LOGGER.warning("[ePSF] comparison star load failed (%s): %s", comp_p.name, _comp_exc)


def _epsf_lc_catalog_ids(platesolve_dir: Path) -> set[str] | None:
    """Full LC star set: active targets + comp pool + all per-target comps (gated PSF coverage)."""
    phot = Path(platesolve_dir) / "photometry"
    ids: set[str] = set()
    at_p = phot / "active_targets.csv"
    if at_p.is_file():
        try:
            at = read_vyvar_csv(at_p, low_memory=False)
            if "catalog_id" in at.columns:
                for _, row in at.iterrows():
                    z = str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower()
                    if z == "catalog_only":
                        continue
                    raw = row.get("catalog_id")
                    if raw is None:
                        continue
                    s = str(raw).strip()
                    if not s or s.lower() in ("nan", "none"):
                        continue
                    try:
                        ids.add(str(normalize_gaia_source_id(s)).strip())
                    except Exception:  # noqa: BLE001
                        ids.add(s)
        except Exception:  # noqa: BLE001
            pass
    for comp_p in (phot / "comparison_stars.csv", Path(platesolve_dir) / "comparison_stars.csv"):
        _add_catalog_ids_from_csv(ids, comp_p)
        if comp_p.is_file():
            break
    _add_catalog_ids_from_csv(ids, phot / "comparison_stars_per_target.csv")
    if ids:
        LOGGER.debug("[ePSF] LC catalog_ids loaded: %d (targets+full comp pool)", len(ids))
    return ids if ids else None


def _epsf_fit_catalog_ids(
    platesolve_dir: Path,
    *,
    psf_photometry_enabled: bool = False,
) -> set[str] | None:
    """Target-only subset (default) vs science set when PSF photometry is enabled."""
    if psf_photometry_enabled:
        from epsf_science_set import build_epsf_science_set

        result = build_epsf_science_set(platesolve_dir)
        if not result.catalog_ids:
            raise ValueError(
                "ePSF science set is empty"
                + (f": {result.empty_reason}" if result.empty_reason else "")
                + "; refusing silent fallback to full LC pool."
            )
        LOGGER.debug(
            "[ePSF] science-set catalog_ids loaded: %d (targets=%d comps=%d checks=%d blended=%d)",
            result.n_total,
            result.n_targets,
            result.n_per_target_comps,
            result.n_check_stars,
            result.n_blended,
        )
        return set(result.catalog_ids)
    return _epsf_target_catalog_ids(platesolve_dir)


def _fill_psf_catalog_columns(
    df: pd.DataFrame,
    data: Any,
    hdr: fits.Header,
    st: dict[str, Any],
    *,
    target_ids: set[str] | None = None,
) -> pd.DataFrame:
    """TODO-8 Phase 2B: PSF photometry columns after aperture (additive; never replaces dao_flux)."""
    _run_epsf = bool(st.get("_run_epsf", False))
    _run_aperture = bool(st.get("_run_aperture", False))
    _epsf_raw = str(st.get("epsf_model_path", "") or "").strip()
    _epsf_path = Path(_epsf_raw) if _epsf_raw else None
    _frame_name = str(st.get("epsf_frame_name") or st.get("current_frame_name") or "")
    _frame_index = st.get("epsf_frame_index")
    _psf_record: dict[str, Any] = {
        "frame_name": _frame_name,
        "frame_index": _frame_index,
        "n_fit": 0,
        "n_ok": 0,
        "exception_class": None,
        "exception_message": None,
        "traceback_tail": None,
    }
    if _run_epsf and _epsf_path is not None and _epsf_path.is_file():
        _n_fit = 0
        _n_ok = 0
        try:
            from psf_photometry import psf_photometry_stars as _psf_phot

            _gain = float(st.get("gain", 1.0) or 1.0)
            _rn = float(st.get("read_noise", 10.0) or 10.0)
            if not math.isfinite(_gain) or _gain <= 0:
                _gain = 1.0
            if not math.isfinite(_rn) or _rn <= 0:
                _rn = 10.0
            d_arr = np.asarray(data, dtype=np.float32)
            _err_map = np.sqrt(np.abs(d_arr) / _gain + (_rn / _gain) ** 2).astype(np.float32)

            if not {"catalog_id", "x", "y"}.issubset(df.columns):
                raise ValueError("catalog missing catalog_id/x/y for PSF photometry")

            _tid = target_ids
            if _tid is None:
                _raw_tid = st.get("epsf_target_ids")
                if isinstance(_raw_tid, (set, frozenset, list, tuple)) and _raw_tid:
                    _tid = {str(x).strip() for x in _raw_tid if str(x).strip()}

            for _c in ("psf_flux", "psf_flux_err", "psf_chi2"):
                if _c not in df.columns:
                    df[_c] = float("nan")
            if "psf_fit_ok" not in df.columns:
                df["psf_fit_ok"] = False
            if "psf_quality" not in df.columns:
                df["psf_quality"] = ""
            if "psf_snr" not in df.columns:
                df["psf_snr"] = float("nan")

            _fit_df = df
            if _tid is not None:
                _cid_key = df["catalog_id"].fillna("").astype(str).str.strip().map(
                    lambda x: (
                        str(normalize_gaia_source_id(x)).strip()
                        if x and str(x).lower() not in ("nan", "none")
                        else ""
                    )
                )
                _fit_df = df.loc[_cid_key.isin(_tid)].copy()

            _n_all = int(len(df))
            _n_fit = int(len(_fit_df))
            if _n_fit > 0:
                _pos = _fit_df[["catalog_id", "x", "y"]].copy()
                if "name" in _fit_df.columns:
                    _pos["name"] = _fit_df["name"]
                else:
                    _pos["name"] = _fit_df["catalog_id"].astype(str)

                _ref_fluxes: np.ndarray | None = None
                if "dao_flux" in _fit_df.columns:
                    _ref_fluxes = pd.to_numeric(_fit_df["dao_flux"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    if len(_ref_fluxes) != len(_pos):
                        _ref_fluxes = None

                _ac_policy = str(st.get("psf_ac_policy") or "p4_none")
                _psf_df = _psf_phot(
                    frame_data=d_arr,
                    frame_hdr=hdr,
                    star_positions=_pos,
                    epsf_model_path=_epsf_path,
                    error=_err_map,
                    ref_fluxes=_ref_fluxes,
                    apply_aperture_correction=(_ac_policy == "chi2_lt5_legacy"),
                    psf_ac_policy=_ac_policy,
                )
                _psf_idx = _psf_df.drop_duplicates(subset=["catalog_id"], keep="last").set_index(
                    "catalog_id"
                )
                for _col in (
                    "psf_flux",
                    "psf_flux_err",
                    "psf_chi2",
                    "psf_fit_ok",
                    "psf_ac_factor",
                    "psf_ac_n_used",
                    "psf_ac_applied",
                    "psf_ac_policy",
                    "psf_snr",
                    "psf_quality",
                    "psf_quality_fallback",
                    "psf_group_n",
                    "x_fit",
                    "y_fit",
                ):
                    if _col in _psf_idx.columns:
                        df[_col] = df["catalog_id"].map(_psf_idx[_col])
                _n_ok = int(pd.to_numeric(df.get("psf_fit_ok"), errors="coerce").fillna(0).astype(bool).sum())
            else:
                _n_ok = 0

            _psf_record.update({"n_fit": _n_fit, "n_ok": _n_ok})
            if _tid is not None:
                LOGGER.info(
                    "[ePSF] fitting %d/%d stars (targeted: variables+comps only), %d psf_fit_ok",
                    _n_fit,
                    _n_all,
                    _n_ok,
                )
            else:
                LOGGER.debug(
                    "[ePSF] frame PSF fit: %d/%d stars ok",
                    _n_ok,
                    _n_all,
                )
        except Exception as _psf_e:  # noqa: BLE001
            import traceback

            _exc_cls = type(_psf_e).__name__
            _exc_msg = str(_psf_e)
            _tb_tail = "\n".join(traceback.format_exc().splitlines()[-8:])
            _psf_record.update(
                {
                    "n_fit": _n_fit,
                    "n_ok": 0,
                    "exception_class": _exc_cls,
                    "exception_message": _exc_msg,
                    "traceback_tail": _tb_tail,
                }
            )
            LOGGER.warning(
                "[ePSF] per-frame PSF failed on frame %s: %s: %s",
                _frame_name or "?",
                _exc_cls,
                _exc_msg,
            )
            for _c in ("psf_flux", "psf_flux_err", "psf_chi2"):
                if _c not in df.columns:
                    df[_c] = float("nan")
            if "psf_fit_ok" not in df.columns:
                df["psf_fit_ok"] = False
            if "psf_quality" not in df.columns:
                df["psf_quality"] = ""
            if "psf_snr" not in df.columns:
                df["psf_snr"] = float("nan")
    else:
        for _c in ("psf_flux", "psf_flux_err", "psf_chi2"):
            if _c not in df.columns:
                df[_c] = float("nan")
        if "psf_fit_ok" not in df.columns:
            df["psf_fit_ok"] = False

    if _run_epsf:
        st["_psf_frame_record"] = _psf_record

    # -- Moffat PSF fit (Step 1 of two-step ePSF pipeline only) -----------------
    if _run_epsf and not bool(st.get("_psf_merge_only", False)):
        try:
            from config import AppConfig
            from psf_photometry import fit_moffat_psf_stars

            _cfg = AppConfig()

            _gain = float(st.get("gain", 1.0) or 1.0)
            _rn = float(st.get("read_noise", 10.0) or 10.0)
            if not math.isfinite(_gain) or _gain <= 0:
                _gain = 1.0
            if not math.isfinite(_rn) or _rn <= 0:
                _rn = 10.0
            d_arr = np.asarray(data, dtype=np.float32)
            _error_map = np.sqrt(np.abs(d_arr) / _gain + (_rn / _gain) ** 2).astype(np.float32)

            if not {"catalog_id", "x", "y"}.issubset(df.columns):
                raise ValueError("catalog missing catalog_id/x/y for Moffat PSF fit")

            try:
                _fwhm_px = float(hdr.get("VY_FWHM", 3.5) or 3.5)
            except Exception:  # noqa: BLE001
                _fwhm_px = 3.5
            if not math.isfinite(_fwhm_px) or _fwhm_px <= 0:
                _fwhm_px = 3.5

            _fit_df = df
            _tid = target_ids
            if _tid is None:
                _raw_tid = st.get("epsf_target_ids")
                if isinstance(_raw_tid, (set, frozenset, list, tuple)) and _raw_tid:
                    _tid = {str(x).strip() for x in _raw_tid if str(x).strip()}
            if _tid is not None:
                _cid_key = df["catalog_id"].fillna("").astype(str).str.strip().map(
                    lambda x: (
                        str(normalize_gaia_source_id(x)).strip()
                        if x and str(x).lower() not in ("nan", "none")
                        else ""
                    )
                )
                _fit_df = df.loc[_cid_key.isin(_tid)].copy()

            if len(_fit_df) > 0:
                _cols_want = ["catalog_id", "x", "y", "peak_dao", "dao_flux"]
                _pos = _fit_df.reindex(columns=_cols_want).copy()

                _moffat_df = fit_moffat_psf_stars(
                    frame_data=d_arr,
                    frame_hdr=hdr,
                    star_positions=_pos,
                    fwhm_guess_px=_fwhm_px,
                    error=_error_map,
                    saturate_limit_adu=float(getattr(_cfg, "saturate_limit_adu", 65000)),
                    peak_col="peak_dao",
                    chi2_limit=float(_MOFFAT_CHI2_LIMIT),  # WAVE-B STEP 6: hardcoded QC internal
                )

                # Moffat aperture correction
                if _moffat_df is not None and len(_moffat_df) > 0:
                    from psf_photometry import _compute_moffat_aperture_correction

                    _dao_arr = pd.to_numeric(
                        _fit_df.get("dao_flux", pd.Series(dtype=float)),
                        errors="coerce",
                    ).values
                    _mac_factor, _mac_n = _compute_moffat_aperture_correction(
                        _moffat_df,
                        _dao_arr,
                        chi2_limit=5.0,
                        min_flux_snr=50000.0,
                    )
                    if _mac_factor != 1.0:
                        log_event(
                            f"Moffat AC: factor={_mac_factor:.4f}, "
                            f"n_ref={_mac_n}"
                        )
                        for _col in ("moffat_flux", "moffat_flux_err"):
                            if _col in _moffat_df.columns:
                                _moffat_df[_col] = (
                                    pd.to_numeric(_moffat_df[_col], errors="coerce")
                                    * _mac_factor
                                )
                    _moffat_df["moffat_ac_factor"] = _mac_factor
                    _moffat_df["moffat_ac_n_used"] = _mac_n

                _moffat_cols = [c for c in _moffat_df.columns if str(c).startswith("moffat_")]
                if _moffat_cols and len(_moffat_df) > 0:
                    _moffat_df["_cid_key"] = _moffat_df["catalog_id"].astype(str).str.strip()
                    df["_cid_key"] = df["catalog_id"].astype(str).str.strip()
                    df = (
                        df.merge(_moffat_df[["_cid_key"] + _moffat_cols], on="_cid_key", how="left")
                        .drop(columns=["_cid_key"])
                    )
            else:
                LOGGER.debug("[PSF] Moffat skipped - no stars selected for fit")
        except Exception as _mex:  # noqa: BLE001
            log_event(f"Moffat PSF fit error (non-fatal): {_mex}")

    return df


def _vyvar_calibrate_multiprocessing_enabled() -> bool:
    """Parallel calibration uses ``spawn`` workers (errors are easy to miss). Set ``VYVAR_CALIBRATE_MP=1`` to enable."""
    v = os.environ.get("VYVAR_CALIBRATE_MP", "").strip().lower()
    return v in ("1", "true", "yes")


def _cfg_calibration_library_native_binning(cfg: Any) -> int | None:
    """Config ``calibration_library_native_binning``: ``None`` = read ``XBINNING`` from each master FITS."""
    raw = cfg.calibration_library_native_binning
    if raw is None:
        return None
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(CALIBRATION_LIBRARY_NATIVE_BINNING)



def _obs_group_key_from_light_path(src: Path) -> str:
    with fits.open(src, memmap=False) as hdul:
        return observation_group_key_from_metadata(
            fits_metadata_from_primary_header(hdul[0].header)
        )


def _light_binning_from_path(src: Path) -> int:
    with fits.open(src, memmap=False) as hdul:
        bx, _ = fits_binning_xy_from_header(hdul[0].header)
    return int(bx)


def _archive_root_from_lights_root(lights_root: Path) -> Path | None:
    p = Path(lights_root).resolve()
    if p.name.casefold() != "lights":
        return None
    parent = p.parent
    if parent.name.casefold() in ("raw", "non_calibrated"):
        return parent.parent
    return None



def _resolve_dark_path_for_light(
    *,
    src: Path,
    obs_group_key: str,
    master_dark_path: Path | None,
    master_dark_by_obs_key: dict[str, str | Path | None] | None,
) -> Path | None:
    md_use = master_dark_path
    if master_dark_by_obs_key:
        _alt = master_dark_by_obs_key.get(obs_group_key)
        if _alt is not None and str(_alt).strip() != "":
            _pa = Path(_alt)
            if _pa.is_file():
                md_use = _pa
    if md_use is not None and Path(md_use).is_file():
        return Path(md_use)
    return None


def _saturation_adu_for_cal_diag(
    hdr: fits.Header,
    *,
    db: VyvarDatabase | None,
    equipment_id: int | None,
) -> float | None:
    eq_sat: float | None = None
    if db is not None and equipment_id is not None:
        try:
            eq_sat = db.get_equipment_saturation_adu(int(equipment_id))
        except Exception:  # noqa: BLE001
            eq_sat = None
    lim, _src = _effective_saturation_limit(
        hdr,
        fallback_adu=None,
        equipment_saturate_adu=eq_sat,
    )
    return lim


def _cal_diag_session_from_export(blob: dict[str, Any] | None) -> CalDiagSession:
    session = CalDiagSession()
    if not blob:
        return session
    for k, raw in (blob.get("keys") or {}).items():
        try:
            session.gate_results[str(k)] = CalDiagGateResult(**raw)
        except TypeError:
            continue
    session.aborted_groups = set(str(x) for x in (blob.get("aborted_groups") or []))
    return session


def _cal_diag_export_for_workers(session: CalDiagSession) -> dict[str, Any] | None:
    if not session.gate_results and not session.aborted_groups:
        return None
    return session.json_export()


# ``_calibrate_one_light_*``: explicit ``None`` = read master FITS; omit param = library default (1x1).
_CALIB_MASTER_NB_UNSET = object()


def _log_calibration_io_preflight(
    *,
    calibrated_root: Path,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
) -> None:
    """Log resolved master paths and best-effort write access to ``calibrated_root``."""
    try:
        calibrated_root.mkdir(parents=True, exist_ok=True)
        probe = calibrated_root / ".vyvar_write_probe"
        probe.write_text("ok", encoding="ascii")
        probe.unlink(missing_ok=True)
        log_event(f"Kalibracia - zapis OK: {calibrated_root.resolve()}")
    except OSError as exc:
        log_event(f"Kalibracia - CHYBA PRAVO ZAPISU (calibrated): {calibrated_root.resolve()} -> {exc}")

    if master_dark_path is not None:
        md = Path(master_dark_path)
        md_r = md.resolve()
        ok = md_r.is_file()
        log_event(f"MasterDark: {md_r} (exists={ok})")
        if not ok:
            log_event("MasterDark: subor neexistuje - dark sa neaplikuje (flat-only / copy-only podla nastavenia).")

    for fk, fp in sorted((masterflat_by_filter or {}).items(), key=lambda x: str(x[0])):
        if fp is None:
            log_event(f"MasterFlat[{fk!r}]: (ziadna cesta)")
            continue
        p = Path(fp)
        pr = p.resolve()
        ok = pr.is_file()
        log_event(f"MasterFlat[{fk!r}]: {pr} (exists={ok})")
        if not ok:
            log_event(f"MasterFlat[{fk!r}]: subor neexistuje - pre tento filter sa flat neaplikuje.")

# Public aliases (historically some callers used ``pipeline.parse_user_*`` / ``pointing_hint_from_header``).
pointing_hint_from_header = _pointing_hint_from_header


LOGGER = logging.getLogger(__name__)

# WAVE-B STEP 6 (HARDCODE): solver / QC internals, formerly AppConfig knobs. Fixed to their
# long-standing defaults (never tuned in config history); solver mechanics, not user tuning.
_MOFFAT_CHI2_LIMIT = 50.0                              # was cfg.moffat_chi2_limit (50.0)
_SKY_ADU_FALLBACK = 1581.6                             # was cfg.sky_adu_fallback (1581.6)
_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG = 1.0  # was cfg.masterstar_solver_use_draft_median_if_hint_sep_deg (1.0)
_MASTERSTAR_OPTIMIZER_MIRROR_EXTRA_LOG = True         # was cfg.masterstar_optimizer_mirror_extra_log (True)
_MASTERSTAR_PLATESOLVE_PREWRITE_RMS_MAX_PX = 30.0     # was cfg.masterstar_platesolve_prewrite_rms_max_px (30.0)
_MASTERSTAR_PLATESOLVE_PREWRITE_RELAXED_RMS_MAX_PX = 35.0  # was cfg.masterstar_platesolve_prewrite_relaxed_rms_max_px (35.0)
_MASTERSTAR_PLATESOLVE_NN_REFINE_MAX_RMS_PX = None    # was cfg.masterstar_platesolve_nn_refine_max_rms_px (None)
_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO = 1.15          # was cfg.masterstar_sip_force_rms_guard_ratio (1.15)
_PLATESOLVE_ANISOTROPY_THRESHOLD = 1.3                # was cfg.platesolve_anisotropy_threshold (1.3)


def _pipeline_ui_info(msg: str) -> None:
    """Log always; during a long job route to the bottom footer instead of ``st.info``."""
    log_event(msg)
    try:
        import streamlit as st

        fs = st.session_state.get("vyvar_footer_state")
        if isinstance(fs, dict) and fs.get("running"):
            st.session_state["vyvar_footer_state"] = {**fs, "status_detail": str(msg)[:800]}
            _fn = st.session_state.get("vyvar_ui_rerender_footer")
            if callable(_fn):
                _fn()
                return
        st.info(msg)
    except Exception:  # noqa: BLE001
        pass


def _pipeline_ui_error(msg: str) -> None:
    """Log always; mirror text to footer during a running job, then ``st.error``."""
    log_event(msg)
    try:
        import streamlit as st

        fs = st.session_state.get("vyvar_footer_state")
        if isinstance(fs, dict) and fs.get("running"):
            st.session_state["vyvar_footer_state"] = {**fs, "status_detail": str(msg)[:800]}
            _fn = st.session_state.get("vyvar_ui_rerender_footer")
            if callable(_fn):
                _fn()
        st.error(msg)
    except Exception:  # noqa: BLE001
        pass


def _ensure_parent_dirs_for_aligned_fits(out_path: Path) -> None:
    """Ensure parent folders exist for nested outputs under ``detrended_aligned/lights/...``."""
    os.makedirs(str(out_path.parent), exist_ok=True)


def _assert_alignment_produced_fits(aligned_root: Path) -> None:
    """Fail fast if alignment wrote no FITS under the group folder (recursive; not only top-level)."""
    n = len(_iter_fits_recursive(aligned_root))
    if n == 0:
        raise RuntimeError(
            "Alignment zlyhal - nenasli sa ziadne vystupne subory! "
            f"(ziadne FITS v {aligned_root.resolve()} vratane podadresarov.)"
        )


def _match_and_crop_pair(a: "np.ndarray", b: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Crop 2D arrays to common smallest shape (top-left)."""
    import numpy as np

    a2 = np.asarray(a)
    b2 = np.asarray(b)
    if a2.ndim != 2 or b2.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shapes {a2.shape} and {b2.shape}")
    h = min(a2.shape[0], b2.shape[0])
    w = min(a2.shape[1], b2.shape[1])
    return a2[:h, :w], b2[:h, :w]


def _safe_filter_token(text: str) -> str:
    t = (text or "").strip()
    if not t or t.lower() in {"unknown", "none", "nan"}:
        return "NoFilter"
    return t


def observation_group_key_from_metadata(meta: dict[str, Any]) -> str:
    """Must match ``importer.observation_group_key`` (FILTER, EXPTIME, binning from FITS / metadata dict)."""
    flt = _safe_filter_token(str(meta.get("filter") or "NoFilter"))
    try:
        e = float(meta.get("exposure", 0.0))
    except (TypeError, ValueError):
        e = 0.0
    b = max(1, int(meta.get("binning", 1) or 1))
    return f"{flt}|{e:g}|{b}"


def _iter_light_fits(lights_root: Path) -> list[Path]:
    """Collect lights FITS under lights_root (including nested subdirs, e.g. ``filter_exp_binning``)."""
    return _iter_fits_recursive(lights_root)


def _summarize_lights_binning_from_headers(paths: list[Path]) -> dict[str, Any]:
    """Count ``(XBINNING,YBINNING)`` read directly from each primary FITS header."""
    counts: dict[tuple[int, int], int] = {}
    samples: dict[tuple[int, int], Path] = {}
    errors = 0
    for fp in paths:
        try:
            with fits.open(fp, memmap=False) as hdul:
                xb, yb = fits_binning_xy_from_header(hdul[0].header)
            key = (int(xb), int(yb))
            counts[key] = counts.get(key, 0) + 1
            samples.setdefault(key, fp)
        except Exception:  # noqa: BLE001
            errors += 1
    return {"counts": counts, "samples": samples, "errors": errors}


def log_lights_binning_from_headers_preflight(
    paths: list[Path],
    *,
    context: str = "VYVAR",
) -> tuple[int, int] | None:
    """Log binning from FITS headers at pipeline start (before DB/cache overlay)."""
    if not paths:
        return None
    summary = _summarize_lights_binning_from_headers(paths)
    counts: dict[tuple[int, int], int] = summary.get("counts") or {}
    if not counts:
        if int(summary.get("errors") or 0) > 0:
            log_event(f"{context} - binning z hlaviciek: nepodarilo sa precitat ziadny FITS.")
        return None
    log_event(f"{context} - binning z hlaviciek FITS ({len(paths)} suborov):")
    samples: dict[tuple[int, int], Path] = summary.get("samples") or {}
    for (xb, yb), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        sample = samples.get((xb, yb))
        sn = sample.name if sample is not None else "?"
        log_event(f"  {xb}x{yb}: {n}x (priklad: {sn})")
    if len(counts) > 1:
        log_event(
            f"{context} - POZOR: zmiesane binning rezimy v davke - "
            "kalibracia / master match pouzivaju binning z hlavicky kazdeho suboru."
        )
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    return best_key


def _pick_light_for_metadata_diagnostic(paths: list[Path]) -> Path:
    """Representative light for infolog diagnostics: predominant header binning, not sort order."""
    if not paths:
        raise ValueError("empty paths")
    summary = _summarize_lights_binning_from_headers(paths)
    counts: dict[tuple[int, int], int] = summary.get("counts") or {}
    if not counts:
        return paths[0]
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    samples: dict[tuple[int, int], Path] = summary.get("samples") or {}
    return samples.get(best_key) or paths[0]


def _filter_light_paths_maybe(
    files: list[Path],
    only_paths: Sequence[Path | str] | None,
) -> list[Path]:
    """If ``only_paths`` is set, keep only those members (resolved path match, casefold on Windows)."""
    if only_paths is None:
        return files

    def _norm(p: Path) -> str:
        try:
            return str(p.resolve()).casefold()
        except OSError:
            return str(p).casefold()

    by_norm: dict[str, Path] = {}
    for fp in files:
        by_norm.setdefault(_norm(fp), fp)

    ordered: list[Path] = []
    seen: set[str] = set()
    for x in only_paths:
        px = Path(x)
        k = _norm(px)
        if k in seen:
            continue
        hit = by_norm.get(k)
        if hit is not None:
            ordered.append(hit)
            seen.add(k)
            continue
        for fp in files:
            fk = _norm(fp)
            if fk in seen:
                continue
            try:
                if os.path.samefile(fp, px):
                    ordered.append(fp)
                    seen.add(fk)
                    break
            except OSError:
                continue
    return ordered


def _archive_raw_to_calibrated_light(
    archive: Path,
    raw_fp: Path | str,
) -> tuple[Path, Path] | None:
    """Map archived raw light path to ``(calibrated_fits, lights_root_under_archive)``."""
    ap = Path(archive).expanduser().resolve()
    raw_path = Path(raw_fp)
    try:
        r = raw_path.resolve() if raw_path.is_file() else (ap / raw_path).resolve()
    except OSError:
        r = raw_path if raw_path.is_file() else (ap / raw_path)
    pairs: tuple[tuple[Path, Path], ...] = (
        (ap / "non_calibrated" / "lights", ap / "calibrated" / "lights"),
        (ap / "Raw" / "lights", ap / "calibrated" / "lights"),
    )
    for raw_root, cal_root in pairs:
        if not raw_root.is_dir():
            continue
        try:
            rel = r.relative_to(raw_root.resolve())
        except ValueError:
            continue
        cand = cal_root / rel
        if cand.is_file():
            return cand, cal_root
    return None


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


def _resolve_draft_light_raw_path(archive: Path, file_path: Path | str) -> Path | None:
    """Resolve raw light FITS on disk for in-RAM calibration (``manifest files[].FILE_PATH``)."""
    ap = Path(archive).expanduser()
    p = Path(str(file_path))
    if p.is_file():
        return p
    q = ap / p
    if q.is_file():
        return q
    name = p.name
    for sub in (ap / "non_calibrated" / "lights", ap / "Raw" / "lights"):
        cand = sub / name
        if cand.is_file():
            return cand
        # Setup subfolder (e.g. Raw/lights/NoFilter_60_2/BO_CVn_Light_001.fits)
        try:
            for hit in sub.rglob(name):
                if hit.is_file():
                    return hit
        except OSError:
            pass
    return None


def _sat_ctx_from_worker(st: dict[str, Any]) -> Any | None:
    raw = st.get("sat_diag_ctx_dict")
    if not raw:
        return None
    try:
        from sat_diag import PileupResult, SatDiagContext  # noqa: PLC0415

        pileup_raw = raw.get("pileup")
        pileup = PileupResult(**pileup_raw) if isinstance(pileup_raw, dict) else None
        d = dict(raw)
        d.pop("pileup", None)
        d["pileup"] = pileup
        return SatDiagContext(**{k: v for k, v in d.items() if k in SatDiagContext.__dataclass_fields__})
    except Exception:  # noqa: BLE001
        return None


def _load_raw_for_frame(st: dict[str, Any], fname: str) -> Any | None:
    arch = str(st.get("sat_diag_archive") or "").strip()
    if not arch:
        return None
    raw_p = _resolve_draft_light_raw_path(Path(arch), fname)
    if raw_p is None or not raw_p.is_file():
        return None
    try:
        from sat_diag import image_adu_array  # noqa: PLC0415

        with fits.open(raw_p, memmap=False) as hdul:
            if int(hdul[0].header.get("BITPIX", 0)) < 0:
                return None
            return image_adu_array(hdul[0])
    except Exception:  # noqa: BLE001
        return None


def _load_raw_hdr_for_frame(st: dict[str, Any], fname: str) -> fits.Header | None:
    arch = str(st.get("sat_diag_archive") or "").strip()
    if not arch:
        return None
    raw_p = _resolve_draft_light_raw_path(Path(arch), fname)
    if raw_p is None or not raw_p.is_file():
        return None
    try:
        with fits.open(raw_p, memmap=False) as hdul:
            return hdul[0].header.copy()
    except Exception:  # noqa: BLE001
        return None


def find_qc_metrics_csv(
    archive_path: Path | str,
    app_config: AppConfig | None = None,
    *,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Find ``qc_metrics.csv`` under draft lights root (legacy fallback: processed/lights)."""
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    candidates: list[Path] = [
        resolve_draft_lights_root(ap, draft_id=draft_id, db=db) / "qc_metrics.csv",
        ap / "processed" / "lights" / "qc_metrics.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def norm_fits_path_key(path: Path | str) -> str:
    """Normalized absolute path key for qc_metrics / allowlist joins (casefold)."""
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:
        return str(path).casefold()


def load_qc_metrics_status_by_path(qc_csv: Path | str) -> dict[str, str]:
    """Map normalized absolute FITS path -> ``status`` from ``qc_metrics.csv``."""
    p = Path(qc_csv)
    if not p.is_file():
        raise FileNotFoundError(f"qc_metrics.csv not found: {p}")
    df = pd.read_csv(p)
    if "status" not in df.columns:
        raise ValueError(f"qc_metrics.csv missing status column: {p}")
    src_col = "src" if "src" in df.columns else "dst"
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        raw = row.get(src_col)
        if raw is None or (isinstance(raw, float) and not math.isfinite(float(raw))):
            continue
        out[norm_fits_path_key(str(raw))] = str(row["status"]).strip()
    return out


def filter_files_by_qc_metrics_allowlist(
    files: Sequence[Path | str],
    qc_csv: Path | str,
) -> tuple[list[Path], dict[str, str]]:
    """Keep FITS whose ``qc_metrics.csv`` row has ``status == 'ok'`` (exact match).

    Frames on disk but absent from the CSV, or with any non-ok status, are excluded.
    Matching uses normalized resolved absolute paths (casefold).
    """
    status_map = load_qc_metrics_status_by_path(qc_csv)
    ok_keys = {k for k, v in status_map.items() if v == "ok"}
    selected: list[Path] = []
    for raw in files:
        fp = Path(raw)
        if norm_fits_path_key(fp) in ok_keys:
            selected.append(fp)
    return selected, status_map


def build_prefilter_rejected_map(
    all_paths: Sequence[Path | str],
    passing_paths: Sequence[Path | str],
    *,
    reason: str = "rejected_prefilter_fwhm",
) -> dict[Path, str]:
    """Paths in ``all_paths`` not in ``passing_paths`` -> prefilter reject reason."""
    pass_keys = {norm_fits_path_key(p) for p in passing_paths}
    out: dict[Path, str] = {}
    for raw in all_paths:
        fp = Path(raw)
        if norm_fits_path_key(fp) not in pass_keys:
            out[fp.resolve()] = str(reason)
    return out


def _archive_preprocess_lights_root(
    ap: Path | str,
    *,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path:
    """Lights root for alignment (draft lights root or legacy ``detrended/lights``)."""
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    root = draft_archive_root(Path(ap).expanduser())
    lights = resolve_draft_lights_root(root, draft_id=draft_id, db=db)
    for cand in (lights, root / "detrended" / "lights"):
        if cand.is_dir() and _iter_fits_recursive(cand):
            return cand
    return lights


def draft_obs_group_count(archive_path: Path | str) -> int:
    """Number of obs_group subdirs under draft lights root that contain FITS."""
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    pl = resolve_draft_lights_root(ap)
    if not pl.is_dir():
        pl = ap / "processed" / "lights"
    if not pl.is_dir():
        return 0
    return sum(
        1
        for d in pl.iterdir()
        if d.is_dir() and any(_iter_fits_recursive(d))
    )


def draft_is_multi_group_obs(archive_path: Path | str) -> bool:
    return int(draft_obs_group_count(archive_path)) > 1


def resolve_masterstar_input_root(
    archive_path: Path | str,
    setup_name: str | None = None,
    *,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Pick MASTERSTAR input root under draft lights[/setup_name] with robust fallback.

    Uses ``calibrated/lights`` or ``non_calibrated/lights`` (via :func:`resolve_draft_lights_root`).

    When ``setup_name`` is explicitly provided, only that subdir is considered - no cross-group scan.
    Returns ``None`` when the requested setup is missing/empty (multi-group safe).
    """
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    lights_root = resolve_draft_lights_root(ap, draft_id=draft_id, db=db)
    setup_key = str(setup_name or "").strip()

    def _pick_under(base: Path) -> Path | None:
        if setup_key:
            cand = base / setup_key
            if cand.is_dir() and any(_iter_fits_recursive(cand)):
                return cand
            log_event(
                f"WARN: MASTERSTAR input root for setup {setup_key!r} not found under {base}; "
                "refusing cross-group fallback"
            )
            return None
        if base.is_dir():
            subdirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name.casefold())
            for sd in subdirs:
                if any(_iter_fits_recursive(sd)):
                    return sd
            if any(_iter_fits_recursive(base)):
                return base
        return None

    hit = _pick_under(lights_root)
    if hit is not None:
        return hit
    return lights_root if not setup_key else None


def _inspection_jd_from_header(hdr: fits.Header) -> float | None:
    """Julian Date (UTC) for scatter time axis."""
    for k in ("MJD-OBS", "MJD_OBS"):
        v = hdr.get(k)
        if v is not None:
            try:
                mjd = float(v)
                if math.isfinite(mjd):
                    return mjd + 2400000.5
            except (TypeError, ValueError):
                continue
    for k in ("JD-OBS", "JD_OBS", "JD"):
        v = hdr.get(k)
        if v is not None:
            try:
                jd = float(v)
                if math.isfinite(jd):
                    return jd
            except (TypeError, ValueError):
                continue
    date = hdr.get("DATE-OBS")
    if date:
        tim = hdr.get("TIME-OBS", hdr.get("TIME", "00:00:00"))
        d_s = str(date).strip()
        tim_s = str(tim).strip() if tim is not None else "00:00:00"
        if "T" in d_s:
            base = d_s.split("T", 1)[0]
            iso = f"{base}T{tim_s}"
        else:
            iso = f"{d_s}T{tim_s}"
        try:
            t = Time(iso, format="isot", scale="utc")
            return float(t.jd)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0283] DATE-OBS/Time parse failure leaves inspection_jd NULL in manifest files[] QC updates.: %s', exc)
            try:
                t = Time(d_s, scale="utc")
                return float(t.jd)
            except Exception:  # noqa: BLE001
                pass
    return None


def _exposure_sec_from_header(hdr: fits.Header) -> float | None:
    """Exposure duration in seconds from common FITS keywords."""
    for k in ("EXPTIME", "EXPOSURE", "EXPOSURE0", "EXP_TIME", "EXPTIM"):
        v = hdr.get(k)
        if v is None:
            continue
        try:
            sec = float(v)
            if math.isfinite(sec) and sec >= 0.0:
                return sec
        except (TypeError, ValueError):
            continue
    return None


def _quality_inspection_dao_metrics(fp: Path) -> dict[str, Any]:
    """Fast DAOStarFinder + moment FWHM on brightest sources; sky median; star count."""
    import numpy as np

    out0: dict[str, Any] = {
        "fwhm_mean": None,
        "sky_background": None,
        "star_count": 0,
        "inspection_jd": None,
    }
    fp = Path(fp)
    if not fp.is_file():
        return {**out0, "error": "missing_file"}
    try:
        with fits.open(fp, memmap=True) as hdul:
            hdr = hdul[0].header
            data = np.asarray(hdul[0].data, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        return {**out0, "error": str(exc)}
    return _quality_inspection_dao_metrics_array(data, hdr)


def _dao_star_table_mean_roundness(tbl: Any) -> float | None:
    """Mean ``hypot(|roundness1|, |roundness2|)`` over detected sources (DAOStarFinder table)."""
    import numpy as np

    if tbl is None or len(tbl) == 0:
        return None
    try:
        if "roundness1" not in tbl.colnames or "roundness2" not in tbl.colnames:
            return None
        r1 = np.asarray(tbl["roundness1"], dtype=np.float64)
        r2 = np.asarray(tbl["roundness2"], dtype=np.float64)
        per = np.hypot(np.abs(r1), np.abs(r2))
        ok = np.isfinite(per)
        if not np.any(ok):
            return None
        return float(np.mean(per[ok]))
    except Exception:  # noqa: BLE001
        return None



def _quality_inspection_dao_metrics_array(
    data: "np.ndarray",
    hdr: fits.Header,
) -> dict[str, Any]:
    """Same as :func:`_quality_inspection_dao_metrics` but on an in-memory calibrated image.

    FWHM is the median moment-FWHM over many star-like detections (see
    :func:`_robust_frame_fwhm_median`); not a single detection and not
    segmentation islands (which track cosmics/hot pixels after CR removal).
    """
    import numpy as np

    out: dict[str, Any] = {
        "fwhm_mean": None,
        "sky_background": None,
        "star_count": 0,
        "inspection_jd": _inspection_jd_from_header(hdr),
        "exposure_sec": _exposure_sec_from_header(hdr),
        "roundness_mean": None,
        "elongation_mean": None,
    }
    _pra, _pde, _ = _pointing_hint_from_header(hdr)
    out["ra_deg"] = float(_pra) if _pra is not None and math.isfinite(float(_pra)) else None
    out["de_deg"] = float(_pde) if _pde is not None and math.isfinite(float(_pde)) else None
    try:
        arr = np.asarray(data, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        return {**out, "error": str(exc)}

    crop = _qc_center_crop_for_stars(arr)
    finite = np.isfinite(crop)
    if not np.any(finite):
        return out
    _, med, std = plain_mean_med_std(crop[finite], sigma=3.0, maxiters=5)
    std = float(std)
    if not math.isfinite(std) or std <= 0:
        return out
    out["sky_background"] = float(med) if np.isfinite(med) else None

    rob = _robust_frame_fwhm_median(
        arr,
        max_sources=120,
        min_keep=12,
        use_center_crop=True,
    )
    if rob.get("fwhm_px") is None:
        rob = _robust_frame_fwhm_median(
            arr,
            max_sources=120,
            min_keep=5,
            use_center_crop=True,
        )
    out["star_count"] = int(rob.get("n_stars_detected") or 0)
    out["elongation_mean"] = rob.get("elongation")
    if rob.get("fwhm_px") is not None:
        out["fwhm_mean"] = float(rob["fwhm_px"])
    # roundness_mean: keep best-effort from a quick DAO table when available
    try:
        from photutils.detection import DAOStarFinder

        img2 = np.asarray(crop - float(med), dtype=np.float32)
        img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.nanmedian(img2)) < 0:
            # Only for roundness helper; prefer tail test when near zero.
            pos = float(np.count_nonzero(img2 > (4.0 * std)))
            neg = float(np.count_nonzero(img2 < (-4.0 * std)))
            if neg > (pos * 2.0) and neg > 50:
                img2 = -img2
        fwhm_guess = float(max(3.0, min(12.0, _estimate_dao_fwhm_guess(img2, std))))
        daofind = DAOStarFinder(
            fwhm=fwhm_guess,
            threshold=5.0 * std,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = daofind(img2)
        out["roundness_mean"] = _dao_star_table_mean_roundness(tbl)
    except Exception:  # noqa: BLE001
        pass
    return out


def draft_median_pointing_icrs_deg(db: VyvarDatabase, draft_id: int) -> tuple[float | None, float | None]:
    """Median ``RA`` / ``DE`` from draft light rows (degrees ICRS), for preprocess hints when headers lack coords."""
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    ras: list[float] = []
    des: list[float] = []
    for row in rows:
        try:
            ra = row.get("RA")
            de_val = row.get("DE")
            if ra is None or de_val is None:
                continue
            raf = float(ra)
            dec_f = float(de_val)
            if math.isfinite(raf) and math.isfinite(dec_f):
                ras.append(raf)
                des.append(dec_f)
        except (TypeError, ValueError):
            continue
    if not ras:
        return None, None
    import statistics

    return float(statistics.median(ras)), float(statistics.median(des))


def resolve_preprocess_target_coordinates(
    *,
    db: VyvarDatabase,
    draft_id: int | None,
    ui_ra_deg: float | None,
    ui_dec_deg: float | None,
) -> tuple[float | None, float | None]:
    """Resolve preprocess target coordinates with DB-first priority.

    Priority:
    1) ``draft manifest.CENTEROFFIELDRA/DE`` (finite pair; **0/0** sa povazuje za nevyplnene - pokracuje sa dalej)
    2) UI values
    3) median RA/DE from draft light rows
    """
    if draft_id is not None:
        try:
            drow = db.fetch_obs_draft_by_id(int(draft_id)) or {}
            ra_db = drow.get("CENTEROFFIELDRA")
            de_db = drow.get("CENTEROFFIELDDE")
            if ra_db is not None and de_db is not None:
                ra_f = float(ra_db)
                de_f = float(de_db)
                if math.isfinite(ra_f) and math.isfinite(de_f):
                    if not (abs(ra_f) < 1e-9 and abs(de_f) < 1e-9):
                        log_event(
                            f"INFO: Preprocessing forced to DB coordinates RA:{ra_f}, Dec:{de_f} for stability."
                        )
                        return float(ra_f), float(de_f)
                    log_event(
                        "DEBUG: draft manifest center is 0/0 - beriem ako nevyplnene; skusam UI, potom median z manifest files[]."
                    )
        except Exception:  # noqa: BLE001
            pass
    try:
        if ui_ra_deg is not None and ui_dec_deg is not None:
            ra_ui = float(ui_ra_deg)
            de_ui = float(ui_dec_deg)
            if math.isfinite(ra_ui) and math.isfinite(de_ui):
                if not (abs(ra_ui) < 1e-9 and abs(de_ui) < 1e-9):
                    log_event(f"DEBUG: Preprocess using UI fallback coordinates: RA={ra_ui}, Dec={de_ui}")
                    return float(ra_ui), float(de_ui)
    except (TypeError, ValueError):
        pass
    if draft_id is not None:
        try:
            med_ra, med_de = draft_median_pointing_icrs_deg(db, int(draft_id))
            if med_ra is not None and med_de is not None:
                if math.isfinite(float(med_ra)) and math.isfinite(float(med_de)):
                    log_event(
                        f"DEBUG: Preprocess using draft median coordinates: RA={float(med_ra)}, Dec={float(med_de)}"
                    )
                    return float(med_ra), float(med_de)
        except Exception:  # noqa: BLE001
            pass
    return None, None


def _estimate_fov_deg_from_header(hdr: fits.Header) -> float | None:
    """Rough field diameter in degrees from WCS pixel scale x ``NAXIS*`` (fallback when CDELT missing)."""
    try:
        n1 = int(hdr.get("NAXIS1", 0) or 0)
        n2 = int(hdr.get("NAXIS2", 0) or 0)
        if n1 <= 0 or n2 <= 0:
            return None
        d1 = hdr.get("CDELT1")
        d2 = hdr.get("CDELT2")
        if d1 is not None and d2 is not None:
            a, b = abs(float(d1)), abs(float(d2))
            if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0:
                return float(math.hypot(a * n1, b * n2))
        c11 = hdr.get("CD1_1")
        c12 = hdr.get("CD1_2")
        c21 = hdr.get("CD2_1")
        c22 = hdr.get("CD2_2")
        if None not in (c11, c12, c21, c22):
            a11, a12 = float(c11), float(c12)
            a21, a22 = float(c21), float(c22)
            if all(math.isfinite(x) for x in (a11, a12, a21, a22)):
                wx = abs(a11) * n1 + abs(a12) * n2
                hy = abs(a21) * n1 + abs(a22) * n2
                if wx > 0 and hy > 0:
                    return float(math.hypot(wx, hy))
    except (TypeError, ValueError):
        return None
    return None


def _estimate_fov_deg_from_fits_path(fp: Path) -> float | None:
    p = Path(fp)
    if not p.is_file():
        return None
    try:
        with fits.open(p, memmap=False) as hdul:
            return _estimate_fov_deg_from_header(hdul[0].header)
    except Exception:  # noqa: BLE001
        return None



def sync_obs_files_drift_arcmin_for_draft(
    db: VyvarDatabase,
    draft_id: int,
    *,
    ref_ra_deg: float | None,
    ref_de_deg: float | None,
) -> int:
    """Fill ``manifest files[].DRIFT`` (arcmin), ``DRIFT_DRA`` / ``DRIFT_DDE`` (deg plane offsets vs median). Clears when ref missing."""
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = 0
    if ref_ra_deg is None or ref_de_deg is None:
        for row in rows:
            db.update_obs_file_quality_by_id(int(draft_id), int(row["ID"]), clear_drift=True)
            n += 1
        return n
    try:
        rref0, dref0 = float(ref_ra_deg), float(ref_de_deg)
    except (TypeError, ValueError):
        for row in rows:
            db.update_obs_file_quality_by_id(int(draft_id), int(row["ID"]), clear_drift=True)
            n += 1
        return n
    if not (math.isfinite(rref0) and math.isfinite(dref0)):
        for row in rows:
            db.update_obs_file_quality_by_id(int(draft_id), int(row["ID"]), clear_drift=True)
            n += 1
        return n
    rref, dref = rref0, dref0
    for row in rows:
        rid = int(row["ID"])
        try:
            ra = row.get("RA")
            de = row.get("DE")
            if ra is None or de is None:
                db.update_obs_file_quality_by_id(int(draft_id), rid, clear_drift=True)
                n += 1
                continue
            raf, def_ = float(ra), float(de)
            if not (math.isfinite(raf) and math.isfinite(def_)):
                db.update_obs_file_quality_by_id(int(draft_id), rid, clear_drift=True)
                n += 1
                continue
            dra_deg = ((raf - rref) + 180.0) % 360.0 - 180.0
            dde_deg = def_ - dref
            dra_plane = dra_deg * math.cos(math.radians(dref))
            d_arc = math.hypot(dra_plane, dde_deg) * 60.0
            if not math.isfinite(d_arc):
                db.update_obs_file_quality_by_id(int(draft_id), rid, clear_drift=True)
            else:
                db.update_obs_file_quality_by_id(int(draft_id), 
                    rid,
                    drift_arcmin=float(d_arc),
                    drift_dra_deg=float(dra_plane),
                    drift_dde_deg=float(dde_deg),
                )
            n += 1
        except (TypeError, ValueError):
            db.update_obs_file_quality_by_id(int(draft_id), rid, clear_drift=True)
            n += 1
    return n


def generate_observation_hash(db: VyvarDatabase, draft_id: int) -> str:
    """Deterministic processing hashtag: camera (equipment) + telescope + filter/exptime set + JD start.

    JD start is the minimum finite ``INSPECTION_JD`` among draft lights, else ``draft manifest.OBSERVATIONSTARTJD``.
    Filter/exptime signature is sorted unique ``(FILTER, EXPTIME)`` pairs from ``manifest files[]`` lights.
    """
    import hashlib

    drow = db.fetch_obs_draft_by_id(int(draft_id))
    if drow is None:
        raise ValueError(f"Draft id={int(draft_id)} not found")
    id_eq = int(drow.get("ID_EQUIPMENTS") or 0)
    id_tel = int(drow.get("ID_TELESCOPE") or 0)
    lights = db.fetch_draft_light_rows_for_quality(int(draft_id))
    pair_set: set[tuple[str, float]] = set()
    jds: list[float] = []
    for r in lights:
        flt = str(r.get("FILTER") or "").strip() or "None"
        ex_raw = r.get("EXPTIME")
        try:
            ex_f = float(ex_raw) if ex_raw is not None else 0.0
        except (TypeError, ValueError):
            ex_f = 0.0
        if math.isfinite(ex_f):
            pair_set.add((flt, round(ex_f, 6)))
        jd_raw = r.get("INSPECTION_JD")
        try:
            jf = float(jd_raw) if jd_raw is not None else float("nan")
            if math.isfinite(jf):
                jds.append(jf)
        except (TypeError, ValueError):
            pass
    pairs_sorted = "|".join(f"{a}:{b:.6f}" for a, b in sorted(pair_set))
    try:
        jd0 = float(drow.get("OBSERVATIONSTARTJD") or 0.0)
    except (TypeError, ValueError):
        jd0 = 0.0
    if jds:
        jd0 = min(jds)
    if not math.isfinite(jd0):
        jd0 = 0.0
    payload = f"{id_eq}|{id_tel}|{pairs_sorted}|{jd0:.8f}"
    digest = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    date_prefix = VyvarDatabase._jd_to_yyyymmdd(jd0)
    return f"{date_prefix}_{digest}"


def run_quality_analysis(
    *,
    db: VyvarDatabase,
    draft_id: int,
    archive_path: Path | str,
    progress_cb: Callable[[int, int, str], None] | None = None,
    roundness_reject_above: float | None = None,
) -> dict[str, Any]:
    """Per-draft light: DAO metrics -> ``manifest files[]``; FWHM x1.5 and optional DAO roundness auto-reject.

    When Streamlit is available, updates ``st.session_state`` keys ``fwhm_threshold``, ``center_ra``,
    ``center_de`` from computed medians (same as RAM QC path).

    The Quality Dashboard assigns ``frame_index`` 1...N in the same order as
    :meth:`VyvarDatabase.fetch_draft_light_rows_for_quality` (stable table <-> plot alignment).

    After metrics, :func:`sync_obs_files_drift_arcmin_for_draft` fills ``DRIFT`` / ``DRIFT_DRA`` / ``DRIFT_DDE``.
    """
    import numpy as np

    _rn = 1.25 if roundness_reject_above is None else float(roundness_reject_above)
    rlim_active = math.isfinite(_rn) and _rn > 0.0

    ap = Path(archive_path).expanduser()
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = len(rows)
    rid_to_scan: dict[int, int] = {}
    for _r in rows:
        try:
            rid_to_scan[int(_r["ID"])] = int(_r.get("ID_SCANNING") or 0)
        except Exception:  # noqa: BLE001
            continue
    fwhm_by_id: dict[int, float] = {}
    roundness_by_id: dict[int, float] = {}
    errors: list[str] = []
    fov_sample_deg: float | None = None

    for i, row in enumerate(rows, start=1):
        rid = int(row["ID"])
        raw = Path(str(row.get("FILE_PATH") or ""))
        tgt = _resolve_light_fits_for_quality_inspection(ap, raw)
        if tgt is None:
            errors.append(f"missing {raw.name}")
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Skip missing {raw.name}")
            continue
        m = _quality_inspection_dao_metrics(tgt)
        if fov_sample_deg is None and not m.get("error"):
            fov_sample_deg = _estimate_fov_deg_from_fits_path(tgt)
        if m.get("error"):
            errors.append(f"{tgt.name}: {m['error']}")
        _rm = m.get("roundness_mean")
        _rm_db = float(_rm) if _rm is not None and math.isfinite(float(_rm)) and float(_rm) >= 0.0 else None
        _el = m.get("elongation_mean")
        _el_db = float(_el) if _el is not None and math.isfinite(float(_el)) and float(_el) > 0.0 else None
        db.update_obs_file_quality_by_id(int(draft_id), rid,
            fwhm=m.get("fwhm_mean"),
            sky_level=m.get("sky_background"),
            star_count=int(m.get("star_count") or 0),
            rejected_auto=0,
            inspection_jd=m.get("inspection_jd"),
            ra_deg=m.get("ra_deg"),
            de_deg=m.get("de_deg"),
            exptime_sec=m.get("exposure_sec"),
            roundness_mean=_rm_db,
            elongation_mean=_el_db,
        )
        fv = m.get("fwhm_mean")
        if fv is not None and math.isfinite(float(fv)):
            fwhm_by_id[rid] = float(fv)
        if _rm_db is not None:
            roundness_by_id[rid] = float(_rm_db)
        if progress_cb is not None:
            progress_cb(i, n, f"Quality {tgt.name}")

    vals = [v for v in fwhm_by_id.values() if math.isfinite(v) and v > 0]
    med: float | None
    if vals:
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
        if not math.isfinite(med) or med <= 0:
            med = None
    else:
        med = None

    thr = med * 1.5 if med is not None else None
    light_rows2 = db.fetch_draft_light_rows_for_quality(int(draft_id))
    auto_n = 0
    for row in light_rows2:
        rid = int(row["ID"])
        rej = 0
        if med is not None and thr is not None:
            fv = fwhm_by_id.get(rid)
            if fv is not None and math.isfinite(float(fv)) and float(fv) > thr:
                rej = 1
        if rlim_active:
            rv = roundness_by_id.get(rid)
            if rv is not None and math.isfinite(float(rv)) and float(rv) > _rn:
                rej = 1
        if rej:
            auto_n += 1
        db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=rej)

    med_ra, med_de = draft_median_pointing_icrs_deg(db, int(draft_id))
    sync_obs_files_drift_arcmin_for_draft(db, int(draft_id), ref_ra_deg=med_ra, ref_de_deg=med_de)
    _dl_suggest = 5.0
    if fov_sample_deg is not None and math.isfinite(float(fov_sample_deg)) and float(fov_sample_deg) > 0:
        _dl_suggest = max(0.5, min(180.0, 0.1 * float(fov_sample_deg) * 60.0))
    result: dict[str, Any] = {
        "draft_id": int(draft_id),
        "n_lights": n,
        "n_successful_fwhm": int(len(fwhm_by_id)),
        "median_fwhm": med,
        "median_ra_deg": med_ra,
        "median_de_deg": med_de,
        "auto_rejected": int(auto_n),
        "errors": errors,
        "suggested_drift_limit_arcmin": float(_dl_suggest),
    }
    try:
        import streamlit as st

        _upd: dict[str, Any] = {}
        if med is not None and math.isfinite(float(med)) and float(med) > 0:
            _upd["fwhm_threshold"] = float(med)
        if med_ra is not None and math.isfinite(float(med_ra)):
            _upd["center_ra"] = float(med_ra)
            _upd["cur_draft_ra"] = float(med_ra)
        if med_de is not None and math.isfinite(float(med_de)):
            _upd["center_de"] = float(med_de)
            _upd["cur_draft_de"] = float(med_de)
        _upd["drift_limit_arcmin"] = float(_dl_suggest)
        st.session_state.update(_upd)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
    return result


def _perf10_lookup_qc(
    perf10_qc_results: dict[str, dict[str, Any]],
    archive: Path,
    file_path: str,
) -> dict[str, Any] | None:
    """Match ``perf10_qc_results`` keyed by resolved raw FITS path."""
    raw_fp = _resolve_draft_light_raw_path(archive, file_path)
    if raw_fp is None:
        return None
    for key in (str(raw_fp.resolve()), str(raw_fp)):
        qc = perf10_qc_results.get(key)
        if isinstance(qc, dict) and qc:
            return qc
    return None


def apply_perf10_dao_qc_to_obs_files(
    *,
    db: VyvarDatabase,
    draft_id: int,
    archive_path: Path | str,
    perf10_qc_results: dict[str, dict[str, Any]],
    roundness_reject_above: float | None = 1.25,
) -> dict[str, Any]:
    """Write calibration-time DAO QC to ``manifest files[]`` (same columns as RAM QC step 5)."""
    import numpy as np

    ap = Path(archive_path).expanduser()
    _rn = 1.25 if roundness_reject_above is None else float(roundness_reject_above)
    rlim_active = math.isfinite(_rn) and _rn > 0.0

    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    fwhm_by_id: dict[int, float] = {}
    roundness_by_id: dict[int, float] = {}
    n_updated = 0
    errors: list[str] = []

    for row in rows:
        rid = int(row["ID"])
        qc = _perf10_lookup_qc(perf10_qc_results, ap, str(row.get("FILE_PATH") or ""))
        if qc is None or qc.get("error"):
            continue
        try:
            _rm = qc.get("roundness_mean")
            _rm_db = (
                float(_rm)
                if _rm is not None and math.isfinite(float(_rm)) and float(_rm) >= 0.0
                else None
            )
            _el = qc.get("elongation_mean")
            _el_db = (
                float(_el)
                if _el is not None and math.isfinite(float(_el)) and float(_el) > 0.0
                else None
            )
            db.update_obs_file_quality_by_id(int(draft_id), rid,
                fwhm=qc.get("fwhm_mean"),
                sky_level=qc.get("sky_background"),
                star_count=int(qc.get("star_count") or 0),
                rejected_auto=0,
                inspection_jd=qc.get("inspection_jd"),
                ra_deg=qc.get("ra_deg"),
                de_deg=qc.get("de_deg"),
                exptime_sec=qc.get("exposure_sec"),
                roundness_mean=_rm_db,
                elongation_mean=_el_db,
            )
            n_updated += 1
            fv = qc.get("fwhm_mean")
            if fv is not None and math.isfinite(float(fv)):
                fwhm_by_id[rid] = float(fv)
            if _rm_db is not None:
                roundness_by_id[rid] = float(_rm_db)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('FILE_PATH')}: {exc}")

    logging.info("[PERF-10] Updated manifest files[] QC for %d frames during calibration", n_updated)

    vals = [v for v in fwhm_by_id.values() if math.isfinite(v) and v > 0]
    med: float | None
    if vals:
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
        if not math.isfinite(med) or med <= 0:
            med = None
    else:
        med = None

    thr = med * 1.5 if med is not None else None
    auto_n = 0
    for row in rows:
        rid = int(row["ID"])
        rej = 0
        if med is not None and thr is not None:
            fv = fwhm_by_id.get(rid)
            if fv is not None and math.isfinite(float(fv)) and float(fv) > thr:
                rej = 1
        if rlim_active:
            rv = roundness_by_id.get(rid)
            if rv is not None and math.isfinite(float(rv)) and float(rv) > _rn:
                rej = 1
        if rej:
            auto_n += 1
        db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=rej)

    med_ra, med_de = draft_median_pointing_icrs_deg(db, int(draft_id))
    sync_obs_files_drift_arcmin_for_draft(db, int(draft_id), ref_ra_deg=med_ra, ref_de_deg=med_de)
    db._try_refresh_draft_manifest(int(draft_id))
    _dl_suggest = 5.0
    return {
        "draft_id": int(draft_id),
        "n_lights": len(rows),
        "n_successful_fwhm": int(len(fwhm_by_id)),
        "median_fwhm": med,
        "median_ra_deg": med_ra,
        "median_de_deg": med_de,
        "auto_rejected": int(auto_n),
        "errors": errors,
        "perf10_n_updated": int(n_updated),
        "suggested_drift_limit_arcmin": float(_dl_suggest),
    }


def run_draft_ram_calibration_qc_to_obs_files(
    *,
    db: VyvarDatabase,
    draft_id: int,
    archive_path: Path | str,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
    masterflat_by_obs_key: dict[str, str | Path | None] | None = None,
    master_dark_by_obs_key: dict[str, str | Path | None] | None = None,
    equipment_id: int | None = None,
    pipeline_config: AppConfig | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    roundness_reject_above: float | None = None,
) -> dict[str, Any]:
    """Calibrate each draft light **in RAM only**, DAO metrics -> ``manifest files[]``; FWHM x1.5 and optional roundness reject.

    No calibrated FITS are written. Uses the same master selection rules as :func:`calibrate_lights_to_calibrated`.
    After QC, :func:`sync_obs_files_drift_arcmin_for_draft` writes ``DRIFT`` / ``DRIFT_DRA`` / ``DRIFT_DDE``.
    """
    import numpy as np

    _rn = 1.25 if roundness_reject_above is None else float(roundness_reject_above)
    rlim_active = math.isfinite(_rn) and _rn > 0.0

    ap = Path(archive_path).expanduser()
    cfg = pipeline_config or AppConfig()
    db_cal = _db_for_calibration_tasks(None)

    mf_merged: dict[str, Path | None] = {}
    for k, v in (masterflat_by_filter or {}).items():
        mf_merged[str(k)] = None if v is None else Path(v)
    for k, v in (masterflat_by_obs_key or {}).items():
        mf_merged[str(k)] = None if v is None or str(v).strip() == "" else Path(v)

    md_pre: Any = None
    md_path_ok: Path | None = None
    if master_dark_path is not None and Path(master_dark_path).exists():
        md_path_ok = Path(master_dark_path)
        with fits.open(md_path_ok, memmap=False) as hdul:
            md_pre = np.array(hdul[0].data, dtype=np.float32, copy=True)

    dark_cache: dict[str, Any] = {}
    _native_b = _cfg_calibration_library_native_binning(cfg)

    flat_cache: dict[str, Any] = {}
    flat_median_scale: dict[str, float] = {}
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = len(rows)
    fwhm_by_id: dict[int, float] = {}
    roundness_by_id: dict[int, float] = {}
    errors: list[str] = []
    fov_sample_deg: float | None = None

    cal_diag_session = CalDiagSession()
    _pregate_paths: list[Path] = []
    for row in rows:
        _rf = _resolve_draft_light_raw_path(ap, str(row.get("FILE_PATH") or ""))
        if _rf is not None:
            _pregate_paths.append(_rf)
    if _pregate_paths:
        cal_diag_session = run_cal_diag_pregate(
            _pregate_paths,
            obs_group_key_from_path=_obs_group_key_from_light_path,
            resolve_dark_path=lambda fp, og, lb: _resolve_dark_path_for_light(
                src=fp,
                obs_group_key=og,
                master_dark_path=md_path_ok,
                master_dark_by_obs_key=master_dark_by_obs_key,
            ),
            light_binning_from_path=_light_binning_from_path,
            master_binning=_native_b,
            match_and_crop_pair=_match_and_crop_pair,
            saturation_for_light=lambda fp: _saturation_adu_for_cal_diag(
                fits.getheader(fp, 0),
                db=db_cal,
                equipment_id=equipment_id,
            ),
            ui_error=_pipeline_ui_error,
        )
    write_cal_diag_json(ap, cal_diag_session)

    for i, row in enumerate(rows, start=1):
        rid = int(row["ID"])
        raw_fp = _resolve_draft_light_raw_path(ap, str(row.get("FILE_PATH") or ""))
        if raw_fp is None:
            errors.append(f"missing raw {row.get('FILE_PATH')}")
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Skip missing {Path(str(row.get('FILE_PATH') or '')).name}")
            continue

        try:
            with fits.open(raw_fp, memmap=False) as hdul:
                hdr0 = hdul[0].header
                _ok = observation_group_key_from_metadata(fits_metadata_from_primary_header(hdr0))
                _light_shape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{raw_fp.name}: header {exc}")
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Header fail {raw_fp.name}")
            continue

        if is_obs_group_aborted(cal_diag_session, _ok):
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"CAL-DIAG skip {raw_fp.name}")
            continue

        md_use = md_path_ok
        md_np_use = None
        light_bx, _ = fits_binning_xy_from_header(hdr0)
        if master_dark_by_obs_key:
            _alt = master_dark_by_obs_key.get(_ok)
            if _alt is not None and str(_alt).strip() != "":
                _pa = Path(_alt)
                if _pa.is_file():
                    md_use = _pa
        gr = gate_result_for_frame(
            cal_diag_session,
            obs_group_key=_ok,
            dark_path=md_use,
            light_binning=light_bx,
        )
        if md_use is not None and md_use.is_file():
            if (
                md_pre is not None
                and md_path_ok is not None
                and md_use.resolve() == md_path_ok.resolve()
                and _native_b is not None
                and _native_b == light_bx
                and (gr is None or convention_to_dark_mode(gr.convention) == "sum")
            ):
                md_np_use = md_pre
            else:
                md_np_use = dark_np_for_cal_diag(
                    cal_diag_session,
                    master_binning=_native_b,
                    dark_path=md_use,
                    light_binning=light_bx,
                    light_shape=_light_shape,
                    light_filename=raw_fp.name,
                    gate_result=gr,
                )

        try:
            data, hdr, _ud, _uf = _calibrate_one_light_apply_masters_in_ram(
                src=raw_fp,
                master_dark_path=md_use,
                masterflat_by_filter=mf_merged,
                flat_cache=flat_cache,
                flat_median_scale=flat_median_scale,
                md_data_preload=md_np_use,
                db=db_cal,
                id_equipments=equipment_id,
                calibration_master_native_binning=_native_b,
                cal_diag_gate_result=gr,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{raw_fp.name}: {exc}")
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Cal fail {raw_fp.name}")
            continue

        m = _quality_inspection_dao_metrics_array(data, hdr)
        if fov_sample_deg is None:
            fov_sample_deg = _estimate_fov_deg_from_header(hdr)
        _rm = m.get("roundness_mean")
        _rm_db = float(_rm) if _rm is not None and math.isfinite(float(_rm)) and float(_rm) >= 0.0 else None
        _el = m.get("elongation_mean")
        _el_db = float(_el) if _el is not None and math.isfinite(float(_el)) and float(_el) > 0.0 else None
        db.update_obs_file_quality_by_id(int(draft_id), rid,
            fwhm=m.get("fwhm_mean"),
            sky_level=m.get("sky_background"),
            star_count=int(m.get("star_count") or 0),
            rejected_auto=0,
            inspection_jd=m.get("inspection_jd"),
            ra_deg=m.get("ra_deg"),
            de_deg=m.get("de_deg"),
            exptime_sec=m.get("exposure_sec"),
            roundness_mean=_rm_db,
            elongation_mean=_el_db,
        )
        fv = m.get("fwhm_mean")
        if fv is not None and math.isfinite(float(fv)):
            fwhm_by_id[rid] = float(fv)
        if _rm_db is not None:
            roundness_by_id[rid] = float(_rm_db)
        if progress_cb is not None:
            progress_cb(i, n, f"QC RAM {raw_fp.name}")

    vals = [v for v in fwhm_by_id.values() if math.isfinite(v) and v > 0]
    med: float | None
    if vals:
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
        if not math.isfinite(med) or med <= 0:
            med = None
    else:
        med = None

    thr = med * 1.5 if med is not None else None
    light_rows_ram = db.fetch_draft_light_rows_for_quality(int(draft_id))
    auto_n = 0
    for row in light_rows_ram:
        rid = int(row["ID"])
        rej = 0
        if med is not None and thr is not None:
            fv = fwhm_by_id.get(rid)
            if fv is not None and math.isfinite(float(fv)) and float(fv) > thr:
                rej = 1
        if rlim_active:
            rv = roundness_by_id.get(rid)
            if rv is not None and math.isfinite(float(rv)) and float(rv) > _rn:
                rej = 1
        if rej:
            auto_n += 1
        db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=rej)

    med_ra, med_de = draft_median_pointing_icrs_deg(db, int(draft_id))
    sync_obs_files_drift_arcmin_for_draft(db, int(draft_id), ref_ra_deg=med_ra, ref_de_deg=med_de)
    _dl_suggest_ram = 5.0
    if fov_sample_deg is not None and math.isfinite(float(fov_sample_deg)) and float(fov_sample_deg) > 0:
        _dl_suggest_ram = max(0.5, min(180.0, 0.1 * float(fov_sample_deg) * 60.0))
    result = {
        "draft_id": int(draft_id),
        "n_lights": n,
        "n_successful_fwhm": int(len(fwhm_by_id)),
        "median_fwhm": med,
        "median_ra_deg": med_ra,
        "median_de_deg": med_de,
        "auto_rejected": int(auto_n),
        "errors": errors,
        "suggested_drift_limit_arcmin": float(_dl_suggest_ram),
    }
    try:
        rid_to_scan_local: dict[int, int] = locals().get("rid_to_scan", {})  # type: ignore[assignment]
        by_scan: dict[int, dict[str, Any]] = {}
        for rid, sid in rid_to_scan_local.items():
            if sid <= 0:
                continue
            rec = by_scan.setdefault(int(sid), {"n_rows": 0, "fwhm_vals": [], "round_vals": []})
            rec["n_rows"] = int(rec["n_rows"]) + 1
            fv = fwhm_by_id.get(rid)
            if fv is not None and math.isfinite(float(fv)):
                rec["fwhm_vals"].append(float(fv))
            rv = roundness_by_id.get(rid)
            if rv is not None and math.isfinite(float(rv)):
                rec["round_vals"].append(float(rv))
        if by_scan:
            result["by_scanning"] = {
                str(sid): {
                    "n_rows": int(v["n_rows"]),
                    "median_fwhm": (
                        float(np.median(np.asarray(v["fwhm_vals"], dtype=np.float64)))
                        if v["fwhm_vals"]
                        else None
                    ),
                    "median_roundness": (
                        float(np.median(np.asarray(v["round_vals"], dtype=np.float64)))
                        if v["round_vals"]
                        else None
                    ),
                }
                for sid, v in sorted(by_scan.items())
            }
    except Exception:  # noqa: BLE001
        pass
    try:
        import streamlit as st

        _upd2: dict[str, Any] = {}
        if med is not None and math.isfinite(float(med)) and float(med) > 0:
            _upd2["fwhm_threshold"] = float(med)
        if med_ra is not None and math.isfinite(float(med_ra)):
            _upd2["center_ra"] = float(med_ra)
            _upd2["cur_draft_ra"] = float(med_ra)
        if med_de is not None and math.isfinite(float(med_de)):
            _upd2["center_de"] = float(med_de)
            _upd2["cur_draft_de"] = float(med_de)
        _upd2["drift_limit_arcmin"] = float(_dl_suggest_ram)
        st.session_state.update(_upd2)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
    db._try_refresh_draft_manifest(int(draft_id))
    return result


def calibrated_paths_for_draft_apply_filters(
    archive_path: Path | str,
    db: VyvarDatabase,
    draft_id: int,
    *,
    fwhm_max_px: float,
    drift_max_arcmin: float = 0.0,
    source_dir: Path | str | None = None,
) -> tuple[list[Path], list[Path]]:
    """Calibrated FITS paths that pass QC: ``IS_REJECTED`` 0/NULL + optional ``FWHM`` cap.

    When ``fwhm_max_px`` <= 0, no FWHM filter (still excludes manual reject).
    When ``fwhm_max_px`` > 0, rows with finite ``FWHM`` > max are excluded.
    Rows with missing ``FWHM`` (NULL/NaN) are kept.

    ``drift_max_arcmin`` is kept only for backward-compatible callers and is ignored.
    """
    ap = Path(archive_path).expanduser()
    fwhm_limit = fwhm_max_px
    log_event(
        f"DEBUG: Hladam subory pre Draft {draft_id} s FWHM <= {fwhm_limit}"
    )
    from draft_provenance import reset_manifest_light_is_rejected

    reset_manifest_light_is_rejected(db, int(draft_id))
    log_event(f"DEBUG: Draft {draft_id} reset: all files set to IS_REJECTED=0 before filtering.")
    cal_paths: list[Path] = []
    if source_dir is None:
        from draft_provenance import resolve_draft_lights_root

        src_root = resolve_draft_lights_root(ap, draft_id=int(draft_id), db=db).resolve()
    else:
        src_root = Path(source_dir).expanduser().resolve()
    lim_active = bool(fwhm_max_px is not None and float(fwhm_max_px) > 0)
    lim_v = float(fwhm_max_px) if lim_active else 0.0
    _ = drift_max_arcmin  # backward compatibility: cleaning is FWHM-only
    all_rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    if lim_active:
        _nonnull_cnt = sum(1 for _r in all_rows if _r.get("FWHM") is not None)
        if _nonnull_cnt > 0:
            rows = [
                _r
                for _r in all_rows
                if _r.get("IS_REJECTED") in (None, 0)
                and _r.get("FWHM") is not None
                and float(_r["FWHM"]) <= float(lim_v)
            ]
        else:
            rows = [
                _r
                for _r in all_rows
                if _r.get("IS_REJECTED") in (None, 0)
                and (_r.get("FWHM") is None or float(_r["FWHM"]) <= float(lim_v))
            ]
        log_event(
            f"DEBUG: Preprocess DB filter selected {len(rows)} rows (limit={float(lim_v):.3f} px, strict, nonnull_fwhm={_nonnull_cnt})."
        )
    else:
        log_event("DEBUG: Preprocess filter - FWHM limit disabled (0/None); keeping all non-rejected frames.")
        rows = [_r for _r in all_rows if _r.get("IS_REJECTED") in (None, 0)]
        log_event(f"DEBUG: Preprocess DB filter selected {len(rows)} rows (FWHM disabled).")

    for row in rows:
        raw = Path(str(row.get("FILE_PATH") or ""))
        m = _archive_raw_to_calibrated_light(ap, raw)
        if m is not None:
            cand_m, _cand_root = m
            try:
                if cand_m.is_file() and cand_m.resolve().is_relative_to(src_root):
                    cal_paths.append(cand_m.resolve())
                    continue
            except Exception:  # noqa: BLE001
                pass
        cand = src_root / raw.name
        if cand.is_file():
            cal_paths.append(cand)
            continue
        resolved_raw = _resolve_draft_light_raw_path(ap, raw)
        if resolved_raw is None or not resolved_raw.is_file():
            continue
        try:
            resolved_raw.relative_to(src_root)
        except ValueError:
            continue
        cal_paths.append(resolved_raw)
    if not cal_paths:
        rescue_rows = [
            _r
            for _r in db.fetch_draft_light_rows_for_quality(int(draft_id))
            if _r.get("IS_REJECTED") in (None, 0)
        ]
        if rescue_rows:
            log_event(f"INFO: Rescue pass found {len(rescue_rows)} files by ignoring QC filters.")
        for row in rescue_rows:
            raw = Path(str(row["FILE_PATH"] or ""))
            m = _archive_raw_to_calibrated_light(ap, raw)
            if m is not None:
                cand_m, _cand_root = m
                try:
                    if cand_m.is_file() and cand_m.resolve().is_relative_to(src_root):
                        cal_paths.append(cand_m.resolve())
                        continue
                except Exception:  # noqa: BLE001
                    pass
            cand = src_root / raw.name
            if cand.is_file():
                cal_paths.append(cand)
                continue
            resolved_raw = _resolve_draft_light_raw_path(ap, raw)
            if resolved_raw is None or not resolved_raw.is_file():
                continue
            try:
                resolved_raw.relative_to(src_root)
            except ValueError:
                continue
            cal_paths.append(resolved_raw)
    if not cal_paths and src_root.is_dir():
        # Final safety net: keep DB-filter intent by fuzzy matching selected DB file names to on-disk FITS.
        disk_fits = [p for p in _iter_fits_recursive(src_root) if p.is_file()]
        db_names = [Path(str(r.get("FILE_PATH") or "")).name for r in rows]
        db_stems = {Path(n).stem.lower() for n in db_names if str(n).strip()}
        matched: list[Path] = []
        for fp in disk_fits:
            s = fp.stem.lower()
            if any((st in s) or (s in st) for st in db_stems):
                matched.append(fp)
        if matched:
            cal_paths = sorted(matched)
            log_event(
                f"INFO: Disk fallback selected {len(cal_paths)} FITS files from {src_root} (DB-name matched)."
            )
        elif disk_fits:
            cal_paths = sorted(disk_fits)
            log_event(
                f"WARNING: Disk fallback used all {len(cal_paths)} FITS files from {src_root} (no DB-name match)."
            )
    return cal_paths, []


def format_memory_bytes(n: float | int) -> str:
    """Human-readable binary size (KiB = 1024 B)."""
    try:
        x = float(n)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x) or x <= 0:
        return "0 B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


def _fits_primary_pixel_count(header: fits.Header) -> int:
    """Pixel count of primary image HDU (product of NAXIS*)."""
    try:
        naxis = int(header.get("NAXIS", 0) or 0)
    except (TypeError, ValueError):
        return 0
    if naxis < 1:
        return 0
    prod = 1
    for i in range(1, naxis + 1):
        try:
            prod *= int(header.get(f"NAXIS{i}", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return int(prod)


def _available_system_ram_bytes() -> tuple[int | None, str]:
    """Best-effort free/available RAM (``psutil`` if installed)."""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available), "psutil"
    except Exception:  # noqa: BLE001
        return None, "unknown"


def estimate_memory_from_fits_headers(
    paths: list[Path],
    *,
    sample_headers: int = 48,
) -> dict[str, Any]:
    """Read only FITS primary headers; estimate float32 array size (no decompression of data)."""
    n = len(paths)
    pixels: list[int] = []
    step = max(1, n // max(1, min(sample_headers, n))) if n else 1
    for i in range(0, n, step):
        if len(pixels) >= sample_headers:
            break
        fp = paths[i]
        try:
            hdr = fits.getheader(fp, ext=0, ignore_missing_end=True, memmap=False)
            pixels.append(_fits_primary_pixel_count(hdr))
        except Exception:  # noqa: BLE001
            pixels.append(0)
    pixels = [p for p in pixels if p > 0]
    import numpy as np

    med = int(np.median(pixels)) if pixels else 0
    mx = int(np.max(pixels)) if pixels else 0
    bytes_med = med * 4
    bytes_max = mx * 4
    return {
        "n_files": n,
        "primary_pixels_median": med,
        "primary_pixels_max": mx,
        "bytes_float32_median_frame": bytes_med,
        "bytes_float32_max_frame": bytes_max,
    }


def estimate_archive_memory_profile(archive_path: str | Path) -> dict[str, Any]:
    """Rough RAM hints for QC analyze and optional platesolve ``ram_align_and_catalog`` (header-only scan).

    **QC analyze** is sequential: peak is a fewx the largest single frame (QC temps).

    **RAM handoff** (after detrending) holds ~one float32 copy per successfully aligned frame in memory
    before flush; add a margin for align-time temporaries (reference + source + aligned).
    """
    ap = Path(archive_path)
    avail, avail_src = _available_system_ram_bytes()
    out: dict[str, Any] = {
        "archive_path": str(ap),
        "available_ram_bytes": avail,
        "available_ram_human": format_memory_bytes(avail) if avail is not None else "nezname (nainstaluj ``psutil``)",
        "available_ram_source": avail_src,
        "qc_analyze": None,
        "platesolve_ram_handoff": None,
        "notes_sk": (
            "Odhad z hlaviciek FITS (NAXIS*), predpoklad prace vo float32. Skutocna spotreba zavisi od OS, "
            "Streamlit a dalsich kniznic. Po preprocess este pribudnu docasne polia pri detrendingu."
        ),
    }

    cal = ap / "calibrated" / "lights"
    try:
        from draft_provenance import resolve_draft_lights_root

        _lights_mem = resolve_draft_lights_root(ap)
        if _lights_mem.is_dir():
            cal = _lights_mem
    except Exception:  # noqa: BLE001
        pass
    if cal.is_dir():
        cfiles = _iter_light_fits(cal)
        st = estimate_memory_from_fits_headers(cfiles)
        # Working set: jeden snimok nacitany + QC (hruby faktor)
        qc_factor = 6.0
        peak_qc = int(float(st["bytes_float32_max_frame"]) * qc_factor)
        out["qc_analyze"] = {
            **st,
            "estimated_peak_bytes_sequential": peak_qc,
            "estimated_peak_human": format_memory_bytes(peak_qc),
            "explanation_sk": (
                f"Sekvencne spracovanie ~{st['n_files']} snimok; spicka RAM ~ {qc_factor:.0f}x najvacsi snimok "
                f"({format_memory_bytes(st['bytes_float32_max_frame'])} float32) pri analyze."
            ),
        }

    det = _archive_preprocess_lights_root(ap)
    if det.is_dir():
        dfiles = _iter_fits_recursive(det)
        st2 = estimate_memory_from_fits_headers(dfiles)
        n = int(st2["n_files"])
        med_b = int(st2["bytes_float32_median_frame"])
        max_b = int(st2["bytes_float32_max_frame"])
        # Buffer: jedna kopia zarovnaneho float32 na uspesny snimok (horny odhad = vsetky vstupy zarovnane)
        buffer_est = med_b * max(0, n)
        # Pocas astroalign: ref + src + aligned chvilu naraz
        align_spike = max_b * 3
        total_conservative = buffer_est + align_spike
        out["platesolve_ram_handoff"] = {
            **st2,
            "estimated_aligned_buffer_bytes": buffer_est,
            "estimated_aligned_buffer_human": format_memory_bytes(buffer_est),
            "estimated_align_spike_bytes": align_spike,
            "estimated_align_spike_human": format_memory_bytes(align_spike),
            "estimated_total_conservative_bytes": total_conservative,
            "estimated_total_conservative_human": format_memory_bytes(total_conservative),
            "explanation_sk": (
                f"Rezim 'zarovnanie + katalog v RAM': drzi ~{n} snimok x ~{format_memory_bytes(med_b)} "
                f"(median) + kratkodoba spicka pri zarovnani. Ak je to viac nez volna RAM, vypni RAM handoff."
            ),
        }

    if avail is not None and out.get("platesolve_ram_handoff"):
        tot = int(out["platesolve_ram_handoff"]["estimated_total_conservative_bytes"])
        out["platesolve_ram_handoff"]["estimate_below_available_ram"] = bool(tot <= avail)
        out["platesolve_ram_handoff"]["available_vs_estimated_ratio"] = float(avail) / float(tot) if tot > 0 else None

    if avail is not None and out.get("qc_analyze"):
        pq = int(out["qc_analyze"]["estimated_peak_bytes_sequential"])
        out["qc_analyze"]["estimate_below_available_ram"] = bool(pq <= avail)

    return out



def _path_segments_forbidden_for_masterstar_physical_source(
    p: Path,
    *,
    pre_calibrated: bool = False,
) -> bool:
    """``True`` if resolved path must not be used as MASTERSTAR physical source.

    In VYVAR-calibrated mode, ``non_calibrated/`` and ``raw/`` are forbidden (use processed/calibrated).
    In pre-calibrated mode, ``non_calibrated/lights/`` **is** the valid source tree; only ``raw/`` is forbidden.
    """
    try:
        parts = Path(p).resolve().parts
    except OSError:
        parts = Path(p).parts
    bad = {"raw"} if pre_calibrated else {"raw", "non_calibrated"}
    return any(seg.casefold() in bad for seg in parts)


def _path_is_under_tree(root: Path, p: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _pick_preferred_masterstar_basename_hit(
    hits: list[Path],
    *,
    pre_calibrated: bool = False,
) -> Path | None:
    """Pri viacerych zhodach basename zvol radsej ``proc_*.fits`` (spracovany snimok)."""
    if not hits:
        return None
    clean = [
        h
        for h in hits
        if not _path_segments_forbidden_for_masterstar_physical_source(h, pre_calibrated=pre_calibrated)
    ]
    if not clean:
        return None
    proc_first = [h for h in clean if h.name.casefold().startswith("proc_")]
    use = proc_first if proc_first else clean
    return sorted(use, key=lambda x: str(x).casefold())[0]


def _header_vy_fwhm_px(hdr: fits.Header | None) -> float | None:
    """Measured QC FWHM from ``VY_FWHM`` if present and sane."""
    if hdr is None or "VY_FWHM" not in hdr:
        return None
    try:
        v = float(hdr["VY_FWHM"])
        if math.isfinite(v) and 0.5 < v < 80.0:
            return float(v)
    except (TypeError, ValueError):
        return None
    return None


def _obs_fwhm_basename_map_from_db(db: VyvarDatabase, draft_id: int) -> dict[str, float]:
    """Map ``basename.casefold()`` -> FWHM from ``manifest files[]`` for draft lights (last row wins per name)."""
    out: dict[str, float] = {}
    for row in db.fetch_draft_light_rows_for_quality(int(draft_id)):
        try:
            fv = row.get("FWHM")
            if fv is None:
                continue
            v = float(fv)
            if not math.isfinite(v) or v <= 0.5 or v >= 80.0:
                continue
            bn = Path(str(row.get("FILE_PATH") or "")).name.casefold()
            if bn:
                out[bn] = float(v)
                if bn.startswith("proc_"):
                    out.setdefault(bn[5:], float(v))
                else:
                    out.setdefault(f"proc_{bn}", float(v))
        except (TypeError, ValueError):
            continue
    return out


def _sort_masterstar_paths_by_fwhm(
    files: list[Path],
    *,
    fwhm_by_basename: dict[str, float] | None = None,
) -> list[Path]:
    """Najlepsie prve: najnizsi VY_FWHM (hlavicka alebo DB mapa), nezname na koniec."""
    _fb = fwhm_by_basename or {}
    scored: list[tuple[float, str, Path]] = []
    for fp in files:
        v: float | None = None
        try:
            with fits.open(fp, memmap=False) as h:
                v = _header_vy_fwhm_px(h[0].header)
        except Exception:  # noqa: BLE001
            pass
        if v is None:
            _n = fp.name.casefold()
            vv = _fb.get(_n)
            if vv is None and _n.startswith("proc_"):
                vv = _fb.get(_n[5:])
            if vv is None and not _n.startswith("proc_"):
                vv = _fb.get(f"proc_{_n}")
            if vv is not None and math.isfinite(vv) and vv > 0:
                v = float(vv)
        score = float(v) if v is not None and math.isfinite(v) and v > 0 else float("inf")
        scored.append((score, str(fp).casefold(), fp))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in scored]


def _strip_external_platesolve_header(hdr: fits.Header) -> None:
    """Drop celestial WCS and common third-party plate-solve keywords (ASTAP, astrometry.net, ...).

    VYVAR must establish astrometry via :func:`vyvar_platesolver.solve_wcs_with_local_gaia` only.
    """
    strip_celestial_wcs_keys(hdr)
    strip_vendor_platesolve_metadata(hdr)
    for _k in (
        "WCSAXES",
        "WCSDIM",
        "CROTA1",
        "CROTA2",
        "WCSNAME",
        "VY_PSOLV",
        "VY_SIPRF",
    ):
        try:
            del hdr[_k]
        except KeyError:
            pass


def build_masterstar_from_detrended(
    *,
    detrended_root: Path,
    output_fits: Path,
    only_paths: "Sequence[Path | str] | None" = None,
    fwhm_fallback_px: float | None = None,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
    pre_calibrated: bool = False,
) -> dict[str, Any]:
    """Build MASTERSTAR by copying the single best processed FITS (lowest VY_FWHM)."""
    import shutil

    import numpy as np

    root = Path(detrended_root).resolve()
    _forbid_kw = {"pre_calibrated": bool(pre_calibrated)}
    all_fits = [
        fp
        for fp in _iter_fits_recursive(root)
        if _path_is_under_tree(root, fp)
        and not _path_segments_forbidden_for_masterstar_physical_source(fp, **_forbid_kw)
    ]
    files = _filter_light_paths_maybe(all_fits, only_paths)
    if not files and only_paths is not None:
        remapped: list[Path] = []
        seen_r: set[str] = set()
        for op in only_paths:
            hit = _resolve_best_effort_path_under(root, str(op), pre_calibrated=pre_calibrated)
            if hit is None or not _path_is_under_tree(root, hit):
                continue
            if _path_segments_forbidden_for_masterstar_physical_source(hit, **_forbid_kw):
                log_event(f"MASTERSTAR: mapovanie zahodilo RAW cestu -> {hit}")
                continue
            try:
                rk = str(hit.resolve()).casefold()
            except OSError:
                rk = str(hit).casefold()
            if rk in seen_r:
                continue
            seen_r.add(rk)
            remapped.append(hit)
        if remapped:
            files = remapped
            log_event(
                f"MASTERSTAR: vyber zluceny cez best-effort mapovanie ({len(files)} FITS; path filter bol prazdny)."
            )
    if files:
        if only_paths is None:
            _pipeline_ui_info(
                f"Najdenych {len(files)} suborov v {root} (vratane podadresarov)."
            )
        else:
            _pipeline_ui_info(
                f"Najdenych {len(files)} suborov pre MASTERSTAR v {root} (vyber kandidatov; "
                f"v strome {len(all_fits)} FITS celkom)."
            )
    if not files:
        if only_paths is not None:
            try:
                _want = ", ".join(Path(str(x)).name for x in only_paths[:8])
            except Exception:  # noqa: BLE001
                _want = str(only_paths)
            msg = (
                f"MASTERSTAR: explicitny vyber sa nezhoduje so subormi pod {root}: {_want}"
                + (" ..." if len(list(only_paths)) > 8 else "")
            )
            log_event(msg)
            raise FileNotFoundError(msg)
        # Cely strom (bez filtra kandidatov): deterministicky maly batch z disku.
        log_event("MASTERSTAR: bez filtra kandidatov - beriem prvych N FITS z priecinka.")
        batch = sorted(
            (
                fp
                for fp in _iter_fits_recursive(root)
                if _path_is_under_tree(root, fp)
                and not _path_segments_forbidden_for_masterstar_physical_source(fp, **_forbid_kw)
            ),
            key=lambda p: str(p).casefold(),
        )
        if batch:
            n_take = max(1, min(8, len(batch)))
            files = batch[:n_take]
            all_fits = batch

    if not files:
        if not all_fits:
            msg = (
                f"Nenasli sa ziadne FITS subory v {root} (prehladavana cesta, vratane podadresarov)."
            )
        else:
            msg = (
                f"Ziadne FITS pre MASTERSTAR po vybere kandidatov: {root} obsahuje {len(all_fits)} subor(ov), "
                "ale ziadna cesta nezodpoveda vyberu z databazy."
            )
        _pipeline_ui_info(msg)
        raise FileNotFoundError(msg)
    try:
        _comp_names = [Path(p).name for p in files]
        log_event(f"[folder] MASTERSTAR COMPOSITION: Using [{', '.join(_comp_names)}]")
    except Exception:  # noqa: BLE001
        pass

    sorted_files = _sort_masterstar_paths_by_fwhm(files, fwhm_by_basename=None)

    output_fits.parent.mkdir(parents=True, exist_ok=True)
    _cfg_ms = app_config or AppConfig()

    reference_path = Path(sorted_files[0])
    if _path_segments_forbidden_for_masterstar_physical_source(
        reference_path, **_forbid_kw
    ) or not _path_is_under_tree(root, reference_path):
        raise FileNotFoundError(
            f"MASTERSTAR: odmietnuty zdroj mimo lights stromu alebo z RAW: {reference_path}"
        )
    if not reference_path.exists():
        log_event(f"[X] MASTERSTAR FAIL: Reference file {reference_path} not found.")
        fallback_hits = [
            x
            for x in _iter_fits_recursive(root)
            if x.name == reference_path.name
            and _path_is_under_tree(root, x)
            and not _path_segments_forbidden_for_masterstar_physical_source(x, **_forbid_kw)
        ]
        if fallback_hits:
            reference_path = _pick_preferred_masterstar_basename_hit(fallback_hits) or fallback_hits[0]
            log_event(f"[OK] MASTERSTAR fallback reference found: {reference_path}")
        else:
            raise FileNotFoundError(f"MASTERSTAR reference file not found: {reference_path}")

    best_frame_fwhm_px: float | None = None
    try:
        from astropy.io.fits import getheader

        _ref_hdr = getheader(str(reference_path))
        _bfw = float(_ref_hdr.get("VY_FWHM", 0))
        if 0.5 < _bfw < 80.0:
            best_frame_fwhm_px = float(_bfw)
    except Exception:  # noqa: BLE001
        best_frame_fwhm_px = None

    try:
        with fits.open(reference_path, memmap=False) as hdul0:
            _d0 = hdul0[0].data
            _sh = getattr(_d0, "shape", None)
            if _d0 is None or _sh is None or len(_sh) != 2:
                raise ValueError("MASTERSTAR: referencny FITS nie je platny 2D primary.")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"MASTERSTAR: neviem nacitat referencny FITS: {exc}") from exc

    shutil.copy2(reference_path, output_fits)
    if len(files) <= 1:
        _ms_pick_msg = "jediny kandidat"
    else:
        _ms_pick_msg = f"kandidatov {len(files)}; najlepsi podla VY_FWHM"
    log_event(f"MASTERSTAR: cista kopia -> {output_fits} (zdroj {reference_path.name}, {_ms_pick_msg}).")

    # Auto FWHM: median VY_FWHM zo vsetkych processed FITS v root sade
    _all_processed = list(_iter_fits_recursive(root))
    _fwhm_auto_values: list[float] = []
    for _fp_fw in _all_processed:
        try:
            with fits.open(_fp_fw, memmap=False) as _hf_fw:
                _v_fw = _header_vy_fwhm_px(_hf_fw[0].header)
                if _v_fw is not None and 1.0 < _v_fw < 15.0:
                    _fwhm_auto_values.append(float(_v_fw))
        except Exception:  # noqa: BLE001
            pass
    if _fwhm_auto_values:
        _fwhm_auto = float(np.median(np.asarray(_fwhm_auto_values, dtype=np.float64)))
        log_event(
            f"MASTERSTAR: auto FWHM z {len(_fwhm_auto_values)} processed FITS "
            f"= {_fwhm_auto:.3f} px (median VY_FWHM)"
        )
    else:
        _fwhm_auto = float(fwhm_fallback_px) if fwhm_fallback_px is not None else 4.5
        log_event(f"MASTERSTAR: VY_FWHM nedostupne - fallback FWHM = {_fwhm_auto:.1f} px")

    try:
        with fits.open(output_fits, mode="update", memmap=False) as h:
            hdr = h[0].header
            # Vzdy prepis VY_FWHM hodnotou z auto vypoctu (median sady)
            vy_fwhm = float(_fwhm_auto)
            hdr["VY_FWHM"] = (vy_fwhm, "FWHM [pix] auto z medianu processed FITS")
            log_event(f"MASTERSTAR: VY_FWHM = {vy_fwhm:.3f} px zapisany do FITS.")
            # VY_FWHM_GAUSS sa doplni po plate-solve (2D Gaussian fit na MASTERSTAR).

            _strip_external_platesolve_header(hdr)
            log_event(
                "MASTERSTAR: z MASTERSTAR kopie odstraneny externy WCS/plate-solve metadata "
                "(ASTAP, astrometry.net, ...) - astrometriu nastavi vyhradne VYVAR Gaia solver."
            )

            h.flush()
    except Exception as _exc:  # noqa: BLE001
        log_event(f"MASTERSTAR: VY_FWHM zapis zlyhal: {_exc!s}")

    return {
        "masterstar_path": str(output_fits),
        "frames_used": int(len(files)),
        "reference_path": str(reference_path),
        "reference_index": 0,
        "stacked": False,
        "frames_combined": 1,
        "copied_from": str(reference_path),
        "best_frame_fwhm_px": best_frame_fwhm_px,
    }


def _update_masterstar_obs_file_status(
    *,
    cfg: AppConfig | None,
    draft_id: int | None,
    selected_ref_path: Path | None,
    wcs_ok: bool,
    n_stars: int,
) -> None:
    if draft_id is None or selected_ref_path is None:
        return
    try:
        from draft_provenance import load_or_init_manifest, patch_draft_manifest, resolve_draft_dir_for_id

        app_cfg = cfg or AppConfig()
        db = VyvarDatabase(Path(app_cfg.database_path))
        ar = getattr(app_cfg, "archive_root", None)
        if ar:
            db._archive_root_override = Path(str(ar)).expanduser().resolve()
        try:
            root = resolve_draft_dir_for_id(db, int(draft_id))
            if root is None:
                return
            manifest = load_or_init_manifest(root, int(draft_id))
            files = manifest.get("files")
            if not isinstance(files, list):
                return
            ref_name = selected_ref_path.name
            ref_l = ref_name.casefold()
            raw_l = ref_l[5:] if ref_l.startswith("proc_") else ref_l
            proc_l = raw_l if raw_l.startswith("proc_") else f"proc_{raw_l}"
            updated = 0
            for i, entry in enumerate(files):
                if not isinstance(entry, dict):
                    continue
                fp = str(entry.get("file_path") or "")
                fp_l = Path(fp).name.casefold()
                if not (
                    fp_l == raw_l
                    or fp_l == proc_l
                    or fp_l == ref_l
                    or raw_l in fp_l
                    or proc_l in fp_l
                    or ref_l in fp_l
                ):
                    continue
                patched = dict(entry)
                insp = dict(patched.get("inspection") or {})
                insp["wcs"] = 1 if bool(wcs_ok) else 0
                insp["stars"] = int(max(0, int(n_stars)))
                patched["inspection"] = insp
                files[i] = patched
                updated += 1
            if updated:
                patch_draft_manifest(root, int(draft_id), files=files)
            log_event(
                f"MASTERSTAR manifest update: DRAFT_ID={int(draft_id)}, WCS={1 if wcs_ok else 0}, "
                f"Stars={int(n_stars)}, rows={int(updated)}"
            )
        finally:
            db.conn.close()
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR manifest update skipped: {exc!s}")


def _resolve_best_effort_path_under(
    root: Path,
    raw_path: str,
    *,
    pre_calibrated: bool = False,
) -> Path | None:
    """Map an manifest files[].FILE_PATH (often archived raw path) to an existing file under ``root``.

    Strategy: exact relative join, else basename match (first hit). This is intentionally heuristic because
    imports may store different path bases (absolute vs relative, calibrated vs processed).
    """
    _forbid_kw = {"pre_calibrated": bool(pre_calibrated)}
    rp = str(raw_path or "").strip()
    if not rp:
        return None
    p = Path(rp)
    # If already absolute and exists under root tree, accept.
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(root.resolve())
            cand = (root / rel).resolve()
            if (
                cand.is_file()
                and _path_is_under_tree(root, cand)
                and not _path_segments_forbidden_for_masterstar_physical_source(cand, **_forbid_kw)
            ):
                return cand
        except Exception:  # noqa: BLE001
            pass
    # If relative, try directly.
    cand2 = (root / p).resolve()
    if (
        cand2.is_file()
        and _path_is_under_tree(root, cand2)
        and not _path_segments_forbidden_for_masterstar_physical_source(cand2, **_forbid_kw)
    ):
        return cand2
    # Explicit processed-name fallback in the same folder (e.g. Light_066.fits -> proc_Light_066.fits).
    if not pre_calibrated:
        try:
            cand2_proc = cand2.with_name(_safe_proc_name(cand2.name)).resolve()
            if (
                cand2_proc.is_file()
                and _path_is_under_tree(root, cand2_proc)
                and not _path_segments_forbidden_for_masterstar_physical_source(cand2_proc, **_forbid_kw)
            ):
                return cand2_proc
        except Exception:  # noqa: BLE001
            pass
    # Basename / fuzzy suffix fallback (handles prefixes like ``proc_``).
    name = p.name
    if not name:
        return None
    _name_cf = name.casefold()
    _name_noproc = _name_cf[5:] if _name_cf.startswith("proc_") else _name_cf
    hits = []
    for x in _iter_fits_recursive(root):
        xn = x.name.casefold()
        xn_noproc = xn[5:] if xn.startswith("proc_") else xn
        if (
            xn == _name_cf
            or xn_noproc == _name_noproc
            or xn.endswith(_name_cf)
            or xn_noproc.endswith(_name_noproc)
        ):
            if not _path_segments_forbidden_for_masterstar_physical_source(x, **_forbid_kw):
                hits.append(x)
    return _pick_preferred_masterstar_basename_hit(hits, pre_calibrated=pre_calibrated)


def resolve_obs_file_to_processed_fits(
    archive_path: Path | str | None,
    obs_file_path: str,
    *,
    setup_name: str | None = None,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Map ``manifest files[].FILE_PATH`` onto the draft lights FITS for MASTERSTAR / preprocess.

    VYVAR-calibrated: ``processed/lights/.../proc_*.fits`` (legacy alias kept for callers).
    Pre-calibrated: ``non_calibrated/lights/<setup>/`` - the import frame is the frame (no proc_ remap).
    """
    if archive_path is None:
        return None
    from draft_provenance import draft_archive_root, is_pre_calibrated_draft

    ap = draft_archive_root(Path(archive_path).expanduser())
    if not ap.is_dir():
        return None
    pre_cal = is_pre_calibrated_draft(ap, draft_id=draft_id, db=db)
    root = resolve_masterstar_input_root(
        ap,
        setup_name=setup_name,
        app_config=app_config,
        draft_id=draft_id,
        db=db,
    )
    if root is None or not root.exists():
        return None
    hit = _resolve_best_effort_path_under(
        root,
        str(obs_file_path),
        pre_calibrated=pre_cal,
    )
    if hit is None or _path_segments_forbidden_for_masterstar_physical_source(
        hit, pre_calibrated=pre_cal
    ):
        return None
    return hit


def list_best_processed_light_paths_for_masterstar(
    archive_path: Path | str | None,
    *,
    setup_name: str | None = None,
    draft_id: int | None = None,
    app_config: AppConfig | None = None,
    take_n: int = 5,
) -> list[Path]:
    """Najlepsie (najnizsi FWHM) FITS pod ``processed/lights`` - na UI tabulku a vyber pre MASTERSTAR."""
    if archive_path is None:
        return []
    ap = Path(archive_path).expanduser()
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    if not ap.is_dir():
        return []
    root = resolve_masterstar_input_root(
        ap,
        setup_name=setup_name,
        app_config=app_config,
        draft_id=draft_id,
    )
    if root is None or not root.exists():
        return []
    from draft_provenance import is_pre_calibrated_draft

    pre_cal = is_pre_calibrated_draft(ap, draft_id=draft_id)
    _forbid_kw = {"pre_calibrated": pre_cal}
    files = [
        fp
        for fp in _iter_fits_recursive(root)
        if not _path_segments_forbidden_for_masterstar_physical_source(fp, **_forbid_kw)
    ]
    if not files:
        return []
    tn = max(2, min(5, int(take_n)))
    _fb: dict[str, float] = {}
    if draft_id is not None:
        _dbc = _vyvar_open_database(app_config or AppConfig())
        if _dbc is not None:
            try:
                _fb = _obs_fwhm_basename_map_from_db(_dbc, int(draft_id))
            except Exception:  # noqa: BLE001
                # EXC-0305: T4 -- DB conn.close failure after FWHM map fetch is ignored once ranked paths are computed. (EXCEPT-BULK 2026-07-08)
                _fb = {}
            finally:
                try:
                    _dbc.conn.close()
                except Exception:  # noqa: BLE001
                    pass
    ranked = _sort_masterstar_paths_by_fwhm(files, fwhm_by_basename=_fb or None)
    return ranked[:tn]


def get_masterstar_candidate_rows(
    draft_id: int,
    percentage: float,
    *,
    fwhm_max_px: float | None = None,
    db: VyvarDatabase,
) -> "pd.DataFrame":
    """Rank draft light frames by quality metrics for MASTERSTAR selection.

    Strict filter: include only rows with ``IS_REJECTED`` 0/NULL.
    Score (higher is better) is a normalized version of:

    Score = (1 / fwhm) * (1 / sky_level) * snr_estimate
    where ``snr_estimate`` is approximated from ``STAR_COUNT`` and ``SKY_LEVEL`` when no explicit SNR exists.
    """
    import numpy as np
    import pandas as pd

    did = int(draft_id)
    pct = float(percentage)
    pct = float(max(0.1, min(100.0, pct)))

    rows = db.fetch_draft_light_rows_for_quality(did)
    if not rows:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    # Strict: IS_REJECTED == 0 (or NULL)
    if "IS_REJECTED" in df.columns:
        df["IS_REJECTED"] = pd.to_numeric(df["IS_REJECTED"], errors="coerce").fillna(0).astype(int)
        df = df[df["IS_REJECTED"] == 0].copy()
    lim_active = bool(fwhm_max_px is not None and float(fwhm_max_px) > 0)
    lim_v = float(fwhm_max_px) if lim_active else 0.0
    if lim_active:
        _f = pd.to_numeric(df.get("FWHM"), errors="coerce")
        df = df[_f.notna() & (_f <= lim_v)].copy()
    if df.empty:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    df["FWHM"] = pd.to_numeric(df.get("FWHM"), errors="coerce")
    df["SKY_LEVEL"] = pd.to_numeric(df.get("SKY_LEVEL"), errors="coerce")
    df["STAR_COUNT"] = pd.to_numeric(df.get("STAR_COUNT"), errors="coerce").fillna(0).astype(int)

    # Robust fallbacks: median scales for normalization.
    f_med = float(np.nanmedian(df["FWHM"].values)) if np.isfinite(np.nanmedian(df["FWHM"].values)) else 0.0
    s_med = (
        float(np.nanmedian(df["SKY_LEVEL"].values))
        if np.isfinite(np.nanmedian(df["SKY_LEVEL"].values))
        else 0.0
    )
    if not (math.isfinite(f_med) and f_med > 0):
        f_med = 1.0
    if not (math.isfinite(s_med) and s_med > 0):
        s_med = 1.0

    eps = 1e-9
    f = df["FWHM"].astype(float)
    sky = df["SKY_LEVEL"].astype(float)
    stars = df["STAR_COUNT"].astype(float)

    # snr_estimate: more stars and lower sky -> better SNR (proxy; true SNR not stored in manifest files[]).
    snr_est = (stars + 1.0) / np.sqrt(np.maximum(sky, 0.0) + 1.0)

    f_norm_inv = f_med / (np.maximum(f, 0.0) + eps)
    sky_norm_inv = s_med / (np.maximum(sky, 0.0) + eps)

    score = f_norm_inv * sky_norm_inv * snr_est
    score = np.where(np.isfinite(score), score, 0.0)

    df["SNR_EST"] = snr_est
    df["SCORE"] = score

    df = df.sort_values(["SCORE", "STAR_COUNT"], ascending=[False, False], kind="mergesort").reset_index(
        drop=True
    )

    k = int(max(1, math.ceil(len(df) * (pct / 100.0))))
    return df.head(k).loc[:, ["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"]]


def get_masterstar_candidates(draft_id: int, percentage: float, *, db: VyvarDatabase) -> list[str]:
    """Return FILE_PATH list of top-ranked MASTERSTAR candidates (temporary list; manifest files[] is not modified)."""
    df = get_masterstar_candidate_rows(int(draft_id), float(percentage), db=db)
    if df.empty:
        return []
    return [str(x) for x in df["FILE_PATH"].tolist() if str(x).strip()]


def _vyvar_open_database(cfg: AppConfig) -> VyvarDatabase | None:
    try:
        return VyvarDatabase(Path(cfg.database_path))
    except Exception:  # noqa: BLE001
        return None


def _fits_pixel_raw_to_micrometres(value: float) -> float:
    """Map raw FITS pixel-size keywords to **micrometres** (WCS often uses SI metres)."""
    if not math.isfinite(value) or value <= 0:
        return 0.0
    v = float(value)
    if v < 5e-5:
        return v * 1e6
    if v < 0.2:
        return v * 1000.0
    return v


def _header_focal_length_mm(header: fits.Header) -> float | None:
    """Focal length [mm] from common FITS keys (``FOCALLEN`` / ``FOCLEN`` often in **metres**)."""
    for key in ("FOCALLEN", "FOCLEN", "TELFOCA", "FOCAL_LEN", "FOCALL", "FOC_LEN"):
        if key not in header or header[key] in (None, "", " ", "0", 0):
            continue
        try:
            v = float(header[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        mm = v * 1000.0 if v < 25.0 else v
        if 40.0 <= mm <= 120_000.0:
            return float(mm)
    return None


def resolve_plate_solve_fov_deg_hint(
    hdr: fits.Header,
    h: int,
    w: int,
    *,
    database_path: Path | str | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
) -> float | None:
    """Estimate plate-solve FOV diameter [deg] along chip diagonal (optics from header, else DB scale + NAXIS)."""
    if h <= 0 or w <= 0:
        return None
    f_mm = _header_focal_length_mm(hdr)
    p_um = _db_header_pixel_native_um_mean(hdr)
    if f_mm is not None and p_um is not None and f_mm > 0 and p_um > 0:
        diag_mm = math.hypot(float(w) * float(p_um) * 0.001, float(h) * float(p_um) * 0.001)
        rad = 2.0 * math.atan2(0.5 * diag_mm, float(f_mm))
        if math.isfinite(rad) and rad > 0:
            return float(rad * 180.0 / math.pi)

    dbp = str(database_path or "").strip()
    if not dbp:
        return None
    try:
        db = VyvarDatabase(Path(dbp))
    except Exception:  # noqa: BLE001
        return None
    try:
        eq = int(equipment_id) if equipment_id is not None else None
        tel: int | None = None
        if draft_id is not None:
            dr = db.fetch_obs_draft_by_id(int(draft_id))
            if dr is not None:
                if eq is None and dr.get("ID_EQUIPMENTS") is not None:
                    eq = int(dr["ID_EQUIPMENTS"])
                if dr.get("ID_TELESCOPE") is not None:
                    tel = int(dr["ID_TELESCOPE"])
        xb, yb = fits_binning_xy_from_header(hdr)
        bin_b = max(1, int(xb), int(yb))
        sc = compute_plate_scale_from_db(eq, tel, db.conn, binning=bin_b)
        if sc is None or not math.isfinite(float(sc)) or float(sc) <= 0:
            return None
        nx = int(hdr.get("NAXIS1", w) or w)
        ny = int(hdr.get("NAXIS2", h) or h)
        return plate_solve_fov_deg_diagonal_from_scale(nx, ny, float(sc))
    except Exception:  # noqa: BLE001
        # EXC-0309: T4 -- db.conn.close failure in FOV hint finally block is ignored after the hint is computed o... (EXCEPT-BULK 2026-07-08)
        return None
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass


def get_auto_fov(
    *,
    archive_path: Path | None = None,
    masterstar_path: Path | None = None,
    database_path: Path | str | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
) -> float | None:
    """Auto field diameter [deg] (diagonal) for plate solving.

    Priority:
    - Header optics (focal + pixel) or DB plate-scale x NAXIS diagonal
    - Else WCS corners (after a successful solve)
    """
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    ms = Path(masterstar_path) if masterstar_path is not None else None
    if ms is None and archive_path is not None:
        ap = Path(archive_path)
        cand = ap / "platesolve" / "MASTERSTAR.fits"
        ms = cand if cand.is_file() else None
    if ms is None or not ms.is_file():
        return None

    with fits.open(ms, memmap=False) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
    if data is None:
        return None
    h, w = int(data.shape[0]), int(data.shape[1])

    _dbp = str(database_path or "").strip()
    if not _dbp:
        try:
            _dbp = str(AppConfig().database_path)
        except Exception:  # noqa: BLE001
            _dbp = ""
    _hint = resolve_plate_solve_fov_deg_hint(
        hdr, h, w, database_path=_dbp or None, equipment_id=equipment_id, draft_id=draft_id
    )
    if _hint is not None and math.isfinite(_hint) and _hint > 0:
        return float(_hint)

    # Fall back to WCS-based FOV (after solve).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs0 = WCS(hdr)
        if not getattr(wcs0, "has_celestial", False):
            return None
        corners = np.array(
            [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
            dtype=np.float64,
        )
        sky = wcs0.celestial.pixel_to_world(corners[:, 0], corners[:, 1])
        c0 = SkyCoord(sky[0])
        c2 = SkyCoord(sky[2])
        sep = c0.separation(c2).to(u.deg).value
        if math.isfinite(float(sep)) and float(sep) > 0:
            return float(sep)
    except Exception:  # noqa: BLE001
        return None
    return None


def _focal_mm_plausible(mm: float) -> bool:
    return math.isfinite(mm) and 40.0 <= mm <= 120_000.0


def _resolve_focal_mm_for_plate_scale(
    header: fits.Header | None,
    db: VyvarDatabase | None,
    *,
    telescope_id: int | None = None,
) -> tuple[float | None, str]:
    """Plausible FITS focal first; else ``TELESCOPE.FOCAL``."""
    if header is not None:
        hdr_mm = _header_focal_length_mm(header)
        if hdr_mm is not None:
            hdr_n, hfixed = normalize_telescope_focal_mm_for_plate_scale(hdr_mm)
            if hfixed:
                log_event(
                    f"FOCAL: v hlavicke FITS ohnisko vyzeralo ako 10x preklep ({hdr_mm:g} mm -> {hdr_n:g} mm)."
                )
            hdr_mm = hdr_n
        if hdr_mm is not None and _focal_mm_plausible(hdr_mm):
            return float(hdr_mm), "fits_header"
    if db is not None and telescope_id is not None:
        try:
            tel_raw = db.get_telescope_focal_mm(int(telescope_id))
        except Exception:  # noqa: BLE001
            tel_raw = None
        if tel_raw is not None:
            tel_n, tel_fixed = normalize_telescope_focal_mm_for_plate_scale(float(tel_raw))
            if tel_fixed:
                log_event(
                    f"FOCAL: TELESCOPE.FOCAL (ID={int(telescope_id)}) vyzeralo ako 10x preklep "
                    f"({tel_raw:g} mm -> {tel_n:g} mm) - pouzite pre mierku / solver."
                )
            if _focal_mm_plausible(tel_n):
                return float(tel_n), "database_telescope"
    if db is not None:
        raw = db.get_telescope_focal_mm(None)
        if raw is not None:
            norm, fixed = normalize_telescope_focal_mm_for_plate_scale(float(raw))
            if fixed:
                log_event(
                    f"FOCAL: TELESCOPE.FOCAL v DB vyzeralo ako 10x preklep ({raw:g} mm -> {norm:g} mm) - "
                    "pouzite pre mierku / solver."
                )
            if _focal_mm_plausible(norm):
                return float(norm), "database_telescope"
    return None, "none"


def _merge_equipment_pixel_into_metadata(meta: dict[str, Any], db: VyvarDatabase, equipment_id: int) -> None:
    """If FITS native pixel is missing or nonsense, use ``EQUIPMENTS.PIXELSIZE`` [um, 1x1] x binning."""
    try:
        native = db.get_equipment_pixel_size_um(int(equipment_id))
    except Exception:  # noqa: BLE001
        return
    if native is None:
        return
    try:
        nv = float(native)
    except (TypeError, ValueError):
        return
    if not math.isfinite(nv) or nv <= 0 or nv > 300.0:
        return
    x_bin = max(1, int(meta.get("binning", 1) or 1))
    y_bin = max(1, int(meta.get("binning_y", x_bin) or x_bin))
    prev = meta.get("pixel_size_um_physical")
    # Universal scale: trust EQUIPMENTS.PIXELSIZE (UI/DB) whenever equipment is known; binning stays from FITS.
    meta["pixel_size_um_physical"] = float(nv)
    eff_x = float(nv) * float(x_bin)
    eff_y = float(nv) * float(y_bin)
    meta["pixel_size_um_header"] = (eff_x + eff_y) / 2.0
    meta["effective_pixel_um_plate_scale"] = float(nv) * float(x_bin)
    meta["pixel_size_um_source"] = "equipment_db"
    _eff = float(nv) * float(x_bin)
    if prev is not None:
        try:
            pv = float(prev)
            same = math.isfinite(pv) and abs(pv - float(nv)) < 1e-6
        except (TypeError, ValueError):
            same = False
        if not same:
            log_event(
                f"PIXEL: EQUIPMENTS.PIXELSIZE={nv:g} um x binning {x_bin}x{y_bin} -> efektivny {_eff:g} um "
                f"(preferovane pred FITS; predtym {prev!r})."
            )
    else:
        log_event(
            f"PIXEL: EQUIPMENTS.PIXELSIZE={nv:g} um x binning {x_bin}x{y_bin} -> efektivny {_eff:g} um."
        )


def _recompute_effective_pixel_from_physical(meta: dict[str, Any]) -> None:
    """``effective_pixel_um_plate_scale`` = native [um] x XBINNING (int)."""
    phys = meta.get("pixel_size_um_physical")
    if phys is None:
        return
    try:
        p = float(phys)
    except (TypeError, ValueError):
        return
    if not math.isfinite(p) or p <= 0:
        return
    x_bin = max(1, int(meta.get("binning", 1) or 1))
    y_bin = max(1, int(meta.get("binning_y", x_bin) or x_bin))
    eff_x = p * float(x_bin)
    eff_y = p * float(y_bin)
    meta["pixel_size_um_header"] = (eff_x + eff_y) / 2.0
    meta["effective_pixel_um_plate_scale"] = float(p) * float(x_bin)


def _header_pick_first(header: fits.Header, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return default


def _enrich_calibration_metadata_from_header(
    meta: dict[str, Any],
    header: fits.Header,
    *,
    db: VyvarDatabase | None,
    id_equipment: int | None,
    id_telescope: int | None = None,
) -> None:
    """Add ``focal_length``, ``pixel_size_raw``, ``pixel_um``, ``focal_length_source`` for diagnostics / UI."""

    def _to_f(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    r1 = _to_f(_header_pick_first(header, "PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0))
    r2 = _to_f(_header_pick_first(header, "PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0))
    parts: list[str] = []
    if r1 > 0:
        parts.append(f"PIX1={r1:g}")
    if r2 > 0:
        parts.append(f"PIX2={r2:g}")
    meta["pixel_size_raw"] = " | ".join(parts) if parts else "n/a"

    meta["pixel_um"] = meta.get("effective_pixel_um_plate_scale")

    focal_mm: float | None = None
    src = "none"
    # Universal optics: prefer DB (UI) over FITS FOCALLEN for plate scale / solver hints.
    if db is not None:
        telf = db.get_telescope_focal_mm(int(id_telescope) if id_telescope is not None else None)
        if telf is not None:
            focal_mm, _fx = normalize_telescope_focal_mm_for_plate_scale(float(telf))
            src = "telescope_focal"

    has_focallen = "FOCALLEN" in header and header["FOCALLEN"] not in (None, "", " ", "0", 0)
    if focal_mm is None and has_focallen:
        raw_fc = header["FOCALLEN"]
        log_event(f"DIAG: FITS FOCALLEN (SIPS / zapisovatel hlavicky) = {raw_fc!r}")
        try:
            v = float(raw_fc)
            mm0 = (v * 1000.0) if (math.isfinite(v) and v > 0 and v < 25.0) else v
        except (TypeError, ValueError):
            mm0 = None
        if mm0 is not None and math.isfinite(mm0) and mm0 > 0 and _focal_mm_plausible(float(mm0)):
            focal_mm, _fx = normalize_telescope_focal_mm_for_plate_scale(float(mm0))
            src = "fits_focallen"

    meta["focal_length"] = focal_mm
    meta["focal_length_source"] = src


def _apply_draft_combined_to_pipeline_meta(meta: dict[str, Any], comb: dict[str, Any]) -> None:
    """Overlay ``VyvarDatabase.get_combined_metadata`` onto ``extract_fits_metadata`` output."""
    fl = comb.get("focal_length_mm")
    if fl is not None:
        try:
            fv = float(fl)
            if math.isfinite(fv) and _focal_mm_plausible(fv):
                meta["focal_length"] = fv
                meta["focal_length_source"] = str(comb.get("focal_source") or "draft_combined")
        except (TypeError, ValueError):
            pass
    pn = comb.get("pixel_native_um")
    if pn is not None:
        try:
            pv = float(pn)
            if math.isfinite(pv) and 0 < pv <= 300.0:
                meta["pixel_size_um_physical"] = pv
        except (TypeError, ValueError):
            pass
    xb = max(1, int(comb.get("xbinning", 1) or 1))
    yb = max(1, int(comb.get("ybinning", xb) or xb))
    meta["binning"] = xb
    meta["binning_y"] = yb
    pe = comb.get("pixel_effective_um")
    if pe is not None:
        try:
            ev = float(pe)
            if math.isfinite(ev) and ev > 0:
                meta["effective_pixel_um_plate_scale"] = ev
                if meta.get("pixel_size_um_physical") is not None:
                    pph = float(meta["pixel_size_um_physical"])
                    meta["pixel_size_um_header"] = (pph * float(xb) + pph * float(yb)) / 2.0
                else:
                    meta["pixel_size_um_header"] = ev
                meta["pixel_um"] = ev
        except (TypeError, ValueError):
            pass
    sat = comb.get("saturate_adu")
    if sat is not None:
        try:
            sv = float(sat)
            if math.isfinite(sv) and sv > 0:
                meta["equipment_saturate_adu"] = sv
        except (TypeError, ValueError):
            pass


def _log_calibration_metadata_diagnostic(filename: str, metadata: dict[str, Any]) -> None:
    log_event("--- DIAGNOSTIKA METADAT PRE KALIBRACIU ---")
    log_event(f"Subor: {filename}")
    log_event(f"FOCAL (z DB/FITS): {metadata.get('focal_length')} mm")
    log_event(f"PIXEL_SIZE (surovy): {metadata.get('pixel_size_raw')} um")
    _n1 = metadata.get("naxis1")
    _n2 = metadata.get("naxis2")
    if _n1 or _n2:
        log_event(f"NAXIS: {_n1}x{_n2}")
    _rbx = metadata.get("fits_xbinning_raw")
    _rby = metadata.get("fits_ybinning_raw")
    if _rbx is not None or _rby is not None:
        log_event(f"BINNING (FITS raw): XBINNING={_rbx!r} YBINNING={_rby!r}")
    log_event(f"BINNING (X/Y): {metadata.get('binning')}x{metadata.get('binning_y')}")
    log_event(f"EFEKTIVNY PIXEL (pre vypocet): {metadata.get('pixel_um')} um")
    log_event("------------------------------------------")


def _plate_solve_input_bundle(
    fits_path: Path,
    *,
    app_config: AppConfig | None,
    equipment_id: int | None,
    draft_id: int | None = None,
    telescope_id: int | None = None,
) -> dict[str, Any]:
    """Open DB once: metadata, effective pixel, focal length, expected plate scale [arcsec/pixel]."""
    cfg_u = app_config or AppConfig()
    db_u = _vyvar_open_database(cfg_u)
    _eq_use, _tel_use = resolve_optics_ids_for_platesolve(
        db_u, draft_id, equipment_id=equipment_id, telescope_id=telescope_id
    )
    out: dict[str, Any] = {
        "meta": {},
        "header": None,
        "eff_um": None,
        "focal_mm": None,
        "expected_arcsec_per_px": None,
    }
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            out["header"] = hdul[0].header.copy()
        out["meta"] = extract_fits_metadata(
            fits_path,
            db=db_u,
            app_config=cfg_u,
            id_equipment=_eq_use,
            draft_id=draft_id,
        )
        if draft_id is not None:
            _m = out["meta"]
            if _m.get("focal_length") is None or _m.get("effective_pixel_um_plate_scale") is None:
                raise DraftTechnicalMetadataError(int(draft_id))
        foc: float | None = None
        _mf = out["meta"].get("focal_length")
        if _mf is not None:
            try:
                _fx = float(_mf)
                if math.isfinite(_fx) and _focal_mm_plausible(_fx):
                    foc = _fx
            except (TypeError, ValueError):
                foc = None
        if foc is None:
            foc, _ = _resolve_focal_mm_for_plate_scale(
                out["header"], db_u, telescope_id=_tel_use
            )
        out["focal_mm"] = foc
        eff_um: float | None = None
        x_bin = max(1, int(out["meta"].get("binning", 1) or 1))
        if db_u is not None and _eq_use is not None:
            try:
                _nat = db_u.get_equipment_pixel_size_um(int(_eq_use))
            except Exception:  # noqa: BLE001
                _nat = None
            if _nat is not None:
                try:
                    _nv = float(_nat)
                    if math.isfinite(_nv) and 0.5 < _nv <= 300.0:
                        eff_um = float(_nv) * float(x_bin)
                except (TypeError, ValueError):
                    pass
        if eff_um is None:
            _ev = out["meta"].get("effective_pixel_um_plate_scale")
            if _ev is not None:
                try:
                    _ef = float(_ev)
                    if math.isfinite(_ef) and _ef > 0:
                        eff_um = _ef
                except (TypeError, ValueError):
                    eff_um = None
        out["eff_um"] = eff_um
        if eff_um is not None and foc is not None:
            out["expected_arcsec_per_px"] = plate_scale_arcsec_per_pixel(
                pixel_pitch_um=float(eff_um),
                focal_length_mm=float(foc),
            )
            calculated_scale = out["expected_arcsec_per_px"]
            if calculated_scale is not None:
                log_event(f"MATH CHECK: ({eff_um} / {foc}) * 206.265 = {calculated_scale}")
    except Exception as exc:  # noqa: BLE001
        # EXC-0313: T4 -- db_u.conn.close failure in plate-solve bundle finally is ignored after bundle assembly. (EXCEPT-BULK 2026-07-08)
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().plate_solve_bundle_fail += 1
        LOGGER.error("[PLATE-SOLVE] _plate_solve_input_bundle failed: %s", exc)
        out["bundle_error"] = str(exc)
    finally:
        if db_u is not None:
            try:
                db_u.conn.close()
            except Exception:  # noqa: BLE001
                pass
    return out


def compute_plate_scale_from_db(
    equipment_id: int | None,
    telescope_id: int | None,
    db_conn: Any,
    *,
    binning: int = 1,
) -> float | None:
    """Vypocita plate scale [arcsec/px] z EQUIPMENTS a TELESCOPE tabuliek.

    Formula: plate_scale = (pixel_um * binning) / focal_mm * 206.265
    """
    try:
        pixel_um = None
        focal_mm = None
        bx = max(1, int(binning))
        if equipment_id is not None:
            row = db_conn.execute(
                "SELECT PIXELSIZE FROM EQUIPMENTS WHERE ID = ?",
                (int(equipment_id),),
            ).fetchone()
            if row is not None and row[0] is not None:
                pixel_um = float(row[0])

        if telescope_id is not None:
            row = db_conn.execute(
                "SELECT FOCAL FROM TELESCOPE WHERE ID = ?",
                (int(telescope_id),),
            ).fetchone()
            if row is not None and row[0] is not None and float(row[0]) > 0:
                focal_mm = float(row[0])

        if pixel_um and focal_mm:
            scale = (pixel_um * bx) / focal_mm * 206.265
            return round(scale, 4)
    except Exception:  # noqa: BLE001
        pass
    return None


def _try_rescale_masterstar_linear_wcs_to_expected_plate_scale(
    fits_path: Path,
    *,
    app_config: AppConfig | None,
    equipment_id: int | None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """If DB/optics yield expected arcsec/pixel and the on-disk WCS is linear (no SIP), rescale CD when mismatch is large."""
    out: dict[str, Any] = {"rescaled": False}
    try:
        b = _plate_solve_input_bundle(
            fits_path,
            app_config=app_config or AppConfig(),
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
        exp = b.get("expected_arcsec_per_px")
        if exp is None:
            return out
        exp_f = float(exp)
        if not math.isfinite(exp_f) or exp_f <= 0:
            return out
        fp = Path(fits_path)
        with fits.open(fp, memmap=False) as hdul:
            hdr0 = hdul[0].header.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w0 = WCS(hdr0)
        if not w0.has_celestial or w0.sip is not None:
            return out
        try:
            _psm0 = np.asarray(w0.pixel_scale_matrix, dtype=np.float64)
            _old_scale = float(np.sqrt(np.abs(np.linalg.det(_psm0))) * 3600.0)
        except Exception:  # noqa: BLE001
            _old_scale = float("nan")
        w2, changed = maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel(w0, exp_f)
        if not changed:
            return out
        _target_scale = exp_f
        try:
            _psm2 = np.asarray(w2.pixel_scale_matrix, dtype=np.float64)
            _new_scale = float(np.sqrt(np.abs(np.linalg.det(_psm2))) * 3600.0)
        except Exception:  # noqa: BLE001
            _new_scale = float("nan")
        try:
            wh = w2.to_header(relax=True)
        except Exception:  # noqa: BLE001
            return out
        with fits.open(fp, mode="update", memmap=False) as hdul:
            h = hdul[0].header
            strip_celestial_wcs_keys(h)
            for k in wh:
                if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                    continue
                if k.startswith("NAXIS") and k != "NAXIS":
                    continue
                try:
                    h[k] = wh[k]
                except Exception:  # noqa: BLE001
                    pass
            h.add_history("VYVAR: CD scaled to expected arcsec/pixel from equipment optics")
            hdul.flush()
        if math.isfinite(_old_scale) and math.isfinite(_new_scale):
            log_event(
                f"WCS rescale: {_old_scale:.4f} -> {_new_scale:.4f} arcsec/px "
                f"(target {exp_f:.3f})"
            )
        else:
            log_event(
                f"WCS PLATE SCALE: linearny TAN prisposobeny optickej mierke {exp_f:.3f} arcsec/px."
            )
        _ms_csv = fp.parent / "masterstars_full_match.csv"
        if _ms_csv.is_file():
            try:
                _ms_df = read_vyvar_csv(_ms_csv)
                if _ms_df is not None and not _ms_df.empty and "x" in _ms_df.columns and "y" in _ms_df.columns:
                    _mx = pd.to_numeric(_ms_df["x"], errors="coerce").to_numpy(dtype=np.float64)
                    _my = pd.to_numeric(_ms_df["y"], errors="coerce").to_numpy(dtype=np.float64)
                    _mra, _mdec = _all_pix2world_icrs_deg(w2, _mx, _my)
                    _ms_df["ra_deg"] = _mra
                    _ms_df["dec_deg"] = _mdec
                    _vyvar_df_to_csv(_ms_df, _ms_csv)
                    log_event(
                        "masterstars_full_match.csv ra_deg/dec_deg recomputed after WCS rescale"
                    )
            except Exception as _csv_exc:  # noqa: BLE001
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().masterstars_rescale_coords_fail += 1
                LOGGER.error(
                    "masterstars_full_match.csv ra/dec recompute after WCS rescale failed: %s",
                    _csv_exc,
                )
        out["rescaled"] = True
        out["expected_arcsec_per_px"] = exp_f
        if math.isfinite(_old_scale):
            out["old_scale_arcsec_per_px"] = _old_scale
        if math.isfinite(_new_scale):
            out["new_scale_arcsec_per_px"] = _new_scale
    except Exception as exc:  # noqa: BLE001
        log_event(f"WCS PLATE SCALE: uprava CD preskocena - {exc!s}")
        out["error"] = str(exc)
    return out


def _solve_wcs_solve_field_cli(
    masterstar_path: Path,
    *,
    expected_arcsec_per_pixel: float | None = None,
) -> dict[str, Any]:
    """Run local ``solve-field`` if available (ANSVR / astrometry.net indexes on PATH or ``VYVAR_SOLVE_FIELD_EXE``).

    Uses ``--tweak-order`` (SIP-style distortion) and ``--cpulimit`` from :data:`ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC`.
    Skip with env ``VYVAR_SKIP_SOLVE_FIELD=1``.
    """

    if os.environ.get("VYVAR_SKIP_SOLVE_FIELD", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {"solved": False, "reason": "VYVAR_SKIP_SOLVE_FIELD set"}
    exe = (os.environ.get("VYVAR_SOLVE_FIELD_EXE") or "").strip()
    if not exe:
        exe = shutil.which("solve-field") or ""
    if not exe:
        return {"solved": False, "reason": "solve-field not on PATH (set VYVAR_SOLVE_FIELD_EXE for ANSVR)"}

    mp = Path(masterstar_path).resolve()
    if not mp.is_file():
        return {"solved": False, "reason": f"File not found: {mp}"}

    wcs_path = mp.parent / f"{mp.stem}.wcs"
    wcs_path.unlink(missing_ok=True)

    cmd: list[str] = [
        exe,
        "--cpulimit",
        str(int(ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC)),
        "--tweak-order",
        str(int(effective_astrometry_net_tweak_order())),
        "--no-plots",
        "--overwrite",
    ]
    if expected_arcsec_per_pixel is not None:
        s = float(expected_arcsec_per_pixel)
        if math.isfinite(s) and 0.03 < s < 200.0:
            lo, hi = astrometry_net_scale_bounds_arcsec_per_pix(s)
            cmd.extend(
                [
                    "--scale-low",
                    f"{float(lo):.6g}",
                    "--scale-high",
                    f"{float(hi):.6g}",
                ]
            )
    cmd.append(str(mp))

    log_event(
        f"solve-field (lokalny): {exe} - --cpulimit {ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC}, "
        f"--tweak-order {effective_astrometry_net_tweak_order()}, {mp.name}"
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(mp.parent),
            capture_output=True,
            text=True,
            timeout=900,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"solved": False, "reason": "solve-field subprocess timeout (900s wall)"}
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0319] Generated .wcs read failure returns solved=False without applying any WCS to the image.: %s', exc)
        return {"solved": False, "reason": f"solve-field: {exc!s}"}

    if not wcs_path.is_file():
        tail = (proc.stderr or proc.stdout or "")[-900:]
        if proc.returncode != 0:
            return {"solved": False, "reason": f"solve-field exit {proc.returncode}: {tail!s}"}
        return {"solved": False, "reason": f"solve-field: missing {wcs_path.name} - {tail!s}"}

    try:
        with fits.open(wcs_path, memmap=False) as wh:
            wcs_hdr = wh[0].header.copy()
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"read {wcs_path.name}: {exc!s}"}

    _apply_wcs_header_to_fits(mp, wcs_hdr)
    with fits.open(mp, mode="update", memmap=False) as hdul:
        hdr = hdul[0].header
        hdr["VY_PSOLV"] = (True, "Plate solved by local solve-field (SIP tweak-order)")
        hdr.add_history(
            f"VYVAR: solve-field --cpulimit {ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC} "
            f"--tweak-order {effective_astrometry_net_tweak_order()}"
        )
        hdul.flush()

    log_event(
        f"solve-field OK: WCS so SIP (tweak-order {effective_astrometry_net_tweak_order()}) -> {mp.name}"
    )
    return {"solved": True, "method": "solve-field (local CLI)"}



def _solve_wcs_astrometry_net(
    masterstar_path: Path,
    api_key: str | None = None,
    *,
    expected_arcsec_per_pixel: float | None = None,
) -> dict[str, Any]:
    """Try to solve WCS via astrometry.net (optional). Requires astroquery + API key."""
    import os

    api_key = (api_key or os.environ.get("ASTROMETRY_NET_API_KEY", "")).strip()
    if not api_key:
        return {"solved": False, "reason": "Missing ASTROMETRY_NET_API_KEY"}

    try:
        from astroquery.astrometry_net import AstrometryNet  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"astroquery astrometry_net unavailable: {exc}"}

    ast = AstrometryNet()
    ast.api_key = api_key
    _tw = int(effective_astrometry_net_tweak_order())
    solve_kw: dict[str, Any] = {
        "solve_timeout": 180,
        "verbose": False,
        "tweak_order": max(0, _tw),
    }
    log_event(f"Astrometry.net API: tweak_order={_tw} (SIP / distortion, ~solve-field --tweak-order).")
    if expected_arcsec_per_pixel is not None:
        s = float(expected_arcsec_per_pixel)
        if math.isfinite(s) and 0.03 < s < 200.0:
            lo, hi = astrometry_net_scale_bounds_arcsec_per_pix(s)
            solve_kw.update(
                scale_type="ul",
                scale_units="arcsecperpix",
                scale_lower=float(lo),
                scale_upper=float(hi),
            )
            log_event(
                f"Astrometry.net: obmedzenie mierky ~{s:.3f} arcsec/px "
                f"(scale_low={lo:.3f}, scale_high={hi:.3f}; ~solve-field --scale-low/--scale-high)."
            )
    try:
        wcs_header = ast.solve_from_image(str(masterstar_path), **solve_kw)
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"Astrometry.net solve failed: {exc}"}

    if not wcs_header:
        return {"solved": False, "reason": "Astrometry.net returned no WCS header"}

    _apply_wcs_header_to_fits(masterstar_path, fits.Header(wcs_header))
    with fits.open(masterstar_path, mode="update", memmap=False) as hdul:
        hdr = hdul[0].header
        hdr["VY_PSOLV"] = (True, "Plate solved by Astrometry.net")
        hdr.add_history(
            f"VYVAR: Astrometry.net API tweak_order={int(effective_astrometry_net_tweak_order())} (SIP / distortion)"
        )
        hdul.flush()

    return {"solved": True, "method": "astrometry.net"}


def _apply_wcs_header_to_fits(fits_path: Path, wcs_hdr: fits.Header) -> None:
    """Merge celestial WCS keywords from ``wcs_hdr`` into the image FITS (primary HDU)."""
    with fits.open(fits_path, mode="update", memmap=False) as hdul:
        h = hdul[0].header
        strip_celestial_wcs_keys(h)
        for k in wcs_hdr:
            if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                continue
            if k.startswith("NAXIS") and k != "NAXIS":
                continue
            try:
                h[k] = wcs_hdr[k]
            except Exception:  # noqa: BLE001
                pass
        hdul.flush()


def _solve_wcs_external(
    fits_path: Path,
    *,
    backend: str = "vyvar",
    astrometry_api_key: str | None = None,
    plate_solve_fov_deg: float = 1.0,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
    app_config: AppConfig | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Plate solve: vyhradne VYVAR (lokalna Gaia DB). Backend ``auto`` / Astrometry.net aliasy sa mapuju na VYVAR."""
    be = (backend or "vyvar").strip().lower()
    if be in {"astap", "astap_cli", "local"}:
        LOGGER.warning("platesolve backend %r is no longer supported; using VYVAR.", be)
        be = "vyvar"

    fp_solve = Path(fits_path)

    def _finalize_plate_solve_result(r: dict[str, Any]) -> dict[str, Any]:
        # Post-refine removed; Gaia-based pipeline keeps the WCS as-solved.
        try:
            sm = r.get("sip_meta") if isinstance(r, dict) else None
            if isinstance(sm, dict) and sm.get("initial_wcs_offset_px") is not None:
                _off = float(sm.get("initial_wcs_offset_px"))
                if math.isfinite(_off) and _off > 0:
                    log_event(
                        f"DEBUG: Initial WCS offset detected: {_off:.2f} pixels. Applying coarse correction (Pass 0)..."
                    )
        except Exception:  # noqa: BLE001
            pass
        if isinstance(r, dict) and not bool(r.get("solved", False)):
            try:
                mr = r.get("match_rate")
                mrf = float(mr) if mr is not None else float("nan")
                if math.isfinite(mrf) and mrf < 0.02:
                    log_event(
                        f"WARNING: Plate solve final match rate too low ({mrf * 100.0:.1f}%). "
                        "Returning solved=False to prevent downstream matching on invalid WCS."
                    )
            except Exception:  # noqa: BLE001
                pass
        return r

    def _try_vyvar(*, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg_u = app_config or AppConfig()
        gaia_db = (cfg_u.gaia_db_path or "").strip()
        if not gaia_db:
            return {
                "solved": False,
                "reason": "VYVAR solver: v Settings nastav cestu k lokalnej Gaia DR3 SQLite DB (gaia_db_path).",
            }
        from vyvar_platesolver import solve_wcs_with_local_gaia, _get_masterstar_wcs_parity

        b = bundle or _plate_solve_input_bundle(
            fits_path,
            app_config=cfg_u,
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
        _is_masterstar = fp_solve.name.strip().upper() == "MASTERSTAR.FITS"
        hint_ra = hint_ra_deg
        hint_dec = hint_dec_deg
        _em = b.get("meta") or {}
        eff_um = b.get("eff_um")
        exp_scale = b.get("expected_arcsec_per_px")

        # Per-frame solve MUST NOT invoke blind solver.
        # VYVAR platesolver takes RA/Dec from FITS header (VYTARG*/RA/DEC/WCS); caller hint args are not used.
        # Therefore, for non-MASTERSTAR frames we inject pointing hint from MASTERSTAR WCS (CRVAL1/2).
        # Mirror orientation hint:
        # - MASTERSTAR: can be hinted from MASTERSTAR header (VY_MIRR) to speed/robustify the sweep.
        # - per-frame: do NOT inherit from MASTERSTAR; frames may have different orientation.
        preferred_mirror: str | None = None
        if (not _is_masterstar) and (hint_ra is None or hint_dec is None):
            try:
                # Nacitaj center z MASTERSTAR.fits (ma platny WCS po plate solve)
                masterstar_path: Path | None = None
                _fp_here = Path(fits_path)
                # Guess setup name from parent folder (e.g. processed/lights/R_60_1/*.fits or detrended_aligned/lights/V_60_1/*.fits)
                _setup_guess = ""
                try:
                    _setup_guess = str(_fp_here.parent.name or "").strip()
                except Exception:  # noqa: BLE001
                    _setup_guess = ""
                _candidate = _fp_here
                for _lvl in range(6):  # max 6 urovni hore
                    _candidate = _candidate.parent
                    # New (multi-filter): prefer per-setup MASTERSTAR in platesolve/<setup>/MASTERSTAR.fits
                    _ms_path_setup = (
                        (_candidate / "platesolve" / _setup_guess / "MASTERSTAR.fits")
                        if _setup_guess
                        else None
                    )
                    # Back-compat: old location platesolve/MASTERSTAR.fits (single-setup drafts)
                    _ms_path_root = _candidate / "platesolve" / "MASTERSTAR.fits"
                    _ms_path = _ms_path_setup if _ms_path_setup is not None else _ms_path_root
                    try:
                        if bool(cfg_u.debug_platesolver):
                            log_event(
                                "DEBUG: MASTERSTAR search "
                                f"lvl={_lvl} setup={_setup_guess!r} "
                                f"setup_path={_ms_path_setup} exists_setup={_ms_path_setup.is_file() if _ms_path_setup is not None else False} "
                                f"root_path={_ms_path_root} exists_root={_ms_path_root.is_file()}"
                            )
                    except Exception:  # noqa: BLE001
                        pass
                    # Prefer setup MASTERSTAR if it exists; in multi-group drafts never use root MASTERSTAR.
                    _multi_obs = draft_is_multi_group_obs(_candidate)
                    if _ms_path_setup is not None and _ms_path_setup.is_file():
                        masterstar_path = _ms_path_setup
                        break
                    if _multi_obs and _setup_guess:
                        continue
                    if _ms_path_root.is_file():
                        masterstar_path = _ms_path_root
                        break
                if masterstar_path is not None and masterstar_path.is_file():
                    from astropy.io import fits as _fits

                    with _fits.open(masterstar_path, memmap=False) as _hdul:
                        _mhdr = _hdul[0].header
                        _ms_ra = _mhdr.get("CRVAL1")
                        _ms_dec = _mhdr.get("CRVAL2")
                        if _ms_ra is not None and _ms_dec is not None:
                            hint_ra = float(_ms_ra)
                            hint_dec = float(_ms_dec)
                            log_event(
                                f"INFO: Per-frame hint z MASTERSTAR CRVAL: RA={hint_ra:.4f} Dec={hint_dec:.4f}"
                            )
                    # Do not set preferred_mirror for per-frame solves here.
            except Exception as _e:  # noqa: BLE001
                log_event(f"WARNING: Per-frame hint z MASTERSTAR zlyhal: {_e}")

        # Fallback: if MASTERSTAR hint is missing, try draft manifest center (can be 0/0).
        if (not _is_masterstar) and (hint_ra is None or hint_dec is None) and draft_id is not None:
            try:
                _db_hint = _vyvar_open_database(cfg_u)
                if _db_hint is not None:
                    try:
                        drow = _db_hint.fetch_obs_draft_by_id(int(draft_id)) or {}
                        ra_db = drow.get("CENTEROFFIELDRA")
                        de_db = drow.get("CENTEROFFIELDDE")
                        if ra_db is not None and de_db is not None:
                            ra_f = float(ra_db)
                            de_f = float(de_db)
                            if math.isfinite(ra_f) and math.isfinite(de_f) and not (
                                abs(ra_f) < 1e-9 and abs(de_f) < 1e-9
                            ):
                                hint_ra = float(ra_f)
                                hint_dec = float(de_f)
                                log_event(
                                    f"INFO: Per-frame hint z draft manifest center: RA={hint_ra:.4f} Dec={hint_dec:.4f}"
                                )
                    finally:
                        try:
                            _db_hint.conn.close()
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                pass

        _db_ps = _vyvar_open_database(cfg_u)
        _auto_ps: float | None = None
        if _db_ps is not None:
            try:
                _eq_ps, _tel_ps = resolve_optics_ids_for_platesolve(
                    _db_ps, draft_id, equipment_id=equipment_id
                )
                _bx = max(1, int(_em.get("binning", 1) or 1))
                _auto_ps = compute_plate_scale_from_db(
                    int(_eq_ps) if _eq_ps is not None else None,
                    _tel_ps,
                    _db_ps.conn,
                    binning=_bx,
                )
            except Exception:  # noqa: BLE001
                _auto_ps = None
            finally:
                try:
                    _db_ps.conn.close()
                except Exception:  # noqa: BLE001
                    pass
        exp_scale = _auto_ps or exp_scale or None
        if _auto_ps is not None:
            log_event(
                f"INFO: Plate scale z DB (Equipment+Telescope): {_auto_ps:.4f} arcsec/px"
            )
        elif _is_masterstar:
            log_event(
                "WARNING: Plate scale z DB nedostupna - solver odvodi mierku z FITS alebo None"
            )
        _bx = max(1, int(_em.get("binning", 1) or 1))
        _pix = _em.get("pixel_size_um_physical")
        _foc_mm = b.get("focal_mm")
        _pix_s = f"{float(_pix):.4g}" if _pix is not None else "n/a"
        _foc_s = f"{float(_foc_mm):.4g}" if _foc_mm is not None else "n/a"
        _eff_s = f"{float(eff_um):.4g}" if eff_um is not None else "n/a"
        log_event(
            f"SOLVER INPUT: Focal={_foc_s}mm, Pixel={_pix_s}um, Bin={_bx}x -> Effective Pixel={_eff_s}um"
        )
        if exp_scale is not None and _foc_mm is not None and eff_um is not None:
            log_event(
                f"PLATE SOLVING: Mierka nastavena na {float(exp_scale):.3f} arcsec/px "
                f"(vypocitane z {float(_foc_mm)}mm a {float(eff_um)}um)"
            )
        elif exp_scale is not None:
            log_event(f"PLATE SOLVING: Mierka nastavena na {float(exp_scale):.3f} arcsec/px")

        try:
            if bool(cfg_u.debug_platesolver):
                log_event(
                    f"DEBUG: _try_vyvar hint_ra={hint_ra} hint_dec={hint_dec} is_masterstar={_is_masterstar}"
                )
        except Exception:  # noqa: BLE001
            pass

        # Ensure per-frame images have VYTARGRA/VYTARGDE in FITS header so vyvar_platesolver
        # can avoid blind solving (it reads hints from the header, not from caller args).
        if (not _is_masterstar) and hint_ra is not None and hint_dec is not None:
            try:
                with fits.open(fp_solve, mode="update", memmap=False) as hdul:
                    h0 = hdul[0].header
                    if "VYTARGRA" not in h0 or "VYTARGDE" not in h0:
                        h0["VYTARGRA"] = (float(hint_ra), "VYVAR plate-solve hint RA [deg] ICRS")
                        h0["VYTARGDE"] = (float(hint_dec), "VYVAR plate-solve hint Dec [deg] ICRS")
                        hdul.flush()
                log_event(f"INFO: Per-frame VYTARG zapisany: RA={float(hint_ra):.4f} Dec={float(hint_dec):.4f}")
            except Exception as e:  # noqa: BLE001
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().vytarg_header_write_fail += 1
                LOGGER.error("Per-frame VYTARG header write failed: %s", e)

        _no_sip = os.environ.get("VYVAR_PLATE_SOLVE_NO_SIP", "").strip().lower() in {"1", "true", "yes", "on"}
        # For MASTERSTAR only: hint mirror orientation from its own header (VY_MIRR) / parity.
        if _is_masterstar:
            try:
                preferred_mirror = _get_masterstar_wcs_parity(Path(fp_solve))
            except Exception:  # noqa: BLE001
                preferred_mirror = None

        return _finalize_plate_solve_result(
            solve_wcs_with_local_gaia(
                fp_solve,
                hint_ra_deg=float(hint_ra) if hint_ra is not None else None,
                hint_dec_deg=float(hint_dec) if hint_dec is not None else None,
                fov_diameter_deg=float(plate_solve_fov_deg),
                gaia_db_path=Path(gaia_db),
                enable_sip=not _no_sip,
                effective_pixel_um=eff_um,
                focal_length_mm=float(_foc_mm) if _foc_mm is not None else None,
                expected_plate_scale_arcsec_per_px=exp_scale,
                preferred_mirror=preferred_mirror,
                max_catalog_rows=100000 if _is_masterstar else None,
                faintest_mag_limit=18.0 if _is_masterstar else None,
            )
        )

    if be == "auto":
        cfg_a = app_config or AppConfig()
        b_auto = _plate_solve_input_bundle(
            fits_path, app_config=cfg_a, equipment_id=equipment_id, draft_id=draft_id
        )
        log_event('Plate-solve backend "auto": len VYVAR lokalny solver (bez Astrometry.net).')
        return _try_vyvar(bundle=b_auto)

    # Accept legacy backend aliases for backward compatibility.
    if be in {"vyvar", "vyvar_platesolver", "vyvar_gaia"}:
        return _try_vyvar(bundle=None)

    if be in {"astrometry.net", "astrometry_net", "online", "net"}:
        LOGGER.warning(
            "Plate-solve backend %r uz nie je podporovany - pouzivam VYVAR (lokalna Gaia).", be
        )
        cfg_n = app_config or AppConfig()
        b_net = _plate_solve_input_bundle(
            fits_path, app_config=cfg_n, equipment_id=equipment_id, draft_id=draft_id
        )
        return _try_vyvar(bundle=b_net)

    LOGGER.warning("Unknown platesolve backend %r; using VYVAR.", be)
    return _try_vyvar(bundle=None)


def _has_valid_wcs(header: fits.Header) -> bool:
    return fits_header_has_celestial_wcs(header)



def _bin2d_mean(arr: "np.ndarray", factor: int) -> "np.ndarray":
    """Mean-bin a 2D array by integer ``factor`` (>=1). Used to speed up reference scoring."""
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or int(factor) < 2:
        return arr
    f = int(factor)
    h, w = arr.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return arr
    a = arr[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    return np.mean(a, axis=(1, 3)).astype(np.float32)


def _dao_star_count_from_array(arr: "np.ndarray", *, fwhm_px: float = 3.0) -> int:
    """Count DAOStarFinder sources (same recipe as alignment star detection)."""
    import numpy as np
    from photutils.detection import DAOStarFinder

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return 0
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0
    _, med, std = plain_mean_med_std(arr[finite], sigma=3.0, maxiters=5)
    std = float(std) if np.isfinite(std) else 0.0
    if std <= 0:
        return 0
    img2 = np.nan_to_num((arr - float(med)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        finder = DAOStarFinder(
            fwhm=float(fwhm_px),
            threshold=max(3.0 * std, 1e-6),
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(img2)
    except Exception:  # noqa: BLE001
        return 0
    if tbl is None or len(tbl) == 0:
        return 0
    return int(len(tbl))


def _pick_reference_frame_by_star_count(files: list[Path]) -> tuple[Path, dict[str, int]]:
    """Choose the FITS with the most detected stars as alignment / plate-solve reference.

    Uses 2x2 binning + slightly smaller DAO FWHM for speed; preserves relative rankings.
    """
    import numpy as np

    scores: dict[str, int] = {}
    for fp in files:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            data_b = _bin2d_mean(data, 2)
            # 2x2 binning: FWHM v px ~ sips_dao_fwhm_px/2 (sirsie okno pre komu v rohoch).
            scores[str(fp)] = _dao_star_count_from_array(data_b, fwhm_px=2.5)
        except Exception:  # noqa: BLE001
            scores[str(fp)] = 0
    if not files:
        raise ValueError("no FITS files")
    if not scores or max(scores.values(), default=0) <= 0:
        return files[0], scores
    best_n = max(scores.values())
    for fp in files:
        if scores.get(str(fp), 0) == best_n:
            return fp, scores
    return files[0], scores


def _wcs_field_center_radec_deg(fits_path: Path) -> tuple[float, float] | None:
    """RA/Dec (deg) of image center from existing celestial WCS."""

    try:
        with fits.open(fits_path, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
        h = int(hdr.get("NAXIS2", 0) or 0)
        wpx = int(hdr.get("NAXIS1", 0) or 0)
        if h <= 0 or wpx <= 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not w.has_celestial:
            return None
        c = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:  # noqa: BLE001
        return None


def _catalog_df_cap_brightest_by_mag(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """Keep at most ``max_rows`` catalog rows, brightest first (lowest ``mag``)."""
    if df is None or getattr(df, "empty", True):
        return df
    try:
        cap = int(max_rows) if max_rows is not None else int(AppConfig().catalog_query_max_rows)
    except Exception:  # noqa: BLE001
        cap = 50_000
    cap = max(1000, min(500_000, cap))
    if len(df) <= cap:
        return df
    out = df.copy()
    if "mag" not in out.columns:
        return out.iloc[:cap].copy().reset_index(drop=True)
    m = pd.to_numeric(out["mag"], errors="coerce")
    out = out.assign(_vyvar_mag_sort=m)
    out = out.sort_values("_vyvar_mag_sort", na_position="last").head(int(cap))
    return out.drop(columns=["_vyvar_mag_sort"], errors="ignore").reset_index(drop=True)


def _query_gaia_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    gaia_db_path: Path | None,
    max_mag: float | None = None,
    focal_mm_for_log: float | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Query **local Gaia DR3 SQLite** for the field; return unified dataframe (ICRS deg)."""
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        _ra_l, _de_l = float("nan"), float("nan")
    _f_l = f"{float(focal_mm_for_log):g}" if focal_mm_for_log is not None else "?"
    log_event(
        f"CATALOG SEARCH (GAIA local): Ra={_ra_l}, Dec={_de_l}, Radius={float(radius_deg):.2f} deg (pre {_f_l}mm)"
    )
    if gaia_db_path is None:
        return pd.DataFrame()
    gp = Path(gaia_db_path).expanduser().resolve()
    if not gp.is_file():
        return pd.DataFrame()

    ra_min = float(_ra_l) - float(radius_deg)
    ra_max = float(_ra_l) + float(radius_deg)
    de_min = float(_de_l) - float(radius_deg)
    de_max = float(_de_l) + float(radius_deg)
    # query_local_gaia: mag_limit=None => no g_mag SQL cap; pass max_mag only when set
    # so MASTERSTAR / cone export honor faintest_mag_limit in SQL (not only a redundant pandas filter).
    _ql_kw: dict[str, Any] = {
        "ra_min": ra_min,
        "ra_max": ra_max,
        "dec_min": de_min,
        "dec_max": de_max,
    }
    if max_mag is not None:
        try:
            _mm = float(max_mag)
            if math.isfinite(_mm) and _mm > 0:
                _ql_kw["mag_limit"] = _mm
        except (TypeError, ValueError):
            pass
    # SQLite uses a square ra/dec box + ORDER BY g_mag LIMIT. For wide boxes (optics floor vs WCS) the 100k
    # brightest stars are often far from the field center, so cone stars never enter the result set.
    _cap_out: int | None = None
    if max_rows is not None:
        try:
            _mr0 = int(max_rows)
            if _mr0 > 0:
                _cap_out = _mr0
        except (TypeError, ValueError):
            _cap_out = None
    _sql_fetch = _cap_out
    if _cap_out is not None and float(radius_deg) > 6.0:
        _af = max(1.0, (float(radius_deg) / 5.5) ** 2)
        # Oversample for cone cut; cap keeps SQLite practical (ORDER BY g_mag on huge boxes is costly).
        _sql_fetch = min(800_000, max(_cap_out, int(_cap_out * _af * 2.5)))
    if _sql_fetch is not None:
        _ql_kw["max_rows"] = int(_sql_fetch)
    rows = query_local_gaia(gp, **_ql_kw)
    if not rows:
        return pd.DataFrame()
    df0 = pd.DataFrame(rows)
    if "bp_rp" not in df0.columns and "bp_mag" in df0.columns and "rp_mag" in df0.columns:
        df0["bp_rp"] = pd.to_numeric(df0["bp_mag"], errors="coerce") - pd.to_numeric(
            df0["rp_mag"], errors="coerce"
        )
    df = df0.rename(
        columns={"source_id": "catalog_id", "ra": "ra_deg", "dec": "dec_deg", "g_mag": "mag", "bp_rp": "bp_rp"}
    )
    df["catalog"] = "GAIA_DR3"
    # Great-circle cone cut (query box is square; LIMIT is isotropic in mag, not in radius).
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if bool(_okq.any()):
        sub = df.loc[_okq].copy()
        _coo_q = SkyCoord(
            ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
            dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
            frame="icrs",
        )
        _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
        df = sub.loc[_inner].reset_index(drop=True)
    if max_mag is not None and "mag" in df.columns:
        m = pd.to_numeric(df["mag"], errors="coerce")
        df = df[(m.notna()) & (m <= float(max_mag))].copy()
    df = df.reset_index(drop=True)
    # Gaia provides BP-RP; do not map it into B-V (different color index).
    if "b_v" not in df.columns:
        df = df.copy()
        df["b_v"] = np.nan
    if "bp_rp" in df.columns:
        df["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    return _catalog_df_cap_brightest_by_mag(df, max_rows=_cap_out)


def _query_vsx_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    vsx_db_path: Path | None,
    max_rows: int | None = None,
    require_db: bool = False,
) -> pd.DataFrame:
    """Query **local VSX SQLite** for the field; same kuzel ako Gaia (najprv obdlznik, potom great-circle orez)."""
    from database import count_vsx_local_rows, require_vsx_local_db_path

    if require_db:
        vp = require_vsx_local_db_path(vsx_db_path)
        n_total = count_vsx_local_rows(vp)
    elif vsx_db_path is None:
        return pd.DataFrame()
    else:
        vp = Path(vsx_db_path).expanduser().resolve()
        if not vp.is_file():
            return pd.DataFrame()
        n_total = count_vsx_local_rows(vp)
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        if require_db:
            raise
        return pd.DataFrame()
    ra_min = float(_ra_l) - float(radius_deg)
    ra_max = float(_ra_l) + float(radius_deg)
    de_min = float(_de_l) - float(radius_deg)
    de_max = float(_de_l) + float(radius_deg)
    _cap = max_rows
    if _cap is None:
        try:
            _cap = int(AppConfig().catalog_query_max_rows)
        except Exception:  # noqa: BLE001
            _cap = 500_000
        _cap = max(10_000, min(500_000, int(_cap)))
    if float(radius_deg) > 6.0:
        _af = max(1.0, (float(radius_deg) / 5.5) ** 2)
        _cap = min(800_000, max(int(_cap), int(_cap * _af * 2.5)))

    rows = query_local_vsx(
        vp,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=de_min,
        dec_max=de_max,
        max_rows=int(_cap),
    )
    if not rows:
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        if require_db:
            from database import VSXCatalogError

            raise VSXCatalogError(f"VSX query returned unexpected columns from {vp}.")
        return pd.DataFrame()
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if not bool(_okq.any()):
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        return pd.DataFrame()
    sub = df.loc[_okq].copy()
    _coo_q = SkyCoord(
        ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        frame="icrs",
    )
    _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
    out = sub.loc[_inner].reset_index(drop=True)
    if require_db and out.empty:
        log_event(
            f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
            f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
        )
    elif not out.empty:
        try:
            log_event(
                f"CATALOG SEARCH (VSX local): {len(out)} zdrojov v kuzeli r~{float(radius_deg):.3f} deg "
                f"(Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        except Exception:  # noqa: BLE001
            pass
    return out


_EXO_HOST_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "exo_host_obj_id",
    "exo_host_name",
    "exo_cat_source",
    "exo_disposition",
    "exo_match_sep_arcsec",
)


def _query_exoplanet_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    exoplanet_db_path: Path | None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Query local exoplanet host SQLite for the field; box query + great-circle cone filter."""
    if exoplanet_db_path is None:
        return pd.DataFrame()
    from database import require_exoplanet_local_db_path  # noqa: PLC0415

    ep = require_exoplanet_local_db_path(exoplanet_db_path)
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    ra_min = float(_ra_l) - float(radius_deg)
    ra_max = float(_ra_l) + float(radius_deg)
    de_min = float(_de_l) - float(radius_deg)
    de_max = float(_de_l) + float(radius_deg)
    _cap = max_rows
    if _cap is None:
        try:
            _cap = int(AppConfig().catalog_query_max_rows)
        except Exception:  # noqa: BLE001
            _cap = 500_000
        _cap = max(10_000, min(500_000, int(_cap)))

    rows = query_local_exoplanet(
        ep,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=de_min,
        dec_max=de_max,
        max_rows=int(_cap),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        return pd.DataFrame()
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if not bool(_okq.any()):
        return pd.DataFrame()
    sub = df.loc[_okq].copy()
    _coo_q = SkyCoord(
        ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        frame="icrs",
    )
    _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
    out = sub.loc[_inner].reset_index(drop=True)
    try:
        log_event(
            f"CATALOG SEARCH (exoplanet local): {len(out)} hostov v kuzeli r~{float(radius_deg):.3f} deg "
            f"(Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def _exo_host_annotation_arrays(
    det_coords: SkyCoord,
    exo_df: pd.DataFrame,
    max_sep_arcsec: float,
    *,
    frame_name: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Nearest exoplanet host per detection within ``max_sep_arcsec`` (informational only)."""
    import numpy as np
    from astropy.coordinates import search_around_sky

    n = len(det_coords)
    out: dict[str, Any] = {
        "exo_host_obj_id": np.array([""] * n, dtype=object),
        "exo_host_name": np.array([""] * n, dtype=object),
        "exo_cat_source": np.array([""] * n, dtype=object),
        "exo_disposition": np.array([""] * n, dtype=object),
        "exo_match_sep_arcsec": np.full(n, np.nan, dtype=np.float64),
    }
    warnings_out: list[str] = []
    if exo_df is None or exo_df.empty or n == 0:
        return out, warnings_out

    exc = SkyCoord(
        ra=np.asarray(exo_df["ra_deg"], dtype=float) * u.deg,
        dec=np.asarray(exo_df["dec_deg"], dtype=float) * u.deg,
        frame="icrs",
    )
    max_sep = float(max_sep_arcsec)
    idx_nearest, sep2d, _ = det_coords.match_to_catalog_sky(exc)
    sep_arc = np.asarray(sep2d.to(u.arcsec).value, dtype=np.float64)
    hit = np.isfinite(sep_arc) & (sep_arc <= max_sep)

    try:
        idx_d, idx_e, sep_a, _ = search_around_sky(det_coords, exc, max_sep * u.arcsec)
        counts: dict[int, int] = {}
        for i_d in np.asarray(idx_d, dtype=np.int64):
            counts[int(i_d)] = counts.get(int(i_d), 0) + 1
        for i_d, cnt in counts.items():
            if cnt > 1:
                msg = (
                    f"[EXO MATCH] {cnt} exoplanet hosts within {max_sep:g} arcsec of detection "
                    f"index {i_d}"
                    + (f" ({frame_name})" if frame_name else "")
                )
                warnings_out.append(msg)
                LOGGER.warning(msg)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[EXO MATCH] ambiguity search skipped: %s", exc)

    for i in np.flatnonzero(hit):
        j = int(idx_nearest[i])
        if j < 0 or j >= len(exo_df):
            continue
        row = exo_df.iloc[j]
        out["exo_host_obj_id"][i] = str(row.get("obj_id", "") or "").strip()
        out["exo_host_name"][i] = str(row.get("host_name", "") or "").strip()
        out["exo_cat_source"][i] = str(row.get("cat_source", "") or "").strip()
        out["exo_disposition"][i] = str(row.get("disposition", "") or "").strip()
        out["exo_match_sep_arcsec"][i] = float(sep_arc[i])

    return out, warnings_out


def _slice_exo_annotation(exo_ann: dict[str, Any], keep: Any) -> dict[str, Any]:
    import numpy as np

    k = np.asarray(keep, dtype=bool)
    return {col: np.asarray(exo_ann[col])[k] for col in _EXO_HOST_ANNOTATION_COLUMNS}


def _apply_exo_host_columns_to_proc_df(
    df: pd.DataFrame,
    hdr: fits.Header,
    data_shape: tuple[int, int],
    st: dict[str, Any],
    *,
    frame_name: str = "",
) -> pd.DataFrame:
    """Add informational ``exo_*`` columns when local DB path is configured.

    ``detect_stars_and_match_catalog`` already annotates; this covers the MASTERSTAR fast path.
    """
    if df is None or df.empty:
        return df
    if any(c in df.columns for c in _EXO_HOST_ANNOTATION_COLUMNS):
        return df

    _exo_path: Path | None = None
    try:
        _exs = str(st.get("exoplanet_local_db_path") or "").strip()
        if _exs:
            _exo_path = Path(_exs).expanduser().resolve()
    except Exception:  # noqa: BLE001
        _exo_path = None
    if _exo_path is None or not _exo_path.is_file():
        return df

    exo_max = 3.0
    try:
        exo_max = float(st.get("exoplanet_match_max_sep_arcsec", 3.0))
        if not math.isfinite(exo_max):
            exo_max = 3.0
    except (TypeError, ValueError):
        exo_max = 3.0
    exo_max = max(0.5, min(30.0, float(exo_max)))

    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        return df

    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)
    if not wcs_obj.has_celestial:
        return df

    h, wpx = int(data_shape[0]), int(data_shape[1])
    _fov_hint = st.get("plate_solve_fov_deg")
    try:
        if _fov_hint is not None and not math.isfinite(float(_fov_hint)):
            _fov_hint = None
    except (TypeError, ValueError):
        _fov_hint = None
    if _fov_hint is None:
        try:
            _fov_hint = resolve_plate_solve_fov_deg_hint(
                hdr,
                h,
                wpx,
                database_path=st.get("database_path"),
                equipment_id=st.get("equipment_id"),
                draft_id=st.get("draft_id"),
            )
        except Exception:  # noqa: BLE001
            _fov_hint = None
    if _fov_hint is None:
        try:
            _fov_hint = float(AppConfig().plate_solve_fov_deg)
        except Exception:  # noqa: BLE001
            _fov_hint = None

    center, radius_deg = _effective_field_catalog_cone_radius_deg(
        wcs_obj, h, wpx, _fov_hint, fits_header=hdr
    )
    exo_df = _query_exoplanet_local(
        center=center,
        radius_deg=radius_deg,
        exoplanet_db_path=_exo_path,
    )

    ra = pd.to_numeric(df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.any(np.isfinite(ra) & np.isfinite(de)):
        return df

    det_coords = SkyCoord(ra=ra * u.deg, dec=de * u.deg, frame="icrs")
    exo_ann, _ = _exo_host_annotation_arrays(
        det_coords,
        exo_df if exo_df is not None else pd.DataFrame(),
        exo_max,
        frame_name=frame_name,
    )
    out = df.copy()
    for col in _EXO_HOST_ANNOTATION_COLUMNS:
        out[col] = exo_ann[col]
    return out


def resolve_masterstars_metadata_csv(platesolve_dir: Path | str) -> Path | None:
    """Return masterstars metadata CSV in a platesolve setup dir (full_match preferred)."""
    ps = Path(platesolve_dir)
    for name in ("masterstars_full_match.csv", "masterstars.csv"):
        p = ps / name
        if p.is_file():
            return p
    return None


def _build_exoplanet_promotion_rows_from_masterstars(
    master_df: pd.DataFrame,
    hdr: fits.Header,
    cfg: AppConfig,
    *,
    frame_w_px: int,
    frame_h_px: int,
    margin_px: float = 50.0,
) -> pd.DataFrame:
    """Promote masterstars rows with exoplanet host match within configured separation."""
    _exs = str(cfg.exoplanet_local_db_path or "").strip()
    if not _exs:
        return pd.DataFrame()
    from database import require_exoplanet_local_db_path  # noqa: PLC0415

    try:
        _exo_path = require_exoplanet_local_db_path(_exs)
    except Exception:
        raise
    if master_df is None or master_df.empty:
        return pd.DataFrame()
    if "catalog_id" not in master_df.columns or "ra_deg" not in master_df.columns or "dec_deg" not in master_df.columns:
        return pd.DataFrame()

    import numpy as np

    from gaia_catalog_id import catalog_id_series_for_masterstars_export, masterstar_row_gaia_key

    m = master_df.copy()
    if "catalog_id" in m.columns:
        m["catalog_id"] = catalog_id_series_for_masterstars_export(m)
    cid_s = m["catalog_id"].fillna("").astype(str).str.strip()
    ok_cid = cid_s.ne("") & ~cid_s.str.lower().isin({"nan", "none"})
    ra = pd.to_numeric(m["ra_deg"], errors="coerce")
    de = pd.to_numeric(m["dec_deg"], errors="coerce")
    ok_sky = ra.notna() & de.notna()
    m = m.loc[ok_cid & ok_sky].copy()
    if m.empty:
        return pd.DataFrame()

    if "x" in m.columns and "y" in m.columns:
        xn = pd.to_numeric(m["x"], errors="coerce")
        yn = pd.to_numeric(m["y"], errors="coerce")
        in_frame = xn.between(-float(margin_px), float(frame_w_px) + float(margin_px)) & yn.between(
            -float(margin_px), float(frame_h_px) + float(margin_px)
        )
        m = m.loc[in_frame].copy()
    if m.empty:
        return pd.DataFrame()

    exo_max = 3.0
    try:
        exo_max = float(cfg.exoplanet_match_max_sep_arcsec)
        if not math.isfinite(exo_max):
            exo_max = 3.0
    except (TypeError, ValueError):
        exo_max = 3.0
    exo_max = max(0.5, min(30.0, float(exo_max)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)
    if not wcs_obj.has_celestial:
        return pd.DataFrame()
    try:
        _fov_hint = float(cfg.plate_solve_fov_deg)
        if not math.isfinite(_fov_hint):
            _fov_hint = None
    except (TypeError, ValueError):
        _fov_hint = None
    center, radius_deg = _effective_field_catalog_cone_radius_deg(
        wcs_obj, int(frame_h_px), int(frame_w_px), _fov_hint, fits_header=hdr
    )
    exo_df = _query_exoplanet_local(
        center=center,
        radius_deg=radius_deg,
        exoplanet_db_path=_exo_path,
    )
    n_hosts_in_field = int(len(exo_df)) if exo_df is not None and not exo_df.empty else 0

    ra_arr = pd.to_numeric(m["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de_arr = pd.to_numeric(m["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    det_coords = SkyCoord(ra=ra_arr * u.deg, dec=de_arr * u.deg, frame="icrs")
    exo_ann, _ = _exo_host_annotation_arrays(
        det_coords,
        exo_df if exo_df is not None else pd.DataFrame(),
        exo_max,
    )

    exo_ids = np.asarray(exo_ann["exo_host_obj_id"], dtype=object)
    exo_sep = np.asarray(exo_ann["exo_match_sep_arcsec"], dtype=np.float64)
    promote_mask = np.zeros(len(m), dtype=bool)
    for i in range(len(m)):
        oid = str(exo_ids[i] or "").strip()
        sep = float(exo_sep[i]) if math.isfinite(float(exo_sep[i])) else float("nan")
        promote_mask[i] = bool(oid) and math.isfinite(sep) and sep <= exo_max
    if not bool(np.any(promote_mask)):
        log_event(
            f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
            f"promoted=0 sep_max={exo_max:g} arcsec"
        )
        return pd.DataFrame()

    sub = m.loc[promote_mask].copy()
    prom_idx = np.where(promote_mask)[0]

    from gaia_catalog_id import masterstar_row_gaia_key

    rows: list[dict[str, Any]] = []
    for j, (_, row) in enumerate(sub.iterrows()):
        ii = int(prom_idx[j])
        cid_norm = masterstar_row_gaia_key(row)
        if not cid_norm:
            continue
        host_name = str(exo_ann["exo_host_name"][ii] or "").strip()
        obj_id = str(exo_ann["exo_host_obj_id"][ii] or "").strip()
        disp = str(exo_ann["exo_disposition"][ii] or "").strip()
        src = str(exo_ann["exo_cat_source"][ii] or "").strip()
        try:
            mag_v = float(pd.to_numeric(row.get("mag", row.get("phot_g_mean_mag")), errors="coerce"))
        except (TypeError, ValueError):
            mag_v = float("nan")
        if not math.isfinite(mag_v):
            mag_v = float("nan")
        rows.append(
            {
                "name": host_name or obj_id,
                "catalog_id": cid_norm,
                "catalog": "EXOPLANET",
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
                "priority": 2,
                "notes": f"{src} {disp}".strip(),
                "vsx_name": "",
                "vsx_type": "",
                "vsx_period": np.nan,
                "x": pd.to_numeric(row.get("x"), errors="coerce"),
                "y": pd.to_numeric(row.get("y"), errors="coerce"),
                "mag": mag_v,
                "zone": str(row.get("zone", "") or "").strip().lower(),
                "gaia_match_arcsec": float(exo_sep[ii]),
                "gaia_match_quality": "good",
                "gaia_match_source": "masterstars_exo",
                "vsx_mag_max": np.nan,
                "exo_host_obj_id": obj_id,
                "exo_host_name": host_name,
                "exo_cat_source": src,
                "exo_disposition": disp,
                "exo_match_sep_arcsec": float(exo_sep[ii]),
                "target_origin": "EXOPLANET",
            }
        )
    if not rows:
        log_event(
            f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
            f"promoted=0 sep_max={exo_max:g} arcsec (no Gaia catalog_id on matched rows)"
        )
        return pd.DataFrame()
    log_event(
        f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
        f"promoted={len(rows)} sep_max={exo_max:g} arcsec"
    )
    log_event(f"[EXO TARGET] {len(rows)} exoplanet host(s) promoted from masterstars (<={exo_max:g} arcsec)")
    return pd.DataFrame(rows)


def _merge_vsx_exoplanet_variable_targets(
    vsx_df: pd.DataFrame,
    exo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge VSX + exoplanet promotion rows; same Gaia id -> one row with VSX labels + exo_*."""
    from gaia_catalog_id import normalize_gaia_source_id_series

    vsx = vsx_df.copy() if vsx_df is not None else pd.DataFrame()
    exo = exo_df.copy() if exo_df is not None else pd.DataFrame()
    if vsx.empty and exo.empty:
        return vsx
    if "catalog_id" in vsx.columns:
        vsx["catalog_id"] = normalize_gaia_source_id_series(vsx["catalog_id"])
    if not exo.empty and "catalog_id" in exo.columns:
        exo["catalog_id"] = normalize_gaia_source_id_series(exo["catalog_id"])

    exo_extra = list(_EXO_HOST_ANNOTATION_COLUMNS) + ["target_origin"]
    for col in exo_extra:
        if col not in vsx.columns:
            vsx[col] = ""
    if exo.empty:
        return vsx

    if vsx.empty:
        return exo

    exo_by_cid: dict[str, pd.Series] = {}
    for _, er in exo.iterrows():
        cid = str(er.get("catalog_id", "") or "").strip()
        if cid:
            exo_by_cid[cid] = er

    vsx_cids: set[str] = set()
    for i in vsx.index:
        cid = str(vsx.at[i, "catalog_id"] or "").strip()
        if not cid:
            continue
        vsx_cids.add(cid)
        er = exo_by_cid.get(cid)
        if er is not None:
            for col in exo_extra:
                if col in er.index:
                    vsx.at[i, col] = er[col]

    exo_only = exo.loc[~exo["catalog_id"].astype(str).str.strip().isin(vsx_cids)].copy()
    if exo_only.empty:
        return vsx
    merged = pd.concat([vsx, exo_only], ignore_index=True)
    log_event(
        f"[EXO TARGET] variable_targets merge: VSX={len(vsx)} exo-only added={len(exo_only)} "
        f"total={len(merged)}"
    )
    return merged


def _query_vsx_local_frame_bbox(
    *,
    wcs: Any,
    width_px: int,
    height_px: int,
    vsx_db_path: Path | None,
    margin_px: float = 50.0,
    center: SkyCoord | None = None,
    require_db: bool = False,
) -> pd.DataFrame:
    """Query **local VSX** within the FRAME footprint (frame bbox + ``margin_px``), spatial-first.

    Completeness must not depend on row order: the frame bbox is tiny (sub-degree), so the SQL
    result is small and the global ``catalog_query_max_rows`` cap never truncates it (unlike the
    3.5 deg cone in ``_query_vsx_local``, which hit the 15000-row cap and dropped a contiguous Dec
    slice). Returns raw VSX rows over the bbox; the caller applies the precise in-frame pixel
    filter (matching ``margin_px``). RA wrap is handled via centre-relative offsets.
    """
    from database import count_vsx_local_rows, require_vsx_local_db_path

    if require_db:
        vp = require_vsx_local_db_path(vsx_db_path)
        n_total = count_vsx_local_rows(vp)
    elif vsx_db_path is None:
        return pd.DataFrame()
    else:
        vp = Path(vsx_db_path).expanduser().resolve()
        if not vp.is_file():
            return pd.DataFrame()
        n_total = 0
    m = float(margin_px)
    w = float(width_px)
    h = float(height_px)
    # Pixel-space samples covering the frame border + margin (corners + edge midpoints).
    xs_px = np.asarray([-m, w * 0.5, w + m, -m, w + m, -m, w * 0.5, w + m], dtype=np.float64)
    ys_px = np.asarray([-m, -m, -m, h * 0.5, h * 0.5, h + m, h + m, h + m], dtype=np.float64)
    try:
        world = wcs.all_pix2world(xs_px, ys_px, 0)
        ras = np.asarray(world[0], dtype=np.float64)
        decs = np.asarray(world[1], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().vsx_frame_bbox_wcs_fail += 1
        LOGGER.error("[VSX] frame-bbox WCS all_pix2world failed: %s", exc)
        return pd.DataFrame()
    ok = np.isfinite(ras) & np.isfinite(decs)
    if not bool(ok.any()):
        return pd.DataFrame()
    ras = ras[ok]
    decs = decs[ok]
    de_min = float(np.min(decs))
    de_max = float(np.max(decs))
    if center is not None:
        try:
            ra0 = float(center.icrs.ra.deg)
        except Exception:  # noqa: BLE001
            ra0 = float(np.median(ras))
    else:
        ra0 = float(np.median(ras))
    dra = ((ras - ra0 + 180.0) % 360.0) - 180.0
    ra_min = ra0 + float(np.min(dra))
    ra_max = ra0 + float(np.max(dra))
    pad = 1.0 / 3600.0  # 1 arcsec rounding pad
    rows = query_local_vsx(
        vp,
        ra_min=ra_min - pad,
        ra_max=ra_max + pad,
        dec_min=de_min - pad,
        dec_max=de_max + pad,
        max_rows=None,  # frame bbox is tiny -> no cap needed (spatial-first completeness)
    )
    if not rows:
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(frame bbox+{int(m)}px, RA=[{ra_min:.4f},{ra_max:.4f}], Dec=[{de_min:.4f},{de_max:.4f}])"
            )
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        if require_db:
            from database import VSXCatalogError

            raise VSXCatalogError(f"VSX query returned unexpected columns from {vp}.")
        return pd.DataFrame()
    try:
        log_event(
            f"CATALOG SEARCH (VSX local, frame bbox+{int(m)}px): {len(df)} zdrojov "
            f"(RA=[{ra_min:.4f},{ra_max:.4f}], Dec=[{de_min:.4f},{de_max:.4f}], bez cap)"
        )
    except Exception:  # noqa: BLE001
        pass
    return df


def _saturate_limit_adu_from_header(hdr: fits.Header) -> float | None:
    """Return saturation / linearity ceiling in image units (ADU, e-, ...) if present in header."""
    import math

    for key in ("SATURATE", "MAXLIN", "ESATUR", "LINLIMIT", "MAXADU"):
        if key not in hdr:
            continue
        try:
            v = float(hdr[key])
            if math.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


# SAT-LIMIT-01 / GAIN-DOMAIN-01: 16-bit FITS container clip (pile-up at 65535, not 65532).
SAT_LIMIT_CONTAINER_CLIP_ADU = 65535.0
# Peak-test fraction when the linearity knee is unmeasured (D1-2 / SAT-LIMIT-01 B3).
SAT_LIMIT_NO_KNEE_FRAC = 0.80
# Provenance string for the INV-SAT-LIMIT peak-test (catalog zone + per-frame clean).
SAT_LIMIT_PEAK_TEST_SOURCE = (
    f"INV-SAT-LIMIT peak-test {SAT_LIMIT_NO_KNEE_FRAC:.2f}x "
    f"container_clip_{SAT_LIMIT_CONTAINER_CLIP_ADU:.0f}"
)


def inv_sat_limit_peak_test_adu() -> tuple[float, str]:
    """Catalog-zone peak-test (INV-SAT-LIMIT). One authority for zone and per-frame.

    Hard container clip is ``SAT_LIMIT_CONTAINER_CLIP_ADU`` (named separately;
    not used as the clean-frame test). This returns the peak-test used for
    ``zone==saturated``: currently 0.80 x 65535 = 52428 ADU when the D1-2 knee
    is unmeasured. n_stack is not applied here: per-frame CSVs are one exposure.
    """
    peak = float(SAT_LIMIT_CONTAINER_CLIP_ADU) * float(SAT_LIMIT_NO_KNEE_FRAC)
    return peak, SAT_LIMIT_PEAK_TEST_SOURCE


def _finite_positive_adu(value: Any) -> float | None:
    """Return a finite positive ADU, else None (NaN/inf/<=0/unparsable)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(v) and v > 0:
        return v
    return None


def _sat_adu_from_draft_sat_diag(archive_path: Path | None) -> float | None:
    """Read ``sat_diag.json`` ``sat_adu`` from a draft archive root when present."""
    if archive_path is None:
        return None
    try:
        from sat_diag import load_sat_diag_json  # noqa: PLC0415

        ctx = load_sat_diag_json(Path(archive_path) / "sat_diag.json")
    except Exception:  # noqa: BLE001
        return None
    if ctx is None:
        return None
    return _finite_positive_adu(getattr(ctx, "sat_adu", None))


def _infer_sat_limit_from_bitpix(hdr: fits.Header) -> float | None:
    """Infer linearity ceiling from FITS integer layout (e.g. unsigned 16-bit -> 65535)."""
    import math

    try:
        bitpix = int(hdr.get("BITPIX", 0))
    except (TypeError, ValueError):
        return None
    bzero = float(hdr.get("BZERO", 0.0))
    bscale = float(hdr.get("BSCALE", 1.0))
    if not math.isfinite(bzero) or not math.isfinite(bscale) or bscale <= 0:
        return None
    if bitpix == 16:
        # Unsigned 16-bit (common): physical 0...65535 stored with BZERO=32768
        if abs(bzero - 32768.0) < 1.0 and abs(bscale - 1.0) < 1e-9:
            return 65535.0
        # Native signed 16-bit
        if abs(bzero) < 1e-6 and abs(bscale - 1.0) < 1e-9:
            return 32767.0
    return None


def _equipment_saturate_adu_from_db(equipment_id: int | None) -> float | None:
    """Read ``EQUIPMENTS.SATURATE_ADU`` when a valid equipment id is given."""
    if equipment_id is None:
        return None
    try:
        eid = int(equipment_id)
    except (TypeError, ValueError):
        return None
    if eid <= 0:
        return None
    try:
        cfg = AppConfig()
        db = VyvarDatabase(cfg.database_path)
        return db.get_equipment_saturation_adu(eid)
    except Exception:  # noqa: BLE001
        return None


def _effective_saturation_limit(
    hdr: fits.Header,
    *,
    fallback_adu: float | None,
    equipment_saturate_adu: float | None = None,
) -> tuple[float, str]:
    """Resolve saturation ceiling: header keywords -> ``EQUIPMENTS`` / caller ``equipment_saturate_adu`` ->
    ``DATAMAX`` / ``MAXPIX`` -> BITPIX guess -> optional ``fallback_adu`` -> GAIN-DOMAIN-01
    container clip (INV-SAT-LIMIT; never None).
    """
    import math

    lim = _saturate_limit_adu_from_header(hdr)
    if lim is not None:
        return lim, "header_keyword"

    if equipment_saturate_adu is not None:
        fe = float(equipment_saturate_adu)
        if math.isfinite(fe) and fe > 0:
            return fe, "equipment_db"

    for dk in ("DATAMAX", "MAXPIX"):
        if dk not in hdr:
            continue
        try:
            v = float(hdr[dk])
            if math.isfinite(v) and v > 0:
                return v, f"header_{dk.lower()}"
        except (TypeError, ValueError):
            continue

    lim2 = _infer_sat_limit_from_bitpix(hdr)
    if lim2 is not None:
        return lim2, "bitpix"

    if fallback_adu is not None:
        fa = float(fallback_adu)
        if math.isfinite(fa) and fa > 0:
            return fa, "config_fallback"

    # INV-SAT-LIMIT: never return None. MASTERSTAR stacks are float (BITPIX -32),
    # EQUIPMENTS.SATURATE_ADU may be NULL (QHY294MM migration), headers often omit
    # SATURATE/MAXLIN. Silent None made peak>limit comparisons False for the whole catalog.
    logging.warning(
        "[INV-SAT-LIMIT] saturation clip unresolved (header/equipment/BITPIX none); "
        "using GAIN-DOMAIN-01 container clip %.0f ADU",
        SAT_LIMIT_CONTAINER_CLIP_ADU,
    )
    return float(SAT_LIMIT_CONTAINER_CLIP_ADU), "conservative_default_container_clip_65535"


def _box_peak_max_adu(data: "np.ndarray", x: float, y: float, half: int = 3) -> float:
    """Maximum pixel value in ``(2*half+1)^2`` box around ``(x,y)`` on the **original** image (linear units)."""
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim != 2:
        return float("nan")
    h, w = arr.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    if y0 >= y1 or x0 >= x1:
        return float("nan")
    return float(np.nanmax(arr[y0:y1, x0:x1]))


def _box_peaks_at_centroids(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    half: int = 3,
) -> "np.ndarray":
    """Maximum ADU in each ``(2*half+1)^2`` box centred on ``round(x), round(y)`` (vectorized).

    Used for per-star saturation on thousands of DAO detections; matches ``_box_peak_max_adu`` on
    interior pixels. Falls back to a Python loop if SciPy is unavailable.
    """
    import numpy as np

    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(xa.size)
    if n == 0:
        return np.zeros([], dtype=np.float64)
    if n != int(ya.size):
        raise ValueError("x and y must have the same length")

    try:
        from scipy.ndimage import maximum_filter  # type: ignore
    except ImportError:
        return np.asarray(
            [_box_peak_max_adu(arr, float(xa[i]), float(ya[i]), half=half) for i in range(n)],
            dtype=np.float64,
        )

    a = np.asarray(np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    size = int(2 * int(half) + 1)
    mf = maximum_filter(a, size=size, mode="nearest")
    xi = np.rint(xa).astype(np.intp, copy=False)
    yi = np.rint(ya).astype(np.intp, copy=False)
    h, w = mf.shape
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    return np.asarray(mf[yi, xi], dtype=np.float64)


def _icrs_deg_to_unitxyz(ra_deg: "np.ndarray", dec_deg: "np.ndarray") -> "np.ndarray":
    """ICRS degrees -> unit direction vectors on the celestial sphere (N,3)."""
    import numpy as np

    ra = np.radians(np.asarray(ra_deg, dtype=np.float64).ravel())
    de = np.radians(np.asarray(dec_deg, dtype=np.float64).ravel())
    cd = np.cos(de)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(de)])


def _chord_to_arcsec(dist_chord: "np.ndarray") -> "np.ndarray":
    """Chord length between unit sphere points (0...2) -> great-circle separation in arcseconds."""
    import numpy as np

    d = np.asarray(dist_chord, dtype=np.float64)
    half = np.clip(d * 0.5, 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(half)) * 3600.0


def build_ucac_catalog_kdtree(cat_df: pd.DataFrame) -> tuple[Any, "np.ndarray"] | None:
    """Build a SciPy ``cKDTree`` on finite ``ra_deg``/``dec_deg`` rows and row indices into ``cat_df``.

    Read-only ``query`` calls are thread-safe across workers sharing the same tree.
    """
    import numpy as np

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return None
    ra = np.asarray(pd.to_numeric(cat_df["ra_deg"], errors="coerce"), dtype=np.float64)
    de = np.asarray(pd.to_numeric(cat_df["dec_deg"], errors="coerce"), dtype=np.float64)
    m = np.isfinite(ra) & np.isfinite(de)
    if not np.any(m):
        return None
    orig_idx = np.nonzero(m)[0].astype(np.int64)
    xyz = _icrs_deg_to_unitxyz(ra[m], de[m])
    return cKDTree(xyz), orig_idx


def nearest_sky_nn_kdtree(
    tree: Any,
    det_ra_deg: "np.ndarray",
    det_dec_deg: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Nearest catalog point on the sphere for each detection (k=1, same idea as ``match_to_catalog_sky``).

    Returns ``(idx_compact, sep_arcsec)`` where ``idx_compact`` indexes the finite subset used to build
    ``tree``, or ``-1`` if invalid.
    """
    import numpy as np

    det_xyz = _icrs_deg_to_unitxyz(det_ra_deg, det_dec_deg)
    dist, idx = tree.query(det_xyz, k=1)
    sep = _chord_to_arcsec(dist)
    idx_a = np.asarray(idx, dtype=np.int64).ravel()
    dist_a = np.asarray(dist, dtype=np.float64).ravel()
    ntree = int(getattr(tree, "n", 0))
    bad = ~np.isfinite(dist_a) | (idx_a < 0) | (idx_a >= max(ntree, 1))
    sep = np.asarray(sep, dtype=np.float64).ravel()
    sep[bad] = np.inf
    idx_a = idx_a.copy()
    idx_a[bad] = -1
    return idx_a, sep


def _saturated_core_plateau(
    data: "np.ndarray",
    x: float,
    y: float,
    *,
    half_inner: int = 1,
    plateau_rel: float = 0.996,
    min_plateau_pixels: int = 5,
) -> bool:
    """Detect a clipped / ``flat-top`` core: many pixels in the central box sit near the local maximum.

    Mirrors what eyeballing a radial profile shows for saturated stars (plateau vs a smooth Gaussian peak).
    Works on **any** linear image scale (raw or calibrated floats).
    """
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        return False
    h, w = arr.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    hi = max(0, int(half_inner))
    y0, y1 = max(0, yi - hi), min(h, yi + hi + 1)
    x0, x1 = max(0, xi - hi), min(w, xi + hi + 1)
    if y0 >= y1 or x0 >= x1:
        return False
    patch = arr[y0:y1, x0:x1]
    if patch.size < int(min_plateau_pixels):
        return False
    pmax = float(np.nanmax(patch))
    if not np.isfinite(pmax) or pmax <= 0:
        return False
    thr = pmax * float(plateau_rel)
    n_high = int(np.sum(np.isfinite(patch) & (patch >= thr)))
    return n_high >= int(min_plateau_pixels)


def _star_saturation_flags(
    arr: "np.ndarray",
    x: float,
    y: float,
    *,
    sat_limit: float | None,
    sat_frac: float,
    peak_dao_val: float | None,
    peak_max_adu: float | None = None,
) -> dict[str, Any]:
    """Per-star saturation: ADU limit crossing + central plateau (flat core)."""
    import numpy as np

    pmax = float(peak_max_adu) if peak_max_adu is not None else _box_peak_max_adu(arr, float(x), float(y), half=3)
    lim = sat_limit
    sat_by_peak = bool(
        lim is not None and np.isfinite(pmax) and pmax >= float(lim) * float(sat_frac)
    )
    if lim is None:
        sat_by_plateau = _saturated_core_plateau(arr, float(x), float(y))
    else:
        plateau_skip_thr = float(lim) * float(sat_frac) * 0.55
        if np.isfinite(pmax) and pmax < plateau_skip_thr:
            sat_by_plateau = False
        else:
            sat_by_plateau = _saturated_core_plateau(arr, float(x), float(y))
    likely = bool(sat_by_peak)
    return {
        "peak_dao": float(peak_dao_val) if peak_dao_val is not None and np.isfinite(peak_dao_val) else None,
        "peak_max_adu": float(pmax) if np.isfinite(pmax) else None,
        "saturate_limit_adu": float(lim) if lim is not None else None,
        "saturated_from_peak": sat_by_peak,
        "saturated_plateau": sat_by_plateau,
        "likely_saturated": likely,
        "photometry_ok": not likely,
    }


def _all_pix2world_icrs_deg(wcs_obj: WCS, x: "np.ndarray", y: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Vectorized pixel -> world (degrees) for celestial axes; single WCS call (no per-star Python loops)."""
    import numpy as np

    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        world0, world1 = wcs_obj.all_pix2world(xa, ya, 0)
    return (
        np.asarray(world0, dtype=np.float64).ravel(),
        np.asarray(world1, dtype=np.float64).ravel(),
    )


def _saturated_core_plateau_vectorized(
    data: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    half_inner: int = 1,
    plateau_rel: float = 0.996,
    min_plateau_pixels: int = 5,
) -> "np.ndarray":
    """Same criterion as ``_saturated_core_plateau``, vectorized over ``(x,y)`` centroids (3x3 patch per star)."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        return np.zeros(len(np.asarray(x).ravel()), dtype=bool)
    h, w = arr.shape
    n = int(np.asarray(x, dtype=np.float64).size)
    yi = np.rint(np.asarray(y, dtype=np.float64).ravel()).astype(np.int32, copy=False)
    xi = np.rint(np.asarray(x, dtype=np.float64).ravel()).astype(np.int32, copy=False)
    hi = max(0, int(half_inner))
    offs = np.arange(-hi, hi + 1, dtype=np.int32)
    yy = yi.reshape(-1, 1, 1) + offs.reshape(1, -1, 1)
    xx = xi.reshape(-1, 1, 1) + offs.reshape(1, 1, -1)
    yy = np.clip(yy, 0, h - 1)
    xx = np.clip(xx, 0, w - 1)
    patches = arr[yy, xx]
    pmax = np.nanmax(patches.reshape(n, -1), axis=1)
    thr = pmax * float(plateau_rel)
    thr3 = thr.reshape(-1, 1, 1)
    n_high = np.sum(np.isfinite(patches) & (patches >= thr3), axis=(1, 2))
    return (n_high >= int(min_plateau_pixels)) & np.isfinite(pmax) & (pmax > 0)


def _vectorized_star_saturation_columns(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    sat_limit: float | None,
    sat_frac: float,
    peak_dao: "np.ndarray",
    peak_max_adu: "np.ndarray",
) -> dict[str, "np.ndarray"]:
    """Per-star saturation flags as column arrays (replaces ``N`` calls to ``_star_saturation_flags``)."""
    import numpy as np

    n = int(np.asarray(x, dtype=np.float64).size)
    pmax = np.asarray(peak_max_adu, dtype=np.float64).reshape(-1)
    lim = sat_limit
    sf = float(sat_frac)
    pdv = np.asarray(peak_dao, dtype=np.float64).reshape(-1)

    if lim is not None:
        sat_peak = np.isfinite(pmax) & (pmax >= float(lim) * sf)
        plateau_skip_thr = float(lim) * sf * 0.55
        need_plateau = np.isfinite(pmax) & (pmax >= plateau_skip_thr)
    else:
        sat_peak = np.zeros(n, dtype=bool)
        need_plateau = np.ones(n, dtype=bool)

    pl_full = _saturated_core_plateau_vectorized(arr, x, y, half_inner=1, plateau_rel=0.996, min_plateau_pixels=5)
    if lim is None:
        sat_plateau = pl_full
    else:
        sat_plateau = np.where(need_plateau, pl_full, False)

    likely = sat_peak | sat_plateau
    peak_dao_col = pdv.copy()
    peak_dao_col[~np.isfinite(peak_dao_col)] = np.nan
    peak_max_col = pmax.copy()
    peak_max_col[~np.isfinite(peak_max_col)] = np.nan
    sl = np.full(n, np.nan, dtype=np.float64)
    if lim is not None:
        sl[:] = float(lim)
    return {
        "peak_dao": peak_dao_col,
        "peak_max_adu": peak_max_col,
        "saturate_limit_adu": sl,
        "saturated_from_peak": sat_peak,
        "saturated_plateau": sat_plateau,
        "likely_saturated": likely,
        "photometry_ok": ~likely,
    }


def _proc_sat_block_for_csv(sat_block: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """Omit obsolete proc columns; return (csv_block, n_saturated_from_peak, n_saturated_plateau) for meta."""
    import numpy as np

    pk = sat_block.get("saturated_from_peak")
    pl = sat_block.get("saturated_plateau")
    n_pk = int(np.asarray(pk, dtype=bool).sum()) if pk is not None else 0
    n_pl = int(np.asarray(pl, dtype=bool).sum()) if pl is not None else 0
    csv_block = {
        k: v for k, v in sat_block.items() if k not in ("saturated_from_peak", "saturated_plateau")
    }
    return csv_block, n_pk, n_pl


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


def _fits_header_first_positive_float(hdr: fits.Header, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k not in hdr:
            continue
        try:
            v = float(hdr[k])
            if math.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _gaia_catalog_cone_radius_optics_floor_deg(
    hdr: fits.Header | None,
    *,
    naxis1: int,
    naxis2: int,
    plate_solve_fov_fallback_deg: float,
) -> float:
    """Minimum Gaia cone radius from FOCALLEN + PIXSIZE + binning (ignores flawed linear WCS at chip edges).

    A solved TAN-without-SIP header often under-predicts corner separation vs true sky; the rectangular
    SQL prefilter then clips the catalog so QA shows 'missing Gaia' stripes at left/right edges.
    """
    if hdr is None:
        return 0.0
    foc = _fits_header_first_positive_float(
        hdr, ("FOCALLEN", "FOCALLENGTH", "FOCAL", "FOC_LEN")
    )
    pix = _fits_header_first_positive_float(
        hdr, ("PIXSIZE", "XPIXSZ", "PIXSZ", "PIXELSIZE", "PIX_SIZE")
    )
    if foc is None or pix is None:
        return 0.0
    try:
        foc_n, _ = normalize_telescope_focal_mm_for_plate_scale(float(foc))
        xb, _yb = fits_binning_xy_from_header(hdr)
        eff_um = effective_binned_pixel_pitch_um(base_pixel_um_1x1=float(pix), binning=int(xb))
        # margin ~2.0: VYVAR-solver-class corner coverage + CRVAL slop / high-Dec geometry
        return float(
            catalog_cone_radius_deg_from_optics(
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                pixel_pitch_um=float(eff_um),
                focal_length_mm=float(foc_n),
                margin=2.05,
                fov_diameter_fallback_deg=float(plate_solve_fov_fallback_deg),
            )
        )
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().gaia_cone_optics_floor_fail += 1
        LOGGER.error("[CATALOG] Gaia cone optics floor failed; using FOV fallback: %s", exc)
        fb = max(0.05, float(plate_solve_fov_fallback_deg))
        return float(max(MIN_GAIA_CONE_RADIUS_DEG, fb * 0.65))


def _field_center_and_radius_from_wcs(w: WCS, h: int, wpx: int) -> tuple[SkyCoord, float]:
    """Footprint for **one** Vizier/Gaia cone query tied to the detector - not 'cela obloha'.

    Uses the geometric pixel centre, then the **maximum** great-circle separation from that centre to a
    dense sample of the **full rectangle border** (not only corners). With sip / distortion, the farthest
    sky point from the centre is not always a corner - undersized cones produced a visible circular
    'matched only in the middle' QA overlay. Adds multiplicative + absolute margin for edge stars and
    small plate-solve errors.
    """
    center = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
    max_sep_deg = 0.0
    step = max(1, int(round(float(max(wpx, h)) / 120.0)))
    pixels: set[tuple[int, int]] = set()
    for cx, cy in ((0, 0), (wpx - 1, 0), (0, h - 1), (wpx - 1, h - 1)):
        pixels.add((int(cx), int(cy)))
    for x in range(0, wpx, step):
        pixels.add((x, 0))
        pixels.add((x, h - 1))
    for y in range(0, h, step):
        pixels.add((0, y))
        pixels.add((wpx - 1, y))
    for cx, cy in pixels:
        cc = SkyCoord.from_pixel(float(cx), float(cy), wcs=w, origin=0)
        max_sep_deg = max(max_sep_deg, float(center.separation(cc).deg))
    # Margin: WCS at edges, rectangle vs cone on sphere, plate-solve error. Larger margin because
    # ``field_catalog_cone.csv`` built from a cropped MASTERSTAR must not underserve full-chip frames.
    radius_deg = max(max_sep_deg * 1.38, max_sep_deg + 45.0 / 3600.0)
    # If CD scale is wrong (too 'zoomed' on sky), spherical sampling collapses - blend in tangent-plane
    # half-diagonal from pixel scales so the cone stays physically plausible.
    try:
        scales = w.proj_plane_pixel_scales()
        sx_deg = abs(float(scales[0].to(u.deg).value))
        sy_deg = abs(float(scales[1].to(u.deg).value))
        r_cd = float(math.hypot(0.5 * (wpx - 1) * sx_deg, 0.5 * (h - 1) * sy_deg))
        radius_deg = max(radius_deg, r_cd * 1.22)
    except Exception:  # noqa: BLE001
        pass
    radius_deg = max(float(radius_deg), float(MIN_GAIA_CONE_RADIUS_DEG))
    return center, radius_deg


def _effective_field_catalog_cone_radius_deg(
    w: WCS,
    h: int,
    wpx: int,
    plate_solve_fov_deg: float | None,
    fits_header: fits.Header | None = None,
) -> tuple[SkyCoord, float]:
    """WCS-derived cone radius, optics floor from FITS (FOCALLEN+PIXSIZE), optional UI FOV minimum."""
    center, r = _field_center_and_radius_from_wcs(w, h, wpx)
    try:
        _pf_fb = float(plate_solve_fov_deg) if plate_solve_fov_deg is not None else 1.5
        if not math.isfinite(_pf_fb) or _pf_fb <= 0:
            _pf_fb = 1.5
    except (TypeError, ValueError):
        _pf_fb = 1.5
    r_opt = _gaia_catalog_cone_radius_optics_floor_deg(
        fits_header,
        naxis1=int(wpx),
        naxis2=int(h),
        plate_solve_fov_fallback_deg=float(_pf_fb),
    )
    r = max(float(r), float(r_opt))
    r_physical = float(r)
    if plate_solve_fov_deg is not None:
        try:
            pf = float(plate_solve_fov_deg)
            if math.isfinite(pf) and pf > 0:
                r_fov = catalog_cone_radius_from_fov_diameter_deg(pf)
                # UI FOV je bezpecnostne minimum, ale nesmie 'rozsirit' kuzel ovela nad realny cip:
                # pri zle nastavenom velkom FOV (napr. 20 deg+) by inak vznikol polomer ~13 deg a SQL by
                # tahalo 500k+ hviezd (minuty behu). Ak uz WCS + optika davaju rozumny polomer,
                # obmedzime prispevok z FOV na ~30 % nad fyzikalnu stopu.
                if r_physical >= 2.5:
                    r_fov_eff = min(float(r_fov), r_physical * 1.30 + 0.35)
                else:
                    r_fov_eff = float(r_fov)
                r = min(22.0, max(r_physical, r_fov_eff))
        except (TypeError, ValueError):
            pass
    return center, float(r)


def _invalidate_field_catalog_cone_cache_if_needed(
    field_catalog_csv: Path,
    *,
    plate_solve_fov_deg: float | None,
    effective_radius_deg: float,
) -> None:
    """Remove ``field_catalog_cone.csv`` + meta when UI FOV or required cone size no longer matches cache."""
    p_csv = Path(field_catalog_csv)
    meta_p = _field_catalog_cone_meta_path(p_csv)
    if not p_csv.is_file() and not meta_p.is_file():
        return
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    except Exception:  # noqa: BLE001
        meta = {}
    r_stored = float(meta.get("cone_radius_deg") or 0.0)
    fov_stored = meta.get("plate_solve_fov_deg")
    slack_deg = 45.0 / 3600.0
    reasons: list[str] = []

    if plate_solve_fov_deg is not None:
        try:
            pf = float(plate_solve_fov_deg)
            if math.isfinite(pf):
                if fov_stored is None:
                    reasons.append("meta chyba plate_solve_fov_deg (stary cache)")
                else:
                    try:
                        if abs(float(fov_stored) - pf) > 1e-4:
                            reasons.append(f"plate_solve_fov_deg {float(fov_stored):.6f} -> {pf:.6f}")
                    except (TypeError, ValueError):
                        reasons.append("neplatny ulozeny plate_solve_fov_deg")
        except (TypeError, ValueError):
            pass

    r_eff = float(effective_radius_deg)
    if r_stored > 0 and r_eff > r_stored * 1.02 + slack_deg:
        reasons.append(f"kuzel r~{r_eff:.4f} deg > ulozene {r_stored:.4f} deg")

    if not reasons:
        return
    p_csv.unlink(missing_ok=True)
    meta_p.unlink(missing_ok=True)
    log_event("Katalog: field_catalog_cone cache vymazana - " + "; ".join(reasons) + " (nacitam nanovo).")


def _field_catalog_cone_meta_path(field_catalog_csv: Path) -> Path:
    return field_catalog_csv.parent / "field_catalog_cone_meta.json"


def _write_field_catalog_cone_meta(
    field_catalog_csv: Path,
    *,
    center: SkyCoord,
    radius_deg: float,
    naxis1: int,
    naxis2: int,
    plate_solve_fov_deg: float | None = None,
) -> None:
    """Persist cone parameters used to build ``field_catalog_cone.csv`` (cache invalidation for larger chips)."""
    p = _field_catalog_cone_meta_path(field_catalog_csv)
    rec = {
        "cone_radius_deg": float(radius_deg),
        "center_ra_icrs_deg": float(center.ra.deg),
        "center_dec_icrs_deg": float(center.dec.deg),
        "naxis1": int(naxis1),
        "naxis2": int(naxis2),
        "plate_solve_fov_deg": float(plate_solve_fov_deg)
        if plate_solve_fov_deg is not None and math.isfinite(float(plate_solve_fov_deg))
        else None,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")


def select_comparison_stars_spatial_grid(
    df: pd.DataFrame,
    *,
    width_px: float,
    height_px: float,
    n_comp: int = 0,
    require_catalog_match: bool = True,
    require_photometry_ok: bool = True,
    require_non_variable: bool = True,
    variable_targets_df: pd.DataFrame | None = None,
    safe_bbox: tuple[float, float, float, float] | None = None,
    exclude_nonlinear_badcolumn: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose comparison stars for ensemble photometry: ~one brightest candidate per spatial grid cell.

    Uses catalog-matched, photometry-safe detections, stratified on a nxxny grid sized for ``n_comp`` and
    image aspect ratio so stars are spread across the full detector (typical 100-200 comps for APASS-like work).

    When ``require_non_variable`` is True and column ``catalog_known_variable`` exists, stars flagged as
    known variables (VSX and/or Gaia ``phot_variable_flag``) are excluded from the ensemble.
    """
    import math

    import numpy as np

    if df.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "empty"}

    work = df.copy()
    if "is_usable" in work.columns:
        work = work[work["is_usable"].fillna(False).astype(bool)]
    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_usable_filter"}
    try:
        from config import AppConfig  # noqa: PLC0415

        _seed_pool = bool(getattr(AppConfig(), "masterstar_forced_seed_comp_pool_enabled", False))
    except Exception:  # noqa: BLE001
        _seed_pool = False
    if not _seed_pool:
        if "source_state" in work.columns:
            work = work[work["source_state"].astype(str).str.strip() != "FORCED_SEED"]
        if "forced_photometry" in work.columns:
            _fp = work["forced_photometry"].astype(str).str.strip().str.lower()
            work = work[~_fp.isin(["true", "1", "yes"])]
        if work.empty:
            return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_seed_pool_filter"}
    if require_catalog_match and "catalog" in work.columns:
        work = work[work["catalog"].fillna("").astype(str).str.strip() != ""]
    if require_photometry_ok and "photometry_ok" in work.columns:
        work = work[work["photometry_ok"].fillna(True).astype(bool)]
    if require_non_variable and "catalog_known_variable" in work.columns:
        work = work[~work["catalog_known_variable"].fillna(False).astype(bool)]
    # Proximity veto: exclude comp candidates within 10 arcsec of any VSX variable target.
    # This is intentionally coordinate-based (not catalog_id-based), because VSX->Gaia cross-id can resolve
    # to a different nearby Gaia source_id than the comp candidate, yet still represent the same physical star.
    if (
        variable_targets_df is not None
        and not getattr(variable_targets_df, "empty", True)
        and "ra_deg" in work.columns
        and "dec_deg" in work.columns
        and "ra_deg" in variable_targets_df.columns
        and "dec_deg" in variable_targets_df.columns
    ):
        try:
            from astropy.coordinates import SkyCoord  # noqa: PLC0415
            import astropy.units as u  # noqa: PLC0415

            w_ra = pd.to_numeric(work["ra_deg"], errors="coerce")
            w_de = pd.to_numeric(work["dec_deg"], errors="coerce")
            ok_w = w_ra.notna() & w_de.notna()
            v_ra = pd.to_numeric(variable_targets_df["ra_deg"], errors="coerce")
            v_de = pd.to_numeric(variable_targets_df["dec_deg"], errors="coerce")
            ok_v = v_ra.notna() & v_de.notna()
            if bool(ok_w.any()) and bool(ok_v.any()):
                comp_coo = SkyCoord(
                    ra=w_ra.loc[ok_w].astype(float).to_numpy() * u.deg,
                    dec=w_de.loc[ok_w].astype(float).to_numpy() * u.deg,
                    frame="icrs",
                )
                vsx_coo = SkyCoord(
                    ra=v_ra.loc[ok_v].astype(float).to_numpy() * u.deg,
                    dec=v_de.loc[ok_v].astype(float).to_numpy() * u.deg,
                    frame="icrs",
                )
                _idx, sep2d, _ = comp_coo.match_to_catalog_sky(vsx_coo)
                veto_arcsec = 10.0
                near = sep2d <= (float(veto_arcsec) * u.arcsec)
                n_removed = int(getattr(near, "sum", lambda: 0)())
                if n_removed > 0:
                    drop_idx = w_ra.loc[ok_w].index[np.asarray(near, dtype=bool)]
                    work = work.drop(index=drop_idx)
                log_event(
                    f"[COMP SELECT] Proximity veto: removed {n_removed} comp candidates within {float(veto_arcsec):.0f} arcsec of VSX targets"
                )
        except Exception:  # noqa: BLE001
            pass
    # Annulus-aware border filter: keep only stars whose centroid is within the safe bbox
    # (intersection of aligned frames shrunk by sky-annulus outer radius).
    if safe_bbox is not None and "x" in work.columns and "y" in work.columns:
        try:
            from aperture_policy import stars_fit_on_chip  # noqa: PLC0415

            x0b, y0b, x1b, y1b = safe_bbox
            before = int(len(work))
            # 4-tuple naxis = precomputed safe bbox (already shrunk by resolver r_out).
            _on = stars_fit_on_chip(
                work["x"], work["y"], (0.0, 0.0, 0.0), (float(x0b), float(y0b), float(x1b), float(y1b))
            )
            work = work[np.asarray(_on, dtype=bool)]
            removed = before - int(len(work))
            if removed > 0:
                logging.info(
                    f"[BORDER] Comp selection: removed {removed} candidates outside safe bbox "
                    f"(annulus-aware intersection)"
                )
        except Exception:  # noqa: BLE001
            pass
    if exclude_nonlinear_badcolumn and "likely_nonlinear" in work.columns:
        _ln = pd.to_numeric(work["likely_nonlinear"], errors="coerce").fillna(0).astype(int)
        work = work[_ln == 0]
    if exclude_nonlinear_badcolumn and "on_bad_column" in work.columns:
        _ob = pd.to_numeric(work["on_bad_column"], errors="coerce").fillna(0).astype(int)
        work = work[_ob == 0]

    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_filter"}

    w = float(width_px)
    h = float(height_px)
    if w <= 0 or h <= 0:
        w = float(work["x"].max()) + 1.0
        h = float(work["y"].max()) + 1.0

    if "flux" not in work.columns:
        work["flux"] = 0.0
    work["_flux_key"] = pd.to_numeric(work["flux"], errors="coerce").fillna(0.0)

    # COMP-POOL-01: n_comp <= 0 means uncapped pool (no grid truncate).
    if int(n_comp) <= 0:
        picked = work.sort_values("_flux_key", ascending=False).drop(
            columns=["_flux_key"], errors="ignore"
        )
        picked.insert(0, "comp_id", [f"COMP_{i+1:04d}" for i in range(len(picked))])
        picked.insert(1, "role", "comparison")
        meta = {
            "n_selected": int(len(picked)),
            "grid_nx": 0,
            "grid_ny": 0,
            "n_comp_requested": 0,
            "pool_uncapped": True,
            "width_px": float(w),
            "height_px": float(h),
            "require_non_variable": bool(require_non_variable),
            "exclude_nonlinear_badcolumn": bool(exclude_nonlinear_badcolumn),
        }
        return picked, meta

    ar = w / h
    nc = max(1, int(n_comp))
    ny = max(1, int(round(math.sqrt(nc / ar))))
    nx = max(1, int(math.ceil(nc / ny)))
    while nx * ny < nc:
        nx += 1

    cw = w / float(nx)
    ch = h / float(ny)

    ix = np.floor(np.clip(work["x"].to_numpy(dtype=float), 0.0, w - 1e-6) / cw).astype(int)
    iy = np.floor(np.clip(work["y"].to_numpy(dtype=float), 0.0, h - 1e-6) / ch).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    work["_cell"] = ix + iy * nx

    # Brightest per cell
    picked = (
        work.sort_values("_flux_key", ascending=False)
        .groupby("_cell", as_index=False, sort=False)
        .head(1)
    )

    if len(picked) > nc:
        picked = picked.nlargest(nc, "_flux_key")

    picked = picked.drop(columns=["_cell", "_flux_key"], errors="ignore")
    picked.insert(0, "comp_id", [f"COMP_{i+1:04d}" for i in range(len(picked))])
    picked.insert(1, "role", "comparison")

    meta = {
        "n_selected": int(len(picked)),
        "grid_nx": int(nx),
        "grid_ny": int(ny),
        "n_comp_requested": int(nc),
        "pool_uncapped": False,
        "width_px": float(w),
        "height_px": float(h),
        "require_non_variable": bool(require_non_variable),
        "exclude_nonlinear_badcolumn": bool(exclude_nonlinear_badcolumn),
    }
    return picked, meta


_MASTERSTAR_ZONE_LOG_ONCE: set[str] = set()

def _masterstar_zone_linear_threshold(
    dao_detection_n_equiv: Any,
) -> float | None:
    """Linear-zone lower bound from DAO detection significance (peak_dao/bg_sigma >= N_equiv)."""
    try:
        t1 = float(dao_detection_n_equiv)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(t1) or t1 <= 0:
        return None
    return t1


def _detect_empirical_clip_level_adu(data: "np.ndarray") -> float | None:
    """Return the ADU level of frame truncation pile-up, or None if the frame is not clipped."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    vmax = float(np.max(finite))
    if not math.isfinite(vmax) or vmax <= 0:
        return None
    at_max = np.abs(finite - vmax) <= 0.5
    if int(np.count_nonzero(at_max)) >= 8:
        return vmax
    for clip_candidate in (65535.0, 32767.0):
        near = np.abs(finite - clip_candidate) <= 0.5
        if int(np.count_nonzero(near)) >= 8:
            return clip_candidate
    return None


def _resolve_peak_saturation_limit_adu(
    *,
    camera_sat_limit_adu: float | None,
    saturate_fraction: float,
    n_stack: int = 1,
    sky_median_adu: float | None = None,
    frame_max_adu: float | None = None,
    empirical_clip_adu: float | None = None,
) -> float | None:
    """Peak-test saturation ceiling: empirical clip first; camera constant only on raw-scale data."""
    ns = max(1, int(n_stack))
    frac = float(saturate_fraction)
    if empirical_clip_adu is not None:
        try:
            ec = float(empirical_clip_adu)
            if math.isfinite(ec) and ec > 0:
                return ec * frac / float(ns)
        except (TypeError, ValueError):
            pass
    if camera_sat_limit_adu is None:
        return None
    try:
        raw_cam = float(camera_sat_limit_adu)
        cam_lim = raw_cam * frac / float(ns)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cam_lim) or cam_lim <= 0:
        return None
    sky = float("nan")
    fmax = float("nan")
    try:
        if sky_median_adu is not None:
            sky = float(sky_median_adu)
        if frame_max_adu is not None:
            fmax = float(frame_max_adu)
    except (TypeError, ValueError):
        pass
    # Float MASTERSTAR stacks can overshoot the 16-bit clip by a few percent
    # (interpolation). Veto only when the image is clearly a different unit scale
    # (SAT-LIMIT-01: 515 stack max 68429 vs clip 65535 is not a unit change).
    if math.isfinite(fmax) and math.isfinite(raw_cam) and fmax > raw_cam * 1.20:
        return None
    if math.isfinite(sky) and sky > 0.20 * cam_lim:
        return None
    return cam_lim


def _masterstar_zone_log_once(key: str, msg: str) -> None:
    if key in _MASTERSTAR_ZONE_LOG_ONCE:
        return
    _MASTERSTAR_ZONE_LOG_ONCE.add(key)
    logging.warning(msg)


def _resolve_masterstar_bg_sigma_adu(
    *,
    sigma_px: Any,
    noise_floor_adu: Any,
    sky_median_adu: Any,
    prematch_peak_sigma_floor: Any,
) -> float | None:
    """Return per-pixel background sigma used in the DAO noise-floor formula."""
    try:
        if sigma_px is not None and str(sigma_px).strip() != "":
            sp = float(sigma_px)
            if math.isfinite(sp) and sp > 0:
                return sp
    except (TypeError, ValueError):
        pass
    try:
        nf = float(noise_floor_adu)
        sky = float(sky_median_adu)
        k = float(prematch_peak_sigma_floor)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(nf) and math.isfinite(sky) and math.isfinite(k) and k > 0):
        return None
    inv = (nf - sky) / k
    if math.isfinite(inv) and inv > 0:
        _masterstar_zone_log_once(
            "sigma_inverted",
            "[MASTERSTAR zone] bg_sigma_adu inferred from noise_floor_adu, sky_median_adu and "
            "prematch_peak_sigma_floor (prefer det_meta['bg_sigma_adu'] when available).",
        )
        return float(inv)
    return None


def _annotate_masterstars_flux_zones(
    df: pd.DataFrame,
    *,
    noise_floor_adu: Any,
    equipment_saturate_adu: float | None,
    saturate_limit_adu_fallback: Any = None,
    n_stack: int | None = None,
    saturate_limit_fraction: float = SAT_LIMIT_NO_KNEE_FRAC,
    sigma_px: Any = None,
    sky_median_adu: Any = None,
    prematch_peak_sigma_floor: Any = None,
    frame_max_adu: Any = None,
    empirical_clip_adu: Any = None,
    dao_detection_n_equiv: Any = None,
) -> pd.DataFrame:
    """Tag MASTERSTAR catalog rows by peak significance (peak_dao/bg_sigma) and saturation.

    Linear/usability boundary T1 = ``dao_detection_n_equiv`` (same N-sigma as DAOFIND threshold).
    Noisy sub-bands step below in whole sigma (``_MASTERSTAR_ZONE_SIGMA_STEP``).

    ``noise_floor_adu`` must match the DAO pre-match SNR filter (``median + kxsigma``) from
    :func:`detect_stars_and_match_catalog` (see ``det_meta["noise_floor_adu"]``).
    """
    import numpy as np

    out = df.copy()
    if "flux" not in out.columns:
        return out
    flux_s = pd.to_numeric(out["flux"], errors="coerce")

    nf: float | None = None
    try:
        if noise_floor_adu is not None and str(noise_floor_adu).strip() != "":
            nf = float(noise_floor_adu)
            if not math.isfinite(nf):
                nf = None
    except (TypeError, ValueError):
        nf = None

    ns = int(n_stack) if n_stack is not None else 1
    ns = max(1, ns)

    # RAW camera/container clip (do not pre-scale; _resolve_peak_saturation_limit_adu applies frac).
    sat_lim_raw = _finite_positive_adu(equipment_saturate_adu)
    peak_frac = float(saturate_limit_fraction)
    unresolved_clip = sat_lim_raw is None
    if unresolved_clip:
        sat_lim_raw = _finite_positive_adu(saturate_limit_adu_fallback)
        unresolved_clip = sat_lim_raw is None
    if unresolved_clip:
        sat_lim_raw = float(SAT_LIMIT_CONTAINER_CLIP_ADU)
        peak_frac = float(SAT_LIMIT_NO_KNEE_FRAC)
        _masterstar_zone_log_once(
            "sat_limit_unresolved",
            "[INV-SAT-LIMIT] header/equipment/sat_diag clip unresolved; "
            f"using clip={SAT_LIMIT_CONTAINER_CLIP_ADU:.0f} ADU (GAIN-DOMAIN-01 container), "
            f"peak-test frac={SAT_LIMIT_NO_KNEE_FRAC:.2f} "
            f"(peak-test={SAT_LIMIT_CONTAINER_CLIP_ADU * SAT_LIMIT_NO_KNEE_FRAC:.1f} ADU). "
            "Never silently admit.",
        )

    out["noise_floor_adu"] = nf if nf is not None else np.nan
    out["saturate_limit_adu"] = float(sat_lim_raw) if sat_lim_raw is not None else np.nan

    if "peak_max_adu" in out.columns:
        peak_col_used = "peak_max_adu"
        peak_s = pd.to_numeric(out["peak_max_adu"], errors="coerce")
    elif "peak_dao" in out.columns:
        peak_col_used = "peak_dao"
        peak_s = pd.to_numeric(out["peak_dao"], errors="coerce")
    else:
        peak_col_used = "flux"
        peak_s = flux_s

    peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_lim_raw,
        saturate_fraction=peak_frac,
        n_stack=ns,
        sky_median_adu=sky_median_adu,
        frame_max_adu=frame_max_adu,
        empirical_clip_adu=empirical_clip_adu,
    )
    if peak_sat_lim is None and unresolved_clip:
        # Scale veto rejected even the conservative default. INV-SAT-LIMIT: still apply peak test.
        peak_sat_lim = float(sat_lim_raw) * float(peak_frac) / float(ns)
        _masterstar_zone_log_once(
            "sat_limit_unresolved_scale_veto",
            "[INV-SAT-LIMIT] peak-test scale veto ignored for unresolved clip; "
            f"applying peak-test={float(peak_sat_lim):.1f} ADU.",
        )
    if peak_sat_lim is None:
        peak_sat_lim = float(sat_lim_raw) if sat_lim_raw is not None else None
        _masterstar_zone_log_once(
            "sat_limit_85pct_nan",
            "[ZONE-SAT-01] saturate_limit_adu_85pct unresolved; peak test uses saturate_limit_adu "
            f"({peak_sat_lim}).",
        )
    out["saturate_limit_adu_85pct"] = peak_sat_lim if peak_sat_lim is not None else np.nan
    out["zone_peak_column"] = peak_col_used
    out["zone_sat_limit_used"] = float(peak_sat_lim) if peak_sat_lim is not None else np.nan

    out["zone"] = ""
    if peak_sat_lim is not None:
        out.loc[peak_s > float(peak_sat_lim), "zone"] = "saturated"

    bg_sigma = _resolve_masterstar_bg_sigma_adu(
        sigma_px=sigma_px,
        noise_floor_adu=noise_floor_adu,
        sky_median_adu=sky_median_adu,
        prematch_peak_sigma_floor=prematch_peak_sigma_floor,
    )
    if bg_sigma is None:
        _masterstar_zone_log_once(
            "sigma_unresolvable",
            "[MASTERSTAR zone] bg_sigma_adu unresolvable - "
            "leaving zone empty (maps to neznama_zona downstream).",
        )
    else:
        t1 = _masterstar_zone_linear_threshold(dao_detection_n_equiv)
        if t1 is None:
            _masterstar_zone_log_once(
                "n_equiv_missing",
                "[MASTERSTAR zone] dao_detection_n_equiv unresolvable - "
                "leaving zone empty (maps to neznama_zona downstream).",
            )
        else:
            if "peak_dao" in out.columns:
                peak_dao_s = pd.to_numeric(out["peak_dao"], errors="coerce")
            else:
                peak_dao_s = pd.Series(np.nan, index=out.index, dtype=float)
            peak_sig = peak_dao_s / float(bg_sigma)
            unsat = out["zone"].eq("")
            sig_ok = unsat & peak_sig.notna()
            sig_miss = unsat & peak_sig.isna()
            out.loc[sig_miss, "zone"] = "unknown"
            out.loc[sig_ok & (peak_sig >= t1), "zone"] = "linear"
            out.loc[sig_ok & (peak_sig < t1), "zone"] = "noise"
            if sig_miss.any():
                _masterstar_zone_log_once(
                    "peak_dao_missing",
                    "[MASTERSTAR zone] peak_dao missing/NaN rows "
                    "marked zone=unknown (not flux fallback).",
                )

    if peak_sat_lim is not None:
        out["is_saturated"] = (peak_s > float(peak_sat_lim)).fillna(False)
    else:
        # Camera clip was supplied but rejected as wrong-scale (precalibrated stack).
        out["is_saturated"] = False

    out["is_noisy"] = out["zone"].eq("noise")
    out["is_usable"] = out["zone"].eq("linear") & flux_s.notna()
    return out


def write_photometry_plan_files(
    *,
    platesolve_dir: Path,
    masterstar_fits: Path,
    masterstars_csv: Path,
    n_comparison_stars: int = 0,
    require_non_variable: bool = True,
    draft_id: int | None = None,
    database_path: Path | str | None = None,
    aligned_files: list[Path] | None = None,
    aligned_ram_frames: list[tuple[str, Any, Any]] | None = None,
    require_safe_bbox: bool = False,
) -> dict[str, Any]:
    """Write ``comparison_stars.csv`` from ``masterstars.csv`` + image size; stub ``variable_targets.csv``."""
    import numpy as np
    import json


    ps = Path(platesolve_dir)
    ps.mkdir(parents=True, exist_ok=True)

    if not masterstars_csv.is_file():
        return {"comparison_stars_csv": "", "variable_targets_csv": "", "error": "missing masterstars.csv"}

    # Read IDs as strings (avoid float64 precision loss for 19-digit Gaia IDs).
    df = pd.read_csv(masterstars_csv, low_memory=False, dtype={"catalog_id": str, "name": str})
    try:
        from gaia_catalog_id import catalog_id_series_for_masterstars_export  # noqa: PLC0415

        if "catalog_id" in df.columns:
            df = df.copy()
            df["catalog_id"] = catalog_id_series_for_masterstars_export(df)
    except Exception:  # noqa: BLE001
        pass

    _enrichment_provenance: dict[str, Any] = {"source": "masterstars_full_match.csv"}
    try:
        from masterstars_enrichment import missing_comp_selection_enrichment_columns  # noqa: PLC0415

        _missing_enrich = missing_comp_selection_enrichment_columns(df)
        if _missing_enrich:
            _msg = (
                "comp-selection enrichment unavailable for this draft: columns "
                f"{_missing_enrich} absent from masterstars_full_match.csv - "
                "selection proceeds without nonlinearity/bad-column exclusion"
            )
            logging.error("[COMP-SELECTION] %s", _msg)
            log_event(f"ERROR: {_msg}")
            _enrichment_provenance = {
                "source": "masterstars_full_match.csv",
                "enrichment_missing_columns": list(_missing_enrich),
                "nonlinearity_badcolumn_exclusion": False,
                "message": _msg,
            }
        else:
            _enrichment_provenance["nonlinearity_badcolumn_exclusion"] = True
    except Exception as _enrich_exc:  # noqa: BLE001
        logging.error("[COMP-SELECTION] enrichment column check failed: %s", _enrich_exc)
    try:
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header
    except Exception as _ms_open_exc:  # noqa: BLE001
        log_event(f"write_photometry_plan_files: nepodarilo sa otvorit MASTERSTAR.fits ({masterstar_fits}): {_ms_open_exc!s}")
        return {"comparison_stars_csv": "", "variable_targets_csv": "", "error": "MASTERSTAR open failed"}
    wpx = int(hdr.get("NAXIS1", 0) or 0)
    h = int(hdr.get("NAXIS2", 0) or 0)
    if wpx <= 0 or h <= 0:
        return {"comparison_stars_csv": "", "error": "MASTERSTAR has no data"}

    _cfg_plan = AppConfig()

    comp_path = ps / "comparison_stars.csv"
    var_path = ps / "variable_targets.csv"

    # --- Annulus-aware intersection bbox (Variant A2) ---
    # Compute a "safe" bbox from the intersection of aligned frames, shrunk by the sky-annulus outer radius.
    # CONSOLIDATE-01B A2: FWHM source stays MASTERSTAR VY_FWHM (qc vs header differ on all
    # 134 era04 frames at FITS-card rounding; night-median qc vs MASTERSTAR is also not
    # last-ulp identical). Do not invent 3.5 / 10.5.
    _safe_bbox: tuple[float, float, float, float] | None = None
    _r_out: float | None = None
    try:
        from aperture_policy import fwhm_from_header_vy_fwhm, resolve_aperture_geometry  # noqa: PLC0415

        _fwhm_px = fwhm_from_header_vy_fwhm(hdr)
        if _fwhm_px is None:
            raise ValueError("MASTERSTAR VY_FWHM missing; cannot resolve annulus r_out")
        _r_ap_b, _r_in_b, _r_out = resolve_aperture_geometry(
            f=float(_cfg_plan.aperture_fwhm_factor),
            fwhm_px=float(_fwhm_px),
            annulus_inner_fwhm=float(_cfg_plan.annulus_inner_fwhm),
            annulus_outer_fwhm=float(_cfg_plan.annulus_outer_fwhm),
        )
        _ = (_r_ap_b, _r_in_b)

        draft_root = ps.parent.parent
        aligned_dir = draft_root / "detrended_aligned" / "lights" / ps.name
        if aligned_files is not None:
            all_aligned = [Path(p) for p in aligned_files if Path(p).exists()]
            LOGGER.info(f"[BORDER] Using {len(all_aligned)} pre-supplied aligned frame paths")
        else:
            all_aligned = sorted(aligned_dir.glob("proc_*.fits"))
            LOGGER.info(f"[BORDER] Glob found {len(all_aligned)} aligned frames in {aligned_dir}")

        ram_arrays: list[Any] = []
        if aligned_ram_frames:
            for _name, _hdr, _arr in aligned_ram_frames:
                if _arr is not None:
                    ram_arrays.append(_arr)
            if ram_arrays:
                LOGGER.info(f"[BORDER] Using {len(ram_arrays)} aligned RAM frames for intersection bbox")

        n_frame_sources = len(all_aligned) + (len(ram_arrays) if not all_aligned else 0)
        if n_frame_sources == 0:
            if require_safe_bbox:
                raise RuntimeError(
                    "[BORDER] Post-alignment border filter requires aligned frames but none "
                    "were found on disk or in RAM handoff"
                )
            log_event(
                "[BORDER] Deferred: no aligned proc_*.fits on disk yet "
                "(pre-alignment or RAM-handoff); border filter skipped"
            )
        elif n_frame_sources < 2:
            if require_safe_bbox:
                raise RuntimeError(
                    f"[BORDER] Post-alignment border filter requires >=2 aligned frames, got {n_frame_sources}"
                )
            log_event(f"[BORDER] Not enough aligned frames for intersection bbox: {n_frame_sources}")
        else:
            frames_for_bbox = all_aligned if all_aligned else []
            ram_for_bbox = [] if all_aligned else ram_arrays
            try:
                if draft_id is not None and database_path:
                    dbp2 = Path(str(database_path))
                    if dbp2.is_file():
                        jr = detect_field_jumps(
                            db=VyvarDatabase(dbp2), draft_id=int(draft_id)
                        )
                        if bool(jr.get("has_jump")) and int(jr.get("n_groups") or 0) > 0:
                            dom = None
                            for g in (jr.get("groups") or []):
                                if bool(g.get("is_dominant")):
                                    dom = g
                                    break
                            if dom is not None:
                                fs = int(dom.get("frame_start") or 0)
                                fe = int(dom.get("frame_end") or 0)
                                if all_aligned:
                                    frames_for_bbox = all_aligned[fs : fe + 1]
                                elif aligned_ram_frames:
                                    ram_for_bbox = [
                                        arr for _n, _h, arr in aligned_ram_frames[fs : fe + 1] if arr is not None
                                    ]
                                log_event(
                                    f"[BORDER] Field jump detected - using dominant group frames {fs}-{fe} "
                                    f"({len(frames_for_bbox) or len(ram_for_bbox)} frames) for intersection bbox"
                                )
                            else:
                                log_event(
                                    f"[BORDER] Field jump detected - no dominant group found; using all "
                                    f"{len(frames_for_bbox) or len(ram_for_bbox)} frames"
                                )
                        else:
                            log_event(
                                f"[BORDER] Field stable - using all "
                                f"{len(frames_for_bbox) or len(ram_for_bbox)} aligned frames for intersection bbox"
                            )
            except Exception as _jr_exc:  # noqa: BLE001
                log_event(f"[BORDER] detect_field_jumps failed: {_jr_exc!s} - using all aligned frames")

            from photometry_core import (  # noqa: PLC0415
                common_field_intersection_bbox_px,
                common_field_intersection_bbox_px_from_arrays,
            )

            raw_bbox = None
            if len(frames_for_bbox) >= 2:
                raw_bbox = common_field_intersection_bbox_px(frame_paths=frames_for_bbox, finite_stride=16)
            elif len(ram_for_bbox) >= 2:
                raw_bbox = common_field_intersection_bbox_px_from_arrays(
                    frame_arrays=ram_for_bbox, finite_stride=16
                )

            if raw_bbox is None:
                if require_safe_bbox:
                    raise RuntimeError(
                        "[BORDER] Post-alignment border filter failed: intersection bbox returned None "
                        f"(disk={len(frames_for_bbox)}, ram={len(ram_for_bbox)})"
                    )
                log_event("[BORDER] intersection bbox returned None - no border filter applied")
            else:
                x0, y0, x1, y1 = raw_bbox
                sb = (float(x0) + _r_out, float(y0) + _r_out, float(x1) - _r_out, float(y1) - _r_out)
                if sb[2] > sb[0] and sb[3] > sb[1]:
                    _safe_bbox = sb
                    log_event(
                        f"[BORDER] Safe bbox (shrunk by r_out={_r_out:.1f}px): "
                        f"x=[{_safe_bbox[0]:.0f},{_safe_bbox[2]:.0f}] y=[{_safe_bbox[1]:.0f},{_safe_bbox[3]:.0f}]"
                    )
                elif require_safe_bbox:
                    raise RuntimeError(
                        f"[BORDER] Post-alignment safe bbox invalid after shrink (r_out={_r_out:.1f}px)"
                    )
                else:
                    log_event(
                        f"[BORDER] Safe bbox invalid after shrink (r_out={_r_out:.1f}px); skipping border filter"
                    )
                    _safe_bbox = None
    except RuntimeError:
        raise
    except Exception as _bbox_exc:  # noqa: BLE001
        if require_safe_bbox:
            raise RuntimeError(f"[BORDER] safe_bbox computation failed: {_bbox_exc!s}") from _bbox_exc
        log_event(f"[BORDER] safe_bbox computation failed: {_bbox_exc!s} - skipping border filter")
        _safe_bbox = None

    # VSX variable targets for the field (frame bbox), with pixel coords from MASTERSTAR WCS.
    var_cols = [
        "name",
        "catalog_id",
        "catalog",
        "ra_deg",
        "dec_deg",
        "priority",
        "notes",
        "vsx_name",
        "vsx_type",
        "vsx_period",
        "x",
        "y",
        "mag",
        "zone",
        "gaia_match_arcsec",
        "gaia_match_quality",
        "gaia_match_source",
        "vsx_mag_max",
        "exo_host_obj_id",
        "exo_host_name",
        "exo_cat_source",
        "exo_disposition",
        "exo_match_sep_arcsec",
        "target_origin",
    ]
    vsx_out = pd.DataFrame(columns=var_cols)
    _vsx_n_cone = 0
    _vsx_diag: dict[str, Any] = {}
    try:
        from database import VSXCatalogError, require_vsx_local_db_path

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w0 = WCS(hdr)
        if not bool(getattr(w0, "has_celestial", False)):
            raise RuntimeError("MASTERSTAR WCS has no celestial axes.")
        if w0.has_celestial:
            center, radius_deg = _effective_field_catalog_cone_radius_deg(
                w0, int(h), int(wpx), plate_solve_fov_deg=None, fits_header=hdr
            )
            _vsx_p3 = require_vsx_local_db_path(_cfg_plan.vsx_local_db_path)
            # B-cap fix: variable_targets are driven by the FRAME footprint (bbox + the same
            # 50px margin used by the in-frame pixel filter below), not the 3.5 deg cone box. The
            # cone box hit the 15000-row catalog_query_max_rows cap (no ORDER BY) and silently
            # dropped a contiguous Dec slice (northern half of the field, incl. bright named
            # variables). The frame bbox is tiny -> no cap -> spatial-first completeness. Note this
            # list also drives the global comparison-pool veto, so the now-complete variable set
            # correctly purges newly-recognised variables from the comp ensemble (Milan-approved,
            # CURSOR_RESULT_round1).
            vsx_df = _query_vsx_local_frame_bbox(
                wcs=w0,
                width_px=int(wpx),
                height_px=int(h),
                vsx_db_path=_vsx_p3,
                margin_px=50.0,
                center=center,
                require_db=True,
            )
            n_vsx_in_cone = int(len(vsx_df)) if vsx_df is not None else 0
            _vsx_n_cone = n_vsx_in_cone

            # Masterstars catalogue (Phase 0 DAO membership only; not the VSX->Gaia match RHS).
            ga = df.copy()
            if "catalog_id" in ga.columns and "ra_deg" in ga.columns and "dec_deg" in ga.columns:
                ga["catalog_id"] = ga["catalog_id"].fillna("").astype(str).str.strip()
                ga = ga[ga["catalog_id"].ne("")].copy()
                ga["ra_deg"] = pd.to_numeric(ga["ra_deg"], errors="coerce")
                ga["dec_deg"] = pd.to_numeric(ga["dec_deg"], errors="coerce")
                ga = ga[ga["ra_deg"].notna() & ga["dec_deg"].notna()].copy()
            else:
                ga = ga.iloc[0:0].copy()
            from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

            _ms_ids: set[str] = set()
            if not ga.empty:
                _ms_ids = {
                    str(normalize_gaia_source_id(x)).strip()
                    for x in ga["catalog_id"].astype(str).tolist()
                    if str(x).strip()
                }

            if vsx_df is not None and not vsx_df.empty and "ra_deg" in vsx_df.columns and "dec_deg" in vsx_df.columns:
                v = vsx_df.copy()
                v["ra_deg"] = pd.to_numeric(v["ra_deg"], errors="coerce")
                v["dec_deg"] = pd.to_numeric(v["dec_deg"], errors="coerce")
                v = v[v["ra_deg"].notna() & v["dec_deg"].notna()].copy()
                if not v.empty:
                    # Pixel coords from WCS (origin=0).
                    xy = w0.all_world2pix(v["ra_deg"].astype(float).to_numpy(), v["dec_deg"].astype(float).to_numpy(), 0)
                    x = np.asarray(xy[0], dtype=float)
                    y = np.asarray(xy[1], dtype=float)
                    v["x"] = x
                    v["y"] = y
                    in_frame = (v["x"] >= -50.0) & (v["y"] >= -50.0) & (v["x"] <= float(wpx) + 50.0) & (v["y"] <= float(h) + 50.0)
                    v = v.loc[in_frame].copy()

                n_vsx_in_frame = int(len(v))

                # Density-aware VSX -> Gaia DR3 cross-match (Marrese 2017/2019; Sutherland & Saunders 1992).
                # RHS is the deep local Gaia catalogue over the frame bbox; masterstars membership is derived after.
                cat_id_out = [""] * int(len(v))
                mag_out: list[float] = [float("nan")] * int(len(v))
                zone_out: list[str] = [""] * int(len(v))
                sep_out: list[float] = [float("nan")] * int(len(v))
                quality_out: list[str] = [""] * int(len(v))
                source_out: list[str] = [""] * int(len(v))
                _cm_diag: dict[str, Any] = {}
                if len(v) > 0:
                    from database import get_gaia_db_max_g_mag  # noqa: PLC0415
                    from vsx_gaia_crossmatch import (  # noqa: PLC0415
                        VsxGaiaCrossmatchError,
                        field_area_deg2_from_wcs,
                        match_vsx_to_gaia_density_aware,
                        query_gaia_for_frame_bbox,
                    )

                    try:
                        gaia_db = str(_cfg_plan.gaia_db_path or "").strip()
                    except Exception:  # noqa: BLE001
                        gaia_db = ""
                    if not gaia_db:
                        raise RuntimeError("VSX->Gaia cross-match refused: gaia_db_path not configured")

                    _field_area = field_area_deg2_from_wcs(w0, int(wpx), int(h))
                    _gaia_db_max_g = float(get_gaia_db_max_g_mag(gaia_db))
                    _gaia_rows = query_gaia_for_frame_bbox(
                        gaia_db,
                        w0,
                        int(wpx),
                        int(h),
                        margin_px=50.0,
                        center=center,
                    )
                    _gdf = pd.DataFrame(_gaia_rows) if _gaia_rows else pd.DataFrame()
                    if _gdf.empty:
                        raise RuntimeError(
                            "VSX->Gaia cross-match refused: no Gaia DR3 sources in frame bbox"
                        )
                    _sid_col = "source_id" if "source_id" in _gdf.columns else "catalog_id"
                    _ra_col = "ra" if "ra" in _gdf.columns else "ra_deg"
                    _dec_col = "dec" if "dec" in _gdf.columns else "dec_deg"
                    _gdf = _gdf.copy()
                    _gdf["catalog_id"] = _gdf[_sid_col].apply(
                        lambda x: str(normalize_gaia_source_id(x)).strip()
                    )
                    _gdf["ra_deg"] = pd.to_numeric(_gdf[_ra_col], errors="coerce")
                    _gdf["dec_deg"] = pd.to_numeric(_gdf[_dec_col], errors="coerce")
                    _gdf = _gdf[
                        _gdf["catalog_id"].ne("")
                        & _gdf["ra_deg"].notna()
                        & _gdf["dec_deg"].notna()
                    ].copy()
                    _pmra = (
                        pd.to_numeric(_gdf["pmra"], errors="coerce").to_numpy(dtype=float)
                        if "pmra" in _gdf.columns
                        else None
                    )
                    _pmdec = (
                        pd.to_numeric(_gdf["pmdec"], errors="coerce").to_numpy(dtype=float)
                        if "pmdec" in _gdf.columns
                        else None
                    )
                    _vsx_mag_max = (
                        pd.to_numeric(v.get("mag_max"), errors="coerce").to_numpy(dtype=float)
                        if "mag_max" in v.columns
                        else None
                    )
                    try:
                        _match_rows, _cm_diag_obj = match_vsx_to_gaia_density_aware(
                            v["ra_deg"].astype(float).to_numpy(),
                            v["dec_deg"].astype(float).to_numpy(),
                            _gdf["catalog_id"].astype(str).to_numpy(),
                            _gdf["ra_deg"].astype(float).to_numpy(),
                            _gdf["dec_deg"].astype(float).to_numpy(),
                            field_area_deg2=float(_field_area),
                            pmra=_pmra,
                            pmdec=_pmdec,
                            gaia_db_max_g=_gaia_db_max_g,
                            vsx_mag_max=_vsx_mag_max,
                            masterstars_ids=_ms_ids,
                        )
                        _cm_diag = {
                            "q_fit": _cm_diag_obj.q_fit,
                            "w_fit": _cm_diag_obj.w_fit,
                            "sigma_narrow_arcsec": _cm_diag_obj.sigma_narrow_arcsec,
                            "sigma_broad_arcsec": _cm_diag_obj.sigma_broad_arcsec,
                            "sigma_arcsec": _cm_diag_obj.sigma_narrow_arcsec,
                            "rho_per_deg2": _cm_diag_obj.rho_per_deg2,
                            "mean_nn_arcsec": _cm_diag_obj.mean_nn_arcsec,
                            "pm_path": _cm_diag_obj.pm_path,
                            "pm_columns_present": _cm_diag_obj.pm_columns_present,
                            "n_pm_finite": _cm_diag_obj.n_pm_finite,
                            "n_vsx": _cm_diag_obj.n_vsx,
                            "n_gaia": _cm_diag_obj.n_gaia,
                            "n_accepted": _cm_diag_obj.n_accepted,
                            "expected_contamination_fraction": _cm_diag_obj.expected_contamination_fraction,
                            "r_max_arcsec": _cm_diag_obj.r_max_arcsec,
                            "candidate_multiplicity": dict(_cm_diag_obj.candidate_multiplicity),
                            "multi_candidate_fraction": _cm_diag_obj.multi_candidate_fraction,
                            "fit_degenerate_warn": _cm_diag_obj.fit_degenerate_warn,
                            "gaia_db_max_g": _cm_diag_obj.gaia_db_max_g,
                            "vsx_fainter_than_gaia_db": _cm_diag_obj.vsx_fainter_than_gaia_db,
                            "sep_quantiles_before_pm": _cm_diag_obj.sep_quantiles_before_pm,
                            "sep_quantiles_after_pm": _cm_diag_obj.sep_quantiles_after_pm,
                            "sep_quantiles_accepted": _cm_diag_obj.sep_quantiles_accepted,
                            "masterstars_in_frame": _cm_diag_obj.masterstars_in_frame,
                            "masterstars_eligible": _cm_diag_obj.masterstars_eligible,
                            "masterstars_accepted": _cm_diag_obj.masterstars_accepted,
                            "outcome_check": _cm_diag_obj.outcome_check,
                        }
                        _gaia_by_cid = {
                            str(r["catalog_id"]): r for _, r in _gdf.iterrows()
                        }
                        for i, mr in enumerate(_match_rows):
                            if mr.accepted and mr.catalog_id:
                                cat_id_out[i] = mr.catalog_id
                                sep_out[i] = float(mr.sep_arcsec)
                                quality_out[i] = mr.quality
                                if mr.catalog_id in _ms_ids:
                                    source_out[i] = "masterstars"
                                    gro = ga.loc[ga["catalog_id"].astype(str) == mr.catalog_id]
                                    if not gro.empty:
                                        gro = gro.iloc[0]
                                        try:
                                            _mg = gro.get("mag")
                                            mag_out[i] = (
                                                float(_mg)
                                                if _mg is not None and str(_mg).strip() != ""
                                                else float("nan")
                                            )
                                        except (TypeError, ValueError):
                                            mag_out[i] = float("nan")
                                        try:
                                            zr = gro.get("zone")
                                            zone_out[i] = str(zr).strip().lower() if zr is not None else ""
                                        except Exception:  # noqa: BLE001
                                            zone_out[i] = ""
                                else:
                                    source_out[i] = "gaia_dr3_direct"
                                    grow = _gaia_by_cid.get(mr.catalog_id)
                                    if grow is not None:
                                        try:
                                            _gm = grow.get("g_mag")
                                            mag_out[i] = (
                                                float(_gm)
                                                if _gm is not None and str(_gm).strip() != ""
                                                else float("nan")
                                            )
                                        except (TypeError, ValueError):
                                            mag_out[i] = float("nan")
                            else:
                                source_out[i] = "no_match"
                                try:
                                    vsx_name0 = str(v.iloc[i].get("name", "") or "").strip()
                                    vsx_ra = float(v.iloc[i]["ra_deg"])
                                    vsx_dec = float(v.iloc[i]["dec_deg"])
                                    log_event(
                                        f"VSX no Gaia match: {vsx_name0} ra={vsx_ra:.4f} dec={vsx_dec:.4f} "
                                        "- hviezda nebude sledovana"
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                    except VsxGaiaCrossmatchError as _cm_exc:
                        log_event(f"VSX->Gaia DR3 cross-match refused: {_cm_exc!s}")
                        raise RuntimeError(str(_cm_exc)) from _cm_exc

                # Period column varies by VSX schema.
                _pcol = None
                for c in ("period", "varperiod", "var_period", "Period", "VarPeriod", "Var_Period"):
                    if c in v.columns:
                        _pcol = c
                        break
                if _pcol is None:
                    v["vsx_period"] = np.nan
                else:
                    v["vsx_period"] = pd.to_numeric(v[_pcol], errors="coerce")

                vname = v.get("name", pd.Series([""] * len(v))).fillna("").astype(str).str.strip()
                vtype = v.get("var_type", pd.Series([""] * len(v))).fillna("").astype(str).str.strip()
                notes = []
                for t, p in zip(vtype.tolist(), v["vsx_period"].tolist(), strict=False):
                    t0 = str(t or "").strip()
                    p0 = float(p) if p is not None and pd.notna(p) else None
                    if p0 is not None:
                        notes.append(f"{t0} P={p0}")
                    else:
                        notes.append(t0)

                _mmx = pd.to_numeric(v.get("mag_max"), errors="coerce") if "mag_max" in v.columns else pd.Series(
                    [float("nan")] * len(v), dtype=float
                )
                vsx_out = pd.DataFrame(
                    {
                        "name": vname,
                        "catalog_id": cat_id_out,
                        "catalog": "VSX",
                        "ra_deg": v["ra_deg"].astype(float).to_numpy(),
                        "dec_deg": v["dec_deg"].astype(float).to_numpy(),
                        "priority": 1,
                        "notes": notes,
                        "vsx_name": vname,
                        "vsx_type": vtype,
                        "vsx_period": v["vsx_period"].to_numpy(),
                        "x": pd.to_numeric(v.get("x"), errors="coerce"),
                        "y": pd.to_numeric(v.get("y"), errors="coerce"),
                        "mag": np.asarray(mag_out, dtype=float),
                        "zone": zone_out,
                        "gaia_match_arcsec": np.asarray(sep_out, dtype=float),
                        "gaia_match_quality": quality_out,
                        "gaia_match_source": source_out,
                        "vsx_mag_max": _mmx.to_numpy(dtype=float),
                    }
                )
                n_gaia_ok = int(sum(1 for c in cat_id_out if str(c).strip()))
                _zone_hist: dict[str, int] = {}
                n_phase0_hint = int(n_gaia_ok)
                for i in range(len(zone_out)):
                    if not str(cat_id_out[i]).strip():
                        continue
                    z = str(zone_out[i]).strip().lower()
                    _zone_hist[z] = int(_zone_hist.get(z, 0)) + 1
                _vsx_diag = {
                    "vsx_rows_in_frame_bbox": int(n_vsx_in_cone),
                    "vsx_rows_after_in_frame_margin": int(n_vsx_in_frame),
                    "n_with_masterstar_match": int(n_gaia_ok),
                    "vsx_rows_written_csv": int(len(vsx_out)),
                    "gaia_matches_within_10arcsec": int(n_gaia_ok),
                    "masterstars_zone_counts_among_gaia_matched": _zone_hist,
                    "phase0_active_targets_hint_count": int(n_phase0_hint),
                    "vsx_gaia_crossmatch": dict(_cm_diag),
                    "phase0_note_sk": (
                        "Faza 0 (`select_active_targets` v photometry_core.py): vsetky zony z masterstars "
                        "prejdu do active_targets.csv s `zone_flag`; saturovane maju `skip_photometry=True` "
                        "a Faza 2A ich nefotometruje (pozri `run_phase2a`). Vylucene su len prazdny "
                        "`catalog_id` a ciele mimo snimky (okrajovy filter)."
                    ),
                }
                _vsx_n_cone = int(n_vsx_in_cone)
    except VSXCatalogError:
        raise
    except Exception as _vsx_exc:  # noqa: BLE001
        log_event(f"variable_targets.csv (VSX export) preskoceny: {_vsx_exc!s}")
        vsx_out = pd.DataFrame(columns=var_cols)

    exo_promo = _build_exoplanet_promotion_rows_from_masterstars(
        df,
        hdr,
        _cfg_plan,
        frame_w_px=int(wpx),
        frame_h_px=int(h),
        margin_px=50.0,
    )
    merged_var = _merge_vsx_exoplanet_variable_targets(vsx_out, exo_promo)
    _proximity_veto_df: pd.DataFrame | None = None
    if merged_var is not None and not merged_var.empty:
        _ra_v = pd.to_numeric(merged_var.get("ra_deg"), errors="coerce")
        _de_v = pd.to_numeric(merged_var.get("dec_deg"), errors="coerce")
        _proximity_veto_df = merged_var.loc[_ra_v.notna() & _de_v.notna()].copy()

    comp_df, cmeta = select_comparison_stars_spatial_grid(
        df,
        width_px=float(wpx),
        height_px=float(h),
        n_comp=int(n_comparison_stars),
        require_non_variable=bool(require_non_variable),
        variable_targets_df=_proximity_veto_df,
        safe_bbox=_safe_bbox,
    )
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in comp_df.columns:
            comp_df = comp_df.copy()
            comp_df["catalog_id"] = normalize_gaia_source_id_series(comp_df["catalog_id"])
    except Exception:  # noqa: BLE001
        pass
    _vyvar_df_to_csv(comp_df, comp_path)

    # Always overwrite (even if it exists) so UI sees current field cone + exo promotions.
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in merged_var.columns:
            merged_var = merged_var.copy()
            merged_var["catalog_id"] = normalize_gaia_source_id_series(merged_var["catalog_id"])
    except Exception:  # noqa: BLE001
        pass
    _vyvar_df_to_csv(merged_var, var_path)
    if _vsx_diag:
        log_event(
            "variable_targets.csv (VSX): "
            f"VSX in frame bbox={_vsx_diag.get('vsx_rows_in_frame_bbox')} -> "
            f"in-frame margin={_vsx_diag.get('vsx_rows_after_in_frame_margin')} -> "
            f"detection-limited (DAO+Gaia match)={_vsx_diag.get('n_with_masterstar_match')} -> "
            f"CSV={int(len(merged_var))}. "
            f"Odhad VSX s Gaia ID (Faza 0 potom cross-match na masterstars): {_vsx_diag.get('phase0_active_targets_hint_count')}."
        )
    else:
        log_event(
            f"variable_targets.csv (VSX export): cone={_vsx_n_cone} -> zapisane={int(len(merged_var))} "
            f"({var_path.name})"
        )

    plan_path = ps / "photometry_plan.json"
    plan = {
        "purpose": "VYVAR photometry: comparison ensemble + variable targets",
        "comparison_stars": str(comp_path),
        "variable_targets_template": str(var_path),
        "masterstars": str(masterstars_csv),
        "masterstar_fits": str(masterstar_fits),
        "comparison_selection": cmeta,
        "comp_selection_enrichment": _enrichment_provenance,
        "variable_targets_diagnostics": _vsx_diag,
        "safe_bbox_px": list(_safe_bbox) if _safe_bbox is not None else None,
        "safe_bbox_r_out_px": float(_r_out) if _r_out is not None else None,
        "next_steps": [
            "Fill variable_targets.csv with programme stars (catalog IDs / coordinates).",
            "Use comparison_stars.csv comp_id / catalog_id to tie ensemble photometry on detrended_aligned frames.",
            "Run validate_comparison_ensemble_flatness on detrended_aligned frames to check comp stars stay flat vs ensemble.",
            "Light curves: aperture photometry per frame vs time (JD), then search for new variables vs field behavior.",
        ],
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    return {
        "comparison_stars_csv": str(comp_path),
        "variable_targets_csv": str(var_path),
        "photometry_plan_json": str(plan_path),
        "comparison_selection": cmeta,
        "comp_selection_enrichment": _enrichment_provenance,
    }


def _sync_comparison_stars_across_setups(platesolve_root: Path) -> None:
    """Copy comparison_stars.csv from a reference setup to all other setup dirs under platesolve/.

    Reference preference:
    - first setup containing 'R_' in its name with comparison_stars.csv
    - else first setup (sorted) that has comparison_stars.csv
    """
    try:
        root = Path(platesolve_root)
        if not root.is_dir():
            return
        setup_dirs = [
            d
            for d in root.iterdir()
            if d.is_dir() and (d / "per_frame_catalog_index.csv").exists()
        ]
        if len(setup_dirs) < 2:
            return
        ref_dir: Path | None = None
        for d in setup_dirs:
            if "R_" in d.name and (d / "comparison_stars.csv").exists():
                ref_dir = d
                break
        if ref_dir is None:
            for d in sorted(setup_dirs, key=lambda p: p.name.casefold()):
                if (d / "comparison_stars.csv").exists():
                    ref_dir = d
                    break
        if ref_dir is None:
            return
        ref_comp = ref_dir / "comparison_stars.csv"
        if not ref_comp.is_file():
            return
        import shutil

        for d in setup_dirs:
            if d == ref_dir:
                continue
            target_comp = d / "comparison_stars.csv"
            try:
                shutil.copy2(ref_comp, target_comp)
                log_event(
                    f"INFO: Comparison stars skopirovane z {ref_dir.name} -> {d.name}"
                )
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PIPE] comparison-star sync across setups skipped: %s", exc)
        return


def _dao_auto_binning_factor(h: int, w: int) -> int:
    """2x2 mean binning for DAO on large chips (~4x fewer pixels); skipped below ~5 MP."""
    mp = float(int(h) * int(w)) / 1_000_000.0
    if mp < 5.0:
        return 1
    return 2


def _pixel_noise_sigma_pp_adu(arr: Any) -> float:
    """Gradient-immune pixel noise from adjacent differences (MAD estimator).

    For smooth large-scale structure, ``I[i+1]-I[i]`` is nearly constant so MAD tracks
    pixel-to-pixel noise, not sky gradient. Validated on BO CVn MASTERSTAR (~46 ADU stable
    across draft 435/450 while ``plain_mean_med_std`` moved 26%; see SIGMA-ESTIMATOR-VERIFY).
    """
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.size < 64:
        return float("nan")
    finite = np.isfinite(a)
    chunks: list[np.ndarray] = []
    dh = a[:, 1:] - a[:, :-1]
    mh = finite[:, 1:] & finite[:, :-1]
    if int(np.count_nonzero(mh)) >= 32:
        chunks.append(dh[mh].ravel())
    dv = a[1:, :] - a[:-1, :]
    mv = finite[1:, :] & finite[:-1, :]
    if int(np.count_nonzero(mv)) >= 32:
        chunks.append(dv[mv].ravel())
    if not chunks:
        return float("nan")
    diffs = np.concatenate(chunks)
    if diffs.size < 64:
        return float("nan")
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    if not math.isfinite(mad) or mad <= 0:
        return float("nan")
    return float(mad / 0.674489750196082 / math.sqrt(2.0))


def _dao_noise_sigma_adu(
    arr: Any,
    *,
    bfac: int,
    fallback_std: float,
    data_dao: Any | None = None,
) -> float:
    """Legacy per-pixel noise scale (diagnostic only; not used for DAO threshold under option B)."""

    sigma_pp = _pixel_noise_sigma_pp_adu(arr)
    if math.isfinite(float(sigma_pp)) and float(sigma_pp) > 0:
        return float(sigma_pp) / math.sqrt(float(max(1, bfac)))
    if int(bfac) > 1 and data_dao is not None:
        _, _, std_dao = plain_mean_med_std(data_dao, sigma=3.0, maxiters=3)
        logging.warning("[DAO] sigma_pp unavailable; falling back to plain_mean_med_std on binned frame")
        return float(std_dao)
    logging.warning("[DAO] sigma_pp unavailable; falling back to global plain_mean_med_std")
    return float(fallback_std)


def _dao_convolved_background_rms_adu(
    data_dao: Any,
    *,
    fwhm_px: float,
    sigma_radius: float = 1.5,
) -> tuple[float, float]:
    """Robust RMS of the DAO convolved detection image (option B threshold basis).

    Builds the same zero-sum kernel DAOStarFinder uses, convolves ``data_dao``, and
    returns (rms_convolved, kernel.rel_err). Caller sets ``scale_threshold=False`` and
    ``threshold = N * rms_convolved`` so nominal N-sigma holds on correlated/resampled data.
    """
    import numpy as np
    from photutils.detection.core import _StarFinderKernel
    from scipy.ndimage import convolve

    arr = np.asarray(data_dao, dtype=np.float32)
    if arr.ndim != 2 or arr.size < 64:
        return float("nan"), float("nan")
    fwhm = max(1.2, float(fwhm_px))
    kernel = _StarFinderKernel(fwhm=fwhm, sigma_radius=float(sigma_radius))
    conv = convolve(arr, kernel.data, mode="nearest")
    _, _, rms = plain_mean_med_std(conv, sigma=3.0, maxiters=3)
    return float(rms), float(kernel.rel_err)


_BATCH_E_N_EQUIV_LOGGED = False


def _dao_detection_threshold_adu(
    rms_conv: float,
    *,
    cfg: Any | None,
    dao_threshold_sigma: float,
) -> tuple[float, float]:
    """T4-1 Option B: threshold = N_equiv * rms_conv (measured N from Part 2b)."""
    global _BATCH_E_N_EQUIV_LOGGED  # noqa: PLW0603
    rms = float(rms_conv)
    if not math.isfinite(rms) or rms <= 0:
        return 1e-6, float("nan")
    n_eff = float(getattr(cfg, "dao_detection_n_equiv", dao_threshold_sigma)) if cfg is not None else float(
        dao_threshold_sigma
    )
    if not math.isfinite(n_eff) or n_eff <= 0:
        n_eff = float(dao_threshold_sigma)
    if cfg is not None and not _BATCH_E_N_EQUIV_LOGGED:
        LOGGER.info("[BATCH-E E.4] N_equiv=%.2f applied for DAO detection threshold", n_eff)
        _BATCH_E_N_EQUIV_LOGGED = True
    return max(n_eff * rms, 1e-6), n_eff


def _apply_dao_centroid_wcs_guard(
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    matched: "np.ndarray",
    safe: "np.ndarray",
    master_df: pd.DataFrame,
    fwhm_px: float,
    max_shift_fwhm: float,
) -> tuple["np.ndarray", "np.ndarray", int]:
    """Replace DAO centroids with MASTERSTAR reference pixels when shift exceeds guard."""
    import numpy as np

    xo = np.asarray(x, dtype=np.float64).copy()
    yo = np.asarray(y, dtype=np.float64).copy()
    if master_df is None or master_df.empty or "x" not in master_df.columns or "y" not in master_df.columns:
        return xo, yo, 0
    mx = pd.to_numeric(master_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    my = pd.to_numeric(master_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.asarray(matched, dtype=bool)
    if not m.any():
        return xo, yo, 0
    s = np.clip(np.asarray(safe, dtype=np.int64), 0, max(len(mx) - 1, 0))
    x_ref = mx[s]
    y_ref = my[s]
    ok = m & np.isfinite(x_ref) & np.isfinite(y_ref) & np.isfinite(xo) & np.isfinite(yo)
    if not ok.any():
        return xo, yo, 0
    max_px = float(max(0.1, max_shift_fwhm)) * float(max(1.2, fwhm_px))
    shift = np.hypot(xo - x_ref, yo - y_ref)
    use_wcs = ok & (shift > max_px)
    xo[use_wcs] = x_ref[use_wcs]
    yo[use_wcs] = y_ref[use_wcs]
    return xo, yo, int(np.count_nonzero(use_wcs))


def _lock_matched_centroids_to_master_grid(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    matched: "np.ndarray",
    safe: "np.ndarray",
    master_df: pd.DataFrame,
    fwhm_px: float,
    search_fwhm: float = 2.5,
) -> tuple["np.ndarray", "np.ndarray", int]:
    """Lock matched catalog stars to MASTERSTAR grid with local peak refinement.

    Master-reference per-frame catalogs must measure on the shared alignment grid,
    not on arbitrary DAO detections that can land on faint neighbours after transform
    smearing.  Each matched row is snapped to the master (x, y) then refined within
    a small search window for the brightest pixel (sub-pixel centre not required).
    """
    import numpy as np

    xo = np.asarray(x, dtype=np.float64).copy()
    yo = np.asarray(y, dtype=np.float64).copy()
    if master_df is None or master_df.empty or "x" not in master_df.columns or "y" not in master_df.columns:
        return xo, yo, 0
    img = np.asarray(arr, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        return xo, yo, 0
    h_img, w_img = int(img.shape[0]), int(img.shape[1])
    mx = pd.to_numeric(master_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    my = pd.to_numeric(master_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.asarray(matched, dtype=bool)
    if not m.any():
        return xo, yo, 0
    s = np.clip(np.asarray(safe, dtype=np.int64), 0, max(len(mx) - 1, 0))
    radius = int(max(3, math.ceil(float(max(1.2, fwhm_px)) * float(max(1.0, search_fwhm)))))
    n_locked = 0
    for i in np.nonzero(m)[0]:
        si = int(s[i])
        if si < 0 or si >= len(mx):
            continue
        x_ref = float(mx[si])
        y_ref = float(my[si])
        if not (math.isfinite(x_ref) and math.isfinite(y_ref)):
            continue
        xi = int(round(x_ref))
        yi = int(round(y_ref))
        x_lo = max(0, xi - radius)
        x_hi = min(w_img, xi + radius + 1)
        y_lo = max(0, yi - radius)
        y_hi = min(h_img, yi + radius + 1)
        if x_lo >= x_hi or y_lo >= y_hi:
            xo[i] = x_ref
            yo[i] = y_ref
            n_locked += 1
            continue
        patch = img[y_lo:y_hi, x_lo:x_hi]
        if patch.size == 0 or not np.any(np.isfinite(patch)):
            xo[i] = x_ref
            yo[i] = y_ref
            n_locked += 1
            continue
        flat_idx = int(np.nanargmax(patch))
        py, px = np.unravel_index(flat_idx, patch.shape)
        xo[i] = float(x_lo + int(px))
        yo[i] = float(y_lo + int(py))
        n_locked += 1
    return xo, yo, n_locked


def _proc_deduplicate_matched_catalog_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per non-empty ``catalog_id`` (highest peak / flux wins).

    Multiple DAO detections can match the same Gaia source in one frame; duplicates
    inflate Phase-1 comp RMS and can select faint spurious matches.
    """
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    out = df.copy()
    cid = out["catalog_id"].fillna("").astype(str).str.strip()
    matched = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    if not bool(matched.any()):
        return out
    # Always a Series: out.get("flux") is None when the column is absent (empty-DAO
    # frames after forced-phot inject - draft 515 crash), and pd.to_numeric(None)
    # collapses to numpy.float64 without .fillna.
    if "peak_max_adu" in out.columns:
        score = pd.to_numeric(out["peak_max_adu"], errors="coerce")
    elif "dao_flux" in out.columns:
        score = pd.to_numeric(out["dao_flux"], errors="coerce")
    elif "flux" in out.columns:
        score = pd.to_numeric(out["flux"], errors="coerce")
    else:
        score = pd.Series(float("nan"), index=out.index, dtype="float64")
    score = pd.Series(pd.to_numeric(score, errors="coerce"), index=out.index, dtype="float64")
    out["_dedupe_score"] = score.fillna(-1.0)
    keep_idx = (
        out.loc[matched]
        .sort_values("_dedupe_score", ascending=False, kind="mergesort")
        .drop_duplicates(subset=["catalog_id"], keep="first")
        .index
    )
    keep_mask = out.index.isin(keep_idx) | ~matched
    out = out.loc[keep_mask].drop(columns=["_dedupe_score"]).reset_index(drop=True)
    n_dropped = int(matched.sum()) - int(len(keep_idx))
    if n_dropped > 0:
        LOGGER.debug("[PROC] deduplicated %d duplicate catalog_id rows in per-frame CSV", n_dropped)
    return out


def _mean_bin2d_for_dao(data0: "np.ndarray", factor: int) -> tuple["np.ndarray", int]:
    import numpy as np

    f = int(factor)
    a = np.asarray(data0, dtype=np.float32)
    if f <= 1:
        return a, 1
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return a, 1
    b = a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3)).astype(np.float32)
    return b, f


from masterstar_gaia_accounting import _dao_xy_binned_to_full  # noqa: E402,F401


from masterstar_gaia_accounting import _dao_full_to_binned_xy  # noqa: E402,F401


def _catalog_match_radius_px(
    wcs_obj: Any,
    *,
    match_sep_arcsec: float,
    wpx: int,
    h: int,
) -> float:
    """Pixel matching radius for Gaia<->DAO (floor 10 px)."""
    import numpy as np
    from astropy.wcs.utils import proj_plane_pixel_scales

    floor_px = 10.0
    try:
        scales_deg = proj_plane_pixel_scales(wcs_obj)
        arcsec_per_px = float(np.mean(scales_deg)) * 3600.0
        if math.isfinite(arcsec_per_px) and arcsec_per_px > 0:
            return max(floor_px, float(match_sep_arcsec) / arcsec_per_px)
    except Exception:  # noqa: BLE001
        pass
    return floor_px


from masterstar_gaia_accounting import _dao_pass2_annulus_stats  # noqa: E402,F401


def _gaia_chip_xy_from_catalog(
    cat_df: pd.DataFrame | None,
    wcs_obj: Any,
    *,
    wpx: int,
    h: int,
) -> pd.DataFrame:
    """On-chip Gaia rows with x_gaia/y_gaia for star-mask and born-owned pass2."""
    import numpy as np

    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return pd.DataFrame()
    ra = pd.to_numeric(cat_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(cat_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)
    if not bool(ok.any()):
        return pd.DataFrame()
    gx, gy = wcs_obj.world_to_pixel_values(ra[ok], de[ok])
    sub = cat_df.loc[ok].copy().reset_index(drop=True)
    sub["x_gaia"] = gx
    sub["y_gaia"] = gy
    sub["g_mag"] = pd.to_numeric(sub.get("mag"), errors="coerce")
    inb = (
        (sub["x_gaia"] >= 0)
        & (sub["x_gaia"] < float(wpx))
        & (sub["y_gaia"] >= 0)
        & (sub["y_gaia"] < float(h))
    )
    return sub.loc[inb].reset_index(drop=True)


def _merge_dao_pass1_pass2_tables(
    tbl_pass1: Any,
    pass2_rows: list[dict[str, float]],
    *,
    bfac: int,
    dedup_px: float = 3.0,
) -> Any:
    """Append pass-2 detections; drop pass-2 if within ``dedup_px`` of pass-1 (full image coords).

    Adds integer column ``vy_dao_pass`` (1=pass1, 2=pass2) so the prematch peak gate can
    exempt pass-2 recoveries (SNR-GATE-01).
    """
    import numpy as np
    from astropy.table import Table, vstack

    if tbl_pass1 is not None and len(tbl_pass1) > 0 and "vy_dao_pass" not in tbl_pass1.colnames:
        tbl_pass1 = Table(tbl_pass1, copy=True)
        tbl_pass1["vy_dao_pass"] = np.ones(len(tbl_pass1), dtype=np.int16)

    if not pass2_rows:
        return tbl_pass1
    p1x = np.asarray([], dtype=np.float64)
    p1y = np.asarray([], dtype=np.float64)
    if tbl_pass1 is not None and len(tbl_pass1) > 0:
        xb = np.asarray(tbl_pass1["x_centroid"], dtype=np.float64)
        yb = np.asarray(tbl_pass1["y_centroid"], dtype=np.float64)
        p1x, p1y = _dao_xy_binned_to_full(xb, yb, int(bfac))
    kept: list[dict[str, float]] = []
    for row in pass2_rows:
        xf = float(row["x_full"])
        yf = float(row["y_full"])
        if p1x.size:
            d = np.hypot(p1x - xf, p1y - yf)
            if float(np.min(d)) < float(dedup_px):
                continue
        xb, yb = _dao_full_to_binned_xy(xf, yf, int(bfac))
        kept.append(
            {
                "x_centroid": xb,
                "y_centroid": yb,
                "flux": float(row["flux"]),
                "peak": float(row.get("peak", row["flux"])),
                "vy_dao_pass": 2,
            }
        )
    if not kept:
        return tbl_pass1
    t2 = Table(kept)
    if tbl_pass1 is None or len(tbl_pass1) == 0:
        return t2
    return vstack([tbl_pass1, t2])


def _dao_targeted_pass2_unmatched_gaia(
    data0: "np.ndarray",
    tbl_pass1: Any,
    *,
    cat_df: pd.DataFrame,
    wcs_obj: Any,
    bfac: int,
    fwhm_px: float,
    pass2_sigma: float,
    pass2_center_tol_px: float = 5.0,
    match_sep_arcsec: float,
    wpx: int,
    h: int,
) -> tuple[Any, int, int]:
    """Run local DAO on Gaia positions with no pass-1 DAO neighbor. Returns (merged_tbl, n_unmatched, n_pass2)."""
    import numpy as np
    from photutils.detection import DAOStarFinder  # type: ignore

    from masterstar_gaia_accounting import Pass2AcceptParams, dao_pass2_try_at_position

    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return tbl_pass1, 0, 0

    ra = pd.to_numeric(cat_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(cat_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)
    if not bool(ok.any()):
        return tbl_pass1, 0, 0

    gx, gy = wcs_obj.world_to_pixel_values(ra[ok], de[ok])
    inb = (gx >= 0) & (gx < float(wpx)) & (gy >= 0) & (gy < float(h))
    gx = gx[inb]
    gy = gy[inb]
    n_gaia_in = int(gx.size)
    if n_gaia_in == 0:
        return tbl_pass1, 0, 0

    match_r_px = _catalog_match_radius_px(wcs_obj, match_sep_arcsec=float(match_sep_arcsec), wpx=wpx, h=h)
    dao_x = np.asarray([], dtype=np.float64)
    dao_y = np.asarray([], dtype=np.float64)
    if tbl_pass1 is not None and len(tbl_pass1) > 0:
        xb = np.asarray(tbl_pass1["x_centroid"], dtype=np.float64)
        yb = np.asarray(tbl_pass1["y_centroid"], dtype=np.float64)
        dao_x, dao_y = _dao_xy_binned_to_full(xb, yb, int(bfac))

    unmatched_idx: list[int] = []
    if dao_x.size == 0:
        unmatched_idx = list(range(n_gaia_in))
    else:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.column_stack([dao_x, dao_y]))
        dist, _ = tree.query(np.column_stack([gx, gy]), distance_upper_bound=float(match_r_px))
        unmatched_idx = [i for i, d in enumerate(dist) if not np.isfinite(d) or float(d) > float(match_r_px)]

    n_unmatched = len(unmatched_idx)
    if n_unmatched == 0:
        return tbl_pass1, 0, 0

    sigma_p2 = max(1.5, min(20.0, float(pass2_sigma)))
    fwhm_cut = max(1.2, min(20.0, float(fwhm_px)))
    center_tol = max(0.5, min(10.0, float(pass2_center_tol_px)))
    p2_params = Pass2AcceptParams(
        sigma=sigma_p2,
        center_tol_px=center_tol,
        fwhm_px=float(fwhm_cut),
    )
    pass2_rows: list[dict[str, float]] = []
    n_empty_cutouts = 0

    for i in unmatched_idx:
        x0 = float(gx[i])
        y0 = float(gy[i])
        hit = dao_pass2_try_at_position(
            data0, x0, y0, wpx=int(wpx), h=int(h), params=p2_params
        )
        if not hit.get("accepted"):
            if str(hit.get("reason", "")) == "no_detection":
                n_empty_cutouts += 1
            continue
        pass2_rows.append(
            {
                "x_full": float(hit["x_det"]),
                "y_full": float(hit["y_det"]),
                "flux": float(hit["flux"]),
                "peak": float(hit.get("peak", hit["flux"])),
            }
        )

    if n_empty_cutouts > 0:
        LOGGER.info(
            "[DAO pass 2] %d/%d targeted cutouts had no detection "
            "(NoDetectionsWarning suppressed)",
            int(n_empty_cutouts),
            int(n_unmatched),
        )
    merged = _merge_dao_pass1_pass2_tables(tbl_pass1, pass2_rows, bfac=int(bfac), dedup_px=3.0)
    return merged, n_unmatched, len(pass2_rows)


def _proc_rename_det_names_to_catalog_id(df: pd.DataFrame) -> pd.DataFrame:
    """Matched rows still named DET_* get ``name`` = ``catalog_id`` (stale master names)."""
    if df is None or df.empty or "catalog_id" not in df.columns or "name" not in df.columns:
        return df
    _matched_mask = (
        df["catalog_id"].notna()
        & (df["catalog_id"].astype(str).str.strip() != "")
        & (df["name"].astype(str).str.startswith("DET_"))
    )
    if not _matched_mask.any():
        return df
    out = df.copy()
    out.loc[_matched_mask, "name"] = out.loc[_matched_mask, "catalog_id"].astype(str)
    return out


def _proc_drop_unmatched_dao_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unmatched DAO_ONLY detections (no catalog_id) BEFORE expensive per-star work.

    GAIA_MATCHED rows carry a catalog_id and are kept; DAO_ONLY rows
    (empty catalog_id) are discarded by the final keep-matched filter anyway. Keying on
    catalog_id (not source_type) is correct in both export paths, including the in-process path
    where source_type=GAIA_MATCHED is assigned only later.
    """
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    cid = df["catalog_id"].fillna("").astype(str).str.strip()
    keep = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    return df.loc[keep].reset_index(drop=True)


def _proc_catalog_keep_matched_rows_only(df: pd.DataFrame) -> pd.DataFrame:
    """TODO-13: keep catalog rows with valid ``catalog_id`` and ``source_type``."""
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    cid = df["catalog_id"].fillna("").astype(str).str.strip()
    _valid_cid = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    if "source_type" in df.columns:
        _valid_source = (
            df["source_type"].fillna("").astype(str).str.strip().str.upper().eq("GAIA_MATCHED")
        )
        keep = _valid_cid & _valid_source
    else:
        keep = _valid_cid
    return df.loc[keep].reset_index(drop=True)


def _prefilter_dao_table_brightest(tbl: Any, keep_top: int) -> Any:
    """Cap DAO rows before sorting when the finder returns an enormous table (slow + useless)."""
    import numpy as np

    if tbl is None or len(tbl) <= int(keep_top):
        return tbl
    flux_np = np.asarray(tbl["flux"], dtype=np.float64)
    k = int(keep_top)
    take = np.argpartition(flux_np, -k)[-k:]
    return tbl[take]


def _dao_spatial_flux_cap_row_indices(
    tbl: Any,
    *,
    max_n: int,
    width_px: float,
    height_px: float,
) -> "np.ndarray":
    """Row indices into ``tbl`` for up to ``max_n`` sources, spread across the detector.

    Sorting globally by flux and truncating biases toward the frame centre when vignetting or
    gradients make edge stars fainter in ADU - QA then falsely looks like a ``catalog disc``. This
    fills a coarse grid brightest-first per cell, then tops up by global flux (same cap).
    """
    import numpy as np

    n = len(tbl)
    m = int(max_n)
    if n <= m:
        return np.arange(n, dtype=np.int64)
    x = np.asarray(tbl["x_centroid"], dtype=np.float64)
    y = np.asarray(tbl["y_centroid"], dtype=np.float64)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    w = max(float(width_px), 1.0)
    h = max(float(height_px), 1.0)
    # ~25 sources per cell on average, bounded grid
    ncell_target = max(48, min(512, max(1, m // 25)))
    aspect = w / h
    ny = max(4, int(round((ncell_target / aspect) ** 0.5)))
    nx = max(4, int(round(ncell_target / float(ny))))
    ncells = nx * ny
    per_cell = max(1, m // ncells)
    ix = np.clip((x / w * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((y / h * ny).astype(np.int64), 0, ny - 1)
    cell_id = ix + iy * nx
    order = np.argsort(-flux)
    taken = np.zeros(n, dtype=bool)
    picked: list[int] = []
    counts = np.zeros(ncells, dtype=np.int32)
    for idx in order:
        c = int(cell_id[idx])
        if counts[c] < per_cell and len(picked) < m:
            picked.append(int(idx))
            counts[c] += 1
            taken[idx] = True
    for idx in order:
        if len(picked) >= m:
            break
        if not taken[idx]:
            picked.append(int(idx))
            taken[idx] = True
    return np.asarray(sorted(picked), dtype=np.int64)


def detect_stars_match_master_reference(
    data: "np.ndarray",
    hdr: fits.Header,
    master_df: pd.DataFrame,
    *,
    max_catalog_rows: int = 12000,
    match_sep_arcsec: float = 8.0,
    saturate_level_fraction: float = 0.999,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    fallback_saturate_adu: float | None = None,
    equipment_saturate_adu: float | None = None,
    frame_name: str = "",
    sat_diag_ctx: Any | None = None,
    raw_data: "np.ndarray | None" = None,
    raw_hdr: fits.Header | None = None,
    ref_ra_deg: float | None = None,
    ref_dec_deg: float | None = None,
    drift_ref_catalog_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """DAO on this frame + nearest-neighbor match to ``masterstars.csv`` (no Vizier / cone).

    Catalog IDs and static columns come from the master row; ``x``, ``y``, ``flux``, ``peak_*`` and
    saturation flags are per-frame. Intended for ``detrended_aligned`` data whose WCS matches
    ``MASTERSTAR.fits`` astrometry.
    """
    import numpy as np

    m = master_df
    # Primary mode: sky match using per-frame WCS (DAO x/y -> RA/Dec), then NN match against MASTERSTAR ra/dec.
    # Fallback: if frame has no usable celestial WCS, do a pixel NN match (15 px) against MASTERSTAR x/y.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)

    # MASTERSTAR sky catalog
    if "ra_deg" not in m.columns or "dec_deg" not in m.columns:
        raise ValueError("masterstars table must contain ra_deg, dec_deg")
    _ra = pd.to_numeric(m["ra_deg"], errors="coerce")
    _de = pd.to_numeric(m["dec_deg"], errors="coerce")
    okm_sky = _ra.notna() & _de.notna()
    if not bool(okm_sky.any()):
        raise ValueError("No valid ra_deg/dec_deg rows in masterstars")
    m_valid = m.loc[okm_sky].reset_index(drop=True)
    ra_m = pd.to_numeric(m_valid["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de_m = pd.to_numeric(m_valid["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    master_coords = SkyCoord(ra=ra_m * u.deg, dec=de_m * u.deg, frame="icrs")

    match_mode = "sky"
    plate_scale_arcsec_per_px = None

    arr = np.asarray(data, dtype=np.float32)
    mean, med, std = plain_mean_med_std(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = 1
    _cfg_dao = AppConfig()
    _fb_m = float(dao_fwhm_px) if dao_fwhm_px is not None else float(_cfg_dao.sips_dao_fwhm_px)
    _base_fw_m = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_m)
    try:
        from photutils.detection import DAOStarFinder  # type: ignore

        thr_s = float(dao_threshold_sigma)
        thr_s = max(0.5, min(20.0, thr_s))
        dao_scale = _dao_auto_binning_factor(*data0.shape)
        data_dao, bfac = _mean_bin2d_for_dao(data0, dao_scale)
        fwhm_eff = max(1.2, _base_fw_m / float(bfac))
        rms_conv, _dao_rel_err = _dao_convolved_background_rms_adu(data_dao, fwhm_px=fwhm_eff)
        sigma_pp_diag = _dao_noise_sigma_adu(arr, bfac=bfac, fallback_std=float(std), data_dao=data_dao)
        std_dao = rms_conv
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            try:
                std_dao = float(np.nanstd(arr))
            except Exception:  # noqa: BLE001
                std_dao = float("nan")
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            try:
                finite = np.isfinite(arr)
                n_finite = int(np.count_nonzero(finite))
                if n_finite > 0:
                    vals = arr[finite]
                    n_unique = int(len(np.unique(vals)))
                    mn = float(np.nanmin(vals))
                    mx = float(np.nanmax(vals))
                else:
                    n_unique, mn, mx = 0, float("nan"), float("nan")
                print(
                    f"DEBUG std=0: {_nm} n_unique={n_unique} n_finite={n_finite} "
                    f"min={mn:.1f} max={mx:.1f}"
                )
            except Exception:  # noqa: BLE001
                pass
            # If the frame isn't constant (min != max), try std from non-zero finite pixels.
            try:
                nonzero_mask = (arr != 0) & np.isfinite(arr)
                if int(np.count_nonzero(nonzero_mask)) > 100:
                    std_dao = float(np.std(arr[nonzero_mask]))
            except Exception:  # noqa: BLE001
                pass
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(f"WARNING: {_nm} std=0 aj po fallback, preskakujem")
            empty_meta: dict[str, Any] = {"status": "std_zero"}
            return pd.DataFrame(), empty_meta
        _thr, _n_equiv_used = _dao_detection_threshold_adu(
            float(std_dao), cfg=_cfg_dao, dao_threshold_sigma=float(thr_s),
        )
        try:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(
                f"DEBUG DAO INPUT: {_nm} mean={float(np.nanmean(arr)):.1f} std={float(np.nanstd(arr)):.1f} "
                f"threshold={float(_thr):.1f} rms_conv={float(rms_conv):.2f} n_equiv={float(_n_equiv_used):.2f} "
                f"sigma_pp_diag={float(sigma_pp_diag):.2f} "
                f"fwhm={float(fwhm_eff):.2f}"
            )
        except Exception:  # noqa: BLE001
            pass
        finder = DAOStarFinder(
            fwhm=float(fwhm_eff),
            threshold=float(_thr),
            scale_threshold=False,
            n_brightest=None,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(data_dao)
        if tbl is not None and len(tbl) > 0:
            tbl = _prefilter_dao_table_brightest(tbl, max(int(max_catalog_rows) * 12, 36_000))
            from astropy.table import Table as _AstropyTable

            tbl = _AstropyTable(tbl, copy=True)
            tbl["vy_dao_pass"] = np.ones(len(tbl), dtype=np.int16)
        n_pass1_dao_m = int(len(tbl)) if tbl is not None else 0
        _h_m, _wpx_m = int(data0.shape[0]), int(data0.shape[1])
        try:
            _sigma_p2_m = float(_cfg_dao.masterstar_dao_pass2_sigma)
        except (TypeError, ValueError):
            _sigma_p2_m = 1.9
        try:
            _ctol_p2_m = float(_cfg_dao.masterstar_dao_pass2_center_tol_px)
        except (TypeError, ValueError):
            _ctol_p2_m = 5.0
        tbl, _n_unmatched_master, _n_pass2_raw_m = _dao_targeted_pass2_unmatched_gaia(
            data0,
            tbl,
            cat_df=m_valid,
            wcs_obj=wcs_obj,
            bfac=int(bfac),
            fwhm_px=float(max(1.2, _base_fw_m)),
            pass2_sigma=float(_sigma_p2_m),
            pass2_center_tol_px=float(_ctol_p2_m),
            match_sep_arcsec=float(match_sep_arcsec),
            wpx=int(_wpx_m),
            h=int(_h_m),
        )
        n_merged_dao_m = int(len(tbl)) if tbl is not None else 0
        LOGGER.info(
            "[DAO pass 1 master] %d detections, %d MASTERSTAR unmatched",
            int(n_pass1_dao_m),
            int(_n_unmatched_master),
        )
        LOGGER.info(
            "[DAO pass 2 master] %d additional detections from %d targeted positions",
            int(_n_pass2_raw_m),
            int(_n_unmatched_master),
        )
        LOGGER.info("[DAO total master] %d detections after merge", int(n_merged_dao_m))
    except Exception:  # noqa: BLE001
        tbl = None
        bfac = 1

    _fb_sat = fallback_saturate_adu

    if sat_diag_ctx is not None and getattr(sat_diag_ctx, "sat_adu", None) is not None:
        sat_limit = float(sat_diag_ctx.sat_adu)
        sat_limit_src = str(getattr(sat_diag_ctx, "sat_source", "sat_diag"))
    else:
        sat_limit, sat_limit_src = _effective_saturation_limit(
            hdr, fallback_adu=_fb_sat, equipment_saturate_adu=equipment_saturate_adu
        )
    sat_frac = float(saturate_level_fraction)
    sat_frac = min(max(sat_frac, 0.5), 1.0)

    foot_meta = {
        "catalog_footprint": {"method": "master_reference_only"},
        "saturation": {
            "effective_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "limit_source": sat_limit_src,
            "plateau_half_inner_px": 1,
            "plateau_rel": 0.996,
            "plateau_min_pixels": 5,
        },
    }

    empty_meta = {
        "n_detected": 0,
        "n_detected_dao": 0,
        "n_matched": 0,
        "n_matched_before_mag_limit": 0,
        "catalog_rows": int(len(master_df)),
        "catalog_match_mode": export_catalog_match_mode_from_internal(match_mode),
        "n_likely_saturated": 0,
        "n_saturated_from_peak": 0,
        "n_saturated_plateau": 0,
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": 0,
        "n_gaia_variable_in_field": 0,
        "field_catalog_cone_csv": None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": float(max(1.2, _base_fw_m)),
        "dao_detect_binning": 1,
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_sep_arcsec),
        "plate_scale_arcsec_per_px": (
            float(plate_scale_arcsec_per_px) if plate_scale_arcsec_per_px is not None else None
        ),
        **foot_meta,
    }

    _chip_h, _chip_wpx = int(arr.shape[0]), int(arr.shape[1])

    if tbl is None or len(tbl) == 0:
        df_out = pd.DataFrame()
        df_out = _proc_rename_det_names_to_catalog_id(df_out)
        empty_meta["faintest_mag_limit"] = float(faintest_mag_limit) if faintest_mag_limit is not None else None
        empty_meta["n_dropped_fainter_than_limit"] = 0
        empty_meta["n_detected"] = int(len(df_out))
        empty_meta["n_matched"] = int(
            df_out["catalog_id"].fillna("").astype(str).str.strip().ne("").sum()
        ) if len(df_out) and "catalog_id" in df_out.columns else 0
        return df_out, empty_meta

    _fwhm_used_m = float(max(1.2, _base_fw_m / float(bfac)))
    _d_h, _d_w = int(data_dao.shape[0]), int(data_dao.shape[1])
    _keep = _dao_spatial_flux_cap_row_indices(
        tbl, max_n=int(max_catalog_rows), width_px=float(_d_w), height_px=float(_d_h)
    )
    tbl = tbl[_keep]
    tbl.sort("flux")
    tbl = tbl[::-1]
    n = len(tbl)
    xb = np.asarray(tbl["x_centroid"], dtype=np.float64)
    yb = np.asarray(tbl["y_centroid"], dtype=np.float64)
    x, y = _dao_xy_binned_to_full(xb, yb, bfac)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    peak_dao = np.asarray(tbl["peak"], dtype=np.float64) if "peak" in tbl.colnames else np.full(n, np.nan)

    match_thr = float(match_sep_arcsec)

    def _pixel_nn_match(*, dist_thr_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Pixel-space NN match against MASTERSTAR x/y.

        Returns: (icomp, sep_arr, oix)
        - icomp: indices into the *filtered* master_xy array (as from KDTree query)
        - sep_arr: distance in px (NaN when unmatched)
        - oix: mapping from filtered master_xy indices back to m_valid row indices
        """
        ic: np.ndarray
        sep: np.ndarray
        ox: np.ndarray | None
        if "x" not in m_valid.columns or "y" not in m_valid.columns:
            ic = np.zeros(n, dtype=np.int64)
            sep = np.full(n, np.nan, dtype=np.float64)
            ox = None
            return ic, sep, ox
        try:
            from scipy.spatial import cKDTree  # type: ignore

            mx = pd.to_numeric(m_valid["x"], errors="coerce").to_numpy(dtype=np.float64)
            my = pd.to_numeric(m_valid["y"], errors="coerce").to_numpy(dtype=np.float64)
            okxy = np.isfinite(mx) & np.isfinite(my)
            mx2 = mx[okxy]
            my2 = my[okxy]
            ox = np.nonzero(okxy)[0].astype(np.int64)
            if mx2.size == 0:
                ic = np.zeros(n, dtype=np.int64)
                sep = np.full(n, np.nan, dtype=np.float64)
                return ic, sep, ox
            master_xy = np.column_stack([mx2, my2])
            det_xy = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
            tree = cKDTree(master_xy)
            dist_px, idx = tree.query(det_xy, distance_upper_bound=float(dist_thr_px))
            ic = np.asarray(idx, dtype=np.int64)
            sep = np.asarray(dist_px, dtype=np.float64)  # px
            sep[~np.isfinite(sep)] = np.nan
            return ic, sep, ox
        except Exception:  # noqa: BLE001
            ic = np.zeros(n, dtype=np.int64)
            sep = np.full(n, np.nan, dtype=np.float64)
            ox = None
            return ic, sep, ox

    icomp: np.ndarray | None = None
    sep_arcsec_arr: np.ndarray
    oix: np.ndarray | None = None
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    _frame_on_ref_grid = _fits_header_vy_algn_aligned(hdr)

    # Robust strategy:
    # - If celestial WCS exists, try sky-match first (arcsec threshold).
    # - If sky-match looks suspiciously bad (e.g. WCS offset), fall back to pixel match
    #   only when the frame is on the reference alignment grid (VY_ALGN=True).
    # Matching unaligned DAO xy to the reference pixel grid is invalid by construction.
    if getattr(wcs_obj, "has_celestial", False):
        ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
        det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        idx_sky, sep2d, _ = det_coords.match_to_catalog_sky(master_coords)
        sep_sky = np.asarray(sep2d.to_value(u.arcsec), dtype=np.float64)
        ic_sky = np.asarray(idx_sky, dtype=np.int64)

        # Heuristic: if typical nearest-neighbor sep is far above the threshold,
        # the WCS is likely offset (e.g. wrong reference grid / flip / stale solve).
        med_sep = float(np.nanmedian(sep_sky)) if np.isfinite(np.nanmedian(sep_sky)) else float("inf")
        if (not np.isfinite(med_sep)) or (med_sep > max(30.0, match_thr * 3.0)):
            if not _frame_on_ref_grid:
                icomp = ic_sky
                sep_arcsec_arr = sep_sky
                oix = None
                match_mode = "sky_unaligned_no_pixel_fallback"
            else:
                ic_px, sep_px, oix_px = _pixel_nn_match(dist_thr_px=float(6.0))
                # Prefer pixel match if it yields any finite distances.
                if int(np.count_nonzero(np.isfinite(sep_px))) > 0:
                    icomp = ic_px
                    sep_arcsec_arr = sep_px  # px (kept numeric for mask)
                    oix = oix_px
                    match_mode = "pixel_fallback_bad_wcs"
                    ra_deg = np.full(n, np.nan, dtype=np.float64)
                    dec_deg = np.full(n, np.nan, dtype=np.float64)
                else:
                    icomp = ic_sky
                    sep_arcsec_arr = sep_sky
                    oix = None
                    match_mode = "sky"
        else:
            icomp = ic_sky
            sep_arcsec_arr = sep_sky
            oix = None
            match_mode = "sky"
    else:
        ra_deg = np.full(n, np.nan, dtype=np.float64)
        dec_deg = np.full(n, np.nan, dtype=np.float64)
        if not _frame_on_ref_grid:
            match_mode = "nondet_unaligned_no_wcs"
            icomp = np.zeros(n, dtype=np.int64)
            sep_arcsec_arr = np.full(n, np.nan, dtype=np.float64)
            oix = None
        else:
            match_mode = "pixel_fallback_no_wcs"
            ic_px, sep_px, oix_px = _pixel_nn_match(dist_thr_px=float(15.0))
            icomp = ic_px
            sep_arcsec_arr = sep_px  # px (kept numeric for mask)
            oix = oix_px

    nm_m = len(m_valid)
    idx_det = np.arange(1, n + 1, dtype=np.int32)
    det_str = np.array([f"DET_{i:04d}" for i in idx_det], dtype=object)
    finite_sep = np.isfinite(sep_arcsec_arr)
    icomp_a = icomp if icomp is not None else np.zeros(n, dtype=np.int64)
    if match_mode in ("pixel_fallback_no_wcs", "pixel_fallback_bad_wcs") and oix is not None:
        ok_ix = (icomp_a >= 0) & (icomp_a < int(len(oix)))
        cat_row = np.full(n, -1, dtype=np.int64)
        cat_row[ok_ix] = oix[icomp_a[ok_ix]]
        thr_px = float(15.0) if match_mode == "pixel_fallback_no_wcs" else float(6.0)
        matched = finite_sep & ok_ix & (sep_arcsec_arr <= thr_px) & (cat_row >= 0)
        # For output semantics, keep match_sep_arcsec as NaN when unmatched (already handled).
    else:
        cat_row = np.clip(icomp_a.astype(np.int64, copy=False), 0, max(nm_m - 1, 0))
        matched = finite_sep & (sep_arcsec_arr <= match_thr) & (nm_m > 0)

    safe = np.clip(np.where(matched, cat_row, 0), 0, max(nm_m - 1, 0))
    cat_s_m = (
        m_valid["catalog"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "catalog" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    cid_m = (
        m_valid["catalog_id"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "catalog_id" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    nm_c = (
        m_valid["name"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "name" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    mag_m = (
        pd.to_numeric(m_valid["mag"], errors="coerce").to_numpy(dtype=np.float64)
        if "mag" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    bv_m = (
        pd.to_numeric(m_valid["b_v"], errors="coerce").to_numpy(dtype=np.float64)
        if "b_v" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gn_m = (
        pd.to_numeric(m_valid["gaia_nss"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_nss" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gq_m = (
        pd.to_numeric(m_valid["gaia_qso"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_qso" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gg_m = (
        pd.to_numeric(m_valid["gaia_gal"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_gal" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    vx_m = (
        m_valid["vsx_known_variable"].fillna(False).astype(bool).to_numpy()
        if "vsx_known_variable" in m_valid.columns
        else np.zeros(nm_m, dtype=bool)
    )
    gv_m = (
        m_valid["gaia_dr3_variable_catalog"].fillna(False).astype(bool).to_numpy()
        if "gaia_dr3_variable_catalog" in m_valid.columns
        else np.zeros(nm_m, dtype=bool)
    )
    if "catalog_known_variable" in m_valid.columns:
        ck_m = m_valid["catalog_known_variable"].fillna(False).astype(bool).to_numpy()
    else:
        ck_m = vx_m | gv_m

    _fwhm_cent = float(max(1.2, _base_fw_m / float(bfac)))
    if _frame_on_ref_grid:
        x, y, _n_dao_wcs_fallback = _lock_matched_centroids_to_master_grid(
            arr,
            x,
            y,
            matched=matched,
            safe=safe,
            master_df=m_valid,
            fwhm_px=_fwhm_cent,
        )
    else:
        x, y, _n_dao_wcs_fallback = _apply_dao_centroid_wcs_guard(
            x,
            y,
            matched=matched,
            safe=safe,
            master_df=m_valid,
            fwhm_px=_fwhm_cent,
            max_shift_fwhm=float(getattr(_cfg_dao, "dao_centroid_max_shift_fwhm", 1.0)),
        )

    pmax_arr = _box_peaks_at_centroids(arr, x, y)
    _frame_max_adu = float(np.nanmax(arr))
    _empirical_clip_adu = _detect_empirical_clip_level_adu(arr)
    _peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_limit,
        saturate_fraction=sat_frac,
        sky_median_adu=float(med),
        frame_max_adu=_frame_max_adu,
        empirical_clip_adu=_empirical_clip_adu,
    )
    _sat_block = _vectorized_star_saturation_columns(
        arr,
        x,
        y,
        sat_limit=_peak_sat_lim,
        sat_frac=sat_frac,
        peak_dao=peak_dao,
        peak_max_adu=pmax_arr,
    )
    _sat_csv, n_sat_pk, n_sat_pl = _proc_sat_block_for_csv(_sat_block)

    cat_sel = cat_s_m[safe]
    cid_sel = cid_m[safe]
    nm_sel = nm_c[safe]
    name_cand = np.where(
        nm_sel != "",
        nm_sel,
        np.where(cid_sel != "", cid_sel, np.where(cat_sel != "", cat_sel, det_str)),
    )
    name_out = np.where(matched, name_cand, det_str)

    mag_out = np.full(n, np.nan, dtype=np.float64)
    bv_out = np.full(n, np.nan, dtype=np.float64)
    gn_out = np.full(n, np.nan, dtype=np.float64)
    gq_out = np.full(n, np.nan, dtype=np.float64)
    gg_out = np.full(n, np.nan, dtype=np.float64)
    mag_out[matched] = mag_m[safe[matched]]
    bv_out[matched] = bv_m[safe[matched]]
    gn_out[matched] = gn_m[safe[matched]]
    gq_out[matched] = gq_m[safe[matched]]
    gg_out[matched] = gg_m[safe[matched]]

    vx_out = np.zeros(n, dtype=bool)
    gv_out = np.zeros(n, dtype=bool)
    ck_out = np.zeros(n, dtype=bool)
    vx_out[matched] = vx_m[safe[matched]]
    gv_out[matched] = gv_m[safe[matched]]
    ck_out[matched] = ck_m[safe[matched]]

    cat_out = np.array([""] * n, dtype=object)
    cid_out = np.array([""] * n, dtype=object)
    cat_out[matched] = cat_s_m[safe[matched]]
    cid_out[matched] = cid_m[safe[matched]]

    n_matched = int(np.count_nonzero(matched & (cat_s_m[safe].astype(str) != "")))

    df_out = pd.DataFrame(
        {
            "name": name_out,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "mag": mag_out,
            "b_v": bv_out,
            "catalog": cat_out,
            "catalog_id": cid_out,
            "x": x,
            "y": y,
            "flux": flux,
            "vsx_known_variable": vx_out,
            "gaia_dr3_variable_catalog": gv_out,
            **_sat_csv,
        }
    )
    n_detected_dao = int(n)
    n_matched_before_mag = int(n_matched)
    n_before_mag = len(df_out)
    if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit)):
        lim_m = float(faintest_mag_limit)
        mcol = pd.to_numeric(df_out["mag"], errors="coerce")
        # Drop only matched stars fainter than limit; keep unmatched detections (no catalog mag) for QA.
        df_out = df_out.loc[mcol.isna() | (mcol <= lim_m)].reset_index(drop=True)
        meta_mag = {
            "faintest_mag_limit": lim_m,
            "n_dropped_fainter_than_limit": int(n_before_mag - len(df_out)),
        }
    else:
        meta_mag = {"faintest_mag_limit": None, "n_dropped_fainter_than_limit": 0}

    df_out = _proc_rename_det_names_to_catalog_id(df_out)

    n_sat = int(df_out["likely_saturated"].sum()) if len(df_out) and "likely_saturated" in df_out.columns else 0
    cat_nonempty = (
        df_out["catalog_id"].fillna("").astype(str).str.strip().ne("")
        if len(df_out) and "catalog_id" in df_out.columns
        else pd.Series([], dtype=bool)
    )
    n_matched_final = int(cat_nonempty.sum()) if len(df_out) else 0
    meta = {
        "n_detected_dao": n_detected_dao,
        "n_detected": int(len(df_out)),
        "n_matched_before_mag_limit": n_matched_before_mag,
        "n_matched": n_matched_final,
        "catalog_rows": int(len(master_df)),
        "catalog_match_mode": export_catalog_match_mode_from_internal(match_mode),
        "n_likely_saturated": n_sat,
        "n_saturated_from_peak": n_sat_pk,
        "n_saturated_plateau": n_sat_pl,
        "n_dao_wcs_centroid_fallback": int(_n_dao_wcs_fallback),
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": 0,
        "n_gaia_variable_in_field": 0,
        **foot_meta,
        "field_catalog_cone_csv": None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": _fwhm_used_m,
        "dao_detect_binning": int(bfac),
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_thr),
        "plate_scale_arcsec_per_px": (
            float(plate_scale_arcsec_per_px) if plate_scale_arcsec_per_px is not None else None
        ),
        **meta_mag,
    }
    if int(_n_dao_wcs_fallback) > 0:
        LOGGER.info(
            "[BATCH-E E.2] centroid WCS fallback triggered on %d star-frames",
            int(_n_dao_wcs_fallback),
        )
    if (
        sat_diag_ctx is not None
        and raw_data is not None
        and raw_hdr is not None
        and len(df_out) > 0
    ):
        try:
            from sat_diag import apply_raw_peaks_to_proc_df  # noqa: PLC0415

            apply_raw_peaks_to_proc_df(
                df_out,
                np.asarray(raw_data),
                raw_hdr,
                sat_diag_ctx,
                ref_ra=ref_ra_deg,
                ref_dec=ref_dec_deg,
                drift_ref_catalog_id=drift_ref_catalog_id,
                aligned_hdr=hdr,
            )
            n_sat = (
                int(df_out["likely_saturated_raw"].sum())
                if "likely_saturated_raw" in df_out.columns
                else n_sat
            )
            meta["sat_limit_source"] = str(getattr(sat_diag_ctx, "sat_source", sat_limit_src))
            meta["sat_peak_source"] = str(
                getattr(sat_diag_ctx, "sat_peak_source", "PLACED_APERTURE")
            )
            meta["raw_peaks_used"] = True
        except Exception as _sat_exc:  # noqa: BLE001
            LOGGER.warning("[SAT-DIAG] raw peak merge failed for %s: %s", frame_name, _sat_exc)
    return df_out, meta


def _merge_platesolve_gaia_pairs_into_masterstars_df(
    df: pd.DataFrame,
    *,
    pairs_x: list[float],
    pairs_y: list[float],
    pairs_ra: list[float],
    pairs_de: list[float],
    pairs_catalog_id: list[str],
    max_pair_px: float = 12.0,
) -> pd.DataFrame:
    """Map VYVAR plate-solve Gaia pairs onto DAO rows so ``astrometry_optimizer`` sees catalog_id + small sep."""
    import numpy as np

    if df is None or df.empty or not pairs_x:
        return df
    n = len(pairs_x)
    if len(pairs_y) != n or len(pairs_ra) != n or len(pairs_de) != n or len(pairs_catalog_id) != n:
        return df
    if "x" not in df.columns or "y" not in df.columns:
        return df
    out = df.copy()
    x = pd.to_numeric(out["x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out["y"], errors="coerce").to_numpy(dtype=np.float64)
    max2 = float(max_pair_px) ** 2
    used: set[int] = set()
    for i in range(n):
        cid = str(pairs_catalog_id[i] or "").strip()
        if not cid:
            continue
        d2 = (x - float(pairs_x[i])) ** 2 + (y - float(pairs_y[i])) ** 2
        d2[~np.isfinite(d2)] = np.inf
        j = int(np.argmin(d2))
        if j in used or float(d2[j]) > max2:
            continue
        used.add(j)
        if "catalog_id" in out.columns:
            out.iat[j, out.columns.get_loc("catalog_id")] = cid
        if "ra_deg" in out.columns:
            out.iat[j, out.columns.get_loc("ra_deg")] = float(pairs_ra[i])
        if "dec_deg" in out.columns:
            out.iat[j, out.columns.get_loc("dec_deg")] = float(pairs_de[i])
        if "match_sep_arcsec" in out.columns:
            out.iat[j, out.columns.get_loc("match_sep_arcsec")] = 0.25
    return out


def detect_stars_and_match_catalog(
    data: "np.ndarray",
    hdr: fits.Header,
    *,
    max_catalog_rows: int = 12000,
    cat_df: pd.DataFrame | None = None,
    vsx_df: pd.DataFrame | None = None,
    exo_df: pd.DataFrame | None = None,
    gaia_variable_df: pd.DataFrame | None = None,
    match_sep_arcsec: float = 8.0,
    vsx_match_max_sep_arcsec: float = 5.0,
    gaia_variable_match_max_sep_arcsec: float = 2.0,
    saturate_level_fraction: float = 0.999,
    faintest_mag_limit: float | None = None,
    gaia_db_path: Path | None = None,
    field_catalog_export_path: Path | None = None,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    fallback_saturate_adu: float | None = None,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    catalog_kd_pack: tuple[Any, "np.ndarray"] | None = None,
    plate_solve_fov_deg: float | None = None,
    fov_database_path: Path | str | None = None,
    fov_equipment_id: int | None = None,
    fov_draft_id: int | None = None,
    prematch_peak_sigma_floor: float = 10.0,
    prematch_exempt_pass2: bool = True,
    frame_name: str = "",
    dao_fwhm_bypass_header: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect stars (DAOStarFinder), sky coordinates from WCS, match to **local Gaia** (or pre-fetched ``cat_df``).

    If ``cat_df`` is None, builds a local Gaia cone/box catalog using ``AppConfig.gaia_db_path``.
    Pass the same ``cat_df``
    for every frame in a sequence with identical pointing/WCS scale to avoid repeated work.

    Pass ``catalog_kd_pack`` from ``build_ucac_catalog_kdtree(cat_df)`` when exporting many frames with the same
    cone table to avoid rebuilding match structures every call.

    **Known variables:** handled via Gaia flags and/or optional VSX checks (no Gaia TAP here).

    **Faintest magnitude:** if ``faintest_mag_limit`` is set (e.g. ``14``), **matched** stars with catalog
    ``mag`` fainter than the limit are dropped. **Unmatched** detections (no ``mag``) are kept for QA.

    ``match_sep_arcsec`` requested value is recorded; D1 sets the one-pass
    effective radius to max(12 arcsec, 3 x FWHM_dao_px x plate_scale).
    ``solve_rms_px`` is stamped as a diagnostic and does not enter the radius.
    There is no match-rate widening. A final tightening to ~4.5 arcsec is
    applied only when most loose matches survive it. Low match rate remains a
    WARN.

    ``max_catalog_rows`` caps DAO detections written per frame. Rows are chosen with **spatial
    stratification** (brightest per coarse grid cell, then global flux top-up) so vignetting does not
    mimic a ``catalog disc`` the way a plain brightest-N sort does.

    If ``field_catalog_export_path`` is set, the **full** cone table (``cat_df``) is written there for
    QA overlays - many more rows than DAO detections in ``masterstars.csv``.

    ``dao_threshold_sigma``: DAOStarFinder threshold = sigma x std(background); lower values detect more faint
    sources (cf. SIPS ~2.5sigma).

    ``prematch_peak_sigma_floor`` (default 10): before catalog matching, drop **pass-1** DAO rows whose local
    ``peak`` is below ``sky_median + k x sky_mad_sigma`` (SNR-GATE-01). Pass-2 recoveries are exempt when
    ``prematch_exempt_pass2`` is True (local annulus test already applied). Lower **k** keeps more faint pass-1
    detections (MASTERSTAR / ``config.json`` / DAO-STARS typicky **1.8-3.5**).

    Saturation: (1) ``peak_max_adu`` vs resolved ceiling from FITS keywords / ``EQUIPMENTS.SATURATE_ADU`` (before BITPIX);
    (2) **plateau core** - many pixels in the central 3x3
    near the local maximum (flat-top clipping, similar to a saturated radial profile). Row flags:
    ``saturated_from_peak``, ``saturated_plateau``, ``likely_saturated`` (OR), ``photometry_ok`` (not OR).
    """
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)
    if not wcs_obj.has_celestial:
        raise ValueError("FITS header has no usable celestial WCS for catalog matching.")

    arr = np.asarray(data, dtype=np.float32)
    h, wpx = arr.shape
    _fov_hint = plate_solve_fov_deg
    if _fov_hint is None:
        try:
            _fov_hint = resolve_plate_solve_fov_deg_hint(
                hdr,
                int(h),
                int(wpx),
                database_path=fov_database_path,
                equipment_id=fov_equipment_id,
                draft_id=fov_draft_id,
            )
        except Exception:  # noqa: BLE001
            _fov_hint = None
    if _fov_hint is None:
        try:
            _fov_hint = float(AppConfig().plate_solve_fov_deg)
        except Exception:  # noqa: BLE001
            _fov_hint = None
    center, radius_deg = _effective_field_catalog_cone_radius_deg(
        wcs_obj, h, wpx, _fov_hint, fits_header=hdr
    )
    if gaia_db_path is not None:
        _gaia_db_path = Path(gaia_db_path)
    else:
        _gaia_db_path: Path | None = None
    try:
        cfg = AppConfig()
        gp = (cfg.gaia_db_path or "").strip()
        if gp:
            _gaia_db_path = Path(gp)
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0360] field_catalog_cone_meta.json write failure leaves stale cone-radius metadata for cache ...: %s', exc)
        _gaia_db_path = None
    _fb_sat = fallback_saturate_adu
    try:
        _cfg_cap = int(AppConfig().catalog_query_max_rows)
    except Exception:  # noqa: BLE001
        _cfg_cap = 50_000
    _cat_cap_eff = max(int(max_catalog_rows), 50_000, int(_cfg_cap))
    if cat_df is None:
        _max_mag = float(faintest_mag_limit) if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit)) else None
        cat_df = _query_gaia_local(
            center=center,
            radius_deg=radius_deg,
            gaia_db_path=_gaia_db_path,
            max_mag=_max_mag,
            max_rows=int(_cat_cap_eff),
        )
    cat_df = _catalog_df_cap_brightest_by_mag(cat_df, max_rows=_cat_cap_eff)
    if field_catalog_export_path is not None and cat_df is not None and len(cat_df) > 0:
        _fcp = Path(field_catalog_export_path)
        _fcp.parent.mkdir(parents=True, exist_ok=True)
        _vyvar_df_to_csv(cat_df, _fcp)
        log_event(
            f"Vykreslujem katalog pre cele zorne pole: {int(wpx)}x{int(h)} pixelov "
            f"(export {len(cat_df)} riadkov do field_catalog_cone.csv, cap={int(_cat_cap_eff)}, kuzel r~{float(radius_deg):.2f} deg)."
        )
        log_event(
            f"KATALOG TARGET: export {_cat_cap_eff} riadkov do field_catalog_cone.csv "
            f"(ak je dostupnych >= {_cat_cap_eff})."
        )
        try:
            _write_field_catalog_cone_meta(
                _fcp,
                center=center,
                radius_deg=float(radius_deg),
                naxis1=int(wpx),
                naxis2=int(h),
                plate_solve_fov_deg=float(_fov_hint) if _fov_hint is not None else None,
            )
        except Exception:  # noqa: BLE001
            pass
    _ = catalog_local_gaia_only
    # ``vsx_df`` prazdny DataFrame z prefetch = 'uz sme skusali'; doplnaj len ak volajuci nepredal tabulku (``None``).
    if vsx_df is None:
        _vx: Path | None = None
        try:
            _vxs = str(cfg.vsx_local_db_path or "").strip()
            if _vxs:
                _vx = Path(_vxs).expanduser().resolve()
        except Exception:  # noqa: BLE001
            _vx = None
        vsx_df = _query_vsx_local(center=center, radius_deg=radius_deg, vsx_db_path=_vx)
    exo_annotation_active = False
    exo_max = 3.0
    try:
        exo_max = float(cfg.exoplanet_match_max_sep_arcsec)
        if not math.isfinite(exo_max):
            exo_max = 3.0
    except Exception:  # noqa: BLE001
        exo_max = 3.0
    exo_max = max(0.5, min(30.0, float(exo_max)))
    _exo_path: Path | None = None
    try:
        _exs = str(cfg.exoplanet_local_db_path or "").strip()
        if _exs:
            _exo_path = Path(_exs).expanduser().resolve()
    except Exception:  # noqa: BLE001
        _exo_path = None
    if _exo_path is not None and _exo_path.is_file():
        exo_annotation_active = True
    if exo_df is None and exo_annotation_active:
        exo_df = _query_exoplanet_local(
            center=center,
            radius_deg=radius_deg,
            exoplanet_db_path=_exo_path,
        )
    elif not exo_annotation_active:
        exo_df = pd.DataFrame()
    exo_ann: dict[str, Any] = {}
    if gaia_variable_df is None:
        gaia_variable_df = pd.DataFrame()

    mean, med, std = plain_mean_med_std(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = 1
    n_raw_dao = 0
    _cfg_df = AppConfig()
    _fb_c = float(dao_fwhm_px) if dao_fwhm_px is not None else float(_cfg_df.sips_dao_fwhm_px)
    if dao_fwhm_bypass_header and dao_fwhm_px is not None:
        try:
            _dao_arg = float(dao_fwhm_px)
            _base_fw = max(1.2, min(20.0, _dao_arg)) if math.isfinite(_dao_arg) else dao_detection_fwhm_pixels(
                hdr, configured_fallback=_fb_c
            )
        except (TypeError, ValueError):
            _base_fw = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_c)
    else:
        _base_fw = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_c)
    _dao_n_equiv_used: float | None = None
    _derived_tol: Any = None
    try:
        from photutils.detection import DAOStarFinder  # type: ignore

        _ds = float(dao_threshold_sigma)
        _ds = max(0.5, min(20.0, _ds))
        dao_scale = _dao_auto_binning_factor(*data0.shape)
        data_dao, bfac = _mean_bin2d_for_dao(data0, dao_scale)
        fwhm_eff = max(1.2, _base_fw / float(bfac))
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            Pass2AcceptParams,
            dao_pass2_born_owned_rows,
            dedup_pass1_spatial,
            estimate_star_masked_sky_sigma,
            merge_dao_pass1_pass2_born_owned,
            star_mask_from_gaia_xy,
        )

        _gaia_chip_det = _gaia_chip_xy_from_catalog(cat_df, wcs_obj, wpx=int(wpx), h=int(h))
        _gx_sm = (
            pd.to_numeric(_gaia_chip_det["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
            if len(_gaia_chip_det)
            else np.asarray([], dtype=np.float64)
        )
        _gy_sm = (
            pd.to_numeric(_gaia_chip_det["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
            if len(_gaia_chip_det)
            else np.asarray([], dtype=np.float64)
        )
        _smask = star_mask_from_gaia_xy(_gx_sm, _gy_sm, wpx=int(wpx), h=int(h), fwhm_px=float(_base_fw))
        sky_sig, _sky_med_det = estimate_star_masked_sky_sigma(data0, star_mask=_smask)
        rms_conv, _dao_rel_err = _dao_convolved_background_rms_adu(data_dao, fwhm_px=fwhm_eff)
        sigma_pp_diag = _dao_noise_sigma_adu(arr, bfac=bfac, fallback_std=float(std), data_dao=data_dao)
        std_dao = sky_sig
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            std_dao = rms_conv
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            try:
                std_dao = float(np.nanstd(arr))
            except Exception:  # noqa: BLE001
                std_dao = float("nan")
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            try:
                finite = np.isfinite(arr)
                n_finite = int(np.count_nonzero(finite))
                if n_finite > 0:
                    vals = arr[finite]
                    n_unique = int(len(np.unique(vals)))
                    mn = float(np.nanmin(vals))
                    mx = float(np.nanmax(vals))
                else:
                    n_unique, mn, mx = 0, float("nan"), float("nan")
                print(
                    f"DEBUG std=0: {_nm} n_unique={n_unique} n_finite={n_finite} "
                    f"min={mn:.1f} max={mx:.1f}"
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                nonzero_mask = (arr != 0) & np.isfinite(arr)
                if int(np.count_nonzero(nonzero_mask)) > 100:
                    std_dao = float(np.std(arr[nonzero_mask]))
            except Exception:  # noqa: BLE001
                pass
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(f"WARNING: {_nm} std=0 aj po fallback, preskakujem")
            return pd.DataFrame(), {
                "n_detected": 0,
                "n_detected_dao": 0,
                "n_matched": 0,
                "n_matched_before_mag_limit": 0,
                "catalog_rows": int(len(cat_df)) if cat_df is not None else 0,
                "catalog_match_mode": "full_cone",
                "reason": "std_dao_zero",
            }
        _thr_sigma = float(getattr(_cfg_df, "masterstar_dao_threshold_sigma", _ds))
        if not math.isfinite(_thr_sigma) or _thr_sigma <= 0:
            _thr_sigma = float(_ds)
        _thr = max(_thr_sigma * float(std_dao), 1e-6)
        _dao_n_equiv_used = float(_thr_sigma)
        # Adaptive threshold monitoring: match-rate check runs after first catalog match pass (below).
        try:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(
                f"DEBUG DAO INPUT: {_nm} mean={float(np.nanmean(arr)):.1f} std={float(np.nanstd(arr)):.1f} "
                f"threshold={float(_thr):.1f} sky_sigma={float(std_dao):.2f} n_sigma={float(_thr_sigma):.2f} "
                f"rms_conv_diag={float(rms_conv):.2f} sigma_pp_diag={float(sigma_pp_diag):.2f} "
                f"fwhm={float(fwhm_eff):.2f}"
            )
        except Exception:  # noqa: BLE001
            pass
        finder = DAOStarFinder(
            fwhm=float(fwhm_eff),
            threshold=float(_thr),
            scale_threshold=False,
            n_brightest=None,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(data_dao)
        n_raw_dao = int(len(tbl)) if tbl is not None else 0
        if tbl is not None and len(tbl) > 0:
            tbl = _prefilter_dao_table_brightest(tbl, max(int(max_catalog_rows) * 12, 36_000))
            from astropy.table import Table as _AstropyTable

            tbl = _AstropyTable(tbl, copy=True)
            tbl["vy_dao_pass"] = np.ones(len(tbl), dtype=np.int16)
        n_pass1_dao = int(len(tbl)) if tbl is not None else 0
        try:
            _dedup_px = float(_cfg_df.masterstar_dao_pass1_dedup_px)
        except (TypeError, ValueError):
            _dedup_px = 0.75
        if tbl is not None and len(tbl) > 0:
            tbl = dedup_pass1_spatial(tbl, sep_px=max(0.25, min(2.0, _dedup_px)))
        try:
            _sigma_p2_cfg = float(_cfg_df.masterstar_dao_pass2_sigma)
        except (TypeError, ValueError):
            _sigma_p2_cfg = 4.0
        try:
            _depth_p2 = float(_cfg_df.masterstar_gaia_census_target_depth_g)
        except (TypeError, ValueError):
            _depth_p2 = 15.0
        try:
            _edge_p2 = float(_cfg_df.masterstar_gaia_census_edge_margin_px)
        except (TypeError, ValueError):
            _edge_p2 = 10.0
        _match_r_coarse_px = _catalog_match_radius_px(
            wcs_obj, match_sep_arcsec=float(match_sep_arcsec), wpx=int(wpx), h=int(h)
        )
        _dao_x_p1 = np.asarray([], dtype=np.float64)
        _dao_y_p1 = np.asarray([], dtype=np.float64)
        if tbl is not None and len(tbl) > 0:
            xb = np.asarray(tbl["x_centroid"], dtype=np.float64)
            yb = np.asarray(tbl["y_centroid"], dtype=np.float64)
            _dao_x_p1, _dao_y_p1 = _dao_xy_binned_to_full(xb, yb, int(bfac))
        from dao_gaia_calibration import (  # noqa: PLC0415
            compute_pass1_astrometric_residuals_px,
            derive_tolerances_from_residuals,
            plate_scale_arcsec_per_px_from_wcs_nan,
        )

        _gg_sm = (
            pd.to_numeric(_gaia_chip_det.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64)
            if len(_gaia_chip_det) and "g_mag" in _gaia_chip_det.columns
            else (
                pd.to_numeric(_gaia_chip_det.get("mag"), errors="coerce").to_numpy(dtype=np.float64)
                if len(_gaia_chip_det) and "mag" in _gaia_chip_det.columns
                else np.asarray([], dtype=np.float64)
            )
        )
        _res_dr = compute_pass1_astrometric_residuals_px(
            _dao_x_p1,
            _dao_y_p1,
            _gx_sm,
            _gy_sm,
            coarse_match_px=float(_match_r_coarse_px),
        )
        _derived_tol = derive_tolerances_from_residuals(
            _res_dr,
            np.asarray([], dtype=np.float64),
            fwhm_px=float(max(1.2, _base_fw)),
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px_from_wcs_nan(wcs_obj),
            pass1_sigma=float(_thr_sigma),
            pass2_sigma=float(_sigma_p2_cfg),
            match_k=float(getattr(_cfg_df, "masterstar_dao_match_radius_k", 1.7)),
            centroid_floor_px=float(getattr(_cfg_df, "masterstar_dao_centroid_qa_floor_px", 1.0)),
            centroid_cap_px=float(getattr(_cfg_df, "masterstar_dao_centroid_qa_cap_px", 3.0)),
        )
        _match_r_px_p2 = float(_derived_tol.match_radius_px)
        p2_params = Pass2AcceptParams(
            sigma=max(1.5, min(20.0, float(_sigma_p2_cfg))),
            center_tol_px=max(0.5, min(10.0, float(_derived_tol.pass2_center_tol_px))),
            fwhm_px=float(max(1.2, _base_fw)),
        )
        _pass2_rows, _n_unmatched_gaia, _n_pass2_raw, _amb_p2 = dao_pass2_born_owned_rows(
            data0,
            tbl,
            gaia_chip=_gaia_chip_det,
            bfac=int(bfac),
            fwhm_px=float(max(1.2, _base_fw)),
            pass2_params=p2_params,
            target_depth_g=float(_depth_p2),
            edge_margin_px=float(_edge_p2),
            match_r_px=float(_match_r_px_p2),
            wpx=int(wpx),
            h=int(h),
        )
        tbl = merge_dao_pass1_pass2_born_owned(
            tbl, _pass2_rows, bfac=int(bfac), gaia_chip=_gaia_chip_det
        )
        n_merged_dao = int(len(tbl)) if tbl is not None else 0
        LOGGER.info(
            "[DAO pass 1] %d detections, %d Gaia unmatched",
            int(n_pass1_dao),
            int(_n_unmatched_gaia),
        )
        LOGGER.info(
            "[DAO pass 2] %d additional detections from %d targeted positions",
            int(_n_pass2_raw),
            int(_n_unmatched_gaia),
        )
        LOGGER.info("[DAO total] %d detections after merge", int(n_merged_dao))
    except Exception:  # noqa: BLE001
        tbl = None
        bfac = 1
        n_raw_dao = 0

    sat_limit, sat_limit_src = _effective_saturation_limit(
        hdr, fallback_adu=_fb_sat, equipment_saturate_adu=equipment_saturate_adu
    )
    foot_meta = {
        "catalog_footprint": {
            "center_ra_icrs_deg": float(center.ra.deg),
            "center_dec_icrs_deg": float(center.dec.deg),
            "cone_radius_deg": float(radius_deg),
            "naxis1_px": int(wpx),
            "naxis2_px": int(h),
            "method": "circumscribed_cone_border_sample_plus_margin",
            "reference_catalog": "gaia_local_sqlite",
        },
        "saturation": {
            "effective_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "limit_source": sat_limit_src,
            "plateau_half_inner_px": 1,
            "plateau_rel": 0.996,
            "plateau_min_pixels": 5,
        },
    }
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame(), {
            "n_detected": 0,
            "n_detected_dao": 0,
            "n_matched": 0,
            "n_matched_before_mag_limit": 0,
            "catalog_rows": int(len(cat_df)),
            "catalog_match_mode": "full_cone",
            "n_likely_saturated": 0,
            "n_saturated_from_peak": 0,
            "n_saturated_plateau": 0,
            "n_vsx_in_field": int(len(vsx_df)) if vsx_df is not None else 0,
            "n_exo_hosts_in_field": (
                int(len(exo_df)) if exo_annotation_active and exo_df is not None else 0
            ),
            "n_gaia_variable_in_field": int(len(gaia_variable_df)) if gaia_variable_df is not None else 0,
            "faintest_mag_limit": float(faintest_mag_limit) if faintest_mag_limit is not None else None,
            "n_dropped_fainter_than_limit": 0,
            "field_catalog_cone_csv": str(Path(field_catalog_export_path)) if field_catalog_export_path else None,
            "dao_threshold_sigma": float(dao_threshold_sigma),
            "dao_fwhm_px": float(max(1.2, _base_fw)),
            "dao_detect_binning": int(bfac),
            "match_sep_arcsec_requested": float(match_sep_arcsec),
            "match_sep_arcsec_effective": float(match_sep_arcsec),
            "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "saturate_limit_source": sat_limit_src,
            **foot_meta,
        }

    _fwhm_used = float(max(1.2, _base_fw / float(bfac)))
    _d_h2, _d_w2 = int(data_dao.shape[0]), int(data_dao.shape[1])
    # Stratify brightest DAO sources on a coarse grid over the **full** chip (not radial distance from center).
    _keep2 = _dao_spatial_flux_cap_row_indices(
        tbl, max_n=int(max_catalog_rows), width_px=float(_d_w2), height_px=float(_d_h2)
    )
    tbl = tbl[_keep2]
    tbl.sort("flux")
    tbl = tbl[::-1]
    n_spatial = int(len(tbl))
    log_event(
        f"DAO na snimku: raw={n_raw_dao} (po brightest-prefilter max {max(int(max_catalog_rows) * 12, 36_000):d}) -> "
        f"po priestorovom strope max_n={int(max_catalog_rows)}: {n_spatial} bodov (binning DAO={bfac}x)."
    )
    n = n_spatial
    xb = np.asarray(tbl["x_centroid"], dtype=np.float64)
    yb = np.asarray(tbl["y_centroid"], dtype=np.float64)
    x, y = _dao_xy_binned_to_full(xb, yb, bfac)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    peak_dao = np.asarray(tbl["peak"], dtype=np.float64) if "peak" in tbl.colnames else np.full(n, np.nan)
    if "vy_seed_catalog_id" in tbl.colnames:
        vy_seed_cid = np.asarray(tbl["vy_seed_catalog_id"], dtype=object)
    else:
        vy_seed_cid = np.array([""] * n, dtype=object)
    if "vy_ambiguous_owner" in tbl.colnames:
        vy_amb_owner = np.asarray(tbl["vy_ambiguous_owner"], dtype=bool)
    else:
        vy_amb_owner = np.zeros(n, dtype=bool)
    ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
    det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    vsx_max = float(vsx_match_max_sep_arcsec)
    gvar_max = float(gaia_variable_match_max_sep_arcsec)
    if vsx_df is not None and not vsx_df.empty:
        vxc = SkyCoord(
            ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
            dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
        )
        _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
        vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
    else:
        vsx_hit = np.zeros(n, dtype=bool)
    if gaia_variable_df is not None and not gaia_variable_df.empty:
        gvc = SkyCoord(
            ra=np.asarray(gaia_variable_df["ra_deg"], dtype=float) * u.deg,
            dec=np.asarray(gaia_variable_df["dec_deg"], dtype=float) * u.deg,
        )
        _, sepgv, _ = det_coords.match_to_catalog_sky(gvc)
        gvar_hit = np.asarray(sepgv.arcsec <= gvar_max, dtype=bool)
    else:
        gvar_hit = np.zeros(n, dtype=bool)
    catalog_known_variable = np.asarray(vsx_hit, dtype=bool) | np.asarray(gvar_hit, dtype=bool)
    if exo_annotation_active:
        exo_ann, _exo_warns = _exo_host_annotation_arrays(
            det_coords,
            exo_df if exo_df is not None else pd.DataFrame(),
            exo_max,
            frame_name=frame_name,
        )

    sat_frac = float(saturate_level_fraction)
    sat_frac = min(max(sat_frac, 0.5), 1.0)

    pmax_arr = _box_peaks_at_centroids(arr, x, y)
    _frame_max_adu = float(np.nanmax(arr))
    _empirical_clip_adu = _detect_empirical_clip_level_adu(arr)
    _peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_limit,
        saturate_fraction=sat_frac,
        sky_median_adu=float(med),
        frame_max_adu=_frame_max_adu,
        empirical_clip_adu=_empirical_clip_adu,
    )
    _sat_block = _vectorized_star_saturation_columns(
        arr,
        x,
        y,
        sat_limit=_peak_sat_lim,
        sat_frac=sat_frac,
        peak_dao=peak_dao,
        peak_max_adu=pmax_arr,
    )
    _sat_csv, n_sat_pk, n_sat_pl = _proc_sat_block_for_csv(_sat_block)
    # Prematch peak gate (SNR-GATE-01):
    # - noise scale = sky MAD on pixels <= median (not full-frame sample std / scene variance)
    # - pass-2 recoveries already passed a local annulus test; exempt them from this global cut
    _snr_k = float(prematch_peak_sigma_floor)
    if not math.isfinite(_snr_k):
        _snr_k = 10.0
    # Spodna hranica 0.5 = zhoda s AppConfig / DAO-STARS pre MASTERSTAR; horna 15 = per-frame default k=10 zostane platny.
    _snr_k = min(15.0, max(0.5, _snr_k))
    _sky_med_gate, _sky_sig_gate = sky_mad_sigma_adu(arr)
    if not (math.isfinite(_sky_sig_gate) and float(_sky_sig_gate) > 0):
        _sky_sig_gate = float(std) if np.isfinite(std) else 1.0
    if not math.isfinite(_sky_med_gate):
        _sky_med_gate = float(med)
    _bg_sigma_adu = max(float(_sky_sig_gate), 1.0)
    noise_floor = float(float(_sky_med_gate) + _snr_k * _bg_sigma_adu)
    if "vy_dao_pass" in tbl.colnames:
        _dao_pass = np.asarray(tbl["vy_dao_pass"], dtype=np.int16)
        if int(_dao_pass.size) != int(n):
            # Spatial / other pre-filters may have shortened peak arrays; fall back to pass-1.
            _dao_pass = np.ones(n, dtype=np.int16)
    else:
        _dao_pass = np.ones(n, dtype=np.int16)
    _is_pass2 = _dao_pass == 2
    if bool(prematch_exempt_pass2):
        snr_keep = _is_pass2 | (np.isfinite(pmax_arr) & (pmax_arr > noise_floor))
    else:
        snr_keep = np.isfinite(pmax_arr) & (pmax_arr > noise_floor)
    n_snr = int(np.count_nonzero(snr_keep))
    n_gate_drop = int(np.count_nonzero(~snr_keep))
    n_pass2_kept = int(np.count_nonzero(snr_keep & _is_pass2))
    _exempt_tag = "pass2 exempt" if bool(prematch_exempt_pass2) else "pass2 gated"
    if 0 < n_snr < n:
        x = x[snr_keep]
        y = y[snr_keep]
        flux = flux[snr_keep]
        peak_dao = peak_dao[snr_keep]
        ra_deg = ra_deg[snr_keep]
        dec_deg = dec_deg[snr_keep]
        det_coords = det_coords[snr_keep]
        vsx_hit = vsx_hit[snr_keep]
        gvar_hit = gvar_hit[snr_keep]
        catalog_known_variable = catalog_known_variable[snr_keep]
        if exo_annotation_active:
            exo_ann = _slice_exo_annotation(exo_ann, snr_keep)
        pmax_arr = pmax_arr[snr_keep]
        _sat_block = {k: np.asarray(v)[snr_keep] for k, v in _sat_block.items()}
        _sat_csv, n_sat_pk, n_sat_pl = _proc_sat_block_for_csv(_sat_block)
        vy_seed_cid = vy_seed_cid[snr_keep]
        vy_amb_owner = vy_amb_owner[snr_keep]
        _dao_pass = _dao_pass[snr_keep]
        n = int(n_snr)
        log_event(
            f"DAO po SNR filtri (sky_mad median+{_snr_k:.1f}xsigma; {_exempt_tag}): {n}/{n_spatial} bodov "
            f"(noise_floor~{noise_floor:.1f} ADU sky_sig~{_bg_sigma_adu:.1f}; "
            f"pass2_kept={n_pass2_kept}; dropped={n_gate_drop}; pred matchom s katalogom)."
        )
    elif n_snr == 0:
        log_event(
            f"DAO SNR filter by zahodil vsetko - ponechavam {n_spatial} bodov pred matchom."
        )

    idx_det = np.arange(1, n + 1, dtype=np.int32)
    det_str = np.array([f"DET_{i:04d}" for i in idx_det], dtype=object)
    n_matched = 0
    match_sep_used = max(12.0, float(match_sep_arcsec))
    _match_sep_formula_inputs: dict[str, Any] = {}
    _wcs_refine_iters = 0
    from wcs_invertibility import empty_identity_gate_acc

    _identity_gate_acc = empty_identity_gate_acc()
    if cat_df.empty:
        df_out = pd.DataFrame(
            {
                "name": det_str,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "mag": np.full(n, np.nan, dtype=np.float64),
                "b_v": np.full(n, np.nan, dtype=np.float64),
                "catalog": np.array([""] * n, dtype=object),
                "catalog_id": np.array([""] * n, dtype=object),
                "match_sep_arcsec": np.full(n, np.nan, dtype=np.float64),
                "x": x,
                "y": y,
                "flux": flux,
                "vsx_known_variable": vsx_hit,
                "gaia_dr3_variable_catalog": gvar_hit,
                **(exo_ann if exo_annotation_active else {}),
                **_sat_csv,
            }
        )
    else:
        nc = 0
        cid_c = np.empty(0, dtype=object)
        cat_c = np.empty(0, dtype=object)
        mag_c = np.empty(0, dtype=np.float64)
        bv_c = np.empty(0, dtype=np.float64)
        gn_c = np.empty(0, dtype=np.float64)
        gq_c = np.empty(0, dtype=np.float64)
        gg_c = np.empty(0, dtype=np.float64)
        tree_pack = catalog_kd_pack
        if tree_pack is None and len(cat_df) >= 120:
            tree_pack = build_ucac_catalog_kdtree(cat_df)
        oix_rows: np.ndarray | None
        icomp: np.ndarray
        sepa: np.ndarray
        if tree_pack is not None:
            tr, oix_rows = tree_pack
            icomp, sepa = nearest_sky_nn_kdtree(tr, ra_deg, dec_deg)
        else:
            tr = None
            cat_coords = SkyCoord(
                ra=np.asarray(cat_df["ra_deg"], dtype=float) * u.deg,
                dec=np.asarray(cat_df["dec_deg"], dtype=float) * u.deg,
            )
            idx, sep2d, _ = det_coords.match_to_catalog_sky(cat_coords)
            icomp = np.asarray(idx, dtype=np.int64)
            sepa = np.asarray(sep2d.to_value(u.arcsec), dtype=np.float64)
            oix_rows = None

        def _bind_gaia_catalog_columns() -> None:
            nonlocal nc, cid_c, cat_c, mag_c, bv_c, gn_c, gq_c, gg_c
            nc = int(len(cat_df))
            cid_c = (
                cat_df["catalog_id"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
                if "catalog_id" in cat_df.columns
                else np.array([""] * nc, dtype=object)
            )
            cat_c = (
                cat_df["catalog"].fillna("").astype(str).to_numpy(dtype=object)
                if "catalog" in cat_df.columns
                else np.array([""] * nc, dtype=object)
            )
            mag_c = (
                pd.to_numeric(cat_df["mag"], errors="coerce").to_numpy(dtype=np.float64)
                if "mag" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )
            bv_c = (
                pd.to_numeric(cat_df["b_v"], errors="coerce").to_numpy(dtype=np.float64)
                if "b_v" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )
            gn_c = (
                pd.to_numeric(cat_df["gaia_nss"], errors="coerce").to_numpy(dtype=np.float64)
                if "gaia_nss" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )
            gq_c = (
                pd.to_numeric(cat_df["gaia_qso"], errors="coerce").to_numpy(dtype=np.float64)
                if "gaia_qso" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )
            gg_c = (
                pd.to_numeric(cat_df["gaia_gal"], errors="coerce").to_numpy(dtype=np.float64)
                if "gaia_gal" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )

        _bind_gaia_catalog_columns()

        def _assign_catalog_at_threshold(thr: float) -> tuple[pd.DataFrame, int]:
            thr_f = float(thr)
            sepa_eff = np.asarray(sepa, dtype=np.float64)
            if tr is not None and oix_rows is not None:
                # Greedy 1:1 matching on k-nearest sphere neighbors (avoids many detections sharing one Gaia row).
                ntree = int(getattr(tr, "n", 0))
                _nk = min(48, max(1, ntree))
                det_xyz = _icrs_deg_to_unitxyz(ra_deg, dec_deg)
                dist, idx = tr.query(det_xyz, k=_nk)
                dist = np.asarray(dist, dtype=np.float64)
                idx = np.asarray(idx, dtype=np.int64)
                if dist.ndim == 1:
                    dist = dist.reshape(-1, 1)
                    idx = idx.reshape(-1, 1)
                sep_k = _chord_to_arcsec(dist)
                n_oix = int(len(oix_rows))
                pairs: list[tuple[float, int, int]] = []
                for i in range(n):
                    for kk in range(int(idx.shape[1])):
                        j_comp = int(idx[i, kk])
                        if j_comp < 0 or j_comp >= n_oix:
                            continue
                        s = float(sep_k[i, kk])
                        if not np.isfinite(s) or s > thr_f:
                            continue
                        cr = int(oix_rows[j_comp])
                        if 0 <= cr < nc:
                            pairs.append((s, i, cr))
                pairs.sort(key=lambda t: t[0])
                used_det: set[int] = set()
                used_cat: set[int] = set()
                cat_row = np.full(n, -1, dtype=np.int64)
                sepa_out = np.full(n, np.nan, dtype=np.float64)
                # Born-owned pass2: pre-lock seed catalog_id (no greedy reassignment).
                _cid_to_cr = {
                    str(cid_c[_k]).strip(): _k for _k in range(nc) if str(cid_c[_k]).strip()
                }
                for i in range(n):
                    if int(_dao_pass[i]) != 2:
                        continue
                    sc = str(vy_seed_cid[i] if i < len(vy_seed_cid) else "").strip()
                    if not sc:
                        continue
                    cr_b = _cid_to_cr.get(sc)
                    if cr_b is None or cr_b in used_cat or i in used_det:
                        continue
                    used_det.add(i)
                    used_cat.add(cr_b)
                    cat_row[i] = int(cr_b)
                    sepa_out[i] = 0.0
                for s, i, cr in pairs:
                    if i in used_det or cr in used_cat:
                        continue
                    used_det.add(i)
                    used_cat.add(cr)
                    cat_row[i] = cr
                    sepa_out[i] = s
                matched_l = cat_row >= 0
                sepa_eff = np.where(matched_l, sepa_out, sepa.astype(np.float64))
                finite_sep = np.isfinite(sepa_eff)
            else:
                finite_sep = np.isfinite(sepa)
                cat_row = icomp.astype(np.int64, copy=False)
                cat_row = np.clip(cat_row, 0, max(nc - 1, 0))
                matched_l = finite_sep & (sepa <= thr_f) & (nc > 0)
            n_ma = int(np.count_nonzero(matched_l))
            safe_l = np.clip(np.where(matched_l, cat_row, 0), 0, max(nc - 1, 0))
            cid_sel = cid_c[safe_l]
            cat_sel = cat_c[safe_l]
            cid_st = pd.Series(cid_sel, dtype=object).astype(str).str.strip()
            cat_st = pd.Series(cat_sel, dtype=object).astype(str).str.strip()
            empty_cid = cid_st.eq("").to_numpy()
            cat_lab = np.where(cat_st.ne("").to_numpy(), cat_sel, "CAT")
            name_fb = np.array(
                [f"{cat_lab[i]!s}_{int(idx_det[i]):04d}" for i in range(n)],
                dtype=object,
            )
            name_cand = np.where(empty_cid, name_fb, cid_st.to_numpy())
            name_out = np.where(matched_l, name_cand, det_str)
            mag_out = np.full(n, np.nan, dtype=np.float64)
            bv_out = np.full(n, np.nan, dtype=np.float64)
            gn_out = np.full(n, np.nan, dtype=np.float64)
            gq_out = np.full(n, np.nan, dtype=np.float64)
            gg_out = np.full(n, np.nan, dtype=np.float64)
            mag_out[matched_l] = mag_c[safe_l[matched_l]]
            bv_out[matched_l] = bv_c[safe_l[matched_l]]
            gn_out[matched_l] = gn_c[safe_l[matched_l]]
            gq_out[matched_l] = gq_c[safe_l[matched_l]]
            gg_out[matched_l] = gg_c[safe_l[matched_l]]
            cat_out = np.array([""] * n, dtype=object)
            cid_out = np.array([""] * n, dtype=object)
            cat_out[matched_l] = cat_c[safe_l[matched_l]]
            cid_out[matched_l] = cid_c[safe_l[matched_l]]
            df_l = pd.DataFrame(
                {
                    "name": name_out,
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "mag": mag_out,
                    "b_v": bv_out,
                    "catalog": cat_out,
                    "catalog_id": cid_out,
                    "match_sep_arcsec": np.where(matched_l, sepa_eff, np.nan),
                    "x": x,
                    "y": y,
                    "flux": flux,
                    "vsx_known_variable": vsx_hit,
                    "gaia_dr3_variable_catalog": gvar_hit,
                    **(exo_ann if exo_annotation_active else {}),
                    **_sat_csv,
                }
            )
            return df_l, n_ma

        def _apply_post_match_identity_gate() -> None:
            nonlocal df_out, n_matched, _identity_gate_acc
            try:
                from wcs_invertibility import (
                    accumulate_identity_gate,
                    apply_post_match_identity_gate_df,
                    gaia_radec_map_from_table,
                )

                _fwhm_gate = float(_fwhm_used)
                if not math.isfinite(_fwhm_gate) or _fwhm_gate <= 0:
                    _fwhm_gate = 3.5
                _gmap = gaia_radec_map_from_table(cat_df)
                _det_fb = None
                if len(df_out) == int(len(det_str)):
                    import pandas as _pd

                    _det_fb = _pd.Series(det_str, index=df_out.index)
                df_out, _idc = apply_post_match_identity_gate_df(
                    df_out,
                    wcs_obj,
                    gaia_ra_dec_by_cid=_gmap,
                    fwhm_px=_fwhm_gate,
                    log_fn=log_event,
                    det_fallback_names=_det_fb,
                )
                n_matched = int(
                    df_out.get("catalog_id", pd.Series([""] * len(df_out)))
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .sum()
                )
                _identity_gate_acc = accumulate_identity_gate(_identity_gate_acc, _idc, n_matched)
            except Exception as _idg_exc:  # noqa: BLE001
                log_event(f"post_match_identity_gate skipped: {_idg_exc!s}")

        def _run_full_match_pass() -> None:
            nonlocal df_out, n_matched, match_sep_used, _match_sep_formula_inputs
            from dao_gaia_calibration import (
                catalog_match_radius_d1_arcsec,
                plate_scale_arcsec_per_px_from_wcs_nan,
                solve_rms_px_from_fits_header,
            )

            _ps_match = float(plate_scale_arcsec_per_px_from_wcs_nan(wcs_obj))
            _rms_match = solve_rms_px_from_fits_header(hdr)
            match_sep_used, _d1_inputs = catalog_match_radius_d1_arcsec(
                solve_rms_px=_rms_match,
                fwhm_dao_px=float(_fwhm_used),
                plate_scale_arcsec_per_px=_ps_match,
                floor_arcsec=12.0,
            )
            LOGGER.info(
                "Catalog match: radius = max(12, 3 x FWHM_dao=%.3f px x scale=%.4f arcsec/px) "
                "-> %.2f arcsec (formula=%.2f; solve_rms=%.3f px diagnostic only)",
                float(_d1_inputs["fwhm_dao_px"]),
                float(_d1_inputs["plate_scale_arcsec_per_px"] or float("nan")),
                float(match_sep_used),
                float(_d1_inputs["formula_arcsec"] or float("nan")),
                float(_d1_inputs["solve_rms_px"] or float("nan")),
            )
            df_out, n_matched = _assign_catalog_at_threshold(match_sep_used)
            # After a successful loose initial match, tighten for cleaner final IDs (only if most matches survive).
            _tight_sec = 4.5
            if n_matched >= max(10, int(0.20 * max(1, n))) and float(match_sep_used) > _tight_sec + 1e-9:
                df_tight, n_tight = _assign_catalog_at_threshold(_tight_sec)
                if n_tight >= max(8, int(0.92 * max(1, n_matched))):
                    LOGGER.info(
                        "Catalog match: pociatocny loose match %.2f arcsec -> finalne zuzenie na %.2f arcsec (matches %d -> %d)",
                        float(match_sep_used),
                        _tight_sec,
                        int(n_matched),
                        int(n_tight),
                    )
                    df_out, n_matched, match_sep_used = df_tight, n_tight, _tight_sec
            _apply_post_match_identity_gate()
            _match_sep_formula_inputs = dict(_d1_inputs)

        _run_full_match_pass()
        if len(df_out) == int(n):
            df_out["vy_dao_pass"] = np.asarray(_dao_pass, dtype=np.int16)
            df_out["ambiguous_owner"] = np.asarray(vy_amb_owner, dtype=bool)
        if n >= 8:
            _dao_match_rate = float(n_matched) / float(max(1, n))
            if _dao_match_rate < 0.88:
                LOGGER.warning(
                    "[DAO] Match rate %.1f%% below 88%% threshold, "
                    "consider lowering masterstar_dao_threshold_sigma in config.json",
                    100.0 * _dao_match_rate,
                )
        # Gaia / DAO pixel NN TAN refit when sky match fraction stays below ~95% (fixes offset / scale drift).
        if tr is not None and oix_rows is not None and n >= 12 and float(n_matched) / float(max(1, n)) < 0.95:
            try:
                from vyvar_platesolver import _refine_wcs_tan_nn_gaia

                _target_mf = 0.95
                diag = float(np.hypot(float(wpx), float(h)))
                # Allow cross-chip distances when the initial plate solve is badly offset (Gaia world2pix vs DAO).
                max_px = float(min(0.98 * diag, max(800.0, 0.88 * float(max(wpx, h)))))
                det_order_idx = np.argsort(-np.asarray(flux, dtype=np.float64), kind="stable")
                for _wr in range(10):
                    cat_df_snap = cat_df.copy()
                    ra_cat = pd.to_numeric(cat_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
                    de_cat = pd.to_numeric(cat_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
                    r_now = float(n_matched) / float(max(1, n))
                    if r_now >= _target_mf:
                        break
                    n_before_wcs = int(n_matched)
                    hdr_snapshot = hdr.copy()
                    w_try = WCS(hdr)
                    w_new, _meta_wcs = _refine_wcs_tan_nn_gaia(
                        w_try,
                        xs_det=x,
                        ys_det=y,
                        ra_cat_full_deg=ra_cat,
                        dec_cat_full_deg=de_cat,
                        max_match_px=max_px,
                        min_pairs=10,
                        det_order_idx=det_order_idx,
                    )
                    if w_new is None:
                        w_new, _meta_wcs = _refine_wcs_tan_nn_gaia(
                            w_try,
                            xs_det=x,
                            ys_det=y,
                            ra_cat_full_deg=ra_cat,
                            dec_cat_full_deg=de_cat,
                            max_match_px=max_px,
                            min_pairs=8,
                            det_order_idx=det_order_idx,
                        )
                    if w_new is None:
                        max_px = min(max_px * 1.32, 1.52 * diag)
                        if max_px >= 1.48 * diag:
                            break
                        continue
                    _rms_w = _meta_wcs.get("rms_px")
                    if _rms_w is not None and math.isfinite(float(_rms_w)) and float(_rms_w) > 10.0:
                        LOGGER.info(
                            "Catalog match: WCS refine zamietnuty (rms=%.2fpx > 10) - sirsi pixelovy matching.",
                            float(_rms_w),
                        )
                        max_px = min(max_px * 1.32, 1.52 * diag)
                        if max_px >= 1.48 * diag:
                            break
                        continue
                    _apply_wcs_tan_fragment_to_header(
                        hdr,
                        w_new.to_header(relax=True),
                        f"VYVAR: Gaia/pixel NN WCS refine (match {100.0 * r_now:.1f}%, goal {_target_mf * 100:.0f}%)",
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        wcs_obj = WCS(hdr)
                    center2, radius2 = _effective_field_catalog_cone_radius_deg(
                        wcs_obj, h, wpx, _fov_hint, fits_header=hdr
                    )
                    _mag_lim = (
                        float(faintest_mag_limit)
                        if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit))
                        else None
                    )
                    if _gaia_db_path is not None and Path(_gaia_db_path).is_file():
                        # Wide-field cones already subsume the chip; tangent-plane WCS nudges do not warrant
                        # re-running multi-hundred-k row SQLite queries on every refine iteration (was ~10x per frame).
                        _skip_gaia_rerequery = float(radius2) >= 5.0
                        if _skip_gaia_rerequery:
                            LOGGER.info(
                                "Catalog match: WCS refine - ponechavam existujuci lokalny Gaia vyrez "
                                f"(r={float(radius2):.2f} deg >= 5 deg; bez opatovneho SQL dotazu)."
                            )
                        else:
                            cat_df_new = _catalog_df_cap_brightest_by_mag(
                                _query_gaia_local(
                                    center=center2,
                                    radius_deg=radius2,
                                    gaia_db_path=_gaia_db_path,
                                    max_mag=_mag_lim,
                                    max_rows=int(_cat_cap_eff),
                                ),
                                max_rows=int(_cat_cap_eff),
                            )
                            if len(cat_df_new) < 120:
                                LOGGER.info(
                                    "Catalog match: WCS refine - Gaia re-query < 120 hviezd; refine zruseny."
                                )
                                hdr.clear()
                                hdr.extend(hdr_snapshot.cards)
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", FITSFixedWarning)
                                    wcs_obj = WCS(hdr)
                                ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
                                det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                                if vsx_df is not None and not vsx_df.empty:
                                    vxc = SkyCoord(
                                        ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
                                        dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
                                    )
                                    _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
                                    vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
                                else:
                                    vsx_hit = np.zeros(n, dtype=bool)
                                if gaia_variable_df is not None and not gaia_variable_df.empty:
                                    gvc = SkyCoord(
                                        ra=np.asarray(gaia_variable_df["ra_deg"], dtype=float) * u.deg,
                                        dec=np.asarray(gaia_variable_df["dec_deg"], dtype=float) * u.deg,
                                    )
                                    _, sepgv, _ = det_coords.match_to_catalog_sky(gvc)
                                    gvar_hit = np.asarray(sepgv.arcsec <= gvar_max, dtype=bool)
                                else:
                                    gvar_hit = np.zeros(n, dtype=bool)
                                catalog_known_variable = np.asarray(vsx_hit, dtype=bool) | np.asarray(
                                    gvar_hit, dtype=bool
                                )
                                if exo_annotation_active:
                                    exo_ann, _ = _exo_host_annotation_arrays(
                                        det_coords,
                                        exo_df if exo_df is not None else pd.DataFrame(),
                                        exo_max,
                                        frame_name=frame_name,
                                    )
                                if tree_pack is not None:
                                    tr, oix_rows = tree_pack
                                    icomp, sepa = nearest_sky_nn_kdtree(tr, ra_deg, dec_deg)
                                _run_full_match_pass()
                                break
                            cat_df = cat_df_new
                            _bind_gaia_catalog_columns()
                            tree_pack = build_ucac_catalog_kdtree(cat_df)
                            tr, oix_rows = tree_pack
                    else:
                        LOGGER.info("Catalog match: WCS refine bez noveho Gaia kuzela (gaia_db_path).")
                    ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
                    det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                    if vsx_df is not None and not vsx_df.empty:
                        vxc = SkyCoord(
                            ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
                            dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
                        )
                        _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
                        vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
                    else:
                        vsx_hit = np.zeros(n, dtype=bool)
                    if gaia_variable_df is not None and not gaia_variable_df.empty:
                        gvc = SkyCoord(
                            ra=np.asarray(gaia_variable_df["ra_deg"], dtype=float) * u.deg,
                            dec=np.asarray(gaia_variable_df["dec_deg"], dtype=float) * u.deg,
                        )
                        _, sepgv, _ = det_coords.match_to_catalog_sky(gvc)
                        gvar_hit = np.asarray(sepgv.arcsec <= gvar_max, dtype=bool)
                    else:
                        gvar_hit = np.zeros(n, dtype=bool)
                    catalog_known_variable = np.asarray(vsx_hit, dtype=bool) | np.asarray(gvar_hit, dtype=bool)
                    if exo_annotation_active:
                        exo_ann, _ = _exo_host_annotation_arrays(
                            det_coords,
                            exo_df if exo_df is not None else pd.DataFrame(),
                            exo_max,
                            frame_name=frame_name,
                        )
                    icomp, sepa = nearest_sky_nn_kdtree(tr, ra_deg, dec_deg)
                    _run_full_match_pass()
                    _wcs_refine_iters += 1
                    # Revert only on a large regression (refit can briefly reshuffle pairs).
                    if int(n_matched) < int(0.88 * max(1, n_before_wcs)):
                        hdr.clear()
                        hdr.extend(hdr_snapshot.cards)
                        cat_df = cat_df_snap
                        _bind_gaia_catalog_columns()
                        tree_pack = build_ucac_catalog_kdtree(cat_df) if len(cat_df) >= 120 else None
                        if tree_pack is None:
                            tr = None
                            oix_rows = None
                        else:
                            tr, oix_rows = tree_pack
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FITSFixedWarning)
                            wcs_obj = WCS(hdr)
                        ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
                        det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                        if vsx_df is not None and not vsx_df.empty:
                            vxc = SkyCoord(
                                ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
                                dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
                            )
                            _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
                            vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
                        else:
                            vsx_hit = np.zeros(n, dtype=bool)
                        if gaia_variable_df is not None and not gaia_variable_df.empty:
                            gvc = SkyCoord(
                                ra=np.asarray(gaia_variable_df["ra_deg"], dtype=float) * u.deg,
                                dec=np.asarray(gaia_variable_df["dec_deg"], dtype=float) * u.deg,
                            )
                            _, sepgv, _ = det_coords.match_to_catalog_sky(gvc)
                            gvar_hit = np.asarray(sepgv.arcsec <= gvar_max, dtype=bool)
                        else:
                            gvar_hit = np.zeros(n, dtype=bool)
                        catalog_known_variable = np.asarray(vsx_hit, dtype=bool) | np.asarray(gvar_hit, dtype=bool)
                        if exo_annotation_active:
                            exo_ann, _ = _exo_host_annotation_arrays(
                                det_coords,
                                exo_df if exo_df is not None else pd.DataFrame(),
                                exo_max,
                                frame_name=frame_name,
                            )
                        icomp, sepa = nearest_sky_nn_kdtree(tr, ra_deg, dec_deg)
                        _run_full_match_pass()
                        break
                    r_after = float(n_matched) / float(max(1, n))
                    if r_after < 0.88:
                        max_px = min(max_px * 1.1, 1.48 * diag)
                    else:
                        max_px = min(max(max_px * 0.92, 0.42 * diag), 1.2 * diag)
            except Exception as exc:  # noqa: BLE001
                LOGGER.info("Catalog match: WCS Gaia/pixel refine skipped: %s", exc)
    n_detected_dao = int(n)
    n_matched_before_mag = int(n_matched)
    n_before_mag = len(df_out)
    if (
        len(cat_df) > 0
        and len(df_out) >= 30
        and n_matched_before_mag < max(5, int(0.12 * len(df_out)))
    ):
        try:
            _sep_c = pd.to_numeric(df_out["match_sep_arcsec"], errors="coerce")
            _arr = _sep_c.to_numpy(dtype=np.float64)
            _ok = _arr[np.isfinite(_arr)]
            if len(_ok) >= 20:
                med_nn = float(np.nanmedian(_ok))
                if med_nn > float(match_sep_used) * 1.15:
                    LOGGER.warning(
                        "Catalog match je slaby: %s/%s detekcii v ramci %.2f arcsec; median vzdialenosti k najblizsiemu "
                        "katalogu ~ %.2f arcsec - skus zvacsit 'Max catalog match distance arcsec, overit plate solve (FOV, RA/Dec) "
                        "a lokalna Gaia DR3.%s",
                        n_matched_before_mag,
                        len(df_out),
                        float(match_sep_used),
                        med_nn,
                        wcs_distortion_log_suffix(hdr),
                    )
        except Exception:  # noqa: BLE001
            pass
    if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit)):
        lim = float(faintest_mag_limit)
        mcol = pd.to_numeric(df_out["mag"], errors="coerce")
        df_out = df_out.loc[mcol.isna() | (mcol <= lim)].reset_index(drop=True)
        meta_mag = {
            "faintest_mag_limit": lim,
            "n_dropped_fainter_than_limit": int(n_before_mag - len(df_out)),
        }
    else:
        meta_mag = {"faintest_mag_limit": None, "n_dropped_fainter_than_limit": 0}

    n_sat = int(df_out["likely_saturated"].sum()) if len(df_out) and "likely_saturated" in df_out.columns else 0
    cat_nonempty = (
        df_out["catalog"].fillna("").astype(str).str.strip().ne("")
        if len(df_out) and "catalog" in df_out.columns
        else pd.Series([], dtype=bool)
    )
    n_matched_final = int(cat_nonempty.sum()) if len(df_out) else 0
    _ps_idg: float | None = None
    try:
        from dao_gaia_calibration import plate_scale_arcsec_per_px_from_wcs_nan as _ps_idg_fn

        _ps_try = float(_ps_idg_fn(wcs_obj))
        if math.isfinite(_ps_try) and _ps_try > 0:
            _ps_idg = _ps_try
    except Exception:  # noqa: BLE001
        _ps_idg = None
    meta = {
        "noise_floor_adu": float(noise_floor),
        "sky_median_adu": float(_sky_med_gate),
        "bg_sigma_adu": float(_bg_sigma_adu),
        "bg_sigma_estimator": "sky_mad_le_median",
        "prematch_pass2_exempt": bool(prematch_exempt_pass2),
        "frame_max_adu": float(_frame_max_adu),
        "empirical_clip_adu": _empirical_clip_adu,
        "n_detected_dao_raw": int(n_raw_dao),
        "n_dao_after_spatial_cap": int(n_spatial),
        "n_detected_dao": n_detected_dao,
        "n_detected": int(len(df_out)),
        "n_matched_before_mag_limit": n_matched_before_mag,
        "n_matched": n_matched_final,
        "catalog_rows": int(len(cat_df)),
        "catalog_match_mode": "full_cone",
        "n_likely_saturated": n_sat,
        "n_saturated_from_peak": n_sat_pk,
        "n_saturated_plateau": n_sat_pl,
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": int(len(vsx_df)) if vsx_df is not None else 0,
        "n_exo_hosts_in_field": int(len(exo_df)) if exo_annotation_active and exo_df is not None else 0,
        "n_gaia_variable_in_field": int(len(gaia_variable_df)) if gaia_variable_df is not None else 0,
        **foot_meta,
        "field_catalog_cone_csv": str(Path(field_catalog_export_path)) if field_catalog_export_path else None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_detection_n_equiv": (
            float(_dao_n_equiv_used) if _dao_n_equiv_used is not None and math.isfinite(_dao_n_equiv_used) else None
        ),
        "dao_fwhm_px": _fwhm_used,
        "dao_detect_binning": int(bfac),
        "prematch_peak_sigma_floor": float(_snr_k),
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_sep_used),
        "match_sep_formula_inputs": dict(_match_sep_formula_inputs),
        "wcs_gaia_pixel_refine_iters": int(_wcs_refine_iters),
        "catalog_match_fraction_target": 0.95,
        "catalog_match_fraction_met": (
            bool((float(n_matched_final) / float(max(1, len(df_out)))) >= 0.95) if len(df_out) else True
        ),
        "dao_gaia_derived_tol": (
            _derived_tol.to_dict() if _derived_tol is not None else None
        ),
        "identity_gate": {
            **dict(_identity_gate_acc),
            "fwhm_px": float(_fwhm_used),
            "plate_scale_arcsec_per_px": (
                float(_ps_idg) if _ps_idg is not None and math.isfinite(float(_ps_idg)) else None
            ),
            "fail_threshold_px": float(3.0 * float(_fwhm_used)),
            "fail_threshold_arcsec": (
                float(3.0 * float(_fwhm_used) * float(_ps_idg))
                if _ps_idg is not None and math.isfinite(float(_ps_idg))
                else None
            ),
        },
        **meta_mag,
    }
    _catalog_rows = int(meta.get("catalog_rows", 0))
    if _catalog_rows > 0:
        if "catalog_id" in df_out.columns:
            _cid_u = (
                df_out["catalog_id"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            _n_gaia_detected = int(_cid_u[_cid_u != ""].nunique())
        else:
            _n_gaia_detected = int(n_matched_final)
        _gaia_dao_rate = 100.0 * float(_n_gaia_detected) / float(_catalog_rows)
        LOGGER.info(
            "[DAO] Gaia->DAO completeness (raw): "
            "%d/%d Gaia stars detected (%.1f%%) "
            "| catalog_only (undetected): %d",
            _n_gaia_detected,
            _catalog_rows,
            _gaia_dao_rate,
            _catalog_rows - _n_gaia_detected,
        )
        meta["gaia_dao_completeness_raw_pct"] = round(_gaia_dao_rate, 2)
        meta["n_gaia_detected"] = int(_n_gaia_detected)
        meta["n_gaia_undetected"] = int(_catalog_rows - _n_gaia_detected)
        try:
            _plate_recon = None
            if getattr(wcs_obj, "has_celestial", False):
                try:
                    from astropy.wcs.utils import proj_plane_pixel_scales

                    _plate_recon = float(np.mean(proj_plane_pixel_scales(wcs_obj) * 3600.0))
                except Exception:  # noqa: BLE001
                    pass
            if _gaia_db_path is not None and getattr(wcs_obj, "has_celestial", False):
                _max_mag_recon = (
                    float(faintest_mag_limit)
                    if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit))
                    else 18.0
                )
                _recon = compute_gaia_dao_reconcile(
                    df_out,
                    gaia_db_path=_gaia_db_path,
                    wcs=wcs_obj,
                    naxis1=int(wpx),
                    naxis2=int(h),
                    fwhm_px=float(_fwhm_used),
                    plate_scale_arcsec=_plate_recon,
                    mag_limit=_max_mag_recon,
                    match_sep_arcsec=float(match_sep_used),
                    cone_df=cat_df,
                )
                _md = resolve_effective_match_depth(meta, is_masterstar=False)
                _recon.update(_md)
                meta.update(reconcile_to_pipeline_meta(_recon))
                LOGGER.info(
                    "[DAO] Gaia->DAO reconcile: completeness_50=%.1f%% (matched=%d missed=%d "
                    "off_frame=%d below_limit=%d blended=%d) G_lim_50=%.2f fit=%s",
                    float(meta.get("gaia_dao_completeness_pct") or 0.0),
                    int(meta.get("n_gaia_matched") or 0),
                    int(meta.get("n_gaia_missed") or 0),
                    int(meta.get("n_gaia_off_frame") or 0),
                    int(meta.get("n_gaia_below_limit") or 0),
                    int(meta.get("n_gaia_blended") or 0),
                    float(meta.get("g_lim_50") or meta.get("g_lim_est") or 0.0),
                    str(meta.get("fit_method") or "?"),
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[DAO] Gaia reconcile decomposition failed: %s", exc)
            meta["gaia_dao_completeness_pct"] = round(_gaia_dao_rate, 2)
        _corr = meta.get("gaia_dao_completeness_pct")
        if _corr is not None and float(_corr) < 80.0:
            LOGGER.warning(
                "[DAO] Gaia->DAO corrected completeness LOW: %.1f%% (%d genuinely-missed in-frame)",
                float(_corr),
                int(meta.get("n_gaia_missed") or 0),
            )
    else:
        LOGGER.debug("[DAO] catalog_rows not available - Gaia->DAO skip")
    df_out = _proc_rename_det_names_to_catalog_id(df_out)
    try:
        _gain_ms = float(getattr(_cfg_df, "gain", 1.0) or 1.0)
    except (TypeError, ValueError):
        _gain_ms = 1.0
    if not math.isfinite(_gain_ms) or _gain_ms <= 0:
        _gain_ms = 1.0
    try:
        _rn_ms = float(getattr(_cfg_df, "read_noise", 10.0) or 10.0)
    except (TypeError, ValueError):
        _rn_ms = 10.0
    try:
        _ap_fac = float(getattr(_cfg_df, "aperture_fwhm_factor", 1.9) or 1.9)
    except (TypeError, ValueError):
        _ap_fac = 1.9
    try:
        _ann_in = float(getattr(_cfg_df, "annulus_inner_fwhm", 4.75) or 4.75)
    except (TypeError, ValueError):
        _ann_in = 4.75
    try:
        _ann_out = float(getattr(_cfg_df, "annulus_outer_fwhm", 9.0) or 9.0)
    except (TypeError, ValueError):
        _ann_out = 9.0
    df_out = stamp_masterstar_snr_columns(
        df_out,
        image=arr,
        fwhm_dao_px=float(_fwhm_used),
        bg_sigma_adu=float(_bg_sigma_adu),
        gain=_gain_ms,
        read_noise=_rn_ms,
        aperture_fwhm_factor=_ap_fac,
        annulus_inner_fwhm=_ann_in,
        annulus_outer_fwhm=_ann_out,
    )
    return df_out, meta


def _vyvar_parallel_use_processes() -> bool:
    """CPU-heavy parallel steps default to subprocesses. Set ``VYVAR_PARALLEL_BACKEND=thread`` for threads."""
    v = (os.environ.get("VYVAR_PARALLEL_BACKEND") or "process").strip().lower()
    return v not in ("thread", "threads")


@contextlib.contextmanager
def _vyvar_parallel_pool(max_workers: int):
    if _vyvar_parallel_use_processes():
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            yield ex
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            yield ex


def _icrs_center_radius_from_hdr_data(
    hdr: fits.Header,
    data: Any,
    *,
    plate_solve_fov_deg: float | None = None,
) -> tuple[SkyCoord, float, int, int] | None:
    """ICRS center + cone radius from an in-memory frame (RAM handoff), same semantics as disk scan."""
    import numpy as np

    d = np.asarray(data)
    if d.ndim != 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not w.has_celestial:
            return None
        h_i, wpx_i = int(d.shape[0]), int(d.shape[1])
        c_i, r_i = _effective_field_catalog_cone_radius_deg(
            w, h_i, wpx_i, plate_solve_fov_deg, fits_header=hdr
        )
        return c_i, r_i, wpx_i, h_i
    except Exception:  # noqa: BLE001
        return None


def _export_first_icrs_center_radius(
    files: list[Path],
    *,
    plate_solve_fov_deg: float | None = None,
) -> tuple[SkyCoord, float, int, int] | None:
    import numpy as np

    for fp in files:
        try:
            with fits.open(fp, memmap=False) as h:
                hdr = h[0].header
                d = np.asarray(h[0].data)
                if d.ndim != 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    w = WCS(hdr)
                if not w.has_celestial:
                    continue
                h_i, wpx_i = int(d.shape[0]), int(d.shape[1])
                c_i, r_i = _effective_field_catalog_cone_radius_deg(
                    w, h_i, wpx_i, plate_solve_fov_deg, fits_header=hdr
                )
                return c_i, r_i, wpx_i, h_i
        except Exception:  # noqa: BLE001
            continue
    return None


def _prefetch_export_shared_catalog_for_process_pool(
    *,
    files: list[Path] | None = None,
    reference_hdr_data: tuple[Any, Any] | None = None,
    field_cat_path: Path,
    cat_df: pd.DataFrame | None,
    vsx_df: pd.DataFrame | None,
    gaia_variable_df: pd.DataFrame | None,
    gaia_db_path: Path | None,
    gaia_local_max_mag: float | None,
    export_cat_local: bool,
    plate_solve_fov_deg: float | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, Any]:
    """Load cone + VSX/Gaia + KD-tree in the parent before ``ProcessPoolExecutor`` (workers share no memory)."""
    cr: tuple[SkyCoord, float, int, int] | None = None
    if reference_hdr_data is not None:
        _h0, _d0 = reference_hdr_data
        cr = _icrs_center_radius_from_hdr_data(_h0, _d0, plate_solve_fov_deg=plate_solve_fov_deg)
    if cr is None and files:
        cr = _export_first_icrs_center_radius(files, plate_solve_fov_deg=plate_solve_fov_deg)
    c_df = cat_df
    v_df = vsx_df
    g_df = gaia_variable_df

    if cr is not None:
        c_i, r_i, wpx_i, h_i = cr
        _cfg_pf = AppConfig()
        if c_df is None or getattr(c_df, "empty", True):
            _gaia_db_path = Path(gaia_db_path) if gaia_db_path is not None else None
            if _gaia_db_path is None:
                try:
                    _gp = (_cfg_pf.gaia_db_path or "").strip()
                    if _gp:
                        _gaia_db_path = Path(_gp)
                except Exception:  # noqa: BLE001
                    _gaia_db_path = None
            c_df = _query_gaia_local(
                center=c_i,
                radius_deg=r_i,
                gaia_db_path=_gaia_db_path,
                max_mag=float(gaia_local_max_mag) if gaia_local_max_mag is not None else None,
            )
            if c_df is not None and len(c_df) > 0:
                try:
                    field_cat_path.parent.mkdir(parents=True, exist_ok=True)
                    _vyvar_df_to_csv(c_df, field_cat_path)
                    _write_field_catalog_cone_meta(
                        field_cat_path,
                        center=c_i,
                        radius_deg=float(r_i),
                        naxis1=int(wpx_i),
                        naxis2=int(h_i),
                        plate_solve_fov_deg=plate_solve_fov_deg,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("[PIPE] shared-catalog prefetch write skipped: %s", exc)
                    pass

        if v_df is None or getattr(v_df, "empty", True):
            _vsx_p: Path | None = None
            try:
                _vsp = str(_cfg_pf.vsx_local_db_path or "").strip()
                if _vsp:
                    _vsx_p = Path(_vsp).expanduser().resolve()
            except Exception:  # noqa: BLE001
                _vsx_p = None
            if _vsx_p is not None and _vsx_p.is_file():
                v_df = _query_vsx_local(
                    center=c_i,
                    radius_deg=float(r_i),
                    vsx_db_path=_vsx_p,
                )
            else:
                v_df = pd.DataFrame()
        if g_df is None:
            g_df = pd.DataFrame()

    kd_pack = None
    if c_df is not None and not getattr(c_df, "empty", True) and len(c_df) >= 120:
        kd_pack = build_ucac_catalog_kdtree(c_df)
        if kd_pack is not None:
            LOGGER.info(
                "Per-frame catalog: shared cKDTree for %s cone rows (process pool)",
                len(c_df),
            )
    return c_df, v_df, g_df, kd_pack


_EXPORT_PER_FRAME_WORKER_STATE: dict[str, Any] = {}
_PIXEL_MATCH_DEBUG_LOGGED = False


def _init_export_per_frame_worker(state: dict[str, Any]) -> None:
    global _EXPORT_PER_FRAME_WORKER_STATE
    _EXPORT_PER_FRAME_WORKER_STATE = state
    if not bool(state.get("use_master_fast_path")):
        return
    mpath = str(state.get("masterstar_fits_path") or "").strip()
    if not mpath or not Path(mpath).is_file():
        return
    try:
        with fits.open(mpath, memmap=False) as h:
            w = WCS(h[0].header)
            if w.has_celestial:
                _EXPORT_PER_FRAME_WORKER_STATE["ref_wcs"] = w
            d = h[0].data
            if d is not None and getattr(d, "ndim", 0) == 2:
                _EXPORT_PER_FRAME_WORKER_STATE["masterstar_data_shape"] = (
                    int(d.shape[0]),
                    int(d.shape[1]),
                )
    except Exception:  # noqa: BLE001
        pass


def _airmass_from_altitude_deg(alt_deg: float) -> float:
    """Kasten & Young (1989) airmass from altitude in degrees."""
    alt_rad = math.radians(float(alt_deg))
    return 1.0 / (
        math.sin(alt_rad) + 0.50572 * (float(alt_deg) + 6.07995) ** (-1.6364)
    )


def _compute_airmass_from_altaz(
    hdr: fits.Header,
    cfg: AppConfig | None = None,
    *,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> float:
    """Compute airmass from field center RA/Dec + observer site + mid-exposure JD (AltAz).

    Site is resolved via the unified resolver (per-draft ID_LOCATION -> header
    SITELAT -> flagged config) so airmass uses the same location as BJD/HJD and is
    no longer tied to ``cfg.observer_*`` alone (config-drift trap).
    """
    from time_utils import resolve_observer_location  # noqa: PLC0415

    lat_r, lon_r, alt_r = resolve_observer_location(hdr, db, draft_id, cfg=cfg)
    if lat_r is None or lon_r is None:
        return float("nan")
    lat = float(lat_r)
    lon = float(lon_r)
    alt_m = float(alt_r) if alt_r is not None else 0.0
    if lat == 0.0 and lon == 0.0:
        return float("nan")

    from time_utils import mid_exposure_jd  # noqa: PLC0415

    jd = mid_exposure_jd(hdr)
    if jd is None:
        return float("nan")

    # Safe: airmass from CRVAL1/2 or OBJCTRA/DEC; returns nan if missing - handled downstream.
    ra = _header_float_tu(hdr, "CRVAL1")
    dec = _header_float_tu(hdr, "CRVAL2")
    if ra is None or dec is None:
        if "OBJCTRA" in hdr and "OBJCTDEC" in hdr:
            try:
                from time_utils import _parse_objctradec  # noqa: PLC0415

                ra, dec = _parse_objctradec(str(hdr["OBJCTRA"]), str(hdr["OBJCTDEC"]))
            except Exception:  # noqa: BLE001
                ra, dec = None, None
    if ra is None or dec is None:
        return float("nan")

    try:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord  # noqa: PLC0415
        from astropy.time import Time  # noqa: PLC0415

        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt_m * u.m)
        t = Time(float(jd), format="jd", scale="utc", location=location)
        target = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
        altaz = target.transform_to(AltAz(obstime=t, location=location))
        alt_deg = float(altaz.alt.deg)
        if not math.isfinite(alt_deg) or alt_deg < 5.0 or alt_deg > 90.0:
            return float("nan")
        am = _airmass_from_altitude_deg(alt_deg)
        if 0.9 <= am <= 10.0:
            LOGGER.info(
                "Airmass from AltAz fallback: %.4f (alt=%.2f deg)",
                am,
                alt_deg,
            )
            return round(am, 5)
    except Exception:  # noqa: BLE001
        return float("nan")
    return float("nan")


def _extract_airmass_from_header(
    hdr: fits.Header,
    cfg: AppConfig | None = None,
    *,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> float:
    """Extract airmass from a FITS header for per-frame catalog metadata.

    Priority order:
    1. ``AIRMASS`` / ``AIRMAS`` / ``SECZ`` header keywords when in the physical range.
    2. ``ALT_OBJ`` / ``OBJCTALT`` / ``ALTITUDE`` / ``TELALT`` converted via
       Kasten & Young (1989) (:func:`_airmass_from_altitude_deg`).
    3. Field-center AltAz computed from WCS/object coordinates, observer site, and
       mid-exposure JD (:func:`_compute_airmass_from_altaz`), also using Kasten & Young (1989).

    Returns ``float('nan')`` when no path yields a finite value.
    """
    # Priamy keyword
    for kw in ("AIRMASS", "AIRMAS", "SECZ"):
        val = hdr.get(kw)
        if val is not None:
            try:
                v = float(val)
                if 0.9 <= v <= 10.0:  # fyzikalny rozsah
                    return v
            except (TypeError, ValueError):
                pass

    # Fallback: altitude -> airmass cez aproximaciu Kasten & Young (1989)
    for kw in ("ALT_OBJ", "OBJCTALT", "ALTITUDE", "TELALT"):
        val = hdr.get(kw)
        if val is not None:
            try:
                alt_deg = float(val)
                if 5.0 <= alt_deg <= 90.0:
                    am = _airmass_from_altitude_deg(alt_deg)
                    if 0.9 <= am <= 10.0:
                        return round(am, 5)
            except (TypeError, ValueError):
                pass

    am_altaz = _compute_airmass_from_altaz(hdr, cfg, db=db, draft_id=draft_id)
    if math.isfinite(am_altaz):
        return am_altaz

    return float("nan")


def _cfg_from_export_worker_state(st: dict[str, Any]) -> AppConfig:
    """Rebuild minimal AppConfig observer fields from per-frame worker state dict."""
    c = AppConfig()
    c.observer_lat = float(st.get("observer_lat", 0.0) or 0.0)
    c.observer_lon = float(st.get("observer_lon", 0.0) or 0.0)
    c.observer_alt_m = float(st.get("observer_alt_m", 0.0) or 0.0)
    return c


def _export_per_frame_run_catalog_core(
    base_path: Path,
    hdr: fits.Header,
    data: Any,
    st: dict[str, Any],
) -> dict[str, Any]:
    log_event("DEBUG: per-frame worker entry point called")
    fname = base_path.name
    st["epsf_frame_name"] = fname
    _idx_map = st.get("epsf_frame_index_by_name")
    if isinstance(_idx_map, dict):
        st["epsf_frame_index"] = _idx_map.get(fname)
    debug_pixel_match: dict[str, Any] = {
        "file": fname,
        "use_fast": bool(st.get("use_master_fast_path")),
        "master_cols": None,
        "have_x": None,
        "have_y": None,
        "match_mode": None,
        "plate_scale_arcsec_per_px": None,
        "n_matched": None,
    }
    deferred_writes: list[tuple[str, pd.DataFrame]] = []
    if not _has_valid_wcs(hdr):
        return {
            "file": fname,
            "status": "no_wcs",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [],
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        _ = WCS(hdr)
    h_i, wpx_i = data.shape

    master_tab = st.get("master_tab")
    use_master_fast_path = bool(st.get("use_master_fast_path"))
    master_only_mode = bool(st.get("master_only_mode"))

    use_fast = (
        use_master_fast_path
        and master_tab is not None
        and not getattr(master_tab, "empty", True)
    )
    if master_only_mode and not use_fast:
        return {
            "file": fname,
            "status": "error: master_only_mode requires masterstars_full_match.csv",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [
                f"Per-frame catalog {fname}: master_only_mode requires masterstars_full_match.csv",
            ],
        }

    cat_df = st.get("cat_df")
    vsx_df = st.get("vsx_df")
    gaia_variable_df = st.get("gaia_variable_df")
    kd_pack = st.get("kd_pack")
    export_cat_local = bool(st.get("export_cat_local"))

    df: pd.DataFrame
    meta: dict[str, Any]
    if use_fast:
        try:
            # One-time debug: verify MASTERSTAR pixel columns used for KD-tree matching.
            try:
                if not bool(st.get("_debug_logged_master_xy")) and master_tab is not None:
                    st["_debug_logged_master_xy"] = True
                    cols = [c for c in ("x", "y", "ra_deg", "dec_deg") if c in master_tab.columns]
                    if cols:
                        log_event(
                            "DEBUG: masterstars_full_match.csv sample columns "
                            + ",".join(cols)
                            + ":\n"
                            + master_tab.loc[:, cols].head(5).to_string(index=False)
                        )
            except Exception:  # noqa: BLE001
                pass
            try:
                if master_tab is not None and "catalog_id" in master_tab.columns and "ra_deg" in master_tab.columns:
                    print(
                        "DEBUG SKY-MATCH: "
                        f"master_tab rows={len(master_tab)}, "
                        f"catalog_id notna={int(master_tab['catalog_id'].notna().sum())}, "
                        f"ra_deg notna={int(master_tab['ra_deg'].notna().sum())}"
                    )
            except Exception:  # noqa: BLE001
                pass
            try:
                if master_tab is not None:
                    debug_pixel_match["master_cols"] = list(master_tab.columns[:10])
                    debug_pixel_match["have_x"] = bool("x" in master_tab.columns)
                    debug_pixel_match["have_y"] = bool("y" in master_tab.columns)
            except Exception:  # noqa: BLE001
                pass
            df, meta = detect_stars_match_master_reference(
                data,
                hdr,
                master_tab,
                max_catalog_rows=int(st["max_catalog_rows"]),
                match_sep_arcsec=float(st["catalog_match_max_sep_arcsec"]),
                saturate_level_fraction=float(st["saturate_level_fraction"]),
                faintest_mag_limit=st.get("faintest_mag_limit"),
                dao_threshold_sigma=float(st["dao_threshold_sigma"]),
                dao_fwhm_px=float(st.get("dao_fwhm_px", 2.5)),
                equipment_saturate_adu=st.get("equipment_saturate_adu"),
                frame_name=fname,
                sat_diag_ctx=_sat_ctx_from_worker(st),
                raw_data=_load_raw_for_frame(st, fname),
                raw_hdr=_load_raw_hdr_for_frame(st, fname),
                ref_ra_deg=st.get("sat_diag_ref_ra"),
                ref_dec_deg=st.get("sat_diag_ref_dec"),
                drift_ref_catalog_id=st.get("sat_diag_ref_catalog_id"),
            )
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0373] detect_stars_and_match_catalog exception returns status error with empty csv for that f...: %s', exc)
            return {
                "file": fname,
                "status": f"error: {exc}",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [f"Per-frame catalog {fname}: {exc}"],
                "debug_pixel_match": debug_pixel_match,
            }
    else:
        if master_only_mode:
            return {
                "file": fname,
                "status": "error: master_only_mode fallback to Gaia cone is disabled",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [
                    f"Per-frame catalog {fname}: master_only_mode fallback to Gaia cone is disabled",
                ],
            }
        try:
            df, meta = detect_stars_and_match_catalog(
                data,
                hdr,
                max_catalog_rows=int(st["max_catalog_rows"]),
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                match_sep_arcsec=float(st["catalog_match_max_sep_arcsec"]),
                saturate_level_fraction=float(st["saturate_level_fraction"]),
                faintest_mag_limit=st.get("faintest_mag_limit"),
                field_catalog_export_path=None,
                dao_threshold_sigma=float(st["dao_threshold_sigma"]),
                dao_fwhm_px=float(st.get("dao_fwhm_px", 2.5)),
                equipment_saturate_adu=st.get("equipment_saturate_adu"),
                catalog_local_gaia_only=export_cat_local,
                catalog_kd_pack=kd_pack,
                plate_solve_fov_deg=st.get("plate_solve_fov_deg"),
                fov_database_path=st.get("database_path"),
                fov_equipment_id=int(st["equipment_id"]) if st.get("equipment_id") is not None else None,
                fov_draft_id=int(st["draft_id"]) if st.get("draft_id") is not None else None,
                frame_name=fname,
            )
        except Exception as exc:  # noqa: BLE001
            # EXC-0374: T1 -- [SILENT-DROP] MASTERSTAR zone/is_usable/bp_rp merge failure leaves per-frame proc rows ... (EXCEPT-BULK 2026-07-08)
            return {
                "file": fname,
                "status": f"error: {exc}",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [f"Per-frame catalog {fname}: {exc}"],
                "debug_pixel_match": debug_pixel_match,
            }

    df = _apply_exo_host_columns_to_proc_df(df, hdr, (h_i, wpx_i), st, frame_name=fname)

    _before_dao = len(df)
    df = _proc_drop_unmatched_dao_rows(df)
    LOGGER.debug("[TODO-13] catalog-only pre-filter (detect): %d -> %d rows", _before_dao, len(df))

    # --- Join MASTERSTAR annotations into per-frame catalog (via catalog_id) ---
    # For matched rows (catalog_id non-empty), bring stable MASTERSTAR columns like zone/is_usable/bp_rp/etc.
    try:
        if master_tab is not None and "catalog_id" in df.columns and "catalog_id" in master_tab.columns:
            _JOIN_COLS = [
                "zone",
                "is_saturated",
                "is_usable",
                "bp_rp",
                "phot_g_mean_mag",
                "catalog_mag",
                "vy_identity_gate",
                "gaia_dao_resid_px",
                "edge_safe_10px",
                "snr50_ok",
                "noise_floor_adu",
                "saturate_limit_adu_85pct",
                "source_type",
            ]
            join_cols = [c for c in _JOIN_COLS if c in master_tab.columns]
            if join_cols:
                master_lookup = master_tab[["catalog_id"] + join_cols].copy()
                master_lookup["catalog_id"] = (
                    master_lookup["catalog_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
                )
                master_lookup = master_lookup.dropna(subset=["catalog_id"])
                master_lookup = master_lookup.drop_duplicates(subset=["catalog_id"], keep="first")

                df = df.copy()
                df["catalog_id"] = df["catalog_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

                # Avoid column collisions; MASTERSTAR values should win for these columns.
                df = df.drop(columns=[c for c in join_cols if c in df.columns], errors="ignore")
                df = df.merge(master_lookup, on="catalog_id", how="left")
    except Exception:  # noqa: BLE001
        pass

    # FORCED-PHOT-01: inject force-eligible MASTERSTAR members missing from DAO match.
    # DAO path above is unchanged (n_raw_dao / match counts preserved). Forced rows are
    # measured at locked MASTERSTAR (x,y) with bounded peak refine.
    try:
        if master_tab is not None and bool(st.get("forced_photometry_enabled", True)):
            from forced_photometry import inject_forced_masterstar_rows  # noqa: PLC0415

            _force_ids = st.get("forced_photometry_catalog_ids")
            _force_set = (
                {str(x).strip() for x in _force_ids if str(x).strip()}
                if _force_ids
                else None
            )
            _fwhm_fp = float(st.get("dao_fwhm_px", 2.5) or 2.5)
            _bound = float(st.get("forced_photometry_centroid_bound_fwhm", 2.5) or 2.5)
            _margin = float(st.get("forced_photometry_margin_px", 0.0) or 0.0)
            df, _fp_meta = inject_forced_masterstar_rows(
                df,
                master_tab,
                image=np.asarray(data) if data is not None else None,
                fwhm_px=_fwhm_fp,
                centroid_bound_fwhm=_bound,
                margin_px=_margin,
                force_ids=_force_set,
            )
            try:
                meta["forced_photometry"] = dict(_fp_meta)
                debug_pixel_match["forced_photometry"] = dict(_fp_meta)
            except Exception:  # noqa: BLE001
                pass
            LOGGER.debug(
                "[FORCED-PHOT] %s injected=%s geometry_miss=%s",
                fname,
                _fp_meta.get("n_injected"),
                _fp_meta.get("n_geometry_miss"),
            )
    except Exception as _fp_exc:  # noqa: BLE001
        LOGGER.error("[FORCED-PHOT] inject failed (DAO path kept): %s", _fp_exc)

    try:
        debug_pixel_match["match_mode"] = meta.get("catalog_match_mode")
        debug_pixel_match["plate_scale_arcsec_per_px"] = meta.get("plate_scale_arcsec_per_px")
        debug_pixel_match["n_matched"] = meta.get("n_matched")
    except Exception:  # noqa: BLE001
        pass

    _run_aperture = bool(st.get("_run_aperture", True))
    _run_epsf = bool(st.get("_run_epsf", False))
    if _run_aperture:
        st["current_frame_name"] = fname
        df = _apply_aperture_catalog_enhancements_from_st(df, data, hdr, st)
    _ps_dir = str(st.get("platesolve_dir") or "").strip()
    _psf_on = bool(st.get("psf_photometry_enabled", False))
    _epsf_ids = (
        _epsf_fit_catalog_ids(Path(_ps_dir), psf_photometry_enabled=_psf_on)
        if _ps_dir
        else None
    )
    LOGGER.info(
        "[ePSF] _export_per_frame_run_catalog_core %s: platesolve_dir=%r n_psf_ids=%s lc_set=%s",
        fname,
        _ps_dir or None,
        len(_epsf_ids) if _epsf_ids is not None else "ALL",
        _psf_on,
    )
    if not _run_aperture and _run_epsf:
        # PSF-only mode: psf_flux promoted to primary.
        pass
    df = _fill_psf_catalog_columns(df, data, hdr, st, target_ids=_epsf_ids)

    # --- Time columns (JD / HJD / BJD) ---
    _db_tc = None
    try:
        from time_utils import compute_time_columns

        _dbp = str(st.get("database_path") or "").strip()
        _did_tc = st.get("draft_id")
        if _dbp and _did_tc is not None:
            try:
                _db_tc = VyvarDatabase(Path(_dbp))
            except Exception:  # noqa: BLE001
                _db_tc = None
        _geo_cfg = _cfg_from_export_worker_state(st)
        _time_cols = compute_time_columns(
            hdr,
            db=_db_tc,
            draft_id=int(_did_tc) if _did_tc is not None else None,
            cfg=_geo_cfg,
        )
        _tk = ("jd_mid", "hjd_mid", "bjd_tdb_mid")
        _cols_base = list(df.columns)
        _anchors = [c for c in ("jd", "inspection_jd") if c in _cols_base]
        if _anchors:
            _pos = max(_cols_base.index(c) for c in _anchors) + 1
        else:
            _flux_first = next((c for c in _cols_base if c in ("dao_flux", "flux")), None)
            _pos = _cols_base.index(_flux_first) if _flux_first is not None else len(_cols_base)
        for _i, _nm in enumerate(_tk):
            df.insert(_pos + _i, _nm, _time_cols[_nm])

        # Airmass - rovnaka hodnota pre vsetky hviezdy v snimke (frame-level)
        _am_val = _extract_airmass_from_header(
            hdr,
            cfg=_geo_cfg,
            db=_db_tc,
            draft_id=int(_did_tc) if _did_tc is not None else None,
        )
        if "airmass" not in df.columns:
            _am_insert_pos = _pos + len(_tk)
            df.insert(_am_insert_pos, "airmass", _am_val)
    except Exception as _tc_exc:  # noqa: BLE001
        log_event(f"Time columns skipped: {_tc_exc}")
        for _tc in ("jd_mid", "hjd_mid", "bjd_tdb_mid"):
            if _tc not in df.columns:
                df[_tc] = None
        if "airmass" not in df.columns:
            df["airmass"] = float("nan")
    finally:
        if _db_tc is not None:
            try:
                _db_tc.conn.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)

    df2 = df.copy()
    df2.insert(0, "source_file", fname)
    _cmm_export = str(meta.get("catalog_match_mode") or "").strip()
    if _cmm_export:
        df2["catalog_match_mode"] = _cmm_export

    _before_cat = len(df2)
    df2 = _proc_catalog_keep_matched_rows_only(df2)
    LOGGER.debug("[TODO-13] catalog-only filter: %d -> %d rows", _before_cat, len(df2))

    # FORCED-PHOT / DRAFT-514-TRIAGE A2: multiple DAO detections can match the same
    # Gaia id; dedupe before write. Alternate export path already called this;
    # production ``_export_per_frame_run_catalog_core`` did not (root cause of
    # 127 duplicate rows in proc_BO_CVn_Light_001 on draft 514).
    _before_dedupe = len(df2)
    df2 = _proc_deduplicate_matched_catalog_rows(df2)
    if len(df2) != _before_dedupe:
        LOGGER.debug(
            "[PROC] dedupe catalog_id: %d -> %d rows",
            _before_dedupe,
            len(df2),
        )

    csv_paths: list[str] = []
    write_sidecar = bool(st.get("write_sidecar_csv_next_to_fits"))
    mirror_flat = bool(st.get("mirror_flat_platesolve_folder"))
    defer = bool(st.get("defer_disk_writes"))
    out_flat = Path(str(st.get("out_flat") or "."))

    if write_sidecar:
        sidecar = proc_csv_path_for_aligned_fits(base_path)
        if defer:
            deferred_writes.append((str(sidecar), df2.copy()))
        else:
            _vyvar_df_to_csv(df2, sidecar)
        csv_paths.append(str(sidecar))

    if mirror_flat:
        stem = Path(fname).stem
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:120]
        flat_path = out_flat / f"{safe}_catalog.csv"
        if defer:
            deferred_writes.append((str(flat_path), df2.copy()))
        else:
            _vyvar_df_to_csv(df2, flat_path)
        csv_paths.append(str(flat_path))

    primary_csv = csv_paths[0] if csv_paths else ""
    _psf_rec = st.get("_psf_frame_record")
    return {
        "file": fname,
        "status": "ok",
        "csv": primary_csv,
        "csv_paths": ";".join(csv_paths),
        "n_detected": meta.get("n_detected"),
        "n_matched": meta.get("n_matched"),
        "catalog_match_mode": ("master_reference_locked" if master_only_mode else meta.get("catalog_match_mode", "full_cone")),
        "deferred_writes": deferred_writes,
        "infolog_messages": [],
        "debug_pixel_match": debug_pixel_match,
        "psf_frame_record": dict(_psf_rec) if isinstance(_psf_rec, dict) else None,
    }


def _export_per_frame_disk_worker_task(fp_str: str) -> dict[str, Any]:
    print("DEBUG: REAL WORKER CALLED")
    import numpy as np

    st = _EXPORT_PER_FRAME_WORKER_STATE
    fp = Path(fp_str)
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fp.name,
            "status": f"read_error: {exc}",
            "csv": "",
            "deferred_writes": [],
            "infolog_messages": [f"Per-frame catalog {fp.name}: read_error: {exc}"],
        }
    return _export_per_frame_run_catalog_core(fp, hdr, data, st)


def _export_per_frame_ram_worker_task(
    packed: tuple[str, bytes, bytes, int, int],
) -> dict[str, Any]:
    """Picklable worker: (fits_path_str, pickled_fits_Header, float32_bytes, ny, nx)."""
    print("DEBUG: REAL WORKER CALLED")
    import numpy as np

    st = _EXPORT_PER_FRAME_WORKER_STATE
    base_str, hdr_pkl, raw, ny, nx = packed
    fname = Path(base_str).name
    empty_def: list[tuple[str, pd.DataFrame]] = []
    try:
        hdr = pickle.loads(hdr_pkl)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fname,
            "status": f"header_error: {exc}",
            "csv": "",
            "deferred_writes": empty_def,
            "infolog_messages": [f"Per-frame catalog {fname}: header_error: {exc}"],
        }
    try:
        data = np.frombuffer(raw, dtype=np.float32, count=ny * nx).reshape((ny, nx)).copy()
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fname,
            "status": f"buffer_error: {exc}",
            "csv": "",
            "deferred_writes": empty_def,
            "infolog_messages": [f"Per-frame catalog {fname}: buffer_error: {exc}"],
        }
    return _export_per_frame_run_catalog_core(Path(base_str), hdr, data, st)


def _finalize_hybrid_bkg_fallback_sidecar(
    proc_dir: Path,
    *,
    err_background_mode: str,
    write_sidecar: bool,
    gain: float,
    read_noise: float,
    setup_label: str,
) -> dict[str, Any]:
    """Post-export hybrid Howell fallback scaling when sidecar proc CSVs exist."""
    _ = err_background_mode  # CONSOLIDATE-01D: empirical is the only policy
    if not write_sidecar:
        return {}
    try:
        from photometry_core import finalize_hybrid_bkg_fallback_proc_dir

        return finalize_hybrid_bkg_fallback_proc_dir(
            proc_dir,
            gain=float(gain),
            read_noise=float(read_noise),
            setup_label=str(setup_label),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHOT] hybrid bkg fallback finalize skipped: %s", exc)
        return {}


def export_per_frame_catalogs(
    *,
    frames_root: Path,
    platesolve_dir: Path,
    max_catalog_rows: int = 12000,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    write_sidecar_csv_next_to_fits: bool = True,
    mirror_flat_platesolve_folder: bool = False,
    progress_cb: "callable | None" = None,
    masterstars_csv: Path | str | None = None,
    masterstar_fits: Path | str | None = None,
    use_master_fast_path: bool = True,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    aligned_ram: "Sequence[tuple[str, fits.Header, Any]] | None" = None,
    aligned_target_dir: Path | str | None = None,
    defer_disk_writes: bool = False,
    app_config: AppConfig | None = None,
    plate_solve_fov_deg: float | None = None,
    master_dark_path: Path | str | None = None,
    draft_id: int | None = None,
    equipment_id: int | None = None,
    full_catalog_export: bool = False,
) -> dict[str, Any]:
    """For each FITS under ``frames_root`` with WCS: DAO + catalog table, write one CSV per frame.

    **Optional fast path:** if ``use_master_fast_path=True`` and ``masterstars_csv`` + ``masterstar_fits`` match
    the frame (same WCS and **same array shape** as MASTERSTAR), exposures are matched only to
    ``masterstars.csv`` sky positions (faster; on chip edges NN distances can exceed the match threshold).
    ``astrometry_align_and_build_masterstar`` defaults this to **off** unless ``VYVAR_PER_FRAME_MASTER_FAST=1``.

    **Fallback:** if paths are missing or WCS differs, uses one shared cone (see ``field_catalog_cone.csv``)
    and full ``detect_stars_and_match_catalog`` per frame (local Gaia).
    A sidecar ``field_catalog_cone_meta.json`` records the angular cone radius used; if a **cropped** MASTERSTAR
    built a too-small cone, full-chip frames automatically trigger a refetch instead of reusing the stale CSV.

    ``<platesolve_dir>/per_frame_catalog_index.csv`` lists every file and CSV path.

    **Performance:** each frame still runs DAO + catalog match + disk CSV write (dominant cost for many lights).
    Parallelism: jednotny pocet z ``app_config`` / env (``VYVAR_PARALLEL_WORKERS`` alebo legacy env, pozri
    :func:`_vyvar_parallel_worker_count`). When ``>1``, uses ``ProcessPoolExecutor`` (``spawn``); the
    parent prefetches Gaia cone. Worker count is capped using ``psutil`` and
    ``per_frame_mp_reserve_ram_gb``. RAM handoff ``aligned_ram`` uses the same process pool with serialized
    headers + float32 pixels. Lower ``max_catalog_rows`` in the UI to reduce DAO work per file.

    **RAM handoff:** pass ``aligned_ram`` as ``(filename, header, ndarray)`` tuples plus ``aligned_target_dir`` to
    run catalog matching **without** re-reading aligned FITS from disk. With ``defer_disk_writes=True``, sidecar
    CSV (and optional flat mirror) are returned in ``deferred_csv_writes`` for the caller to flush after FITS.

    ``master_dark_path``: optional CalibrationLibrary master dark; enables ``*_dark_bpm.json`` column flags when present.
    """
    import numpy as np

    _cfg_ap = app_config or AppConfig()
    _md_bpm_str = ""
    if master_dark_path is not None and str(master_dark_path).strip():
        _mp = Path(str(master_dark_path))
        if _mp.is_file():
            _md_bpm_str = str(_mp.resolve())

    ps = Path(platesolve_dir) if platesolve_dir is not None else None
    _qc_fwhm_by_name: dict[str, float] = {}
    _fwhm_night_median_px: float | None = None
    try:
        from aperture_policy import load_qc_fwhm_map, normalize_aperture_policy_mode  # noqa: PLC0415

        _qc_root = None
        if ps is not None:
            try:
                _qc_root = Path(ps).resolve().parents[1]
            except IndexError:
                _qc_root = None
        _qc_csv = find_qc_metrics_csv(_qc_root) if _qc_root is not None else None
        _qc_fwhm_by_name, _fwhm_night_median_px = load_qc_fwhm_map(_qc_csv)
        _ap_policy_mode = normalize_aperture_policy_mode(
            getattr(_cfg_ap, "aperture_policy_mode", "f_fixed_night")
        )
    except Exception as _qc_exc:  # noqa: BLE001
        LOGGER.warning("[APERTURE-01] QC FWHM map not loaded: %s", _qc_exc)
        _ap_policy_mode = "f_fixed_night"

    _ap_st: dict[str, Any] = {
        "aperture_photometry_enabled": bool(_cfg_ap.aperture_photometry_enabled),
        "aperture_fwhm_factor": float(_cfg_ap.aperture_fwhm_factor),
        "annulus_inner_fwhm": float(_cfg_ap.annulus_inner_fwhm),
        "annulus_outer_fwhm": float(_cfg_ap.annulus_outer_fwhm),
        "nonlinearity_peak_percentile": float(_cfg_ap.nonlinearity_peak_percentile),
        "nonlinearity_fwhm_ratio": float(_cfg_ap.nonlinearity_fwhm_ratio),
        "bpm_dark_mad_sigma": float(_cfg_ap.bpm_dark_mad_sigma),
        "master_dark_path": _md_bpm_str,
        "database_path": str(Path(_cfg_ap.database_path).resolve()),
        "draft_id": int(draft_id) if draft_id is not None else None,
        "equipment_id": int(equipment_id) if equipment_id is not None else None,
        "aperture_correction_enabled": bool(_cfg_ap.aperture_correction_enabled),
        "aperture_fwhm_factor_small": float(_cfg_ap.aperture_snr_sizing.get("small", 1.5)),
        "aperture_fwhm_factor_large": float(_cfg_ap.aperture_snr_sizing.get("large", 4.0)),
        "aperture_policy_mode": str(_ap_policy_mode),
        "fwhm_night_median_px": _fwhm_night_median_px,
        "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
        "platesolve_dir": str(ps.resolve()) if ps is not None else "",
        "cog_aperture_correction_enabled": bool(_cfg_ap.cog_aperture_correction_enabled),
        "cog_ref_fwhm": float(_cfg_ap.cog_ref_fwhm),
        "cog_min_stars": int(_cfg_ap.cog_min_stars),
        "cog_isolation_fwhm": float(_cfg_ap.cog_isolation_fwhm),
        "cog_snr_min": float(_cfg_ap.cog_snr_min),
        "cog_sat_frac": float(_cfg_ap.cog_sat_frac),
        "cog_ladder_step_px": float(_cfg_ap.cog_ladder_step_px),
        "cog_ladder_step_fwhm": getattr(_cfg_ap, "cog_ladder_step_fwhm", None),
        "cog_ac_factor_max": float(_cfg_ap.cog_ac_factor_max),
        "gain": float(_cfg_ap.gain),
        "read_noise": float(_cfg_ap.read_noise),
        "err_background_mode": "empirical",
        "err_empty_apertures_n": int(_cfg_ap.err_empty_apertures_n),
        "err_empty_apertures_min": int(_cfg_ap.err_empty_apertures_min),
    }

    use_ram_inputs = aligned_ram is not None
    if use_ram_inputs and aligned_target_dir is None:
        raise ValueError("export_per_frame_catalogs: aligned_target_dir is required when aligned_ram is set")

    if (
        not write_sidecar_csv_next_to_fits
        and not mirror_flat_platesolve_folder
        and not defer_disk_writes
    ):
        write_sidecar_csv_next_to_fits = True

    root = Path(aligned_target_dir) if use_ram_inputs else Path(frames_root)
    if ps is None:
        ps = Path(platesolve_dir)
    _ap_st.update(_export_catalog_psf_st_fields(_cfg_ap, ps))
    _epsf_science_meta: dict[str, Any] | None = None
    if bool(_cfg_ap.psf_photometry_enabled) and bool(_ap_st.get("_run_epsf")):
        from epsf_science_set import build_epsf_science_set

        _sci = build_epsf_science_set(ps)
        if not _sci.catalog_ids:
            raise ValueError(
                "ePSF science set is empty"
                + (f": {_sci.empty_reason}" if _sci.empty_reason else "")
                + "; refusing silent fallback to full LC pool."
            )
        _epsf_science_meta = _sci.to_meta_dict()
        _ap_st["epsf_science_set_meta"] = _epsf_science_meta
    out_flat = ps / "per_frame_catalogs"
    if mirror_flat_platesolve_folder:
        out_flat.mkdir(parents=True, exist_ok=True)

    work_ram: list[tuple[str, fits.Header, Any]] | None = None
    if use_ram_inputs:
        root.mkdir(parents=True, exist_ok=True)
        work_ram = sorted(list(aligned_ram), key=lambda t: t[0])
        files = [root / name for name, _, _ in work_ram]
    else:
        if full_catalog_export:
            files = sorted(_iter_fits_recursive(root))
        else:
            from epsf_frame_accounting import list_epsf_science_light_fits

            files = list_epsf_science_light_fits(root)

    _frame_index_by_name = {Path(f).name: i for i, f in enumerate(files)}
    _ap_st["epsf_frame_index_by_name"] = _frame_index_by_name

    if not files:
        return {
            "written": 0,
            "per_frame_dir": str(root),
            "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else "none",
            "index_csv": "",
            "frames": [],
            "deferred_csv_writes": [],
        }

    master_only_mode = bool(use_master_fast_path)
    field_cat_path = ps / "field_catalog_cone.csv"
    meta_path = _field_catalog_cone_meta_path(field_cat_path)
    cat_df: pd.DataFrame | None = None

    _pfov_res: float | None = None
    try:
        _pf0 = float(plate_solve_fov_deg) if plate_solve_fov_deg is not None else float("nan")
        if math.isfinite(_pf0) and _pf0 > 0:
            _pfov_res = _pf0
    except (TypeError, ValueError):
        _pfov_res = None
    if _pfov_res is None and files:
        try:
            _rf0 = files[0]
            with fits.open(_rf0, memmap=False) as _h0:
                _hd0 = _h0[0].header.copy()
                _ar0 = np.asarray(_h0[0].data)
            if _ar0.ndim == 2:
                _pfov_res = resolve_plate_solve_fov_deg_hint(
                    _hd0,
                    int(_ar0.shape[0]),
                    int(_ar0.shape[1]),
                    database_path=_cfg_ap.database_path,
                    equipment_id=int(equipment_id) if equipment_id is not None else None,
                    draft_id=int(draft_id) if draft_id is not None else None,
                )
        except Exception:  # noqa: BLE001
            _pfov_res = None
    if _pfov_res is None:
        try:
            _pfov_res = float(_cfg_ap.plate_solve_fov_deg)
        except Exception:  # noqa: BLE001
            _pfov_res = None

    r_need_deg: float | None = None
    try:
        _ref_fp = files[0]
        with fits.open(_ref_fp, memmap=False) as _hdu0:
            _hdr0 = _hdu0[0].header
            _dat0 = np.asarray(_hdu0[0].data)
            if _dat0.ndim == 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    _w0 = WCS(_hdr0)
                if _w0.has_celestial:
                    _h0, _w0px = _dat0.shape
                    _, r_need_deg = _effective_field_catalog_cone_radius_deg(
                        _w0, _h0, _w0px, _pfov_res, fits_header=_hdr0
                    )
    except Exception:  # noqa: BLE001
        r_need_deg = None

    if (not master_only_mode) and r_need_deg is not None:
        _invalidate_field_catalog_cone_cache_if_needed(
            field_cat_path,
            plate_solve_fov_deg=_pfov_res,
            effective_radius_deg=float(r_need_deg),
        )

    if (not master_only_mode) and field_cat_path.is_file():
        try:
            _hdr_fc = pd.read_csv(field_cat_path, nrows=0)
            _dtype_fc: dict[str, type] = {}
            if "catalog_id" in _hdr_fc.columns:
                _dtype_fc["catalog_id"] = str
            _kw_fc: dict[str, Any] = {}
            if _dtype_fc:
                _kw_fc["dtype"] = _dtype_fc
            _cdf = pd.read_csv(field_cat_path, **_kw_fc)
            if len(_cdf) > 0:
                _reuse = True
                if r_need_deg is not None and meta_path.is_file():
                    try:
                        _meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        r_stored = float(_meta.get("cone_radius_deg", 0.0))
                        slack_deg = 45.0 / 3600.0
                        if r_stored <= 0.0 or r_need_deg > r_stored * 1.02 + slack_deg:
                            _reuse = False
                            LOGGER.info(
                                "Per-frame catalog: ignoring cached %s (full chip needs cone_radius_deg~%.6f, "
                                "cached %.6f from %s) - fetching larger Gaia cone",
                                field_cat_path,
                                r_need_deg,
                                r_stored,
                                meta_path.name,
                            )
                    except Exception:  # noqa: BLE001
                        pass
                if _reuse:
                    cat_df = _cdf
                    LOGGER.info(
                        "Per-frame catalog: reusing %s (%s rows) - skipping duplicate cone query",
                        field_cat_path,
                        len(cat_df),
                    )
        except Exception:  # noqa: BLE001
            cat_df = None

    kd_cell: list[tuple[Any, Any]] = []

    def _refresh_shared_cat_kdtree() -> None:
        if kd_cell or cat_df is None or getattr(cat_df, "empty", True) or len(cat_df) < 120:
            return
        bk = build_ucac_catalog_kdtree(cat_df)
        if bk is not None:
            kd_cell.append(bk)
            LOGGER.info(
                "Per-frame catalog: shared cKDTree for %s cone rows (fast sky matching)",
                len(cat_df),
            )

    vsx_df: pd.DataFrame | None = None
    gaia_variable_df: pd.DataFrame | None = None
    total = len(files)
    _debug_logged_once = False
    _uc_um = float(faintest_mag_limit) if faintest_mag_limit is not None else None
    try:
        _cfg_e = AppConfig()
        _gp = (_cfg_e.gaia_db_path or "").strip()
        _uc_root_e = Path(_gp) if _gp else None
    except Exception:  # noqa: BLE001
        _uc_root_e = None
    _export_cat_local = (not master_only_mode)

    master_tab: pd.DataFrame | None = None
    ref_wcs: WCS | None = None
    masterstar_data_shape: tuple[int, int] | None = None
    if use_master_fast_path and masterstars_csv is not None:
        _msp = Path(masterstars_csv)
        if _msp.is_file():
            try:
                _hdr_ms = pd.read_csv(_msp, nrows=0)
                _dtype_ms: dict[str, type] = {}
                if "catalog_id" in _hdr_ms.columns:
                    _dtype_ms["catalog_id"] = str
                if "name" in _hdr_ms.columns:
                    _dtype_ms["name"] = str
                _kw_ms: dict[str, Any] = {}
                if _dtype_ms:
                    _kw_ms["dtype"] = _dtype_ms
                _mt = pd.read_csv(_msp, **_kw_ms)
                if not _mt.empty and "ra_deg" in _mt.columns and "dec_deg" in _mt.columns:
                    master_tab = _mt
            except Exception:  # noqa: BLE001
                master_tab = None
    if use_master_fast_path and masterstar_fits is not None and master_tab is not None:
        _msf = Path(masterstar_fits)
        if _msf.is_file():
            try:
                with fits.open(_msf, memmap=False) as _mh:
                    _marr = np.asarray(_mh[0].data)
                    if _marr.ndim == 2:
                        masterstar_data_shape = (int(_marr.shape[0]), int(_marr.shape[1]))
                    ref_wcs = WCS(_mh[0].header)
                if not ref_wcs.has_celestial:
                    ref_wcs = None
            except Exception:  # noqa: BLE001
                ref_wcs = None
                masterstar_data_shape = None
        else:
            ref_wcs = None
    else:
        ref_wcs = None

    if master_tab is not None:
        LOGGER.info(
            "Per-frame catalog: MASTERSTAR lock enabled (single catalog: masterstars_full_match.csv, %s rows)",
            len(master_tab),
        )
    elif master_only_mode:
        raise RuntimeError(
            "Per-frame catalog lock requested, but masterstars_full_match.csv is missing or invalid."
        )

    _sat_diag_ctx: Any | None = None
    _sat_diag_archive: str = ""
    _ref_ra_deg: float | None = None
    _ref_dec_deg: float | None = None
    _drift_ref_catalog_id: str | None = None
    try:
        from sat_diag import (  # noqa: PLC0415
            draft_archive_from_platesolve,
            run_sat_diag,
        )

        _arch = draft_archive_from_platesolve(ps)
        if _arch is not None:
            _sat_diag_archive = str(_arch)
            _eq_sat = equipment_saturate_adu
            if _eq_sat is None and equipment_id is not None:
                _eq_sat = _equipment_saturate_adu_from_db(int(equipment_id))
            _ref_hdr = fits.Header()
            if files:
                with fits.open(files[0], memmap=False) as _rh:
                    _ref_hdr = _rh[0].header
            _sat_diag_ctx = run_sat_diag(_arch, equipment_adu=_eq_sat, hdr=_ref_hdr)
            if _sat_diag_ctx.sat_adu is not None:
                equipment_saturate_adu = float(_sat_diag_ctx.sat_adu)
            from sat_diag import resolve_drift_ref_sky_deg  # noqa: PLC0415

            _frame_hint = Path(files[0]).name if files else None
            _ref_ra_deg, _ref_dec_deg, _drift_ref_catalog_id = resolve_drift_ref_sky_deg(
                ps, frame_name_hint=_frame_hint
            )
            if _ref_ra_deg is None and master_tab is not None and not getattr(master_tab, "empty", True):
                if "ra_deg" in master_tab.columns and "dec_deg" in master_tab.columns:
                    _mra = pd.to_numeric(master_tab["ra_deg"], errors="coerce")
                    _mde = pd.to_numeric(master_tab["dec_deg"], errors="coerce")
                    _flux_col = "flux" if "flux" in master_tab.columns else None
                    if _flux_col:
                        _ord = pd.to_numeric(master_tab[_flux_col], errors="coerce").fillna(0)
                        _j = int(_ord.idxmax())
                    else:
                        _j = 0
                    if math.isfinite(float(_mra.iloc[_j])) and math.isfinite(float(_mde.iloc[_j])):
                        _ref_ra_deg = float(_mra.iloc[_j])
                        _ref_dec_deg = float(_mde.iloc[_j])
            if _ref_ra_deg is not None:
                LOGGER.info(
                    "[SAT-DIAG] drift reference sky (%.5f, %.5f) from %s",
                    _ref_ra_deg,
                    _ref_dec_deg,
                    _frame_hint or "platesolve",
                )
            LOGGER.info(
                "[SAT-DIAG] sat_adu=%s source=%s lin_adu=%s (archive %s)",
                _sat_diag_ctx.sat_adu,
                _sat_diag_ctx.sat_source,
                _sat_diag_ctx.lin_adu,
                _sat_diag_archive,
            )
    except Exception as _sd_exc:  # noqa: BLE001
        LOGGER.warning("[SAT-DIAG] init skipped: %s", _sd_exc)

    _gauss_override: float | None = None
    try:
        if masterstar_fits is not None:
            _ms_gauss = Path(masterstar_fits)
            if _ms_gauss.is_file():
                with fits.open(_ms_gauss, memmap=False) as _gfh:
                    _ghdr = _gfh[0].header
                    # PRIORITA 1: VY_FWHM_GAUSS - 2D Gaussian fit, closest to SExtractor
                    for _gk in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN"):
                        _vg = _ghdr.get(_gk)
                        if _vg is None:
                            continue
                        try:
                            _vgf = float(_vg)
                            if 1.0 <= _vgf <= 15.0:
                                _gauss_override = _vgf
                                LOGGER.debug(
                                    "[FWHM] gaussian_override from %s: %.3f px",
                                    _gk,
                                    _gauss_override,
                                )
                                break
                        except (TypeError, ValueError):
                            pass

                    # PRIORITA 2: VY_FWHM x 0.667 fallback
                    if _gauss_override is None:
                        _vy = _ghdr.get("VY_FWHM")
                        if _vy is not None:
                            try:
                                _vyf = float(_vy)
                                if 1.0 <= _vyf <= 15.0:
                                    _gauss_override = _vyf * (1.0 / 1.5)
                                    LOGGER.debug(
                                        "[FWHM] gaussian_override from VY_FWHMx0.667: %.3f px",
                                        _gauss_override,
                                    )
                            except (TypeError, ValueError):
                                pass
    except Exception:  # noqa: BLE001
        _gauss_override = None
    _ap_st["gaussian_fwhm_px_override"] = _gauss_override
    if _gauss_override is not None:
        log_event(
            f"[PHOT] gaussian_fwhm_px_override = {float(_gauss_override):.4f}px "
            "(z VY_FWHM alebo VY_FWHM_GAUSS)"
        )
    else:
        log_event("[PHOT] gaussian_fwhm_px_override = None -> fallback na momentx0.619 per frame")

    cfg_for_workers = app_config if app_config is not None else AppConfig()
    _dao_fw_export = (
        float(dao_fwhm_px)
        if dao_fwhm_px is not None
        else float(cfg_for_workers.sips_dao_fwhm_px)
    )
    n_workers = _vyvar_per_frame_csv_workers(cfg_for_workers)
    _ny, _nx = _estimate_catalog_frame_hw(work_ram if use_ram_inputs else None, files)
    n_workers = _vyvar_cap_mp_workers_for_catalog(
        n_workers,
        (_ny, _nx),
        reserve_gb=float(cfg_for_workers.per_frame_mp_reserve_ram_gb),
    )
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "Per-frame catalog: up to %s process worker(s); jednotny parallel count + RAM cap (psutil); "
            "env VYVAR_PARALLEL_WORKERS / legacy",
            n_workers,
        )

    use_parallel_mp = n_workers > 1 and total > 1
    kd_pack_mp: Any = None
    if use_parallel_mp and not master_only_mode:
        if use_ram_inputs and work_ram is not None:
            _h_ref, _d_ref = work_ram[0][1], np.asarray(work_ram[0][2], dtype=np.float32)
            cat_df, vsx_df, gaia_variable_df, kd_pack_mp = _prefetch_export_shared_catalog_for_process_pool(
                files=None,
                reference_hdr_data=(_h_ref.copy(), _d_ref),
                field_cat_path=field_cat_path,
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                gaia_db_path=_uc_root_e,
                gaia_local_max_mag=_uc_um,
                export_cat_local=_export_cat_local,
                plate_solve_fov_deg=_pfov_res,
            )
        else:
            cat_df, vsx_df, gaia_variable_df, kd_pack_mp = _prefetch_export_shared_catalog_for_process_pool(
                files=list(files),
                reference_hdr_data=None,
                field_cat_path=field_cat_path,
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                gaia_db_path=_uc_root_e,
                gaia_local_max_mag=_uc_um,
                export_cat_local=_export_cat_local,
                plate_solve_fov_deg=_pfov_res,
            )
        LOGGER.info(
        "Per-frame catalog: parallel backend=process (%s workers); Gaia cone prefetched in parent",
            n_workers,
        )
    else:
        _refresh_shared_cat_kdtree()

    _prog_seq = 0
    deferred_csv_writes: list[tuple[Path, pd.DataFrame]] = []

    def _append_deferred_csv(p: Path, df: pd.DataFrame) -> None:
        deferred_csv_writes.append((p, df))

    def _ensure_cone_and_variables(
        c_i: SkyCoord, r_i: float, *, naxis1: int = 0, naxis2: int = 0
    ) -> None:
        def _fill() -> None:
            nonlocal cat_df, vsx_df, gaia_variable_df
            if master_only_mode:
                return
            if cat_df is None or cat_df.empty:
                _gaia_db_path: Path | None = None
                try:
                    _gp = (_cfg_ap.gaia_db_path or "").strip()
                    if _gp:
                        _gaia_db_path = Path(_gp)
                except Exception:  # noqa: BLE001
                    _gaia_db_path = None
                cat_df = _query_gaia_local(
                    center=c_i,
                    radius_deg=r_i,
                    gaia_db_path=_gaia_db_path,
                    max_mag=float(_uc_um) if _uc_um is not None else None,
                )
                if cat_df is not None and len(cat_df) > 0:
                    try:
                        field_cat_path.parent.mkdir(parents=True, exist_ok=True)
                        _vyvar_df_to_csv(cat_df, field_cat_path)
                        _write_field_catalog_cone_meta(
                            field_cat_path,
                            center=c_i,
                            radius_deg=float(r_i),
                            naxis1=int(naxis1),
                            naxis2=int(naxis2),
                            plate_solve_fov_deg=_pfov_res,
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug("[PIPE] cone/variables CSV write failed: %s", exc)
                        pass
            if vsx_df is None or getattr(vsx_df, "empty", True):
                _vsx_p2: Path | None = None
                try:
                    _vsp2 = str(_cfg_ap.vsx_local_db_path or "").strip()
                    if _vsp2:
                        _vsx_p2 = Path(_vsp2).expanduser().resolve()
                except Exception:  # noqa: BLE001
                    _vsx_p2 = None
                if _vsx_p2 is not None and _vsx_p2.is_file():
                    vsx_df = _query_vsx_local(
                        center=c_i,
                        radius_deg=float(r_i),
                        vsx_db_path=_vsx_p2,
                    )
                else:
                    vsx_df = pd.DataFrame()
            if gaia_variable_df is None:
                gaia_variable_df = pd.DataFrame()
            _refresh_shared_cat_kdtree()

        _fill()

    def _run_one_catalog(base_path: Path, hdr: fits.Header, data: np.ndarray) -> dict[str, Any]:
        fname = base_path.name
        nonlocal _prog_seq
        if progress_cb is not None and n_workers <= 1:
            _prog_seq += 1
            progress_cb(_prog_seq, total, f"Catalog: {fname}")
        if not _has_valid_wcs(hdr):
            return {"file": fname, "status": "no_wcs", "csv": ""}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w_i = WCS(hdr)
        h_i, wpx_i = data.shape

        use_fast = (
            bool(use_master_fast_path)
            and master_tab is not None
            and (not getattr(master_tab, "empty", True))
        )
        if master_only_mode and not use_fast:
            return {"file": fname, "status": "error: master_only_mode requires masterstars_full_match.csv", "csv": ""}

        df: pd.DataFrame
        meta: dict[str, Any]
        if use_fast:
            try:
                _raw_arr = None
                _raw_hdr = None
                if _sat_diag_ctx is not None and _sat_diag_archive:
                    _raw_p = _resolve_draft_light_raw_path(Path(_sat_diag_archive), fname)
                    if _raw_p is not None and _raw_p.is_file():
                        try:
                            from sat_diag import image_adu_array  # noqa: PLC0415

                            with fits.open(_raw_p, memmap=False) as _rhd:
                                if int(_rhd[0].header.get("BITPIX", 0)) >= 0:
                                    _raw_arr = image_adu_array(_rhd[0])
                                    _raw_hdr = _rhd[0].header.copy()
                        except Exception:  # noqa: BLE001
                            _raw_arr = None
                df, meta = detect_stars_match_master_reference(
                    data,
                    hdr,
                    master_tab,
                    max_catalog_rows=int(max_catalog_rows),
                    match_sep_arcsec=float(catalog_match_max_sep_arcsec),
                    saturate_level_fraction=float(saturate_level_fraction),
                    faintest_mag_limit=faintest_mag_limit,
                    dao_threshold_sigma=float(dao_threshold_sigma),
                    dao_fwhm_px=float(_dao_fw_export),
                    equipment_saturate_adu=equipment_saturate_adu,
                    frame_name=fname,
                    sat_diag_ctx=_sat_diag_ctx,
                    raw_data=_raw_arr,
                    raw_hdr=_raw_hdr,
                    ref_ra_deg=_ref_ra_deg,
                    ref_dec_deg=_ref_dec_deg,
                    drift_ref_catalog_id=_drift_ref_catalog_id,
                )
            except Exception as exc:  # noqa: BLE001
                return {"file": fname, "status": f"error: {exc}", "csv": ""}
        else:
            if master_only_mode:
                return {"file": fname, "status": "error: master_only_mode fallback to Gaia cone is disabled", "csv": ""}
            c_i, r_i = _effective_field_catalog_cone_radius_deg(
                w_i, h_i, wpx_i, _pfov_res, fits_header=hdr
            )
            _ensure_cone_and_variables(c_i, r_i, naxis1=int(wpx_i), naxis2=int(h_i))

            try:
                df, meta = detect_stars_and_match_catalog(
                    data,
                    hdr,
                    max_catalog_rows=int(max_catalog_rows),
                    cat_df=cat_df,
                    vsx_df=vsx_df,
                    gaia_variable_df=gaia_variable_df,
                    match_sep_arcsec=float(catalog_match_max_sep_arcsec),
                    saturate_level_fraction=float(saturate_level_fraction),
                    faintest_mag_limit=faintest_mag_limit,
                    field_catalog_export_path=None,
                    dao_threshold_sigma=float(dao_threshold_sigma),
                    dao_fwhm_px=float(_dao_fw_export),
                    equipment_saturate_adu=equipment_saturate_adu,
                    catalog_local_gaia_only=_export_cat_local,
                    catalog_kd_pack=kd_cell[0] if kd_cell else None,
                    plate_solve_fov_deg=_pfov_res,
                    fov_database_path=_cfg_ap.database_path,
                    fov_equipment_id=int(equipment_id) if equipment_id is not None else None,
                    fov_draft_id=int(draft_id) if draft_id is not None else None,
                )
            except Exception as exc:  # noqa: BLE001
                return {"file": fname, "status": f"error: {exc}", "csv": ""}

        _exo_st = {
            "exoplanet_local_db_path": str(_cfg_ap.exoplanet_local_db_path or ""),
            "exoplanet_match_max_sep_arcsec": float(_cfg_ap.exoplanet_match_max_sep_arcsec),
            "plate_solve_fov_deg": _pfov_res,
            "database_path": str(Path(_cfg_ap.database_path).resolve()),
            "equipment_id": equipment_id,
            "draft_id": draft_id,
        }
        df = _apply_exo_host_columns_to_proc_df(df, hdr, (h_i, wpx_i), _exo_st, frame_name=fname)

        _before_dao = len(df)
        df = _proc_drop_unmatched_dao_rows(df)
        LOGGER.debug("[TODO-13] catalog-only pre-filter (detect): %d -> %d rows", _before_dao, len(df))

        _run_aperture = bool(_ap_st.get("_run_aperture", True))
        _run_epsf = bool(_ap_st.get("_run_epsf", False))
        if _run_aperture:
            _ap_st["current_frame_name"] = fname
            df = _apply_aperture_catalog_enhancements_from_st(df, data, hdr, _ap_st)
        _psf_on = bool(_ap_st.get("psf_photometry_enabled", False))
        _epsf_ids = (
            _epsf_fit_catalog_ids(ps, psf_photometry_enabled=_psf_on)
            if ps is not None
            else None
        )
        LOGGER.info(
            "[ePSF] _run_one_catalog %s: platesolve_dir=%r n_psf_ids=%s lc_set=%s",
            fname,
            str(ps.resolve()) if ps is not None else None,
            len(_epsf_ids) if _epsf_ids is not None else "ALL",
            _psf_on,
        )
        if not _run_aperture and _run_epsf:
            # PSF-only mode: psf_flux promoted to primary.
            pass
        df = _fill_psf_catalog_columns(df, data, hdr, _ap_st, target_ids=_epsf_ids)

        # --- Time columns (JD / HJD / BJD) ---
        _db_tc = None
        try:
            from time_utils import compute_time_columns

            _dbp = str(_ap_st.get("database_path") or "").strip()
            _did_tc = _ap_st.get("draft_id")
            if _dbp and _did_tc is not None:
                try:
                    _db_tc = VyvarDatabase(Path(_dbp))
                except Exception:  # noqa: BLE001
                    _db_tc = None
            _time_cols = compute_time_columns(
                hdr,
                db=_db_tc,
                draft_id=int(_did_tc) if _did_tc is not None else None,
                cfg=_cfg_ap,
            )
            _tk = ("jd_mid", "hjd_mid", "bjd_tdb_mid")
            _cols_base = list(df.columns)
            _anchors = [c for c in ("jd", "inspection_jd") if c in _cols_base]
            if _anchors:
                _pos = max(_cols_base.index(c) for c in _anchors) + 1
            else:
                _flux_first = next((c for c in _cols_base if c in ("dao_flux", "flux")), None)
                _pos = _cols_base.index(_flux_first) if _flux_first is not None else len(_cols_base)
            for _i, _nm in enumerate(_tk):
                df.insert(_pos + _i, _nm, _time_cols[_nm])

            # Airmass - frame-level hodnota z FITS hlavicky
            _am_val = _extract_airmass_from_header(
                hdr,
                cfg=_cfg_ap,
                db=_db_tc,
                draft_id=int(_did_tc) if _did_tc is not None else None,
            )
            if "airmass" not in df.columns:
                _am_insert_pos = _pos + len(_tk)
                df.insert(_am_insert_pos, "airmass", _am_val)
        except Exception as _tc_exc:  # noqa: BLE001
            log_event(f"Time columns skipped: {_tc_exc}")
            for _tc in ("jd_mid", "hjd_mid", "bjd_tdb_mid"):
                if _tc not in df.columns:
                    df[_tc] = None
            if "airmass" not in df.columns:
                df["airmass"] = float("nan")
        finally:
            if _db_tc is not None:
                try:
                    _db_tc.conn.close()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)

        df2 = df.copy()
        df2.insert(0, "source_file", fname)

        _before_dedupe = len(df2)
        df2 = _proc_deduplicate_matched_catalog_rows(df2)
        if len(df2) != _before_dedupe:
            LOGGER.debug(
                "[PROC] per-frame catalog dedupe: %d -> %d rows (%s)",
                _before_dedupe,
                len(df2),
                fname,
            )
        _before_cat = len(df2)
        df2 = _proc_catalog_keep_matched_rows_only(df2)
        LOGGER.debug("[TODO-13] catalog-only filter: %d -> %d rows", _before_cat, len(df2))

        csv_paths: list[str] = []
        if write_sidecar_csv_next_to_fits:
            sidecar = proc_csv_path_for_aligned_fits(base_path)
            if defer_disk_writes:
                _append_deferred_csv(sidecar, df2.copy())
            else:
                _vyvar_df_to_csv(df2, sidecar)
            csv_paths.append(str(sidecar))

        if mirror_flat_platesolve_folder:
            stem = Path(fname).stem
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:120]
            flat_path = out_flat / f"{safe}_catalog.csv"
            if defer_disk_writes:
                _append_deferred_csv(flat_path, df2.copy())
            else:
                _vyvar_df_to_csv(df2, flat_path)
            csv_paths.append(str(flat_path))

        primary_csv = csv_paths[0] if csv_paths else ""
        return {
            "file": fname,
            "status": "ok",
            "csv": primary_csv,
            "csv_paths": ";".join(csv_paths),
            "n_detected": meta.get("n_detected"),
            "n_matched": meta.get("n_matched"),
            "catalog_match_mode": ("master_reference_locked" if master_only_mode else meta.get("catalog_match_mode", "full_cone")),
            "raw_peaks_used": bool(meta.get("raw_peaks_used")),
        }

    def _process_frame(fp: Path) -> dict[str, Any]:
        try:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header.copy()
                    data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            except Exception as exc:  # noqa: BLE001
                return {"file": fp.name, "status": f"read_error: {exc}", "csv": ""}
            return _run_one_catalog(fp, hdr, data)
        except Exception as exc:  # noqa: BLE001
            return {"file": fp.name, "status": f"error: {exc}", "csv": ""}

    def _process_ram_item(item: tuple[str, fits.Header, Any]) -> dict[str, Any]:
        name, hdr0, arr0 = item
        try:
            base = Path(aligned_target_dir) / name
            return _run_one_catalog(base, hdr0.copy(), np.asarray(arr0, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            return {"file": name, "status": f"error: {exc}", "csv": ""}

    def _catalog_worker_state() -> dict[str, Any]:
        return {
            "cat_df": cat_df,
            "vsx_df": vsx_df,
            "gaia_variable_df": gaia_variable_df,
            "kd_pack": kd_pack_mp,
            "master_tab": master_tab,
            "masterstar_fits_path": (
                str(Path(masterstar_fits).resolve())
                if masterstar_fits is not None and Path(masterstar_fits).is_file()
                else ""
            ),
            "use_master_fast_path": bool(use_master_fast_path),
            "masterstar_data_shape": masterstar_data_shape,
            "max_catalog_rows": int(max_catalog_rows),
            "catalog_match_max_sep_arcsec": float(catalog_match_max_sep_arcsec),
            "saturate_level_fraction": float(saturate_level_fraction),
            "faintest_mag_limit": faintest_mag_limit,
            "dao_threshold_sigma": float(dao_threshold_sigma),
            "dao_fwhm_px": float(_dao_fw_export),
            "forced_photometry_enabled": bool(
                getattr(cfg_for_workers, "forced_photometry_enabled", True)
            ),
            "forced_photometry_centroid_bound_fwhm": float(
                getattr(cfg_for_workers, "forced_photometry_centroid_bound_fwhm", 2.5) or 2.5
            ),
            "forced_photometry_margin_px": float(
                getattr(cfg_for_workers, "forced_photometry_margin_px", 0.0) or 0.0
            ),
            "equipment_saturate_adu": equipment_saturate_adu,
            "sat_diag_ctx_dict": (
                _sat_diag_ctx.to_json_dict() if _sat_diag_ctx is not None else None
            ),
            "sat_diag_archive": _sat_diag_archive,
            "sat_diag_ref_ra": _ref_ra_deg,
            "sat_diag_ref_dec": _ref_dec_deg,
            "sat_diag_ref_catalog_id": _drift_ref_catalog_id,
            "export_cat_local": _export_cat_local,
            "master_only_mode": bool(master_only_mode),
            "plate_solve_fov_deg": _pfov_res,
            "write_sidecar_csv_next_to_fits": write_sidecar_csv_next_to_fits,
            "mirror_flat_platesolve_folder": mirror_flat_platesolve_folder,
            "defer_disk_writes": defer_disk_writes,
            "out_flat": str(out_flat.resolve()),
            "aperture_photometry_enabled": bool(_cfg_ap.aperture_photometry_enabled),
            "aperture_fwhm_factor": float(_cfg_ap.aperture_fwhm_factor),
            "annulus_inner_fwhm": float(_cfg_ap.annulus_inner_fwhm),
            "annulus_outer_fwhm": float(_cfg_ap.annulus_outer_fwhm),
            "nonlinearity_peak_percentile": float(_cfg_ap.nonlinearity_peak_percentile),
            "nonlinearity_fwhm_ratio": float(_cfg_ap.nonlinearity_fwhm_ratio),
            "bpm_dark_mad_sigma": float(_cfg_ap.bpm_dark_mad_sigma),
            "master_dark_path": _md_bpm_str,
            "database_path": str(Path(_cfg_ap.database_path).resolve()),
            "draft_id": int(draft_id) if draft_id is not None else None,
            "equipment_id": int(equipment_id) if equipment_id is not None else None,
            "gaussian_fwhm_px_override": _gauss_override,
            "aperture_correction_enabled": bool(_cfg_ap.aperture_correction_enabled),
            "aperture_fwhm_factor_small": float(_cfg_ap.aperture_snr_sizing.get("small", 1.5)),
            "aperture_fwhm_factor_large": float(_cfg_ap.aperture_snr_sizing.get("large", 4.0)),
            "aperture_policy_mode": str(_ap_policy_mode),
            "fwhm_night_median_px": _fwhm_night_median_px,
            "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
            "platesolve_dir": str(ps.resolve()),
            "observer_lat": float(_cfg_ap.observer_lat),
            "observer_lon": float(_cfg_ap.observer_lon),
            "observer_alt_m": float(_cfg_ap.observer_alt_m),
            "exoplanet_local_db_path": str(_cfg_ap.exoplanet_local_db_path or ""),
            "exoplanet_match_max_sep_arcsec": float(_cfg_ap.exoplanet_match_max_sep_arcsec),
            **_export_catalog_psf_st_fields(_cfg_ap, ps),
            "epsf_frame_index_by_name": dict(_frame_index_by_name),
            "epsf_science_set_meta": _epsf_science_meta,
        }

    if use_parallel_mp and use_ram_inputs and work_ram is not None:
        ctx = multiprocessing.get_context("spawn")
        ws = _catalog_worker_state()
        packs: list[tuple[str, bytes, bytes, int, int]] = []
        for _name, _hdr, _arr in work_ram:
            base_p = (root / _name).resolve()
            d = np.asarray(_arr, dtype=np.float32)
            if not d.flags.c_contiguous:
                d = np.ascontiguousarray(d, dtype=np.float32)
            packs.append(
                (
                    str(base_p),
                    pickle.dumps(_hdr.copy()),
                    d.tobytes(),
                    int(d.shape[0]),
                    int(d.shape[1]),
                )
            )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_export_per_frame_worker,
            initargs=(ws,),
        ) as ex:
            futs = {ex.submit(_export_per_frame_ram_worker_task, pack): pack[0] for pack in packs}
            by_name: dict[str, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                base_str = futs[fut]
                res = fut.result()
                nm = Path(base_str).name
                if (not _debug_logged_once) and isinstance(res, dict) and res.get("debug_pixel_match") is not None:
                    _debug_logged_once = True
                    try:
                        log_event(
                            "DEBUG: per-frame debug_pixel_match (first frame): "
                            + json.dumps(res.get("debug_pixel_match"), ensure_ascii=False, default=str)
                        )
                    except Exception:  # noqa: BLE001
                        log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {res.get('debug_pixel_match')}")
                for msg in res.get("infolog_messages") or []:
                    if msg:
                        log_event(str(msg))
                for p_str, dff in res.get("deferred_writes") or []:
                    deferred_csv_writes.append((Path(p_str), dff))
                by_name[nm] = res
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Catalog: {nm}")
        rows_out = []
        for it in work_ram:
            rr = dict(by_name[str(it[0])])
            rr.pop("deferred_writes", None)
            rr.pop("infolog_messages", None)
            rows_out.append(rr)
    elif use_parallel_mp:
        ctx = multiprocessing.get_context("spawn")
        ws = _catalog_worker_state()
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_export_per_frame_worker,
            initargs=(ws,),
        ) as ex:
            futs = {ex.submit(_export_per_frame_disk_worker_task, str(fp.resolve())): fp for fp in files}
            by_fp: dict[Path, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                fp = futs[fut]
                res = fut.result()
                if (not _debug_logged_once) and isinstance(res, dict) and res.get("debug_pixel_match") is not None:
                    _debug_logged_once = True
                    try:
                        log_event(
                            "DEBUG: per-frame debug_pixel_match (first frame): "
                            + json.dumps(res.get("debug_pixel_match"), ensure_ascii=False, default=str)
                        )
                    except Exception:  # noqa: BLE001
                        log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {res.get('debug_pixel_match')}")
                for msg in res.get("infolog_messages") or []:
                    if msg:
                        log_event(str(msg))
                by_fp[fp] = res
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Catalog: {fp.name}")
                for p_str, dff in res.get("deferred_writes") or []:
                    deferred_csv_writes.append((Path(p_str), dff))
        rows_out = []
        for fp in files:
            rr = dict(by_fp[fp])
            rr.pop("deferred_writes", None)
            rr.pop("infolog_messages", None)
            rows_out.append(rr)
    elif use_ram_inputs and work_ram is not None:
        rows_out = []
        for i, it in enumerate(work_ram, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Catalog: {it[0]}")
            r = _process_ram_item(it)
            if (not _debug_logged_once) and isinstance(r, dict) and r.get("debug_pixel_match") is not None:
                _debug_logged_once = True
                try:
                    log_event(
                        "DEBUG: per-frame debug_pixel_match (first frame): "
                        + json.dumps(r.get("debug_pixel_match"), ensure_ascii=False, default=str)
                    )
                except Exception:  # noqa: BLE001
                    log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {r.get('debug_pixel_match')}")
            for msg in r.get("infolog_messages") or []:
                if msg:
                    log_event(str(msg))
            r.pop("infolog_messages", None)
            rows_out.append(r)
    else:
        rows_out = []
        for i, fp in enumerate(files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Catalog: {fp.name}")
            r = _process_frame(fp)
            if (not _debug_logged_once) and isinstance(r, dict) and r.get("debug_pixel_match") is not None:
                _debug_logged_once = True
                try:
                    log_event(
                        "DEBUG: per-frame debug_pixel_match (first frame): "
                        + json.dumps(r.get("debug_pixel_match"), ensure_ascii=False, default=str)
                    )
                except Exception:  # noqa: BLE001
                    log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {r.get('debug_pixel_match')}")
            for msg in r.get("infolog_messages") or []:
                if msg:
                    log_event(str(msg))
            r.pop("infolog_messages", None)
            rows_out.append(r)

    index_path = ps / "per_frame_catalog_index.csv"
    if not defer_disk_writes:
        index_rows = [
            {k: v for k, v in row.items() if k not in ("deferred_writes", "infolog_messages")}
            for row in rows_out
        ]
        _vyvar_df_to_csv(pd.DataFrame(index_rows), index_path)
    n_ok = sum(1 for r in rows_out if r.get("status") == "ok")
    n_master_ref = sum(1 for r in rows_out if r.get("catalog_match_mode") == "master_reference")
    _hybrid_stats: dict[str, Any] = {}
    if not defer_disk_writes:
        _hybrid_stats = _finalize_hybrid_bkg_fallback_sidecar(
            root,
            err_background_mode="empirical",
            write_sidecar=bool(write_sidecar_csv_next_to_fits),
            gain=float(_ap_st.get("gain", _cfg_ap.gain)),
            read_noise=float(_ap_st.get("read_noise", _cfg_ap.read_noise)),
            setup_label=str(root.name),
        )
    if _sat_diag_ctx is not None and _sat_diag_archive:
        try:
            from sat_diag import commit_sat_diag_provenance  # noqa: PLC0415

            _placed_raw = any(bool(r.get("raw_peaks_used")) for r in rows_out)
            commit_sat_diag_provenance(
                _sat_diag_ctx,
                _sat_diag_archive,
                placed_aperture_used=_placed_raw,
            )
        except Exception as _sd_write_exc:  # noqa: BLE001
            LOGGER.warning("[SAT-DIAG] provenance commit skipped: %s", _sd_write_exc)
    _epsf_job_summary: dict[str, Any] | None = None
    if bool(_cfg_ap.psf_photometry_enabled) and bool(_ap_st.get("_run_epsf")):
        from epsf_frame_accounting import finalize_epsf_frame_job

        _psf_recs = [
            r["psf_frame_record"]
            for r in rows_out
            if isinstance(r.get("psf_frame_record"), dict)
        ]
        if _psf_recs:
            _epsf_job_summary = finalize_epsf_frame_job(
                _psf_recs,
                platesolve_dir=ps,
                science_set_meta=_epsf_science_meta,
            )
    return {
        "written": int(n_ok),
        "per_frame_dir": str(root),
        "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else ("flat_mirror" if mirror_flat_platesolve_folder else "none"),
        "index_csv": str(index_path),
        "frames": rows_out,
        "mirror_flat_platesolve_folder": bool(mirror_flat_platesolve_folder),
        "frames_master_reference_match": int(n_master_ref),
        "deferred_csv_writes": list(deferred_csv_writes) if defer_disk_writes else [],
        "hybrid_bkg_fallback": _hybrid_stats,
        "epsf_job_summary": _epsf_job_summary,
    }


def validate_comparison_ensemble_flatness(
    *,
    frames_root: Path,
    comparison_stars_csv: Path,
    flux_col: str = "flux",
    name_col: str = "name",
    max_relative_rms: float = 0.03,
    min_frames_per_star: int = 5,
    min_stars_per_frame: int = 3,
    output_report_csv: Path | None = None,
) -> dict[str, Any]:
    """Check that comparison stars stay flat vs the **per-frame** ensemble (median flux of comps on that frame).

    Uses DAO ``flux`` from sidecar CSVs next to each aligned FITS under ``frames_root`` (same layout as
    ``export_per_frame_catalogs``). For each exposure, builds relative flux ``f_i / median(f_all comps present)``.
    A good comparison star has low RMS of that ratio over time (instrument / transparency drifts divide out).

    Catalog non-variable filtering (VSX / Gaia) is handled earlier when building ``comparison_stars.csv``; this
    step is the **photometric** sanity check among the selected ensemble.

    Returns summary counts and per-star metrics; optionally writes ``output_report_csv``.
    """
    import numpy as np

    comp_path = Path(comparison_stars_csv)
    if not comp_path.is_file():
        return {"error": f"missing {comp_path}", "rows": []}

    _hdr_comp = pd.read_csv(comp_path, nrows=0)
    _dtype_comp: dict[str, type] = {}
    _nc = str(name_col)
    if _nc in _hdr_comp.columns:
        _dtype_comp[_nc] = str
    if "catalog_id" in _hdr_comp.columns:
        _dtype_comp["catalog_id"] = str
    if "name" in _hdr_comp.columns:
        _dtype_comp["name"] = str
    comp_df = pd.read_csv(comp_path, dtype=_dtype_comp)
    if name_col not in comp_df.columns:
        return {"error": f"comparison table missing column {name_col!r}", "rows": []}
    names = [str(x).strip() for x in comp_df[name_col].dropna().astype(str).unique() if str(x).strip()]
    if not names:
        return {"error": "no comparison star names", "rows": []}

    by_jd: dict[float, dict[str, float]] = {}
    root = Path(frames_root)
    files_n = 0
    _cfg_for_workers = AppConfig()
    for fp in sorted(_iter_fits_recursive(root)):
        sidecar = proc_csv_path_for_aligned_fits(fp)
        if not sidecar.is_file():
            continue
        meta = extract_fits_metadata(fp, app_config=_cfg_for_workers)
        jd = float(meta.get("jd_start") or 0.0)
        if jd <= 0.0:
            continue
        try:
            _hdr_sc = pd.read_csv(sidecar, nrows=0)
            _dtype_sc: dict[str, type] = {str(name_col): str}
            if "catalog_id" in _hdr_sc.columns:
                _dtype_sc["catalog_id"] = str
            if "name" in _hdr_sc.columns:
                _dtype_sc["name"] = str
            dff = pd.read_csv(sidecar, dtype=_dtype_sc)
        except Exception as exc:  # noqa: BLE001
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().stress_sidecar_skip += 1
            LOGGER.error("[STRESS] sidecar CSV read skip %s: %s", sidecar.name, exc)
            continue
        if name_col not in dff.columns or flux_col not in dff.columns:
            continue
        files_n += 1
        rowmap: dict[str, float] = {}
        for nm in names:
            m = dff.loc[dff[name_col].astype(str).str.strip() == nm]
            if m.empty:
                continue
            fl = float(m.iloc[0][flux_col])
            if np.isfinite(fl) and fl > 0:
                rowmap[nm] = fl
        if len(rowmap) >= int(min_stars_per_frame):
            by_jd[jd] = rowmap

    rel_lists: dict[str, list[float]] = {nm: [] for nm in names}
    for _jd, rowmap in sorted(by_jd.items(), key=lambda t: t[0]):
        vals = np.array(list(rowmap.values()), dtype=np.float64)
        med = float(np.median(vals))
        if not np.isfinite(med) or med <= 0:
            continue
        for nm, fl in rowmap.items():
            rel_lists[nm].append(float(fl / med))

    rows_out: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    for nm in names:
        arr = np.array(rel_lists[nm], dtype=np.float64)
        n_fr = int(len(arr))
        if n_fr < int(min_frames_per_star):
            rows_out.append(
                {
                    "name": nm,
                    "n_frames": n_fr,
                    "relative_rms": None,
                    "flatness_ok": False,
                    "reason": "too_few_frames",
                }
            )
            n_fail += 1
            continue
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        ok = rms <= float(max_relative_rms)
        rows_out.append(
            {
                "name": nm,
                "n_frames": n_fr,
                "relative_rms": rms,
                "flatness_ok": ok,
                "reason": "" if ok else "high_rms",
            }
        )
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    rep_path: str | None = None
    if output_report_csv is not None:
        outp = Path(output_report_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(outp, index=False)
        rep_path = str(outp)

    return {
        "comparison_stars_csv": str(comp_path),
        "frames_sampled": int(files_n),
        "frames_used_ensemble": int(len(by_jd)),
        "n_comparison_names": int(len(names)),
        "n_pass_flatness": int(n_pass),
        "n_fail_flatness": int(n_fail),
        "max_relative_rms_threshold": float(max_relative_rms),
        "min_frames_per_star": int(min_frames_per_star),
        "rows": rows_out,
        "report_csv": rep_path,
    }


def _apply_wcs_tan_fragment_to_header(h: fits.Header, wh: fits.Header, history_note: str) -> None:
    strip_celestial_wcs_keys(h)
    for k in wh:
        if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
            continue
        if k.startswith("NAXIS") and k != "NAXIS":
            continue
        try:
            h[k] = wh[k]
        except Exception:  # noqa: BLE001
            pass
    h.add_history(history_note)




def _fill_masterstars_gaia_matched_bp_rp_from_local_db(
    df: pd.DataFrame,
    *,
    gaia_db_path: str,
) -> tuple[pd.DataFrame, int, int]:
    """Dopln ``bp_rp`` / ``b_v`` z lokalnej Gaia SQLite pre ``GAIA_MATCHED`` bez farby v masterstars CSV.

    Frame-wide Gaia dotaz (``ORDER BY g_mag LIMIT``) casto vynecha uz omatchovane hviezdy na snimke;
    tato davka ide priamo podla ``source_id``.
    """
    if df is None or getattr(df, "empty", True):
        return df, 0, 0
    need = {"catalog_id", "bp_rp", "source_type"}
    if not need.issubset(df.columns):
        return df, 0, 0

    st_ok = df["source_type"].astype(str).str.strip().eq("GAIA_MATCHED")
    bpr = pd.to_numeric(df["bp_rp"], errors="coerce")
    gid = df["catalog_id"].map(normalize_gaia_source_id)
    gid_ok = gid.ne("")
    mask = st_ok & bpr.isna() & gid_ok
    n_missing = int(mask.sum())
    if n_missing <= 0:
        return df, 0, 0

    gdb = str(gaia_db_path or "").strip()
    try:
        gdb_ok = bool(gdb) and Path(gdb).is_file()
    except OSError:
        gdb_ok = False
    if not gdb_ok:
        return df, 0, n_missing

    out = df.copy()

    sub_idx = out.index[mask]
    keys_series = gid.loc[sub_idx]
    uniq_keys = sorted({k for k in keys_series.tolist() if k})
    if not uniq_keys:
        return out, 0, n_missing

    gaia_map = query_local_gaia_by_source_ids(gdb, uniq_keys)
    bprp_raw = keys_series.map(lambda k: (gaia_map.get(k) or {}).get("bp_rp"))
    bprp_num = pd.to_numeric(bprp_raw, errors="coerce")
    fill_ok = bprp_num.notna()
    to_fill = bprp_num.index[fill_ok]
    if len(to_fill) > 0:
        out.loc[to_fill, "bp_rp"] = bprp_num.loc[to_fill].astype(float)
    n_filled = int(fill_ok.sum())
    return out, n_filled, n_missing


def generate_masterstar_and_catalog(
    *,
    archive_path: Path,
    max_catalog_rows: int = 12000,
    astrometry_api_key: str | None = None,
    source_root: Path | None = None,
    platesolve_dir: Path | None = None,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    app_config: AppConfig | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
    telescope_id: int | None = None,
    master_dark_path: Path | str | None = None,
    masterstar_candidate_paths: "Sequence[str] | None" = None,
    masterstar_selection_pct: float | None = None,
    setup_name: str | None = None,
    masterstar_basename: str = "MASTERSTAR.fits",
    masterstars_csv_basename: str = "masterstars_full_match.csv",
    masterstar_fits_only: bool = False,
    masterstar_skip_build: bool = False,
    masterstar_platesolve_only: bool = False,
    masterstar_platesolve_skip_solve: bool = False,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
) -> dict[str, Any]:
    """Create MASTERSTAR.fits, plate-solve it, and export masterstars.csv.

    Ak je ``masterstar_fits_only=True``, po zostaveni FITS v ``platesolve/`` sa skonci (ziadny plate-solve ani CSV).
    Ak je ``masterstar_skip_build=True``, preskoci sa build z processed - pouzije sa existujuci ``MASTERSTAR.fits`` v ``platesolve/`` a bezi solver + katalog.
    Ak je ``masterstar_platesolve_only=True``, po uspesnom plate-solve a uprave mierky WCS sa skonci (bez DAO CSV, ``masterstars_full_match.csv``, fotometrickeho planu a zapisu MASTER_SOURCES).
    """
    max_catalog_rows = max(int(max_catalog_rows), 100000)
    import numpy as np

    ap = Path(archive_path).expanduser()
    # Draft UI moze poslat .../draft_x/non_calibrated - MASTERSTAR a platesolve patria pod koren draftu.
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    if equipment_saturate_adu is None:
        _sd_clip = _sat_adu_from_draft_sat_diag(ap)
        if _sd_clip is not None:
            equipment_saturate_adu = _sd_clip
            logging.info(
                "[INV-SAT-LIMIT] EQUIPMENTS.SATURATE_ADU missing; using sat_diag.json sat_adu=%.0f",
                _sd_clip,
            )
    detrended_root: Path | None = None
    if masterstar_skip_build:
        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        if not masterstar_fits.is_file():
            raise FileNotFoundError(
                f"MASTERSTAR plate-solve: v {platesolve_dir} chyba subor {_ms_name}. "
                "Najprv spusti **MAKE MASTERSTAR** na archive alebo vytvor MASTERSTAR inak (FITS QA -> referencny snimok)."
            )
        _match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
        if _match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
            log_event(
                f"MASTERSTAR: catalog match sep eff={_match_sep_eff:.2f} arcsec (min 10 for initial match)."
            )
        log_event(
            f"MASTERSTAR platesolve-from-disk: {masterstar_fits.resolve()} - VYVAR solver + katalog "
            "(bez noveho buildu z processed)."
        )
        ms_selection_meta = {
            "source": "platesolve_existing",
            "file": str(masterstar_fits.resolve()),
        }
        try:
            _ms_resolved = str(masterstar_fits.resolve())
        except OSError:
            _ms_resolved = str(masterstar_fits)
        info = {
            "masterstar_path": _ms_resolved,
            "frames_used": 1,
            "reference_path": _ms_resolved,
            "reference_index": 0,
            "stacked": False,
            "frames_combined": 1,
        }
    if not masterstar_skip_build:
        # MASTERSTAR-only reads from processed/lights/setup_name (robust folder-based discovery).
        if source_root is not None:
            detrended_root = Path(source_root)
        else:
            detrended_root = resolve_masterstar_input_root(
                ap,
                setup_name=setup_name,
                app_config=app_config,
                draft_id=draft_id,
            )
            if detrended_root is None:
                raise FileNotFoundError(
                    f"MASTERSTAR input root for setup {str(setup_name)!r} not found under "
                    f"{ap / 'processed' / 'lights'} (refusing cross-group fallback)."
                )
        if not detrended_root.exists():
            if setup_name:
                raise FileNotFoundError(
                    f"MASTERSTAR input root for setup {str(setup_name)!r} not found: {detrended_root}"
                )
            log_event(f"[X] MASTERSTAR FAIL: Input path {detrended_root} not found.")
            processed_lights = ap / "processed" / "lights"
            if processed_lights.is_dir():
                subdirs = sorted(
                    [d for d in processed_lights.iterdir() if d.is_dir()],
                    key=lambda p: p.name.casefold(),
                )
                if subdirs:
                    detrended_root = subdirs[0]
                    log_event(f"[OK] MASTERSTAR fallback input found: {detrended_root}")
        if not detrended_root.exists():
            raise FileNotFoundError(f"Missing processed/detrended lights: {detrended_root}")
        # If root exists but has no FITS, try first setup subfolder under processed lights (single-group only).
        if not setup_name and not _iter_fits_recursive(detrended_root):
            processed_lights = ap / "processed" / "lights"
            if processed_lights.is_dir():
                subdirs = sorted(
                    [d for d in processed_lights.iterdir() if d.is_dir()],
                    key=lambda p: p.name.casefold(),
                )
                for sd in subdirs:
                    if _iter_fits_recursive(sd):
                        log_event(f"[OK] MASTERSTAR fallback to setup subdir: {sd}")
                        detrended_root = sd
                        break

        log_event(f"[search] MASTERSTAR: Searching for candidates in {Path(detrended_root).resolve()}")
        log_event(f"Vstupny priecinok pre Masterstar: {Path(detrended_root).resolve()}")
        from draft_provenance import is_pre_calibrated_draft

        _pre_cal_ms = is_pre_calibrated_draft(ap, draft_id=draft_id)
        if _pre_cal_ms:
            log_event(
                "MASTERSTAR: pre-calibrated draft - candidates resolved directly under "
                f"{Path(detrended_root).resolve()} (no processed/calibrated remap)."
            )
        _match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
        if _match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
            log_event(
                f"MASTERSTAR: catalog match sep zvyseny na {_match_sep_eff:.2f}\" "
                f"(pozadovane minimum pre pociatocny match)."
            )

        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        only_ms_paths: list[Path] | None = None
        ms_selection_meta: dict[str, Any] = {}
        #: When True, ``masterstar_candidate_paths`` mapped to disk - do not append unrelated FITS
        #: for "best-of-N" pool (that would override a deliberate single-frame pick in the UI).
        explicit_ui_masterstar_paths = False

        def _map_qc_paths_to_disk(raw_paths: list[str]) -> list[Path]:
            """Map UI / DB paths onto draft lights FITS under ``detrended_root``.

            Pre-calibrated: match by basename under ``non_calibrated/lights/<setup>/``.
            VYVAR-calibrated: prefer ``processed/lights/.../proc_*.fits`` via remap helpers.
            """

            def _mapped_hit_ok(hit: Path) -> bool:
                if not hit.is_file() or _path_segments_forbidden_for_masterstar_physical_source(
                    hit, pre_calibrated=_pre_cal_ms
                ):
                    return False
                if _pre_cal_ms:
                    return _path_is_under_tree(Path(detrended_root), hit)
                pl = ap / "processed" / "lights"
                if pl.is_dir():
                    try:
                        hit.resolve().relative_to(pl.resolve())
                        return True
                    except ValueError:
                        return False
                return _path_is_under_tree(Path(detrended_root), hit)

            out: list[Path] = []
            for rp in raw_paths:
                s = str(rp).strip()
                if not s:
                    continue
                hit = _resolve_best_effort_path_under(
                    Path(detrended_root),
                    s,
                    pre_calibrated=_pre_cal_ms,
                )
                if hit is not None and _mapped_hit_ok(hit):
                    out.append(hit)
                    continue
                if _pre_cal_ms:
                    continue
                try:
                    hit2 = resolve_obs_file_to_processed_fits(
                        ap,
                        s,
                        setup_name=setup_name,
                        app_config=app_config,
                        draft_id=draft_id,
                    )
                except Exception:  # noqa: BLE001
                    hit2 = None
                if hit2 is not None and _mapped_hit_ok(hit2):
                    out.append(hit2)
            return out

        def _disk_stack_fallback_paths(input_dir: Path, *, max_frames: int = 8) -> list[Path]:
            """When QC paths / DB mapping fail: pick best frames from disk (deterministic order)."""
            all_on_disk = sorted(
                (
                    fp
                    for fp in _iter_fits_recursive(input_dir)
                    if _path_is_under_tree(input_dir, fp)
                    and not _path_segments_forbidden_for_masterstar_physical_source(
                        fp, pre_calibrated=_pre_cal_ms
                    )
                ),
                key=lambda p: str(p).casefold(),
            )
            if not all_on_disk:
                return []
            n = max(1, min(int(max_frames), len(all_on_disk)))
            return all_on_disk[:n]

        try:
            _pct_eff = float(masterstar_selection_pct) if masterstar_selection_pct is not None else 10.0
        except (TypeError, ValueError):
            _pct_eff = 10.0
        if not math.isfinite(_pct_eff) or _pct_eff <= 0:
            _pct_eff = 10.0
        _pct_eff = max(0.1, min(100.0, _pct_eff))

        cand_paths = [str(x) for x in (masterstar_candidate_paths or []) if str(x).strip()]
        if cand_paths:
            mapped = _map_qc_paths_to_disk(cand_paths)
            if mapped:
                only_ms_paths = mapped
                explicit_ui_masterstar_paths = True
                ms_selection_meta = {
                    "source": "ui_paths",
                    "requested": int(len(cand_paths)),
                    "mapped_found": int(len(mapped)),
                    "explicit_ui_lock": True,
                }
            else:
                raise FileNotFoundError(
                    "MASTERSTAR: z UI/job prisli explicitne cesty k referencnemu snimku, ale ziadna sa nenasla "
                    f"ako ``processed/lights/.../proc_*.fits`` (koren vyberu: {Path(detrended_root).resolve()}). "
                    "Skontroluj preprocess, archiv a vyber vo FITS QA (potvrd znovu po **Create Archive & Do Calibration**). "
                    f"Pozadovane ({len(cand_paths)}): " + "; ".join(cand_paths[:6]) + (" ..." if len(cand_paths) > 6 else "")
                )

        _multi_obs = draft_is_multi_group_obs(ap)
        if only_ms_paths is None and draft_id is not None and not (_multi_obs and setup_name):
            _db_ms = _vyvar_open_database(app_config or AppConfig())
            if _db_ms is not None:
                try:
                    # FITS QA 'Potvrdit vyber MASTERSTAR' -> ``get_obs_draft_masterstar_source_path``
                    # (``draft manifest.MASTERSTAR_PATH`` = zdrojovy frame, nie hotovy ``MASTERSTAR.fits``).
                    # Musi ist pred automatickym top-% z ``manifest files[]``, inak sa pouzivatelsky vyber prepise.
                    _src = _db_ms.get_obs_draft_masterstar_source_path(int(draft_id))
                    if _src and str(_src).strip():
                        mapped_src = _map_qc_paths_to_disk([str(_src).strip()])
                        if mapped_src:
                            only_ms_paths = mapped_src
                            explicit_ui_masterstar_paths = True
                            ms_selection_meta = {
                                "source": "db_masterstar_source_path",
                                "draft_id": int(draft_id),
                                "mapped_found": int(len(mapped_src)),
                                "explicit_ui_lock": True,
                            }
                            log_event(
                                f"MASTERSTAR: FITS QA vyber (DB source, draft {int(draft_id)}) -> {mapped_src[0].name}"
                            )
                    if only_ms_paths is None:
                        db_paths = get_masterstar_candidates(int(draft_id), _pct_eff, db=_db_ms)
                        mapped_db = _map_qc_paths_to_disk([str(x) for x in db_paths if str(x).strip()])
                        if mapped_db:
                            only_ms_paths = mapped_db
                            ms_selection_meta = {
                                "source": "db_top_pct",
                                "draft_id": int(draft_id),
                                "pct": float(_pct_eff),
                                "mapped_found": int(len(mapped_db)),
                            }
                            log_event(
                                f"MASTERSTAR: vyber z DB (draft {int(draft_id)}, top {_pct_eff:g} %) -> "
                                f"{len(mapped_db)} kandidatov (najlepsi sa skopiruje do platesolve)."
                            )
                        else:
                            log_event(
                                f"MASTERSTAR: DB vyber (draft {int(draft_id)}) sa nepodarilo namapovat na FITS pod {detrended_root}."
                            )
                except Exception as exc:  # noqa: BLE001
                    # EXC-0394: T4 -- `_dbc_fw.conn.close()` cleanup only; no radiometry or frame data touched. (EXCEPT-BULK 2026-07-08)
                    logging.error('[EXC-0393] DB FWHM median fetch `pass` leaves `_ms_fwhm_fb` at config default instead of draft QC ...: %s', exc)
                    log_event(f"MASTERSTAR: DB vyber kandidatov zlyhal ({exc!s}).")
                finally:
                    try:
                        _db_ms.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        if only_ms_paths is None:
            disk_batch = _disk_stack_fallback_paths(Path(detrended_root), max_frames=8)
            if disk_batch:
                only_ms_paths = disk_batch
                ms_selection_meta = {
                    "source": "disk_fallback_stack",
                    "mapped_found": int(len(disk_batch)),
                }
                log_event(
                    f"MASTERSTAR disk fallback: {len(disk_batch)} kandidatov z disku (bez platneho QC vyberu)."
                )

        if only_ms_paths is None:
            raise FileNotFoundError(
                f"MASTERSTAR: v {detrended_root} nie su ziadne FITS pre vyber ani po UI/DB."
            )

        _cfg_stack = app_config or AppConfig()
        try:
            _ms_fwhm_fb = float(_cfg_stack.sips_dao_fwhm_px)
        except (TypeError, ValueError):
            _ms_fwhm_fb = 2.5
        if not math.isfinite(_ms_fwhm_fb) or _ms_fwhm_fb <= 0:
            _ms_fwhm_fb = 2.5
        if draft_id is not None:
            _dbc_fw = _vyvar_open_database(_cfg_stack)
            if _dbc_fw is not None:
                try:
                    _fdf = get_masterstar_candidate_rows(int(draft_id), 100.0, db=_dbc_fw)
                    if _fdf is not None and not _fdf.empty and "FWHM" in _fdf.columns:
                        _vals = pd.to_numeric(_fdf["FWHM"], errors="coerce").to_numpy(dtype=float)
                        _vals = _vals[np.isfinite(_vals) & (_vals > 0.5) & (_vals < 80.0)]
                        if _vals.size:
                            _ms_fwhm_fb = float(np.median(_vals))
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    try:
                        _dbc_fw.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        # Build MASTERSTAR with best-of-N fallback: try a few top candidates if build/selection is brittle.
        try:
            _best_n = int(float(_cfg_stack.masterstar_best_of_n))
        except (TypeError, ValueError):
            _best_n = 10
        _best_n = max(1, min(25, int(_best_n)))
        _cand_all = [Path(p) for p in (only_ms_paths or []) if Path(p).is_file()]
        # If UI/DB mapping yields too few candidates, expand from disk for best-of-N robustness -
        # but never when the user explicitly passed ``masterstar_candidate_paths`` (would replace e.g.
        # a single chosen frame with unrelated lights and pick lowest VY_FWHM among them).
        try:
            if not explicit_ui_masterstar_paths and len(_cand_all) < max(2, _best_n):
                _disk_more = _disk_stack_fallback_paths(Path(detrended_root), max_frames=max(8, _best_n * 2))
                for p in _disk_more:
                    if p not in _cand_all and p.is_file():
                        _cand_all.append(p)
        except Exception:  # noqa: BLE001
            pass
        if not _cand_all:
            raise FileNotFoundError(f"MASTERSTAR: v {detrended_root} nie su ziadne FITS pre vyber.")
        _cand_singletons = _cand_all[:_best_n]

        _db_ms_build: VyvarDatabase | None = None
        if draft_id is not None:
            _db_ms_build = _vyvar_open_database(_cfg_stack)
        try:
            last_exc: Exception | None = None
            info = {}
            # Try pool first (as before), then single best-of-N frames.
            attempt_lists: list[tuple[str, list[Path]]] = [("pool", _cand_all)]
            for i, p in enumerate(_cand_singletons, start=1):
                attempt_lists.append((f"single_{i:02d}_of_{len(_cand_singletons):02d}", [p]))
            for label, paths_try in attempt_lists:
                try:
                    log_event(f"MASTERSTAR build attempt: {label} (n={len(paths_try)})")
                    info = build_masterstar_from_detrended(
                        detrended_root=detrended_root,
                        output_fits=masterstar_fits,
                        only_paths=paths_try,
                        fwhm_fallback_px=float(_ms_fwhm_fb),
                        app_config=_cfg_stack,
                        draft_id=draft_id,
                        db=_db_ms_build,
                        pre_calibrated=_pre_cal_ms,
                    )
                    # Update selection metadata for traceability.
                    ms_selection_meta = dict(ms_selection_meta or {})
                    ms_selection_meta["best_of_n"] = int(_best_n)
                    ms_selection_meta["build_attempt"] = str(label)
                    ms_selection_meta["build_only_paths"] = [str(p.name) for p in paths_try]
                    _bfpx = info.get("best_frame_fwhm_px")
                    if _bfpx is not None:
                        try:
                            _bfv = float(_bfpx)
                            if math.isfinite(_bfv) and 0.5 < _bfv < 80.0:
                                ms_selection_meta["best_frame_fwhm_px"] = float(_bfv)
                        except (TypeError, ValueError):
                            pass
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    log_event(f"MASTERSTAR build attempt failed: {label}: {exc!s}")
                    continue
            if last_exc is not None:
                raise last_exc
        finally:
            if _db_ms_build is not None:
                try:
                    _db_ms_build.conn.close()
                except Exception:  # noqa: BLE001
                    pass
        try:
            _legacy_master = Path(detrended_root) / "MASTERSTAR.fits"
            if _legacy_master.is_file() and _legacy_master.resolve() != masterstar_fits.resolve():
                _legacy_master.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    _selected_ref_path: Path | None = None
    try:
        _rp = str(info.get("reference_path") or "").strip()
        _selected_ref_path = Path(_rp) if _rp else None
    except Exception:  # noqa: BLE001
        _selected_ref_path = None

    if masterstar_fits_only:
        _cfg_fast = app_config or AppConfig()
        try:
            _ms_out = str(Path(masterstar_fits).resolve())
        except OSError:
            _ms_out = str(masterstar_fits)
        log_event(
            f"MASTERSTAR (len FITS, bez plate-solve): zapisane {_ms_out} | "
            f"zkombinovanych snimok={info.get('frames_combined', info.get('frames_used', '?'))}"
        )
        out_fast: dict[str, Any] = {
            "masterstar_fits": _ms_out,
            "masterstars_csv": "",
            "frames_used": int(info.get("frames_used", 0)),
            "masterstar_selection": ms_selection_meta or None,
            "masterstar_build_info": info,
            "n_raw_dao": 0,
            "detected_stars": 0,
            "catalog_matched": 0,
            "catalog_rows": 0,
            "catalog_match_max_sep_arcsec": float(_match_sep_eff),
            "solve": {"skipped": True, "reason": "masterstar_fits_only"},
        }
        try:
            if draft_id is not None:
                _db_ms = VyvarDatabase(Path(_cfg_fast.database_path))
                try:
                    _db_ms.set_obs_draft_masterstar_fits_path(int(draft_id), _ms_out)
                finally:
                    _db_ms.conn.close()
        except Exception as exc:  # noqa: BLE001
            out_fast["masterstar_path_store_error"] = str(exc)
        return out_fast

    # Solve WCS (MASTERSTAR): vyhradne VYVAR lokalny Gaia solver (ziadny ASTAP / astrometry.net).

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)

    _cfg_ms = app_config or AppConfig()

    try:
        _dao_sigma_eff = float(_cfg_ms.masterstar_dao_threshold_sigma)
    except (TypeError, ValueError):
        _dao_sigma_eff = 1.8
    if not math.isfinite(_dao_sigma_eff) or _dao_sigma_eff <= 0:
        _dao_sigma_eff = 1.8
    _dao_sigma_eff = max(0.1, min(6.0, float(_dao_sigma_eff)))
    log_event(
        f"MASTERSTAR: DAO threshold sigmaxRMS = {_dao_sigma_eff:.2f} "
        f"(config masterstar_dao_threshold_sigma; plate solve + katalog)"
    )

    _full_db = str(_cfg_ms.gaia_db_path or "").strip()
    if not _full_db:
        raise RuntimeError(
            "MASTERSTAR: v Settings nastavte gaia_db_path (plna lokalna Gaia DR3 SQLite DB)."
        )
    from vyvar_platesolver import solve_wcs_with_local_gaia

    log_event("MASTERSTAR WCS: VYVAR solver + plna Gaia DB (gaia_db_path).")
    try:
        _sip_ms = int(_cfg_ms.masterstar_platesolve_sip_max_order)
    except (TypeError, ValueError):
        _sip_ms = 5
    _sip_ms = max(2, min(5, _sip_ms))
    try:
        _sip_lo = int(_cfg_ms.masterstar_platesolve_sip_min_order)
    except (TypeError, ValueError):
        _sip_lo = 3
    _sip_lo = max(2, min(5, _sip_lo))
    if _sip_lo > _sip_ms:
        _sip_lo = _sip_ms
    log_event(
        f"MASTERSTAR: SIP skusanie {_sip_ms}->...->{_sip_lo} (config max/min plate-solve SIP)."
    )
    try:
        _xb_ms, _yb_ms = fits_binning_xy_from_header(hdr)
        _bin_ms = max(1, int(_xb_ms), int(_yb_ms))
    except Exception:  # noqa: BLE001
        _bin_ms = 1

    _auto_scale_ms: float | None = None
    _db_scale = _vyvar_open_database(_cfg_ms)
    _eq_ms: int | None = int(equipment_id) if equipment_id is not None else None
    _tel_ms: int | None = None
    if _db_scale is not None:
        try:
            _eq_ms, _tel_ms = resolve_optics_ids_for_platesolve(
                _db_scale, draft_id, equipment_id=equipment_id, telescope_id=telescope_id
            )
            _auto_scale_ms = compute_plate_scale_from_db(
                _eq_ms, _tel_ms, _db_scale.conn, binning=_bin_ms
            )
        except Exception:  # noqa: BLE001
            _auto_scale_ms = None
        finally:
            try:
                _db_scale.conn.close()
            except Exception:  # noqa: BLE001
                pass

    if _auto_scale_ms is not None:
        log_event(
            f"INFO: Plate scale z DB (Equipment+Telescope): {_auto_scale_ms:.4f} arcsec/px"
        )
    else:
        log_event(
            "WARNING: Plate scale z DB nedostupna - solver odvodi mierku z FITS alebo None"
        )

    _plate_scale_ms = _auto_scale_ms or None
    # Pull more complete optics hints (focal + effective pixel) from DB/FITS.
    # This is critical when FITS headers lack FOCALLEN/PIXSIZE and the solver would otherwise
    # overestimate FOV / cone radius and fail triangle matching.
    _bundle = _plate_solve_input_bundle(
        Path(masterstar_fits),
        app_config=_cfg_ms,
        equipment_id=_eq_ms,
        draft_id=int(draft_id) if draft_id is not None else None,
        telescope_id=_tel_ms if _tel_ms is not None else telescope_id,
    )
    _eff_um = _bundle.get("eff_um")
    _foc_mm = _bundle.get("focal_mm")
    _expected_bundle = _bundle.get("expected_arcsec_per_px")
    # D1/S6: a FITS/config/UI scale must not overwrite Equipment+Telescope DB
    # scale. On 520 g_60_4 the bundle used 200 mm / 15.511 "/px (Zeiss-wide
    # default) while the AZ800 row is 0.566 "/px; the triangle filter then
    # rejected every match. First auto-scale from DB wins.
    try:
        _bundle_scale = (
            float(_expected_bundle)
            if _expected_bundle is not None
            and math.isfinite(float(_expected_bundle))
            and float(_expected_bundle) > 0
            else None
        )
    except (TypeError, ValueError):
        _bundle_scale = None
    if _auto_scale_ms is not None:
        _plate_scale_ms = float(_auto_scale_ms)
        if (
            _bundle_scale is not None
            and abs(float(_bundle_scale) - float(_auto_scale_ms)) / float(_auto_scale_ms) > 0.05
        ):
            log_event(
                f"WARNING: MASTERSTAR plate-scale from FITS/config/UI "
                f"({_bundle_scale:.4f} arcsec/px) disagrees with DB Equipment+Telescope "
                f"({_auto_scale_ms:.4f} arcsec/px) - keeping DB scale for the triangle filter."
            )
    elif _bundle_scale is not None:
        _plate_scale_ms = float(_bundle_scale)

    _skip_independent_solve = bool(masterstar_platesolve_skip_solve) or (
        str(hdr.get("VY_CRT", "")).strip().lower() == "sibling_recovered"
        and _has_valid_wcs(hdr)
    )
    solve_meta: dict[str, Any] = {}
    if _skip_independent_solve:
        log_event(
            "MASTERSTAR: sibling-recovered WCS on disk - skipping independent Pass-1 plate-solve."
        )
        try:
            _vy_sodd = int(hdr.get("VY_SODD", 0) or 0)
        except (TypeError, ValueError):
            _vy_sodd = 0
        solve_meta = {
            "solved": True,
            "method": "sibling_recovered",
            "match_rate": 1.0,
            "sip_meta": {
                "masterstar_verified": True,
                "route": "sibling_recovered",
                "n_matched_tight": _vy_sodd,
            },
        }

    if not _skip_independent_solve:

        _mra, _mde, _ = _pointing_hint_from_header(hdr)
        if hint_ra_deg is not None and hint_dec_deg is not None:
            try:
                _hra_ov = float(hint_ra_deg)
                _hde_ov = float(hint_dec_deg)
                if math.isfinite(_hra_ov) and math.isfinite(_hde_ov):
                    _mra, _mde = _hra_ov, _hde_ov
                    log_event(
                        "MASTERSTAR: hint_ra_deg / hint_dec_deg z volania prepisuju hint z FITS "
                        "(druhy MASTERSTAR / detrended aligned)."
                    )
            except (TypeError, ValueError):
                pass
        try:
            _hint_sep_thr = float(_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG)
        except (TypeError, ValueError):
            _hint_sep_thr = 1.0
        if not math.isfinite(_hint_sep_thr) or _hint_sep_thr < 0:
            _hint_sep_thr = 1.0
        if draft_id is not None:
            _dbc_hint = _vyvar_open_database(_cfg_ms)
            if _dbc_hint is not None:
                try:
                    med_ra, med_de = draft_median_pointing_icrs_deg(_dbc_hint, int(draft_id))
                    if med_ra is not None and med_de is not None:
                        if _mra is None or _mde is None:
                            _mra, _mde = med_ra, med_de
                            log_event(
                                "MASTERSTAR solve: pouzivam median RA/Dec z manifest files[] (hlavicka bez spolahliveho hintu)."
                            )
                        else:
                            sc_h = SkyCoord(ra=float(_mra) * u.deg, dec=float(_mde) * u.deg, frame="icrs")
                            sc_d = SkyCoord(ra=float(med_ra) * u.deg, dec=float(med_de) * u.deg, frame="icrs")
                            sep = float(sc_h.separation(sc_d).deg)
                            if sep > float(_hint_sep_thr):
                                log_event(
                                    f"MASTERSTAR solve: hint vs draft median = {sep:.3f} deg > {_hint_sep_thr} deg "
                                    "- pouzivam draft median z manifest files[]."
                                )
                                _mra, _mde = med_ra, med_de
                            elif sep > 0.05:
                                log_event(
                                    f"MASTERSTAR solve: hint vs draft median = {sep:.3f} deg (skontrolujte pointing)."
                                )
                finally:
                    try:
                        _dbc_hint.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        _fov_ms_solve = resolve_plate_solve_fov_deg_hint(
            hdr,
            int(data.shape[0]),
            int(data.shape[1]),
            database_path=_cfg_ms.database_path,
            equipment_id=_eq_ms,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
        if _fov_ms_solve is None:
            try:
                _pf_ms = float(plate_solve_fov_deg)
                if math.isfinite(_pf_ms) and _pf_ms > 0:
                    _fov_ms_solve = _pf_ms
            except (TypeError, ValueError):
                pass
        if _fov_ms_solve is None:
            _fov_ms_solve = float(_cfg_ms.plate_solve_fov_deg)
        _prms = _MASTERSTAR_PLATESOLVE_PREWRITE_RMS_MAX_PX
        _prms_r = _MASTERSTAR_PLATESOLVE_PREWRITE_RELAXED_RMS_MAX_PX
        _nnrms = _MASTERSTAR_PLATESOLVE_NN_REFINE_MAX_RMS_PX
        # MASTERSTAR platesolve: always single best processed FITS (copy mode).
        _ms_vyvar_max_rows = 30000

        def _run_masterstar_vyvar_solve(*, enable_sip: bool, sip_max_order: int, fov_deg: float, max_rows: int) -> dict[str, Any]:
            return solve_wcs_with_local_gaia(
                masterstar_fits,
                hint_ra_deg=_mra,
                hint_dec_deg=_mde,
                fov_diameter_deg=float(fov_deg),
                gaia_db_path=Path(_full_db),
                enable_sip=bool(enable_sip),
                sip_max_order=int(sip_max_order),
                ransac_refinement=True,
                max_catalog_rows=int(max_rows),
                faintest_mag_limit=18.0,
                dao_threshold_sigma=float(_dao_sigma_eff),
                effective_pixel_um=float(_eff_um) if _eff_um is not None else None,
                focal_length_mm=float(_foc_mm) if _foc_mm is not None else None,
                expected_plate_scale_arcsec_per_px=(
                    float(_plate_scale_ms) if _plate_scale_ms is not None else None
                ),
                masterstar_prewrite_rms_max_px=float(_prms) if _prms is not None else None,
                masterstar_prewrite_relaxed_rms_max_px=float(_prms_r) if _prms_r is not None else None,
                masterstar_nn_refine_max_rms_px=float(_nnrms) if _nnrms is not None else None,
                masterstar_sip_min_order=int(_sip_lo),
                app_config=_cfg_ms,
                solver_use_cone_for_sip=True,
                solver_fits_header_hint_sep_escape=True,
                solver_legacy_masterstar_mirror_sweep=True,
                solver_apply_roworder_yflip=False,
            )

        solve_meta = _run_masterstar_vyvar_solve(
            enable_sip=True,
            sip_max_order=int(_sip_ms),
            fov_deg=float(_fov_ms_solve),
            max_rows=int(_ms_vyvar_max_rows),
        )
        if not isinstance(solve_meta, dict) or not bool(solve_meta.get("solved", False)):
            raise RuntimeError(
                "MASTERSTAR plate-solve zlyhal. "
                f"Back-end returned: {solve_meta!r}. "
                "Cannot safely continue with photometry / source extraction."
            )

        # Refresh header/data after solve attempt (solver overwrote MASTERSTAR.fits header)
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        if not _has_valid_wcs(hdr):
            raise RuntimeError(
                "MASTERSTAR: po plate-solve chyba platny WCS. Skontroluj gaia_db_path, RA/Dec a mierku v hlavicke "
                "(FOCALLEN/PIXSIZE alebo SECPIX) a vystup solvera."
            )

        # Pipeline-level acceptance criteria (stricter than solver's minimal guard):
        # - match_rate: allow 60% on the first solve (optimizer refines later)
        try:
            _mr = float(solve_meta.get("match_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            _mr = 0.0
        from vyvar_platesolver import MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE

        _min_mr = float(MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE)
        if _mr < _min_mr:
            raise RuntimeError(
                f"MASTERSTAR plate-solve zamietnuty: match_rate={_mr * 100.0:.1f}% < {_min_mr * 100.0:.0f}%. "
                "Skus zvysit n_stack alebo upravit hint/DAO prahy."
            )

        try:
            _aniso_thr = float(_PLATESOLVE_ANISOTROPY_THRESHOLD)
        except (TypeError, ValueError):
            _aniso_thr = 1.3
        if not math.isfinite(_aniso_thr) or _aniso_thr <= 0:
            _aniso_thr = 1.3
        _aniso_thr = max(1.01, min(5.0, float(_aniso_thr)))

        # Post-solve anisotropy validation: reject strongly anisotropic pixel scale and retry solver once.
        try:
            wcs0 = WCS(hdr)
            scale_x = abs(float(wcs0.pixel_scale_matrix[0, 0])) * 3600.0  # arcsec/px
            scale_y = abs(float(wcs0.pixel_scale_matrix[1, 1])) * 3600.0  # arcsec/px
            if math.isfinite(scale_x) and math.isfinite(scale_y) and scale_x > 0 and scale_y > 0:
                scale_ratio = max(scale_x, scale_y) / min(scale_x, scale_y)
            else:
                scale_ratio = float("nan")
        except Exception:  # noqa: BLE001
            scale_ratio = float("nan")

        if math.isfinite(scale_ratio) and scale_ratio > _aniso_thr:
            log_event(
                f"VAROVANIE: Anizotropna mierka ratio={scale_ratio:.2f} - plate-solve zamietnuty, restartujem solver (relaxed)."
            )
            # Retry with relaxed knobs:
            # - slightly larger FOV diameter (hint-vs-solved tolerance),
            # - more Gaia rows,
            # - no SIP (simpler model can be more stable when the fit goes off-rails).
            solve_meta2 = _run_masterstar_vyvar_solve(
                enable_sip=False,
                sip_max_order=0,
                fov_deg=float(_fov_ms_solve) * 1.25,
                max_rows=int(max(_ms_vyvar_max_rows, 30000)),
            )
            if not isinstance(solve_meta2, dict) or not bool(solve_meta2.get("solved", False)):
                raise RuntimeError(
                    f"MASTERSTAR platesolve retry zlyhal po anizotropii. Back-end returned: {solve_meta2!r}"
                )
            solve_meta = solve_meta2
            # Reload header after retry
            with fits.open(masterstar_fits, memmap=False) as hdul:
                hdr = hdul[0].header.copy()
                data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            if not _has_valid_wcs(hdr):
                raise RuntimeError("MASTERSTAR: po retry plate-solve chyba platny WCS.")
            try:
                wcs1 = WCS(hdr)
                sx = abs(float(wcs1.pixel_scale_matrix[0, 0])) * 3600.0
                sy = abs(float(wcs1.pixel_scale_matrix[1, 1])) * 3600.0
                if math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0:
                    scale_ratio2 = max(sx, sy) / min(sx, sy)
                else:
                    scale_ratio2 = float("nan")
            except Exception:  # noqa: BLE001
                scale_ratio2 = float("nan")
            if math.isfinite(scale_ratio2) and scale_ratio2 > _aniso_thr:
                raise RuntimeError(
                    f"MASTERSTAR plate-solve zamietnuty: anizotropna mierka po retry ratio={scale_ratio2:.2f} (>{_aniso_thr})."
                )

    _exp_scale_apx: float | None = None
    if _plate_scale_ms is not None:
        try:
            _ea2 = float(_plate_scale_ms)
            if math.isfinite(_ea2) and _ea2 > 0:
                _exp_scale_apx = float(_ea2)
        except (TypeError, ValueError):
            _exp_scale_apx = None
    if _exp_scale_apx is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                _w_hdr = WCS(hdr)
            if getattr(_w_hdr, "has_celestial", False):
                _pm0 = _w_hdr.pixel_scale_matrix
                _sx0 = abs(float(_pm0[0, 0])) * 3600.0
                _sy0 = abs(float(_pm0[1, 1])) * 3600.0
                if math.isfinite(_sx0) and math.isfinite(_sy0) and _sx0 > 0 and _sy0 > 0:
                    _exp_scale_apx = float((_sx0 + _sy0) / 2.0)
        except Exception:  # noqa: BLE001
            _exp_scale_apx = None
    if _exp_scale_apx is None or (not math.isfinite(_exp_scale_apx)) or _exp_scale_apx <= 0:
        # derive-or-None (DR6 pattern): all principled sources exhausted - do not guess.
        _exp_scale_apx = None
        log_event(
            "WARNING: MASTERSTAR plate scale not derivable (DB/FITS/WCS exhausted) - "
            "expected scale unknown; VY_PLTS will not be written."
        )

    _wcs_ok = False
    try:
        with fits.open(masterstar_fits, memmap=False) as _hd_wq:
            _w_check = WCS(_hd_wq[0].header)
        if _exp_scale_apx is None:
            log_event(
                "MASTERSTAR WCS quality: expected plate scale unknown "
                "(no DB/FITS/WCS/config) - check skipped."
            )
            _wcs_q = None
        else:
            _wcs_q = masterstar_wcs_quality(
                _w_check, float(_exp_scale_apx), anisotropy_limit=float(_aniso_thr)
            )
            _wcs_ok = bool(_wcs_q.get("ok", False))
        if _wcs_q is not None and not _wcs_ok:
            _rq = _wcs_q.get("ratio")
            _se = _wcs_q.get("scale_err_pct")
            try:
                _rq_s = f"{float(_rq):.2f}" if _rq is not None and math.isfinite(float(_rq)) else str(_rq)
            except (TypeError, ValueError):
                _rq_s = str(_rq)
            try:
                _se_s = f"{float(_se):.1f}" if _se is not None and math.isfinite(float(_se)) else str(_se)
            except (TypeError, ValueError):
                _se_s = str(_se)
            log_event(
                f"MASTERSTAR WCS kvalita: zla (ratio={_rq_s}, scale_err={_se_s}%) - "
                "pokracujem bez externeho plate-solve (ocakava sa FITS metadata / buduci blind solver)."
            )
    except Exception as _wq_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR WCS check failed: {_wq_exc}")
        _wcs_ok = False

    try:
        _pscale_adj = _try_rescale_masterstar_linear_wcs_to_expected_plate_scale(
            masterstar_fits,
            app_config=app_config or AppConfig(),
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(f"WCS PLATE SCALE: neocakavana chyba - {exc!s}")
        _pscale_adj = {"rescaled": False, "error": str(exc)}
    solve_meta["wcs_plate_scale_adjustment"] = _pscale_adj

    # Write calibrated plate scale to MASTERSTAR header
    _vy_plts = None
    try:
        if isinstance(solve_meta, dict):
            _vy_plts = solve_meta.get("plate_scale_arcsec_px")
        if _vy_plts is None and isinstance(_pscale_adj, dict):
            _vy_plts = _pscale_adj.get("new_scale_arcsec_per_px") or _pscale_adj.get(
                "expected_arcsec_per_px"
            )
        if _vy_plts is None:
            _vy_plts = _exp_scale_apx
    except Exception:  # noqa: BLE001
        _vy_plts = None

    if _vy_plts is not None:
        try:
            _vy_plts_f = float(_vy_plts)
            if math.isfinite(_vy_plts_f) and _vy_plts_f > 0:
                with fits.open(masterstar_fits, mode="update") as hdul:
                    hdul[0].header["VY_PLTS"] = (
                        _vy_plts_f,
                        "VYVAR plate scale arcsec/px",
                    )
                    hdul.flush()
                log_event(f"VY_PLTS={_vy_plts_f:.4f} written to MASTERSTAR.fits")
        except Exception as exc:  # noqa: BLE001
            log_event(f"Could not write VY_PLTS to MASTERSTAR: {exc}")
    else:
        log_event(
            "WARNING: MASTERSTAR VY_PLTS not written - plate scale not derivable "
            "(derive-or-None; no rig/global constant written to header)."
        )

    try:
        from wcs_invertibility import evaluate_wcs_roundtrip

        _nax1 = int(hdr.get("NAXIS1") or (data.shape[1] if data.ndim >= 2 else 0))
        _nax2 = int(hdr.get("NAXIS2") or (data.shape[0] if data.ndim >= 1 else 0))
        if _has_valid_wcs(hdr) and _nax1 > 0 and _nax2 > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                _w_rt = WCS(hdr)
            _rt0 = evaluate_wcs_roundtrip(_w_rt, naxis1=_nax1, naxis2=_nax2)
            if isinstance(solve_meta, dict):
                solve_meta["wcs_roundtrip_p99_px"] = _rt0.get("wcs_roundtrip_p99_px")
                solve_meta["wcs_roundtrip_pass"] = bool(_rt0.get("pass"))
            if not _rt0.get("pass"):
                log_event(
                    f"WARNING: MASTERSTAR initial-solve WCS round-trip p99="
                    f"{_rt0.get('wcs_roundtrip_p99_px'):.4f}px (threshold "
                    f"{_rt0.get('p99_threshold_px')}px) - provenance flag set; continuing."
                )
            else:
                log_event(
                    f"MASTERSTAR WCS round-trip PASS (initial solve): p99="
                    f"{_rt0.get('wcs_roundtrip_p99_px'):.4f}px"
                )
    except Exception as _rt_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR WCS round-trip check skipped: {_rt_exc!s}")

    if masterstar_platesolve_only:
        _cfg_early = app_config or AppConfig()
        try:
            _ms_out_early = str(Path(masterstar_fits).resolve())
        except OSError:
            _ms_out_early = str(masterstar_fits)
        log_event(
            f"ONLY MASTER (test): plate-solve + uprava mierky WCS hotove -> {_ms_out_early} "
            "(preskakujem DAO export, masterstars CSV, fotometricky plan, MASTER_SOURCES)."
        )
        out_ps: dict[str, Any] = {
            "masterstar_fits": _ms_out_early,
            "masterstars_csv": "",
            "frames_used": int(info.get("frames_used", 0)),
            "masterstar_selection": ms_selection_meta or None,
            "masterstar_build_info": info,
            "n_raw_dao": 0,
            "detected_stars": 0,
            "catalog_matched": 0,
            "catalog_rows": 0,
            "catalog_match_max_sep_arcsec": float(_match_sep_eff),
            "solve": solve_meta,
            "masterstar_platesolve_only": True,
            "comparison_stars_csv": "",
            "variable_targets_csv": "",
            "photometry_plan_json": "",
        }
        try:
            if draft_id is not None:
                _db_early = VyvarDatabase(Path(_cfg_early.database_path))
                try:
                    _db_early.set_obs_draft_masterstar_fits_path(int(draft_id), _ms_out_early)
                finally:
                    _db_early.conn.close()
        except Exception as exc:  # noqa: BLE001
            out_ps["masterstar_path_store_error"] = str(exc)
        return out_ps

    # _cfg_ms / _dao_sigma_eff uz vyssie (rovnake DAO sigma pre plate solve aj katalog).
    _ms_fwhm = float(_cfg_ms.sips_dao_fwhm_px)
    if not math.isfinite(_ms_fwhm) or _ms_fwhm <= 0:
        _ms_fwhm = 2.5
    _ms_meta = ms_selection_meta if isinstance(ms_selection_meta, dict) else {}
    _best_fwhm = _ms_meta.get("best_frame_fwhm_px")
    try:
        _best_fwhm_f = float(_best_fwhm) if _best_fwhm is not None else float("nan")
    except (TypeError, ValueError):
        _best_fwhm_f = float("nan")
    _use_best_frame_fwhm = bool(_cfg_ms.masterstar_use_best_frame_fwhm)
    if (
        _use_best_frame_fwhm
        and math.isfinite(_best_fwhm_f)
        and 1.2 <= _best_fwhm_f <= 20.0
    ):
        dao_fwhm_px_for_ms = float(_best_fwhm_f)
        _dao_fwhm_bypass_hdr = True
        log_event(
            f"MASTERSTAR DAO: dao_fwhm_px={dao_fwhm_px_for_ms:.3f} from best_frame_fwhm_px "
            f"(header VY_FWHM median ignored)"
        )
    else:
        dao_fwhm_px_for_ms = float(_ms_fwhm)
        _dao_fwhm_bypass_hdr = False
        log_event(
            f"MASTERSTAR DAO: dao_fwhm_px={dao_fwhm_px_for_ms:.3f} from sips_dao_fwhm_px / header VY_FWHM"
        )

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    data = np.ascontiguousarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Global median (BOJ) background subtraction is intentionally OFF (handled downstream).
    _ms_mean = float(np.nanmean(data))
    log_event("MASTERSTAR: globalne odcitanie medianu (BOJ) vypnute.")
    log_event(f"MASTERSTAR po nan_to_num: mean={_ms_mean:.6f}")
    _ms_min = float(np.nanmin(data))
    _ms_max = float(np.nanmax(data))
    log_event(f"MASTERSTAR levels: noise_floor(min)={_ms_min:.2f}, saturation_proxy(max)={_ms_max:.2f}")

    if platesolve_dir is None:
        raise ValueError("generate_masterstar_and_catalog: platesolve_dir is required (got None).")
    platesolve_dir.mkdir(parents=True, exist_ok=True)
    _fov_job: float | None = None
    try:
        _fj = float(plate_solve_fov_deg)
        if math.isfinite(_fj) and _fj > 0:
            _fov_job = _fj
    except (TypeError, ValueError):
        _fov_job = None
    if _fov_job is None:
        _fov_job = resolve_plate_solve_fov_deg_hint(
            hdr,
            int(data.shape[0]),
            int(data.shape[1]),
            database_path=_cfg_ms.database_path,
            equipment_id=int(equipment_id) if equipment_id is not None else None,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
    if _fov_job is None:
        _fov_job = float(_cfg_ms.plate_solve_fov_deg)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_pre = WCS(hdr)
        if _w_pre.has_celestial:
            _, _r_cat_need = _effective_field_catalog_cone_radius_deg(
                _w_pre, int(data.shape[0]), int(data.shape[1]), float(_fov_job), fits_header=hdr
            )
            _invalidate_field_catalog_cone_cache_if_needed(
                platesolve_dir / "field_catalog_cone.csv",
                plate_solve_fov_deg=float(_fov_job),
                effective_radius_deg=float(_r_cat_need),
            )
    except Exception as exc:  # noqa: BLE001
        log_event(f"Katalog: kontrola cache field_catalog_cone preskocena - {exc!s}")

    # Full-field MASTERSTAR depth: keep deeper Gaia and larger catalog rows for corner recovery.
    _ms_max_catalog_rows_eff = max(int(max_catalog_rows), 100000)
    if faintest_mag_limit is None:
        _ms_faintest_mag_eff: float | None = 18.0
    else:
        try:
            _ms_faintest_mag_eff = max(float(faintest_mag_limit), 18.0)
        except (TypeError, ValueError):
            _ms_faintest_mag_eff = 18.0
    df_out, det_meta = detect_stars_and_match_catalog(
        data,
        hdr,
        max_catalog_rows=int(_ms_max_catalog_rows_eff),
        cat_df=None,
        match_sep_arcsec=float(_match_sep_eff),
        saturate_level_fraction=float(saturate_level_fraction),
        faintest_mag_limit=_ms_faintest_mag_eff,
        field_catalog_export_path=platesolve_dir / "field_catalog_cone.csv",
        dao_threshold_sigma=float(_dao_sigma_eff),
        dao_fwhm_px=dao_fwhm_px_for_ms,
        equipment_saturate_adu=equipment_saturate_adu,
        catalog_local_gaia_only=catalog_local_gaia_only,
        plate_solve_fov_deg=float(_fov_job),
        fov_database_path=_cfg_ms.database_path,
        fov_equipment_id=int(equipment_id) if equipment_id is not None else None,
        fov_draft_id=int(draft_id) if draft_id is not None else None,
        prematch_peak_sigma_floor=float(
            _cfg_ms.masterstar_prematch_peak_sigma_floor
        ),
        dao_fwhm_bypass_header=bool(_dao_fwhm_bypass_hdr),
    )
    try:
        if isinstance(solve_meta, dict) and bool(solve_meta.get("solved")):
            _px = solve_meta.get("pairs_x")
            _py = solve_meta.get("pairs_y")
            _pra = solve_meta.get("pairs_ra")
            _pde = solve_meta.get("pairs_de")
            _pids = solve_meta.get("pairs_catalog_id")
            if (
                isinstance(_px, list)
                and isinstance(_py, list)
                and isinstance(_pra, list)
                and isinstance(_pde, list)
                and isinstance(_pids, list)
                and len(_px) > 0
                and len(_px) == len(_py) == len(_pra) == len(_pde) == len(_pids)
            ):
                _sm0 = solve_meta.get("sip_meta") if isinstance(solve_meta.get("sip_meta"), dict) else {}
                _mir = str((_sm0 or {}).get("det_mirror_orientation") or "").strip()
                df_out = _merge_platesolve_gaia_pairs_into_masterstars_df(
                    df_out,
                    pairs_x=[float(t) for t in _px],
                    pairs_y=[float(t) for t in _py],
                    pairs_ra=[float(t) for t in _pra],
                    pairs_de=[float(t) for t in _pde],
                    pairs_catalog_id=[str(t) for t in _pids],
                )
                log_event(
                    f"MASTERSTAR: VYVAR pary ({len(_px)}) zlucene do katalogu "
                    f"(mirror={_mir or 'native'}, pre astrometry optimizer)."
                )
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR: zlucenie VYVAR parov preskocene - {exc!s}")

    _fwhm_dao = float(det_meta.get("dao_fwhm_px") or 0.0)
    if not math.isfinite(_fwhm_dao) or _fwhm_dao <= 0:
        _fwhm_dao = float((det_meta.get("identity_gate") or {}).get("fwhm_px") or 1.25)

    try:
        from wcs_invertibility import (
            accumulate_identity_gate,
            apply_post_match_identity_gate_df,
            gaia_radec_map_from_table,
        )

        _gmap_ms: dict[str, tuple[float, float]] = {}
        _cone_p = Path(platesolve_dir) / "field_catalog_cone.csv"
        if _cone_p.is_file():
            _cone_df = pd.read_csv(_cone_p, low_memory=False, dtype={"catalog_id": str, "source_id": str})
            _gmap_ms.update(gaia_radec_map_from_table(_cone_df))
        if isinstance(solve_meta, dict):
            _pids = solve_meta.get("pairs_catalog_id") or []
            _pra = solve_meta.get("pairs_ra") or []
            _pde = solve_meta.get("pairs_de") or []
            from gaia_catalog_id import normalize_gaia_source_id as _norm_gid

            for _i, _pid in enumerate(_pids):
                _k = _norm_gid(str(_pid))
                if _k and _i < len(_pra) and _i < len(_pde):
                    _gmap_ms[_k] = (float(_pra[_i]), float(_pde[_i]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_gate = WCS(hdr)
        df_out, _idc_merge = apply_post_match_identity_gate_df(
            df_out,
            _w_gate,
            gaia_ra_dec_by_cid=_gmap_ms,
            fwhm_px=_fwhm_dao,
            log_fn=log_event,
        )
        _acc = dict(det_meta.get("identity_gate") or {})
        _n_out = int(
            df_out.get("catalog_id", pd.Series([""] * len(df_out)))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        det_meta["identity_gate"] = accumulate_identity_gate(_acc, _idc_merge, _n_out)
        det_meta["identity_gate"]["fwhm_px"] = float(_fwhm_dao)
    except Exception as _mg_exc:  # noqa: BLE001
        log_event(f"post_match_identity_gate (post-merge) skipped: {_mg_exc!s}")

    if "b_v" in df_out.columns and "bp_rp" not in df_out.columns:
        df_out = df_out.copy()
        df_out["bp_rp"] = pd.to_numeric(df_out["b_v"], errors="coerce")
    if "mag" in df_out.columns:
        df_out = df_out.copy()
        df_out["phot_g_mean_mag"] = pd.to_numeric(df_out["mag"], errors="coerce")

    if int(det_meta.get("n_detected", 0)) == 0:
        raise RuntimeError("No stars detected on MASTERSTAR.")
    _n_det_raw = int(det_meta.get("n_detected", 0) or 0)
    _n_mat_raw = int(det_meta.get("n_matched", 0) or 0)
    _rate_raw = (100.0 * float(_n_mat_raw) / float(_n_det_raw)) if _n_det_raw > 0 else 0.0
    _cat_rows = int(det_meta.get("catalog_rows", 0) or 0)
    if "catalog_id" in df_out.columns:
        _cid_raw = df_out["catalog_id"].fillna("").astype(str).str.strip()
        _n_gaia_det_raw = int(_cid_raw[_cid_raw != ""].nunique())
    else:
        _n_gaia_det_raw = int(_n_mat_raw)
    _gaia_rate_raw = (100.0 * float(_n_gaia_det_raw) / float(_cat_rows)) if _cat_rows > 0 else 0.0
    log_event(
        f"[chart] MATCH STATS (raw): Found {_n_det_raw} stars on image | {_n_mat_raw} matched with Gaia | "
        f"Match Rate: {_rate_raw:.2f}% | Gaia->DAO: {_gaia_rate_raw:.2f}% ({_n_gaia_det_raw}/{_cat_rows})"
    )
    if _cat_rows > 0:
        LOGGER.info(
            "[MASTERSTAR] Gaia->DAO completeness: "
            "%d/%d (%.1f%%) | catalog_only: %d",
            _n_gaia_det_raw,
            _cat_rows,
            _gaia_rate_raw,
            _cat_rows - _n_gaia_det_raw,
        )
    _update_masterstar_obs_file_status(
        cfg=_cfg_ms,
        draft_id=draft_id,
        selected_ref_path=_selected_ref_path,
        wcs_ok=bool(_has_valid_wcs(hdr)),
        n_stars=_n_det_raw,
    )
    temp_csv = platesolve_dir / "masterstars.csv"
    _msc_name = str(masterstars_csv_basename or "masterstars_full_match.csv").strip() or "masterstars_full_match.csv"
    csv_path = platesolve_dir / _msc_name
    _vyvar_df_to_csv(df_out, temp_csv)
    _opt_stats_last: dict[str, Any] = {}
    try:
        from astrometry_optimizer import optimize_masterstar_matches

        _gdb_opt = str(_cfg_ms.gaia_db_path or "").strip()
        if _gdb_opt:
            _mir_extra = bool(_MASTERSTAR_OPTIMIZER_MIRROR_EXTRA_LOG)
            _idg_n_out = int((det_meta.get("identity_gate") or {}).get("n_matched_out") or 0)
            try:
                with fits.open(masterstar_fits, mode="update", memmap=False) as _hf_dao:
                    _hf_dao[0].header["VY_FWHM_DAO"] = (
                        float(_fwhm_dao),
                        "DAO-domain FWHM [pix] for identity gate",
                    )
                    _hf_dao.flush()
            except Exception as _dao_hdr_exc:  # noqa: BLE001
                log_event(f"MASTERSTAR VY_FWHM_DAO stamp skipped: {_dao_hdr_exc!s}")
            csv_path = optimize_masterstar_matches(
                masterstars_csv=temp_csv,
                masterstar_fits=masterstar_fits,
                gaia_db_path=_gdb_opt,
                output_csv=csv_path,
                gaia_mag_limit=float(_ms_faintest_mag_eff),
                gaia_max_catalog_rows=int(_ms_max_catalog_rows_eff),
                mirror_orientation_extra_log=_mir_extra,
                sip_force_rms_guard_ratio=_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO,
                fwhm_dao_px=float(_fwhm_dao),
                identity_gate_n_out=_idg_n_out,
                stats_out=_opt_stats_last,
            )
            # Force one more pass after WCS displacement update for final edge recovery.
            # Identity-count contract is first-entry only; rematch may add honest pairs.
            csv_path = optimize_masterstar_matches(
                masterstars_csv=csv_path,
                masterstar_fits=masterstar_fits,
                gaia_db_path=_gdb_opt,
                output_csv=csv_path,
                gaia_mag_limit=float(_ms_faintest_mag_eff),
                gaia_max_catalog_rows=int(_ms_max_catalog_rows_eff),
                mirror_orientation_extra_log=_mir_extra,
                sip_force_rms_guard_ratio=_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO,
                fwhm_dao_px=float(_fwhm_dao),
                identity_gate_n_out=None,
                stats_out=_opt_stats_last,
            )
            log_event("MASTERSTAR optimizer: forced final re-match pass completed.")
            # Final safety: repair any residual precision-loss IDs in masterstars_full_match.csv via Gaia RA/DEC lookup.
            try:
                from repair_catalog_ids import repair_csv_catalog_ids_from_gaia_db  # noqa: PLC0415

                rep = repair_csv_catalog_ids_from_gaia_db(
                    csv_path=Path(csv_path),
                    gaia_db_path=Path(_gdb_opt),
                    id_col="catalog_id",
                    backup=True,
                    max_sep_arcsec=2.0,
                    log_fn=log_event,
                    skip_unmatched_placeholders=True,
                )
                if int(rep.get("repaired") or 0) > 0:
                    log_event(
                        f"MASTERSTAR repair: repaired={rep.get('repaired')} warnings={rep.get('warnings')} ({Path(csv_path).name})"
                    )
            except Exception as _rep_exc:  # noqa: BLE001
                log_event(f"MASTERSTAR repair skipped: {_rep_exc!s}")
        else:
            _vyvar_df_to_csv(df_out, csv_path)
    except Exception as exc:  # noqa: BLE001
        from invariants_runtime import InvariantViolation as _InvMatchId  # noqa: PLC0415

        if isinstance(exc, _InvMatchId):
            raise
        log_event(f"MASTERSTAR optimizer skipped/fallback: {exc!s}")
        _vyvar_df_to_csv(df_out, csv_path)
    try:
        # Critical: keep Gaia IDs as strings (avoid float/scientific precision loss).
        df_final = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str, "name": str})
        # Preserve DAO pass provenance through astrometry_optimizer CSV round-trip.
        for _prov_col in ("vy_dao_pass", "ambiguous_owner"):
            if _prov_col not in df_out.columns:
                continue
            _cid_out = df_out.get("catalog_id", pd.Series([""] * len(df_out))).map(
                lambda c: str(c).strip()
            )
            _map = df_out.assign(_cid=_cid_out).drop_duplicates("_cid", keep="last").set_index("_cid")[
                _prov_col
            ]
            _cid_fin = df_final.get("catalog_id", pd.Series([""] * len(df_final))).map(
                lambda c: str(c).strip()
            )
            df_final[_prov_col] = _cid_fin.map(_map)
            if _prov_col == "vy_dao_pass":
                df_final[_prov_col] = pd.to_numeric(df_final[_prov_col], errors="coerce").fillna(1)
            elif _prov_col == "ambiguous_owner":
                df_final[_prov_col] = df_final[_prov_col].fillna(False).astype(bool)
        if len(df_final) == len(df_out):
            for _idcol in ("vy_identity_gate", "gaia_dao_resid_px"):
                if _idcol in df_out.columns and _idcol not in df_final.columns:
                    df_final[_idcol] = df_out[_idcol].to_numpy()
    except Exception as _df_final_exc:  # noqa: BLE001
        log_event(
            f"MASTERSTAR: re-read of {Path(csv_path).name} failed ({_df_final_exc!s}); "
            "using in-memory df_out and re-asserting catalog_id/name as str."
        )
        df_final = df_out.copy()
        # df_out.copy() can carry catalog_id/name as non-string dtypes -> re-assert to avoid
        # reintroducing float/scientific precision loss on Gaia IDs downstream.
        for _idcol in ("catalog_id", "name"):
            if _idcol in df_final.columns:
                df_final[_idcol] = df_final[_idcol].astype(str)
    _wcs_rt_p99: float | None = None
    _wcs_rt_pass: bool | None = None
    _identity_qa: dict[str, Any] = {}
    try:
        from wcs_invertibility import (
            evaluate_matched_world2pix_identity_px,
            evaluate_wcs_roundtrip,
            finalize_masterstar_sky_coords,
        )

        with fits.open(masterstar_fits, memmap=False) as _hf:
            _hdr_fin = _hf[0].header
            _w_fin = WCS(_hdr_fin)
        df_final = finalize_masterstar_sky_coords(
            df_final,
            _w_fin,
            gaia_db_path=str(_cfg_ms.gaia_db_path or ""),
            log_fn=log_event,
        )
        _nax1f = int(_hdr_fin.get("NAXIS1") or 0)
        _nax2f = int(_hdr_fin.get("NAXIS2") or 0)
        _rt_fin = evaluate_wcs_roundtrip(_w_fin, naxis1=_nax1f, naxis2=_nax2f)
        _wcs_rt_p99 = _rt_fin.get("wcs_roundtrip_p99_px")
        _wcs_rt_pass = bool(_rt_fin.get("pass"))
        _identity_qa = evaluate_matched_world2pix_identity_px(
            df_final,
            _w_fin,
            gaia_db_path=str(_cfg_ms.gaia_db_path or ""),
            log_fn=log_event,
        )
        _p95 = _identity_qa.get("matched_world2pix_identity_p95_px")
        try:
            _p95f = float(_p95) if _p95 is not None else float("nan")
        except (TypeError, ValueError):
            _p95f = float("nan")
        # Standing series WARN (Anchor #3 / draft_435 baseline p95~1.54 px): soft threshold only.
        # INV-WCS-01: same band, recorded into pipeline_meta invariants at merge below.
        _IDENTITY_P95_WARN_PX = 2.0
        if math.isfinite(_p95f) and _p95f > _IDENTITY_P95_WARN_PX:
            logging.warning(
                "[IDENTITY-QA] matched_world2pix_identity_p95_px=%.3f exceeds WARN threshold %.1f px "
                "(series baseline draft_435 p95~1.54; no FAIL)",
                _p95f,
                _IDENTITY_P95_WARN_PX,
            )
            log_event(
                f"IDENTITY-QA WARN: p95={_p95f:.3f} px > {_IDENTITY_P95_WARN_PX:.1f} px threshold"
            )
        try:
            from invariants_runtime import check_wcs_identity_p95  # noqa: PLC0415
            from invariants_runtime import inv_check  # noqa: PLC0415

            _ok_w, _det_w = check_wcs_identity_p95(_p95f if math.isfinite(_p95f) else None)
            _inv_meta_wcs: dict = {"invariants": []}
            inv_check(_inv_meta_wcs, "INV-WCS-01", _ok_w, policy="WARN", detail=_det_w)
            _identity_qa = dict(_identity_qa or {})
            _identity_qa["_inv_wcs_01"] = _inv_meta_wcs.get("invariants") or []
        except Exception as _inv_wcs_exc:  # noqa: BLE001
            logging.debug("[INV-WCS-01] record skipped: %s", _inv_wcs_exc)
    except Exception as _fin_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR coordinate finalization / round-trip QA skipped: {_fin_exc!s}")
        _wcs_rt_p99 = None
        _wcs_rt_pass = None
        _identity_qa = {}
    # DAO-GAIA-ERA-01 M1: expand detection table to catalog-derived membership before zones/enrich.
    # INV-MS-EXPAND-01: when cone+WCS exist, expand must succeed or raise (no silent skip).
    _chip_ms: pd.DataFrame | None = None
    _membership_expand_meta: dict[str, Any] = {}
    _cone_gaia_pre = Path(platesolve_dir) / "field_catalog_cone.csv"
    _wcs_ok_pre = bool(_has_valid_wcs(hdr))
    if _cone_gaia_pre.is_file() and _wcs_ok_pre:
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            expand_detection_to_catalog_membership,
            gaia_on_chip_from_cone,
        )
        from astropy.wcs import WCS as _WCS_expand  # noqa: PLC0415

        LOGGER.info(
            "[M1] catalog membership expand: cone=%s wcs_ok=%s n_ms_in=%d",
            True,
            True,
            int(len(df_final)),
        )
        _cone_df_pre = read_vyvar_csv(_cone_gaia_pre, low_memory=False, dtype={"catalog_id": str})
        _nax1_pre = int(hdr.get("NAXIS1") or 0)
        _nax2_pre = int(hdr.get("NAXIS2") or 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _wcs_pre = _WCS_expand(hdr)
        _ra_pre = pd.to_numeric(_cone_df_pre["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _de_pre = pd.to_numeric(_cone_df_pre["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _ok_pre = np.isfinite(_ra_pre) & np.isfinite(_de_pre)
        _gx_pre, _gy_pre = _wcs_pre.world_to_pixel_values(_ra_pre[_ok_pre], _de_pre[_ok_pre])
        _chip_ms = gaia_on_chip_from_cone(
            _cone_df_pre, gx=_gx_pre, gy=_gy_pre, ok_mask=_ok_pre, wpx=_nax1_pre, h=_nax2_pre
        )
        _membership_depth_g = float(
            getattr(_cfg_ms, "masterstar_gaia_census_target_depth_g", None) or 15.0
        )
        df_final, _membership_expand_meta = expand_detection_to_catalog_membership(
            df_final,
            _chip_ms,
            membership_depth_g=_membership_depth_g,
            wpx=_nax1_pre,
            h=_nax2_pre,
        )
        det_meta["catalog_derived_membership"] = dict(_membership_expand_meta)
        LOGGER.info(
            "[M1] catalog-derived membership: +%d Gaia rows (depth G<=%.1f), n_out=%d",
            int(_membership_expand_meta.get("n_catalog_rows_added", 0)),
            _membership_depth_g,
            int(_membership_expand_meta.get("n_rows_out", len(df_final))),
        )
        log_event(
            "MASTERSTAR catalog-derived membership: "
            f"+{int(_membership_expand_meta.get('n_catalog_rows_added', 0))} Gaia rows "
            f"(depth G<={_membership_depth_g:.1f}), "
            f"n_out={int(_membership_expand_meta.get('n_rows_out', len(df_final)))}"
        )
    elif _cone_gaia_pre.is_file() or _wcs_ok_pre:
        raise RuntimeError(
            "INV-MS-EXPAND-01: catalog membership expand blocked "
            f"(cone={_cone_gaia_pre.is_file()} wcs_ok={_wcs_ok_pre})"
        )
    # VSX stamp deferred until after write_photometry_plan_files (VT CSV created there).
    df_final = _annotate_masterstars_flux_zones(
        df_final,
        noise_floor_adu=det_meta.get("noise_floor_adu"),
        equipment_saturate_adu=equipment_saturate_adu,
        saturate_limit_adu_fallback=det_meta.get("saturate_limit_adu"),
        saturate_limit_fraction=float(_cfg_ms.saturate_limit_fraction),
        sigma_px=det_meta.get("bg_sigma_adu"),
        sky_median_adu=det_meta.get("sky_median_adu"),
        prematch_peak_sigma_floor=det_meta.get("prematch_peak_sigma_floor"),
        frame_max_adu=det_meta.get("frame_max_adu"),
        empirical_clip_adu=det_meta.get("empirical_clip_adu"),
        dao_detection_n_equiv=(
            det_meta.get("dao_detection_n_equiv")
            if det_meta.get("dao_detection_n_equiv") is not None
            else float(_cfg_ms.dao_detection_n_equiv)
        ),
    )
    _dao_class_meta: dict[str, Any] = {}
    _recon_ms: dict[str, Any] | None = None
    try:
        cid = df_final.get("catalog_id", pd.Series([""] * len(df_final))).fillna("").astype(str).str.strip()
        df_final["source_type"] = np.where(cid.ne(""), "GAIA_MATCHED", "DAO_ONLY")
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            enrich_masterstar_gaia_complete,
            gaia_on_chip_from_cone,
            write_gaia_census_and_verify,
        )
        from astropy.wcs import WCS as _WCS_enrich  # noqa: PLC0415

        _cone_gaia = Path(platesolve_dir) / "field_catalog_cone.csv"
        if not _cone_gaia.is_file():
            raise RuntimeError("MASTERSTAR Gaia-complete enrich: missing field_catalog_cone.csv")
        if not _has_valid_wcs(hdr):
            raise RuntimeError("MASTERSTAR Gaia-complete enrich: MASTERSTAR WCS missing/invalid")
        _cone_df = read_vyvar_csv(_cone_gaia, low_memory=False, dtype={"catalog_id": str})
        _nax1_g = int(hdr.get("NAXIS1") or 0)
        _nax2_g = int(hdr.get("NAXIS2") or 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _wcs_g = _WCS_enrich(hdr)
        _ra_g = pd.to_numeric(_cone_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _de_g = pd.to_numeric(_cone_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _ok_g = np.isfinite(_ra_g) & np.isfinite(_de_g)
        _gx, _gy = _wcs_g.world_to_pixel_values(_ra_g[_ok_g], _de_g[_ok_g])
        _chip = (
            _chip_ms
            if _chip_ms is not None and len(_chip_ms)
            else gaia_on_chip_from_cone(
                _cone_df, gx=_gx, gy=_gy, ok_mask=_ok_g, wpx=_nax1_g, h=_nax2_g
            )
        )
        with fits.open(masterstar_fits, memmap=False) as _hg:
            _raw_g = np.asarray(_hg[0].data, dtype=np.float32)
        _, _med_g, _ = plain_mean_med_std(_raw_g, sigma=3.0, maxiters=3)
        _data0_g = np.nan_to_num(
            (_raw_g - _med_g).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        _fwhm_g = float(det_meta.get("dao_fwhm_px") or header_core_fwhm_px(hdr) or 3.5)
        _membership_depth_g = float(
            getattr(_cfg_ms, "masterstar_gaia_census_target_depth_g", None) or 15.0
        )
        _census_depth_g = 17.5  # M1-amend: census accounting depth; G 15-17.5 census-only
        df_final, _gaia_census, _gaia_meta = enrich_masterstar_gaia_complete(
            df_final,
            data0=_data0_g,
            gaia_on_chip=_chip,
            cfg=_cfg_ms,
            wpx=_nax1_g,
            h=_nax2_g,
            fwhm_px=_fwhm_g,
            target_depth_g=_census_depth_g,
            sat_limit_adu=det_meta.get("saturate_limit_adu"),
            identity_lock_only=False,
            catalog_derived_membership=bool(_membership_expand_meta),
            tolerance_overrides=det_meta.get("dao_gaia_derived_tol"),
        )
        _census_inv = write_gaia_census_and_verify(
            _gaia_census,
            n_on_chip=len(_chip),
            census_path=Path(platesolve_dir) / "gaia_source_state_census.csv",
        )
        try:
            from dao_gaia_calibration import (  # noqa: PLC0415
                build_calibration_certificate,
                write_calibration_certificate,
            )

            _setup_nm = str(setup_name or platesolve_dir.name or "MASTERSTAR")
            _tol_d = det_meta.get("dao_gaia_derived_tol") or {}
            _dao_x_f = pd.to_numeric(df_final.get("x"), errors="coerce").to_numpy(dtype=np.float64)
            _dao_y_f = pd.to_numeric(df_final.get("y"), errors="coerce").to_numpy(dtype=np.float64)
            _cert = build_calibration_certificate(
                setup=_setup_nm,
                wcs_obj=_wcs_g,
                data0=_data0_g,
                dao_x=_dao_x_f,
                dao_y=_dao_y_f,
                gaia_x=pd.to_numeric(_chip.get("x_gaia"), errors="coerce").to_numpy(dtype=np.float64),
                gaia_y=pd.to_numeric(_chip.get("y_gaia"), errors="coerce").to_numpy(dtype=np.float64),
                gaia_g=pd.to_numeric(_chip.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64),
                fwhm_px=float(_fwhm_g),
                pass1_sigma=float(_tol_d.get("pass1_sigma") or _cfg_ms.masterstar_dao_threshold_sigma),
                pass2_sigma=float(_tol_d.get("pass2_sigma") or _cfg_ms.masterstar_dao_pass2_sigma),
                seed_snr_min=float(_cfg_ms.masterstar_forced_seed_snr_min),
                target_depth_g=float(_census_depth_g),
                edge_margin_px=float(_cfg_ms.masterstar_gaia_census_edge_margin_px),
                cfg=_cfg_ms,
                ms_df=df_final,
                census_df=_gaia_census,
                repo_root=Path(__file__).resolve().parent.parent,
            )
            _cert_path = write_calibration_certificate(
                _cert, Path(platesolve_dir), fail_closed=True
            )
            if _membership_expand_meta:
                from masterstar_gaia_accounting import verify_ms_expand_guard  # noqa: PLC0415

                _ok_exp, _det_exp = verify_ms_expand_guard(
                    _membership_expand_meta,
                    census_path=Path(platesolve_dir) / "gaia_source_state_census.csv",
                    cert_path=_cert_path,
                )
                if not _ok_exp:
                    from invariants_runtime import InvariantViolation  # noqa: PLC0415

                    raise InvariantViolation("INV-MS-EXPAND-01", _det_exp)
                log_event(f"INV-MS-EXPAND-01 PASS: {_det_exp}")
            det_meta["dao_gaia_calibration"] = _cert.to_dict()
            det_meta["dao_gaia_calibration_path"] = str(_cert_path)
            log_event(
                f"DAO-Gaia calibration certificate {_cert.status}: "
                f"match_r={_cert.derived.match_radius_px:.1f}px "
                f"centroid={_cert.derived.pass2_center_tol_px:.1f}px "
                f"empty-sky det={_cert.empty_sky.inv_det} seed={_cert.empty_sky.inv_seed}"
            )
        except Exception as _cal_exc:  # noqa: BLE001
            from invariants_runtime import InvariantViolation  # noqa: PLC0415

            if isinstance(_cal_exc, InvariantViolation):
                raise
            if _membership_expand_meta:
                raise RuntimeError(
                    f"INV-MS-EXPAND-01: certificate write failed: {_cal_exc!s}"
                ) from _cal_exc
            log_event(f"DAO-Gaia calibration certificate skipped: {_cal_exc!s}")
        det_meta["gaia_census_meta"] = _gaia_meta
        det_meta["gaia_census_invariants"] = _census_inv.get("invariants") or []
        log_event(
            f"MASTERSTAR Gaia census: {len(_chip)} on-chip, "
            f"forced_seed={_gaia_meta.get('n_forced_seed', 0)}, "
            f"leftover_promotions={_gaia_meta.get('n_leftover_promotions', 0)}, "
            f"INV-MS-CENSUS-01 {_census_inv.get('detail')}"
        )
        _gdb_fill = str(_cfg_ms.gaia_db_path or "").strip()
        df_final, _n_bp_fill, _n_bp_miss = _fill_masterstars_gaia_matched_bp_rp_from_local_db(
            df_final,
            gaia_db_path=_gdb_fill,
        )
        if _n_bp_miss > 0:
            log_event(f"masterstars bp_rp fallback: {_n_bp_fill}/{_n_bp_miss} doplnenych z Gaia DB")
        _fleming_sigma: float | None = None
        if _gdb_fill:
            try:
                from dao_reconcile import (  # noqa: PLC0415
                    annotate_dao_only_magnitude_classes,
                    compute_gaia_dao_reconcile,
                    fit_fleming_completeness,
                    format_dao_only_census_log,
                    resolve_effective_match_depth,
                )

                _cone_df_cls = None
                _cone_csv_cls = Path(platesolve_dir) / "field_catalog_cone.csv"
                if _cone_csv_cls.is_file():
                    _cone_df_cls = read_vyvar_csv(_cone_csv_cls, low_memory=False, dtype={"catalog_id": str})
                _fwhm_cls = float(det_meta.get("dao_fwhm_px") or 0.0)
                if not (_fwhm_cls > 0.0):
                    _fwhm_cls = float(header_core_fwhm_px(hdr) or 3.5)
                _md_cls = resolve_effective_match_depth(det_meta, is_masterstar=True)
                _cone_lim_cls: float | None = None
                try:
                    _raw_lim = det_meta.get("faintest_mag_limit")
                    if _raw_lim is not None and math.isfinite(float(_raw_lim)):
                        _cone_lim_cls = float(_raw_lim)
                except (TypeError, ValueError):
                    _cone_lim_cls = None
                _noise_cls = det_meta.get("noise_floor_adu")
                _nax1_cls = int(hdr.get("NAXIS1") or 0)
                _nax2_cls = int(hdr.get("NAXIS2") or 0)
                _wcs_cls = None
                _plate_cls = None
                if _has_valid_wcs(hdr) and _nax1_cls > 0 and _nax2_cls > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        _wcs_cls = WCS(hdr)
                    try:
                        from astropy.wcs.utils import proj_plane_pixel_scales

                        _plate_cls = float(np.mean(proj_plane_pixel_scales(_wcs_cls) * 3600.0))
                    except Exception:  # noqa: BLE001
                        pass
                if _wcs_cls is not None:
                    _recon_ms = compute_gaia_dao_reconcile(
                        df_final,
                        gaia_db_path=_gdb_fill,
                        wcs=_wcs_cls,
                        naxis1=_nax1_cls,
                        naxis2=_nax2_cls,
                        fwhm_px=_fwhm_cls,
                        plate_scale_arcsec=_plate_cls,
                        mag_limit=float(det_meta.get("faintest_mag_limit") or 18.0),
                        match_sep_arcsec=float(
                            det_meta.get("match_sep_arcsec_effective")
                            or det_meta.get("match_sep_arcsec_requested")
                            or 8.0
                        ),
                        cone_df=_cone_df_cls,
                    )
                    _recon_ms.update(_md_cls)
                    _ff = fit_fleming_completeness(_recon_ms.get("completeness_curve") or [])
                    _fleming_sigma = _ff.sigma_mag
                df_final, _dao_class_meta = annotate_dao_only_magnitude_classes(
                    df_final,
                    gaia_db_path=_gdb_fill,
                    effective_match_depth=_md_cls.get("match_depth"),
                    cone_query_mag_limit=_cone_lim_cls,
                    fleming_sigma_mag=_fleming_sigma,
                    frame_noise_adu=_noise_cls,
                )
                if _recon_ms is not None:
                    _recon_ms["dao_only_class_meta"] = _dao_class_meta
                _ms_info_msg = format_dao_only_census_log(_dao_class_meta, n_total=len(df_final))
                LOGGER.info(_ms_info_msg)
                log_event(_ms_info_msg)
            except Exception as _ms_census_exc:  # noqa: BLE001
                LOGGER.debug("[MASTERSTAR-DAO-CENSUS] skipped: %s", _ms_census_exc)
                try:
                    from invariants_runtime import dao_only_fraction_from_masterstars  # noqa: PLC0415

                    _frac_ms = float(dao_only_fraction_from_masterstars(df_final))
                    _n_dao_ms = int(round(_frac_ms * float(len(df_final))))
                    _ms_info_msg = (
                        f"MASTERSTAR DAO_ONLY census: {_n_dao_ms}/{len(df_final)} "
                        f"(fraction={_frac_ms:.3f}) -- informational, not a gate"
                    )
                    LOGGER.info(_ms_info_msg)
                    log_event(_ms_info_msg)
                except Exception:  # noqa: BLE001
                    pass
        else:
            try:
                from invariants_runtime import dao_only_fraction_from_masterstars  # noqa: PLC0415

                _frac_ms = float(dao_only_fraction_from_masterstars(df_final))
                _n_dao_ms = int(round(_frac_ms * float(len(df_final))))
                _ms_info_msg = (
                    f"MASTERSTAR DAO_ONLY census: {_n_dao_ms}/{len(df_final)} "
                    f"(fraction={_frac_ms:.3f}) -- informational, not a gate"
                )
                LOGGER.info(_ms_info_msg)
                log_event(_ms_info_msg)
            except Exception as _ms_census_exc:  # noqa: BLE001
                LOGGER.debug("[MASTERSTAR-DAO-CENSUS] skipped: %s", _ms_census_exc)
    except Exception as exc:  # noqa: BLE001
        from invariants_runtime import InvariantViolation  # noqa: PLC0415

        if isinstance(exc, InvariantViolation):
            raise
        _cone_gaia_fail = Path(platesolve_dir) / "field_catalog_cone.csv"
        if _cone_gaia_fail.is_file() and _has_valid_wcs(hdr):
            raise RuntimeError(
                f"MASTERSTAR Gaia-complete enrich failed on production path: {exc!s}"
            ) from exc
        LOGGER.exception("[M1] MASTERSTAR source_type annotate failed: %s", exc)
        log_event(f"MASTERSTAR source_type annotate failed: {exc!s}")
    _vyvar_df_to_csv(df_final, csv_path)
    _n_det = int(len(df_final))
    _n_mat = int(
        df_final.get("catalog_id", pd.Series([""] * len(df_final)))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    _rate = (100.0 * float(_n_mat) / float(_n_det)) if _n_det > 0 else 0.0
    _cat_rows_opt = int(det_meta.get("catalog_rows", 0) or 0)
    if "catalog_id" in df_final.columns:
        _cid_opt = df_final["catalog_id"].fillna("").astype(str).str.strip()
        _n_gaia_det_opt = int(_cid_opt[_cid_opt != ""].nunique())
    else:
        _n_gaia_det_opt = int(_n_mat)
    _gaia_rate_opt = (100.0 * float(_n_gaia_det_opt) / float(_cat_rows_opt)) if _cat_rows_opt > 0 else 0.0
    log_event(
        f"[chart] MATCH STATS (optimized): Found {_n_det} stars on image | {_n_mat} matched with Gaia | "
        f"Match Rate: {_rate:.2f}% | Gaia->DAO: {_gaia_rate_opt:.2f}% ({_n_gaia_det_opt}/{_cat_rows_opt})"
    )
    if _cat_rows_opt > 0:
        LOGGER.info(
            "[MASTERSTAR] Gaia->DAO completeness: "
            "%d/%d (%.1f%%) | catalog_only: %d",
            _n_gaia_det_opt,
            _cat_rows_opt,
            _gaia_rate_opt,
            _cat_rows_opt - _n_gaia_det_opt,
        )
    log_event(
        f"MASTERSTAR JSON consistency: n_raw_dao={int(det_meta.get('n_detected_dao_raw', 0) or 0)}, "
        f"detected_stars={_n_det}, catalog_matched={_n_mat}, "
        f"gaia_dao_completeness_pct={round(_gaia_rate_opt, 2) if _cat_rows_opt > 0 else None}, "
        f"n_gaia_undetected={(_cat_rows_opt - _n_gaia_det_opt) if _cat_rows_opt > 0 else None}"
    )
    # TODO-25: persist to pipeline_meta.json so UI can read single source of truth
    if _cat_rows_opt > 0:
        _meta_patch: dict[str, Any] = {
            "gaia_dao_completeness_raw_pct": round(float(_gaia_rate_opt), 2),
            "catalog_rows": int(_cat_rows_opt),
            "n_gaia_detected": int(_n_gaia_det_opt),
            "n_gaia_undetected": int(_cat_rows_opt - _n_gaia_det_opt),
        }
        _idg_stamp = dict(det_meta.get("identity_gate") or {})
        _gmeta = det_meta.get("gaia_census_meta") or {}
        if isinstance(_gmeta, dict) and "n_lock_geometry_reject" in _gmeta:
            _idg_stamp["n_lock_geometry_reject"] = int(_gmeta.get("n_lock_geometry_reject") or 0)
        if _idg_stamp:
            _meta_patch["identity_gate"] = _idg_stamp
        try:
            from dao_gaia_calibration import effective_tol_stamps  # noqa: PLC0415

            _meta_patch["dao_gaia_tol"] = effective_tol_stamps(
                det_meta.get("dao_gaia_derived_tol")
                if isinstance(det_meta.get("dao_gaia_derived_tol"), dict)
                else None,
                _cfg_ms,
                fwhm_px=float(det_meta.get("dao_fwhm_px") or _idg_stamp.get("fwhm_px") or 3.5),
                census_meta=_gmeta if isinstance(_gmeta, dict) else None,
            )
        except Exception:  # noqa: BLE001
            pass
        for _mk in (
            "match_sep_arcsec_requested",
            "match_sep_arcsec_effective",
            "match_sep_formula_inputs",
            "wcs_gaia_pixel_refine_iters",
        ):
            if det_meta.get(_mk) is not None:
                _meta_patch[_mk] = det_meta.get(_mk)
        if _opt_stats_last:
            _meta_patch["optimizer_refit"] = dict(_opt_stats_last)
        if _wcs_rt_p99 is not None:
            _meta_patch["wcs_roundtrip_p99_px"] = float(_wcs_rt_p99)
            _meta_patch["wcs_roundtrip_pass"] = bool(_wcs_rt_pass)
        if _identity_qa:
            _inv_wcs_recs = _identity_qa.pop("_inv_wcs_01", None)
            _meta_patch.update(_identity_qa)
            if _inv_wcs_recs:
                _meta_patch.setdefault("invariants", [])
                if isinstance(_meta_patch["invariants"], list):
                    _meta_patch["invariants"].extend(list(_inv_wcs_recs))
        try:
            if _recon_ms is not None:
                _meta_patch.update(reconcile_to_pipeline_meta(_recon_ms))
            elif _dao_class_meta:
                from dao_reconcile import dao_only_class_meta_flat  # noqa: PLC0415

                _meta_patch.update(dao_only_class_meta_flat(_dao_class_meta))
            else:
                _cone_df = None
                _cone_csv = Path(platesolve_dir) / "field_catalog_cone.csv"
                if _cone_csv.is_file():
                    _cone_df = read_vyvar_csv(_cone_csv, low_memory=False, dtype={"catalog_id": str})
                _fwhm_recon = float(det_meta.get("dao_fwhm_px") or 0.0)
                if not (_fwhm_recon > 0.0):
                    _fwhm_recon = float(header_core_fwhm_px(hdr) or 3.5)
                _wcs_recon = None
                _plate_recon = None
                _nax1 = int(hdr.get("NAXIS1") or 0)
                _nax2 = int(hdr.get("NAXIS2") or 0)
                if _has_valid_wcs(hdr) and _nax1 > 0 and _nax2 > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        _wcs_recon = WCS(hdr)
                    try:
                        from astropy.wcs.utils import proj_plane_pixel_scales

                        _plate_recon = float(np.mean(proj_plane_pixel_scales(_wcs_recon) * 3600.0))
                    except Exception:  # noqa: BLE001
                        pass
                _gdb_recon = str(_cfg_ms.gaia_db_path or "").strip()
                _faintest_recon = float(det_meta.get("faintest_mag_limit") or 18.0)
                _match_sep_recon = float(
                    det_meta.get("match_sep_arcsec_effective")
                    or det_meta.get("match_sep_arcsec_requested")
                    or 8.0
                )
                if _wcs_recon is not None and _gdb_recon:
                    _recon = compute_gaia_dao_reconcile(
                        df_final,
                        gaia_db_path=_gdb_recon,
                        wcs=_wcs_recon,
                        naxis1=_nax1,
                        naxis2=_nax2,
                        fwhm_px=_fwhm_recon,
                        plate_scale_arcsec=_plate_recon,
                        mag_limit=_faintest_recon,
                        match_sep_arcsec=_match_sep_recon,
                        cone_df=_cone_df,
                    )
                    _md = resolve_effective_match_depth(det_meta, is_masterstar=True)
                    _recon.update(_md)
                    _meta_patch.update(reconcile_to_pipeline_meta(_recon))
        except Exception as exc:  # noqa: BLE001
            log_event(f"MASTERSTAR Gaia reconcile decomposition skipped: {exc!s}")
            _meta_patch["gaia_dao_completeness_pct"] = round(float(_gaia_rate_opt), 2)
        merge_photometry_pipeline_meta(
            Path(platesolve_dir) / "photometry",
            _meta_patch,
            _cfg_ms,
            entry_point="generate_masterstar_and_catalog",
        )
        # INV-DAG-01: masterstar stage stamp (cold-start OK if earlier stages absent).
        try:
            from invariants_runtime import stamp_stage_on_disk  # noqa: PLC0415

            stamp_stage_on_disk(
                Path(platesolve_dir) / "photometry",
                "masterstar",
                enforce_upstream=True,
            )
        except Exception as _dag_exc:  # noqa: BLE001
            logging.debug("[INV-DAG-01] masterstar stamp skipped: %s", _dag_exc)
    log_event(
        f"MASTERSTAR katalog: {Path(csv_path).name} - {len(df_final)} riadkov "
        f"(DAO + katalog na celom poli; ziadne orezanie podla vzdialenosti od stredu snimku)."
    )
    # Gaussian FWHM (2D fit) -> hlavicka; VY_FWHM je DAO odhad, nie moment FWHM - nepouzivaj 0.619.
    masterstars_df = df_final
    if (
        masterstars_df is None
        or len(masterstars_df) == 0
        or "x" not in masterstars_df.columns
        or "y" not in masterstars_df.columns
    ):
        masterstars_df = df_out
    try:
        from photometry_phase2a import measure_fwhm_from_masterstar

        _ms_path = Path(masterstar_fits)
        if "mag" in masterstars_df.columns:
            _star_pos = masterstars_df[["x", "y", "mag"]].dropna().head(50)
        elif "phot_g_mean_mag" in masterstars_df.columns:
            _star_pos = (
                masterstars_df[["x", "y", "phot_g_mean_mag"]]
                .dropna()
                .rename(columns={"phot_g_mean_mag": "mag"})
                .head(50)
            )
        else:
            _star_pos = masterstars_df[["x", "y"]].dropna().head(50)
        with fits.open(_ms_path, memmap=False) as _hint_hdul:
            _vy_hint = _hint_hdul[0].header.get("VY_FWHM", 3.5)
            _vy_fwhm_hint = float(_vy_hint) if _vy_hint is not None else 3.5
        _gaussian_fwhm = measure_fwhm_from_masterstar(
            _ms_path,
            _star_pos,
            dao_fwhm_hint=_vy_fwhm_hint,
            n_stars=30,
        )
        with fits.open(_ms_path, mode="update", memmap=False) as _hdul:
            _hdul[0].header["VY_FWHM_GAUSS"] = (
                round(float(_gaussian_fwhm), 4),
                "Gaussian FWHM px (2D fit)",
            )
            _n_raw_dao_hdr = int(det_meta.get("n_detected_dao_raw", 0) or 0)
            _hdul[0].header["VY_NDAO"] = (
                _n_raw_dao_hdr,
                "VYVAR: raw DAO detections on MASTERSTAR (stars/Mpx density)",
            )
            _hdul.flush()
        logging.info(
            f"[MASTERSTAR] VY_FWHM_GAUSS={float(_gaussian_fwhm):.3f}px ulozene do hlavicky (2D fit)"
        )
    except Exception as e:  # noqa: BLE001
        logging.error('[EXC-0409] Cross-setup `comparison_stars.csv` sync failure leaves B/V/R setups with inconsistent c...: %s', e)
        log_event(f"[ERROR] VY_FWHM_GAUSS fit ZLYHAL: {e}\n{traceback.format_exc()}")
    try:
        _ms_path_tag = Path(masterstar_fits)
        _n_raw_dao_hdr = int(det_meta.get("n_detected_dao_raw", 0) or 0)
        with fits.open(_ms_path_tag, mode="update", memmap=False) as _hdul_tag:
            if "VY_NDAO" not in _hdul_tag[0].header:
                _hdul_tag[0].header["VY_NDAO"] = (
                    _n_raw_dao_hdr,
                    "VYVAR: raw DAO detections on MASTERSTAR (stars/Mpx density)",
                )
                _hdul_tag.flush()
    except Exception:  # noqa: BLE001
        pass
    # Small flush pause: UI may read CSV immediately after this returns.
    time.sleep(0.5)
    # Drop stale pre-optimizer dataframe to avoid accidental reuse ("ghost rows").
    try:
        del df_out
    except Exception:  # noqa: BLE001
        pass
    # Keep platesolve clean: remove temporary/duplicate artifacts.
    for _dup in (
        platesolve_dir / "MASTERSTAR_full.fits",
        platesolve_dir / "MASTERSTAR_full.jpg",
        temp_csv,
    ):
        try:
            if Path(_dup).is_file() and Path(_dup).resolve() != Path(csv_path).resolve():
                Path(_dup).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    photo_plan = write_photometry_plan_files(
        platesolve_dir=platesolve_dir,
        masterstar_fits=masterstar_fits,
        masterstars_csv=csv_path,
        n_comparison_stars=int(n_comparison_stars),
        require_non_variable=bool(require_non_variable_comparisons),
    )
    # F-431 / C1: stamp AFTER VT exists (write_photometry_plan_files).
    try:
        _vt_stamp_path = platesolve_dir / "variable_targets.csv"
        if _vt_stamp_path.is_file():
            from photometry_core import stamp_vsx_known_variable_on_masterstars  # noqa: PLC0415

            _ms_for_stamp = pd.read_csv(
                csv_path, low_memory=False, dtype={"catalog_id": str, "name": str}
            )
            _vt_stamp_df = pd.read_csv(_vt_stamp_path, low_memory=False, dtype={"catalog_id": str})
            _ms_for_stamp, _vsx_stamp_final = stamp_vsx_known_variable_on_masterstars(
                _ms_for_stamp,
                _vt_stamp_df,
                log_fn=log_event,
            )
            _vyvar_df_to_csv(_ms_for_stamp, csv_path)
            df_final = _ms_for_stamp
            log_event(
                f"MASTERSTAR VSX catalog_id stamp (post-VT): "
                f"id_join={_vsx_stamp_final.get('id_join')} "
                f"positional_fallback={_vsx_stamp_final.get('positional_fallback')}"
            )
        else:
            log_event("MASTERSTAR VSX stamp skipped: variable_targets.csv missing after photometry plan.")
    except Exception as _vsx_final_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR VSX catalog_id stamp (post-VT) skipped: {_vsx_final_exc!s}")
    # Multi-filter support: keep comparison stars consistent across platesolve/<setup>/ folders.
    try:
        _sync_comparison_stars_across_setups(Path(platesolve_dir).parent)
    except Exception as _sync_exc:  # noqa: BLE001
        log_event(
            f"MASTERSTAR: comparison-star cross-setup sync failed ({_sync_exc!s}); "
            "B/V/R comp sets may be inconsistent across setups."
        )

    out: dict[str, Any] = {
        "masterstar_fits": str(masterstar_fits),
        "masterstars_csv": str(csv_path),
        "frames_used": int(info.get("frames_used", 0)),
        "masterstar_selection": ms_selection_meta or None,
        "n_raw_dao": int(det_meta.get("n_detected_dao_raw", 0) or 0),
        "detected_stars": int(_n_det),
        "catalog_matched": int(_n_mat),
        "catalog_rows": int(det_meta.get("catalog_rows", 0)),
        "catalog_match_max_sep_arcsec": float(_match_sep_eff),
        "max_catalog_rows": int(_ms_max_catalog_rows_eff),
        "n_likely_saturated": int(det_meta.get("n_likely_saturated", 0)),
        "saturate_limit_adu": det_meta.get("saturate_limit_adu"),
        "saturate_limit_source": det_meta.get("saturate_limit_source"),
        "solve": solve_meta,
        "n_comparison_stars_requested": int(n_comparison_stars),
        "faintest_mag_limit": det_meta.get("faintest_mag_limit"),
        "n_dropped_fainter_than_limit": det_meta.get("n_dropped_fainter_than_limit"),
        "field_catalog_cone_csv": det_meta.get("field_catalog_cone_csv"),
        "catalog_derived_membership": det_meta.get("catalog_derived_membership"),
        "dao_threshold_sigma": det_meta.get("dao_threshold_sigma"),
        "masterstar_match_png": "",
    }
    out.update(photo_plan)
    # Enrichment columns for masterstars_full_match.csv (formerly MASTER_SOURCES DB).
    try:
        gaia_db = str(_cfg_ms.gaia_db_path or "").strip()
        if gaia_db and draft_id is not None and "ra_deg" in df_final.columns and "dec_deg" in df_final.columns:
            det = df_final.copy()
            det["ra_deg"] = pd.to_numeric(det["ra_deg"], errors="coerce")
            det["dec_deg"] = pd.to_numeric(det["dec_deg"], errors="coerce")
            det = det[det["ra_deg"].notna() & det["dec_deg"].notna()].copy()
            if not det.empty:
                ra_min = float(det["ra_deg"].min()) - 0.01
                ra_max = float(det["ra_deg"].max()) + 0.01
                de_min = float(det["dec_deg"].min()) - 0.01
                de_max = float(det["dec_deg"].max()) + 0.01
                ga = query_local_gaia(
                    gaia_db,
                    ra_min=ra_min,
                    ra_max=ra_max,
                    dec_min=de_min,
                    dec_max=de_max,
                    mag_limit=None,
                )
                if ga:
                    gdf = pd.DataFrame(ga)
                    gcoo = SkyCoord(
                        ra=pd.to_numeric(gdf["ra"], errors="coerce").astype(float).values * u.deg,
                        dec=pd.to_numeric(gdf["dec"], errors="coerce").astype(float).values * u.deg,
                        frame="icrs",
                    )
                    dcoo = SkyCoord(
                        ra=det["ra_deg"].astype(float).values * u.deg,
                        dec=det["dec_deg"].astype(float).values * u.deg,
                        frame="icrs",
                    )
                    idx, sep2d, _ = dcoo.match_to_catalog_sky(gcoo)
                    ok = sep2d.to(u.arcsec).value <= 2.0
                    if bool(np.any(ok)):
                        # Geometry + blending pruning and dynamic photometric binning.
                        nax1 = int(hdr.get("NAXIS1", 0) or 0) or int(data.shape[1])
                        nax2 = int(hdr.get("NAXIS2", 0) or 0) or int(data.shape[0])
                        border_px = 50.0

                        try:
                            from astropy.coordinates import search_around_sky

                            pairs_i, pairs_j, _, _ = search_around_sky(gcoo, gcoo, 5.0 * u.arcsec)
                            gmag_all = (
                                pd.to_numeric(gdf.get("g_mag"), errors="coerce")
                                .astype(float)
                                .to_numpy()
                            )
                            blended_idx: set[int] = set()
                            for a, b in zip(pairs_i, pairs_j, strict=False):
                                ia = int(a)
                                ib = int(b)
                                if ia == ib:
                                    continue
                                ma = gmag_all[ia] if ia < len(gmag_all) else float("nan")
                                mb = gmag_all[ib] if ib < len(gmag_all) else float("nan")
                                if not (math.isfinite(ma) and math.isfinite(mb)):
                                    continue
                                if abs(ma - mb) < 3.0:
                                    blended_idx.add(ia)
                                    blended_idx.add(ib)
                        except Exception:  # noqa: BLE001
                            blended_idx = set()

                        filt = str(det_meta.get("filter") or hdr.get("FILTER") or "Clear").strip() or "Clear"
                        if filt.lower() in {"nofilter", "none", "null"}:
                            filt = "Clear"

                        def _bin_step(v: float, step: float) -> float:
                            if not math.isfinite(v):
                                return float("nan")
                            return math.floor((float(v) / float(step)) + 0.5) * float(step)

                        # Saturation threshold for MASTERSTAR (FITS + EQUIPMENTS; no global config fallback)
                        sat_limit = det_meta.get("saturate_limit_adu")
                        if sat_limit is None:
                            _eq_sat_ms = equipment_saturate_adu
                            if _eq_sat_ms is None and equipment_id is not None:
                                _eq_sat_ms = _equipment_saturate_adu_from_db(equipment_id)
                            sat_limit, _ = _effective_saturation_limit(
                                hdr,
                                fallback_adu=None,
                                equipment_saturate_adu=_eq_sat_ms,
                            )
                        if (
                            sat_limit is not None
                            and math.isfinite(float(sat_limit))
                            and float(sat_limit) > 0
                        ):
                            sat_thr = float(sat_limit) * float(saturate_level_fraction)
                        else:
                            # INV-SAT-LIMIT: never admit against +inf.
                            sat_thr = float(SAT_LIMIT_CONTAINER_CLIP_ADU) * float(SAT_LIMIT_NO_KNEE_FRAC)
                            logging.warning(
                                "[INV-SAT-LIMIT] MASTERSTAR sat_thr unresolved; "
                                "using peak-test %.1f ADU (0.80 x container clip)",
                                sat_thr,
                            )

                        rows_ms: list[dict[str, Any]] = []
                        det_ok = det.iloc[np.where(ok)[0]].reset_index(drop=True)
                        g_ok = gdf.iloc[idx[np.where(ok)[0]]].reset_index(drop=True)
                        g_ok_idx = idx[np.where(ok)[0]]
                        # Aperture optimization: estimate per-star FWHM on MASTERSTAR, then take medians per color.
                        try:
                            import numpy as _np

                            arr_ms = _np.asarray(data, dtype=_np.float32)

                            fwhm_est = [
                                _fwhm_moment_at(
                                    arr_ms,
                                    float(det_ok["x"].iloc[i]) if "x" in det_ok.columns and pd.notna(det_ok["x"].iloc[i]) else float("nan"),
                                    float(det_ok["y"].iloc[i]) if "y" in det_ok.columns and pd.notna(det_ok["y"].iloc[i]) else float("nan"),
                                    half=6,
                                )
                                for i in range(len(det_ok))
                            ]
                            fwhm_med_px = float(_np.nanmedian(_np.asarray(fwhm_est, dtype=_np.float64)))
                        except Exception:  # noqa: BLE001
                            fwhm_est = [float("nan")] * len(det_ok)
                            fwhm_med_px = float("nan")

                        if not (math.isfinite(fwhm_med_px) and fwhm_med_px > 0):
                            try:
                                fwhm_med_px = float(det_meta.get("dao_fwhm_px") or _ms_fwhm)
                            except Exception:  # noqa: BLE001
                                fwhm_med_px = float(_ms_fwhm)
                        if not (math.isfinite(fwhm_med_px) and fwhm_med_px > 0):
                            fwhm_med_px = float(_ms_fwhm)

                        # Median per coarse color category.
                        def _color_bucket(bp_rp: float) -> str:
                            if not math.isfinite(bp_rp):
                                return "neutral"
                            if bp_rp < 0.5:
                                return "blue"
                            if bp_rp <= 1.5:
                                return "neutral"
                            return "red"

                        by_col: dict[str, list[float]] = {"blue": [], "neutral": [], "red": []}
                        for i in range(len(det_ok)):
                            bprp_v0 = (
                                float(g_ok["bp_rp"].iloc[i])
                                if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i])
                                else float("nan")
                            )
                            fe = float(fwhm_est[i]) if i < len(fwhm_est) else float("nan")
                            if math.isfinite(fe) and fe > 0:
                                by_col[_color_bucket(bprp_v0)].append(fe)
                        fwhm_blue = float(_np.median(by_col["blue"])) if by_col["blue"] else fwhm_med_px
                        fwhm_neu = float(_np.median(by_col["neutral"])) if by_col["neutral"] else fwhm_med_px
                        fwhm_red = float(_np.median(by_col["red"])) if by_col["red"] else fwhm_med_px

                        # Gaia neighbour veto radius in arcsec: 3x median FWHM (px) x plate scale.
                        try:
                            from astropy.wcs.utils import proj_plane_pixel_scales

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", FITSFixedWarning)
                                _w_ms = WCS(hdr)
                            sc = proj_plane_pixel_scales(_w_ms.celestial)  # deg/pix
                            scale_arcsec_per_px = float(_np.nanmedian(_np.asarray(sc, dtype=_np.float64))) * 3600.0
                        except Exception:  # noqa: BLE001
                            scale_arcsec_per_px = float("nan")
                        veto_radius_arcsec = (
                            float(3.0 * fwhm_med_px * scale_arcsec_per_px)
                            if math.isfinite(scale_arcsec_per_px) and scale_arcsec_per_px > 0
                            else float("nan")
                        )
                        import numpy as _np
                        # photometry.py is legacy; use the merged core module.
                        from photometry_core import bad_columns_for_light_frame

                        _thr_nl = float("nan")
                        _peaks_nl: list[float] = []
                        for _i in range(len(det_ok)):
                            _pm = (
                                det_ok["peak_max_adu"].iloc[_i]
                                if "peak_max_adu" in det_ok.columns and pd.notna(det_ok["peak_max_adu"].iloc[_i])
                                else None
                            )
                            if _pm is not None and math.isfinite(float(_pm)):
                                _peaks_nl.append(float(_pm))
                        if _peaks_nl:
                            _pp = float(_cfg_ms.nonlinearity_peak_percentile)
                            _thr_nl = float(
                                _np.percentile(
                                    _np.asarray(_peaks_nl, dtype=_np.float64),
                                    min(100.0, max(0.0, 100.0 - _pp)),
                                )
                            )
                        _nl_ratio = float(_cfg_ms.nonlinearity_fwhm_ratio)
                        _bpm_js = None
                        if master_dark_path and str(master_dark_path).strip():
                            _mdp = Path(str(master_dark_path))
                            _bj = _mdp.parent / f"{_mdp.stem}_dark_bpm.json"
                            if _bj.is_file():
                                try:
                                    _bpm_js = json.loads(_bj.read_text(encoding="utf-8"))
                                except Exception:  # noqa: BLE001
                                    _bpm_js = None
                        _bad_x = bad_columns_for_light_frame(_bpm_js, light_header=hdr)
                        for i in range(len(det_ok)):
                            x = float(det_ok["x"].iloc[i]) if "x" in det_ok.columns else None
                            y = float(det_ok["y"].iloc[i]) if "y" in det_ok.columns else None
                            pmax = float(det_ok["peak_max_adu"].iloc[i]) if "peak_max_adu" in det_ok.columns and pd.notna(det_ok["peak_max_adu"].iloc[i]) else None
                            is_sat = 1 if (pmax is not None and math.isfinite(pmax) and pmax > sat_thr) else 0
                            var_flag = str(g_ok.get("var_flag").iloc[i]) if "var_flag" in g_ok.columns else ""
                            is_var = 1 if var_flag.strip() not in ("", "0", "False", "false", "NO", "No") else 0

                            is_border = (
                                x is not None
                                and y is not None
                                and (
                                    x < border_px
                                    or y < border_px
                                    or x > float(nax1) - border_px
                                    or y > float(nax2) - border_px
                                )
                            )
                            gi = int(g_ok_idx[i]) if i < len(g_ok_idx) else -1
                            is_blend = bool(gi in blended_idx) if gi >= 0 else False
                            excl = "Border" if is_border else ("Blended" if is_blend else None)

                            fe_i = float(fwhm_est[i]) if i < len(fwhm_est) else float("nan")
                            likely_nl = False
                            if (
                                math.isfinite(fe_i)
                                and math.isfinite(fwhm_med_px)
                                and fwhm_med_px > 0
                                and pmax is not None
                                and math.isfinite(float(pmax))
                                and math.isfinite(_thr_nl)
                                and float(pmax) >= _thr_nl
                                and fe_i > _nl_ratio * fwhm_med_px
                            ):
                                likely_nl = True
                            on_bad = False
                            if x is not None and _bad_x:
                                if int(round(float(x))) in _bad_x:
                                    on_bad = True

                            # New Gaia stability/multiplicity filters.
                            gfer = None
                            if "g_flux_error_rel" in g_ok.columns and pd.notna(g_ok["g_flux_error_rel"].iloc[i]):
                                try:
                                    gfer = float(g_ok["g_flux_error_rel"].iloc[i])
                                except (TypeError, ValueError):
                                    gfer = None
                            nss = 0
                            if "non_single_star" in g_ok.columns and pd.notna(g_ok["non_single_star"].iloc[i]):
                                try:
                                    nss = int(float(g_ok["non_single_star"].iloc[i]))
                                except (TypeError, ValueError):
                                    nss = 0
                            pvf = ""
                            if "phot_variable_flag" in g_ok.columns and pd.notna(g_ok["phot_variable_flag"].iloc[i]):
                                pvf = str(g_ok["phot_variable_flag"].iloc[i]).strip()

                            if excl is None:
                                if gfer is not None and math.isfinite(gfer) and float(gfer) > 0.02:
                                    excl = "CatalogNoise"
                                elif int(nss) > 0:
                                    excl = "NonSingle"
                                elif pvf.upper() == "VARIABLE":
                                    excl = "Variable"
                                else:
                                    # Neighbour veto: exclude if Gaia neighbour would change mag by > 0.001.
                                    if (
                                        veto_radius_arcsec is not None
                                        and math.isfinite(float(veto_radius_arcsec))
                                        and float(veto_radius_arcsec) > 0
                                        and "g_mag" in g_ok.columns
                                        and pd.notna(g_ok["g_mag"].iloc[i])
                                    ):
                                        try:
                                            m0 = float(g_ok["g_mag"].iloc[i])
                                        except (TypeError, ValueError):
                                            m0 = float("nan")
                                        if math.isfinite(m0):
                                            try:
                                                from astropy.coordinates import search_around_sky as _sas

                                                # Query neighbours in the Gaia window itself (fast; same gcoo).
                                                # Use the matched Gaia index gi.
                                                gi2 = int(g_ok_idx[i]) if i < len(g_ok_idx) else -1
                                                if gi2 >= 0 and gi2 < len(gcoo):
                                                    c0 = gcoo[gi2]
                                                    _, jj, _, _ = _sas(
                                                        c0,
                                                        gcoo,
                                                        float(veto_radius_arcsec) * u.arcsec,
                                                    )
                                                    ratios: list[float] = []
                                                    for jx in list(jj):
                                                        j = int(jx)
                                                        if j == gi2:
                                                            continue
                                                        try:
                                                            mj = float(gdf["g_mag"].iloc[j])
                                                        except Exception:  # noqa: BLE001
                                                            continue
                                                        if not math.isfinite(mj):
                                                            continue
                                                        ratios.append(10.0 ** (-0.4 * (mj - m0)))
                                                    if ratios:
                                                        dm = -2.5 * math.log10(1.0 + float(sum(ratios)))
                                                        if abs(dm) > 0.001:
                                                            excl = "Gaia neighbor blend"
                                            except Exception:  # noqa: BLE001
                                                pass
                            if likely_nl and excl is None:
                                excl = "Nonlinear FWHM"
                            if on_bad and excl is None:
                                excl = "Bad column"
                            safe = 0 if excl is not None else 1

                            gmag_v = (
                                float(g_ok["g_mag"].iloc[i])
                                if "g_mag" in g_ok.columns and pd.notna(g_ok["g_mag"].iloc[i])
                                else float("nan")
                            )
                            bprp_v = (
                                float(g_ok["bp_rp"].iloc[i])
                                if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i])
                                else float("nan")
                            )
                            mb = _bin_step(gmag_v, 0.5)
                            cb = _bin_step(bprp_v, 0.25)
                            phot_cat = (
                                f"{filt}_mag_{mb:.1f}_col_{cb:.2f}"
                                if math.isfinite(mb) and math.isfinite(cb)
                                else f"{filt}_mag_nan_col_nan"
                            )
                            rows_ms.append(
                                {
                                    "x_master": x,
                                    "y_master": y,
                                    "ra": float(g_ok["ra"].iloc[i]) if pd.notna(g_ok["ra"].iloc[i]) else float(det_ok["ra_deg"].iloc[i]),
                                    "dec": float(g_ok["dec"].iloc[i]) if pd.notna(g_ok["dec"].iloc[i]) else float(det_ok["dec_deg"].iloc[i]),
                                    "g_mag": float(g_ok["g_mag"].iloc[i]) if "g_mag" in g_ok.columns and pd.notna(g_ok["g_mag"].iloc[i]) else None,
                                    "bp_rp": float(g_ok["bp_rp"].iloc[i]) if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i]) else None,
                                    "is_var": is_var,
                                    "is_saturated": is_sat,
                                    "source_id_gaia": str(g_ok["source_id"].iloc[i]) if "source_id" in g_ok.columns else "",
                                    "g_flux_error_rel": gfer,
                                    "non_single_star": int(nss),
                                    "phot_variable_flag": pvf,
                                    "filter_name": filt,
                                    "phot_category": phot_cat,
                                    "recommended_aperture": recommended_aperture_by_color(
                                        bp_rp=bprp_v if math.isfinite(bprp_v) else None,
                                        median_fwhm_blue=fwhm_blue,
                                        median_fwhm_neutral=fwhm_neu,
                                        median_fwhm_red=fwhm_red,
                                    ),
                                    "is_safe_comp": safe,
                                    "exclusion_reason": excl,
                                    "safe_override": 0,
                                    "likely_nonlinear": 1 if likely_nl else 0,
                                    "on_bad_column": 1 if on_bad else 0,
                                }
                            )
                        try:
                            from masterstars_enrichment import (  # noqa: PLC0415
                                apply_common_field_bbox_exclusion,
                                apply_stress_rms_to_rows_ms,
                                apply_vsx_variable_flags,
                                merge_enrichment_into_masterstars_df,
                            )

                            df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                            _vyvar_df_to_csv(df_final, csv_path)
                            out["masterstars_enrichment_written"] = int(len(rows_ms))
                            try:
                                _wp2 = write_photometry_plan_files(
                                    platesolve_dir=platesolve_dir,
                                    masterstar_fits=masterstar_fits,
                                    masterstars_csv=csv_path,
                                    n_comparison_stars=int(n_comparison_stars),
                                    require_non_variable=bool(require_non_variable_comparisons),
                                    draft_id=int(draft_id),
                                )
                                out.update(_wp2)
                            except Exception as _wp2_exc:  # noqa: BLE001
                                log_event(
                                    f"MASTERSTAR: enriched photometry-plan rewrite failed ({_wp2_exc!s}); "
                                    "keeping the prior photometry plan."
                                )
                        except Exception as exc:  # noqa: BLE001
                            out["masterstars_enrichment_error"] = str(exc)

                        # Stress-test: 10% random sample, exclude Border/Blended by default (soft-crop).
                        try:
                            from masterstars_enrichment import merge_enrichment_into_masterstars_df  # noqa: PLC0415

                            root_frames = (
                                Path(source_root)
                                if source_root is not None
                                else (Path(detrended_root) if detrended_root is not None else ap)
                            )
                            # Common field intersection bbox across MASTERSTAR input frames (finite data overlap).
                            try:
                                _ms_inputs: list[Path] = []
                                if only_ms_paths is not None:
                                    _ms_inputs = [Path(p) for p in only_ms_paths if Path(p).is_file()]
                                else:
                                    # Fallback: approximate using a subset of aligned frames.
                                    _ms_inputs = sorted(_iter_fits_recursive(root_frames))[
                                        : max(2, int(info.get("frames_used", 0)))
                                    ]
                                bbox = common_field_intersection_bbox_px(frame_paths=_ms_inputs, finite_stride=16)
                                if bbox is not None:
                                    x0b, y0b, x1b, y1b = bbox
                                    apply_common_field_bbox_exclusion(
                                        rows_ms,
                                        x0=float(x0b),
                                        x1=float(x1b),
                                        y0=float(y0b),
                                        y1=float(y1b),
                                    )
                                    df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                                    _vyvar_df_to_csv(df_final, csv_path)
                                    out["common_field_bbox_px"] = [float(x0b), float(y0b), float(x1b), float(y1b)]
                            except Exception as exc:  # noqa: BLE001
                                out["common_field_error"] = str(exc)

                            safe_ids = [
                                str(r.get("source_id_gaia") or "").strip()
                                for r in rows_ms
                                if int(r.get("is_safe_comp") or 0) == 1
                            ]
                            st_res = stress_test_relative_rms_from_sidecars(
                                frames_root=root_frames,
                                source_ids=safe_ids,
                                sample_frac=0.10,
                                seed=42,
                            )
                            out["stress_frames_sampled"] = int(st_res.frames_sampled)
                            out["stress_frames_used"] = int(st_res.frames_used)

                            by_bin: dict[str, list[float]] = {}
                            for rr in rows_ms:
                                if int(rr.get("is_safe_comp") or 0) != 1:
                                    continue
                                sid = str(rr.get("source_id_gaia") or "").strip()
                                if not sid or sid not in st_res.per_source_rms:
                                    continue
                                b = str(rr.get("phot_category") or "").strip()
                                if b:
                                    by_bin.setdefault(b, []).append(float(st_res.per_source_rms[sid]))
                            med_by_bin = {b: float(pd.Series(v).median()) for b, v in by_bin.items() if v}
                            apply_stress_rms_to_rows_ms(rows_ms, st_res.per_source_rms, med_by_bin)

                            packed = [
                                {
                                    "source_id_gaia": rr.get("source_id_gaia"),
                                    "phot_category": rr.get("phot_category"),
                                    "stress_rms": rr.get("stress_rms"),
                                    "ra": rr.get("ra"),
                                    "dec": rr.get("dec"),
                                }
                                for rr in rows_ms
                                if rr.get("stress_rms") is not None
                            ]
                            var_ids = vsx_is_known_variable_top3_per_bin(rows=packed)
                            if var_ids:
                                apply_vsx_variable_flags(rows_ms, set(var_ids))
                                out["vsx_flagged_variables"] = int(len(var_ids))

                            df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                            _vyvar_df_to_csv(df_final, csv_path)
                        except Exception as exc:  # noqa: BLE001
                            out["stress_test_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        out["masterstars_enrichment_error"] = str(exc)
    # Persist MASTERSTAR path on draft for later UI reloads / Step 3 continuity.
    try:
        if draft_id is not None:
            _db_ms = VyvarDatabase(Path(_cfg_ms.database_path))
            try:
                _db_ms.set_obs_draft_masterstar_fits_path(int(draft_id), str(Path(masterstar_fits).resolve()))
            finally:
                _db_ms.conn.close()
    except Exception as exc:  # noqa: BLE001
        out["masterstar_path_store_error"] = str(exc)
    return out


def _pass2_sibling_wcs_recovery(
    *,
    reports: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    job_list: list[dict[str, Any]],
    align_kw: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pass 2: recover failed filter sub-groups via verified sibling WCS + bulk-shift."""
    cfg = align_kw.get("app_config") or AppConfig()
    if not bool(getattr(cfg, "masterstar_sibling_recovery_enabled", True)):
        return reports, skipped
    if len(job_list) <= 1 or not skipped or not reports:
        return reports, skipped

    from vyvar_platesolver import (
        filter_code_from_setup_name,
        pick_sibling_donor_filter,
        try_recover_masterstar_sibling_wcs,
    )

    job_by_gkey = {str(j.get("gkey") or ""): j for j in job_list}
    report_by_gkey: dict[str, dict[str, Any]] = {}
    for rep in reports:
        gk = str(rep.get("observation_group_key") or "")
        if gk:
            report_by_gkey[gk] = rep

    verified_filters: set[str] = set()
    for gk in report_by_gkey:
        setup = Path(gk).name if gk else "(root)"
        flt = filter_code_from_setup_name(setup)
        if flt:
            verified_filters.add(flt)

    still_skipped: list[dict[str, Any]] = []
    archive_path = Path(align_kw["archive_path"])

    for sk in skipped:
        gkey = str(sk.get("gkey") or "")
        setup = str(sk.get("setup") or (Path(gkey).name if gkey else "(root)"))
        recipient_filter = filter_code_from_setup_name(setup)
        job = job_by_gkey.get(gkey)
        if not recipient_filter or job is None:
            still_skipped.append(sk)
            continue

        donor_filter = pick_sibling_donor_filter(recipient_filter, verified_filters)
        if donor_filter is None:
            still_skipped.append(sk)
            continue

        donor_gkey: str | None = None
        donor_report: dict[str, Any] | None = None
        for gk, rep in report_by_gkey.items():
            d_setup = Path(gk).name if gk else "(root)"
            if filter_code_from_setup_name(d_setup) == donor_filter:
                donor_gkey = gk
                donor_report = rep
                break
        donor_job = job_by_gkey.get(donor_gkey or "") if donor_gkey else None
        if donor_report is None or donor_job is None:
            still_skipped.append(sk)
            continue

        platesolve_dir = Path(job["platesolve_dir"])
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        recipient_ms = platesolve_dir / "MASTERSTAR.fits"
        donor_ms_str = str(donor_report.get("masterstar_fits") or "").strip()
        donor_ms = (
            Path(donor_ms_str)
            if donor_ms_str
            else Path(donor_job["platesolve_dir"]) / "MASTERSTAR.fits"
        )

        try:
            if not recipient_ms.is_file():
                generate_masterstar_and_catalog(
                    archive_path=archive_path,
                    source_root=Path(job["detrended_root"]),
                    platesolve_dir=platesolve_dir,
                    setup_name=gkey or None,
                    masterstar_fits_only=True,
                    app_config=cfg,
                    equipment_id=align_kw.get("id_equipment"),
                    draft_id=align_kw.get("draft_id"),
                    master_dark_path=align_kw.get("master_dark_path"),
                    platesolve_backend=str(align_kw.get("platesolve_backend") or "vyvar"),
                    plate_solve_fov_deg=float(
                        align_kw.get("plate_solve_fov_deg") or cfg.plate_solve_fov_deg
                    ),
                )

            _bundle = _plate_solve_input_bundle(
                recipient_ms if recipient_ms.is_file() else donor_ms,
                app_config=cfg,
                equipment_id=align_kw.get("id_equipment"),
                draft_id=align_kw.get("draft_id"),
            )
            with fits.open(recipient_ms, memmap=False) as _hd_ms:
                _ms_hdr = _hd_ms[0].header
                _ms_data = _hd_ms[0].data
            _fov = resolve_plate_solve_fov_deg_hint(
                _ms_hdr,
                int(_ms_data.shape[0]),
                int(_ms_data.shape[1]),
                database_path=cfg.database_path,
                equipment_id=align_kw.get("id_equipment"),
                draft_id=align_kw.get("draft_id"),
            )
            if _fov is None:
                _fov = float(align_kw.get("plate_solve_fov_deg") or cfg.plate_solve_fov_deg)

            rec_result = try_recover_masterstar_sibling_wcs(
                recipient_masterstar_fits=recipient_ms,
                donor_masterstar_fits=donor_ms,
                recipient_filter=recipient_filter,
                donor_filter=donor_filter,
                frame_paths=[Path(f) for f in (job.get("files") or [])],
                app_config=cfg,
                plate_solve_fov_deg=float(_fov),
                expected_plate_scale_arcsec_per_px=_bundle.get("expected_arcsec_per_px"),
                effective_pixel_um=_bundle.get("eff_um"),
                focal_length_mm=_bundle.get("focal_mm"),
            )
            if not rec_result.get("confirmed"):
                still_skipped.append(sk)
                continue

            generate_masterstar_and_catalog(
                archive_path=archive_path,
                platesolve_dir=platesolve_dir,
                setup_name=gkey or None,
                masterstar_skip_build=True,
                masterstar_platesolve_skip_solve=True,
                app_config=cfg,
                equipment_id=align_kw.get("id_equipment"),
                draft_id=align_kw.get("draft_id"),
                platesolve_backend=str(align_kw.get("platesolve_backend") or "vyvar"),
                plate_solve_fov_deg=float(_fov),
                catalog_match_max_sep_arcsec=float(
                    align_kw.get("catalog_match_max_sep_arcsec") or 25.0
                ),
                max_catalog_rows=int(align_kw.get("max_catalog_rows") or 12000),
                dao_threshold_sigma=float(
                    align_kw.get("dao_threshold_sigma") or cfg.masterstar_dao_threshold_sigma
                ),
                catalog_local_gaia_only=align_kw.get("catalog_local_gaia_only"),
            )

            rep = _astrometry_align_impl_body(
                job=job,
                sibling_recovery_use_masterstar=True,
                build_masterstar_and_catalogs=False,
                **{
                    k: v
                    for k, v in align_kw.items()
                    if k not in {"build_masterstar_and_catalogs"}
                },
            )
            rep["sibling_recovery"] = rec_result
            reports.append(rep)
            report_by_gkey[gkey] = rep
            verified_filters.add(recipient_filter)
            log_event(
                f"SIBLING-WCS Pass2: recovered {setup} via donor {donor_filter} - "
                f"n_tight={((rec_result.get('after') or {}).get('n_matched_tight'))}"
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Sibling-WCS Pass2 failed for %s: %s", setup, exc)
            log_event(f"! Sibling-WCS Pass2: set {setup} zostava preskoceny - {exc}")
            still_skipped.append(sk)

    return reports, still_skipped


def _partition_detrended_by_subfolder(files: list[Path], detrended_root: Path) -> dict[str, list[Path]]:
    """Group detrended FITS by full parent subpath under ``detrended_root``.

    This preserves on-disk structure (e.g. ``NoFilter_120_2`` or deeper nested layouts)
    so alignment, zero-level correction and DAO detection run on the exact same file tree.
    """
    root = detrended_root.resolve()
    out: dict[str, list[Path]] = {}
    for fp in files:
        p = Path(fp)
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        parent_rel = rel.parent
        key = "" if str(parent_rel) == "." else parent_rel.as_posix()
        out.setdefault(key, []).append(p)
    for k in out:
        out[k].sort()
    return out


def _merge_astrometry_group_reports(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]
    tot_aligned = sum(int(x.get("aligned_frames") or 0) for x in rows)
    tot_in = sum(int(x.get("input_frames") or 0) for x in rows)
    merged = dict(rows[-1])
    merged["aligned_frames"] = int(tot_aligned)
    merged["input_frames"] = int(tot_in)
    merged["observation_subgroup_reports"] = rows
    merged["masterstar_fits"] = "; ".join(
        str(x.get("masterstar_fits") or "").strip()
        for x in rows
        if str(x.get("masterstar_fits") or "").strip()
    )
    merged["masterstars_csv"] = "; ".join(
        str(x.get("masterstars_csv") or "").strip()
        for x in rows
        if str(x.get("masterstars_csv") or "").strip()
    )
    log_event(
        f"Astrometria: dokoncene {len(rows)} pod-skupin - zarovnanych snimok spolu {tot_aligned} / {tot_in}."
    )
    return merged


def _run_osc_multi_group_alignment(
    *,
    job_list: list[dict[str, Any]],
    osc_bundles: dict[str, dict[str, dict[str, Any]]],
    align_kw: dict[str, Any],
) -> dict[str, Any]:
    """OSC-2: oneRGGB full solve+align first; R/G/B reuse WCS + registration handoff."""
    from osc_align import (
        OSC_REGISTRATION_HANDOFF,
        load_registration_handoff,
        log_channel_match_rate_verification,
        propagate_wcs_between_fits,
        require_osc_donor_products,
        write_wcs_propagation_meta,
    )

    build_ms = bool(align_kw.get("build_masterstar_and_catalogs"))
    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    archive_path = Path(align_kw["archive_path"])
    cfg = align_kw.get("app_config") or AppConfig()

    for base in sorted(osc_bundles.keys(), key=str.casefold):
        bundle = osc_bundles[base]
        one_job = dict(bundle["oneRGGB"])
        one_job["osc_write_registration_handoff"] = True
        _setup = Path(str(one_job.get("gkey") or "")).name or base
        one_rggb_failed = False
        try:
            one_rep = _astrometry_align_impl_body(job=one_job, **align_kw)
            reports.append(one_rep)
        except Exception as exc:  # noqa: BLE001
            one_rggb_failed = True
            LOGGER.error("OSC oneRGGB alignment failed for %s: %s", _setup, exc)
            skipped.append({"gkey": one_job.get("gkey"), "setup": _setup, "skipped_reason": str(exc)})
            continue

        donor_ps = Path(one_job["platesolve_dir"])
        try:
            require_osc_donor_products(donor_ps)
            handoff = load_registration_handoff(donor_ps / OSC_REGISTRATION_HANDOFF)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("OSC donor products missing for %s: %s", base, exc)
            skipped.append({"gkey": one_job.get("gkey"), "setup": _setup, "skipped_reason": str(exc)})
            continue

        for ch in ("R", "G", "B"):
            ch_job = dict(bundle[ch])
            ch_setup = Path(str(ch_job.get("gkey") or "")).name or f"{base}_{ch}"
            ch_ps = Path(ch_job["platesolve_dir"])
            ch_ps.mkdir(parents=True, exist_ok=True)
            try:
                if build_ms:
                    generate_masterstar_and_catalog(
                        archive_path=archive_path,
                        source_root=Path(ch_job["detrended_root"]),
                        platesolve_dir=ch_ps,
                        setup_name=str(ch_job.get("gkey") or "") or None,
                        masterstar_fits_only=True,
                        app_config=cfg,
                        equipment_id=align_kw.get("id_equipment"),
                        draft_id=align_kw.get("draft_id"),
                        master_dark_path=align_kw.get("master_dark_path"),
                        platesolve_backend=str(align_kw.get("platesolve_backend") or "vyvar"),
                        plate_solve_fov_deg=float(
                            align_kw.get("plate_solve_fov_deg") or cfg.plate_solve_fov_deg
                        ),
                    )
                    ch_ms = ch_ps / "MASTERSTAR.fits"
                    propagate_wcs_between_fits(donor_ps / "MASTERSTAR.fits", ch_ms)
                    cat_info = generate_masterstar_and_catalog(
                        archive_path=archive_path,
                        platesolve_dir=ch_ps,
                        setup_name=str(ch_job.get("gkey") or "") or None,
                        masterstar_skip_build=True,
                        masterstar_platesolve_skip_solve=True,
                        app_config=cfg,
                        equipment_id=align_kw.get("id_equipment"),
                        draft_id=align_kw.get("draft_id"),
                        platesolve_backend=str(align_kw.get("platesolve_backend") or "vyvar"),
                        plate_solve_fov_deg=float(
                            align_kw.get("plate_solve_fov_deg") or cfg.plate_solve_fov_deg
                        ),
                        catalog_match_max_sep_arcsec=float(
                            align_kw.get("catalog_match_max_sep_arcsec") or 25.0
                        ),
                        max_catalog_rows=int(align_kw.get("max_catalog_rows") or 12000),
                        dao_threshold_sigma=float(
                            align_kw.get("dao_threshold_sigma") or cfg.masterstar_dao_threshold_sigma
                        ),
                        catalog_local_gaia_only=align_kw.get("catalog_local_gaia_only"),
                    )
                    mr = None
                    if isinstance(cat_info, dict):
                        sm = cat_info.get("solve_meta") or cat_info
                        if isinstance(sm, dict):
                            mr = sm.get("match_rate")
                    write_wcs_propagation_meta(
                        ch_ps,
                        donor_dir=donor_ps,
                        channel=str(ch),
                        match_rate=float(mr) if mr is not None else None,
                    )
                    log_channel_match_rate_verification(
                        channel=str(ch),
                        match_rate=float(mr) if mr is not None else None,
                        one_rggb_failed=one_rggb_failed,
                        log_event=log_event,
                    )
                ch_kw = {
                    k: v
                    for k, v in align_kw.items()
                    if k not in {"build_masterstar_and_catalogs"}
                }
                ch_rep = _astrometry_align_impl_body(
                    job=ch_job,
                    osc_registration_handoff=handoff,
                    sibling_recovery_use_masterstar=True,
                    build_masterstar_and_catalogs=False,
                    **ch_kw,
                )
                ch_rep["osc_channel"] = ch
                ch_rep["osc_donor"] = str(donor_ps)
                reports.append(ch_rep)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("OSC channel %s alignment failed for %s: %s", ch, ch_setup, exc)
                skipped.append({"gkey": ch_job.get("gkey"), "setup": ch_setup, "skipped_reason": str(exc)})

    from osc_align import parse_osc_channel_from_setup

    mono_jobs: list[dict[str, Any]] = []
    for j in job_list:
        gkey = str(j.get("gkey") or "")
        setup = Path(gkey).name if gkey else ""
        _, ch = parse_osc_channel_from_setup(setup)
        if ch is None:
            mono_jobs.append(j)

    for j in mono_jobs:
        try:
            reports.append(_astrometry_align_impl_body(job=j, **align_kw))
        except Exception as exc:  # noqa: BLE001
            setup = Path(str(j.get("gkey") or "")).name if j.get("gkey") else "(root)"
            skipped.append({"gkey": j.get("gkey"), "setup": setup, "skipped_reason": str(exc)})

    if not reports:
        raise RuntimeError(
            "OSC astrometry: no group succeeded. "
            + "; ".join(f"{s.get('setup')}: {s.get('skipped_reason')}" for s in skipped)
        )
    merged = _merge_astrometry_group_reports(reports)
    if skipped:
        merged["skipped_subgroups"] = skipped
    merged["osc_orchestration"] = True
    return merged


def _astrometry_align_impl_body(
    *,
    job: dict[str, Any],
    archive_path: Path,
    astrometry_api_key: str | None = None,
    max_control_points: int = 80,
    min_detected_stars: int = 100,
    max_detected_stars: int = 500,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    max_extra_platesolve: int = 0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    max_catalog_rows: int = 12000,
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    id_equipment: int | None = None,
    draft_id: int | None = None,
    catalog_local_gaia_only: bool | None = None,
    build_masterstar_and_catalogs: bool = False,
    progress_cb: "callable | None" = None,
    ram_align_and_catalog: bool = False,
    app_config: AppConfig | None = None,
    sibling_recovery_use_masterstar: bool = False,
    osc_registration_handoff: dict[str, Any] | None = None,
    osc_write_registration_handoff: bool = False,
) -> dict[str, Any]:
    """Internal: astrometry + alignment + per-frame CSV for one observation subtree (``job``)."""
    import numpy as np

    ap = Path(archive_path)
    obs_group_key = str(job.get("gkey", "") or "")
    scanning_id = int(job.get("scanning_id", 0) or 0)
    _job_md = job.get("master_dark_path")
    _master_dark_bpm_path: Path | None = Path(str(_job_md)) if _job_md and str(_job_md).strip() else None
    if _master_dark_bpm_path is not None and not _master_dark_bpm_path.is_file():
        _master_dark_bpm_path = None
    detrended_root = Path(job["detrended_root"])
    aligned_root = Path(job["aligned_root"])
    platesolve_dir = Path(job["platesolve_dir"])
    files = list(job["files"])
    # Zarovnane FITS: {archive}/detrended_aligned/lights/{filter_exp_binning}/... (vnorene cesty podla vstupu)
    os.makedirs(str(aligned_root), exist_ok=True)
    os.makedirs(str(platesolve_dir), exist_ok=True)
    _cfg_align = app_config or AppConfig()
    _align_star_cap = max(10, min(5000, int(_cfg_align.alignment_max_stars)))
    # Keep alignment input stable on dense fields: use at most TOP 200 brightest stars.
    _align_star_cap = min(_align_star_cap, 200)
    _sips_sig = float(_cfg_align.sips_dao_threshold_sigma)
    if not math.isfinite(_sips_sig) or _sips_sig <= 0:
        _sips_sig = 3.5
    _cfg_align_sig = float(_cfg_align.alignment_detection_sigma)
    if not math.isfinite(_cfg_align_sig) or _cfg_align_sig <= 0:
        _cfg_align_sig = _sips_sig
    try:
        _ui_sig = float(dao_threshold_sigma)
    except (TypeError, ValueError):
        _ui_sig = 0.0
    if not math.isfinite(_ui_sig) or _ui_sig <= 0:
        _ui_sig = 0.0
    # Session override > Settings alignment_detection_sigma > sips_dao_threshold_sigma.
    _align_det_sigma = max(
        0.8,
        min(20.0, _ui_sig if _ui_sig > 0 else _cfg_align_sig),
    )
    _fb_align = float(_cfg_align.sips_dao_fwhm_px)
    if not math.isfinite(_fb_align) or _fb_align <= 0:
        _fb_align = 2.5
    _pfov_align: float | None = None
    if build_masterstar_and_catalogs:
        LOGGER.info("Astrometria + MASTERSTAR + per-frame CSV: archiv %s", ap)
    else:
        LOGGER.info("Astrometria + zarovnanie + per-frame CSV (bez MASTERSTAR): archiv %s", ap)
    # MASTERSTAR initial match: allow a looser sep (min 10") for robust first-pass Gaia join.
    _catalog_match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
    if _catalog_match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
        _pipeline_ui_info(
            f"Katalogovy match prah zvyseny na {_catalog_match_sep_eff:.2f}\" "
            "(minimum pre robustny pociatocny cross-match)."
        )

    _cat_loc_only = bool(catalog_local_gaia_only) if catalog_local_gaia_only is not None else True
    if _cat_loc_only:
        LOGGER.info("Katalog: rezim lokalny Gaia (SQLite)")
    equip_sat_adu = _equipment_saturate_adu_from_db(id_equipment)
    if draft_id is not None and files:
        try:
            _db_sat = VyvarDatabase(Path(_cfg_align.database_path))
            try:
                _cmb_sat = _db_sat.get_combined_metadata(files[0], int(draft_id))
                if _cmb_sat.get("saturate_adu") is not None:
                    equip_sat_adu = _cmb_sat["saturate_adu"]
            finally:
                _db_sat.conn.close()
        except Exception:  # noqa: BLE001
            # EXC-0414: T4 -- `_db_pf.conn.close()` after plate-scale lookup for per-frame match sep; cleanup only. (EXCEPT-BULK-2 2026-07-08)
            pass
    if not files:
        raise FileNotFoundError(
            f"Chybaju FITS v {detrended_root}. Plate solve cita len **spracovane** snimky. "
            "Najprv spusti **MAKE MASTERSTAR** po kroku **Analyze** (zapis do "
            f"`{ap / 'processed' / 'lights'}` alebo starsie `{ap / 'detrended' / 'lights'}`)."
        )

    _t_step3_start = time.time()
    n_files = len(files)
    ref_fp, ref_star_scores = _pick_reference_frame_by_star_count(files)
    if osc_registration_handoff is not None:
        ref_name = str(osc_registration_handoff.get("reference_file") or "")
        for _rf in files:
            if _rf.name == ref_name:
                ref_fp = _rf
                break
    # Read reference once (no lock during solve step).
    with fits.open(ref_fp, memmap=False) as hdul:
        ref_hdr = hdul[0].header.copy()
        ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
    _rh, _rw = int(ref_data.shape[0]), int(ref_data.shape[1])
    try:
        _pf_try = float(plate_solve_fov_deg)
        if math.isfinite(_pf_try) and _pf_try > 0:
            _pfov_align = _pf_try
    except (TypeError, ValueError):
        _pfov_align = None
    if _pfov_align is None:
        _pfov_align = resolve_plate_solve_fov_deg_hint(
            ref_hdr,
            _rh,
            _rw,
            database_path=_cfg_align.database_path,
            equipment_id=int(id_equipment) if id_equipment is not None else None,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
    if _pfov_align is None:
        _pfov_align = float(_cfg_align.plate_solve_fov_deg)

    _scale_pf: float | None = None
    _db_pf = _vyvar_open_database(_cfg_align)
    if _db_pf is not None:
        try:
            _eq_pf, _tel_pf = resolve_optics_ids_for_platesolve(
                _db_pf, draft_id, equipment_id=id_equipment
            )
            _xb_pf, _yb_pf = fits_binning_xy_from_header(ref_hdr)
            _bin_pf = max(1, int(_xb_pf), int(_yb_pf))
            _scale_pf = compute_plate_scale_from_db(_eq_pf, _tel_pf, _db_pf.conn, binning=_bin_pf)
        except Exception:  # noqa: BLE001
            _scale_pf = None
        finally:
            try:
                _db_pf.conn.close()
            except Exception:  # noqa: BLE001
                pass
    _j_psep = job.get("per_frame_catalog_match_sep_arcsec")
    if _j_psep is not None:
        try:
            per_frame_match_sep = float(_j_psep)
        except (TypeError, ValueError):
            per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)
    else:
        per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)
    if not math.isfinite(per_frame_match_sep) or per_frame_match_sep <= 0:
        per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)

    has_wcs = _has_valid_wcs(ref_hdr)
    solve_steps = 0 if has_wcs else 1
    master_steps = 1 if build_masterstar_and_catalogs else 0
    global_total = max(1, 1 + solve_steps + n_files + master_steps + n_files)
    prog_i = [0]

    def _prog(msg: str) -> None:
        if progress_cb is None:
            return
        prog_i[0] += 1
        progress_cb(prog_i[0], global_total, msg)

    # --- MASTERSTAR build + plate-solve (per-setup platesolve/) before alignment ---
    # IMPORTANT (multi-filter): each setup must have its own MASTERSTAR + catalogs, otherwise
    # R/V/B runs overwrite each other (MASTERSTAR.fits, masterstars_full_match.csv, VY_MIRR, ...)
    # and reference/per-frame astrometry becomes unstable.
    _masterstar_built = False
    _cat_info_root: dict[str, Any] = {}
    _ps_root = platesolve_dir
    _t_platesolve = time.time()
    if build_masterstar_and_catalogs:
        _prog("platesolve/MASTERSTAR: referencny snimok + plate-solve + katalogy...")
        _cat_info_root = generate_masterstar_and_catalog(
            archive_path=ap,
            max_catalog_rows=int(max_catalog_rows),
            astrometry_api_key=astrometry_api_key,
            source_root=detrended_root,
            platesolve_dir=_ps_root,
            platesolve_backend=platesolve_backend,
            plate_solve_fov_deg=float(_pfov_align),
            catalog_match_max_sep_arcsec=float(_catalog_match_sep_eff),
            saturate_level_fraction=float(saturate_level_fraction),
            n_comparison_stars=int(n_comparison_stars),
            require_non_variable_comparisons=bool(require_non_variable_comparisons),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            app_config=_cfg_align,
            equipment_id=id_equipment,
            draft_id=draft_id,
            master_dark_path=_master_dark_bpm_path,
            masterstar_candidate_paths=job.get("masterstar_candidate_paths"),
            masterstar_selection_pct=job.get("masterstar_selection_pct"),
            setup_name=obs_group_key or None,
            masterstar_basename="MASTERSTAR.fits",
            masterstars_csv_basename="masterstars_full_match.csv",
            masterstar_fits_only=False,
            masterstar_skip_build=False,
        )
        _masterstar_built = True

        # Prefer MASTERSTAR as the canonical alignment reference when available.
        # This guarantees that:
        # - the output pixel grid matches MASTERSTAR (no WCS/data grid mismatch),
        # - per-frame matching against masterstars_full_match.csv works reliably.
        try:
            _ms_fp = _cat_info_root.get("masterstar_fits") if isinstance(_cat_info_root, dict) else None
            if _ms_fp:
                _ms_path = Path(str(_ms_fp)).resolve()
                if _ms_path.is_file():
                    with fits.open(_ms_path, memmap=False) as hdul:
                        ref_hdr = hdul[0].header.copy()
                        ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                    ref_fp = _ms_path
                    has_wcs = _has_valid_wcs(ref_hdr)
                    log_event(f"INFO: Alignment reference set to MASTERSTAR: {ref_fp.name}")
        except Exception as _ms_ref_exc:  # noqa: BLE001
            # EXC-0416: T1 -- Copying MASTERSTAR WCS onto reference FITS fails silently; aligned products can carry a... (EXCEPT-BULK-2 2026-07-08)
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().masterstar_ref_swap_fail += 1
            LOGGER.error("Using MASTERSTAR as alignment reference failed: %s", _ms_ref_exc)

    _prog(
        f"detrended_aligned/lights: pripravujem zarovnanie ({n_files} snimok z {detrended_root.name}/...)..."
    )

    # If MASTERSTAR was built for this setup and has a valid WCS, prefer it as the canonical
    # WCS for detrended_aligned products. Some frames already carry a WCS that can be offset by
    # arcminutes from MASTERSTAR; using it would break per-frame Gaia matching (master_reference_sky).
    if build_masterstar_and_catalogs and isinstance(_cat_info_root, dict):
        try:
            _ms_fp = _cat_info_root.get("masterstar_fits")
            if _ms_fp:
                _ms_path = Path(str(_ms_fp)).resolve()
                if _ms_path.is_file():
                    with fits.open(_ms_path, memmap=False) as _ms_hdul:
                        _ms_hdr = _ms_hdul[0].header.copy()
                    if _has_valid_wcs(_ms_hdr):
                        _apply_wcs_header_to_fits(ref_fp, _ms_hdr)
                        with fits.open(ref_fp, memmap=False) as hdul:
                            ref_hdr = hdul[0].header.copy()
                            ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                        has_wcs = True
                        log_event(
                            f"INFO: Reference WCS prevzate z MASTERSTAR ({_ms_path.name}) - pouzijem MASTERSTAR WCS pre alignment aj per-frame match."
                        )
        except Exception as _wcs_copy_exc:  # noqa: BLE001
            try:
                log_event(f"DEBUG: Reference WCS copy from MASTERSTAR failed: {_wcs_copy_exc}")
            except Exception:  # noqa: BLE001
                pass

    if (
        not build_masterstar_and_catalogs
        and sibling_recovery_use_masterstar
        and not _masterstar_built
    ):
        try:
            _ms_path = (platesolve_dir / "MASTERSTAR.fits").resolve()
            if _ms_path.is_file():
                with fits.open(_ms_path, memmap=False) as hdul:
                    _ms_hdr = hdul[0].header.copy()
                    ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                if _has_valid_wcs(_ms_hdr):
                    ref_fp = _ms_path
                    ref_hdr = _ms_hdr
                    has_wcs = True
                    _cat_info_root = {
                        "masterstar_fits": str(_ms_path),
                        "masterstars_csv": str(platesolve_dir / "masterstars_full_match.csv"),
                    }
                    log_event(
                        f"INFO: Sibling-recovery alignment using existing MASTERSTAR: {_ms_path.name}"
                    )
        except Exception as _sib_ms_exc:  # noqa: BLE001
            # EXC-0418: T3 -- DEBUG min/max/mean/NaN stats logging `pass`; alignment detection proceeds regardless. (EXCEPT-BULK-2 2026-07-08)
            log_event(f"DEBUG: Sibling-recovery MASTERSTAR load failed: {_sib_ms_exc}")

    if not has_wcs:
        _prog("Plate solve referencie (moze chvilu trvat)...")

    if not has_wcs:
        # Solve reference file in-place (no open handle on Windows).
        _hra, _hdec, _ = _pointing_hint_from_header(ref_hdr)
        solve = _solve_wcs_external(
            ref_fp,
            backend=platesolve_backend,
            astrometry_api_key=astrometry_api_key,
            plate_solve_fov_deg=float(_pfov_align),
            hint_ra_deg=_hra,
            hint_dec_deg=_hdec,
            app_config=_cfg_align,
            equipment_id=id_equipment,
            draft_id=draft_id,
        )
        if not solve.get("solved", False):
            raise RuntimeError(f"Reference astrometry failed: {solve.get('reason')}")
        # Reload solved header/data
        with fits.open(ref_fp, memmap=False) as hdul:
            ref_hdr = hdul[0].header.copy()
            ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)

    print(f"  Plate solve: {time.time() - _t_platesolve:.1f}s")
    _t_align = time.time()

    # Use the same FWHM rule as per-frame alignment (VY_FWHM / header), not only ``sips_dao_fwhm_px``.
    # A fixed ~2.5 px kernel on the reference while sources use ~5 px yields different brightest-N
    # orderings -> bogus point pairs -> astroalign "triangles exhausted" and identity/no_wcs cascades.
    _raw_ref_fw = dao_detection_fwhm_pixels(ref_hdr, configured_fallback=_fb_align)
    try:
        _fwv = float(_raw_ref_fw) if _raw_ref_fw is not None else float("nan")
    except (TypeError, ValueError):
        _fwv = float("nan")
    _align_fwhm_ref = float(_fwv) if math.isfinite(_fwv) and _fwv > 0 else float(_fb_align)

    hint_center = _wcs_field_center_radec_deg(ref_fp)
    hint_ra: float | None = hint_center[0] if hint_center else None
    hint_dec: float | None = hint_center[1] if hint_center else None

    extra_platesolve_results: list[dict[str, Any]] = []

    try:
        pass  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0419] `estimate_archive_memory_profile` failure leaves `use_ram_handoff=True`; may exhaust RA...: %s', exc)
        raise RuntimeError(f"astroalign required for frame registration: {exc}") from exc

    log_event(
        f"Detekcia hviezd: Pouzite FWHM={_align_fwhm_ref:.2f}, Sigma={_align_det_sigma:.2f}"
    )

    def _maybe_refine_aligned(
        hdr_mut: fits.Header, data_mut: np.ndarray, label: str, *, dao_fwhm_px_frame: float
    ) -> None:
        _ = (hdr_mut, data_mut, label, dao_fwhm_px_frame)
        return

    # Adaptive alignment star budget:
    # - if STAR_COUNT > 1000 -> use top 300 brightest
    # - if STAR_COUNT < 100 -> use all
    # - else -> cap at 300
    data_to_detect = np.asarray(ref_data, dtype=np.float32)
    try:
        log_event(
            "DEBUG: Data stats for alignment - "
            f"Min: {np.min(data_to_detect):.2f}, "
            f"Max: {np.max(data_to_detect):.2f}, "
            f"Mean: {np.mean(data_to_detect):.2f}, "
            f"NaN count: {np.isnan(data_to_detect).sum()}"
        )
    except Exception:  # noqa: BLE001
        pass
    ref_xy_all = _alignment_detect_xy(
        data_to_detect,
        int(max(100, max_detected_stars)),
        det_sigma=_align_det_sigma,
        fwhm_px=_align_fwhm_ref,
        label=ref_fp.name,
        log_sink=None,
    )
    n_ref = int(len(ref_xy_all))
    if n_ref > 1000:
        n_keep = 300
    elif n_ref < 100:
        n_keep = n_ref
    else:
        n_keep = min(300, n_ref)
    ref_xy = ref_xy_all[:n_keep]
    if len(ref_xy) < int(min_detected_stars):
        raise RuntimeError(
            f"Reference frame has too few detected stars ({len(ref_xy)} < {min_detected_stars})."
        )
    ref_xy_fit = ref_xy[: int(min(_align_star_cap, len(ref_xy)))]
    log_event(
        f"Zarovnanie referencia {ref_fp.name}: DAO hviezd={len(ref_xy)}, "
        f"cap pre transform={_align_star_cap}, DAO sigma={_align_det_sigma:.2f}, FWHM={_align_fwhm_ref:.2f}px "
        f"(QC VY_FWHM alebo sips_dao_fwhm_px)"
    )

    # Auto RAM management: default in-memory, but switch to disk when estimated working set exceeds 70% of available RAM.
    use_ram_handoff = bool(ram_align_and_catalog)
    try:
        mp = estimate_archive_memory_profile(ap)
        avail = mp.get("available_ram_bytes")
        prh = mp.get("platesolve_ram_handoff") or {}
        tot = prh.get("estimated_total_conservative_bytes")
        if isinstance(avail, int) and isinstance(tot, int) and avail > 0 and tot > 0:
            if tot > int(0.70 * avail):
                use_ram_handoff = False
    except Exception:  # noqa: BLE001
        # EXC-0420: T1 -- `_vy_fwhm_header_value` returns `None` on read error; MASTERSTAR match heuristic falls ... (EXCEPT-BULK-2 2026-07-08)
        pass

    aligned_ram_buffer: list[tuple[str, fits.Header, np.ndarray]] = []
    aligned_files: list[Path] = []
    star_counts: list[dict[str, Any]] = []
    rotation_ref_angle_deg: float | None = None
    rotation_flip_frame_indices_1based: list[int] = []
    rotation_flip_first_index_1based: int | None = None
    _flip_logged = False

    try:
        rotation_ref_angle_deg = wcs_rotation_angle_deg(ref_hdr)
    except Exception:  # noqa: BLE001
        # EXC-0421: T2 -- `_aligned_masterstar_matches_platesolve` stat/read failure returns `False`, allowing al... (EXCEPT-BULK-2 2026-07-08)
        rotation_ref_angle_deg = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            ref_wcs_obj = WCS(ref_hdr) if _has_valid_wcs(ref_hdr) else None
    except Exception:  # noqa: BLE001
        ref_wcs_obj = None

    # save reference as aligned baseline, keep WCS
    _prog(
        f"detrended_aligned/lights: {'RAM - referencia' if use_ram_handoff else 'zapisujem FITS'} "
        f"{ref_fp.name} (1/{n_files})"
    )
    try:
        ref_rel = ref_fp.relative_to(detrended_root)
    except Exception:  # noqa: BLE001
        # Reference can live outside detrended_root (e.g. MASTERSTAR in platesolve/...).
        # In that case, store it at the aligned root top-level.
        ref_rel = Path(ref_fp.name)
    ref_out = aligned_root / ref_rel
    _ensure_parent_dirs_for_aligned_fits(ref_out)

    def _vy_fwhm_header_value(path: Path) -> float | None:
        try:
            with fits.open(path, memmap=False) as _h:
                _v = _h[0].header.get("VY_FWHM")
            if _v is None:
                return None
            vv = float(_v)
            return float(vv) if math.isfinite(vv) and vv > 0 else None
        except Exception:  # noqa: BLE001
            return None

    def _aligned_masterstar_matches_platesolve(*, platesolve_masterstar: Path, aligned_masterstar: Path) -> bool:
        """Heuristic match: size+mtime OR size+VY_FWHM.

        This is intentionally lightweight to prevent unintended overwrites of the aligned MASTERSTAR baseline.
        """
        try:
            if not platesolve_masterstar.is_file() or not aligned_masterstar.is_file():
                return False
            s_src = platesolve_masterstar.stat()
            s_dst = aligned_masterstar.stat()
            if int(s_src.st_size) != int(s_dst.st_size):
                return False
            # Preferred fast path: preserved mtime via shutil.copy2
            if int(s_src.st_mtime) == int(s_dst.st_mtime):
                return True
        except Exception:  # noqa: BLE001
            return False
        # Header fallback: VY_FWHM should be stable for a given MASTERSTAR (copied+solved).
        v_src = _vy_fwhm_header_value(platesolve_masterstar)
        v_dst = _vy_fwhm_header_value(aligned_masterstar)
        if v_src is None or v_dst is None:
            return False
        return abs(float(v_src) - float(v_dst)) <= 1e-9

    def _ensure_aligned_masterstar_copy(*, platesolve_masterstar: Path, aligned_masterstar: Path) -> None:
        import shutil  # local import: narrow scope

        if aligned_masterstar.is_file() and _aligned_masterstar_matches_platesolve(
            platesolve_masterstar=platesolve_masterstar, aligned_masterstar=aligned_masterstar
        ):
            log_event(
                "[MASTERSTAR] aligned ref already exists and matches platesolve source - skipping write"
            )
            return
        log_event("[MASTERSTAR] copying platesolve MASTERSTAR -> detrended_aligned ref")
        shutil.copy2(platesolve_masterstar, aligned_masterstar)

    _ps_masterstar = (platesolve_dir / "MASTERSTAR.fits")
    _is_platesolve_masterstar_ref = False
    try:
        _is_platesolve_masterstar_ref = (
            ref_fp.name.strip().casefold() == "masterstar.fits"
            and _ps_masterstar.is_file()
            and ref_fp.resolve() == _ps_masterstar.resolve()
            and ref_out.name.strip().casefold() == "masterstar.fits"
        )
    except Exception:  # noqa: BLE001
        _is_platesolve_masterstar_ref = False

    if _is_platesolve_masterstar_ref:
        _ensure_aligned_masterstar_copy(
            platesolve_masterstar=_ps_masterstar.resolve(),
            aligned_masterstar=ref_out,
        )
    else:
        with fits.open(ref_fp, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = _as_fits_float32_image(hdul[0].data)
        hdr["VY_ALGN"] = (True, "Aligned to reference")
        hdr["VYALGOK"] = (True, "Alignment OK")
        hdr["VY_REF"] = (ref_fp.name[:60], "Reference frame for alignment")
        _maybe_refine_aligned(hdr, data, ref_fp.name, dao_fwhm_px_frame=_align_fwhm_ref)
        if use_ram_handoff:
            aligned_ram_buffer.append((ref_rel.as_posix(), hdr.copy(), np.copy(data)))
        else:
            fits.writeto(ref_out, data, header=hdr, overwrite=True)
    aligned_files.append(ref_out)
    star_counts.append(
        {
            "file": ref_fp.name,
            "frame_index": int(files.index(ref_fp) + 1) if ref_fp in files else 1,
            "detected_stars": int(len(ref_xy)),
            "aligned": True,
            "alignment_method": "reference",
            "is_flipped": False,
            "rotation_angle_deg": rotation_ref_angle_deg,
        }
    )

    # Align every other frame to reference (skip duplicate if ref is not files[0]).
    # Astroalign control-point cap from cfg (decoupled from detection ladder max_stars).
    align_cp = int(max(12, min(500, int(_cfg_align.alignment_max_control_points))))
    ref_pts = np.asarray(ref_xy_fit, dtype=np.float32)
    if ref_pts is None or len(ref_pts) == 0:
        raise ValueError("Referencne hviezdy su prazdne pred startom alignmentu!")
    # Keep immutable backup of reference points; never overwrite with per-frame source detections.
    fixed_target_pts = np.copy(ref_pts).astype("float32")
    log_event(f"DEBUG: Start alignment, reference stars N = {len(fixed_target_pts)}")
    # Brute-force isolation from any shared numpy buffers.
    REFERENCE_LIST = fixed_target_pts.tolist()
    LOGGER.info(
        "Astrometry alignment: astroalign uses up to %s control points on up to %s DAO sources per frame",
        align_cp,
        _align_star_cap,
    )

    n_written_align = 1
    n_align_workers = _vyvar_parallel_worker_count(_cfg_align)
    align_tasks: list[tuple[str, int]] = []
    for frame_index_1based, fp in enumerate(files, start=1):
        if fp == ref_fp:
            continue
        align_tasks.append((str(fp.resolve()), int(frame_index_1based)))

    _align_ctx: dict[str, Any] = {
        "ref_data": ref_data,
        "ref_hdr": ref_hdr.copy(),
        "ref_fp_name": ref_fp.name,
        "fixed_target_pts": np.copy(fixed_target_pts).astype(np.float32, copy=False),
        "reference_list": list(REFERENCE_LIST),
        "has_ref_wcs": ref_wcs_obj is not None,
        "platesolve_dir": str(platesolve_dir),
        "align_star_cap": int(_align_star_cap),
        "max_control_points": int(align_cp),
        "min_detected_stars": int(min_detected_stars),
        "max_detected_stars": int(max_detected_stars),
        "fb_align": float(_fb_align),
        "rotation_ref_angle_deg": rotation_ref_angle_deg,
    }
    _osc_registration_capture: dict[str, dict[str, Any]] = {}

    def _flush_one_alignment(res: dict[str, Any]) -> None:
        nonlocal n_written_align, _flip_logged, rotation_flip_first_index_1based
        idx = int(res["frame_index_1based"])
        fp = Path(res["fp"])
        if bool(res.get("is_flipped", False)):
            rotation_flip_frame_indices_1based.append(idx)
            if rotation_flip_first_index_1based is None:
                rotation_flip_first_index_1based = idx
            if not _flip_logged:
                log_event(
                    f"Physical rotation change detected at frame index {idx}. "
                    "Adjusting alignment strategy."
                )
                _flip_logged = True
        if res.get("kind") == "failed_skip":
            star_counts.append(res["star_count"])
            return
        hdr_out = res["hdr"]
        aligned_data = res["aligned_data"]
        fw_i = float(res["fw_i"])
        _maybe_refine_aligned(hdr_out, aligned_data, fp.name, dao_fwhm_px_frame=fw_i)
        try:
            fp_rel = fp.relative_to(detrended_root)
        except Exception:  # noqa: BLE001
            fp_rel = Path(fp.name)
        out_fp = aligned_root / fp_rel
        _ensure_parent_dirs_for_aligned_fits(out_fp)
        n_written_align += 1
        _prog(
            f"detrended_aligned/lights: "
            f"{'RAM - zarovnanie' if use_ram_handoff else 'zapisujem FITS'} "
            f"{fp.name} ({n_written_align}/{n_files})..."
        )
        if use_ram_handoff:
            aligned_ram_buffer.append((fp_rel.as_posix(), hdr_out.copy(), np.copy(aligned_data)))
        else:
            fits.writeto(out_fp, aligned_data, header=hdr_out, overwrite=True)
        aligned_files.append(out_fp)
        star_counts.append(res["star_count"])
        rh = res.get("registration_handoff")
        if isinstance(rh, dict) and res.get("star_count", {}).get("file"):
            _osc_registration_capture[str(res["star_count"]["file"])] = dict(rh)

    if osc_registration_handoff is not None:
        from osc_align import apply_registration_handoff_to_frame

        handoff_frames = dict(osc_registration_handoff.get("frames") or {})
        ref_name = str(osc_registration_handoff.get("reference_file") or ref_fp.name)
        for fp in files:
            if fp == ref_fp and fp.name != ref_name:
                continue
            if fp.name == ref_name:
                continue
            entry = handoff_frames.get(fp.name) or {}
            if entry and not bool(entry.get("aligned", True)):
                star_counts.append(
                    {
                        "file": fp.name,
                        "frame_index": int(files.index(fp) + 1) if fp in files else 0,
                        "detected_stars": 0,
                        "aligned": False,
                        "reason": "donor_not_aligned",
                        "alignment_method": "osc_handoff_skip",
                        "is_flipped": False,
                    }
                )
                continue
            with fits.open(fp, memmap=False) as hdul:
                raw_hdr = hdul[0].header.copy()
                raw_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
            aligned_data, hdr_out, method = apply_registration_handoff_to_frame(
                frame_path=fp,
                frame_data=raw_data,
                frame_hdr=raw_hdr,
                ref_data=ref_data,
                ref_hdr=ref_hdr,
                handoff_entry=entry,
            )
            _flush_one_alignment(
                {
                    "kind": "aligned",
                    "fp": str(fp.resolve()),
                    "frame_index_1based": int(files.index(fp) + 1) if fp in files else 0,
                    "is_flipped": False,
                    "hdr": hdr_out,
                    "aligned_data": aligned_data,
                    "aligned_method": method,
                    "fw_i": float(_fb_align),
                    "star_count": {
                        "file": fp.name,
                        "frame_index": int(files.index(fp) + 1) if fp in files else 0,
                        "detected_stars": 0,
                        "aligned": True,
                        "alignment_method": method,
                        "is_flipped": False,
                    },
                }
            )
    elif n_align_workers > 1 and len(align_tasks) > 1:
        _mp_ctx: dict[str, Any] = {
            "ref_data": np.ascontiguousarray(np.copy(_align_ctx["ref_data"])),
            "ref_hdr": _align_ctx["ref_hdr"].copy(),
            "ref_fp_name": _align_ctx["ref_fp_name"],
            "fixed_target_pts": np.copy(_align_ctx["fixed_target_pts"]).astype(np.float32, copy=False),
            "reference_list": list(_align_ctx["reference_list"]),
            "has_ref_wcs": bool(_align_ctx["has_ref_wcs"]),
            "platesolve_dir": str(_align_ctx["platesolve_dir"]),
            "align_star_cap": int(_align_ctx["align_star_cap"]),
            "max_control_points": int(_align_ctx["max_control_points"]),
            "min_detected_stars": int(_align_ctx["min_detected_stars"]),
            "max_detected_stars": int(_align_ctx["max_detected_stars"]),
            "fb_align": float(_align_ctx["fb_align"]),
            "rotation_ref_angle_deg": _align_ctx["rotation_ref_angle_deg"],
        }
        def _run_alignment_single_process() -> None:
            for fp_s, idx in align_tasks:
                res = _alignment_compute_one_frame(Path(fp_s), int(idx), _align_ctx, None)
                _flush_one_alignment(res)

        # A-durable: resolve the MP init/task by FRESH module attribute at call time, so the
        # objects handed to the spawn pool are exactly what sys.modules resolves - even if the
        # Streamlit file-watcher reloaded vyvar_alignment_frame after pipeline.py was imported
        # (the import-time `from ... import` binding would otherwise go stale -> PicklingError).
        _mp_init = vyvar_alignment_frame._astrometry_align_mp_init
        _mp_task = vyvar_alignment_frame._astrometry_align_mp_task
        try:
            with ProcessPoolExecutor(
                max_workers=n_align_workers,
                initializer=_mp_init,
                initargs=(_mp_ctx,),
            ) as pool:
                raw_list = list(pool.map(_mp_task, align_tasks, chunksize=1))
            for res in raw_list:
                for ln in res.get("log_events", ()):
                    log_event(ln)
                res_flush = {k: v for k, v in res.items() if k != "log_events"}
                _flush_one_alignment(res_flush)
        except pickle.PicklingError as _pkl_err:
            # Robust fallback: if the spawn pool cannot pickle the MP funcs (e.g. mid-run module
            # reload defeats the fresh-attr lookup), run alignment single-process instead of
            # aborting. Photometry is byte-identical to the MP path (same per-frame compute).
            # PicklingError is raised at task-submission (pickling the worker funcs) before any
            # result is flushed, so no partial per-frame state exists here - run single-process.
            _pipeline_ui_info(
                f"Alignment: multiprocessing dispatch failed to pickle worker functions "
                f"({_pkl_err}); falling back to single-process alignment."
            )
            _run_alignment_single_process()
    else:
        for fp_s, idx in align_tasks:
            res = _alignment_compute_one_frame(Path(fp_s), int(idx), _align_ctx, None)
            _flush_one_alignment(res)

    n_aligned = int(sum(1 for r in star_counts if r.get("aligned")))
    n_failed_align = int(sum(1 for r in star_counts if not bool(r.get("aligned"))))
    if n_failed_align > 0:
        reasons: dict[str, int] = {}
        for r in star_counts:
            if bool(r.get("aligned")):
                continue
            rr = str(r.get("reason") or "unknown")
            reasons[rr] = int(reasons.get(rr, 0)) + 1
        reason_txt = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:5])
        _pipeline_ui_info(
            f"Alignment warning: zlyhalo {n_failed_align}/{len(files)} snimok v {aligned_root.resolve()} "
            f"(dovody: {reason_txt})."
        )
    if n_aligned <= 1:
        msg = (
            f"Alignment zlyhal: uspesne zarovnana len referencia (1/{len(files)}). "
            f"Skontroluj DAO prah/FWHM a WCS vstupy. Vystupny priecinok: {aligned_root.resolve()}."
        )
        _pipeline_ui_error(msg)
        raise RuntimeError(msg)
    rep_path = platesolve_dir / "alignment_report.csv"
    pd.DataFrame(star_counts).to_csv(rep_path, index=False)
    if osc_write_registration_handoff or bool(job.get("osc_write_registration_handoff")):
        from osc_align import write_registration_handoff

        write_registration_handoff(
            platesolve_dir,
            reference_file=str(ref_fp.name),
            frames=_osc_registration_capture,
        )

    print(f"  Zarovnanie: {time.time() - _t_align:.1f}s")
    _t_csv = time.time()

    # If we aligned in RAM, flush aligned FITS to disk before MASTERSTAR (needs files on disk).
    _ram_flushed_before_masterstar = False
    if use_ram_handoff and aligned_ram_buffer and build_masterstar_and_catalogs:
        _prog("detrended_aligned/lights: zapisujem FITS na disk (RAM -> disk, pred MASTERSTAR)...")
        for name, hdr, arr in aligned_ram_buffer:
            _target = aligned_root / Path(name)
            _ensure_parent_dirs_for_aligned_fits(_target)
            fits.writeto(
                _target,
                _as_fits_float32_image(arr),
                header=hdr,
                overwrite=True,
            )
        _ram_flushed_before_masterstar = True
        _aligned_file_list = sorted(aligned_root.glob("proc_*.fits"))
        LOGGER.info(f"[BORDER] RAM flush done - {len(_aligned_file_list)} aligned frames on disk")

    cat_info: dict[str, Any] = {}
    ms_csv: Path | None = None
    ms_fits: Path | None = None
    use_master_fast = False

    if build_masterstar_and_catalogs:
        # Use per-setup platesolve/ artifacts built before alignment.
        cat_info = dict(_cat_info_root or {})
        try:
            ms_csv = Path(str((cat_info.get("masterstars_csv") or (_ps_root / "masterstars_full_match.csv")))).resolve()
        except Exception:  # noqa: BLE001
            # EXC-0422: T3 -- Optional `variable_targets_csv` path probe `pass`; UI metadata field may be absent. (EXCEPT-BULK-2 2026-07-08)
            ms_csv = Path(str(cat_info.get("masterstars_csv") or (_ps_root / "masterstars_full_match.csv")))
        try:
            ms_fits = Path(str((cat_info.get("masterstar_fits") or (_ps_root / "MASTERSTAR.fits")))).resolve()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0423] Post-RAM-flush DB-aware photometry plan rewrite `pass`es; border-safe bbox may use stal...: %s', exc)
            ms_fits = Path(str(cat_info.get("masterstar_fits") or (_ps_root / "MASTERSTAR.fits")))

        # comparison_stars.csv / variable_targets.csv are produced in this setup directory already.
        try:
            _comp = platesolve_dir / "comparison_stars.csv"
            _var = platesolve_dir / "variable_targets.csv"
            if _comp.is_file():
                cat_info["comparison_stars_csv"] = str(_comp)
            if _var.is_file():
                cat_info["variable_targets_csv"] = str(_var)
        except Exception:  # noqa: BLE001
            pass

        # Masterstar lock for Step 3: per-frame catalogs must use one fixed reference list.
        use_master_fast = True

        # Recompute photometry plan after alignment so border-safe bbox uses aligned frames.
        try:
            _aligned_disk: list[Path] = []
            if use_ram_handoff and aligned_ram_buffer and _ram_flushed_before_masterstar:
                _aligned_disk = list(_aligned_file_list)
            else:
                try:
                    _aligned_disk = sorted(aligned_root.glob("proc_*.fits"))
                except Exception:  # noqa: BLE001
                    _aligned_disk = []
            _has_aligned = bool(_aligned_disk) or bool(use_ram_handoff and aligned_ram_buffer)
            if build_masterstar_and_catalogs and _has_aligned:
                _wp_aligned = write_photometry_plan_files(
                    platesolve_dir=platesolve_dir,
                    masterstar_fits=ms_fits or (platesolve_dir / "MASTERSTAR.fits"),
                    masterstars_csv=ms_csv or (platesolve_dir / "masterstars_full_match.csv"),
                    n_comparison_stars=int(n_comparison_stars),
                    require_non_variable=bool(require_non_variable_comparisons),
                    draft_id=int(draft_id) if draft_id is not None else None,
                    database_path=(
                        Path(str(_cfg_align.database_path))
                        if str(_cfg_align.database_path or "").strip()
                        else None
                    ),
                    aligned_files=_aligned_disk if _aligned_disk else None,
                    aligned_ram_frames=aligned_ram_buffer if use_ram_handoff and aligned_ram_buffer else None,
                    require_safe_bbox=True,
                )
                cat_info.update(_wp_aligned or {})
        except RuntimeError as _wp_exc:
            log_event(f"[BORDER] Post-alignment photometry plan rewrite failed: {_wp_exc!s}")
            if "[BORDER]" in str(_wp_exc):
                raise
        except Exception as _wp_exc:  # noqa: BLE001
            log_event(f"[BORDER] Post-alignment photometry plan rewrite failed: {_wp_exc!s}")

    export_base = prog_i[0]
    _catalog_app_cfg = _cfg_align
    _, _run_epsf = _photometry_mode_run_flags(
        _catalog_app_cfg,
        platesolve_dir=platesolve_dir,
    )

    # TODO-8: Build ePSF model after MASTERSTAR (Phase 2B prep)
    if _run_epsf:
        try:
            if draft_id is not None and str(_catalog_app_cfg.database_path or "").strip():
                _ms_for_epsf = ms_fits
                if _ms_for_epsf is None or not Path(_ms_for_epsf).is_file():
                    _ms_try = platesolve_dir / "MASTERSTAR.fits"
                    if _ms_try.is_file():
                        _ms_for_epsf = _ms_try
                _ms_csv_epsf = ms_csv
                if _ms_csv_epsf is None or not Path(_ms_csv_epsf).is_file():
                    _ms_csv_try = platesolve_dir / "masterstars_full_match.csv"
                    if _ms_csv_try.is_file():
                        _ms_csv_epsf = _ms_csv_try
                if (
                    _ms_for_epsf is not None
                    and _ms_csv_epsf is not None
                    and Path(_ms_for_epsf).is_file()
                    and Path(_ms_csv_epsf).is_file()
                ):
                    from psf_photometry import build_epsf_model

                    _db_epsf = VyvarDatabase(Path(str(_catalog_app_cfg.database_path)))
                    try:
                        _epsf_path = build_epsf_model(
                            masterstar_fits_path=Path(_ms_for_epsf),
                            masterstars_csv_path=Path(_ms_csv_epsf),
                            db=_db_epsf,
                            draft_id=int(draft_id),
                            # TODO-PSF-PHASE2: moffat_centroids not yet available at MASTERSTAR
                            # build time - requires per-frame Moffat run first then aggregate
                            # centroids. Implement in next session.
                        )
                        LOGGER.info("[ePSF] Model built: %s", _epsf_path)
                    finally:
                        _db_epsf.conn.close()
        except Exception as _e:  # noqa: BLE001
            # EXC-0424: T1 -- `normalize_gaia_source_id_series` failure `pass`es before deferred CSV write; per-frame... (EXCEPT-BULK-2 2026-07-08)
            LOGGER.warning("[ePSF] build_epsf_model failed (non-fatal): %s", _e)

    def _cat_prog(i: int, tot: int, msg: str) -> None:
        if progress_cb is None:
            return
        progress_cb(
            min(export_base + i, global_total),
            global_total,
            f"detrended_aligned/lights: CSV ({i}/{tot}) - {msg}",
        )

    if use_ram_handoff:
        per_cat = export_per_frame_catalogs(
            frames_root=aligned_root,
            platesolve_dir=platesolve_dir,
            max_catalog_rows=int(max_catalog_rows),
            catalog_match_max_sep_arcsec=float(per_frame_match_sep),
            saturate_level_fraction=float(saturate_level_fraction),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            masterstars_csv=ms_csv,
            masterstar_fits=ms_fits,
            use_master_fast_path=use_master_fast,
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            progress_cb=_cat_prog if progress_cb is not None else None,
            aligned_ram=aligned_ram_buffer,
            aligned_target_dir=aligned_root,
            defer_disk_writes=True,
            app_config=_catalog_app_cfg,
            plate_solve_fov_deg=float(_pfov_align),
            master_dark_path=_master_dark_bpm_path,
            draft_id=draft_id,
            equipment_id=id_equipment,
            full_catalog_export=True,
        )
        _prog("detrended_aligned/lights: zapisujem FITS + CSV na disk (davka po praci v RAM)...")
        if not _ram_flushed_before_masterstar:
            for name, hdr, arr in aligned_ram_buffer:
                _target = aligned_root / Path(name)
                _ensure_parent_dirs_for_aligned_fits(_target)
                fits.writeto(
                    _target,
                    _as_fits_float32_image(arr),
                    header=hdr,
                    overwrite=True,
                )
        for pcsv, df in per_cat.get("deferred_csv_writes", []):
            try:
                from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

                if isinstance(df, pd.DataFrame) and "catalog_id" in df.columns:
                    df = df.copy()
                    df["catalog_id"] = normalize_gaia_source_id_series(df["catalog_id"])
            except Exception:  # noqa: BLE001
                pass
            df.to_csv(pcsv, index=False)
        pd.DataFrame(per_cat.get("frames", [])).to_csv(Path(per_cat["index_csv"]), index=False)
        if per_cat.get("deferred_csv_writes"):
            _hybrid_ram = _finalize_hybrid_bkg_fallback_sidecar(
                aligned_root,
                err_background_mode="empirical",
                write_sidecar=True,
                gain=float(_catalog_app_cfg.gain),
                read_noise=float(_catalog_app_cfg.read_noise),
                setup_label=str(aligned_root.name),
            )
            if _hybrid_ram:
                per_cat["hybrid_bkg_fallback"] = _hybrid_ram
    else:
        per_cat = export_per_frame_catalogs(
            frames_root=aligned_root,
            platesolve_dir=platesolve_dir,
            max_catalog_rows=int(max_catalog_rows),
            catalog_match_max_sep_arcsec=float(per_frame_match_sep),
            saturate_level_fraction=float(saturate_level_fraction),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            masterstars_csv=ms_csv,
            masterstar_fits=ms_fits,
            use_master_fast_path=use_master_fast,
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            progress_cb=_cat_prog if progress_cb is not None else None,
            app_config=_catalog_app_cfg,
            plate_solve_fov_deg=float(_pfov_align),
            master_dark_path=_master_dark_bpm_path,
            draft_id=draft_id,
            equipment_id=id_equipment,
            full_catalog_export=True,
        )

    _assert_alignment_produced_fits(aligned_root)

    print(f"  Per-frame CSV: {time.time() - _t_csv:.1f}s")
    print(f"CELKOM krok 3 ({obs_group_key or detrended_root.name}): {time.time() - _t_step3_start:.1f}s")

    LOGGER.info(
        "Astrometria dokoncena: zarovnane %s / %s snimok; per-frame CSV: %s; MASTERSTAR: %s; RAM handoff: %s",
        n_aligned,
        len(files),
        int(per_cat.get("written", 0)),
        "ano" if build_masterstar_and_catalogs else "nie",
        "ano" if use_ram_handoff else "nie",
    )

    return {
        "ram_align_handoff_used": bool(use_ram_handoff),
        "detrended_input_root": str(detrended_root),
        "detrended_files_used": len(files),
        "reference_frame": str(ref_fp),
        "reference_star_counts": dict(ref_star_scores),
        "reference_hint_ra_dec_deg": {"ra": hint_ra, "dec": hint_dec} if hint_ra is not None else None,
        "extra_platesolve": extra_platesolve_results,
        "alignment_max_control_points_used": align_cp,
        "alignment_max_stars_cap": int(_align_star_cap),
        "alignment_detection_sigma": float(_align_det_sigma),
        "aligned_root": str(aligned_root),
        "aligned_frames": n_aligned,
        "input_frames": int(len(files)),
        "alignment_report_csv": str(rep_path),
        "rotation_ref_angle_deg": rotation_ref_angle_deg,
        "rotation_flip_frame_indices_1based": rotation_flip_frame_indices_1based,
        "rotation_flip_first_index_1based": rotation_flip_first_index_1based,
        "build_masterstar_and_catalogs": bool(build_masterstar_and_catalogs),
        "masterstar_built": bool(_masterstar_built) if build_masterstar_and_catalogs else False,
        "masterstar_fits": str(ms_fits) if build_masterstar_and_catalogs and ms_fits is not None else "",
        "masterstars_csv": str(ms_csv) if build_masterstar_and_catalogs and ms_csv is not None else "",
        "catalog_match_max_sep_arcsec": float(_catalog_match_sep_eff),
        "saturate_level_fraction": float(saturate_level_fraction),
        "saturate_limit_adu": (cat_info.get("saturate_limit_adu") if build_masterstar_and_catalogs else None),
        "saturate_limit_source": (cat_info.get("saturate_limit_source") if build_masterstar_and_catalogs else None),
        "max_catalog_rows": int(max_catalog_rows),
        "faintest_mag_limit": (
            cat_info.get("faintest_mag_limit") if build_masterstar_and_catalogs else faintest_mag_limit
        ),
        "per_frame_catalog_dir": per_cat.get("per_frame_dir"),
        "per_frame_catalog_index_csv": per_cat.get("index_csv"),
        "per_frame_catalogs_written": per_cat.get("written"),
        "comparison_stars_csv": cat_info.get("comparison_stars_csv", "") if build_masterstar_and_catalogs else "",
        "variable_targets_csv": cat_info.get("variable_targets_csv", "") if build_masterstar_and_catalogs else "",
        "photometry_plan_json": cat_info.get("photometry_plan_json", "") if build_masterstar_and_catalogs else "",
        "n_comparison_stars_requested": (
            cat_info.get("n_comparison_stars_requested") if build_masterstar_and_catalogs else None
        ),
        "comparison_selection": cat_info.get("comparison_selection") if build_masterstar_and_catalogs else None,
        "id_equipment": int(id_equipment) if id_equipment is not None else None,
        "equipment_saturate_adu_resolved": equip_sat_adu,
        "catalog_local_gaia_only": _cat_loc_only,
        "observation_group_key": obs_group_key,
        "scanning_id": (scanning_id if scanning_id > 0 else None),
    }


def astrometry_align_and_build_masterstar(
    *,
    archive_path: Path,
    astrometry_api_key: str | None = None,
    max_control_points: int = 80,
    min_detected_stars: int = 100,
    max_detected_stars: int = 500,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    max_extra_platesolve: int = 0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    max_catalog_rows: int = 12000,
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    id_equipment: int | None = None,
    draft_id: int | None = None,
    catalog_local_gaia_only: bool | None = None,
    build_masterstar_and_catalogs: bool = False,
    progress_cb: "callable | None" = None,
    ram_align_and_catalog: bool = False,
    app_config: AppConfig | None = None,
    masterstar_candidate_paths: "Sequence[str] | None" = None,
    masterstar_selection_pct: float | None = None,
    master_dark_path: Path | str | None = None,
) -> dict[str, Any]:
    """Astrometry + alignment + per-frame catalog CSV (mandatory outputs).

    Preprocessed frames under ``<archive>/processed/lights`` (or legacy ``detrended/lights``) are grouped
    by full relative parent folder (multi-observation / FILTER+EXP+BIN layout). Each group gets its own
    alignment, ``MASTERSTAR``, and ``platesolve/<group>/`` outputs. Frames stored directly in ``lights/``
    form a single group.
    """
    ap = Path(archive_path)
    _cfg_align_root = app_config or AppConfig()
    input_root = _archive_preprocess_lights_root(
        ap,
        app_config=_cfg_align_root,
        draft_id=draft_id,
        db=None,
    )
    if not input_root.exists():
        log_event(f"[X] ERROR: Input path {input_root} does not exist! Trying fallback...")
        processed_lights = ap / "processed" / "lights"
        subdirs: list[Path] = []
        try:
            if processed_lights.is_dir():
                subdirs = [d for d in processed_lights.iterdir() if d.is_dir()]
        except Exception:  # noqa: BLE001
            subdirs = []
        if subdirs:
            subdirs = sorted(subdirs, key=lambda p: p.name.casefold())
            input_root = subdirs[0]
            log_event(f"[OK] Fallback found: {input_root}")
    det_top = input_root
    ali_top = ap / "detrended_aligned" / "lights"
    os.makedirs(str(ali_top), exist_ok=True)
    ps_top = ap / "platesolve"
    os.makedirs(str(ps_top), exist_ok=True)
    files_all = _iter_fits_recursive(det_top)
    _n_before_qc = len(files_all)
    _qc_csv = find_qc_metrics_csv(
        ap,
        app_config=_cfg_align_root,
        draft_id=draft_id,
        db=None,
    )
    if _qc_csv is None or not _qc_csv.is_file():
        raise FileNotFoundError(
            "Preprocess QC step required; run Analyze/preprocess first to produce qc_metrics.csv"
        )
    files_all, _qc_status_map = filter_files_by_qc_metrics_allowlist(files_all, _qc_csv)
    log_event(
        f"QC allowlist: using {len(files_all)}/{_n_before_qc} lights FITS "
        f"(qc_metrics.csv) for alignment from {det_top}"
    )
    from invariants_runtime import check_qc01_skipproc_alignment  # noqa: PLC0415

    check_qc01_skipproc_alignment(files_all, _qc_csv, meta={"invariants": []})
    _bayermask = None
    if id_equipment is not None and _cfg_align_root is not None:
        try:
            from database import VyvarDatabase

            _osc_db = VyvarDatabase(str(_cfg_align_root.database_path))
            _bayermask = _osc_db.get_equipment_bayermask(int(id_equipment))
        except Exception:  # noqa: BLE001
            _bayermask = None
    from invariants_runtime import check_osc01_channel_extraction_required  # noqa: PLC0415

    check_osc01_channel_extraction_required(
        files_all,
        equipment_bayermask=_bayermask,
        meta={"invariants": []},
    )
    _n_root_only = len(list(det_top.glob("*.fits"))) if det_top.exists() else 0
    _n_all = len(files_all)
    if _n_root_only != _n_all:
        log_event(
            f"FITS celkom: {_n_all} pod {det_top} ({_n_root_only} priamo v koreni; "
            f"ostatne v podpriecinkoch napr. filter/exp/binning)."
        )
    else:
        log_event(f"FITS celkom: {_n_all} v {det_top}")
    log_event(f"Alignment input root: {det_top}")
    if not files_all:
        raise FileNotFoundError(
            f"Chybaju FITS v {det_top}. Plate solve cita len **spracovane** snimky. "
            "Najprv spusti **MAKE MASTERSTAR** po kroku **Analyze** (zapis z "
            f"`{ap / 'calibrated' / 'lights'}` -> `{ap / 'processed' / 'lights'}` alebo starsie `{ap / 'detrended' / 'lights'}`)."
        )
    job_list: list[dict[str, Any]] = []
    # Group strictly by real folder structure under processed/detrended lights.
    # DO NOT create scan_<id> subfolders: users expect platesolve/ + detrended_aligned/ to mirror
    # processed/lights/<setup_name>/ layout (as in older drafts like draft_000231).
    groups = _partition_detrended_by_subfolder(files_all, det_top)
    for gkey in sorted(groups.keys()):
        gfs = groups[gkey]
        if not gfs:
            continue
        _setup_name = Path(gkey).name if gkey else ""
        if _setup_name:
            log_event(
                f"Alignment subgroup detected: setup_name={_setup_name} "
                f"(input={det_top / gkey}, files={len(gfs)})"
            )
        if gkey:
            job_list.append(
                {
                    "gkey": gkey,
                    "scanning_id": None,
                    "detrended_root": det_top / gkey,
                    "aligned_root": ali_top / gkey,
                    "platesolve_dir": ps_top / gkey,
                    "files": gfs,
                }
            )
        else:
            job_list.append(
                {
                    "gkey": "",
                    "scanning_id": None,
                    "detrended_root": det_top,
                    "aligned_root": ali_top,
                    "platesolve_dir": ps_top,
                    "files": gfs,
                }
            )
    if len(job_list) > 1:
        log_event(
            "Astrometria: viacero pod-pozorovani (podpriecinky v processed|detrended/lights) - "
            "samostatne zarovnanie, MASTERSTAR a katalogy pre kazdu skupinu."
        )
    from osc_align import partition_jobs_for_osc_alignment

    job_list, osc_meta = partition_jobs_for_osc_alignment(job_list)
    if osc_meta.get("has_osc_bundles"):
        from invariants_runtime import check_osc02_unified_frame_sets

        check_osc02_unified_frame_sets(osc_meta["bundles"], meta={"invariants": []})
    _ms_paths = [str(x).strip() for x in (masterstar_candidate_paths or []) if str(x).strip()]
    try:
        _ms_pct = float(masterstar_selection_pct) if masterstar_selection_pct is not None else None
    except (TypeError, ValueError):
        _ms_pct = None
    if _ms_pct is not None and (not math.isfinite(_ms_pct) or _ms_pct <= 0):
        _ms_pct = None
    _md_bpm_job: str | None = None
    if master_dark_path is not None and str(master_dark_path).strip():
        _mdp = Path(str(master_dark_path))
        if _mdp.is_file():
            _md_bpm_job = str(_mdp.resolve())
    _multi_obs = len(job_list) > 1
    for _j in job_list:
        _j["masterstar_candidate_paths"] = [] if _multi_obs else _ms_paths
        _j["masterstar_selection_pct"] = _ms_pct
        if _md_bpm_job:
            _j["master_dark_path"] = _md_bpm_job

    _kw: dict[str, Any] = dict(
        archive_path=archive_path,
        astrometry_api_key=astrometry_api_key,
        max_control_points=max_control_points,
        min_detected_stars=min_detected_stars,
        max_detected_stars=max_detected_stars,
        platesolve_backend=platesolve_backend,
        plate_solve_fov_deg=plate_solve_fov_deg,
        max_extra_platesolve=max_extra_platesolve,
        catalog_match_max_sep_arcsec=catalog_match_max_sep_arcsec,
        saturate_level_fraction=saturate_level_fraction,
        max_catalog_rows=max_catalog_rows,
        n_comparison_stars=n_comparison_stars,
        require_non_variable_comparisons=require_non_variable_comparisons,
        faintest_mag_limit=faintest_mag_limit,
        dao_threshold_sigma=dao_threshold_sigma,
        id_equipment=id_equipment,
        draft_id=draft_id,
        catalog_local_gaia_only=catalog_local_gaia_only,
        build_masterstar_and_catalogs=build_masterstar_and_catalogs,
        progress_cb=progress_cb,
        ram_align_and_catalog=ram_align_and_catalog,
        app_config=app_config,
    )
    if osc_meta.get("has_osc_bundles"):
        return _run_osc_multi_group_alignment(
            job_list=job_list,
            osc_bundles=osc_meta["bundles"],
            align_kw=_kw,
        )
    if len(job_list) == 1:
        return _astrometry_align_impl_body(job=job_list[0], **_kw)

    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for j in job_list:
        _gkey = str(j.get("gkey") or "")
        _setup = Path(_gkey).name if _gkey else "(root)"
        try:
            reports.append(_astrometry_align_impl_body(job=j, **_kw))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Astrometria/MASTERSTAR failed for set %s: %s", _setup, exc)
            log_event(
                f"! Set {_setup}: plate-solve/MASTERSTAR zlyhal - set sa preskakuje, "
                f"pokracujem dalsim setom. Dovod: {exc}"
            )
            skipped.append(
                {"gkey": _gkey, "setup": _setup, "solved": False, "skipped_reason": str(exc)}
            )
            continue

    reports, skipped = _pass2_sibling_wcs_recovery(
        reports=reports,
        skipped=skipped,
        job_list=job_list,
        align_kw=_kw,
    )

    if not reports:
        raise RuntimeError(
            "Astrometria: ziadny set nepresiel plate-solve/MASTERSTAR. "
            + "; ".join(f"{s['setup']}: {s['skipped_reason']}" for s in skipped)
        )

    merged = _merge_astrometry_group_reports(reports)
    if skipped:
        merged["skipped_subgroups"] = skipped
        log_event(
            f"Astrometria: {len(reports)} setov OK, {len(skipped)} preskocenych: "
            + ", ".join(s["setup"] for s in skipped)
        )
    return merged


def _fits_meta_ra_deg(value: Any) -> float:
    r = _fits_header_parse_ra_deg(value)
    return float(r) if r is not None else 0.0


def _fits_meta_dec_deg(value: Any) -> float:
    d = _fits_header_parse_dec_deg(value)
    return float(d) if d is not None else 0.0


def _db_for_calibration_tasks(
    qc_opt: dict[str, Any] | None,
) -> VyvarDatabase | None:
    """Open DB once per worker / sequential pass when post-calibrate QC needs it."""
    q = qc_opt or {}
    p: str | None = None
    if q.get("enabled") and q.get("db_path"):
        p = str(q["db_path"])
    if not p:
        return None
    try:
        return VyvarDatabase(Path(p))
    except Exception:  # noqa: BLE001
        return None


def _qc_pack_from_config(
    cfg: AppConfig,
    *,
    draft_id: int | None,
    observation_id: str | None,
) -> dict[str, Any]:
    """Post-calibration QC limits + DB linkage (``manifest files[]`` update by raw light path)."""
    en = bool(cfg.qc_after_calibrate_enabled)
    _dao = float(cfg.qc_dao_detection_sigma)
    if not math.isfinite(_dao) or _dao <= 0:
        _dao = 5.0
    id_equipments: int | None = None
    if draft_id is not None and str(cfg.database_path).strip():
        try:
            _dbp = Path(cfg.database_path)
            if _dbp.is_file():
                _vdb = VyvarDatabase(_dbp)
                id_equipments = _vdb.get_draft_equipment_id(int(draft_id))
        except Exception:  # noqa: BLE001
            id_equipments = None
    return {
        "enabled": en,
        "max_hfr": float(cfg.qc_max_hfr),
        "max_hfr_fwhm_ratio": getattr(cfg, "qc_max_hfr_fwhm_ratio", None),
        "min_stars": int(cfg.qc_min_stars),
        "max_bg_rms": cfg.qc_max_background_rms,
        "dao_detection_sigma": _dao,
        "db_path": str(Path(cfg.database_path).resolve()) if en else None,
        "draft_id": draft_id,
        "observation_id": observation_id,
        "id_equipments": id_equipments,
        "dao_qc_in_calibrate": bool(cfg.dao_qc_in_calibrate),
    }


def _qc_center_crop_for_stars(data: "np.ndarray", max_side: int = 1000) -> "np.ndarray":
    """Central crop for star metrics when the frame is larger than ``max_side``."""
    import numpy as np

    a = np.asarray(data, dtype=np.float32)
    if a.ndim != 2:
        return a
    h, w = int(a.shape[0]), int(a.shape[1])
    if h <= max_side and w <= max_side:
        return a
    cy, cx = h // 2, w // 2
    hs, ws = max_side // 2, max_side // 2
    y0, y1 = max(0, cy - hs), min(h, cy + hs)
    x0, x1 = max(0, cx - ws), min(w, cx + ws)
    return np.asarray(a[y0:y1, x0:x1], dtype=np.float32)


def _half_flux_radius_in_cutout(cut: "np.ndarray", xc: float, yc: float) -> float:
    import numpy as np

    cut = np.asarray(cut, dtype=np.float64)
    h, w = cut.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2).ravel()
    pix = np.maximum(cut.ravel(), 0.0)
    order = np.argsort(r)
    r = r[order]
    pix = pix[order]
    tot = float(np.sum(pix))
    if tot <= 0 or not math.isfinite(tot):
        return float("nan")
    cum = np.cumsum(pix)
    idx = int(np.searchsorted(cum, 0.5 * tot))
    idx = min(max(idx, 0), cum.size - 1)
    return float(r[idx])


def _mean_hfr_bright_stars_dao(
    crop: "np.ndarray",
    *,
    max_stars: int = 50,
    dao_detection_sigma: float = 5.0,
) -> tuple[float | None, int, float | None]:
    """Median half-flux radius [px] on up to ``max_stars`` brightest DAO sources.

    Returns (HFR, n_detected, fwhm_guess_px).
    """
    import numpy as np
    from photutils.detection import DAOStarFinder

    img = np.asarray(crop, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return None, 0, None
    _, med, std = plain_mean_med_std(img[finite], sigma=3.0, maxiters=5)
    std = float(std)
    if not math.isfinite(std) or std <= 0:
        return None, 0, None
    img2 = np.asarray(img - float(med), dtype=np.float32)
    img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmedian(img2)) < 0:
        img2 = -img2
    fwhm_guess = _estimate_dao_fwhm_guess(img2, std)
    sig0 = float(dao_detection_sigma) if math.isfinite(float(dao_detection_sigma)) and float(dao_detection_sigma) > 0 else 5.0
    daofind = DAOStarFinder(
        fwhm=float(fwhm_guess),
        threshold=sig0 * std,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = daofind(img2)
    if tbl is None or len(tbl) == 0:
        daofind = DAOStarFinder(
            fwhm=float(fwhm_guess),
            threshold=max(3.5, 0.7 * sig0) * std,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = daofind(img2)
    if tbl is None or len(tbl) == 0:
        return None, 0, float(fwhm_guess)
    n_det = int(len(tbl))
    tbl.sort("flux")
    tbl = tbl[::-1]
    h, w = img2.shape
    half = 10
    hfrs: list[float] = []
    for i in range(min(max_stars, len(tbl))):
        x0 = float(tbl["x_centroid"][i])
        y0 = float(tbl["y_centroid"][i])
        xi, yi = int(round(x0)), int(round(y0))
        y1, y2 = max(0, yi - half), min(h, yi + half + 1)
        x1, x2 = max(0, xi - half), min(w, xi + half + 1)
        sl = img2[y1:y2, x1:x2]
        if sl.shape[0] < 5 or sl.shape[1] < 5:
            continue
        hfr = _half_flux_radius_in_cutout(sl, x0 - x1, y0 - y1)
        if math.isfinite(hfr) and 0.2 < hfr < 50.0:
            hfrs.append(hfr)
    if not hfrs:
        return None, n_det, float(fwhm_guess)
    return float(np.nanmedian(np.asarray(hfrs, dtype=np.float64))), n_det, float(fwhm_guess)


def _post_calibration_qc_eval(
    data: "np.ndarray",
    *,
    limits: dict[str, Any],
    light_basename: str = "",
) -> dict[str, Any]:
    """Sky stats on full frame; star HFR/count on central crop if frame is large."""
    import numpy as np

    img = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        out = {
            "qc_passed": False,
            "hfr": None,
            "n_stars": 0,
            "sky_mean": None,
            "sky_median": None,
            "sky_rms": None,
            "reject_reasons": ["no finite pixels"],
        }
        if light_basename:
            log_event(f"Frame {light_basename} REJECTED (no finite pixels)")
        return out

    sky_mean, sky_med, sky_rms = plain_mean_med_std(img[finite], sigma=3.0, maxiters=5)
    sky_mean = float(sky_mean)
    sky_med = float(sky_med)
    sky_rms = float(sky_rms)

    crop = _qc_center_crop_for_stars(img, 1000)
    _ds = float(limits.get("dao_detection_sigma", 5.0))
    if not math.isfinite(_ds) or _ds <= 0:
        _ds = 5.0
    hfr_m, n_star, fwhm_guess = _mean_hfr_bright_stars_dao(crop, max_stars=50, dao_detection_sigma=_ds)

    from unit_resolver import resolve_hfr_limit_px

    class _QcHfrCfg:
        qc_max_hfr = float(limits.get("max_hfr", 5.0))
        qc_max_hfr_fwhm_ratio = limits.get("max_hfr_fwhm_ratio")

    max_h = resolve_hfr_limit_px(_QcHfrCfg(), fwhm_px=fwhm_guess)
    min_star = int(limits.get("min_stars", 10))
    max_rms = limits.get("max_bg_rms")

    reasons: list[str] = []
    ok = True
    if hfr_m is None or not math.isfinite(hfr_m):
        ok = False
        reasons.append("HFR unavailable")
    elif hfr_m > max_h:
        ok = False
        reasons.append(f"HFR: {hfr_m:.2f} > limit {max_h:.2f}")
    if n_star < min_star:
        ok = False
        reasons.append(f"stars: {n_star} < min {min_star}")
    if max_rms is not None and math.isfinite(float(max_rms)) and math.isfinite(sky_rms):
        if sky_rms > float(max_rms):
            ok = False
            reasons.append(f"background RMS: {sky_rms:.4g} > limit {float(max_rms):.4g}")

    if not ok and light_basename:
        if len(reasons) == 1 and reasons[0].startswith("HFR:"):
            log_event(f"Frame {light_basename} REJECTED ({reasons[0]})")
        else:
            log_event(f"Frame {light_basename} REJECTED ({'; '.join(reasons)})")

    return {
        "qc_passed": ok,
        "hfr": hfr_m,
        "n_stars": int(n_star),
        "sky_mean": sky_mean,
        "sky_median": sky_med,
        "sky_rms": sky_rms,
        "reject_reasons": reasons,
    }



def _strip_raw_linearity_header_keywords(hdr: fits.Header) -> None:
    """Remove FITS keys that describe the **raw** detector linearity range.

    After ``(light - dark) / flat`` the pixel scale is no longer raw ADU; keeping SATURATE/DATAMAX from the
    light frame makes viewers and automated limits disagree with the actual array values.
    """
    for key in (
        "SATURATE",
        "MAXLIN",
        "ESATUR",
        "LINLIMIT",
        "MAXADU",
        "DATAMAX",
        "MAXPIX",
    ):
        if key in hdr:
            try:
                del hdr[key]
            except KeyError:
                pass


def _vy_calib_status_numeric(flags: str) -> int:
    """Map ``VY_CFLAG`` to CALIB_STATUS-like 0/1/2 (reference: full / partial / raw)."""
    f = (flags or "").upper()
    if f == "DF":
        return 2
    if f == "D":
        return 1
    return 0


def _hdr_vy_cflag_str(hdr: fits.Header) -> str:
    raw = hdr.get("VY_CFLAG")
    if isinstance(raw, tuple):
        return str(raw[0]).strip().upper() or "P"
    if raw is None:
        return "P"
    return str(raw).strip().upper() or "P"


def _calibration_flags(
    *,
    used_dark: bool,
    used_flat: bool,
    passthrough: bool,
    flat_skipped_no_dark: bool = False,
) -> str:
    """Build ``VY_CFLAG`` (D=dark, F=flat, DF=full, FS=flat skipped without dark, P=passthrough/raw)."""
    if passthrough:
        return "P"
    if flat_skipped_no_dark:
        return "FS"
    out = ""
    if used_dark:
        out += "D"
    if used_flat:
        out += "F"
    return out or "P"


def _calibration_type_from_flags(flags: str) -> str:
    f = (flags or "").upper()
    if f == "P":
        return "PASSTHROUGH"
    if f == "DF":
        return "DARK+FLAT"
    if f == "D":
        return "DARK_ONLY"
    if f == "F":
        return "FLAT_ONLY"
    if f == "FS":
        return "RAW_FLAT_SKIPPED"
    return "PASSTHROUGH"


def _calibrate_one_light_apply_masters_in_ram(
    *,
    src: Path,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
    flat_norm_floor: float = 0.15,
    flat_cache: dict[str, Any] | None = None,
    flat_median_scale: dict[str, float] | None = None,
    md_data_preload: Any = None,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    calibration_master_native_binning: int | None | object = _CALIB_MASTER_NB_UNSET,
    cal_diag_gate_result: CalDiagGateResult | None = None,
) -> tuple[Any, fits.Header, bool, bool]:
    """Apply dark/flat in RAM; return ``(data_float32, header, used_dark, used_flat)`` (no disk write).

    ``calibration_master_native_binning`` defaults to :data:`calibration.CALIBRATION_LIBRARY_NATIVE_BINNING`
    (CalibrationLibrary stores native masters; resample in RAM to match light ``XBINNING``).
    Pass ``None`` explicitly to read ``XBINNING`` from each master FITS (``calibration_library_native_binning: null`` in config).
    """
    import numpy as np

    if calibration_master_native_binning is _CALIB_MASTER_NB_UNSET:
        _mb_lib = int(CALIBRATION_LIBRARY_NATIVE_BINNING)
    elif calibration_master_native_binning is None:
        _mb_lib = None
    else:
        _mb_lib = max(1, int(calibration_master_native_binning))

    fc: dict[str, Any] = flat_cache if flat_cache is not None else {}
    fms: dict[str, float] = flat_median_scale if flat_median_scale is not None else {}

    with fits.open(src, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)

    light_bx, light_by = fits_binning_xy_from_header(hdr)
    hdr["VY_CLBX"] = (int(light_bx), "Light XBINNING used for master matching / resampling")
    hdr["VY_CLBY"] = (int(light_by), "Light YBINNING (diagnostic)")
    if _mb_lib is None:
        hdr["VY_MBNC"] = (
            -1,
            "VYVAR: native XBINNING read from each CalibrationLibrary master FITS (XBINNING)",
        )
    else:
        hdr["VY_MBNC"] = (
            int(_mb_lib),
            "VYVAR: assumed native XBINNING of CalibrationLibrary master before resample to light",
        )

    md_data: np.ndarray | None = md_data_preload
    if _mb_lib is None or (md_data is not None and md_data.shape != data.shape):
        md_data = None
    if md_data is not None and master_dark_path is not None and master_dark_path.exists():
        if _mb_lib != light_bx:
            md_data = None

    if md_data is None and master_dark_path is not None and master_dark_path.exists():
        _drm = "sum"
        if cal_diag_gate_result is not None:
            _drm = convention_to_dark_mode(cal_diag_gate_result.convention)
        pm = get_processed_master(
            master_dark_path,
            light_bx,
            kind="dark",
            master_binning=_mb_lib,
            light_shape=data.shape,
            light_filename=src.name,
            dark_resample_mode=_drm,
        )
        if pm.resampled:
            log_event(
                f"Calibration: library master native {pm.master_binning}x{pm.master_binning} -> "
                f"resampled (RAM) to {light_bx}x{light_bx} for Light [{src.name}]"
            )
        md_data = pm.data

    used_dark = False
    used_flat = False

    if md_data is not None:
        data2, md2 = _match_and_crop_pair(data, md_data)
        data = data2 - md2
        used_dark = True

    flt = _safe_filter_token(str(hdr.get("FILTER") or hdr.get("FILT") or "NoFilter"))
    _obs_k = observation_group_key_from_metadata(fits_metadata_from_primary_header(hdr))
    hdr["VY_OBSG"] = (_obs_k, "VYVAR observation group: FILTER|EXPTIME|XBINNING")
    mf_path = None
    if masterflat_by_filter:
        c_obs = masterflat_by_filter.get(_obs_k)
        if c_obs is not None:
            p_obs = Path(c_obs)
            if p_obs.is_file():
                mf_path = p_obs
    if mf_path is None and masterflat_by_filter:
        c_f = masterflat_by_filter.get(flt) or masterflat_by_filter.get("NoFilter")
        if c_f is not None:
            p_f = Path(c_f)
            if p_f.is_file():
                mf_path = p_f
    if mf_path is not None and mf_path.exists() and used_dark:
        key = f"{flt}|{mf_path!s}|lb{light_bx}"
        if key not in fc:
            pmf = get_processed_master(
                mf_path,
                light_bx,
                kind="flat",
                master_binning=_mb_lib,
                light_shape=data.shape,
                light_filename=src.name,
                db=db,
                id_equipments=id_equipments,
            )
            if pmf.resampled:
                log_event(
                    f"Calibration: library flat native {pmf.master_binning}x{pmf.master_binning} -> "
                    f"resampled (RAM) to {light_bx}x{light_bx} for Light [{src.name}]"
                )
            flat = pmf.data
            flat = np.where(np.isfinite(flat) & (flat > 0), flat, 1.0).astype(np.float32)
            if pmf.flat_normalized_at_calibrate:
                fm = pmf.flat_median_adu_before_norm
                if fm is None or not np.isfinite(fm) or fm <= 0:
                    fm = float(np.nanmedian(flat))
                    if not np.isfinite(fm) or fm <= 0:
                        fm = 1.0
            else:
                fm = float(np.nanmedian(flat))
                if not np.isfinite(fm) or fm <= 0:
                    fm = 1.0
                flat = (flat / fm).astype(np.float32)
            flat = np.maximum(flat, float(flat_norm_floor)).astype(np.float32)
            fc[key] = flat
            fms[key] = fm
        flat_arr = fc[key]
        data2, flat2 = _match_and_crop_pair(data, flat_arr)
        data = data2 / flat2
        used_flat = True
    flat_skipped_no_dark = bool(mf_path is not None and mf_path.exists() and not used_dark)
    if flat_skipped_no_dark:
        log_event("Flat skipped because no Dark/Bias was subtracted to avoid over-correction.")

    if used_dark or used_flat:
        if used_flat or not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    hdr["VYVARCAL"] = (True, "VYVAR calibrated output")
    hdr["VY_DARK"] = (bool(used_dark), "MasterDark applied")
    hdr["VY_FLAT"] = (bool(used_flat), "MasterFlat applied")
    _flags = _calibration_flags(
        used_dark=bool(used_dark),
        used_flat=bool(used_flat),
        passthrough=False,
        flat_skipped_no_dark=flat_skipped_no_dark,
    )
    hdr["VY_CFLAG"] = (
        _flags,
        "VYVAR flags: DF=full, D=partial, FS=flat skipped (no dark), P=passthrough/raw",
    )
    hdr["VY_CALIB"] = (_calibration_type_from_flags(_flags), "Calibration mode")
    hdr["VY_CALST"] = (
        int(_vy_calib_status_numeric(_flags)),
        "VYVAR CALIB_STATUS: 2=full DF, 1=partial D, 0=raw (incl. FS passthrough)",
    )
    if _flags == "FS":
        hdr["VY_WARN"] = (
            True,
            "Flat available but not applied: subtract MasterDark first (over-correction risk).",
        )
    if master_dark_path is not None:
        hdr["VY_MDP"] = (str(master_dark_path.name)[:68], "MasterDark filename")
    if mf_path is not None:
        hdr["VY_MFP"] = (str(Path(mf_path).name)[:68], "MasterFlat filename")
    if used_flat:
        try:
            key_m = f"{flt}|{mf_path!s}|lb{light_bx}"
            hdr["VY_FLATM"] = (
                float(fms[key_m]),
                "Median ADU of master flat at target resample before normalize-to-1 (legacy: before pipeline division)",
            )
            hdr["VY_FLFL"] = (
                float(flat_norm_floor),
                "Min normalized flat before division (limits local gain from flat only)",
            )
        except KeyError:
            pass

    if used_dark or used_flat:
        _strip_raw_linearity_header_keywords(hdr)
        hdr.add_history(
            "VYVAR: cleared raw SATURATE/DATAMAX/... "
            "(pixels = (light-dark)/median-norm flat; not raw ADU)."
        )

    apply_cal_diag_headers(hdr, cal_diag_gate_result)

    return data, hdr, used_dark, used_flat


def _sync_obs_calibration_state_with_retry(
    db: VyvarDatabase | None,
    *,
    raw_light_path: Path,
    draft_id: int | None,
    observation_id: str | None,
    is_calibrated: int,
    calib_type: str,
    calib_flags: str,
    stats: dict[str, Any] | None = None,
) -> bool:
    """Sync manifest files[] cal state after successful calibrate; retry once then ERROR + count."""
    if db is None:
        return True
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            db.update_obs_file_calibration_state_by_raw_light_path(
                raw_light_path,
                draft_id=draft_id,
                observation_id=observation_id,
                is_calibrated=is_calibrated,
                calib_type=calib_type,
                calib_flags=calib_flags,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0426] [CAL-DIAG] `_quality_inspection_dao_metrics_array` in calibrate path logs WARNING and o...: %s', exc)
            last_exc = exc
            if attempt == 0:
                LOGGER.warning(
                    "CAL-DIAG: manifest files[] cal sync retry after failure (%s): %s",
                    raw_light_path.name,
                    exc,
                )
    if last_exc is not None:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().calibrate_db_sync_fail += 1
        LOGGER.error(
            "CAL-DIAG: manifest files[] cal sync failed after calibrate (%s): %s",
            raw_light_path.name,
            last_exc,
        )
        if stats is not None:
            stats["cal_db_sync_failures"] = int(stats.get("cal_db_sync_failures", 0)) + 1
            errs = stats.setdefault("cal_db_sync_errors", [])
            if isinstance(errs, list):
                errs.append({"file": raw_light_path.name, "error": str(last_exc)})
    return False


def _calibrate_one_light_disk(
    *,
    src: Path,
    dst: Path,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
    flat_norm_floor: float = 0.15,
    flat_cache: dict[str, Any] | None = None,
    flat_median_scale: dict[str, float] | None = None,
    md_data_preload: Any = None,
    db: VyvarDatabase | None = None,
    qc_pack: dict[str, Any] | None = None,
    calibration_master_native_binning: int | None | object = _CALIB_MASTER_NB_UNSET,
    cal_diag_gate_result: CalDiagGateResult | None = None,
) -> tuple[bool, bool, dict[str, Any] | None, str, dict[str, Any] | None]:
    """Apply master dark / flat to one light FITS and write ``dst``.

    Uses ``fits.open(..., memmap=False)`` and ``with`` blocks so BZERO/BSCALE frames load reliably and
    file handles close after each read (arrays are copied into RAM before processing).

    Returns ``(used_dark, used_flat, qc_summary, vy_cflag, perf10_qc)`` where ``qc_summary`` is set when
    post-calibration QC ran; ``perf10_qc`` holds DAO inspection metrics when ``dao_qc_in_calibrate``;
    ``vy_cflag`` matches ``VY_CFLAG`` written to the FITS (see calibration decision table).
    """
    import numpy as np

    dst.parent.mkdir(parents=True, exist_ok=True)

    _id_eq = None
    if qc_pack is not None:
        try:
            _raw = qc_pack.get("id_equipments")
            _id_eq = int(_raw) if _raw is not None else None
        except (TypeError, ValueError):
            _id_eq = None
    data, hdr, used_dark, used_flat = _calibrate_one_light_apply_masters_in_ram(
        src=src,
        master_dark_path=master_dark_path,
        masterflat_by_filter=masterflat_by_filter,
        flat_norm_floor=flat_norm_floor,
        flat_cache=flat_cache,
        flat_median_scale=flat_median_scale,
        md_data_preload=md_data_preload,
        db=db,
        id_equipments=_id_eq,
        calibration_master_native_binning=calibration_master_native_binning,
        cal_diag_gate_result=cal_diag_gate_result,
    )

    qc_summary: dict[str, Any] | None = None
    _osc_mosaic = _valid_bayerpat_from_header(hdr) is not None
    if (used_dark or used_flat) and qc_pack and qc_pack.get("enabled") and not _osc_mosaic:
        limits = {
            "max_hfr": float(qc_pack.get("max_hfr", 5.0)),
            "min_stars": int(qc_pack.get("min_stars", 10)),
            "max_bg_rms": qc_pack.get("max_bg_rms"),
            "dao_detection_sigma": float(qc_pack.get("dao_detection_sigma", 5.0)),
        }
        qc_summary = _post_calibration_qc_eval(
            np.asarray(data, dtype=np.float32),
            limits=limits,
            light_basename=src.name,
        )
        hdr["VYQCPASS"] = (bool(qc_summary["qc_passed"]), "Post-calibration QC pass")
        hfrv = qc_summary.get("hfr")
        if hfrv is not None and math.isfinite(float(hfrv)):
            hdr["VY_QCHFR"] = (float(hfrv), "QC median HFR [px]")
        hdr["VY_QCNS"] = (int(qc_summary["n_stars"]), "QC DAO detections (central crop if large)")
        sm = qc_summary.get("sky_median")
        if sm is not None and math.isfinite(float(sm)):
            hdr["VY_QCBG"] = (float(sm), "QC sigma-clipped sky median")
        sr = qc_summary.get("sky_rms")
        if sr is not None and math.isfinite(float(sr)):
            hdr["VY_QCRMS"] = (float(sr), "QC sigma-clipped sky RMS")
        if qc_pack.get("draft_id") is not None or (
            qc_pack.get("observation_id") not in (None, "")
        ):
            try:
                db_q = db if db is not None else _db_for_calibration_tasks(qc_pack)
                if db_q is not None:
                    db_q.update_obs_file_qc_by_raw_light_path(
                        src,
                        draft_id=qc_pack.get("draft_id"),
                        observation_id=qc_pack.get("observation_id"),
                        qc_hfr=qc_summary.get("hfr"),
                        qc_stars=int(qc_summary["n_stars"]),
                        qc_background=qc_summary.get("sky_median"),
                        qc_bg_rms=qc_summary.get("sky_rms"),
                        qc_passed=bool(qc_summary["qc_passed"]),
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("manifest files[] QC update failed: %s", exc)

    perf10_qc: dict[str, Any] | None = None
    if (
        not _osc_mosaic
        and qc_pack
        and bool(qc_pack.get("dao_qc_in_calibrate"))
    ):
        try:
            perf10_qc = _quality_inspection_dao_metrics_array(
                np.asarray(data, dtype=np.float32),
                hdr,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("[PERF-10] DAO QC in calibrate failed for %s: %s", src.name, exc)

    from cal_stage import stamp_cal_stage_headers  # noqa: PLC0415

    cal_stage_token = "PURE"
    cal_datasum = stamp_cal_stage_headers(hdr, data, stage=cal_stage_token)
    fits.writeto(dst, _as_fits_float32_image(data), header=hdr, overwrite=True)
    if qc_pack is not None and qc_pack.get("draft_id") is not None:
        try:
            db_cs = db if db is not None else _db_for_calibration_tasks(qc_pack)
            if db_cs is not None:
                db_cs.update_obs_file_cal_stage_by_raw_light_path(
                    src,
                    draft_id=int(qc_pack["draft_id"]),
                    observation_id=qc_pack.get("observation_id"),
                    cal_stage=cal_stage_token,
                    cal_datasum=cal_datasum,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("manifest cal_stage sync failed after calibrate: %s", exc)
    return used_dark, used_flat, qc_summary, _hdr_vy_cflag_str(hdr), perf10_qc


_cal_batch_flat_cache: dict[str, Any] | None = None
_cal_batch_flat_median: dict[str, float] | None = None
_cal_batch_md_preload: Any = None
_cal_batch_native_binning: int | None = 1
_cal_batch_cal_diag: CalDiagSession | None = None


def _init_calibrate_batch_worker(
    initargs: tuple[str | None, int | None, dict[str, Any] | None],
) -> None:
    """Per-subprocess caches; ``native_binning`` = CalibrationLibrary master convention (``None`` = read FITS)."""
    global _cal_batch_flat_cache, _cal_batch_flat_median, _cal_batch_md_preload
    global _cal_batch_native_binning, _cal_batch_cal_diag
    _md_s, native_b, cal_diag_blob = initargs
    _ = _md_s  # path reserved for future worker-side dark preload
    _cal_batch_flat_cache = {}
    _cal_batch_flat_median = {}
    _cal_batch_md_preload = None
    _cal_batch_cal_diag = _cal_diag_session_from_export(cal_diag_blob)
    if native_b is None:
        _cal_batch_native_binning = None
    else:
        try:
            _cal_batch_native_binning = max(1, int(native_b))
        except (TypeError, ValueError):
            _cal_batch_native_binning = int(CALIBRATION_LIBRARY_NATIVE_BINNING)


def _calibrate_batch_process_one(
    item: tuple[
        str,
        str,
        str | None,
        dict[str, str | None],
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]
    | tuple[str, str, str | None, dict[str, str | None], dict[str, Any] | None]
    | tuple[str, str, str | None, dict[str, str | None]],
) -> dict[str, Any]:
    """Picklable worker: calibrate one light; returns ``dst`` path on success."""
    global _cal_batch_flat_cache, _cal_batch_flat_median, _cal_batch_md_preload
    global _cal_batch_native_binning, _cal_batch_cal_diag
    qc_opt: dict[str, Any] | None = None
    if len(item) == 4:
        src_s, dst_s, md_s, mf_map = item  # type: ignore[misc]
    else:
        src_s, dst_s, md_s, mf_map, qc_opt = item  # type: ignore[misc]
    fc = _cal_batch_flat_cache
    fm = _cal_batch_flat_median
    if fc is None or fm is None:
        fc, fm = {}, {}
    src_p = Path(src_s)
    dst_p = Path(dst_s)
    try:
        _ok = _obs_group_key_from_light_path(src_p)
        if _cal_batch_cal_diag is not None and is_obs_group_aborted(_cal_batch_cal_diag, _ok):
            if dst_p.exists():
                try:
                    dst_p.unlink()
                except OSError:
                    pass
            return {
                "src": src_s,
                "dst": dst_s,
                "ok": True,
                "error": None,
                "qc_summary": None,
                "perf10_qc": None,
                "traceback": None,
                "vy_cflag": "P",
                "skipped": True,
            }
        md_use: Path | None = Path(md_s) if md_s else None
        _md_obs = (qc_opt or {}).get("master_dark_by_obs_key") or {}
        if _md_obs:
            _alt = _md_obs.get(_ok)
            if _alt is not None and str(_alt).strip() != "":
                _pa = Path(str(_alt))
                if _pa.is_file():
                    md_use = _pa
        light_bx = _light_binning_from_path(src_p)
        gr = None
        if _cal_batch_cal_diag is not None:
            gr = gate_result_for_frame(
                _cal_batch_cal_diag,
                obs_group_key=_ok,
                dark_path=md_use,
                light_binning=light_bx,
            )
        md_np = _cal_batch_md_preload
        if md_use is not None and md_use.is_file():
            with fits.open(src_p, memmap=False) as hdul:
                lshape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
            md_np = dark_np_for_cal_diag(
                _cal_batch_cal_diag or CalDiagSession(),
                master_binning=_cal_batch_native_binning,
                dark_path=md_use,
                light_binning=light_bx,
                light_shape=lshape,
                light_filename=src_p.name,
                gate_result=gr,
            )
        mf: dict[str, Path | None] = {str(k): Path(v) if v else None for k, v in mf_map.items()}
        db_w = _db_for_calibration_tasks(qc_opt)
        _ud, _uf, qc_sum, _cf, perf10_qc = _calibrate_one_light_disk(
            src=src_p,
            dst=dst_p,
            master_dark_path=md_use,
            masterflat_by_filter=mf,
            flat_cache=fc,
            flat_median_scale=fm,
            md_data_preload=md_np,
            db=db_w,
            qc_pack=qc_opt,
            calibration_master_native_binning=_cal_batch_native_binning,
            cal_diag_gate_result=gr,
        )
        return {
            "src": src_s,
            "dst": dst_s,
            "ok": True,
            "error": None,
            "qc_summary": qc_sum,
            "perf10_qc": perf10_qc,
            "traceback": None,
            "vy_cflag": _cf,
        }
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0428] [CAL-DIAG] Passthrough reuse path: `update_obs_file_calibration_state_by_raw_light_path...: %s', exc)
        tb = traceback.format_exc()
        LOGGER.error("calibrate_batch worker: %s -> %s\n%s", src_s, exc, tb)
        try:
            log_exception(f"CHYBA WORKERA: {Path(src_s).name}", exc)
        except Exception:  # noqa: BLE001
            pass
        return {
            "src": src_s,
            "dst": None,
            "ok": False,
            "error": str(exc),
            "qc_summary": None,
            "traceback": tb,
        }


def _has_usable_master_dark(path: Path | None) -> bool:
    return bool(path is not None and Path(path).is_file())



def _passthrough_lights_to_calibrated(
    *,
    lights_root: Path,
    calibrated_root: Path,
    progress_cb: "callable | None" = None,
    database_path: Path | None = None,
    draft_id: int | None = None,
    observation_id: str | None = None,
) -> dict[str, Any]:
    """Passthrough mode: copy raw lights to calibrated and mark FITS header."""
    files = _iter_light_fits(lights_root)
    total = len(files)
    calibrated_root.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "processed": 0,
        "used_dark": 0,
        "used_flat": 0,
        "copied_only": 0,
        "errors": 0,
        "calibrate_workers": 1,
        "qc_evaluated": 0,
        "qc_rejected": 0,
        "passthrough_mode": True,
        "passthrough_existing_reused": 0,
    }
    db_pt: VyvarDatabase | None = None
    try:
        if database_path is not None and Path(database_path).is_file():
            db_pt = VyvarDatabase(Path(database_path))
    except Exception:  # noqa: BLE001
        db_pt = None
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(lights_root)
        dst = calibrated_root / rel
        if progress_cb is not None:
            progress_cb(i, total, f"Passthrough {src.name}")
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                # Reuse previous calibrated output; do not fail when rerunning.
                stats["processed"] += 1
                stats["copied_only"] += 1
                stats["passthrough_existing_reused"] += 1
                if db_pt is not None:
                    try:
                        db_pt.update_obs_file_calibration_state_by_raw_light_path(
                            src,
                            draft_id=draft_id,
                            observation_id=observation_id,
                            is_calibrated=0,
                            calib_type="PASSTHROUGH",
                            calib_flags="P",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                continue
            with fits.open(src, memmap=False) as hdul:
                hdr = hdul[0].header.copy()
                data = np.asarray(hdul[0].data, dtype=np.float32)
            hdr["VYVARCAL"] = (True, "VYVAR calibrated output")
            hdr["VY_DARK"] = (False, "MasterDark applied")
            hdr["VY_FLAT"] = (False, "MasterFlat applied")
            hdr["VY_CFLAG"] = ("P", "VYVAR flags: DF=full, D=partial, FS=flat skipped, P=passthrough/raw")
            hdr["VY_CALIB"] = ("PASSTHROUGH", "Calibration mode")
            hdr["VY_CALST"] = (0, "VYVAR CALIB_STATUS: 2=full DF, 1=partial D, 0=raw (passthrough)")
            hdr.add_history("No calibration frames applied.")
            passthrough_cal_diag_headers(hdr)
            from cal_stage import stamp_cal_stage_headers  # noqa: PLC0415

            cal_stage_token = "PASSTHROUGH"
            cal_datasum = stamp_cal_stage_headers(hdr, data, stage=cal_stage_token)
            fits.writeto(dst, _as_fits_float32_image(data), header=hdr, overwrite=True)
            stats["processed"] += 1
            stats["copied_only"] += 1
            if db_pt is not None:
                try:
                    db_pt.update_obs_file_calibration_state_by_raw_light_path(
                        src,
                        draft_id=draft_id,
                        observation_id=observation_id,
                        is_calibrated=0,
                        calib_type="PASSTHROUGH",
                        calib_flags="P",
                    )
                    db_pt.update_obs_file_cal_stage_by_raw_light_path(
                        src,
                        draft_id=draft_id,
                        observation_id=observation_id,
                        cal_stage=cal_stage_token,
                        cal_datasum=cal_datasum,
                    )
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            stats["errors"] += 1
    try:
        if db_pt is not None:
            db_pt.conn.close()
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0431] `filter_light_paths_for_calibration_db` failure logs and skips IS_REJECTED filter, pote...: %s', exc)
        pass
    return stats


def calibrate_lights_to_calibrated(
    *,
    lights_root: Path,
    calibrated_root: Path,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
    progress_cb: "callable | None" = None,
    pipeline_config: AppConfig | None = None,
    equipment_id: int | None = None,
    max_workers: int | None = None,
    draft_id: int | None = None,
    observation_id: str | None = None,
    masterflat_by_obs_key: dict[str, str | Path | None] | None = None,
    master_dark_by_obs_key: dict[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    """Apply available masters to lights and write into calibrated_root.

    Works in all modes:
    - dark+flat
    - only dark
    - none (passthrough copy when dark is unavailable)

    MasterFlat is **median-normalized to ~1.0** before division so calibrated values stay in the same order
    of magnitude as (light - dark) and extreme spikes from tiny flat pixels are avoided.

    After normalization, each flat pixel is **clamped below by a small positive floor** so dust donuts and
    dead columns cannot divide the image by near-zero (which would create nonsense ADU far above the
    detector range in viewers).

    Pixels are **not** expected to match raw ADU counts: calibration is ``(L - D) / F_norm`` with
    ``F_norm = flat / median(flat)``. Raw linearity keywords (SATURATE, DATAMAX, ...) are dropped when any
    calibration is applied so headers stay consistent with the stored float image.

    Parallelism uses ``max_workers`` or auto ``pipeline_config.qc_preprocess_workers`` only when environment
    variable ``VYVAR_CALIBRATE_MP`` is set to ``1``/``true`` (default is sequential for clearer tracebacks).
    """
    import numpy as np

    cfg = pipeline_config or AppConfig()
    qc_pack = _qc_pack_from_config(cfg, draft_id=draft_id, observation_id=observation_id)
    nw = max_workers if max_workers is not None else int(cfg.qc_preprocess_workers)
    nw = max(1, min(32, int(nw)))
    if not _vyvar_calibrate_multiprocessing_enabled():
        nw = 1
    if master_dark_by_obs_key:
        nw = 1

    mf_merged: dict[str, Path | None] = {}
    for k, v in (masterflat_by_filter or {}).items():
        mf_merged[str(k)] = None if v is None else Path(v)
    for k, v in (masterflat_by_obs_key or {}).items():
        mf_merged[str(k)] = None if v is None or str(v).strip() == "" else Path(v)
    masterflat_by_filter = mf_merged

    calibrated_root.mkdir(parents=True, exist_ok=True)
    _log_calibration_io_preflight(
        calibrated_root=calibrated_root,
        master_dark_path=master_dark_path,
        masterflat_by_filter=masterflat_by_filter,
    )

    _has_dark_from_obs = any(
        Path(v).is_file() for v in (master_dark_by_obs_key or {}).values() if v is not None and str(v).strip() != ""
    )
    _has_dark_any = _has_usable_master_dark(master_dark_path) or bool(_has_dark_from_obs)
    # Dark-first policy: if no usable dark exists, keep pipeline alive in passthrough mode.
    if not _has_dark_any:
        log_event(
            "Calibration Passthrough: missing MasterDark -> "
            "copy Raw/lights to calibrated/lights with VY_CALIB=PASSTHROUGH."
        )
        return _passthrough_lights_to_calibrated(
            lights_root=lights_root,
            calibrated_root=calibrated_root,
            progress_cb=progress_cb,
            database_path=Path(cfg.database_path) if str(cfg.database_path).strip() else None,
            draft_id=draft_id,
            observation_id=observation_id,
        )

    md_pre: Any = None
    md_path_ok: Path | None = None
    md_init_str: str | None = None
    if master_dark_path is not None and master_dark_path.exists():
        md_path_ok = master_dark_path
        md_init_str = str(master_dark_path.resolve())
        with fits.open(master_dark_path, memmap=False) as hdul:
            md_pre = np.array(hdul[0].data, dtype=np.float32, copy=True)

    _native_b = _cfg_calibration_library_native_binning(cfg)

    mf_serial: dict[str, str | None] = {}
    for k, v in (masterflat_by_filter or {}).items():
        if v is None:
            mf_serial[str(k)] = None
        else:
            mf_serial[str(k)] = str(Path(v).resolve())

    flat_cache: dict[str, Any] = {}
    flat_median_scale: dict[str, float] = {}
    stats: dict[str, Any] = {
        "processed": 0,
        "used_dark": 0,
        "used_flat": 0,
        "copied_only": 0,
        "errors": 0,
        "calibrate_workers": 1,
        "qc_evaluated": 0,
        "qc_rejected": 0,
        "applied_focal_length": None,
        "applied_pixel_size": None,
        "perf10_qc_results": {},
        "cal_db_sync_failures": 0,
        "cal_db_sync_errors": [],
    }
    perf10_qc_results: dict[str, dict[str, Any]] = stats["perf10_qc_results"]

    files = _iter_light_fits(lights_root)
    _n_before_obs_filter = len(files)
    if cfg.database_path:
        try:
            _dbp = Path(cfg.database_path)
            if _dbp.is_file():
                files = filter_light_paths_for_calibration_db(
                    files,
                    database_path=_dbp,
                    draft_id=draft_id,
                    observation_id=observation_id,
                )
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0432] [CAL-DIAG] First-light metadata diagnostic (`extract_fits_metadata` for focal/pixel log...: %s', exc)
            log_event(f"manifest files[] IS_REJECTED filter skipped (error): {exc}")
    total = len(files)
    if total < _n_before_obs_filter:
        log_event(
            f"Kalibracia - vylucene {_n_before_obs_filter - total} suborov podla manifest files[] (IS_REJECTED=1 alebo mimo DB)"
        )

    _suffix_note = ", ".join(sorted(FITS_SUFFIXES_LOWER))
    _log_bits = [
        f"lights_root={lights_root.resolve()}",
        f"disk_fits_count={total} (suffixes {_suffix_note}, case-insensitive via path.suffix.casefold)",
    ]
    if observation_id or draft_id is not None:
        try:
            _dbc = VyvarDatabase(Path(cfg.database_path))
            if observation_id:
                _n_o = _dbc.count_obs_files_for_observation(str(observation_id))
                _log_bits.append(
                    f"manifest files[] count (obs={observation_id!r}) -> {_n_o} rows"
                )
            if draft_id is not None:
                _n_d = _dbc.count_obs_files_for_draft(int(draft_id))
                _log_bits.append(
                    f"manifest files[] count (draft_id={int(draft_id)}) -> {_n_d} rows"
                )
        except Exception as exc:  # noqa: BLE001
            _log_bits.append(f"manifest files[] count failed: {exc}")
    log_event("Kalibracia - vstupne subory: " + " | ".join(_log_bits))

    if total > 0:
        log_lights_binning_from_headers_preflight(files, context="Kalibracia")
        _diag_light = _pick_light_for_metadata_diagnostic(files)
        try:
            _db_cal_meta = VyvarDatabase(Path(cfg.database_path))
            try:
                _meta0 = extract_fits_metadata(
                    _diag_light,
                    db=_db_cal_meta,
                    app_config=cfg,
                    id_equipment=equipment_id,
                    draft_id=draft_id,
                )
                _log_calibration_metadata_diagnostic(_diag_light.name, _meta0)
                stats["applied_focal_length"] = _meta0.get("focal_length")
                stats["applied_pixel_size"] = _meta0.get("pixel_um")
            finally:
                _db_cal_meta.conn.close()
        except Exception as exc:  # noqa: BLE001
            log_event(f"DIAGNOSTIKA KALIBRACIE: metadata prveho suboru zlyhali: {exc!s}")

    db_main = _db_for_calibration_tasks(qc_pack)

    def _refresh_manifest_after_cal_qc() -> None:
        if draft_id is not None and db_main is not None:
            try:
                db_main._try_refresh_draft_manifest(int(draft_id))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("draft_manifest refresh after calibrate QC: %s", exc)

    cal_diag_session = CalDiagSession()

    def _resolve_dark_for_pregate(fp: Path, og: str, lb: int) -> Path | None:
        return _resolve_dark_path_for_light(
            src=fp,
            obs_group_key=og,
            master_dark_path=md_path_ok,
            master_dark_by_obs_key=master_dark_by_obs_key,
        )

    def _sat_for_pregate(fp: Path) -> float | None:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        return _saturation_adu_for_cal_diag(hdr, db=db_main, equipment_id=equipment_id)

    if total > 0:
        cal_diag_session = run_cal_diag_pregate(
            files,
            obs_group_key_from_path=_obs_group_key_from_light_path,
            resolve_dark_path=_resolve_dark_for_pregate,
            light_binning_from_path=_light_binning_from_path,
            master_binning=_native_b,
            match_and_crop_pair=_match_and_crop_pair,
            saturation_for_light=_sat_for_pregate,
            ui_error=_pipeline_ui_error,
        )
    stats["cal_diag_aborted_groups"] = len(cal_diag_session.aborted_groups)
    cal_diag_worker_blob = _cal_diag_export_for_workers(cal_diag_session)

    dark_cache: dict[str, Any] = {}

    def _one_sequential(i: int, src: Path, dst: Path) -> None:
        nonlocal stats
        if progress_cb is not None:
            progress_cb(i, total, f"Calibrating {src.name}")
        md_use = md_path_ok
        md_np_use = None
        light_bx = 1
        with fits.open(src, memmap=False) as hdul:
            hdr_l = hdul[0].header
            _ok = observation_group_key_from_metadata(fits_metadata_from_primary_header(hdr_l))
            light_bx, _ = fits_binning_xy_from_header(hdr_l)
            _light_shape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
        if is_obs_group_aborted(cal_diag_session, _ok):
            if dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass
            return
        if master_dark_by_obs_key:
            _alt = master_dark_by_obs_key.get(_ok)
            if _alt is not None and str(_alt).strip() != "":
                _pa = Path(_alt)
                if _pa.is_file():
                    md_use = _pa
        gr = gate_result_for_frame(
            cal_diag_session,
            obs_group_key=_ok,
            dark_path=md_use,
            light_binning=light_bx,
        )
        if md_use is not None and md_use.is_file():
            if (
                md_pre is not None
                and md_path_ok is not None
                and md_use.resolve() == md_path_ok.resolve()
                and _native_b is not None
                and _native_b == light_bx
                and (gr is None or convention_to_dark_mode(gr.convention) == "sum")
            ):
                md_np_use = md_pre
            else:
                md_np_use = dark_np_for_cal_diag(
                    cal_diag_session,
                    master_binning=_native_b,
                    dark_path=md_use,
                    light_binning=light_bx,
                    light_shape=_light_shape,
                    light_filename=src.name,
                    gate_result=gr,
                )
        used_dark, used_flat, qc_sum, _flags, perf10_qc = _calibrate_one_light_disk(
            src=src,
            dst=dst,
            master_dark_path=md_use,
            masterflat_by_filter=masterflat_by_filter,
            flat_cache=flat_cache,
            flat_median_scale=flat_median_scale,
            md_data_preload=md_np_use,
            db=db_main,
            qc_pack=qc_pack,
            calibration_master_native_binning=_native_b,
            cal_diag_gate_result=gr,
        )
        if isinstance(perf10_qc, dict) and perf10_qc and not perf10_qc.get("error"):
            perf10_qc_results[str(src.resolve())] = perf10_qc
        _sync_obs_calibration_state_with_retry(
            db_main,
            raw_light_path=src,
            draft_id=draft_id,
            observation_id=observation_id,
            is_calibrated=1 if "D" in _flags else 0,
            calib_type=_calibration_type_from_flags(_flags),
            calib_flags=_flags,
            stats=stats,
        )
        stats["processed"] += 1
        if used_dark:
            stats["used_dark"] += 1
        if used_flat:
            stats["used_flat"] += 1
        if not used_dark and not used_flat:
            stats["copied_only"] += 1
        if qc_sum is not None:
            stats["qc_evaluated"] += 1
            if not bool(qc_sum.get("qc_passed", True)):
                stats["qc_rejected"] += 1

    if _vyvar_calibrate_multiprocessing_enabled() and nw > 1 and total > 1:
        items: list[
            tuple[
                str,
                str,
                str | None,
                dict[str, str | None],
                dict[str, Any] | None,
            ]
        ] = []
        for src in files:
            rel = src.relative_to(lights_root)
            dst = calibrated_root / rel
            _qc_mp = dict(qc_pack) if qc_pack else {}
            if master_dark_by_obs_key:
                _qc_mp["master_dark_by_obs_key"] = {
                    str(k): str(Path(v).resolve()) if v is not None and str(v).strip() != "" else None
                    for k, v in master_dark_by_obs_key.items()
                }
            items.append(
                (
                    str(src.resolve()),
                    str(dst.resolve()),
                    md_init_str,
                    mf_serial,
                    _qc_mp,
                )
            )
        stats["calibrate_workers"] = min(nw, total)
        ctx = multiprocessing.get_context("spawn")
        rows: list[dict[str, Any] | None] = [None] * total
        try:
            with ProcessPoolExecutor(
                max_workers=stats["calibrate_workers"],
                mp_context=ctx,
                initializer=_init_calibrate_batch_worker,
                initargs=(md_init_str, _native_b, cal_diag_worker_blob),
            ) as ex:
                future_map = {
                    ex.submit(_calibrate_batch_process_one, it): idx for idx, it in enumerate(items)
                }
                done = 0
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    rows[idx] = fut.result()
                    done += 1
                    if progress_cb is not None:
                        src_name = Path(items[idx][0]).name
                        progress_cb(done, total, f"Calibrating batch {done}/{total} ({src_name})")
        except Exception as exc:  # noqa: BLE001
            # EXC-0434: T2 -- [CAL-DIAG] MP calibrate batch: same manifest files[] calibration-state update `pass` after wor... (EXCEPT-BULK-2 2026-07-08)
            _tb_pool = traceback.format_exc()
            LOGGER.error("Kalibracia (parallel): pool zlyhal, fallback na sekvencny rezim: %s\n%s", exc, _tb_pool)
            log_exception("CHYBA POOLU KALIBRACIE", exc)
            stats["errors"] = 0
            stats["processed"] = 0
            stats["used_dark"] = 0
            stats["used_flat"] = 0
            stats["copied_only"] = 0
            stats["qc_evaluated"] = 0
            stats["qc_rejected"] = 0
            stats["calibrate_workers"] = 1
            for i, src in enumerate(files, start=1):
                rel = src.relative_to(lights_root)
                dst = calibrated_root / rel
                try:
                    _one_sequential(i, src, dst)
                except Exception as exc2:  # noqa: BLE001
                    _tb2 = traceback.format_exc()
                    LOGGER.error("Kalibracia: subor %s: %s\n%s", src, exc2, _tb2)
                    log_exception(f"CHYBA KALIBRACIE: {src.name}", exc2)
                    stats["errors"] += 1
            _arch = _archive_root_from_lights_root(lights_root)
            if _arch is not None:
                write_cal_diag_json(_arch, cal_diag_session)
            _refresh_manifest_after_cal_qc()
            return stats

        for idx, r in enumerate(rows):
            if r is None or not r.get("ok"):
                stats["errors"] += 1
                if stats["errors"] == 1:
                    _ename = Path(items[idx][0]).name if idx < len(items) else "?"
                    _emsg = (r or {}).get("error") if isinstance(r, dict) else None
                    LOGGER.error(
                        "Kalibracia: worker zlyhal pre %s: %s",
                        _ename,
                        _emsg or r,
                    )
                    if isinstance(r, dict) and r.get("traceback"):
                        log_event(f"CHYBA WORKERA: {_ename}: {r.get('error', '')}")
                        log_event(str(r["traceback"]))
                    elif isinstance(r, dict) and r.get("error"):
                        log_event(f"CHYBA WORKERA: {_ename}: {r.get('error')}")
                    else:
                        log_event(f"CHYBA WORKERA: {_ename}: {_emsg or 'no traceback in result'}")
                continue
            stats["processed"] += 1
            qcs = r.get("qc_summary")
            if qcs is not None:
                stats["qc_evaluated"] += 1
                if not bool(qcs.get("qc_passed", True)):
                    stats["qc_rejected"] += 1
            try:
                with fits.open(Path(items[idx][1]), memmap=False) as hh:
                    h0 = hh[0].header
                    _flags = _hdr_vy_cflag_str(h0)
                    ud = bool(h0.get("VY_DARK", False))
                    uf = bool(h0.get("VY_FLAT", False))
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0435] `_estimate_fwhm_from_image` broad `except` returns hardcoded `3.0` px; QC FWHM reject a...: %s', exc)
                ud = uf = False
                _flags = "P"
            _sync_obs_calibration_state_with_retry(
                db_main,
                raw_light_path=Path(items[idx][0]),
                draft_id=draft_id,
                observation_id=observation_id,
                is_calibrated=1 if "D" in _flags else 0,
                calib_type=_calibration_type_from_flags(_flags),
                calib_flags=_flags,
                stats=stats,
            )
            if ud:
                stats["used_dark"] += 1
            if uf:
                stats["used_flat"] += 1
            if not ud and not uf:
                stats["copied_only"] += 1
            p10 = r.get("perf10_qc") if isinstance(r, dict) else None
            if isinstance(p10, dict) and p10 and not p10.get("error"):
                perf10_qc_results[str(Path(items[idx][0]).resolve())] = p10
        _arch = _archive_root_from_lights_root(lights_root)
        if _arch is not None:
            write_cal_diag_json(_arch, cal_diag_session)
        _refresh_manifest_after_cal_qc()
        return stats

    _seq_tb_logged = False
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(lights_root)
        dst = calibrated_root / rel
        try:
            _one_sequential(i, src, dst)
        except Exception as exc:  # noqa: BLE001
            _tb_seq = traceback.format_exc()
            LOGGER.error("Kalibracia: subor %s: %s\n%s", src, exc, _tb_seq)
            if not _seq_tb_logged:
                log_exception(f"CHYBA KALIBRACIE: {src.name}", exc)
                _seq_tb_logged = True
            else:
                log_event(f"CHYBA KALIBRACIE: {src.name}: {exc!s}")
            stats["errors"] += 1
            continue

    _arch = _archive_root_from_lights_root(lights_root)
    if _arch is not None:
        write_cal_diag_json(_arch, cal_diag_session)

    _refresh_manifest_after_cal_qc()
    return stats


def _safe_proc_name(original_name: str) -> str:
    name = (original_name or "").strip()
    if not name:
        return "proc.fits"
    if name.lower().startswith("proc_"):
        return name
    return "proc_" + name



def _estimate_dao_fwhm_guess(img2: "np.ndarray", std: float) -> float:
    """DAOStarFinder kernel FWHM hint (pixels).

    Returns a star-scale default. Do not derive this from segmentation SourceCatalog
    after CR cleaning was removed -- that path preferred hot pixels / cosmics (~1-2 px)
    and pulled the kernel (and later FWHM) unphysically low.
    """
    _ = (img2, std)
    return 4.5


def _moment_fwhm_elong_peak_at(
    img2: "np.ndarray",
    x0: float,
    y0: float,
    *,
    half: int = 7,
) -> tuple[float | None, float | None, float, float, float]:
    """Moment FWHM/elongation at ``(x0,y0)`` on a background-subtracted image.

    Returns ``(fwhm_px, elongation, peak, flux_sum, concentration)`` where
    ``concentration = peak / flux_sum`` (hot pixels / CRs are near 1).
    """
    import numpy as np

    h, w = img2.shape
    xi = int(round(float(x0)))
    yi = int(round(float(y0)))
    y1 = max(0, yi - half)
    y2 = min(h, yi + half + 1)
    x1 = max(0, xi - half)
    x2 = min(w, xi + half + 1)
    cut = np.asarray(img2[y1:y2, x1:x2], dtype=np.float32)
    if cut.size < 25:
        return None, None, 0.0, 0.0, 1.0
    cut_pos = np.where(cut > 0, cut, 0.0).astype(np.float32)
    flux_sum = float(np.sum(cut_pos))
    peak = float(np.max(cut_pos)) if cut_pos.size else 0.0
    if not math.isfinite(flux_sum) or flux_sum <= 0 or not math.isfinite(peak) or peak <= 0:
        return None, None, peak, flux_sum, 1.0
    conc = float(peak / flux_sum)
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    cx = float(np.sum(xx * cut_pos) / flux_sum)
    cy = float(np.sum(yy * cut_pos) / flux_sum)
    dx = xx - cx
    dy = yy - cy
    mxx = float(np.sum((dx * dx) * cut_pos) / flux_sum)
    myy = float(np.sum((dy * dy) * cut_pos) / flux_sum)
    mxy = float(np.sum((dx * dy) * cut_pos) / flux_sum)
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = tr * tr - 4.0 * det
    if disc < 0:
        return None, None, peak, flux_sum, conc
    l1 = 0.5 * (tr + float(np.sqrt(disc)))
    l2 = 0.5 * (tr - float(np.sqrt(disc)))
    if l1 <= 0 or l2 <= 0:
        return None, None, peak, flux_sum, conc
    sig1 = float(np.sqrt(l1))
    sig2 = float(np.sqrt(l2))
    fwhm = 2.355 * 0.5 * (sig1 + sig2)
    elong = (sig1 / sig2) if sig2 > 0 else float("nan")
    if not (math.isfinite(fwhm) and 0.2 < fwhm < 50):
        return None, None, peak, flux_sum, conc
    if not (math.isfinite(elong) and 0.5 < elong < 20):
        return float(fwhm), None, peak, flux_sum, conc
    return float(fwhm), float(elong), peak, flux_sum, conc


def _robust_frame_fwhm_median(
    data: "np.ndarray",
    *,
    max_sources: int = 120,
    min_keep: int = 12,
    use_center_crop: bool = True,
    sat_adu: float | None = None,
    isol_px: float = 10.0,
    max_concentration: float = 0.22,
    elong_lo: float = 0.75,
    elong_hi: float = 1.55,
) -> dict[str, Any]:
    """Per-frame FWHM = median moment-FWHM over many star-like detections.

    Membership selection only (no sigma-clip of the FWHM sample; Milan 2026-08-12):
    - DAO detections on a background-subtracted (optional center-crop) image
    - Unsaturated (peak < ``sat_adu``)
    - Extended / anti-CR: ``peak/sum <= max_concentration`` (rejects hot pixels / cosmics)
    - Roundish: moment elongation in ``[elong_lo, elong_hi]``
    - Isolated: no brighter accepted neighbor within ``isol_px``
    - Aggregate: median of remaining FWHMs (requires ``min_keep``)

    This is measurement aggregation across stars; it does not alter science pixels.
    """
    import numpy as np
    from photutils.detection import DAOStarFinder

    out: dict[str, Any] = {
        "fwhm_px": None,
        "elongation": None,
        "n_sources": 0,
        "n_stars_detected": 0,
        "n_fwhm_sample": 0,
    }
    img = np.asarray(data, dtype=np.float32)
    if img.ndim != 2:
        return out
    work = _qc_center_crop_for_stars(img) if use_center_crop else img
    finite = np.isfinite(work)
    if not np.any(finite):
        return out
    _, med, std = plain_mean_med_std(work[finite])
    std_f = float(std)
    if not math.isfinite(std_f) or std_f <= 0:
        return out
    img2 = np.asarray(work - float(med), dtype=np.float32)
    img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
    # Polarity: only flip when the negative tail clearly dominates (stars as dips).
    # Do NOT flip on a near-zero median after background subtraction -- float noise
    # around 0 would invert a normal star field and DAO finds almost nothing.
    if math.isfinite(std_f) and std_f > 0:
        pos = float(np.count_nonzero(img2 > (4.0 * std_f)))
        neg = float(np.count_nonzero(img2 < (-4.0 * std_f)))
        if neg > (pos * 2.0) and neg > 100:
            img2 = -img2

    # DAO kernel must not be tuned to CR-scale (~1-2 px) blobs.
    fwhm_guess = float(max(3.0, min(12.0, _estimate_dao_fwhm_guess(img2, std_f))))
    sat = float(sat_adu) if sat_adu is not None and math.isfinite(float(sat_adu)) else 50000.0
    sat = max(1000.0, sat)

    tbl = None
    for thr_k in (5.0, 3.5, 2.5):
        daofind = DAOStarFinder(
            fwhm=float(fwhm_guess),
            threshold=float(thr_k) * std_f,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = daofind(img2)
        if tbl is not None and len(tbl) >= max(8, min_keep):
            break
    if tbl is None or len(tbl) == 0:
        return out
    out["n_stars_detected"] = int(len(tbl))
    tbl.sort("flux")
    tbl = tbl[::-1]

    n_scan = int(min(len(tbl), max(max_sources * 3, max_sources)))
    cand: list[tuple[float, float, float, float, float]] = []
    for i in range(n_scan):
        x0 = float(tbl["x_centroid"][i])
        y0 = float(tbl["y_centroid"][i])
        flux = float(tbl["flux"][i])
        fwhm, elong, peak, _flux_sum, conc = _moment_fwhm_elong_peak_at(img2, x0, y0, half=7)
        if fwhm is None or elong is None:
            continue
        if not math.isfinite(peak) or peak >= sat:
            continue
        if not math.isfinite(conc) or conc > float(max_concentration):
            continue
        if not (float(elong_lo) <= float(elong) <= float(elong_hi)):
            continue
        cand.append((flux, x0, y0, float(fwhm), float(elong)))

    accepted_fwhm: list[float] = []
    accepted_elong: list[float] = []
    accepted_xy: list[tuple[float, float]] = []
    isol2 = float(isol_px) * float(isol_px)
    for _flux, x0, y0, fwhm, elong in cand:
        if len(accepted_fwhm) >= int(max_sources):
            break
        ok_iso = True
        for ax, ay in accepted_xy:
            if (x0 - ax) * (x0 - ax) + (y0 - ay) * (y0 - ay) < isol2:
                ok_iso = False
                break
        if not ok_iso:
            continue
        accepted_fwhm.append(fwhm)
        accepted_elong.append(elong)
        accepted_xy.append((x0, y0))

    out["n_fwhm_sample"] = int(len(accepted_fwhm))
    out["n_sources"] = int(len(accepted_fwhm))
    if len(accepted_fwhm) < int(min_keep):
        return out
    arr_f = np.asarray(accepted_fwhm, dtype=np.float64)
    arr_e = np.asarray(accepted_elong, dtype=np.float64)
    out["fwhm_px"] = float(np.median(arr_f))
    out["elongation"] = float(np.median(arr_e))
    return out


def _qc_fwhm_elongation(
    data: "np.ndarray",
    *,
    max_sources: int = 200,
) -> dict[str, Any]:
    """Estimate average FWHM and elongation (best-effort).

    Robust path (2026-08-12): median moment-FWHM over many star-like DAO detections
    (bright, unsaturated, isolated, extended). Segmentation SourceCatalog is NOT used
    for FWHM -- after CR cleaning was removed it preferentially measured hot pixels /
    cosmics (~1-2 px) and poisoned ``VY_FWHM`` / aperture sizing.

    Returns ``n_stars_detected``: approximate total star-like detections on the frame.
    ``n_sources``: subset used for the FWHM/elongation median.
    """
    _ = max_sources
    rob = _robust_frame_fwhm_median(
        data,
        max_sources=120,
        min_keep=12,
        use_center_crop=True,
    )
    if rob.get("fwhm_px") is not None:
        return {
            "fwhm_px": rob.get("fwhm_px"),
            "elongation": rob.get("elongation"),
            "n_sources": int(rob.get("n_sources") or 0),
            "n_stars_detected": int(rob.get("n_stars_detected") or 0),
        }
    rob2 = _robust_frame_fwhm_median(
        data,
        max_sources=120,
        min_keep=5,
        use_center_crop=True,
    )
    return {
        "fwhm_px": rob2.get("fwhm_px"),
        "elongation": rob2.get("elongation"),
        "n_sources": int(rob2.get("n_sources") or 0),
        "n_stars_detected": int(rob2.get("n_stars_detected") or 0),
    }

def _vyvar_parallel_worker_count(app_config: AppConfig | None = None) -> int:
    """Jednotny pocet workerov pre QC, preprocess, combined, alignment seed, per-frame CSV seed, calibrate MP.

    Prednost: ``VYVAR_PARALLEL_WORKERS`` -> (ak su obe) minimum z legacy ``VYVAR_QC_PREPROCESS_WORKERS`` a
    ``VYVAR_PER_FRAME_CSV_WORKERS`` -> ``app_config.qc_preprocess_workers`` (uz zjednotene) -> host auto z ``config``.
    """
    u = os.environ.get("VYVAR_PARALLEL_WORKERS")
    if u is not None and str(u).strip() != "":
        try:
            return max(1, min(32, int(str(u).strip())))
        except ValueError:
            pass
    legacy: list[int] = []
    for key in ("VYVAR_QC_PREPROCESS_WORKERS", "VYVAR_PER_FRAME_CSV_WORKERS"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            try:
                legacy.append(max(1, min(32, int(str(raw).strip()))))
            except ValueError:
                pass
    if legacy:
        return int(min(legacy))
    if app_config is not None:
        try:
            return max(1, min(32, int(app_config.qc_preprocess_workers)))
        except (TypeError, ValueError):
            pass
    from config import load_config_json, recommended_vyvar_parallel_workers, resolve_data_root

    _install = Path(__file__).resolve().parent.parent
    data = load_config_json(resolve_data_root(_install))
    try:
        res_gb = float(data.get("per_frame_mp_reserve_ram_gb", 1.5))
        if not math.isfinite(res_gb) or res_gb < 0:
            res_gb = 1.5
    except (TypeError, ValueError):
        res_gb = 1.5
    return int(recommended_vyvar_parallel_workers(reserve_ram_gb=res_gb))


def _vyvar_qc_preprocess_workers() -> int:
    """Parallel workers for analyze / preprocess / combined (see :func:`_vyvar_parallel_worker_count`)."""
    return _vyvar_parallel_worker_count(None)


def _estimate_catalog_frame_hw(
    work_ram: Sequence[tuple[str, Any, Any]] | None,
    files: list[Path],
) -> tuple[int, int]:
    """Rough (ny, nx) for RAM cap heuristics (per-frame catalog export)."""
    import numpy as np

    if work_ram:
        d = np.asarray(work_ram[0][2])
        if d.ndim == 2:
            return int(d.shape[0]), int(d.shape[1])
    for fp in files[:1]:
        try:
            with fits.open(fp, memmap=False) as h:
                nh = h[0].header
                ny = int(nh.get("NAXIS2", 0) or 0)
                nx = int(nh.get("NAXIS1", 0) or 0)
                if ny > 0 and nx > 0:
                    return ny, nx
        except Exception:  # noqa: BLE001
            continue
    return 2048, 2048


def _vyvar_cap_mp_workers_for_catalog(
    n_workers: int,
    frame_hw: tuple[int, int],
    *,
    reserve_gb: float,
) -> int:
    """Cap process pool size using available RAM (rough float32 frame footprint per worker)."""
    h, w = frame_hw
    if h <= 0 or w <= 0:
        h, w = 2048, 2048
    per_worker = max(int(h * w * 4 * 3), 1)
    try:
        import psutil
    except ImportError:
        return max(1, n_workers)
    reserve = int(max(0.0, float(reserve_gb)) * (1024**3))
    avail = int(psutil.virtual_memory().available) - reserve
    if avail <= 0:
        return 1
    mx = max(1, avail // per_worker)
    return max(1, min(int(n_workers), mx))


def _vyvar_per_frame_csv_workers(app_config: AppConfig | None = None) -> int:
    """Process workers for plate-solve step 3 per-frame catalog (DAO + match + CSV).

    Rovnaky zaklad ako QC/alignment (see :func:`_vyvar_parallel_worker_count`); dalsi strop podla skutocnej velkosti
    snimku riesi ``_vyvar_cap_mp_workers_for_catalog``.
    """
    return _vyvar_parallel_worker_count(app_config)


def _analyze_calibrated_qc_one(src: Path) -> dict[str, Any]:
    import numpy as np

    src = Path(src)
    try:
        with fits.open(src, memmap=False) as hdul:
            hdr = hdul[0].header
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        qc = _qc_fwhm_elongation(data)
        finite = np.isfinite(data)
        arr = data[finite]
        return {
            "src": str(src),
            "filter": _safe_filter_token(str(hdr.get("FILTER") or hdr.get("FILT") or "NoFilter")),
            "fwhm_px": qc.get("fwhm_px"),
            "elongation": qc.get("elongation"),
            "n_sources": qc.get("n_sources"),
            "n_stars_detected": qc.get("n_stars_detected"),
            "bg_median": float(np.nanmedian(arr)) if arr.size else None,
            "p50": float(np.nanpercentile(arr, 50)) if arr.size else None,
            "p99": float(np.nanpercentile(arr, 99)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
        }
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0439] [SILENT-DROP] analyze calibrated QC one-frame helper failed: %s', exc)
        return {"src": str(src), "status": f"error: {exc}"}


def _fit_subtract_preprocess_sky_surface(
    data: Any,
    *,
    order: int,
    fwhm_px: float | None = None,
    subsample_step: int = 4,
    calm_adu: float = 100.0,
) -> tuple[Any, dict[str, Any]]:
    """Fit and subtract a 2D polynomial sky correction (shared calibrated->processed preprocess).

  Star-masked fit on calm background pixels (|cal-median| < ``calm_adu``).
  Fits order-N to ``work - bg_median`` on the fit set, then subtracts the fitted surface from
  the frame (429-class gradient removal; full surface including constant term).
  No sigma-clip on fit samples (zero-clipping policy 2026-08-12).
    """
    import numpy as np
    from photutils.detection import DAOStarFinder

    arr = np.asarray(data, dtype=np.float32)
    order_i = int(order)
    if order_i <= 0:
        return arr.copy(), {"sky_surface_order": 0, "sky_surface_applied": False}

    order_i = min(2, max(1, order_i))
    h, w = arr.shape
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    work = np.where(finite, arr, fill)

    mask = np.ones((h, w), dtype=bool)
    margin = 40
    if h > 2 * margin and w > 2 * margin:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

    fwhm_eff = max(
        1.2,
        float(fwhm_px) if fwhm_px is not None and math.isfinite(float(fwhm_px)) else 2.5,
    )
    _, med, std = plain_mean_med_std(work)
    data0 = np.nan_to_num((work - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    thr = max(3.0 * float(std), 1e-6)
    finder = DAOStarFinder(
        fwhm=float(fwhm_eff),
        threshold=float(thr),
        n_brightest=5000,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = finder(data0)
    stamp_r = int(max(4, round(3.5 * fwhm_eff)))
    if tbl is not None and len(tbl) > 0:
        r2 = stamp_r * stamp_r
        for row in tbl:
            cy = int(round(float(row["y_centroid"])))
            cx = int(round(float(row["x_centroid"])))
            if not (0 <= cy < h and 0 <= cx < w):
                continue
            y0, y1 = max(0, cy - stamp_r), min(h, cy + stamp_r + 1)
            x0, x1 = max(0, cx - stamp_r), min(w, cx + stamp_r + 1)
            yy_l, xx_l = np.ogrid[y0:y1, x0:x1]
            local_excl = (yy_l - cy) ** 2 + (xx_l - cx) ** 2 <= r2
            mask[y0:y1, x0:x1] &= ~local_excl

    bg_median, _, _ = plain_mean_med_std(work, mask=~mask)
    calm_thr = max(5.0, float(calm_adu))
    fit_mask = mask & (np.abs(work - float(bg_median)) < calm_thr)

    step = max(1, int(subsample_step))
    yy_s, xx_s = np.mgrid[0:h:step, 0:w:step]
    z_s = (work[::step, ::step] - float(bg_median)).astype(np.float64)
    m_s = fit_mask[::step, ::step]
    use_mask = m_s & np.isfinite(z_s)
    min_coef = (order_i + 1) * (order_i + 2) // 2
    if int(np.count_nonzero(use_mask)) < min_coef + 10:
        return arr.copy(), {
            "sky_surface_order": order_i,
            "sky_surface_applied": False,
            "sky_surface_skip_reason": "insufficient_fit_pixels",
        }

    z_samples = z_s[use_mask]
    x_fit = xx_s[use_mask].astype(np.float64)
    y_fit = yy_s[use_mask].astype(np.float64)
    z_fit = z_samples
    if z_fit.size < min_coef + 5:
        return arr.copy(), {
            "sky_surface_order": order_i,
            "sky_surface_applied": False,
            "sky_surface_skip_reason": "insufficient_fit_pixels",
        }

    cols: list[np.ndarray] = []
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols.append((x_fit**i) * (y_fit**j))
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), z_fit, rcond=None)

    yy_f, xx_f = np.mgrid[0:h, 0:w]
    cols_f: list[np.ndarray] = []
    x_flat = xx_f.ravel().astype(np.float64)
    y_flat = yy_f.ravel().astype(np.float64)
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols_f.append((x_flat**i) * (y_flat**j))
    surf = (np.column_stack(cols_f) @ coef).reshape(h, w).astype(np.float32)

    out = (work - surf).astype(np.float32)
    out = np.where(finite, out, np.nan).astype(np.float32)
    # INV-FLAT-01: residual large-scale flatness after full-surface subtract (WARN band).
    _flat_p99 = float("nan")
    try:
        from invariants_runtime import residual_large_scale_p99_adu  # noqa: PLC0415

        _flat_p99 = float(residual_large_scale_p99_adu(out))
    except Exception:  # noqa: BLE001
        _flat_p99 = float("nan")
    return out, {
        "sky_surface_order": order_i,
        "sky_surface_applied": True,
        "sky_surface_p2p_adu": float(np.nanmax(surf) - np.nanmin(surf)),
        "sky_surface_median_adu": float(np.nanmedian(surf)),
        "sky_surface_n_fit_pixels": int(z_fit.size),
        "sky_surface_fwhm_px": float(fwhm_eff),
        "sky_surface_bg_median_adu": float(bg_median),
        "sky_surface_calm_adu": float(calm_thr),
        "residual_flatness_p99_adu": _flat_p99,
    }


def run_osc_channel_extraction_for_archive(
    *,
    calibrated_lights_root: Path,
    db: VyvarDatabase,
    equipment_id: int,
    app_config: AppConfig | None = None,
    progress_cb: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Extract OSC CFA mosaics into four channel obs-groups; QC+sky-surface per channel."""
    from osc_extract import (
        OSC_CHANNELS,
        channel_obs_group_folder,
        extract_one_light_to_channels,
        is_channel_obs_group_folder,
        is_osc_bayermask,
        iter_mosaic_light_fits,
    )

    cfg = app_config or AppConfig()
    bayermask = db.get_equipment_bayermask(int(equipment_id))
    if not is_osc_bayermask(bayermask):
        return {"skipped": True, "reason": "mono equipment"}

    gain_raw, rn_raw = db.get_equipment_cosmic_params(int(equipment_id))
    gain_e = float(gain_raw) if gain_raw is not None and gain_raw > 0 else 1.0
    rn_e = float(rn_raw) if rn_raw is not None and rn_raw >= 0 else 5.0
    osc_bin = int(cfg.osc_channel_binning)
    root = Path(calibrated_lights_root)
    out: dict[str, Any] = {
        "skipped": False,
        "bayermask": bayermask,
        "osc_bin": osc_bin,
        "groups": [],
        "n_extracted": 0,
        "n_mosaic_removed": 0,
    }

    group_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and not is_channel_obs_group_folder(d.name)],
        key=lambda p: p.name.casefold(),
    )
    total_files = sum(len(iter_mosaic_light_fits(g)) for g in group_dirs)
    done = 0
    for group_dir in group_dirs:
        mosaic_files = iter_mosaic_light_fits(group_dir)
        if not mosaic_files:
            continue
        base_name = group_dir.name
        out_dirs = {
            ch: root / channel_obs_group_folder(base_name, ch) for ch in OSC_CHANNELS
        }
        grp_stat: dict[str, Any] = {"base": base_name, "channels": list(OSC_CHANNELS), "n": 0}
        for fp in mosaic_files:
            done += 1
            if progress_cb is not None:
                progress_cb(done, max(total_files, 1), f"OSC extract {fp.name}")
            extract_one_light_to_channels(
                fp,
                out_dirs=out_dirs,
                bayermask=str(bayermask),
                osc_bin=osc_bin,
                gain_e_per_adu=gain_e,
                read_noise_e=rn_e,
            )
            try:
                fp.unlink()
                out["n_mosaic_removed"] += 1
            except OSError as exc:
                LOGGER.warning("[OSC] failed to remove mosaic %s: %s", fp, exc)
            grp_stat["n"] += 1
            out["n_extracted"] += 1
        for ch in OSC_CHANNELS:
            _qc_enrich_calibrated_in_place(
                out_dirs[ch],
                app_config=cfg,
                progress_cb=progress_cb,
            )
        try:
            from osc_align import merge_osc_qc_metrics_at_lights_root, replicate_qc_verdict_from_one_rggb

            replicate_qc_verdict_from_one_rggb(lights_root=root, base_name=base_name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[OSC] QC verdict replication failed for %s: %s", base_name, exc)
        try:
            if not any(group_dir.iterdir()):
                group_dir.rmdir()
        except OSError:
            pass
        out["groups"].append(grp_stat)
    try:
        from osc_align import merge_osc_qc_metrics_at_lights_root

        merged_qc = merge_osc_qc_metrics_at_lights_root(root)
        if merged_qc is not None:
            out["qc_metrics_csv"] = str(merged_qc)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[OSC] qc_metrics merge at lights root failed: %s", exc)
    log_event(
        f"OSC extraction: {out['n_extracted']} mosaic(s) -> {len(out['groups'])} base group(s); "
        f"channels={','.join(OSC_CHANNELS)} bin={osc_bin}"
    )
    return out


class SkySurfaceOrderConflictError(RuntimeError):
    """``VY_SKYSF`` present but ``VYSKYORD`` differs from the requested preprocess order."""


from cal_stage import _header_has_vy_skysf  # noqa: E402,F401  # re-export; survivor in cal_stage.py


def _header_vyskyord(hdr: fits.Header) -> int | None:
    if "VYSKYORD" not in hdr:
        return None
    try:
        return int(hdr["VYSKYORD"])
    except (TypeError, ValueError):
        return None


def _decide_preprocess_sky_action(
    hdr: fits.Header,
    *,
    sky_order: int,
    force_reapply: bool,
) -> str:
    """Return ``apply`` or ``skip``. Raises :class:`SkySurfaceOrderConflictError` on order mismatch."""
    if sky_order <= 0:
        return "skip"
    if force_reapply:
        return "apply"
    if not _header_has_vy_skysf(hdr):
        return "apply"
    existing = _header_vyskyord(hdr)
    if existing is None:
        return "apply"
    if existing == sky_order:
        return "skip"
    raise SkySurfaceOrderConflictError(
        f"VYSKYORD={existing} but preprocess requests order={sky_order}; "
        "recalibration from raw is required to change sky-surface order."
    )


def _qc_enrich_one_frame(
    fp_str: str,
    *,
    sky_order: int,
    force_reapply: bool,
    prefilter_status: str | None,
    target_ra: float | None,
    target_dec: float | None,
    inject_pointing_only_if_missing: bool,
) -> dict[str, Any]:
    """Process one calibrated light frame in-place (picklable worker for parallel QC)."""
    import numpy as np

    from cal_stage import (  # noqa: PLC0415
        compute_skysf_apply_stage,
        stamp_cal_stage_headers,
        verify_fits_datasum,
    )

    fp = Path(fp_str)
    try:
        with fits.open(fp, memmap=False) as hdul:
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            hdr = hdul[0].header.copy()

        sky_stats: dict[str, Any] = {}
        cal_stage_token: str | None = None
        cal_datasum: str | None = None
        cal_pstbg: float | None = None
        skypass: int | None = None
        is_mosaic = _valid_bayerpat_from_header(hdr) is not None and not hdr.get("VY_CHANNEL")
        if sky_order > 0 and not is_mosaic:
            action = _decide_preprocess_sky_action(
                hdr, sky_order=sky_order, force_reapply=force_reapply
            )
            if action == "apply":
                _force_flag = bool(force_reapply and _header_has_vy_skysf(hdr))
                if _force_flag:
                    LOGGER.warning(
                        "preprocess_sky_surface_force_reapply: re-subtracting sky surface on %s",
                        fp.name,
                    )
                data, sky_stats = _fit_subtract_preprocess_sky_surface(data, order=sky_order)
                if _force_flag:
                    sky_stats["sky_surface_force_reapply"] = True
                cal_stage_token, skypass = compute_skysf_apply_stage(
                    hdr,
                    sky_order=sky_order,
                    force_reapply=_force_flag,
                )
                cal_pstbg = float(np.nanmedian(data)) if data.size else None
            else:
                sky_stats = {
                    "sky_surface_applied": False,
                    "sky_surface_skipped": True,
                    "sky_surface_order": sky_order,
                }
                LOGGER.info(
                    "Sky-surface subtract skipped on %s (VY_SKYSF present, VYSKYORD=%s)",
                    fp.name,
                    sky_order,
                )

        qc = _qc_fwhm_elongation(data)
        fwhm = float(qc.get("fwhm_px")) if qc.get("fwhm_px") is not None else float("nan")
        elong = float(qc.get("elongation")) if qc.get("elongation") is not None else float("nan")
        n_stars = int(qc.get("n_stars_detected") or 0)
        status = prefilter_status if prefilter_status else "ok"

        with fits.open(fp, mode="update") as hdul:
            hdul[0].data = _as_fits_float32_image(data)
            hdr = hdul[0].header
            if sky_stats.get("sky_surface_applied"):
                hdr["VY_SKYSF"] = (True, "Sky-surface subtract applied")
                hdr["VYSKYORD"] = (
                    int(sky_stats.get("sky_surface_order") or sky_order),
                    "Preprocess sky-surface polynomial order",
                )
                p2p = sky_stats.get("sky_surface_p2p_adu")
                if p2p is not None and math.isfinite(float(p2p)):
                    hdr["VYSKYP2P"] = (
                        round(float(p2p), 4),
                        "Sky surface peak-to-peak ADU",
                    )
                if cal_stage_token:
                    cal_datasum = stamp_cal_stage_headers(
                        hdr,
                        data,
                        stage=cal_stage_token,
                        pstbg=cal_pstbg,
                        skypass=skypass,
                    )
                    if not verify_fits_datasum(data, cal_datasum):
                        raise RuntimeError(
                            f"INV-CAL-02: VY_CALDATASUM self-check failed on {fp.name}"
                        )
            if math.isfinite(fwhm):
                hdr["VY_FWHM"] = (round(fwhm, 4), "Estimated FWHM [pix]")
            if math.isfinite(elong):
                hdr["VY_ELONG"] = (round(elong, 4), "Estimated elongation (a/b)")
            hdr["VY_NSTAR"] = (n_stars, "Approx. star detections (QC)")
            hdr["VY_QC"] = (status, "QC status")
            # Drop legacy CR provenance if present (L.A.Cosmic removed 2026-08-12).
            if "VY_COSM" in hdr:
                del hdr["VY_COSM"]
            if "VY_COSMNPX" in hdr:
                del hdr["VY_COSMNPX"]
            hdr["VYVARPR"] = (True, "VYVAR pre-processed output")
            if target_ra is not None and target_dec is not None:
                ira = float(target_ra)
                idec = float(target_dec)
                if math.isfinite(ira) and math.isfinite(idec):
                    ex_ra, ex_dec, _ = _pointing_hint_from_header(hdr)
                    do_inject = (not bool(inject_pointing_only_if_missing)) or (
                        ex_ra is None or ex_dec is None
                    )
                    if do_inject:
                        hdr["VYTARGRA"] = (ira, "VYVAR plate-solve hint RA [deg] ICRS")
                        hdr["VYTARGDE"] = (idec, "VYVAR plate-solve hint Dec [deg] ICRS")
                        hdr.add_history("VYVAR: VYTARGRA/VYTARGDE for plate solving (QC in-place)")
            hdul.flush()

        out_row: dict[str, Any] = {
            "src": str(fp),
            "dst": str(fp),
            "status": status,
            "fwhm_px": fwhm,
            "elongation": elong,
            "n_sources": qc.get("n_sources"),
            "n_stars_detected": n_stars,
            "bg_median": float(np.nanmedian(data)) if data.size else None,
            **sky_stats,
        }
        if cal_stage_token and cal_datasum:
            out_row["cal_stage"] = cal_stage_token
            out_row["cal_datasum"] = cal_datasum
            if cal_pstbg is not None:
                out_row["cal_pstbg"] = cal_pstbg
        return out_row
    except SkySurfaceOrderConflictError:
        raise
    except Exception as exc:  # noqa: BLE001
        return {
            "src": str(fp),
            "dst": str(fp),
            "status": f"error: {exc}",
            "fwhm_px": float("nan"),
            "elongation": float("nan"),
            "n_sources": None,
            "n_stars_detected": 0,
            "bg_median": None,
        }


def _infer_raw_light_path_for_calibrated(cal_fp: Path) -> Path | None:
    """Map ``calibrated/lights/...`` frame to matching ``Raw/lights/...`` if present."""
    cal_fp = Path(cal_fp).resolve()
    marker = f"{os.sep}calibrated{os.sep}lights{os.sep}"
    s = str(cal_fp)
    idx = s.find(marker)
    if idx < 0:
        return None
    draft_root = Path(s[:idx])
    rel = Path(s[idx + len(marker) :])
    raw = draft_root / "Raw" / "lights" / rel
    return raw if raw.is_file() else None


def _sync_manifest_cal_stage_from_qc_row(
    row: dict[str, Any],
    *,
    db: VyvarDatabase | None,
    draft_id: int | None,
) -> None:
    if db is None or draft_id is None:
        return
    stage = row.get("cal_stage")
    datasum = row.get("cal_datasum")
    if not stage or not datasum:
        return
    raw = _infer_raw_light_path_for_calibrated(Path(str(row.get("src") or row.get("dst") or "")))
    if raw is None:
        return
    try:
        db.update_obs_file_cal_stage_by_raw_light_path(
            raw,
            draft_id=int(draft_id),
            cal_stage=str(stage),
            cal_datasum=str(datasum),
            cal_pstbg=row.get("cal_pstbg"),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("manifest cal_stage sync failed after preprocess: %s", exc)


def _qc_enrich_calibrated_in_place(
    calibrated_root: Path,
    *,
    app_config: AppConfig | None = None,
    fwhm_reject_limit: float | None = None,
    elong_reject_limit: float | None = None,
    target_ra: float | None = None,
    target_dec: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
    prefilter_rejected: Mapping[Path | str, str] | None = None,
    progress_cb: Callable[..., None] | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> dict[str, Any]:
    """Lightweight QC pass that writes VY_* headers directly onto calibrated FITS files.

    In-place preprocess QC for ``preprocess_calibrated_to_processed`` (skip-only; no ``processed/`` copy tree).
    Does NOT copy files. Does NOT do temporal sigma clipping.
    Writes: VY_FWHM, VY_ELONG, VY_QC, VY_NSTAR, VYVARPR (and optional VYTARG*) to each FITS.

    Skip-mode frame **selection authority** is the DB DAO-FWHM prefilter passed via
    ``prefilter_rejected`` (status ``rejected_prefilter_*``). Segmentation FWHM /
    elongation are measured and recorded as diagnostics only; they do not drive reject
    statuses here. Alignment gating reads ``qc_metrics.csv`` (QC-01).

    Returns a dict with ``results`` rows compatible with the preprocess QC table.
    """
    import numpy as np

    _ = fwhm_reject_limit, elong_reject_limit  # skip-mode: diagnostics only
    cfg = app_config or AppConfig()
    sky_order = int(cfg.preprocess_sky_surface_order)
    force_reapply = bool(getattr(cfg, "preprocess_sky_surface_force_reapply", False))

    calibrated_root = Path(calibrated_root)
    _prefilter: dict[str, str] = {}
    if prefilter_rejected:
        for fp, reason in prefilter_rejected.items():
            _prefilter[norm_fits_path_key(fp)] = str(reason)

    if prefilter_rejected is not None:
        fits_paths = _iter_light_fits(calibrated_root)
    else:
        fits_paths = _filter_light_paths_maybe(_iter_light_fits(calibrated_root), only_paths)
    results: list[dict[str, Any]] = []
    total = len(fits_paths)
    n_workers = _vyvar_qc_preprocess_workers() if total > 1 else 1
    if n_workers > 1:
        LOGGER.info(
            "QC in-place: parallel_workers=%s (%s frames; qc_preprocess_workers)",
            n_workers,
            total,
        )

    if n_workers > 1 and total > 1:
        with _vyvar_parallel_pool(n_workers) as ex:
            futs = {}
            for fp in fits_paths:
                futs[ex.submit(
                    _qc_enrich_one_frame,
                    str(fp),
                    sky_order=sky_order,
                    force_reapply=force_reapply,
                    prefilter_status=_prefilter.get(norm_fits_path_key(fp)),
                    target_ra=target_ra,
                    target_dec=target_dec,
                    inject_pointing_only_if_missing=inject_pointing_only_if_missing,
                )] = fp
            done = 0
            for fut in as_completed(futs):
                fp = futs[fut]
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"QC in-place {fp.name}")
                try:
                    row = fut.result()
                except SkySurfaceOrderConflictError as exc:
                    raise RuntimeError(f"QC in-place failed for {fp.name}: {exc}") from exc
                results.append(row)
                _sync_manifest_cal_stage_from_qc_row(row, db=db, draft_id=draft_id)
                if str(row.get("status", "")).startswith("error"):
                    log_event(f"QC in-place failed for {fp.name}: {row.get('status')}")
                else:
                    log_event(
                        f"QC in-place {fp.name}: {row.get('status')} "
                        f"FWHM={float(row.get('fwhm_px', float('nan'))):.2f} "
                        f"elong={float(row.get('elongation', float('nan'))):.2f}"
                    )
        results.sort(key=lambda r: str(r.get("src", "")))
    else:
        for i, fp in enumerate(fits_paths, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"QC in-place {fp.name}")
            row = _qc_enrich_one_frame(
                str(fp),
                sky_order=sky_order,
                force_reapply=force_reapply,
                prefilter_status=_prefilter.get(norm_fits_path_key(fp)),
                target_ra=target_ra,
                target_dec=target_dec,
                inject_pointing_only_if_missing=inject_pointing_only_if_missing,
            )
            results.append(row)
            _sync_manifest_cal_stage_from_qc_row(row, db=db, draft_id=draft_id)
            if str(row.get("status", "")).startswith("error"):
                log_event(f"QC in-place failed for {fp.name}: {row.get('status')}")
            else:
                log_event(
                    f"QC in-place {fp.name}: {row.get('status')} "
                    f"FWHM={float(row.get('fwhm_px', float('nan'))):.2f} "
                    f"elong={float(row.get('elongation', float('nan'))):.2f}"
                )

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_rejected = sum(1 for r in results if str(r.get("status", "")).startswith("rejected"))
    sky_surface_skip_count = sum(1 for r in results if r.get("sky_surface_skipped"))
    sky_surface_force_reapply = force_reapply and any(
        r.get("sky_surface_force_reapply") for r in results
    )
    log_event(
        f"QC in-place: {n_ok} ok, {n_rejected} rejected, "
        f"{len(results) - n_ok - n_rejected} errors"
    )
    if sky_surface_skip_count:
        log_event(
            f"QC in-place: sky-surface subtract skipped on {sky_surface_skip_count} frame(s) "
            "(VY_SKYSF guard)"
        )
    if sky_surface_force_reapply:
        log_event(
            "QC in-place: preprocess_sky_surface_force_reapply=True (sky-surface guard bypassed)"
        )

    try:
        from invariants_runtime import check_preprocess_large_small_ratio  # noqa: PLC0415
        from invariants_runtime import inv_check  # noqa: PLC0415

        _prep_inv_meta: dict[str, Any] = {"invariants": []}
        _sample_by_group: dict[str, Path] = {}
        for _r in results:
            if str(_r.get("status") or "") != "ok":
                continue
            _src = Path(str(_r.get("src") or ""))
            if not _src.is_file():
                continue
            _grp = str(_src.parent)
            if _grp not in _sample_by_group:
                _sample_by_group[_grp] = _src
        for _grp, _fp in _sample_by_group.items():
            with fits.open(_fp, memmap=False) as _hdul:
                _frame = np.asarray(_hdul[0].data, dtype=np.float32)
            _ok_prep, _det_prep, _ratio_prep = check_preprocess_large_small_ratio(_frame)
            inv_check(
                _prep_inv_meta,
                "INV-PREP-01",
                _ok_prep,
                policy="WARN",
                detail=f"{Path(_grp).name}: {_det_prep}",
            )
            if math.isfinite(_ratio_prep):
                _prep_guard_msg = (
                    f"INV-PREP-01 Preprocess gradient guard ({Path(_grp).name}): {_det_prep}"
                )
                LOGGER.info(_prep_guard_msg)
                log_event(_prep_guard_msg)
    except Exception as _prep_inv_exc:  # noqa: BLE001
        LOGGER.debug("[INV-PREP-01] skipped: %s", _prep_inv_exc)

    _qc_df = pd.DataFrame(results)
    _qc_csv = calibrated_root / "qc_metrics.csv"
    try:
        _qc_df.to_csv(_qc_csv, index=False)
        log_event(f"qc_metrics.csv written to {_qc_csv}")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PIPELINE] qc_metrics.csv write failed: %s", exc)

    return {
        "n_processed": len(results),
        "n_ok": n_ok,
        "n_rejected": n_rejected,
        "sky_surface_skip_count": sky_surface_skip_count,
        "sky_surface_force_reapply": sky_surface_force_reapply,
        "results": results,
        "qc_root": str(calibrated_root),
        "qc_csv": str(_qc_csv),
    }


def qc_enrich_calibrated_lights_in_place(
    *,
    calibrated_root: Path,
    reject_fwhm_px: float | None = None,
    reject_elongation: float | None = None,
    progress_cb: Callable[..., None] | None = None,
    inject_pointing_ra_deg: float | None = None,
    inject_pointing_dec_deg: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
    prefilter_rejected: Mapping[Path | str, str] | None = None,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
    app_config: AppConfig | None = None,
) -> pd.DataFrame:
    """In-place QC enrichment on ``calibrated/lights`` (no ``processed/`` copy tree).

    Mutates calibrated FITS in place:
    - optional order-N polynomial sky-surface subtract (``preprocess_sky_surface_order``)
    - FWHM / elongation QC headers (``VY_FWHM``, ``VY_ELONG``, ``VYVARPR``, ...)
    - INV-CAL-02 stage stamp when pixels change (``VY_CALSTAGE``, ``VY_CALDATASUM``)

    Optional ``inject_pointing_*``: write ``VYTARGRA`` / ``VYTARGDE`` (deg ICRS) for plate-solve hints.

    Parallelism: auto worker count from host CPU (capped) or env
    ``VYVAR_PARALLEL_WORKERS`` / legacy env (see :func:`_vyvar_parallel_worker_count`).
    """
    cfg = app_config or AppConfig()
    calibrated_root = Path(calibrated_root)

    if db is not None and draft_id is not None:
        _ra_eff, _de_eff = resolve_preprocess_target_coordinates(
            db=db,
            draft_id=int(draft_id),
            ui_ra_deg=inject_pointing_ra_deg,
            ui_dec_deg=inject_pointing_dec_deg,
        )
        if _ra_eff is not None and _de_eff is not None:
            inject_pointing_ra_deg = float(_ra_eff)
            inject_pointing_dec_deg = float(_de_eff)

    log_event(
        f"Preprocess: running QC in-place on draft lights ({calibrated_root})"
    )
    log_milestone(f"[PREPROCESS] start in-place QC sky_order={int(cfg.preprocess_sky_surface_order)}")
    _out = _qc_enrich_calibrated_in_place(
        calibrated_root,
        app_config=cfg,
        fwhm_reject_limit=reject_fwhm_px,
        elong_reject_limit=reject_elongation,
        target_ra=inject_pointing_ra_deg,
        target_dec=inject_pointing_dec_deg,
        inject_pointing_only_if_missing=inject_pointing_only_if_missing,
        only_paths=only_paths,
        prefilter_rejected=prefilter_rejected,
        progress_cb=progress_cb,
        draft_id=draft_id,
        db=db,
    )
    _skip_n = int(_out.get("sky_surface_skip_count") or 0)
    if _skip_n:
        log_milestone(f"[PREPROCESS] sky_surface skipped {_skip_n} frame(s) (VY_SKYSF guard)")
    if _out.get("sky_surface_force_reapply"):
        LOGGER.warning(
            "preprocess_sky_surface_force_reapply=True: sky-surface guard bypassed on this run"
        )
    df = pd.DataFrame(_out.get("results") or [])
    df.attrs["preprocess_sky_summary"] = {
        "sky_surface_skip_count": _skip_n,
        "sky_surface_force_reapply": bool(_out.get("sky_surface_force_reapply")),
    }
    return df


def preprocess_calibrated_to_processed(
    *,
    calibrated_root: Path,
    processed_root: Path,
    reject_fwhm_px: float | None = None,
    reject_elongation: float | None = None,
    use_gpu_if_available: bool = False,
    progress_cb: Callable[..., None] | None = None,
    inject_pointing_ra_deg: float | None = None,
    inject_pointing_dec_deg: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
    prefilter_rejected: Mapping[Path | str, str] | None = None,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
    app_config: AppConfig | None = None,
) -> pd.DataFrame:
    """Deprecated alias for :func:`qc_enrich_calibrated_lights_in_place`.

    ``processed_root`` and ``use_gpu_if_available`` are ignored (legacy API).
    """
    _ = processed_root, use_gpu_if_available
    return qc_enrich_calibrated_lights_in_place(
        calibrated_root=calibrated_root,
        reject_fwhm_px=reject_fwhm_px,
        reject_elongation=reject_elongation,
        progress_cb=progress_cb,
        inject_pointing_ra_deg=inject_pointing_ra_deg,
        inject_pointing_dec_deg=inject_pointing_dec_deg,
        inject_pointing_only_if_missing=inject_pointing_only_if_missing,
        only_paths=only_paths,
        prefilter_rejected=prefilter_rejected,
        db=db,
        draft_id=draft_id,
        app_config=app_config,
    )


def preprocess_sky_summary_from_df(df: pd.DataFrame) -> dict[str, Any]:
    """Extract sky-surface guard counters stored on a preprocess result dataframe."""
    raw = getattr(df, "attrs", {}).get("preprocess_sky_summary") or {}
    return {
        "sky_surface_skip_count": int(raw.get("sky_surface_skip_count") or 0),
        "sky_surface_force_reapply": bool(raw.get("sky_surface_force_reapply")),
    }



def analyze_calibrated_qc(
    *,
    calibrated_root: Path,
    max_frames: int | None = None,
    progress_cb: Callable[..., None] | None = None,
    only_paths: Sequence[Path | str] | None = None,
) -> pd.DataFrame:
    """Analyze calibrated frames (QC) without writing /processed outputs.

    Runs FWHM/elongation QC on calibrated data in memory and returns a QC dataframe.

    If ``max_frames`` is None, every light FITS under ``calibrated_root`` is analyzed.

    Parallelism: jednotny pocet workerov (auto CPU/RAM alebo env, pozri :func:`_vyvar_parallel_worker_count`);
    integer ``>1`` uses a process pool by default (``VYVAR_PARALLEL_BACKEND=thread`` for threads).
    """
    calibrated_root = Path(calibrated_root)
    files = _filter_light_paths_maybe(_iter_light_fits(calibrated_root), only_paths)
    if max_frames is not None:
        files = files[: max(0, int(max_frames))]
    total = len(files)
    if total > 0:
        _mh = estimate_memory_from_fits_headers(files)
        _peak = int(float(_mh["bytes_float32_max_frame"]) * 6.0)
        LOGGER.info(
            "VYVAR QC analyze: %s frames; odhad spicky RAM ~%s (float32 + docasne polia)",
            total,
            format_memory_bytes(_peak),
        )
    n_workers = _vyvar_qc_preprocess_workers()
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "VYVAR QC analyze: parallel_workers=%s (paralelne; ~%sx RAM na snimok oproti 1 vlaknu)",
            n_workers,
            n_workers,
        )

    if n_workers > 1 and total > 1:
        with _vyvar_parallel_pool(n_workers) as ex:
            futs = {
                ex.submit(_analyze_calibrated_qc_one, src): src
                for src in files
            }
            by_src: dict[Path, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                src = futs[fut]
                by_src[src] = fut.result()
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Analyzing {src.name}")
            rows = [by_src[s] for s in files]
    else:
        rows = []
        for i, src in enumerate(files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Analyzing {src.name}")
            rows.append(_analyze_calibrated_qc_one(src))

    return pd.DataFrame(rows)


def _qc_suggest_thresholds(df: "pd.DataFrame") -> dict[str, float | int | None]:
    """Compute best/worst and robust suggested reject thresholds."""
    import numpy as np

    out: dict[str, float | int | None] = {
        "fwhm_min": None,
        "fwhm_median": None,
        "fwhm_max": None,
        "elong_min": None,
        "elong_median": None,
        "elong_max": None,
        "suggest_reject_fwhm_px": None,
        "suggest_reject_elongation": None,
        "n_qc": 0,
    }
    if df is None or df.empty:
        return out

    def _robust_suggest(x: "np.ndarray", k: float) -> float | None:
        x = x[np.isfinite(x)]
        if x.size < 5:
            return None
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med))) + 1e-6
        sigma = 1.4826 * mad
        return float(med + k * sigma)

    f = np.asarray(df.get("fwhm_px"), dtype=np.float64)
    e = np.asarray(df.get("elongation"), dtype=np.float64)
    f_ok = f[np.isfinite(f)]
    e_ok = e[np.isfinite(e)]
    out["n_qc"] = int(min(f_ok.size, e_ok.size) if (f_ok.size and e_ok.size) else max(f_ok.size, e_ok.size))

    if f_ok.size:
        out["fwhm_min"] = float(np.nanmin(f_ok))
        out["fwhm_median"] = float(np.nanmedian(f_ok))
        out["fwhm_max"] = float(np.nanmax(f_ok))
        # Suggest: median + 3*MAD_sigma (conservative)
        out["suggest_reject_fwhm_px"] = _robust_suggest(f_ok, k=3.0)
    if e_ok.size:
        out["elong_min"] = float(np.nanmin(e_ok))
        out["elong_median"] = float(np.nanmedian(e_ok))
        out["elong_max"] = float(np.nanmax(e_ok))
        # Suggest: median + 4*MAD_sigma (elongation is often tighter)
        out["suggest_reject_elongation"] = _robust_suggest(e_ok, k=4.0)

    return out


def scan_calibrated_lights_pointing(
    calibrated_root: Path | str,
    *,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Summarize celestial WCS vs object-style pointing keywords under ``calibrated_root`` (lights).

    If ``max_files`` is None, all light FITS under the tree are scanned.
    """
    from astropy.wcs import WCS

    root = Path(calibrated_root)
    files = _iter_light_fits(root)
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    rows: list[dict[str, Any]] = []
    n_wcs = 0
    n_hint = 0
    n_missing = 0

    def _wcs_center_from_header(h: fits.Header) -> tuple[float | None, float | None]:
        try:
            if not _has_valid_wcs(h):
                return None, None
            nax1 = int(h.get("NAXIS1") or 0)
            nax2 = int(h.get("NAXIS2") or 0)
            if nax1 <= 0 or nax2 <= 0:
                return None, None
            w = WCS(h, relax=True)
            cx = 0.5 * float(nax1)
            cy = 0.5 * float(nax2)
            ra, dec = w.celestial.all_pix2world([cx], [cy], 0)
            ra0 = float(ra[0]) % 360.0
            de0 = float(dec[0])
            if math.isfinite(ra0) and math.isfinite(de0) and (-90.0 <= de0 <= 90.0):
                return ra0, de0
        except Exception:  # noqa: BLE001
            # EXC-0442: T2 -- JD parse helper returns `None`; WCS change-point logic excludes frame from segment list. (EXCEPT-BULK-2 2026-07-08)
            return None, None
        return None, None

    def _jd_from_header(h: fits.Header) -> float | None:
        try:
            meta = fits_metadata_from_primary_header(h)
            jd = meta.get("jd_start")
            if jd is None:
                return None
            jd_f = float(jd)
            if math.isfinite(jd_f) and jd_f > 0:
                return jd_f
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0443] Multi-pointing segment detector returns `None` on any internal error; caller treats as ...: %s', exc)
            return None
        return None

    def _detect_segments_from_wcs_centers(
        _rows: list[dict[str, Any]],
        *,
        break_arcmin: float = 10.0,
        min_segment_size: int = 12,
    ) -> dict[str, Any] | None:
        try:
            import numpy as np
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            pts: list[tuple[int, float, float, float]] = []
            for idx, r in enumerate(_rows):
                ra = r.get("wcs_center_ra_deg")
                de = r.get("wcs_center_dec_deg")
                jd = r.get("jd")
                if ra is None or de is None or jd is None:
                    continue
                ra_f = float(ra)
                de_f = float(de)
                jd_f = float(jd)
                if math.isfinite(ra_f) and math.isfinite(de_f) and math.isfinite(jd_f) and jd_f > 0:
                    pts.append((idx, jd_f, ra_f, de_f))
            if len(pts) < max(20, 2 * int(min_segment_size)):
                return None
            pts = sorted(pts, key=lambda t: t[1])
            coords = [SkyCoord(ra=ra * u.deg, dec=de * u.deg, frame="icrs") for _, _, ra, de in pts]
            break_positions: list[int] = []
            seps_arcmin: list[float] = []
            for i in range(1, len(coords)):
                s = float(coords[i - 1].separation(coords[i]).arcminute)
                seps_arcmin.append(s)
                if math.isfinite(s) and s >= float(break_arcmin):
                    break_positions.append(i)
            if not break_positions:
                return None
            cuts = [0] + break_positions + [len(pts)]
            segments: list[dict[str, Any]] = []
            for a, b in itertools.pairwise(cuts):
                seg = pts[a:b]
                if len(seg) < int(min_segment_size):
                    continue
                ras = np.asarray([p[2] for p in seg], dtype=np.float64)
                des = np.asarray([p[3] for p in seg], dtype=np.float64)
                ang = np.deg2rad(ras)
                x = np.nanmedian(np.cos(ang))
                y = np.nanmedian(np.sin(ang))
                ra_med = float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)
                de_med = float(np.nanmedian(des))
                segments.append(
                    {
                        "segment_id": int(len(segments)),
                        "n": int(len(seg)),
                        "jd_min": float(min(p[1] for p in seg)),
                        "jd_max": float(max(p[1] for p in seg)),
                        "median_ra_deg": ra_med,
                        "median_dec_deg": de_med,
                        "member_row_indices": [int(p[0]) for p in seg],
                    }
                )
            if len(segments) < 2:
                return None
            return {
                "detected": True,
                "method": "wcs_center_change_point",
                "break_arcmin": float(break_arcmin),
                "min_segment_size": int(min_segment_size),
                "n_points_used": int(len(pts)),
                "breaks_on_sorted_points": [int(x) for x in break_positions],
                "segments": segments,
                "sep_arcmin_max": float(max(seps_arcmin)) if seps_arcmin else None,
            }
        except Exception:  # noqa: BLE001
            return None
    for fp in files:
        with fits.open(fp, memmap=False) as hdul:
            h = hdul[0].header
        hwcs = _has_valid_wcs(h)
        ha, hd, hs = _pointing_hint_from_header(h)
        wra, wde = _wcs_center_from_header(h)
        jd = _jd_from_header(h)
        da, dd, ds = ha, hd, hs
        if hwcs:
            n_wcs += 1
        elif ha is not None and hd is not None:
            n_hint += 1
        else:
            n_missing += 1
        rows.append(
            {
                "file": fp.name,
                "has_celestial_wcs": hwcs,
                "hint_ra_deg": ha,
                "hint_dec_deg": hd,
                "hint_source": hs,
                "wcs_center_ra_deg": wra,
                "wcs_center_dec_deg": wde,
                "jd": jd,
                "display_ra_deg": da,
                "display_dec_deg": dd,
                "display_source": ds,
            }
        )
    seg = _detect_segments_from_wcs_centers(rows)
    return {
        "calibrated_root": str(root.resolve()),
        "n_files_scanned": len(rows),
        "n_has_celestial_wcs": n_wcs,
        "n_has_object_hint_no_wcs": n_hint,
        "n_no_pointing_hint": n_missing,
        "rows": rows,
        "pointing_segments": seg,
    }


def _parse_fits_binning_int(raw: Any, default: int = 1) -> int:
    from utils import parse_fits_binning_int

    return parse_fits_binning_int(raw, default)


def _log_effective_pixel_pitch(meta: dict[str, Any], *, filepath: str = "") -> None:
    """Infolog: physical pixel, binning, effective pixel for plate-scale style calculations."""
    pixsz_from_header_or_db = meta.get("pixel_size_um_physical")
    binning_x = max(1, int(meta.get("binning", 1) or 1))
    binning_y = max(1, int(meta.get("binning_y", binning_x) or binning_x))
    effective_pixsz = meta.get("effective_pixel_um_plate_scale")
    tail = f" - {filepath}" if filepath else ""
    try:
        ps_s = f"{float(pixsz_from_header_or_db):.4g}" if pixsz_from_header_or_db is not None else "n/a"
        eff_s = f"{float(effective_pixsz):.4g}" if effective_pixsz is not None else "n/a"
        log_event(
            f"DEBUG: Fyzicky pixel: {ps_s} um | Binning: {binning_x}x{binning_y} | "
            f"EFEKTIVNY PIXEL PRE VYPOCET: {eff_s} um{tail}"
        )
    except (TypeError, ValueError):
        pass


def fits_metadata_from_primary_header(
    header: fits.Header,
    *,
    force_physical_pixel_um: float | None = None,
) -> dict[str, Any]:
    """Build the same dict as :func:`extract_fits_metadata` from an already-loaded primary header.

    ``pixel_size_um_header`` is the **effective** on-sky pitch [um] (physical pixel from header x binning).
    ``pixel_size_um_physical`` is the native pitch before binning (after optional ``force_physical_pixel_um``).
    ``effective_pixel_um_plate_scale`` is native x XBINNING [um] for plate scale / solver hints.
    """

    def _pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in header and header[key] not in (None, ""):
                return header[key]
        return default

    jd_obs = _pick("JD", "JD-OBS", default=None)
    if jd_obs is not None:
        jd_start = _to_float_db(jd_obs, 0.0)
    else:
        mjd_obs = _pick("MJD-OBS", default=None)
        if mjd_obs is not None:
            jd_start = _to_float_db(mjd_obs, 0.0) + 2400000.5
        else:
            # Fallback: compute JD from exposure start time keywords.
            # DATE-OBS is typically the start of exposure in UTC (FITS standard).
            date_obs = _pick("DATE-OBS", "DATEOBS", default=None)
            time_obs = _pick("TIME-OBS", "TIMEOBS", default=None)
            jd_start = 0.0
            if date_obs is not None:
                dt_str = str(date_obs).strip()
                if time_obs is not None and "T" not in dt_str:
                    dt_str = f"{dt_str}T{str(time_obs).strip()}"
                try:
                    jd_start = float(Time(dt_str, format="isot", scale="utc").jd)
                except Exception:  # noqa: BLE001
                    jd_start = 0.0

    _raw_xb = _pick("XBINNING", "BINNING", default=1)
    _raw_yb = _pick("YBINNING", default=_raw_xb)
    x_bin = _parse_fits_binning_int(_raw_xb, 1)
    y_bin = _parse_fits_binning_int(_raw_yb, x_bin)

    _ps1 = _fits_pixel_raw_to_micrometres(_to_float_db(_pick("PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0)))
    _ps2 = _fits_pixel_raw_to_micrometres(_to_float_db(_pick("PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0)))

    _force = None
    if force_physical_pixel_um is not None:
        try:
            fv = float(force_physical_pixel_um)
            if math.isfinite(fv) and fv > 0:
                _force = fv
        except (TypeError, ValueError):
            _force = None
    if _force is not None:
        _ps1, _ps2 = _force, _force

    _physical_x = _ps1 if _ps1 > 0 else None
    _physical_y = _ps2 if _ps2 > 0 else None
    if _physical_x is not None and _physical_y is not None:
        _physical_mean = (_physical_x + _physical_y) / 2.0
    elif _physical_x is not None:
        _physical_mean = float(_physical_x)
    elif _physical_y is not None:
        _physical_mean = float(_physical_y)
    else:
        _physical_mean = None

    _eff_x = (_physical_x * float(x_bin)) if _physical_x is not None else None
    _eff_y = (_physical_y * float(y_bin)) if _physical_y is not None else None
    if _eff_x is not None and _eff_y is not None:
        _effective_um = (_eff_x + _eff_y) / 2.0
    elif _eff_x is not None:
        _effective_um = float(_eff_x)
    elif _eff_y is not None:
        _effective_um = float(_eff_y)
    else:
        _effective_um = None

    if _physical_mean is not None:
        _plate_eff = float(_physical_mean) * float(x_bin)
    else:
        _plate_eff = _effective_um

    return {
        "exposure": float(_pick("EXPTIME", "EXPOSURE", default=0.0)),
        "filter": str(_pick("FILTER", "FILT", default="NoFilter")),
        "binning": int(x_bin),
        "binning_y": int(y_bin),
        "fits_xbinning_raw": _raw_xb,
        "fits_ybinning_raw": _raw_yb,
        "naxis1": int(header.get("NAXIS1", 0) or 0),
        "naxis2": int(header.get("NAXIS2", 0) or 0),
        "pixel_size_um_physical": _physical_mean,
        "pixel_size_um_header": _effective_um,
        "effective_pixel_um_plate_scale": _plate_eff,
        "temp": float(_pick("CCD-TEMP", "SENSORTEMP", "SET-TEMP", default=0.0)),
        "gain": int(_pick("GAIN", "GAINER", "CCD-GAIN", default=0) or 0),
        "ra": _fits_meta_ra_deg(
            _pick(
                "OBJCTRA",
                "RA_OBJ",
                "TARGRA",
                "CENTRA",
                "RA",
                "RAJ2000",
                "CAT-RA",
                "CRVAL1",
                default=0.0,
            )
        ),
        "dec": _fits_meta_dec_deg(
            _pick(
                "OBJCTDEC",
                "DEC_OBJ",
                "TARGDEC",
                "CENTDEC",
                "DEC",
                "DEJ2000",
                "CAT-DEC",
                "CRVAL2",
                default=0.0,
            )
        ),
        "jd_start": jd_start,
        "telescope": _pick("TELESCOP", "SCOPE", default=None),
        "camera": _pick("INSTRUME", "CAMERA", default=None),
        "bayerpat": _valid_bayerpat_from_header(header),
    }


def _valid_bayerpat_from_header(header: fits.Header) -> str | None:
    from osc_extract import valid_bayer_pattern_4

    return valid_bayer_pattern_4(str(header.get("BAYERPAT") or ""))


def extract_fits_metadata(
    filepath: Path | str,
    *,
    db: VyvarDatabase | None = None,
    app_config: AppConfig | None = None,
    force_physical_pixel_um: float | None = None,
    id_equipment: int | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Extract key metadata from FITS primary header.

    When ``db`` is set, uses ``FITS_HEADER_CACHE`` when ``FILE_PATH``, ``FILE_SIZE``, and ``MTIME`` match
    the file on disk; otherwise reads the FITS header and refreshes the cache row.

    ``ra`` / ``dec`` are in decimal degrees. Sources (first match wins): OBJCTRA/OBJCTDEC,
    RA_OBJ/DEC_OBJ, TARGRA/TARGDEC, CENTRA/CENTDEC, RA/DEC, RAJ2000/DEJ2000, CAT-RA/CAT-DEC,
    CRVAL1/CRVAL2.
    Sexagesimal strings are accepted with colons or spaces (e.g. ``' 3 39 06.45'`` for RA).

    Returned keys:
    - exposure
    - filter
    - binning (XBINNING / BINNING)
    - binning_y (YBINNING, default same as binning)
    - naxis1, naxis2
    - pixel_size_um_physical (native pitch from header / DB merge)
    - pixel_size_um_header (**effective** pitch [um] = physical x binning for plate solve / WCS)
    - effective_pixel_um_plate_scale (native mean x XBINNING; solver / plate scale)
    - focal_length [mm]: ``FOCALLEN`` in header when present; else ``TELESCOPE.FOCAL``
    - focal_length_source, pixel_size_raw (surove cisla z hlavicky), pixel_um (= effective pixel)
    - temp
    - ra
    - dec
    - jd_start
    - telescope
    - camera
    - bayerpat (FITS BAYERPAT when present, e.g. RGGB)

    Convention:     ``EQUIPMENTS.PIXELSIZE`` in the DB is the **native 1x1** pitch [um]; FITS cache
    ``PIXEL_UM`` stores **effective** pitch after binning.

    When ``draft_id`` is set and ``db`` is given, :meth:`VyvarDatabase.get_combined_metadata` overlays
    FITS+SQL focal/pixel (``XBINNING``-strict effective pixel) and ``EQUIPMENTS.SATURATE_ADU``.
    """
    fp = Path(filepath)
    id_telescope: int | None = None
    if db is not None and draft_id is not None:
        try:
            dr = db.fetch_obs_draft_by_id(int(draft_id)) or {}
            if dr.get("ID_TELESCOPE") is not None:
                id_telescope = int(dr["ID_TELESCOPE"])
        except Exception:  # noqa: BLE001
            id_telescope = None
    st: os.stat_result | None = None
    try:
        st = fp.stat()
    except OSError:
        st = None

    if db is not None and st is not None:
        cached = db.fits_header_cache_try_meta(fp)
        if cached is not None:
            meta = dict(cached)
            if id_equipment is not None:
                _merge_equipment_pixel_into_metadata(meta, db, int(id_equipment))
            _recompute_effective_pixel_from_physical(meta)
            with fits.open(fp, memmap=False) as hdul:
                _enrich_calibration_metadata_from_header(
                    meta,
                    hdul[0].header,
                    db=db,
                    id_equipment=id_equipment,
                    id_telescope=id_telescope,
                )
            if draft_id is not None:
                comb = db.get_combined_metadata(fp, int(draft_id))
                _apply_draft_combined_to_pipeline_meta(meta, comb)
            return meta

    _fpu: float | None = None
    if force_physical_pixel_um is not None:
        try:
            v = float(force_physical_pixel_um)
            if v > 0 and math.isfinite(v):
                _fpu = v
        except (TypeError, ValueError):
            _fpu = None
    with fits.open(fp, memmap=False) as hdul:
        header = hdul[0].header
    meta = fits_metadata_from_primary_header(header, force_physical_pixel_um=_fpu)
    if db is not None and id_equipment is not None:
        _merge_equipment_pixel_into_metadata(meta, db, int(id_equipment))
    _recompute_effective_pixel_from_physical(meta)
    _enrich_calibration_metadata_from_header(
        meta,
        header,
        db=db,
        id_equipment=id_equipment,
        id_telescope=id_telescope,
    )
    if draft_id is not None and db is not None:
        comb = db.get_combined_metadata(fp, int(draft_id))
        _apply_draft_combined_to_pipeline_meta(meta, comb)
    _log_effective_pixel_pitch(meta, filepath=str(fp.name))

    if db is not None and st is not None:
        imagetyp = str(header.get("IMAGETYP") or header.get("FRAME") or header.get("IMTYPE") or "")
        do = header.get("DATE-OBS") or header.get("DATEOBS")
        date_obs = None if do in (None, "") else str(do)
        db.fits_header_cache_upsert_one(
            fp,
            file_size=int(st.st_size),
            mtime=float(st.st_mtime),
            meta=meta,
            imagetyp=imagetyp,
            date_obs=date_obs,
        )

    return meta


def scan_usb_folder(path: Path | str) -> pd.DataFrame:
    """Recursively scan source tree and detect Lights/Darks/Flats by IMAGETYP/FRAME.

    This scan is folder-name agnostic and reports real folder paths.
    If a folder contains mixed types, it will be marked as Mixed, but files can still be
    sorted per-file by IMAGETYP downstream (importer).

    Output columns:
    - Folder Path
    - Type
    - File Count
    - Lights Count
    - Darks Count
    - Flats Count
    - Unknown Count
    - Detected Filters
    - Params
    """
    root = Path(path)
    rows: list[dict[str, Any]] = []

    _light_tokens = frozenset(
        {"light", "light frame", "lights", "object", "science"},
    )

    def _classify(text: str) -> str:
        t = (text or "").strip().lower()
        if "dark" in t:
            return "Darks"
        if "flat" in t:
            return "Flats"
        if t in _light_tokens or "light" in t:
            return "Lights"
        return "Unknown"

    def _fits_files(folder: Path) -> list[Path]:
        files: list[Path] = []
        seen: set[str] = set()
        for fp in folder.iterdir():
            if not fp.is_file():
                continue
            if not path_suffix_is_fits(fp):
                continue
            key = str(fp.resolve()).casefold()
            if key in seen:
                continue
            seen.add(key)
            files.append(fp)
        return sorted(files)

    if not root.exists() or not root.is_dir():
        return pd.DataFrame(columns=["Folder Path", "Type", "File Count", "Params"])

    # scan all subfolders, including root itself
    folders = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for folder in folders:
        files = _fits_files(folder)
        if not files:
            continue

        # classify per-file to detect mixed folders
        type_counts: dict[str, int] = {"Lights": 0, "Darks": 0, "Flats": 0, "Unknown": 0}
        filter_set: set[str] = set()
        for fp in files:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header
                imagetyp = str(hdr.get("IMAGETYP") or hdr.get("FRAME") or hdr.get("IMTYPE") or "")
                cls = _classify(imagetyp)
                flt = str(hdr.get("FILTER") or hdr.get("FILT") or "").strip()
                if flt:
                    filter_set.add(flt)
            except Exception:  # noqa: BLE001
                cls = "Unknown"
            type_counts[cls] = type_counts.get(cls, 0) + 1

        present = [k for k, v in type_counts.items() if v > 0 and k != "Unknown"]
        if len(present) == 1:
            detected = present[0]
        elif len(present) > 1:
            detected = "Mixed"
        else:
            detected = "Unknown"

        # params from first file
        first = files[0]
        try:
            with fits.open(first, memmap=False) as hdul:
                hdr = hdul[0].header
            exp = hdr.get("EXPTIME") or hdr.get("EXPOSURE") or 0
            gain = hdr.get("GAIN") or hdr.get("GAINER") or hdr.get("CCD-GAIN") or 0
            params = f"Exp={float(exp):g}s, Gain={int(gain)}"
        except Exception:  # noqa: BLE001
            params = ""

        rows.append(
            {
                "Folder Path": str(folder),
                "Type": detected,
                "File Count": int(len(files)),
                "Lights Count": int(type_counts.get("Lights", 0)),
                "Darks Count": int(type_counts.get("Darks", 0)),
                "Flats Count": int(type_counts.get("Flats", 0)),
                "Unknown Count": int(type_counts.get("Unknown", 0)),
                "Detected Filters": ", ".join(sorted(filter_set)) if filter_set else "",
                "Params": params,
            }
        )

    return pd.DataFrame(rows).sort_values(["Type", "Folder Path"], ignore_index=True)


class AstroPipeline:
    """Skeleton for the modular variable-star processing workflow."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.db = VyvarDatabase(self.config.database_path)
        self.db._archive_root_override = Path(self.config.archive_root)

    def calibrate(self, session_path: Path | str) -> Any:
        """Calibrate raw images and build nightly masters.

        Planned implementation:
        - library masters: mean stack (dark), median stack (flat); calibration re-use library
        """
        LOGGER.info("Calibrate requested for session: %s", session_path)
        return None

    def quick_calibrate_last_import(
        self,
        *,
        archive_path: Path | str,
        master_dark_path: Path | None,
        masterflat_by_filter: dict[str, Path | None],
        progress_cb: "callable | None" = None,
        equipment_id: int | None = None,
        draft_id: int | None = None,
        observation_id: str | None = None,
        masterflat_by_obs_key: dict[str, str | Path | None] | None = None,
        master_dark_by_obs_key: dict[str, str | Path | None] | None = None,
        roundness_reject_above: float | None = 1.25,
    ) -> dict[str, Any]:
        """Calibrate imported draft/raw lights into `calibrated/` under archive_path."""
        ap = Path(archive_path)
        # Accept both draft root (.../draft_xxx) and direct non_calibrated path (.../draft_xxx/non_calibrated).
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        LOGGER.info("Kalibracia archivu: %s", ap)
        # Support Draft structure: <archive>/Raw/lights and optional <archive>/non_calibrated/lights.
        # Both sources are calibrated/passthrough-written into the single target: <archive>/calibrated/lights.
        outputs: dict[str, Any] = {"archive_path": str(ap_root), "results": {}, "perf10_qsum": {}}
        merged_perf10: dict[str, dict[str, Any]] = {}

        draft_lights = ap_root / "non_calibrated" / "lights"
        if not draft_lights.exists() and ap.name.casefold() == "non_calibrated":
            draft_lights = ap / "lights"
        raw_lights = ap_root / "Raw" / "lights"
        # In non_calibrated draft mode always prefer Draft/non_calibrated/lights.
        if draft_lights.exists():
            out_root = ap_root / "calibrated" / "lights"
            outputs["results"]["non_calibrated"] = calibrate_lights_to_calibrated(
                lights_root=draft_lights,
                calibrated_root=out_root,
                master_dark_path=master_dark_path,
                masterflat_by_filter=masterflat_by_filter,
                progress_cb=progress_cb,
                pipeline_config=self.config,
                equipment_id=equipment_id,
                draft_id=draft_id,
                observation_id=observation_id,
                masterflat_by_obs_key=masterflat_by_obs_key,
                master_dark_by_obs_key=master_dark_by_obs_key,
            )
        elif raw_lights.exists():
            out_root = ap_root / "calibrated" / "lights"
            outputs["results"]["Raw"] = calibrate_lights_to_calibrated(
                lights_root=raw_lights,
                calibrated_root=out_root,
                master_dark_path=master_dark_path,
                masterflat_by_filter=masterflat_by_filter,
                progress_cb=progress_cb,
                pipeline_config=self.config,
                equipment_id=equipment_id,
                draft_id=draft_id,
                observation_id=observation_id,
                masterflat_by_obs_key=masterflat_by_obs_key,
                master_dark_by_obs_key=master_dark_by_obs_key,
            )

        for _sec_stats in (outputs.get("results") or {}).values():
            if isinstance(_sec_stats, dict):
                _p10 = _sec_stats.get("perf10_qc_results")
                if isinstance(_p10, dict):
                    merged_perf10.update(_p10)

        if (
            bool(self.config.dao_qc_in_calibrate)
            and merged_perf10
            and draft_id is not None
        ):
            outputs["perf10_qsum"] = apply_perf10_dao_qc_to_obs_files(
                db=self.db,
                draft_id=int(draft_id),
                archive_path=ap_root,
                perf10_qc_results=merged_perf10,
                roundness_reject_above=roundness_reject_above,
            )

        LOGGER.info("Kalibracia dokoncena (sekcii vystupu: %s)", list((outputs.get("results") or {}).keys()))
        if equipment_id is not None:
            cal_root = ap_root / "calibrated" / "lights"
            if cal_root.is_dir():
                osc_out = run_osc_channel_extraction_for_archive(
                    calibrated_lights_root=cal_root,
                    db=self.db,
                    equipment_id=int(equipment_id),
                    app_config=self.config,
                    progress_cb=progress_cb,
                )
                outputs["osc_extraction"] = osc_out
        return outputs

    def calibrate_batch(
        self,
        *,
        light_paths: Sequence[Path | str],
        lights_root: Path | str,
        calibrated_root: Path | str,
        master_dark_path: Path | str | None,
        masterflat_by_filter: dict[str, Path | str | None],
        max_workers: int | None = None,
        progress_cb: "callable | None" = None,
        equipment_id: int | None = None,
        draft_id: int | None = None,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply master dark/flat to many lights using ``ProcessPoolExecutor`` (``spawn``).

        Output layout matches :func:`calibrate_lights_to_calibrated`: for each input file
        ``dst = calibrated_root / Path(light).relative_to(lights_root)``.

        Returns a dict with:

        - ``output_paths``: list aligned with ``light_paths`` - calibrated FITS path or ``None`` on failure
        - ``results``: list of per-file ``dict``\\ s from workers (``src``, ``dst``, ``ok``, ``error``)
        - ``stats``: processed / ok / failed counts
        """
        import numpy as np

        lr = Path(lights_root).resolve()
        cr = Path(calibrated_root)
        cr.mkdir(parents=True, exist_ok=True)

        light_paths = filter_light_paths_for_calibration_db(
            [Path(lp) for lp in light_paths],
            database_path=self.config.database_path,
            draft_id=draft_id,
            observation_id=observation_id,
        )

        mf_paths: dict[str, Path | None] = {}
        for k, v in (masterflat_by_filter or {}).items():
            if v is None or str(v).strip() == "":
                mf_paths[str(k)] = None
            else:
                mf_paths[str(k)] = Path(v)

        mf_serial: dict[str, str | None] = {
            k: str(p.resolve()) if p is not None else None for k, p in mf_paths.items()
        }

        _md_log = Path(master_dark_path) if master_dark_path is not None else None
        _log_calibration_io_preflight(
            calibrated_root=cr,
            master_dark_path=_md_log,
            masterflat_by_filter=mf_paths,
        )

        md_init: str | None = None
        if master_dark_path is not None:
            md_p = Path(master_dark_path)
            if md_p.is_file():
                md_init = str(md_p.resolve())

        qc_pack = _qc_pack_from_config(
            self.config, draft_id=draft_id, observation_id=observation_id
        )

        items: list[
            tuple[
                str,
                str,
                str | None,
                dict[str, str | None],
                dict[str, Any] | None,
            ]
        ] = []
        for lp in light_paths:
            src_p = Path(lp).resolve()
            rel = src_p.relative_to(lr)
            dst_p = (cr / rel).resolve()
            items.append((str(src_p), str(dst_p), md_init, mf_serial, qc_pack))

        n = len(items)
        if n == 0:
            return {
                "output_paths": [],
                "results": [],
                "stats": {"n_input": 0, "ok": 0, "failed": 0},
            }

        nw = (
            max_workers
            if max_workers is not None
            else max(1, min(32, int(self.config.qc_preprocess_workers)))
        )
        nw = max(1, min(int(nw), n))
        if not _vyvar_calibrate_multiprocessing_enabled():
            nw = 1

        _native_b = _cfg_calibration_library_native_binning(self.config)

        cal_diag_session = CalDiagSession()
        db_cal = _db_for_calibration_tasks(qc_pack)
        _lpaths = [Path(it[0]) for it in items]
        if _lpaths and md_init:
            cal_diag_session = run_cal_diag_pregate(
                _lpaths,
                obs_group_key_from_path=_obs_group_key_from_light_path,
                resolve_dark_path=lambda fp, og, lb: Path(md_init) if md_init else None,
                light_binning_from_path=_light_binning_from_path,
                master_binning=_native_b,
                match_and_crop_pair=_match_and_crop_pair,
                saturation_for_light=lambda fp: _saturation_adu_for_cal_diag(
                    fits.getheader(fp, 0),
                    db=db_cal,
                    equipment_id=equipment_id,
                ),
                ui_error=_pipeline_ui_error,
            )
        cal_diag_worker_blob = _cal_diag_export_for_workers(cal_diag_session)

        rows: list[dict[str, Any]]
        if nw <= 1:
            md_pre: Any = None
            if md_init:
                with fits.open(md_init, memmap=False) as h:
                    md_pre = np.array(h[0].data, dtype=np.float32, copy=True)
            flat_cache: dict[str, Any] = {}
            flat_med: dict[str, float] = {}
            db_main = _db_for_calibration_tasks(qc_pack)
            rows = []
            for i, it in enumerate(items):
                src_s, dst_s, md_s, mf_map, _qopt = it
                src_p = Path(src_s)
                dst_p = Path(dst_s)
                try:
                    _ok = _obs_group_key_from_light_path(src_p)
                    if is_obs_group_aborted(cal_diag_session, _ok):
                        if dst_p.exists():
                            dst_p.unlink(missing_ok=True)
                        rows.append(
                            {
                                "src": src_s,
                                "dst": dst_s,
                                "ok": True,
                                "skipped": True,
                                "error": None,
                                "qc_summary": None,
                                "traceback": None,
                            }
                        )
                        if progress_cb is not None:
                            progress_cb(i + 1, n, f"CAL-DIAG skip {src_p.name}")
                        continue
                    light_bx = _light_binning_from_path(src_p)
                    gr = gate_result_for_frame(
                        cal_diag_session,
                        obs_group_key=_ok,
                        dark_path=Path(md_s) if md_s else None,
                        light_binning=light_bx,
                    )
                    md_np = md_pre
                    if md_s:
                        with fits.open(src_p, memmap=False) as hdul:
                            lshape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
                        md_np = dark_np_for_cal_diag(
                            cal_diag_session,
                            master_binning=_native_b,
                            dark_path=Path(md_s),
                            light_binning=light_bx,
                            light_shape=lshape,
                            light_filename=src_p.name,
                            gate_result=gr,
                        )
                    mf = {str(k): Path(v) if v else None for k, v in mf_map.items()}
                    _ud, _uf, qc_sum, _cf, _p10 = _calibrate_one_light_disk(
                        src=src_p,
                        dst=dst_p,
                        master_dark_path=Path(md_s) if md_s else None,
                        masterflat_by_filter=mf,
                        flat_cache=flat_cache,
                        flat_median_scale=flat_med,
                        md_data_preload=md_np,
                        db=db_main,
                        qc_pack=_qopt,
                        calibration_master_native_binning=_native_b,
                        cal_diag_gate_result=gr,
                    )
                    rows.append(
                        {
                            "src": src_s,
                            "dst": dst_s,
                            "ok": True,
                            "error": None,
                            "qc_summary": qc_sum,
                            "traceback": None,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    _tb_cb = traceback.format_exc()
                    LOGGER.error("calibrate_batch: %s -> %s\n%s", src_s, exc, _tb_cb)
                    log_exception(f"CHYBA WORKERA: {Path(src_s).name}", exc)
                    rows.append(
                        {
                            "src": src_s,
                            "dst": None,
                            "ok": False,
                            "error": str(exc),
                            "traceback": _tb_cb,
                        }
                    )
                if progress_cb is not None:
                    progress_cb(i + 1, n, f"Calibrating {Path(src_s).name}")
        else:
            ctx = multiprocessing.get_context("spawn")
            rows = [None] * n  # type: ignore[misc]
            with ProcessPoolExecutor(
                max_workers=nw,
                mp_context=ctx,
                initializer=_init_calibrate_batch_worker,
                initargs=(md_init, _native_b, cal_diag_worker_blob),
            ) as ex:
                future_map = {ex.submit(_calibrate_batch_process_one, it): idx for idx, it in enumerate(items)}
                done = 0
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    rows[idx] = fut.result()
                    done += 1
                    if progress_cb is not None:
                        progress_cb(done, n, f"Calibrating batch {done}/{n}")

        out_paths: list[str | None] = []
        ok_c = 0
        fail_c = 0
        for r in rows:
            if r.get("ok"):
                ok_c += 1
                out_paths.append(str(r["dst"]) if r.get("dst") else None)
            else:
                fail_c += 1
                out_paths.append(None)

        return {
            "output_paths": out_paths,
            "results": rows,
            "stats": {
                "n_input": n,
                "ok": ok_c,
                "failed": fail_c,
                "max_workers": nw,
            },
        }

    def quick_preprocess_last_import(
        self,
        *,
        archive_path: Path | str,
        run: bool = True,
        reject_fwhm_px: float | None = None,
        reject_elongation: float | None = None,
        use_gpu_if_available: bool = False,
    ) -> dict[str, Any]:
        ap = Path(archive_path)
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        src_cal = ap_root / "calibrated" / "lights"
        src_noncal = ap_root / "non_calibrated" / "lights"
        source_dir = src_cal if src_cal.exists() else src_noncal
        out: dict[str, Any] = {
            "archive_path": str(ap_root),
            "processed": {},
            "qc_suggestions": {},
            "checked": {
                "archive_exists": bool(ap_root.exists()),
                "source_path": str(source_dir),
                "calibrated_path": str(src_cal),
                "non_calibrated_path": str(src_noncal),
                "processed_path": str(ap_root / "processed" / "lights"),
                "detrended_path": str(ap_root / "detrended" / "lights"),
            },
        }

        if source_dir.exists():
            if run:
                df = preprocess_calibrated_to_processed(
                    calibrated_root=source_dir,
                    processed_root=source_dir,
                    reject_fwhm_px=reject_fwhm_px,
                    reject_elongation=reject_elongation,
                    use_gpu_if_available=use_gpu_if_available,
                    progress_cb=None,
                    app_config=self.config,
                )
            else:
                qc_csv_existing = source_dir / "qc_metrics.csv"
                if not qc_csv_existing.exists():
                    qc_csv_existing = ap_root / "detrended" / "lights" / "qc_metrics.csv"
                try:
                    df = pd.read_csv(qc_csv_existing) if qc_csv_existing.exists() else pd.DataFrame()
                except Exception:  # noqa: BLE001
                    df = pd.DataFrame()
            out["processed"]["source"] = {
                "lights_root": str(source_dir),
                "rows": int(len(df)),
                "rejected": int((df["status"].astype(str).str.startswith("rejected")).sum()) if not df.empty else 0,
                "source_dir": str(source_dir),
            }
            try:
                qc_csv = source_dir / "qc_metrics.csv"
                out["processed"]["source"]["qc_csv"] = str(qc_csv)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
            out["qc_suggestions"]["source"] = _qc_suggest_thresholds(df)

        if not out["processed"]:
            out["warning"] = "No calibrated lights found under this archive_path. Run calibration first (Run quick pipeline step)."
        return out

    def quick_analyze_last_import(
        self,
        *,
        archive_path: Path | str,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        ap = Path(archive_path)
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        out: dict[str, Any] = {
            "archive_path": str(ap_root),
            "analysis": {},
            "qc_suggestions": {},
        }
        cal = ap_root / "calibrated" / "lights"
        if cal.exists():
            df = analyze_calibrated_qc(
                calibrated_root=cal,
                max_frames=max_frames,
            )
            out["analysis"]["calibrated"] = {"rows": int(len(df))}
            out["qc_suggestions"]["calibrated"] = _qc_suggest_thresholds(df)
            try:
                qc_csv = ap_root / "calibrated" / "lights" / "qc_analysis.csv"
                df.to_csv(qc_csv, index=False)
                out["analysis"]["calibrated"]["qc_csv"] = str(qc_csv)
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0444] `fetch_draft_light_rows_for_quality` failure logs WARNING and returns empty jump result...: %s', exc)
                LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
        else:
            out["warning"] = "No calibrated lights found. Run calibration first."
        return out

    @staticmethod
    def _first_fits_file(session_path: Path | str) -> Path:
        session = Path(session_path)
        search_roots = [session, session / "Raw"]
        for root in search_roots:
            if not root.exists():
                continue
            for fp in sorted(root.rglob("*")):
                if fp.is_file() and path_suffix_is_fits(fp):
                    return fp
        raise FileNotFoundError(f"No FITS file found in session path: {session}")

    def prepare_observation_from_session(
        self,
        session_path: Path | str,
        *,
        id_equipment: int | None = None,
        id_telescope: int | None = None,
        id_location: int = 1,
    ) -> dict[str, Any]:
        """Read first FITS and build prefilled OBSERVATION payload."""
        first_fits = self._first_fits_file(session_path)
        metadata = extract_fits_metadata(first_fits, db=self.db, app_config=self.config)
        scanning_id = self.db.derive_scanning_id(metadata)

        if id_equipment is None or id_telescope is None:
            raise ValueError(
                "prepare_observation_from_session: id_equipment a id_telescope su povinne "
                "(vyberte kameru a dalekohlad v Session Upload)."
            )
        equipment_id = int(id_equipment)
        telescope_id = int(id_telescope)

        observation_payload = {
            "id_equipments": equipment_id,
            "id_telescope": telescope_id,
            "id_location": int(id_location),
            "id_scanning": scanning_id,
            "center_of_field_ra": float(metadata["ra"]),
            "center_of_field_de": float(metadata["dec"]),
            "observation_start_jd": float(metadata["jd_start"]),
        }
        return {
            "fits_file": str(first_fits),
            "metadata": metadata,
            "observation_payload": observation_payload,
            "missing_telescope": not bool(metadata.get("telescope")),
            "missing_camera": not bool(metadata.get("camera")),
        }

def _field_jump_empty_result() -> dict[str, Any]:
    """Return empty/unknown result when data is insufficient."""
    return {
        "has_jump": False,
        "n_groups": 0,
        "jump_frames": [],
        "groups": [],
        "dominant_group_frac": 1.0,
        "recommended_min_frames_frac": 0.3,
        "warning_text": None,
        "total_frames": 0,
    }


def detect_field_jumps(
    db: "VyvarDatabase",
    draft_id: int,
    jump_threshold_arcmin: float = 3.0,
    min_frames_in_group: int = 5,
) -> dict[str, Any]:
    """
    Detect sudden field position jumps from per-frame DRIFT values.

    Uses DRIFT_DRA / DRIFT_DDE from manifest files[] (already computed by
    sync_obs_files_drift_arcmin_for_draft).

    Returns dict with keys:
      has_jump: bool
      n_groups: int
      jump_frames: list of {frame_index, delta_arcmin, file}
      groups: list of {group_id, frame_start, frame_end,
                       n_frames, frac, is_dominant}
      dominant_group_frac: float
      recommended_min_frames_frac: float
      warning_text: str | None
      total_frames: int
    """
    import math

    import numpy as np  # noqa: F401
    import pandas as pd

    _ = (min_frames_in_group,)  # reserved for future grouping guards

    try:
        rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"[FIELD JUMP] DB fetch failed: {exc}")
        return _field_jump_empty_result()

    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(rows)
    if rows.empty:
        return _field_jump_empty_result()

    if "FILE_PATH" not in rows.columns:
        return _field_jump_empty_result()

    # Filter: light frames if IMAGETYP is present (defensive; fetch_* already does this).
    if "IMAGETYP" in rows.columns:
        rows = rows[rows["IMAGETYP"].astype(str).str.contains("light", case=False, na=False)]

    # Sort by FILE_PATH (typically chronological on acquisition naming).
    rows = rows.sort_values("FILE_PATH").reset_index(drop=True)

    # Require DRIFT plane offsets.
    for col in ("DRIFT_DRA", "DRIFT_DDE", "DRIFT"):
        if col not in rows.columns:
            return _field_jump_empty_result()
        rows[col] = pd.to_numeric(rows[col], errors="coerce")

    # Keep only frames with finite DRIFT_DRA/DRIFT_DDE.
    valid = rows.dropna(subset=["DRIFT_DRA", "DRIFT_DDE"]).copy().reset_index(drop=True)
    if len(valid) < 3:
        return _field_jump_empty_result()

    total = int(len(valid))

    # Frame-to-frame delta (arcmin) in the DRIFT plane (degrees).
    dra = valid["DRIFT_DRA"].to_numpy(dtype=float)
    dde = valid["DRIFT_DDE"].to_numpy(dtype=float)

    deltas: list[float] = []
    for i in range(1, total):
        d = math.sqrt((dra[i] - dra[i - 1]) ** 2 + (dde[i] - dde[i - 1]) ** 2) * 60.0
        deltas.append(float(d))

    # Identify jumps.
    jump_indices: list[dict[str, Any]] = []
    thr = float(jump_threshold_arcmin)
    if not (math.isfinite(thr) and thr > 0):
        thr = 3.0
    for i, d in enumerate(deltas):
        if math.isfinite(float(d)) and float(d) > thr:
            jump_indices.append(
                {
                    "frame_index": int(i + 1),  # boundary at i+1 (0-based index in `valid`)
                    "delta_arcmin": round(float(d), 2),
                    "file": str(valid.iloc[i + 1].get("FILE_PATH", "")),
                }
            )

    if not jump_indices:
        return {
            "has_jump": False,
            "n_groups": 1,
            "jump_frames": [],
            "groups": [
                {
                    "group_id": 1,
                    "frame_start": 0,
                    "frame_end": total - 1,
                    "n_frames": total,
                    "frac": 1.0,
                    "is_dominant": True,
                }
            ],
            "dominant_group_frac": 1.0,
            "recommended_min_frames_frac": 0.3,
            "warning_text": None,
            "total_frames": total,
        }

    # Split into groups using jump boundaries.
    boundaries = [0] + [int(j["frame_index"]) for j in jump_indices] + [total]
    groups: list[dict[str, Any]] = []
    for gid, (start, end) in enumerate(itertools.pairwise(boundaries), 1):
        n = int(end - start)
        groups.append(
            {
                "group_id": int(gid),
                "frame_start": int(start),
                "frame_end": int(end - 1),
                "n_frames": n,
                "frac": round(float(n) / float(total), 3),
                "is_dominant": False,
            }
        )

    dominant = max(groups, key=lambda g: int(g.get("n_frames", 0) or 0))
    dominant["is_dominant"] = True
    dom_frac = float(dominant.get("frac", 0.0) or 0.0)

    recommended = round(dom_frac * 0.95, 2)
    recommended = max(0.3, min(0.95, float(recommended)))

    jump_frame_nos = [int(j["frame_index"]) for j in jump_indices]
    warning = (
        "Field jump detected at frame(s) "
        f"{', '.join(str(f) for f in jump_frame_nos)} "
        f"(threshold {thr:.1f}'). "
        f"{len(groups)} position groups found. "
        f"Dominant group: {int(dominant['n_frames'])}/{total} frames "
        f"({dom_frac * 100:.0f}%). "
        f"Recommended min_frames_frac >= {recommended:.2f}."
    )

    return {
        "has_jump": True,
        "n_groups": int(len(groups)),
        "jump_frames": jump_indices,
        "groups": groups,
        "dominant_group_frac": float(dom_frac),
        "recommended_min_frames_frac": float(recommended),
        "warning_text": warning,
        "total_frames": total,
    }

