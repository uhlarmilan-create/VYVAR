"""Photometry core - zluceny modul (photometry + photometry_phase2a)."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AbstractSet, Any, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.io import fits as astrofits

from stats_core import _flux_to_mag
from comp_pool_rms import attach_comp_rms_to_pool_rows, compute_global_pool_rms_map
from comp_rms_loo import (
    COMP_RMS_FRAMES_BASIS,
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    LN10_OVER_2P5,
    compute_loo_mag_rms_map,
)
from proc_frame_store import (
    PROC_CSV_GLOB,
    PROC_STORE_COLS,
    ProcFrameStore,
    is_masterstar_proc_name,
    proc_csv_path_for_aligned_fits,
)
from config import (
    AppConfig,
    DENSITY_OVERRIDES,
    apply_crowding_overrides,
    apply_density_overrides,
    classify_field_density,
    compute_field_density,
    resolve_comp_sparse_fallback_enabled,
)
from database import query_local_gaia, query_local_gaia_by_source_ids
from gaia_catalog_id import (
    GAIA_PROC_CSV_READ_DTYPE,
    masterstar_row_gaia_key,
    normalize_gaia_source_id,
    read_vyvar_csv,
)
from infolog import log_event
from plain_stats import plain_mean_med_std

from catalog_match_trust import is_wcs_untrusted_catalog_match_mode, normalize_catalog_match_mode
from jd_axis_format import jd_axis_title, jd_series_relative
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from unit_resolver import (
    phase01_chip_interior_margin_px as _resolve_chip_interior_margin_px,
    phase01_comparison_isolation_radius_px as _resolve_isolation_radius_px,
    resolve_max_dist_fallback_deg,
    resolve_px_from_arcsec,
    resolve_px_from_fwhm_factor,
    plate_scale_arcsec_per_px_from_header,
    sips_dao_fwhm_px as _resolve_sips_dao_fwhm_px,
)

LOGGER = logging.getLogger(__name__)

_MAD_CONSISTENCY = 0.6745  # normalizacny faktor MAD -> sigma ekvivalent

# Explicit annulus sky (ADU/px) for Howell err; ``noise_floor_adu`` remains detection-floor legacy.
SKY_ADU_PER_PX_ANNULUS_COL = "sky_adu_per_px_annulus"
SKY_SURFACE_BG_MEDIAN_ADU_COL = "sky_surface_bg_median_adu"

# F-BINGAIN-1: empirical background noise (empty-aperture scatter) + provenance.
SIGMA_BKG_AP_COL = "sigma_bkg_ap"
ERR_BKG_SOURCE_COL = "err_bkg_source"
ERR_BKG_MODE_EMPIRICAL = "empirical"
ERR_BKG_MODE_HOWELL = "howell"
ERR_BKG_SOURCE_EMPIRICAL = "empirical"
ERR_BKG_SOURCE_HOWELL_FALLBACK = "howell_fallback"
ERR_BKG_SOURCE_HOWELL_SCALED = "howell_scaled"
BKG_SCALE_R_CLAMP_LO = 0.05
BKG_SCALE_R_CLAMP_HI = 2.0





# Per-target LC time provenance (F-BJD-1): labels BJD recompute path, does not alter time values.
TIME_BASE_COL = "time_base"
TIME_BASE_BJD_TDB = "BJD_TDB"
TIME_BASE_JD_FALLBACK = "JD_FALLBACK"










# Comp tier: Gaia BP-RP outside this band -> unreliable vs field comps (use B-V fallback).
_BPRP_VALID_MIN = 0.1
_BPRP_VALID_MAX = 3.5

# Gaia ID (`catalog_id`, VSX / masterstars `name`) musi byt str - float64 straca cifry
_GAIA_ID_DTYPE: dict[str, type] = dict(GAIA_PROC_CSV_READ_DTYPE)




_COMP_QUALITY_JSON_META_KEYS = frozenset(
    {
        "selected_tier",
        "tier4_warning",
        "n_tier1",
        "n_tier2",
        "n_tier3",
        "n_tier4",
        "aperture_correction",
        "qa_degraded",
        "qa_degraded_reason",
    }
)









# ---------------------------------------------------------------------------
# Pomocne funkcie
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# KROK 1: Globalna fixna apertura z PSF FWHM (MASTERSTAR VY_FWHM alebo fit)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 2: Aperturna fotometria per snimka - medianovy sky
# ---------------------------------------------------------------------------















































































# ---------------------------------------------------------------------------
# ALG-3: Temporal binning of comparison ensemble (MNRAS 2023)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ALG-5: PyTICS iterative comp intercalibration (RASTI 2026)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 3: Stability check porovnavaciek (Abbeho p2p scatter + MAD)
# ---------------------------------------------------------------------------

# Observed-band / catalog mag before broad Gaia G for SNR-optimal aperture sizing.
_APERTURE_SIZING_MAG_COLS: tuple[str, ...] = (
    "mag",
    "catalog_mag",
    "lc_median_mag",
    "phot_g_mean_mag",
)











# ---------------------------------------------------------------------------
# KROK 4: Ensemble normalizacia
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Color term (BP-RP) - globalny shift na noc
# ---------------------------------------------------------------------------





































_CT_PROTOTYPE_CSV_FIELDS: tuple[str, ...] = (
    "catalog_id",
    "vsx_name",
    "obs_group",
    "n_comp_used",
    "c1",
    "c1_stderr",
    "stderr_ratio",
    "target_bp_rp",
    "comp_med_bp_rp",
    "ct_corr",
    "cat_inst_scatter",
    "cat_inst_scatter_resid",
    "gate_would_pass",
)







# ---------------------------------------------------------------------------
# KROK 5: Outlier detekcia
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
# KROK 6: Vystup - lightcurve CSV
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 6: Vystup - PNG grafy
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Hlavny wrapper - run_phase2a
# ---------------------------------------------------------------------------


_EDGE_FILTER_NOTE_OK = ""
_EDGE_FILTER_NOTE_FAILED = "EDGE-UNFILTERED: edge safety check failed"












# PFS-SEMANTICS-01: never rescue these skip_reason values (TARGET-DEPTH-02 outranks PFS).
PFS_NEVER_RESCUE_REASONS = frozenset({"zone_noise", "below_target_depth"})
PFS_SATURATION_SKIP_REASONS = frozenset(
    {
        "zone_flag",
        "saturovany ciel",
        "per_frame_saturation",
        "likely_saturated",
        "saturated",
    }
)




















_LC_QUALITY_FLAGS: tuple[str, ...] = (
    "good",
    "noisy",
    "noisy_moon",
    "short_baseline",
    "no_data",
    "saturated",
)














_GIT_PROVENANCE_WARNED = False
# src_py/photometry_core.py -> repo root is parent.parent (git cwd + porcelain path base).
_REPO_ROOT_FOR_PROVENANCE = Path(__file__).resolve().parent.parent
































_ADAPTIVE_BLEND_CACHE: dict[str, dict[str, BlendMapEntry]] = {}






from mag_constants import MAG_ERR_SCALE

_PSF_ERR_MAG_SCALE = MAG_ERR_SCALE















































# ======================================================================
# photometry.py (zlucene do photometry_core)
# ======================================================================

from utils import (
    fits_binning_xy_from_header,
    plate_scale_arcsec_per_pixel,
    plate_solve_fov_deg_diagonal_from_scale,
)











# Stlpce nacitavane z per-frame CSV pre bootstrap (78 % uspora pamate)
_PHASE_USECOLS_PERFRAME: list[str] = [
    "name",
    "catalog_id",
    "bjd_tdb_mid",
    "flux",
    "dao_flux",
    "noise_floor_adu",
    "sky_adu_per_px_annulus",
    "aperture_r_px",
    "is_usable",
    "is_saturated",
    "is_noisy",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
]


























def compute_auto_fwhm_limit(
    fwhm_values: np.ndarray | Sequence[float],
    k: float = 1.5,
) -> dict[str, Any]:
    """
    Vypocita automaticky FWHM limit pomocou MAD statistiky.

    Vracia dict:
        median_fwhm, mad, sigma_mad, auto_limit, k, n_total, n_kept, n_cut
    (``auto_limit`` moze byt ``None`` pri prilis malo bodoch.)
    """
    arr = np.asarray(fwhm_values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < 3:
        return {
            "median_fwhm": None,
            "mad": None,
            "sigma_mad": None,
            "auto_limit": None,
            "k": float(k),
            "n_total": int(len(arr)),
            "n_kept": int(len(arr)),
            "n_cut": 0,
        }
    median_f = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median_f)))
    sigma_mad = mad * 1.4826
    auto_limit = median_f + float(k) * sigma_mad
    n_kept = int(np.sum(arr <= auto_limit))
    return {
        "median_fwhm": round(median_f, 3),
        "mad": round(mad, 4),
        "sigma_mad": round(sigma_mad, 4),
        "auto_limit": round(float(auto_limit), 3),
        "k": float(k),
        "n_total": int(len(arr)),
        "n_kept": n_kept,
        "n_cut": int(len(arr) - n_kept),
    }






















































def run_full_photometry_pipeline(
    *,
    masterstar_fits_path: Path,
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    detrended_aligned_dir: Path,
    output_dir: Path,
    cfg: AppConfig | None = None,
    db: Any = None,
    draft_id: int | None = None,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Jedno-krokovy wrapper: Faza 0+1 + Faza 2A ako jeden celok.

    UI to pouziva ako jednu akciu 'RUN Aperture Photometry' pre dany obs_group.
    """
    _cfg = cfg or AppConfig()

    ensure_full_variable_targets_if_presel_stub(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        masterstar_fits=Path(masterstar_fits_path),
        cfg=_cfg,
        draft_id=draft_id,
    )

    def _p(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    # FWHM: prefer header (VY_FWHM_GAUSS/VY_FWHM), inak default z configu.
    fwhm_px = _resolve_sips_dao_fwhm_px(_cfg, fwhm_px=None)
    _ms_header_shared: Any | None = None
    _ms_path_shared = Path(masterstar_fits_path)
    if _ms_path_shared.is_file():
        try:
            from astropy.io import fits as astrofits  # noqa: PLC0415

            with astrofits.open(_ms_path_shared, memmap=False) as _hdul:
                _ms_header_shared = _hdul[0].header.copy()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0220] Shared MASTERSTAR header cache load fails - repeated FITS opens (perf), not science num...: %s', exc)
            logging.warning("[PERF-2] Cannot open MASTERSTAR.fits for header: %s", exc)
    if _ms_header_shared is not None:
        try:
            # Prefer night seeing (VY_FWHM) over Gaussian-core (VY_FWHM_GAUSS) for
            # Phase-1 isolation / blend geometry. Core FWHM under-states the CoG
            # 3-FWHM single-source radius (COMP-ASSIGN-03 / A-1).
            for key in ("VY_FWHM", "VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN"):
                v = _ms_header_shared.get(key)
                if v is None:
                    continue
                fv = float(v)
                if 0.5 < fv < 30.0:
                    fwhm_px = fv
                    break
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0221] VY_FWHM/VY_FWHM_GAUSS header parse fails - pipeline uses default/config FWHM for phase0+1: %s', exc)
            pass

    # -- FAZA 0+1 --
    _p("Faza 0+1: select targets + comparison stars...")
    _plate_scale = _get_plate_scale_from_cfg(
        _cfg,
        db=db,
        draft_id=draft_id,
        fits_path=Path(masterstar_fits_path),
        ms_header=_ms_header_shared,
    )
    if _plate_scale is None:
        _plate_scale = _read_plate_scale_from_fits_path(
            Path(masterstar_fits_path),
            ms_header=_ms_header_shared,
        )
        if _plate_scale is not None and math.isfinite(float(_plate_scale)) and float(_plate_scale) > 0:
            logging.info(
                "[FOV] plate_scale from MASTERSTAR.fits header -> %.4f arcsec/px",
                float(_plate_scale),
            )
    _fw_pipe, _fh_pipe, _frame_hw_src = _resolve_frame_hw_px_from_masterstar(
        Path(masterstar_fits_path),
        frame_w_px=int(_cfg.frame_width_px),
        frame_h_px=int(_cfg.frame_height_px),
        db=db,
        draft_id=draft_id,
    )
    if _frame_hw_src != "caller_default":
        logging.info(
            "[PHASE 0+1] Pipeline frame dimensions %dx%d px from %s",
            int(_fw_pipe),
            int(_fh_pipe),
            _frame_hw_src,
        )
    _pt_mags = _cfg.phase01_tier_mags()
    p01 = run_phase0_and_phase1(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        per_frame_csv_dir=Path(per_frame_csv_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        frame_w_px=int(_fw_pipe),
        frame_h_px=int(_fh_pipe),
        chip_interior_margin_px=_resolve_chip_interior_margin_px(_cfg, arcsec_per_px=_plate_scale),
        plate_scale_arcsec_px=_plate_scale,
        max_dist_deg=_compute_fov_max_dist(
            frame_w_px=int(_fw_pipe),
            frame_h_px=int(_fh_pipe),
            plate_scale=_plate_scale,
            fov_fraction=float(_cfg.phase01_comparison_fov_fraction),
            fallback_deg=resolve_max_dist_fallback_deg(
                _cfg,
                frame_w_px=int(_fw_pipe),
                frame_h_px=int(_fh_pipe),
                plate_scale_arcsec_px=_plate_scale,
            ),
        ),
        max_mag_diff=float(_cfg.phase01_comparison_max_mag_diff),
        comp_max_delta_bprp=float(_cfg.comp_max_delta_bprp),
        max_mag_diff_t1=float(_pt_mags[0]),
        max_mag_diff_t2=float(_pt_mags[1]),
        max_mag_diff_t3=float(_pt_mags[2]),
        max_mag_diff_t4=float(_pt_mags[3]),
        n_comp_min=int(_cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
        max_comp_rms=float(_cfg.phase01_comparison_max_comp_rms),
        min_dist_arcsec=float(_cfg.phase01_comparison_min_dist_arcsec),
        min_frames_frac=float(_cfg.phase01_comparison_min_frames_frac),
        rms_outlier_sigma=3.0,
        exclude_gaia_nss=bool(_cfg.phase01_comparison_exclude_gaia_nss),
        exclude_gaia_extobj=bool(_cfg.phase01_comparison_exclude_gaia_extobj),
        mag_bright_threshold=float(_cfg.phase01_comparison_mag_bright_threshold),
        max_mag_diff_bright_floor=float(
            _cfg.phase01_comparison_max_mag_diff_bright_floor or 0.0
        ),
        max_psf_chi2=float(_cfg.phase01_comparison_max_psf_chi2),
        max_fwhm_factor=float(_cfg.phase01_comparison_max_fwhm_factor),
        isolation_radius_px=_resolve_isolation_radius_px(_cfg, arcsec_per_px=_plate_scale),
        flux_col=_cfg.phase01_flux_col,
        cfg=_cfg,
        progress_cb=progress_cb,
        draft_id=draft_id,
        db=db,
    )

    active_targets_csv = Path(str(p01.get("active_targets_csv") or ""))
    comparison_stars_csv = Path(str(p01.get("comparison_stars_csv") or ""))
    n_active = int(p01.get("n_active_targets") or 0)
    if n_active <= 0:
        return {
            "phase01": p01,
            "phase2a": None,
            "output_dir": str(Path(output_dir)),
            "zero_targets": True,
            "n_active_targets": 0,
        }
    if not active_targets_csv.is_file() or not comparison_stars_csv.is_file():
        return {
            "phase01": p01,
            "phase2a": None,
            "output_dir": str(Path(output_dir)),
            "error": "Faza 0+1 nevygenerovala active_targets/comparison_stars CSV.",
        }

    # INV-DAG-01: phase01 stamp after successful Phase 0+1.
    try:
        from invariants_runtime import stamp_stage_on_disk  # noqa: PLC0415

        stamp_stage_on_disk(Path(output_dir), "phase01", enforce_upstream=True)
    except Exception as _dag_p01_exc:  # noqa: BLE001
        logging.debug("[INV-DAG-01] phase01 stamp skipped: %s", _dag_p01_exc)

    # -- FAZA 2A --
    _p("Faza 2A: aperture photometry + lightcurves...")
    _cfg2a = p01.get("cfg_effective_for_photometry") or _cfg
    p2a = run_phase2a(
        masterstar_fits_path=Path(masterstar_fits_path),
        active_targets_csv=active_targets_csv,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=Path(per_frame_csv_dir),
        detrended_aligned_dir=Path(detrended_aligned_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        annulus_inner_fwhm=float(_cfg.annulus_inner_fwhm),
        annulus_outer_fwhm=float(_cfg.annulus_outer_fwhm),
        cfg=_cfg2a,
        progress_cb=progress_cb,
        db=db,
        draft_id=draft_id,
        proc_frame_store=p01.get("proc_store"),
    )

    sysrem_result: dict[str, Any] | None = None
    if bool(_cfg.sysrem_enabled):
        _p("SysRem: removing systematic trends...")
        _sysrem_lc_dir = Path(output_dir) / "lightcurves"
        sysrem_result = run_sysrem_field(
            _sysrem_lc_dir,
            n_iter=int(_cfg.sysrem_n_iter),
        )
        logging.info(
            "[SysRem] %d stars | RMS improvement %.1f%% (%d iter)",
            int(sysrem_result.get("n_stars", 0)),
            float(sysrem_result.get("rms_improvement_pct", float("nan"))),
            int(sysrem_result.get("n_iter", 0)),
        )

    return {
        "phase01": p01,
        "phase2a": p2a,
        "sysrem": sysrem_result,
        "output_dir": str(Path(output_dir)),
        "proc_frame_store": p01.get("proc_store"),
    }




__all__ = [
    # photometry (legacy)
    "StressTestResult",
    "_get_lc_adaptive",
    "apply_reporting_postprocess",
    "check_comparison_stability",
    "common_field_intersection_bbox_px",
    "compute_aperture_correction",
    "compute_fwhm_gaussian_for_aperture_catalog",
    "compute_lc_rms_ooe",
    "compute_mag_calib_final",
    "compute_optimal_apertures",
    "detect_outliers",
    "empirical_feature_mask_mag",
    "enhance_catalog_dataframe_aperture_bpm",
    "ensemble_normalize",
    "ensure_full_variable_targets_if_presel_stub",
    "load_epsf_metrics_for_draft",
    # photometry_phase2a (legacy)
    "measure_fwhm_from_masterstar",
    "pytics_iterative_weights",
    "read_flux_from_csv",
    "recommended_aperture_by_color",
    "resolve_apply_color_term",
    "run_full_photometry_pipeline",
    "run_phase0_and_phase1",
    "run_phase2a",
    "run_sysrem_field",
    "save_cutout_png",
    "save_field_map_png",
    "save_lightcurve_csv",
    "save_lightcurve_png",
    "save_target_field_map_png",
    "select_active_targets",
    "select_comparison_stars_per_target",
    "stress_test_relative_rms_from_sidecars",
    "vsx_is_known_variable_top3_per_bin",
]

from photometry_ui_helpers import (  # noqa: E402,F401
    resolve_lc_time_base,
    lc_time_axis_short_label,
)

from photometry_shared import (  # noqa: E402,F401
    _safe_polyfit,
    _normalize_gaia_id,
    finalize_hybrid_bkg_fallback_proc_dir,
    stamp_masterstar_snr_columns,
    _target_display_name,
    stamp_vsx_known_variable_on_masterstars,
    build_gs11_summary,
    _get_lc_adaptive,
    _get_plate_scale_from_cfg,
    _resolve_plate_scale_arcsec_per_px,
    _cd_matrix_scale_arcsec_per_px,
    _read_plate_scale_from_fits_path,
    _angular_distance_deg,
    StressTestResult,
    stress_test_relative_rms_from_sidecars,
    vsx_is_known_variable_top3_per_bin,
    common_field_intersection_bbox_px_from_arrays,
    common_field_intersection_bbox_px,
    recommended_aperture_by_color,
    bad_columns_for_light_frame,
    _fwhm_moment_at,
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture_bpm,
    _finite_pixel_bbox_from_array,
    _intersection_bbox_from_frame_bboxes,
    _aperture_flux_sky_batch,
    compute_per_frame_cog_correction,

)

import sys as _sys_e4
_p2a_mod = _sys_e4.modules.get("photometry_phase2a")
if _p2a_mod is None or getattr(_p2a_mod, "run_phase2a", None) is not None:
    from photometry_phase2a import (  # noqa: E402,F401
    parse_comp_quality_json_map,
    _build_csv_lookup,
    _lookup_star_in_csv,
    _sat_limit_peak_adu,
    _mad_sigma_or_std_floor,
    measure_fwhm_from_masterstar,
    compute_optimal_apertures,
    _howell_variance_adu2,
    _photometric_error,
    _photometric_error_with_bkg_mode,
    _phase2a_proc_column_requirements,
    _phase2a_cache_columns,
    _phase2a_empirical_sigma_bkg_ap,
    _sky_pp_for_photometric_error,
    _resolve_phase2a_equipment_id,
    _draft_dir_from_phase2a_paths,
    _require_comparison_stars_per_target_schema,
    _median_sky_from_phase2a_csv_cache,
    _measured_aperture_from_proc_cache,
    _resolve_photometric_aperture_px_for_gs11,
    read_flux_from_csv,
    compute_aperture_correction,
    fit_color_term_c1,
    should_apply_color_term,
    _obs_group_filter_key,
    resolve_apply_color_term,
    _ColorTermGroupFit,
    _group_comp_mag_inst_from_flux_matrix,
    _group_comp_mag_inst_from_proc_csvs,
    _comp_maps_from_comparison_stars_csv,
    _phase2a_attempt_k2_night_fit,
    _compute_group_color_term_fit,
    _ensure_group_comp_pool_csv,
    _target_row_is_vsx_known_variable,
    empirical_feature_mask_mag,
    detect_outliers,
    apply_reporting_postprocess,
    democratic_detrend_lc,
    save_field_map_png,
    _edge_ok_from_masterstar_pipeline,
    resolve_variable_targets_csv,
    auto_export_variability_candidates_csv,
    _phase2a_coerce_skip_photometry,
    build_rms_mag_model,
    expected_rms_from_model,
    classify_lc_quality,
    build_lc_quality_summary,
    _phase2a_write_summary,
    _phase2a_observer_location_dict,
    _sky_surface_meta_from_qc,
    _phase2a_resolve_field_center_ra_dec,
    _phase2a_collect_session_jd_values,
    _Phase2AState,
    _build_phase2a_dynamic_params,
    _phase2a_compute_lunar_context,
    _preserve_nondetection_flags_helper,
    _proc_stem,
    _compute_frame_align_residuals,
    _record_align_residuals_to_report,
    _frame_align_residual_gate_select,
    _propagate_phase2a_skip_reason_to_active,
    _phase2a_finalize_exports,
        run_phase2a,
    )

from photometry_lightcurve import (  # noqa: E402,F401
    _ac_summary_fields,
    _phase2a_empty_comp_summary_row,
    _phase2a_skip_empty_comps_target,
    _coerce_bool_cell,
    _frame_has_usable_cog,
    evaluate_cog_night_apcorr_gate,
    temporal_bin_comp_lc,
    pytics_iterative_weights,
    _common_mode_detrend_comp_lc,
    _comp_lc_frame_ensemble_residual,
    compute_lc_rms_ooe,
    check_comparison_stability,
    ensemble_normalize,
    _ensemble_scatter_by_source_file,
    _combine_err_with_ensemble_scatter_keyed,
    _err_budget_components_keyed,
    _exclude_err_scatter_unmatched_epochs,
    ct_ensemble_reference_maps,
    apply_color_term,
    _check_color_term_extrapolation,
    _ct_prototype_enabled,
    _color_term_cat_inst_scatter_pair,
    _append_ct_prototype_row,
    savgol_detrend_lc,
    compute_mag_calib_final,
    save_lightcurve_csv,
    save_lightcurve_png,
    save_cutout_png,
    save_target_field_map_png,
    pfs_rescue_eligible,
    _keep_recorded_skip_reason,
    decide_target_saturation_policy,
    _per_frame_sat_flags_for_catalog_id,
    _resolve_pfs_peak_test,
    apply_per_frame_saturation_to_active_targets,
    _fits_header_facts,
    _build_phase2a_resolved_facts,
    BlendMapEntry,
    _load_blend_worklist,
    _load_adaptive_blend_map,
    _route_lc_per_frame_err,
    _get_lc,
    _get_comp_bjd_series,
    compute_lc_flux_method,
    _recompute_bjd_hjd_with_status,
    run_sysrem_field,
)

from photometry_gate_helpers import (  # noqa: E402,F401
    _sigma_bkg_r_key,
    _assert_inv_err_sigma_acct_01,
    comp_quality_quality_strings,
    _clamp_err_empty_apertures_n,
    _normalize_err_background_mode,
    _labbe_content_seed_from_header,
    measure_empty_aperture_sigma_bkg,
    estimate_star_free_per_pixel_variance_adu2,
    _howell_bkg_variance_adu2,
    _clamp_bkg_scale_r,
    bkg_scale_ratio_empirical_over_howell,
    compute_setup_bkg_scale_r,
    scaled_sigma_bkg_ap_from_howell,
    measure_growth_curve_ee,
    _phase2a_star_mag_lookup,
    discover_aligned_science_fits,
    _median_bkg_var_from_aligned_frames,
    _estimate_annulus_sky_pp,
    _annulus_sky_subtracted_flux,
    _resolve_star_flux_method,
    _frame_quality_gate_select,
    _recompute_bjd_hjd_per_target,
    photometer_check_star_production_path,
    _compute_fov_max_dist,
    _sky_pp_from_annulus_image,
    _aperture_flux_sky_per_star,
    _clamp_err_empty_apertures_min,
    _robust_scatter_mad,
    _build_star_exclusion_mask,
    _canonicalize_star_xy,
    _labbe_debug_dump_enabled,
    _labbe_debug_dump_path,
    _labbe_append_debug_record,
)

from photometry_exports import (  # noqa: E402,F401
    lc_has_finite_airmass,
    apply_comp_w_rel_for_display,
    ensemble_member_ids,
    _get_lc_psf_strict,
    _get_lc_adaptive_per_star,
    _get_lc_star_method,

)

from epsf_hooks import load_epsf_metrics_for_draft  # noqa: E402,F401

from photometry_provenance import (  # noqa: E402,F401
    _is_import_relevant_py_path,
    _porcelain_status_by_path,
    classify_git_dirty_paths,
    _resolve_git_provenance,
    _json_safe_snapshot_value,
    _complete_config_snapshot,
    _build_pipeline_provenance_block,
    merge_photometry_pipeline_meta,
)

from photometry_comp import (  # noqa: E402,F401
    _sid_int,
    _enrich_comp_bp_rp,
    _ensure_active_target_display_names,
    _variable_targets_looks_like_ct_presel_stub,
    ensure_full_variable_targets_if_presel_stub,
    _normalize_id_value,
    _normalize_id_series,
    _bool_col,
    _phase0_effective_frame_hw_px,
    _active_target_zone_flag,
    _auto_repair_catalog_ids,
    _enrich_active_targets_bp_rp,
    _resolve_frame_hw_px_from_masterstar,
    _read_field_density_inputs,
    _refresh_variable_targets_xy,
    _attach_predicted_dilution_report,
    select_active_targets,
    _batch_enrich_targets_bp_rp_from_gaia_db,
    _enrich_target_bp_rp_from_gaia_db,
    _bprp_tier_ladder_for_selection,
    _select_comps_by_rms_then_color,
    _select_comps_by_color_then_rms,
    _select_comps_tiered,
    build_global_comp_pool,
    _dedupe_comp_pool_by_gaia_key,
    _warn_zero_compstars_edge,
    _count_gate_passing_comps,
    select_comparison_stars_per_target,
    _write_suspected_variables,
)

from phase01_run import run_phase0_and_phase1  # noqa: E402,F401


def __getattr__(name: str) -> Any:
    """PEP 562: LAST_EXCLUDED_TARGETS lives in photometry_comp (D-FACADE-PERMANENT-01)."""
    if name == "LAST_EXCLUDED_TARGETS":
        import photometry_comp as _photometry_comp

        return _photometry_comp.LAST_EXCLUDED_TARGETS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
