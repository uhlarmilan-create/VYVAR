"""Photometry facade: ``run_full_photometry_pipeline`` plus re-exports for dev/ scripts (D-FACADE-PERMANENT-01).

Pipeline modules import from the defining modules, never from here (LAYER-2026-10).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any


from config import (
    AppConfig,
)


# LAYER-2026-10: names, labels and the logger live in the leaf ``photometry_defs``, physical constants
# in ``pipeline_constants``; this facade re-exports them for dev/ scripts (D-FACADE-PERMANENT-01).
from photometry_defs import (  # noqa: F401
    COMP_QUALITY_JSON_META_KEYS,
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    ERR_BKG_SOURCE_HOWELL_FALLBACK,
    ERR_BKG_SOURCE_HOWELL_SCALED,
    FINAL_SKIP_REASONS,
    LC_QUALITY_FLAGS,
    LOGGER,
    PHASE_USECOLS_PERFRAME,
    SIGMA_BKG_AP_COL,
    SKY_ADU_PER_PX_ANNULUS_COL,
    SKY_SURFACE_BG_MEDIAN_ADU_COL,
    TIME_BASE_BJD_TDB,
    TIME_BASE_COL,
    TIME_BASE_JD_FALLBACK,
)
from pipeline_constants import BKG_SCALE_R_CLAMP_HI, BKG_SCALE_R_CLAMP_LO, MAD_CONSISTENCY  # noqa: F401
from mag_constants import MAG_ERR_SCALE

_PSF_ERR_MAG_SCALE = MAG_ERR_SCALE


def run_full_photometry_pipeline(
    *,
    masterstar_fits_path: Path,
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    detrended_aligned_dir: Path,
    output_dir: Path,
    cfg: AppConfig,
    db: Any = None,
    draft_id: int | None = None,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Single-step wrapper: target and comparison-star selection + aperture photometry as one unit.

    The UI uses it as a single 'RUN Aperture Photometry' action for the given obs_group.
    """
    _cfg = cfg


    def _p(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    # FWHM: MASTERSTAR header VY_FWHM, otherwise the config default.
    from pipeline_constants import DAO_FWHM_FALLBACK_PX  # noqa: PLC0415

    fwhm_px = DAO_FWHM_FALLBACK_PX
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
            for key in ("VY_FWHM",):
                v = _ms_header_shared.get(key)
                if v is None:
                    continue
                fv = float(v)
                if 0.5 < fv < 30.0:
                    fwhm_px = fv
                    break
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0221] VY_FWHM header parse fails - pipeline uses default/config FWHM for target and comparison-star selection: %s', exc)
            pass

    # -- target and comparison-star selection --
    _p("Target and comparison-star selection...")
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
    p01 = run_target_and_comp_selection(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        per_frame_csv_dir=Path(per_frame_csv_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        plate_scale_arcsec_px=_plate_scale,
        comp_max_delta_bprp=float(_cfg.comp_max_delta_bprp),
        n_comp_min=int(_cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
        max_comp_rms=float(_cfg.phase01_comparison_max_comp_rms),
        min_frames_frac=float(_cfg.phase01_comparison_min_frames_frac),
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
            "target_comp_selection": p01,
            "aperture_photometry": None,
            "output_dir": str(Path(output_dir)),
            "zero_targets": True,
            "n_active_targets": 0,
        }
    if not active_targets_csv.is_file() or not comparison_stars_csv.is_file():
        return {
            "target_comp_selection": p01,
            "aperture_photometry": None,
            "output_dir": str(Path(output_dir)),
            "error": "Target and comparison-star selection did not produce active_targets/comparison_stars CSV.",
        }

    # INV-DAG-01: target_comp_selection stamp after successful target and comparison-star selection.
    try:
        from invariants_runtime import stamp_stage_on_disk  # noqa: PLC0415

        stamp_stage_on_disk(Path(output_dir), "target_comp_selection", enforce_upstream=True)
    except Exception as _dag_p01_exc:  # noqa: BLE001
        logging.debug("[INV-DAG-01] target_comp_selection stamp skipped: %s", _dag_p01_exc)

    # -- aperture photometry --
    _p("Aperture photometry + light curves...")
    _cfg2a = _cfg
    p2a = run_aperture_photometry(
        masterstar_fits_path=Path(masterstar_fits_path),
        active_targets_csv=active_targets_csv,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=Path(per_frame_csv_dir),
        detrended_aligned_dir=Path(detrended_aligned_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        cfg=_cfg2a,
        progress_cb=progress_cb,
        db=db,
        draft_id=draft_id,
        proc_frame_store=p01.get("proc_store"),
    )

    return {
        "target_comp_selection": p01,
        "aperture_photometry": p2a,
        "output_dir": str(Path(output_dir)),
        "proc_frame_store": p01.get("proc_store"),
    }


__all__ = [
    # photometry (legacy)
    "apply_reporting_postprocess",
    "check_comparison_stability",
    "common_field_intersection_bbox_px",
    "compute_fwhm_gaussian_for_aperture_catalog",
    "compute_lc_rms_ooe",
    "compute_optimal_apertures",
    "empirical_feature_mask_mag",
    "enhance_catalog_dataframe_aperture",
    "ensemble_normalize",
    # aperture_photometry (legacy)
    "measure_fwhm_from_masterstar",
    "read_flux_from_csv",
    "resolve_apply_color_term",
    "run_full_photometry_pipeline",
    "run_target_and_comp_selection",
    "run_aperture_photometry",
    "save_lightcurve_csv",
    "save_lightcurve_png",
    "select_active_targets",
    "select_comparison_stars_per_target",
]

from photometry_lightcurve import resolve_lc_time_base  # noqa: E402,F401

from photometry_shared import (
    _sky_pp_for_photometric_error,
    _safe_polyfit,
    _normalize_gaia_id,
    finalize_hybrid_bkg_fallback_proc_dir,
    stamp_masterstar_snr_columns,
    _target_display_name,
    stamp_vsx_known_variable_on_masterstars,
    build_gs11_summary,
    _get_plate_scale_from_cfg,
    _resolve_plate_scale_arcsec_per_px,
    _read_plate_scale_from_fits_path,
    common_field_intersection_bbox_px_from_headers,
    common_field_intersection_bbox_px,
    _fwhm_moment_at,
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture,
)

from aperture_photometry import (  # noqa: E402,F401
    _mad_sigma_or_std_floor,
    measure_fwhm_from_masterstar,
    compute_optimal_apertures,
    _aperture_proc_column_requirements,
    _aperture_cache_columns,
    _aperture_empirical_sigma_bkg_ap,
    _resolve_aperture_equipment_id,
    _draft_dir_from_aperture_paths,
    _require_comparison_stars_per_target_schema,
    _resolve_photometric_aperture_px_for_gs11,
    read_flux_from_csv,
    fit_color_term_c1,
    _group_comp_mag_inst_from_proc_csvs,
    empirical_feature_mask_mag,
    apply_reporting_postprocess,
    build_rms_mag_model,
    expected_rms_from_model,
    classify_lc_quality,
    build_lc_quality_summary,
    _AperturePhotometryState,
    _propagate_aperture_skip_reason_to_active,
    run_aperture_photometry,
)
from band_classify import should_apply_color_term, resolve_apply_color_term  # noqa: E402,F401
from photometry_gate_helpers import (  # noqa: E402,F401
    _howell_variance_adu2,
    _photometric_error,
    _photometric_error_with_bkg_mode,
)
from comp_table import parse_comp_quality_json_map  # noqa: E402,F401

from photometry_lightcurve import (
    _aperture_skip_empty_comps_target,
    compute_lc_rms_ooe,
    check_comparison_stability,
    ensemble_normalize,
    _ensemble_scatter_by_source_file,
    _combine_err_with_ensemble_scatter_keyed,
    ct_ensemble_reference_maps,
    apply_color_term,
    _check_color_term_extrapolation,
    save_lightcurve_csv,
    save_lightcurve_png,
    _per_frame_sat_flags_for_catalog_id,
    apply_per_frame_saturation_to_active_targets,
    decide_per_frame_saturation,
    _build_aperture_resolved_facts,
    _recompute_bjd_hjd_with_status,
)

from photometry_gate_helpers import (
    _sigma_bkg_r_key,
    _assert_inv_err_sigma_acct_01,
    _clamp_err_empty_apertures_n,
    _labbe_content_seed_from_header,
    measure_empty_aperture_sigma_bkg,
    _howell_bkg_variance_adu2,
    _clamp_bkg_scale_r,
    bkg_scale_ratio_empirical_over_howell,
    compute_setup_bkg_scale_r,
    scaled_sigma_bkg_ap_from_howell,
    _annulus_sky_subtracted_flux,
    _aperture_flux_sky_per_star,
)
from comp_table import comp_quality_quality_strings

from photometry_exports import (  # noqa: E402,F401
    lc_has_finite_airmass,
    apply_comp_w_rel_for_display,

)


from photometry_provenance import (
    _is_import_relevant_py_path,
    classify_git_dirty_paths,
    _resolve_git_provenance,
    _complete_config_snapshot,
    _build_pipeline_provenance_block,
    merge_photometry_pipeline_meta,
)

from photometry_comp import (
    _enrich_comp_bp_rp,
    _normalize_id_value,
    _normalize_id_series,
    _bool_col,
    _resolve_frame_hw_px_from_masterstar,
    select_active_targets,
    _bprp_tier_ladder_for_selection,
    _select_comps_by_rms_then_color,
    build_global_comp_pool,
    _dedupe_comp_pool_by_gaia_key,
    _warn_zero_compstars_edge,
    _count_gate_passing_comps,
    select_comparison_stars_per_target,
)

from target_comp_selection import run_target_and_comp_selection  # noqa: E402,F401
from aperture_photometry_state import _aperture_prepare_shared_state  # noqa: E402,F401
from aperture_photometry_target import _aperture_process_one_target  # noqa: E402,F401


