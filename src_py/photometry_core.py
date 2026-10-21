"""Photometry core - zluceny modul (photometry + photometry_phase2a)."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any


from config import (
    AppConfig,
)
from gaia_catalog_id import (
    GAIA_PROC_CSV_READ_DTYPE,
)


LOGGER = logging.getLogger(__name__)

_MAD_CONSISTENCY = 0.6745  # normalizacny faktor MAD -> sigma ekvivalent

# Explicit annulus sky (ADU/px) for Howell err; ``noise_floor_adu`` remains detection-floor legacy.
SKY_ADU_PER_PX_ANNULUS_COL = "sky_adu_per_px_annulus"
SKY_SURFACE_BG_MEDIAN_ADU_COL = "sky_surface_bg_median_adu"

# F-BINGAIN-1: empirical background noise (empty-aperture scatter) + provenance.
SIGMA_BKG_AP_COL = "sigma_bkg_ap"
ERR_BKG_SOURCE_COL = "err_bkg_source"
ERR_BKG_SOURCE_EMPIRICAL = "empirical"
ERR_BKG_SOURCE_HOWELL_FALLBACK = "howell_fallback"
ERR_BKG_SOURCE_HOWELL_SCALED = "howell_scaled"
BKG_SCALE_R_CLAMP_LO = 0.05
BKG_SCALE_R_CLAMP_HI = 2.0


# Per-target LC time provenance (F-BJD-1): labels BJD recompute path, does not alter time values.
TIME_BASE_COL = "time_base"
TIME_BASE_BJD_TDB = "BJD_TDB"
TIME_BASE_JD_FALLBACK = "JD_FALLBACK"


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
        "aperture_correction",  # legacy key in pre-2026-10 comp_quality JSON (AC removed)
        "qa_degraded",
        "qa_degraded_reason",
        "broeg_weights",  # EXPORT-2026-10: normalised Broeg (2005) ensemble weights
        "comp_band_mag",  # catalogue magnitudes in the zero-point band
        "band_system",
        "zp_band_label",
        "zero_point",  # ZP-ERR-2026-10: comps setting the absolute level + its uncertainty
        "gspc_zero_point",
        "comp_smart",
        "check_star",  # OUTPUT-REVIEW-2026-10 step 6: the check star and its trust statistics
        "color_fallback",  # legacy (tier copies before OUTPUT-REVIEW step 6)
        "max_delta_bprp_used",  # legacy
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
# KROK 3: Stability check porovnavaciek (Abbeho p2p scatter + MAD)
# ---------------------------------------------------------------------------


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


#: TARGET-RULES-2026-10 (Milan 2026-10-13): target skips decided on MASTERSTAR in Phase 0
#: (saturated = upper cut, below_snr_min = lower cut, VSX type out of scope) are final:
#: no comparison selection in Phase 1, never measured, never revived later.
FINAL_SKIP_REASONS = frozenset({"saturated", "below_snr_min", "vsx_type_out_of_scope"})


_LC_QUALITY_FLAGS: tuple[str, ...] = (
    "good",
    "noisy",
    "noisy_moon",
    "short_baseline",
    "no_data",
    "saturated",
)


# src_py/photometry_core.py -> repo root is parent.parent (git cwd + porcelain path base).
_REPO_ROOT_FOR_PROVENANCE = Path(__file__).resolve().parent.parent


from mag_constants import MAG_ERR_SCALE

_PSF_ERR_MAG_SCALE = MAG_ERR_SCALE


# ======================================================================
# photometry.py (zlucene do photometry_core)
# ======================================================================


# Stlpce nacitavane z per-frame CSV pre bootstrap (78 % uspora pamate)
_PHASE_USECOLS_PERFRAME: list[str] = [
    "catalog_id",
    "bjd_tdb_mid",
    "dao_flux",
    "sky_adu_per_px_annulus",
    "aperture_r_px",
    "is_saturated",
    "is_noisy",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
    # APERTURE-K-2026-10: comparison-star RMS on the light-curve aperture grid
    "flux_k100",
    "flux_k125",
    "flux_k150",
    "flux_k175",
    "flux_k200",
]


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
    fwhm_px = float(_cfg.sips_dao_fwhm_px)
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
    p01 = run_phase0_and_phase1(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        per_frame_csv_dir=Path(per_frame_csv_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        frame_w_px=int(_fw_pipe),
        frame_h_px=int(_fh_pipe),
        chip_interior_margin_px=int(_cfg.phase01_chip_interior_margin_px),
        plate_scale_arcsec_px=_plate_scale,
        comp_max_delta_bprp=float(_cfg.comp_max_delta_bprp),
        n_comp_min=int(_cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
        max_comp_rms=float(_cfg.phase01_comparison_max_comp_rms),
        min_dist_arcsec=float(_cfg.phase01_comparison_min_dist_arcsec),
        min_frames_frac=float(_cfg.phase01_comparison_min_frames_frac),
        isolation_radius_px=float(_cfg.phase01_comparison_isolation_radius_px),
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
    _cfg2a = _cfg
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

    return {
        "phase01": p01,
        "phase2a": p2a,
        "output_dir": str(Path(output_dir)),
        "proc_frame_store": p01.get("proc_store"),
    }


__all__ = [
    # photometry (legacy)
    "StressTestResult",
    "apply_reporting_postprocess",
    "check_comparison_stability",
    "common_field_intersection_bbox_px",
    "compute_fwhm_gaussian_for_aperture_catalog",
    "compute_lc_rms_ooe",
    "compute_optimal_apertures",
    "detect_outliers",
    "empirical_feature_mask_mag",
    "enhance_catalog_dataframe_aperture",
    "ensemble_normalize",
    "ensure_full_variable_targets_if_presel_stub",
    # photometry_phase2a (legacy)
    "measure_fwhm_from_masterstar",
    "read_flux_from_csv",
    "recommended_aperture_by_color",
    "resolve_apply_color_term",
    "run_full_photometry_pipeline",
    "run_phase0_and_phase1",
    "run_phase2a",
    "save_lightcurve_csv",
    "save_lightcurve_png",
    "select_active_targets",
    "select_comparison_stars_per_target",
    "stress_test_relative_rms_from_sidecars",
    "vsx_is_known_variable_top3_per_bin",
]

from photometry_ui_helpers import resolve_lc_time_base  # noqa: E402,F401

from photometry_shared import (
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
    StressTestResult,
    stress_test_relative_rms_from_sidecars,
    vsx_is_known_variable_top3_per_bin,
    common_field_intersection_bbox_px_from_arrays,
    common_field_intersection_bbox_px,
    recommended_aperture_by_color,
    _fwhm_moment_at,
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture,
)

import sys as _sys_e4
_p2a_mod = _sys_e4.modules.get("photometry_phase2a")
if _p2a_mod is None or getattr(_p2a_mod, "run_phase2a", None) is not None:
    from photometry_phase2a import (  # noqa: E402,F401
        parse_comp_quality_json_map,
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
        _resolve_photometric_aperture_px_for_gs11,
        read_flux_from_csv,
        fit_color_term_c1,
        should_apply_color_term,
        resolve_apply_color_term,
        _group_comp_mag_inst_from_proc_csvs,
        empirical_feature_mask_mag,
        detect_outliers,
        apply_reporting_postprocess,
        build_rms_mag_model,
        expected_rms_from_model,
        classify_lc_quality,
        build_lc_quality_summary,
        _Phase2AState,
        _propagate_phase2a_skip_reason_to_active,
        run_phase2a,
    )

from photometry_lightcurve import (
    _phase2a_skip_empty_comps_target,
    compute_lc_rms_ooe,
    check_comparison_stability,
    ensemble_normalize,
    _ensemble_scatter_by_source_file,
    _combine_err_with_ensemble_scatter_keyed,
    ct_ensemble_reference_maps,
    apply_color_term,
    _check_color_term_extrapolation,
    _color_term_cat_inst_scatter_pair,
    save_lightcurve_csv,
    save_lightcurve_png,
    _per_frame_sat_flags_for_catalog_id,
    apply_per_frame_saturation_to_active_targets,
    decide_per_frame_saturation,
    _build_phase2a_resolved_facts,
    _recompute_bjd_hjd_with_status,
)

from photometry_gate_helpers import (
    _sigma_bkg_r_key,
    _assert_inv_err_sigma_acct_01,
    comp_quality_quality_strings,
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
    _variable_targets_looks_like_ct_presel_stub,
    ensure_full_variable_targets_if_presel_stub,
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

from phase01_run import run_phase0_and_phase1  # noqa: E402,F401


