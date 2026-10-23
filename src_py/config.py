"""Project configuration for the variable-star processing system."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields as _dc_fields
import difflib
import json
import logging
import math
import os
import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import Any


def is_git_dev_checkout(install_root: Path) -> bool:
    """True when running from a git clone (dev/CI path uses install tree as data root)."""
    return (install_root / ".git").is_dir()


def default_release_data_dir() -> Path:
    """Default user data directory for bundled installs (RELEASE-2 / R1)."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        if local:
            return (Path(local) / "VYVAR").resolve()
    return (Path.home() / ".local" / "share" / "vyvar").resolve()


def resolve_data_root(install_root: Path) -> Path:
    """Resolve the user data root (config, DB, Archive, catalogs).

    Precedence: ``VYVAR_DATA_DIR`` env -> git dev checkout (install root) ->
    bundled install (platform default) when bundle markers/env present ->
    install root (tests and explicit ``project_root`` overrides).
    """
    override = os.environ.get("VYVAR_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    root = install_root.resolve()
    if is_git_dev_checkout(root):
        return root
    bundle_flag = os.environ.get("VYVAR_RELEASE_BUNDLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    bundle_layout = (root / "RUNTIME_PIN.json").is_file() or (
        root / "python" / "python.exe"
    ).is_file() or (root / "python" / "bin" / "python3").is_file()
    if bundle_flag or bundle_layout:
        return default_release_data_dir()
    return root


def resolve_config_path(raw: str | Path, data_root: Path) -> str:
    """Resolve a config path: absolute as-is, relative against ``data_root`` (never CWD)."""
    s = str(raw or "").strip()
    if not s:
        return ""
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = data_root / p
    return str(p.resolve())


def config_json_path(project_root: Path) -> Path:
    return project_root / "config.json"


# CONFIG-HUMAN-EDIT STEP 2: legacy config.json keys the loader migrates silently. These are
# NOT AppConfig fields, so the unknown-key typo warning must skip them (they have explicit
# migration paths in __post_init__). Two families: uppercase env-style aliases superseded by
# lowercase fields, and the WAVE-B scalar tier/aperture keys merged into structured keys.
# CFG-REVIEW-2026-10: keys renamed (old -> new); the old name still loads, with an INFO.
RENAMED_KEYS: dict[str, str] = {
    "snr_cog_isolation_fwhm": "comp_isolation_fwhm",
}


# Keys removed from AppConfig; load_config_json logs INFO once per key (no difflib WARN).
KNOWN_REMOVED_KEYS: dict[str, str] = {
    "gain": (
        "gain removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): gain comes from the FITS header or EQUIPMENTS.GAIN_ADU; unknown -> error (no default)"
    ),
    "phase01_chip_interior_margin_px": (
        "phase01_chip_interior_margin_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): the edge margin is the sky-annulus radius (annulus_outer_fwhm x FWHM) or the annulus-aware safe box"
    ),
    "phase01_comparison_isolation_radius_px": (
        "phase01_comparison_isolation_radius_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): the crude neighbour-flux contamination index it sized was removed; neighbour light is the comparison pool's aperture blend model"
    ),
    "phase01_suspected_interior_margin_px": (
        "phase01_suspected_interior_margin_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): the edge margin is the sky-annulus radius (annulus_outer_fwhm x FWHM) or the annulus-aware safe box"
    ),
    "read_noise": (
        "read_noise removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): read noise comes from the FITS header or EQUIPMENTS.READNOISE_E; unknown -> error (no default)"
    ),
    "sips_dao_fwhm_px": (
        "sips_dao_fwhm_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): the DAO FWHM is the measured VY_FWHM of each frame; without one the fixed fallback pipeline_constants.DAO_FWHM_FALLBACK_PX = 3 px is used"
    ),
    "alignment_max_control_points": (
        "alignment_max_control_points removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.ALIGNMENT_MAX_CONTROL_POINTS = 80); alignment_max_stars stays"
    ),
    "check_select_rms_floor": (
        "check_select_rms_floor removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (check_star_kmag.CHECK_SELECT_RMS_FLOOR = 1e-4)"
    ),
    "check_star_min_epochs": (
        "check_star_min_epochs removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (trust_flag_core.CHECK_STAR_MIN_EPOCHS = 5)"
    ),
    "comp_contamination_penalty_k": (
        "comp_contamination_penalty_k removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (comp_selection_per_target.COMP_CONTAMINATION_PENALTY_K = 3)"
    ),
    "comp_rms_loo_photon_k": (
        "comp_rms_loo_photon_k removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (comp_rms_loo.COMP_RMS_LOO_PHOTON_K_DEFAULT = 5)"
    ),
    "comp_select_rms_floor": (
        "comp_select_rms_floor removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.COMP_SELECT_RMS_FLOOR = 1e-6)"
    ),
    "comp_slope_significance_k": (
        "comp_slope_significance_k removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed default of check_comparison_stability now (3 sigma)"
    ),
    "comparison_stars_pool_n": (
        "comparison_stars_pool_n removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the comparison pool is uncapped (was always 0)"
    ),
    "dao_centroid_max_shift_fwhm": (
        "dao_centroid_max_shift_fwhm removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.DAO_CENTROID_MAX_SHIFT_FWHM = 1)"
    ),
    "epsf_min_stars": (
        "epsf_min_stars removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.EPSF_MIN_BUILD_STARS = 30)"
    ),
    "err_empty_apertures_min": (
        "err_empty_apertures_min removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): fixed defaults of the empty-aperture sky error (64 apertures, at least 16)"
    ),
    "err_empty_apertures_n": (
        "err_empty_apertures_n removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): fixed defaults of the empty-aperture sky error (64 apertures, at least 16)"
    ),
    "exoplanet_match_max_sep_arcsec": (
        "exoplanet_match_max_sep_arcsec removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.EXOPLANET_MATCH_MAX_SEP_ARCSEC = 3 arcsec)"
    ),
    "forced_photometry_enabled": (
        "forced_photometry_enabled removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): forced photometry of MASTERSTAR stars is always on, without extra margin"
    ),
    "forced_photometry_margin_px": (
        "forced_photometry_margin_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): forced photometry of MASTERSTAR stars is always on, without extra margin"
    ),
    "frame_height_px": (
        "frame_height_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the frame size always comes from FITS NAXIS1/2"
    ),
    "frame_width_px": (
        "frame_width_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the frame size always comes from FITS NAXIS1/2"
    ),
    "gs11_dilution_aperture_arcsec": (
        "gs11_dilution_aperture_arcsec removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the dilution aperture is always the target's photometric aperture (0 = auto was the only consistent value)"
    ),
    "gs11_dilution_mag_limit_delta": (
        "gs11_dilution_mag_limit_delta removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed default of compute_dilution_factor now (5 mag)"
    ),
    "gs11_target_min_dilution": (
        "gs11_target_min_dilution removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (dilution.GS11_TARGET_MIN_DILUTION = 0.5)"
    ),
    "hrd_dsc_confirm_prob": (
        "hrd_dsc_confirm_prob removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_enrich_max_candidates": (
        "hrd_enrich_max_candidates removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_enrich_tap_timeout_s": (
        "hrd_enrich_tap_timeout_s removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_max_per_category": (
        "hrd_max_per_category removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_min_per_net": (
        "hrd_min_per_net removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_parallax_min_mas": (
        "hrd_parallax_min_mas removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_parallax_snr_min": (
        "hrd_parallax_snr_min removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_simbad_enrich_enabled": (
        "hrd_simbad_enrich_enabled removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): merged into hrd_online_enrich_enabled (one switch for Gaia TAP and SIMBAD)"
    ),
    "lc_high_err_nsigma": (
        "lc_high_err_nsigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the four 5-sigma LC flag thresholds are one constant now (lc_outlier.OUTLIER_N_SIGMA = 5)"
    ),
    "lc_outlier_adjacent_sigma": (
        "lc_outlier_adjacent_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (lc_outlier.ADJACENT_SIGMA = 3)"
    ),
    "lc_outlier_enabled": (
        "lc_outlier_enabled removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): LC outlier flagging is always on (flags never change photometry)"
    ),
    "lc_outlier_evidence_n_sigma": (
        "lc_outlier_evidence_n_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the four 5-sigma LC flag thresholds are one constant now (lc_outlier.OUTLIER_N_SIGMA = 5)"
    ),
    "lc_outlier_frame_qc_n_sigma": (
        "lc_outlier_frame_qc_n_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the four 5-sigma LC flag thresholds are one constant now (lc_outlier.OUTLIER_N_SIGMA = 5)"
    ),
    "lc_outlier_n_sigma": (
        "lc_outlier_n_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the four 5-sigma LC flag thresholds are one constant now (lc_outlier.OUTLIER_N_SIGMA = 5)"
    ),
    "lc_quality_min_normal_frac": (
        "lc_quality_min_normal_frac removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed default of classify_lc_quality now (3 frames, 0.5)"
    ),
    "lc_quality_short_min_frames": (
        "lc_quality_short_min_frames removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed default of classify_lc_quality now (3 frames, 0.5)"
    ),
    "masterstar_dao_centroid_qa_cap_px": (
        "masterstar_dao_centroid_qa_cap_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_centroid_qa_floor_px": (
        "masterstar_dao_centroid_qa_floor_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_empty_sky_false_accept_max": (
        "masterstar_dao_empty_sky_false_accept_max removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): read by nobody since the empty-sky audit was removed"
    ),
    "masterstar_dao_empty_sky_target_n": (
        "masterstar_dao_empty_sky_target_n removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): read by nobody since the empty-sky audit was removed"
    ),
    "masterstar_dao_match_radius_k": (
        "masterstar_dao_match_radius_k removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_pass1_dedup_px": (
        "masterstar_dao_pass1_dedup_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_pass2_center_tol_px": (
        "masterstar_dao_pass2_center_tol_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_pass2_sigma": (
        "masterstar_dao_pass2_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_forced_seed_centroid_max_px": (
        "masterstar_forced_seed_centroid_max_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_forced_seed_comp_pool_enabled": (
        "masterstar_forced_seed_comp_pool_enabled removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): forced seeds never enter the comparison pool (was always off)"
    ),
    "masterstar_forced_seed_snr_min": (
        "masterstar_forced_seed_snr_min removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_gaia_census_edge_margin_px": (
        "masterstar_gaia_census_edge_margin_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_lock_leftover_radius_px": (
        "masterstar_lock_leftover_radius_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_lock_pair_tol_px": (
        "masterstar_lock_pair_tol_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_use_best_frame_fwhm": (
        "masterstar_use_best_frame_fwhm removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): always on: the MASTERSTAR DAO FWHM is the best frame's FWHM"
    ),
    "phase01_comparison_min_dist_arcsec": (
        "phase01_comparison_min_dist_arcsec removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the target-comparison separation comes from geometry (annulus + aperture x FWHM)"
    ),
    "phase01_ct_extrapolation_tol": (
        "phase01_ct_extrapolation_tol removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the colour term is applied only inside the comparison-star colour range (was always 0)"
    ),
    "phase01_ct_min_comp": (
        "phase01_ct_min_comp removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.CT_MIN_COMP = 7)"
    ),
    "plate_solve_fov_deg": (
        "plate_solve_fov_deg removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the value was never used (the field size comes from FOCALLEN + PIXSIZE + NAXIS)"
    ),
    "preprocess_sky_surface_force_reapply": (
        "preprocess_sky_surface_force_reapply removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): sky surface is subtracted once; a new order needs recalibration from raw"
    ),
    "psf_chi2_locus_nsigma": (
        "psf_chi2_locus_nsigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (psf_chi2_locus.DEFAULT_NSIGMA = 5)"
    ),
    "sparse_trust_T_green": (
        "sparse_trust_T_green removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the sparse-trust band limits are the SparseTrustConfig defaults (1.5 / 4.0 / 0.0004)"
    ),
    "sparse_trust_T_red": (
        "sparse_trust_T_red removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the sparse-trust band limits are the SparseTrustConfig defaults (1.5 / 4.0 / 0.0004)"
    ),
    "sparse_trust_X2_RED": (
        "sparse_trust_X2_RED removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the sparse-trust band limits are the SparseTrustConfig defaults (1.5 / 4.0 / 0.0004)"
    ),
    "tess_contam_max": (
        "tess_contam_max removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (tess_analysis.TESS_CONTAM_MAX = 0.10)"
    ),
    "color_level_k_mag_per_bprp": (
        "color_level_k_mag_per_bprp removed 2026-10-19 (CFG-REVIEW-2026-10): the Clear-filter level colour coefficient was never applied since BANDAUTH-2026-10 (Clear is reported untransformed, TRANS=NO); it only kept the trust gate from seeing the colour term as off"
    ),
    "color_level_k_stderr_mag_per_bprp": (
        "color_level_k_stderr_mag_per_bprp removed 2026-10-19 (CFG-REVIEW-2026-10): the Clear-filter level colour coefficient was never applied since BANDAUTH-2026-10 (Clear is reported untransformed, TRANS=NO); it only kept the trust gate from seeing the colour term as off"
    ),
    "sigma_sys_mag": (
        "sigma_sys_mag removed 2026-10-19 (CFG-REVIEW-2026-10): a per-camera property; it is EQUIPMENTS.SIGMA_SYS_MAG now"
    ),
    "gain_container_scale": (
        "gain_container_scale removed 2026-10-19 (CFG-REVIEW-2026-10): a per-camera property; it is EQUIPMENTS.ADC_BITS now (container scale = 2^(16 - ADC_BITS))"
    ),
    "dao_qc_in_calibrate": (
        "dao_qc_in_calibrate removed 2026-10-19 (CFG-REVIEW-2026-10): the DAO QC during calibration (PERF-10) always runs"
    ),
    "qc_max_hfr": (
        "qc_max_hfr removed 2026-10-19 (CFG-REVIEW-2026-10): the post-calibration QC limits only set a pass/fail flag nobody read (frames were always kept); the QC measurements stay"
    ),
    "qc_min_stars": (
        "qc_min_stars removed 2026-10-19 (CFG-REVIEW-2026-10): the post-calibration QC limits only set a pass/fail flag nobody read (frames were always kept); the QC measurements stay"
    ),
    "qc_max_background_rms": (
        "qc_max_background_rms removed 2026-10-19 (CFG-REVIEW-2026-10): the post-calibration QC limits only set a pass/fail flag nobody read (frames were always kept); the QC measurements stay"
    ),
    "k2_fit_enabled": (
        "k2_fit_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "k2_fit_min_detectability": (
        "k2_fit_min_detectability removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "k2_fit_consistency_sigma": (
        "k2_fit_consistency_sigma removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "k2_fit_lit_factor": (
        "k2_fit_lit_factor removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "k2_ceiling": (
        "k2_ceiling removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "aperture_policy_mode": (
        "aperture_policy_mode removed 2026-10-19 (CFG-REVIEW-2026-10): per_target is the only aperture policy (APERTURE-K-2026-10); f_fixed_night / f_per_frame are removed"
    ),
    "cog_aperture_correction_enabled": (
        "cog_aperture_correction_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_ref_fwhm": (
        "cog_ref_fwhm removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_min_stars": (
        "cog_min_stars removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_isolation_fwhm": (
        "cog_isolation_fwhm removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_snr_min": (
        "cog_snr_min removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_ladder_step_px": (
        "cog_ladder_step_px removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_ac_factor_max": (
        "cog_ac_factor_max removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "aperture_photometry_enabled": (
        "aperture_photometry_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): aperture photometry is the only photometry of the per-frame catalogs"
    ),
    "psf_photometry_enabled": (
        "psf_photometry_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): the ePSF light curves are a separate product (ePSF tab, epsf_export_auto); reports and exports are aperture photometry"
    ),
    "blind_img_select_mode": (
        "blind_img_select_mode removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_img_star_budget": (
        "blind_img_star_budget removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_index_fine_path": (
        "blind_index_fine_path removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_index_select_mode": (
        "blind_index_select_mode removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_index_wide_path": (
        "blind_index_wide_path removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_use_rig_prior": (
        "blind_use_rig_prior removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_early_accept": (
        "blind_verify_early_accept removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_early_floor": (
        "blind_verify_early_floor removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_early_fraction": (
        "blind_verify_early_fraction removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_enabled": (
        "blind_verify_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_inmemory_catalog": (
        "blind_verify_inmemory_catalog removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_match_tol_px": (
        "blind_verify_match_tol_px removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_min_fraction": (
        "blind_verify_min_fraction removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_min_matches": (
        "blind_verify_min_matches removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_top_n": (
        "blind_verify_top_n removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "catalog_query_max_rows": (
        "catalog_query_max_rows removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "debug_platesolver": (
        "debug_platesolver removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_catalog_recovery_min": (
        "masterstar_catalog_recovery_min removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_centre_rms_max_px": (
        "masterstar_centre_rms_max_px removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_detection_cap_adaptive": (
        "masterstar_detection_cap_adaptive removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_detection_cap_k": (
        "masterstar_detection_cap_k removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_detection_cap_max": (
        "masterstar_detection_cap_max removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_detection_cap_min": (
        "masterstar_detection_cap_min removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_distortion_benign_ratio_max": (
        "masterstar_distortion_benign_ratio_max removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_min_matched_floor": (
        "masterstar_min_matched_floor removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_platesolve_sip_max_order": (
        "masterstar_platesolve_sip_max_order removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_platesolve_sip_min_order": (
        "masterstar_platesolve_sip_min_order removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_quality_crowded_n_cat_min": (
        "masterstar_quality_crowded_n_cat_min removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "verify_mag_limit": (
        "verify_mag_limit removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_index_path": (
        "blind_index_path removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "BLIND_INDEX_PATH": (
        "BLIND_INDEX_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "BLIND_INDEX_FINE_PATH": (
        "BLIND_INDEX_FINE_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "BLIND_INDEX_WIDE_PATH": (
        "BLIND_INDEX_WIDE_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "psf_zp_membership": (
        "psf_zp_membership removed 2026-10-19 (CFG-REVIEW-2026-10): both modes admitted the same epochs (psf_fit_ok and a positive flux) since FIT-OK-ADMISSION-01"
    ),
    "psf_zp_for_zp_validated_rigs": (
        "psf_zp_for_zp_validated_rigs removed 2026-10-19 (CFG-REVIEW-2026-10): only selected between two identical psf_zp_membership modes"
    ),
    "calibration_library_native_binning": (
        "calibration_library_native_binning removed 2026-10-19 (CFG-REVIEW-2026-10): it was ignored on load since CAL-2026-10 (master binning comes from each master header)"
    ),
    "vsx_local_db_path": (
        "vsx_local_db_path removed 2026-10-20 (CATALOG-DB-2026-10): VSX is the table vsx_data of objects_db_path (catalogs/vyvar_objects.db, built by catalogs/objects_make.py)"
    ),
    "exoplanet_local_db_path": (
        "exoplanet_local_db_path removed 2026-10-20 (CATALOG-DB-2026-10): exoplanets are the table exoplanet_data of objects_db_path (catalogs/vyvar_objects.db, built by catalogs/objects_make.py)"
    ),
    "GAIA_DB_PATH": (
        "GAIA_DB_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): legacy alias - use gaia_db_path"
    ),
    "VSX_LOCAL_DB_PATH": (
        "VSX_LOCAL_DB_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): legacy alias - use objects_db_path"
    ),
    "EXOPLANET_LOCAL_DB_PATH": (
        "EXOPLANET_LOCAL_DB_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): legacy alias - use objects_db_path"
    ),
    "trust_flag_enabled": (
        "trust_flag_enabled removed 2026-10-22 (TRUST-ALWAYS-2026-10): trust always runs; with it off RED targets were exported for submission"
    ),
    "photon_transfer_ci_max_width_factor": (
        "photon_transfer_ci_max_width_factor removed 2026-10-22 (GAIN-ONE-2026-10): the photon-transfer gain is gone; gain = header / EQUIPMENTS / ADC_BITS"
    ),
    "phase01_plate_scale_arcsec_per_px": (
        "phase01_plate_scale_arcsec_per_px removed 2026-10-22 (PLATE-SCALE-2026-10): the plate scale comes from the solved MASTERSTAR WCS"
    ),
    "plate_scale_arcsec_per_px": (
        "plate_scale_arcsec_per_px removed 2026-10-22 (PLATE-SCALE-2026-10): the plate scale comes from the solved MASTERSTAR WCS"
    ),
    "comp_pool_derived_admission": (
        "comp_pool_derived_admission removed 2026-10-22 (COMP-ADMIT-2026-10): known variables (VSX, Gaia DR3 variability catalogue) are always excluded; the derived noise-curve admission only recomputed that test"
    ),
    "masterstar_prematch_peak_sigma_floor": (
        "masterstar_prematch_peak_sigma_floor removed 2026-10-20 (DAO-FALSE-2026-10): the peak filter before the Gaia match had no effect at the detection threshold (530: 1 of 1957 dropped); false detections are measured on every MASTERSTAR instead (pipeline_meta dao_false_detection)"
    ),
    "observer_location_id": (
        "observer_location_id removed 2026-10-20 (SETTINGS-2026-10): the default site is LOCATION.IS_DEFAULT in the database (Database Explorer); the site of a run is chosen on the import page or with --location-id"
    ),
    "aavso_observer_code": (
        "aavso_observer_code removed 2026-10-19 (CFG-REVIEW-2026-10): legacy alias - use observer_code"
    ),
    "aperture_fwhm_factor_medium": (
        "aperture_fwhm_factor_medium removed (legacy, never read)"
    ),
    "phase01_flux_col": (
        "phase01_flux_col removed 2026-10-19 (CFG-REVIEW-2026-10): comparison-star selection runs before RUN ePSF, so dao_flux was its only usable value"
    ),
    "comp_iterative_clip_enabled": (
        "comp_iterative_clip_enabled removed (COMP-RULES-2026-10: no iterative comp clip)"
    ),
    "comp_tier1_bprp_limit": (
        "comp_tier1_bprp_limit removed 2026-10-19 (CFG-REVIEW-2026-10): legacy form of comp_color_tiers"
    ),
    "comp_tier1_weight": (
        "comp_tier1_weight removed (comparison ensemble weights are Broeg weights)"
    ),
    "comp_tier2_bprp_limit": (
        "comp_tier2_bprp_limit removed 2026-10-19 (CFG-REVIEW-2026-10): legacy form of comp_color_tiers"
    ),
    "comp_tier2_weight": (
        "comp_tier2_weight removed (comparison ensemble weights are Broeg weights)"
    ),
    "comp_tier3_bprp_limit": (
        "comp_tier3_bprp_limit removed 2026-10-19 (CFG-REVIEW-2026-10): legacy form of comp_color_tiers"
    ),
    "comp_tier3_weight": (
        "comp_tier3_weight removed (comparison ensemble weights are Broeg weights)"
    ),
    "comp_tier4_bprp_limit": (
        "comp_tier4_bprp_limit removed 2026-10-19 (CFG-REVIEW-2026-10): legacy form of comp_color_tiers"
    ),
    "comp_tier4_weight": (
        "comp_tier4_weight removed (comparison ensemble weights are Broeg weights)"
    ),
    "psf_spatial_order": (
        "psf_spatial_order removed 2026-10-18 (EPSF-SPATIAL-2026-10: the spatial ePSF is the night-averaged 3x3/2x2 perturbation chosen by the build-star count)"
    ),
    "psf_chi2_threshold": (
        "psf_chi2_threshold removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_fit_ok is set by the night chi2 locus (psf_chi2_locus_nsigma); the legacy absolute cut had no consumer)"
    ),
    "psf_grouper_enabled": (
        "psf_grouper_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: the per-star cutout fitter with SourceGrouper is replaced by the frame-level fit)"
    ),
    "psf_group_sep_fwhm": (
        "psf_group_sep_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: the per-star cutout fitter with SourceGrouper is replaced by the frame-level fit)"
    ),
    "psf_neighbor_include_fwhm": (
        "psf_neighbor_include_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: the per-star cutout fitter with SourceGrouper is replaced by the frame-level fit)"
    ),
    "psf_neighbor_sub_enabled": (
        "psf_neighbor_sub_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_chi2_max": (
        "neighbor_sub_chi2_max removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_residual_rms_max": (
        "neighbor_sub_residual_rms_max removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_refuse_sep_fwhm": (
        "neighbor_sub_refuse_sep_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_centroid_max_fwhm": (
        "neighbor_sub_centroid_max_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_nn_contam_dmag": (
        "neighbor_sub_nn_contam_dmag removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_max_neighbor_overmag": (
        "neighbor_sub_max_neighbor_overmag removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_max_target_undermag": (
        "neighbor_sub_max_target_undermag removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_min_recovered_snr": (
        "neighbor_sub_min_recovered_snr removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_regime_dmag_min": (
        "neighbor_sub_regime_dmag_min removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_regime_sep_max": (
        "neighbor_sub_regime_sep_max removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "psf_spatial_enabled": (
        "psf_spatial_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: the spatial ePSF is always on (night-averaged perturbation, grid by build-star count))"
    ),
    "psf_quality_fallback_enabled": (
        "psf_quality_fallback_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: the per-star quality grade went with the cutout fitter; the night chi2 locus sets psf_fit_ok)"
    ),
    "psf_adaptive_enabled": (
        "psf_adaptive_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: the adaptive aperture/PSF picker had no inputs (psf_ac_applied, psf_quality) after the per-frame ePSF; published light curves are aperture, PSF light curves are internal)"
    ),
    "psf_adaptive_resolve_fwhm": (
        "psf_adaptive_resolve_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: the adaptive aperture/PSF picker is removed)"
    ),
    "psf_adaptive_snr_lo": (
        "psf_adaptive_snr_lo removed 2026-10-18 (EPSF-SPATIAL-2026-10: the adaptive aperture/PSF picker is removed)"
    ),
    "photometry_mode": (
        "photometry_mode removed 2026-10-18 (EPSF-SPATIAL-2026-10); aperture photometry always runs on the MASTERSTAR catalogue, ePSF photometry is the separate RUN ePSF stage with a per-frame ePSF (epsf_auto_run)"
    ),
    "hrd_color_field_enabled": (
        "hrd_color_field_enabled removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_saturation": (
        "hrd_color_saturation removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_highlight_mode": (
        "hrd_color_highlight_mode removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_chroma_snr": (
        "hrd_color_chroma_snr removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_white_point": (
        "hrd_color_white_point removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_chroma_boost": (
        "hrd_color_chroma_boost removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_bg_box_px": (
        "hrd_color_bg_box_px removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "bpm_dark_mad_sigma": (
        "bpm_dark_mad_sigma removed 2026-10-17 (BPM-REMOVE-2026-10); dark bad-column map retired (Milan: not used; the *_dark_bpm.json sidecar was never written)"
    ),
    "forced_photometry_centroid_bound_fwhm": (
        "forced_photometry_centroid_bound_fwhm removed 2026-10-17 (CFG-CLEAN-2026-10-17); forced photometry uses the MASTERSTAR position (no refine since POS-SUBPIX-2026-10), the bound was only echoed into debug meta"
    ),
    "blind_verify_match_tol_arcsec": (
        "blind_verify_match_tol_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "cog_ladder_step_fwhm": (
        "cog_ladder_step_fwhm removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "hrd_color_bg_box_arcsec": (
        "hrd_color_bg_box_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "masterstar_centre_rms_max_arcsec": (
        "masterstar_centre_rms_max_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "masterstar_sibling_recovery_enabled": (
        "masterstar_sibling_recovery_enabled removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_min_matched": (
        "masterstar_sibling_min_matched removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_rms_max_px": (
        "masterstar_sibling_rms_max_px removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_min_quadrants": (
        "masterstar_sibling_min_quadrants removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_stack_n": (
        "masterstar_sibling_stack_n removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_rms_max_arcsec": (
        "masterstar_sibling_rms_max_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "phase01_chip_interior_margin_arcsec": (
        "phase01_chip_interior_margin_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "phase01_comparison_isolation_radius_arcsec": (
        "phase01_comparison_isolation_radius_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "qc_max_hfr_fwhm_ratio": (
        "qc_max_hfr_fwhm_ratio removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "sips_dao_fwhm_fwhm_factor": (
        "sips_dao_fwhm_fwhm_factor removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "auto_fwhm_k_min": (
        "auto_fwhm_k_min removed 2026-10-17 (CFG-CLEAN-2026-10-17); auto_fwhm_k_factor is clamped to 1..4 directly"
    ),
    "auto_fwhm_k_max": (
        "auto_fwhm_k_max removed 2026-10-17 (CFG-CLEAN-2026-10-17); auto_fwhm_k_factor is clamped to 1..4 directly"
    ),
    "phase01_comparison_exclude_gaia_nss": (
        "phase01_comparison_exclude_gaia_nss removed 2026-10-14; a Gaia binary that does not vary is a "
        "valid comparison - comparisons must not be variable (VSX, Gaia variability flag, own stability test)"
    ),
    "masterstar_gaia_census_target_depth_g": (
        "masterstar_gaia_census_target_depth_g removed 2026-10-14; the DAO <-> Gaia matching depth is "
        "measured on each MASTERSTAR (FRAME-DEPTH-2026-10)"
    ),
    "phase01_comparison_exclude_gaia_extobj": "COMP-NSS-2026-10: the local Gaia DB has no QSO/galaxy columns - the filter never ran; key removed",
    "phase01_comparison_max_fwhm_factor": "PHASE1-ONE-PASS-2026-10: the FWHM blend filter only fed a diagnostic column; blends are gated by the neighbour-light rule",
    "gs11_comp_max_dilution": "PHASE1-ONE-PASS-2026-10: dilution never rejected a comparison star (the counter could not increment); key removed",
    "frame_align_residual_gate_enabled": "ALIGN-VERIFY-2026-10: the circular aperture-photometry alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
    "frame_align_residual_max_frac": "ALIGN-VERIFY-2026-10: the circular aperture-photometry alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
    "frame_align_residual_min_keep_frames": "ALIGN-VERIFY-2026-10: the circular aperture-photometry alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
    "comp_qa_enabled": "COMP-QA-2026-10: Sokolovsky LOO comp QA removed; comps are gated at selection (not variable in Gaia/VSX, neighbour light, frame edge)",
    "variability_min_frames": (
        "variability_min_frames removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_min_frames_frac": (
        "variability_min_frames_frac removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_p85_filter": (
        "variability_p85_filter removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_slope_floor": (
        "variability_slope_floor removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_sigma_threshold": (
        "variability_sigma_threshold removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_comp_floor_factor": (
        "variability_comp_floor_factor removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_smoothness_max": (
        "variability_smoothness_max removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_mag_limit": (
        "variability_mag_limit removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_min_rms_pct": (
        "variability_min_rms_pct removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_min_amplitude_mag": (
        "variability_min_amplitude_mag removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_clip_ratio_min": (
        "variability_clip_ratio_min removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_vdi_z_threshold": (
        "variability_vdi_z_threshold removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_min_points_rms": (
        "variability_min_points_rms removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "aperture_f_grid": (
        "aperture_f_grid removed 2026-10; fixed k grid 1.0-2.0 x FWHM, k* by comparison-star "
        "scatter (APERTURE-K-2026-10)"
    ),
    "skip_processed_directory": (
        "skip_processed_directory removed 2026-07; skip behavior is now always on"
    ),
    "vsx_variable_targets_mag_limit": (
        "vsx_variable_targets_mag_limit removed 2026-07; VSX scope is detection-limited (DAO+Gaia match)"
    ),
    "phase01_match_radius_arcsec": (
        "phase01_match_radius_arcsec removed 2026-07; target selection uses catalog_id identity join (PHASE0-IDENTITY-GATE)"
    ),
    "aperture_selection_criterion": (
        "aperture_selection_criterion removed 2026-08-31 CONSOLIDATE-01B; scatter/SNR diagnostic table deleted"
    ),
    "aperture_scatter_r_min_px": (
        "aperture_scatter_r_min_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "aperture_scatter_r_max_px": (
        "aperture_scatter_r_max_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "aperture_scatter_r_step_px": (
        "aperture_scatter_r_step_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "global_comp_pool_enabled": (
        "global_comp_pool_enabled removed 2026-09-01 CONSOLIDATE-01D; global pool is always on (COMP-POOL-01)"
    ),
    "export_err_mode": (
        "export_err_mode removed 2026-09-01 CONSOLIDATE-01D (the ERR-CALIB layer itself was removed in CODE-REVIEW-2026-10 step 6: its sidecar was never written)"
    ),
    "err_background_mode": (
        "err_background_mode removed 2026-09-01 CONSOLIDATE-01D; empirical empty-aperture term is always on (F-BINGAIN-1); Howell math remains as missing-sigma fallback"
    ),
    "masterstar_accept_mode": (
        "masterstar_accept_mode removed 2026-09-01 CONSOLIDATE-01D; MASTERSTAR odds gate is the only accept policy"
    ),
    "psf_ac_policy": (
        "psf_ac_policy removed 2026-09-01 CONSOLIDATE-01D; p4_none is the only ePSF AC policy (ZP-OK v2 / P4)"
    ),
    "qc_fwhm_limit": (
        "qc_fwhm_limit removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "qc_elong_limit": (
        "qc_elong_limit removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "psf_spatial_grid": (
        "psf_spatial_grid removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "psf_spatial_min_stars_per_cell": (
        "psf_spatial_min_stars_per_cell removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "gs11_comp_suspect_dilution": (
        "gs11_comp_suspect_dilution removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "phase01_use_bprp_primary": (
        "phase01_use_bprp_primary removed 2026-09-01 CONSOLIDATE-01D; BP-RP is always the color display basis"
    ),
    "aperture_correction_enabled": (
        "aperture_correction_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; aperture correction Method B deleted (mag_calib_final = mag_calib + CT)"
    ),
    "aperture_correction_min_ref_stars": (
        "aperture_correction_min_ref_stars removed 2026-10-10 SPECIAL-FNS-REMOVE; aperture correction Method B deleted (mag_calib_final = mag_calib + CT)"
    ),
    "aperture_correction_max_contamination": (
        "aperture_correction_max_contamination removed 2026-10-10 SPECIAL-FNS-REMOVE; aperture correction Method B deleted (mag_calib_final = mag_calib + CT)"
    ),
    "aperture_correction_max_scatter_mag": (
        "aperture_correction_max_scatter_mag removed 2026-10-10 SPECIAL-FNS-REMOVE; aperture correction Method B deleted (mag_calib_final = mag_calib + CT)"
    ),
    "temporal_binning_enabled": (
        "temporal_binning_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-3 temporal binning deleted"
    ),
    "temporal_bin_window": (
        "temporal_bin_window removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-3 temporal binning deleted"
    ),
    "savgol_detrend_enabled": (
        "savgol_detrend_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-2 Savitzky-Golay detrend deleted"
    ),
    "savgol_window_frac": (
        "savgol_window_frac removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-2 Savitzky-Golay detrend deleted"
    ),
    "savgol_polyorder": (
        "savgol_polyorder removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-2 Savitzky-Golay detrend deleted"
    ),
    "democratic_detrend_enabled": (
        "democratic_detrend_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-4 Democratic Detrender deleted"
    ),
    "democratic_sg_window_frac": (
        "democratic_sg_window_frac removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-4 Democratic Detrender deleted"
    ),
    "pytics_enabled": (
        "pytics_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-5 PyTICS weights deleted (ensemble uses comp_rms_map weights directly)"
    ),
    "pytics_n_iter": (
        "pytics_n_iter removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-5 PyTICS weights deleted (ensemble uses comp_rms_map weights directly)"
    ),
    "sysrem_enabled": (
        "sysrem_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; SysRem field detrend deleted"
    ),
    "sysrem_n_iter": (
        "sysrem_n_iter removed 2026-10-10 SPECIAL-FNS-REMOVE; SysRem field detrend deleted"
    ),
    "phase2a_airmass_before_outlier": (
        "phase2a_airmass_before_outlier removed 2026-10-10 SPECIAL-FNS-REMOVE; dead switch (no target airmass detrend exists)"
    ),
    "aperture_variable_factor": (
        "aperture_variable_factor removed 2026-10-10 SPECIAL-FNS-REMOVE; role-aware aperture scale was never applied"
    ),
    "aperture_comp_factor": (
        "aperture_comp_factor removed 2026-10-10 SPECIAL-FNS-REMOVE; role-aware aperture scale was never applied"
    ),
    "master_combine_method": (
        "master_combine_method removed 2026-10 CFG-CLEAN; never consumed (the Calibration Library "
        "build dialog chooses median/mean per build)"
    ),
    "export_arcsec_per_px": (
        "export_arcsec_per_px removed 2026-10 CFG-CLEAN; export plate scale is derived from "
        "pipeline_meta.json / MASTERSTAR WCS"
    ),
    "aperture_snr_sizing": (
        "aperture_snr_sizing removed 2026-10 CFG-CLEAN; SNR sizing sweep no longer exists, never consumed"
    ),
    "aperture_fwhm_factor_small": (
        "aperture_fwhm_factor_small removed 2026-10 CFG-CLEAN; SNR sizing sweep no longer exists, never consumed"
    ),
    "aperture_fwhm_factor_large": (
        "aperture_fwhm_factor_large removed 2026-10 CFG-CLEAN; SNR sizing sweep no longer exists, never consumed"
    ),
    "phase01_tiers": (
        "phase01_tiers removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "phase01_tier1_mag": (
        "phase01_tier1_mag removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "phase01_tier2_mag": (
        "phase01_tier2_mag removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "phase01_tier3_mag": (
        "phase01_tier3_mag removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "phase01_tier4_mag": (
        "phase01_tier4_mag removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "sips_dao_threshold_sigma": (
        "sips_dao_threshold_sigma removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "masterstar_dao_threshold_sigma": (
        "masterstar_dao_threshold_sigma removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "dao_detection_n_equiv": (
        "dao_detection_n_equiv removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "alignment_detection_sigma": (
        "alignment_detection_sigma removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "qc_dao_detection_sigma": (
        "qc_dao_detection_sigma removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "saturate_limit_fraction": (
        "saturate_limit_fraction removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "admission_sat_peak_frac": (
        "admission_sat_peak_frac removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "cog_sat_frac": (
        "cog_sat_frac removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "nonlinearity_peak_percentile": (
        "nonlinearity_peak_percentile removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "nonlinearity_fwhm_ratio": (
        "nonlinearity_fwhm_ratio removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "per_frame_saturation_enabled": (
        "per_frame_saturation_enabled removed 2026-10-10; per-frame saturation is now always on (SAT-LIN-2026-10)"
    ),
    "comp_sparse_fallback_enabled": (
        "comp_sparse_fallback_enabled removed 2026-10-10; sparse comp fallback removed, comparison stars always pass the same rules (COMP-RULES-2026-10)"
    ),
    "comp_sparse_fallback_min": (
        "comp_sparse_fallback_min removed 2026-10-10; sparse comp fallback removed, comparison stars always pass the same rules (COMP-RULES-2026-10)"
    ),
    "comp_weight_c_col_mag_per_bprp": (
        "comp_weight_c_col_mag_per_bprp removed 2026-10-10; comparison weights are Broeg (2005) weights computed in photometry_lightcurve.ensemble_normalize"
    ),
    "comp_weight_c_dist_mag_per_deg": (
        "comp_weight_c_dist_mag_per_deg removed 2026-10-10; comparison weights are Broeg (2005) weights computed in photometry_lightcurve.ensemble_normalize"
    ),
    "comp_weight_airmass_span": (
        "comp_weight_airmass_span removed 2026-10-10; comparison weights are Broeg (2005) weights computed in photometry_lightcurve.ensemble_normalize"
    ),
    "comp_weight_optics_kind": (
        "comp_weight_optics_kind removed 2026-10-10; comparison weights are Broeg (2005) weights computed in photometry_lightcurve.ensemble_normalize"
    ),
    "phase01_comparison_max_mag_diff": (
        "phase01_comparison_max_mag_diff removed 2026-10-10 (CLEAN-STEP5-2026-10); no magnitude window in comparison selection (|dmag| never gated; Broeg weights)"
    ),
    "phase01_comparison_max_mag_diff_bright_floor": (
        "phase01_comparison_max_mag_diff_bright_floor removed 2026-10-10 (CLEAN-STEP5-2026-10); no magnitude window in comparison selection (|dmag| never gated; Broeg weights)"
    ),
    "phase01_comparison_mag_bright_threshold": (
        "phase01_comparison_mag_bright_threshold removed 2026-10-10 (CLEAN-STEP5-2026-10); no magnitude window in comparison selection (|dmag| never gated; Broeg weights)"
    ),
    "phase01_comparison_max_mag_diff_absolute": (
        "phase01_comparison_max_mag_diff_absolute removed 2026-10-10 (CLEAN-STEP5-2026-10); no magnitude window in comparison selection (|dmag| never gated; Broeg weights)"
    ),
    "phase01_comparison_max_dist_deg": (
        "phase01_comparison_max_dist_deg removed 2026-10-10 (CLEAN-STEP5-2026-10); no distance limit in comparison selection (distance is only a tie-break after RMS and colour)"
    ),
    "phase01_comparison_max_dist_fov_frac": (
        "phase01_comparison_max_dist_fov_frac removed 2026-10-10 (CLEAN-STEP5-2026-10); no distance limit in comparison selection (distance is only a tie-break after RMS and colour)"
    ),
    "phase01_comparison_fov_fraction": (
        "phase01_comparison_fov_fraction removed 2026-10-10 (CLEAN-STEP5-2026-10); no distance limit in comparison selection (distance is only a tie-break after RMS and colour)"
    ),
    "phase01_comparison_max_psf_chi2": (
        "phase01_comparison_max_psf_chi2 removed 2026-10-10 (CLEAN-STEP5-2026-10); comparison-star selection never applied a PSF chi2 cut (proc CSVs are DAO-era)"
    ),
    "field_density_adaptive_enabled": (
        "field_density_adaptive_enabled removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "field_density_sparse_threshold": (
        "field_density_sparse_threshold removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "field_density_dense_threshold": (
        "field_density_dense_threshold removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "crowding_classifier_enabled": (
        "crowding_classifier_enabled removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "crowding_blend_tighten_threshold": (
        "crowding_blend_tighten_threshold removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "crowding_comp_availability_loosen_count": (
        "crowding_comp_availability_loosen_count removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "crowding_tighten_min_fwhm_px": (
        "crowding_tighten_min_fwhm_px removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
}


DETECTION_THRESHOLD_SIGMA_DEFAULT: float = 4.5
DETECTION_THRESHOLD_SIGMA_MIN: float = 2.0
DETECTION_THRESHOLD_SIGMA_MAX: float = 20.0


def resolve_detection_threshold_sigma(raw: Any) -> float:
    """Coerce a detection threshold (N x sky sigma) to a finite float clamped to 2.0-20.0.

    Accepts a raw number or an object with ``detection_threshold_sigma`` (e.g. AppConfig), so
    the GUI, night runner and every pipeline stage resolve the single threshold identically.
    Non-numeric / non-finite / non-positive input falls back to the 4.5 default.
    """
    if raw is not None and not isinstance(raw, (int, float, str)):
        raw = getattr(raw, "detection_threshold_sigma", None)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return DETECTION_THRESHOLD_SIGMA_DEFAULT
    if not math.isfinite(v) or v <= 0:
        return DETECTION_THRESHOLD_SIGMA_DEFAULT
    return max(DETECTION_THRESHOLD_SIGMA_MIN, min(DETECTION_THRESHOLD_SIGMA_MAX, v))


def strip_jsonc_comments(text: str) -> str:
    """Remove ``//`` line comments that sit OUTSIDE string literals (JSONC-lite).

    A small character state machine so ``//`` inside a string value (e.g. a URL like
    ``"http://x"``) is preserved. Only ``//`` to end-of-line is stripped; block comments
    ``/* */`` and trailing commas are intentionally NOT supported so the format stays
    strict elsewhere and diffs/tools stay sane. Newlines are preserved so line numbers in
    validator error messages stay accurate.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    escape = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_config_text(text: str) -> dict[str, Any]:
    """Parse config.json text tolerating ``//`` line comments; raises json.JSONDecodeError."""
    parsed = json.loads(strip_jsonc_comments(text))
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("config.json top level must be a JSON object", text, 0)
    return parsed


_APPCONFIG_FIELD_NAMES_CACHE: tuple[str, ...] | None = None


def _appconfig_field_names() -> tuple[str, ...] | None:
    """Public AppConfig field names, cached. Returns None if AppConfig is unavailable.

    Some report tests monkeypatch ``config.AppConfig`` with a plain factory function; in
    that case ``dataclasses.fields`` raises and we fall back to the last good cache (or
    None), so the best-effort typo net never breaks an unrelated test.
    """
    global _APPCONFIG_FIELD_NAMES_CACHE
    try:
        names = tuple(
            sorted(f.name for f in _dc_fields(AppConfig) if not f.name.startswith("_"))
        )
    except TypeError:
        return _APPCONFIG_FIELD_NAMES_CACHE
    _APPCONFIG_FIELD_NAMES_CACHE = names
    return names


def _warn_unknown_config_keys(data: dict[str, Any]) -> None:
    """Log a WARN for each config.json key that is neither a field nor a known legacy alias.

    Typo safety net for hand editors: names the unknown key and the closest registered
    key (difflib). Migrated/legacy keys stay silent (they are in the allowlist).
    """
    field_names = _appconfig_field_names()
    if field_names is None:
        return  # cannot resolve the known-key set right now; skip the best-effort typo net
    known = set(field_names)
    for key in data:
        if key in known:
            continue
        if key in KNOWN_REMOVED_KEYS:
            logging.info("config.json: %s", KNOWN_REMOVED_KEYS[key])
            continue
        near = difflib.get_close_matches(key, list(field_names), n=1)
        hint = f" (did you mean '{near[0]}'?)" if near else ""
        logging.warning(
            "config.json: unknown key '%s' is ignored%s. See docs/VYVAR_CONFIG_GUIDE_EN.md "
            "or run 'python dev/scripts/validate_config.py'.",
            key,
            hint,
        )


def load_config_json(project_root: Path) -> dict[str, Any]:
    path = config_json_path(project_root)
    if not path.exists():
        return {}
    try:
        data = parse_config_text(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning(
            "config.json could not be parsed (%s); using defaults. Run "
            "'python dev/scripts/validate_config.py' to locate the error.",
            exc,
        )
        return {}
    for old, new in RENAMED_KEYS.items():
        if old in data:
            val = data.pop(old)
            if new not in data:
                data[new] = val
            logging.info("config.json: %s renamed to %s (CFG-REVIEW-2026-10)", old, new)
    _warn_unknown_config_keys(data)
    return data


class ConfigPersistError(RuntimeError):
    """Raised when config.json persistence is attempted outside an explicit UI save action."""


# CONFIG-WRITE-GUARD (PARAM-OWNERSHIP-WAVE-A STEP 1): config.json must be persisted ONLY from
# explicit user save actions in the Streamlit UI layer. The headless / pipeline path may resolve
# and USE any values it needs, but must never write them back to config.json (run-effective values
# belong in provenance). UI save handlers wrap their write in ``ui_config_persist()``; every other
# caller (pipeline, night-run, baseline check) trips ``ConfigPersistError``.
_CONFIG_PERSIST_ALLOWED = False


@contextmanager
def ui_config_persist() -> Iterator[None]:
    """Context in which ``save_config_json`` is permitted (explicit UI save action only)."""
    global _CONFIG_PERSIST_ALLOWED
    prev = _CONFIG_PERSIST_ALLOWED
    _CONFIG_PERSIST_ALLOWED = True
    try:
        yield
    finally:
        _CONFIG_PERSIST_ALLOWED = prev


# CONFIG-HUMAN-EDIT STEP 3: config.json is written as a grouped, commented, JSONC-lite
# document so a user can edit it in a text editor without the UI. Sections follow pipeline
# order; within a section keys are ordered basic -> advanced -> expert then alphabetically.
_CONFIG_SECTION_ORDER = (
    "observer", "calibration", "qc", "alignment", "detection", "photometry",
    "comp_selection", "trust", "extinction", "reports", "export", "system", "paths",
)
_CONFIG_SECTION_TITLES = {
    "observer": "Observer & export identity",
    "calibration": "Calibration",
    "qc": "Frame quality control (QC)",
    "alignment": "Alignment",
    "detection": "Detection, plate solving & masterstar",
    "photometry": "Photometry",
    "comp_selection": "Comparison-star selection",
    "trust": "Trust & quality flags",
    "extinction": "Atmospheric extinction & color",
    "reports": "Reports & HRD",
    "export": "Export",
    "system": "System & performance",
    "paths": "File & catalog paths",
}
_CONFIG_TIER_ORDER = {"basic": 0, "advanced": 1, "expert": 2}
_CONFIG_COMMENT_WIDTH = 78


def _config_header_lines() -> list[str]:
    """The file-header comment block explaining the static/dynamic model + how to edit."""
    para = [
        "VYVAR config.json -- pipeline settings, safe to edit in a text editor.",
        "",
        "This file holds ONLY user-tunable pipeline settings. Two other kinds of values "
        "live elsewhere and are NOT in this file:",
        "  - Static observatory facts (site coordinates, telescope, camera, catalogs) live "
        "in the DATABASE and are managed in the app (Settings -> Observatory).",
        "  - Dynamic per-run values (gain, read noise, frame size, plate scale, filter, "
        "exposure) are read from the FITS headers at run time and appear in the report's "
        "Resolved Facts section.",
        "",
        "Editing without the UI: '//' line comments are allowed (they are ignored on load). "
        "Trailing commas and block comments are NOT allowed. Unknown keys are ignored with "
        "a warning that suggests the closest real key. After editing, validate with:",
        "    python dev/scripts/validate_config.py",
        "Full explanations of every key: docs/VYVAR_CONFIG_GUIDE_EN.md (English) and "
        "docs/VYVAR_CONFIG_GUIDE_CZ.md (Czech).",
        "",
        "NOTE: saving from the UI regenerates this file, its grouping and its comments from "
        "the parameter registry -- any custom comments you add here are not preserved.",
    ]
    bar = "// " + "=" * (_CONFIG_COMMENT_WIDTH - 3)
    lines = [bar]
    for p in para:
        if p == "":
            lines.append("//")
            continue
        for w in textwrap.wrap(p, width=_CONFIG_COMMENT_WIDTH - 3):
            lines.append("// " + w)
    lines.append(bar)
    return lines


def _comment_block(text: str, indent: str) -> list[str]:
    prefix = indent + "// "
    avail = max(20, _CONFIG_COMMENT_WIDTH - len(prefix))
    return [prefix + w for w in (textwrap.wrap(text, width=avail) or [""])]


def render_config_jsonc(data: dict[str, Any]) -> str:
    """Render ``data`` as the canonical grouped + commented config.json (JSONC-lite).

    Deterministic: file header, then sections in pipeline order each opened by a group
    comment, keys ordered basic->advanced->expert then alphabetical, every key preceded by
    its one-line registry help. Values are emitted verbatim (json) so a save->load round
    trip is value-preserving. Keys not in the registry go to a trailing 'Other' section so
    the writer never drops data.
    """
    try:
        import params_registry as _pr  # lazy: params_registry imports config at module load

        registry = _pr.load_registry()
        phase_help = _pr.load_phase_help()
    except Exception:  # noqa: BLE001 -- writer must never fail on registry issues
        registry = {}
        phase_help = {}

    by_phase: dict[str, list[str]] = {}
    other: list[str] = []
    for key in data:
        entry = registry.get(key)
        phase = entry.get("phase") if entry else None
        if phase in _CONFIG_SECTION_ORDER:
            by_phase.setdefault(phase, []).append(key)
        else:
            other.append(key)

    sections: list[tuple[str, str, list[str]]] = []
    for phase in _CONFIG_SECTION_ORDER:
        keys = by_phase.get(phase)
        if not keys:
            continue
        keys_sorted = sorted(
            keys,
            key=lambda k: (_CONFIG_TIER_ORDER.get((registry.get(k) or {}).get("tier"), 1), k),
        )
        sections.append((_CONFIG_SECTION_TITLES[phase], phase_help.get(phase, ""), keys_sorted))
    if other:
        sections.append(("Other", "", sorted(other)))

    total_keys = sum(len(keys) for _, _, keys in sections)
    lines: list[str] = list(_config_header_lines())
    lines.append("{")
    indent = "  "
    idx = 0
    for si, (title, ph, keys) in enumerate(sections):
        if si > 0:
            lines.append("")
        lines.append(f"{indent}// === {title} ===")
        if ph:
            lines.extend(_comment_block(ph, indent))
        for key in keys:
            entry = registry.get(key)
            help_txt = (entry or {}).get("help") if entry else ""
            if help_txt:
                lines.extend(_comment_block(help_txt, indent))
            idx += 1
            comma = "," if idx < total_keys else ""
            key_str = json.dumps(key, ensure_ascii=False)
            val_str = json.dumps(data[key], ensure_ascii=False)
            lines.append(f"{indent}{key_str}: {val_str}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_bootstrap_config_json(data_root: Path, data: dict[str, Any]) -> None:
    """Write first-run ``config.json`` during release bootstrap (not UI-guarded)."""
    path = config_json_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config_jsonc(data), encoding="utf-8")


def materialize_fresh_config_json(install_root: Path, data_root: Path) -> None:
    """Materialize canonical ``config.json`` with every persisted key at code defaults.

    Used on first bundled launch instead of copying a trimmed template. The canonical
    grouped writer (``render_config_jsonc`` + ``AppConfig.to_json()``) is the same path
    the Settings UI uses on save.
    """
    prev_data = os.environ.get("VYVAR_DATA_DIR")
    os.environ["VYVAR_DATA_DIR"] = str(data_root.resolve())
    try:
        cfg = AppConfig(project_root=install_root.resolve())
        write_bootstrap_config_json(data_root, cfg.to_json())
    finally:
        if prev_data is None:
            os.environ.pop("VYVAR_DATA_DIR", None)
        else:
            os.environ["VYVAR_DATA_DIR"] = prev_data


def save_config_json(project_root: Path, data: dict[str, Any]) -> None:
    if not _CONFIG_PERSIST_ALLOWED:
        raise ConfigPersistError(
            "config.json may be persisted only from an explicit UI save action "
            "(wrap the write in config.ui_config_persist()). The pipeline/headless path "
            "must not write config.json; run-effective values belong in provenance."
        )
    write_bootstrap_config_json(project_root, data)


def recommended_vyvar_parallel_workers(*, reserve_ram_gb: float = 1.5) -> int:
    """Single worker count for QC, preprocess, combined, per-frame CSV (base before the RAM cap), alignment, calibrate MP.

    Takes the minimum of the CPU heuristic and the free-RAM estimate (same memory estimate as the per-frame export),
    so that one value is safe across the whole workflow.
    """
    n = os.cpu_count()
    if n is None or n < 1:
        n = 4
    if n <= 1:
        cpu_cap = 1
    else:
        cpu_cap = max(1, min(32, min(n - 1, 16, max(1, n // 2))))
    h, wpx = 2048, 2048
    per_worker = max(int(h * wpx * 4 * 3), 1)
    try:
        import psutil

        reserve = int(max(0.0, float(reserve_ram_gb)) * (1024**3))
        avail = int(psutil.virtual_memory().available) - reserve
        if avail <= 0:
            ram_cap = 1
        else:
            ram_cap = max(1, min(32, avail // per_worker))
    except Exception:  # noqa: BLE001
        ram_cap = 32
    return max(1, min(32, min(cpu_cap, ram_cap)))


#: CODE-REVIEW-2026-10 step 7: the numeric config.json keys with their allowed range [lo, hi] -
#: the ONE place a range lives. The loader clamps to it (with a warning), the Settings page and
#: dev/scripts/validate_config.py read it through params_registry. Type = type of the default.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "masterdark_validity_days": (1, 3650),
    "masterflat_validity_days": (1, 3650),
    "calibration_master_ccd_temp_tolerance_c": (0.01, 20.0),
    "lc_quality_min_frames": (3, 500),
    "comp_trust_min_comps": (3, 20),
    "preprocess_sky_surface_order": (0, 2),
    "osc_channel_binning": (1, 4),
    "aperture_fwhm_factor": (0.25, 6.0),
    "annulus_inner_fwhm": (1.0, 10.0),
    "annulus_outer_fwhm": (1.5, 12.0),
    "detection_threshold_sigma": (DETECTION_THRESHOLD_SIGMA_MIN, DETECTION_THRESHOLD_SIGMA_MAX),
    "per_frame_mp_reserve_ram_gb": (0.0, 64.0),
    "alignment_max_stars": (10, 5000),
    "phase01_comparison_n_comp_min": (2, 12),
    "phase01_comparison_n_comp_max": (3, 20),
    "snr_min": (3.0, 1000.0),
    "phase01_comparison_max_comp_rms": (0.01, 0.5),
    "phase01_comparison_min_frames_frac": (0.05, 0.95),
    "comp_max_delta_bprp": (0.0, 5.0),
    "comp_isolation_fwhm": (1.0, 12.0),
    "comp_max_slope_mmag_hr": (0.0, 500.0),
    "comp_catalog_offset_max_mag": (0.0, 2.0),
    "per_frame_sat_min_clean_frac": (0.1, 1.0),
    "masterstar_best_of_n": (1, 25),
    "auto_fwhm_k_factor": (1.0, 4.0),
}

#: On / off config.json keys.
BOOL_KEYS: tuple[str, ...] = (
    "hrd_online_enrich_enabled",
    "hrd_nss_category_enabled",
    "save_lightcurve_png",
    "epsf_auto_run",
    "epsf_export_auto",
    "gs11_dilution_enabled",
    "tess_enabled",
    "qc_after_calibrate_enabled",
    "auto_fwhm_enabled",
)

_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _coerce_number(key: str, raw: Any, kind: type, lo: float, hi: float) -> float | int | None:
    """``raw`` as ``kind`` clamped to [lo, hi]; None (with a warning) when it is not a number."""
    try:
        if isinstance(raw, bool):
            raise ValueError
        v = float(raw)
        if not math.isfinite(v):
            raise ValueError
    except (TypeError, ValueError):
        logging.warning("config.json: %s = %r is not a number; the default is used", key, raw)
        return None
    if kind is int:
        v = float(int(v))
    c = max(float(lo), min(float(hi), v))
    if c != v:
        logging.warning("config.json: %s = %r is outside [%s, %s]; %s is used", key, raw, lo, hi, kind(c))
    return kind(c)


def _coerce_bool(key: str, raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    word = str(raw).strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    logging.warning("config.json: %s = %r is not true / false; the default is used", key, raw)
    return None


@dataclass(slots=True)
class AppConfig:
    """Central application config: dataclass defaults, overridden by config.json on load."""

    # src_py/config.py -> repo root is parent.parent; project_root anchors config.json,
    # Archive/, CalibrationLibrary/, vyvar.sqlite3, catalogs/ discovery (all at repo root).
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Calibration validity defaults (days)
    masterdark_validity_days: int = 90
    masterflat_validity_days: int = 200

    #: Max |DeltaT| ( degC) when matching library dark masters to light ``CCD_TEMP`` (``find_best_calibration_library_path``).
    calibration_master_ccd_temp_tolerance_c: float = 0.5

    #: Local Gaia DR3 SQLite (table ``gaia_dr3``), built by ``catalogs/build_gaia_catalog.py``
    #: (default output ``catalogs/vyvar_gaia_dr3.db``). Empty = no catalog (LIMITED MODE).
    gaia_db_path: str = ""
    #: PLATESOLVE-2026-10 blind plate-solving index (astro_index build); empty = next to the Gaia DB.
    astrometry_index_dir: str = ""

    #: HRD extreme objects: online lookups (Gaia TAP teff/logg and SIMBAD otype) for the candidates.
    hrd_online_enrich_enabled: bool = True
    #: Include Gaia NSS binary candidates in HRD Stage-1/2 (default off; reversible flag).
    hrd_nss_category_enabled: bool = False

    #: Objects catalog DB (tables ``vsx_data`` and ``exoplanet_data``; built by catalogs/objects_make.py).
    objects_db_path: str = "catalogs/vyvar_objects.db"
    #: VSX types to leave unmeasured (token-match; empty = inactive). See vsx_type_scope.py.
    vsx_out_of_scope_types: list[str] = field(default_factory=list)

    #: Aperture photometry: write PNG files (light curve, cutout, field map). ``False`` = CSV + summary only; the UI plots from the CSV.
    save_lightcurve_png: bool = False
    #: LC-quality frame floor for ``classify_lc_quality`` (aperture-photometry summary).
    lc_quality_min_frames: int = 20
    #: Trust-only comp floor (RED below); comparison-star selection uses ``phase01_comparison_n_comp_min``.
    comp_trust_min_comps: int = 5
    # Export reports (AAVSO + VAR.ASTRO.CZ)
    observer_name: str = "Unknown Observer"
    observer_code: str = ""
    #: User overrides: filter/setup name (uppercase key) -> AAVSO FILT code (e.g. ``"MYLUM": "CV"``).
    aavso_filter_map: dict[str, str] = field(default_factory=dict)
    # Observer location - used for BJD, airmass, lunar context
    #: Run-time only (not in config.json): LOCATION.ID of the default site (IS_DEFAULT), or the site
    #: chosen for the current run; 0 = no default site in the DB.
    observer_location_id: int = 0
    #: Mirror of the site's LOCATION row (NaN = no site; CODE-REVIEW step 7B: no built-in default site).
    observer_lat: float = float("nan")  # degrees N
    observer_lon: float = float("nan")  # degrees E
    observer_alt_m: float = float("nan")  # metres above sea level
    observer_location_name: str = ""  # display name; filled from DB on load when id > 0
    #: Night-run automatic ePSF stage after aperture photometry (EPSF-CHAIN-01B).
    #: Default OFF.
    #: ``NightRunParams.epsf`` True/False overrides; None reads this key.
    epsf_auto_run: bool = False
    #: EPSF-TARGETS-2026-10: targets RUN ePSF measures - ``ALL`` (every VSX target with aperture photometry,
    #: after ``vsx_out_of_scope_types``) or comma-separated VSX types / star names / catalog ids, ``*`` wildcard
    #: (e.g. ``EA, EB, EW`` or ``E*``). The ePSF tab can override it per run.
    epsf_targets: str = "ALL"
    #: EPSF-EXPORT-2026-10: write the ePSF light-curve files for the user (lightcurves_reports/epsf: bjd_tdb,
    #: delta_mag, err + header) automatically when RUN ePSF finishes. Off: the ePSF tab's button writes them.
    epsf_export_auto: bool = True
    #: Per calibrated light: subtract source-masked sigma-clipped polynomial sky surface in preprocess (0=off, default 2).
    preprocess_sky_surface_order: int = 2
    #: OSC: post-extraction NxN average binning on each channel plane (1-4; default 2). Total scale vs raw = 2 x N.
    osc_channel_binning: int = 2
    # NOTE: These are in units of **Gaussian FWHM** (not moment-FWHM).
    # Aperture/annulus radii are computed as factor x fwhm_gaussian_px.
    #: Legacy single aperture factor - used where multi-aperture (B+C) is not active.
    aperture_fwhm_factor: float = 1.35
    annulus_inner_fwhm: float = 2.7
    annulus_outer_fwhm: float = 5.2

    #: CFG-CLEAN 2026-10: the ONE star-detection threshold, N x sky sigma, used by every DAO
    #: detection (QC after calibration, alignment, VYVAR plate solve, MASTERSTAR pass 1, per-frame
    #: catalogs, zone-classifier T1). Replaces sips_dao_threshold_sigma, masterstar_dao_threshold_sigma,
    #: dao_detection_n_equiv, alignment_detection_sigma and qc_dao_detection_sigma. Clamp 2.0-20.0.
    detection_threshold_sigma: float = DETECTION_THRESHOLD_SIGMA_DEFAULT

    #: Parallelism (QC, preprocess, combined, per-frame CSV, alignment, calibrate MP): one value computed
    #: in ``__post_init__``; not in ``config.json``. Run-time override: ``VYVAR_PARALLEL_WORKERS``.
    qc_preprocess_workers: int = 1
    #: RAM (GB) kept free when capping the parallel workers (``psutil``).
    per_frame_mp_reserve_ram_gb: float = 1.5

    #: Frame alignment (DAO detection ladder): max brightest sources detected per frame (astroalign picks a subset).
    alignment_max_stars: int = 160

    #: Comparison-star selection (``photometry_core.select_comparison_stars_per_target``).
    #: No magnitude window and no maximum distance: candidates are ranked RMS -> colour -> distance.
    phase01_comparison_n_comp_min: int = 3
    phase01_comparison_n_comp_max: int = 8
    #: SNR-MIN-2026-10: faint limit for every star (targets, comparison and check stars):
    #: aperture SNR on MASTERSTAR below this -> not measured; a frame where the target SNR
    #: falls below it is not exported. Default 10 (sigma ~ 0.11 mag).
    snr_min: float = 10.0
    phase01_comparison_max_comp_rms: float = 0.1
    phase01_comparison_min_frames_frac: float = 0.2
    #: Max |DeltaBP-RP| (Gaia) between target and comparison star (hard filter of the selection).
    comp_max_delta_bprp: float = 0.79
    #: Comparison-star colour classes 1-3: rows {"bprp": |dBP-RP| limit (Gaia)}; the next rung is
    #: ``comp_max_delta_bprp`` and above it the red-target T4 fallback (CFG-REVIEW-2026-10: the 4th row
    #: only mirrored comp_max_delta_bprp).
    comp_color_tiers: list[dict[str, float]] = field(
        default_factory=lambda: [{"bprp": 0.15}, {"bprp": 0.30}, {"bprp": 0.55}]
    )
    #: Aperture photometry: apply BP-RP colour-term correction (``auto`` = on for B/V/Rc broadband, off for L/Clear).
    apply_color_term: str = "auto"
    #: A comparison candidate whose nearest catalogue neighbour is closer than this x FWHM is a blend and
    #: is not a comparison star (``photometry_comp._select_comps_by_rms_then_color``). Was snr_cog_isolation_fwhm.
    comp_isolation_fwhm: float = 3.0
    #: Second-order extinction: ``off`` | ``literature``.
    k2_mode: str = "literature"
    #: Optional per-band k'' overrides (mag/airmass/BP-RP); empty dict uses ``k2_extinction`` converter.
    k2_defaults_bprp: dict[str, float] = field(default_factory=dict)

    #: Aperture photometry: exclude comparison stars with |linear slope| above this (mmag/hr) in stability check.
    comp_max_slope_mmag_hr: float = 5.0
    #: COMP-CATOFF-2026-10: exclude (and replace from the reserve) a comparison candidate whose
    #: catalogue - measured magnitude differs from the median of the candidates by more than this
    #: (mag); 0 = off.
    comp_catalog_offset_max_mag: float = 0.20

    # GS11 - Flux dilution correction
    gs11_dilution_enabled: bool = False

    #: Minimum fraction of clean (unsaturated) frames required to measure a target
    #: (per-frame saturation is always on, SAT-LIN-2026-10). Clamped to [0.1, 1.0].
    per_frame_sat_min_clean_frac: float = 0.5

    #: MASTERSTAR: stack N frames and pick best N for ePSF/catalog build.
    masterstar_best_of_n: int = 10

    #: ``True`` = download and analyse TESS FFI through lightkurve (TessCut), UI + ``tess_runner`` + pipeline hook.
    #: ``False`` = off - nothing is downloaded; log ``[TESS] skipped``.
    tess_enabled: bool = False

    #: Post-calibration QC metrics (HFR, stars, sky) on each calibrated light; measurements only.
    qc_after_calibrate_enabled: bool = True

    #: FITS QA dashboard: derive the default FWHM limit from the MAD (median + k x sigma_MAD).
    auto_fwhm_enabled: bool = True
    #: Night FWHM limit = median + k x sigma_MAD; k clamped to 1..4 on load.
    auto_fwhm_k_factor: float = 3.0

    # Paths derived from config.json (must stay after all init=True fields for dataclass(slots=True)).
    data_root: Path = field(init=False)
    archive_root: Path = field(init=False)
    calibration_library_root: Path = field(init=False)
    database_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = resolve_data_root(self.project_root)
        data = load_config_json(self.data_root)

        # A blank ("" / whitespace) path value means "use the project-root default" -- the
        # same as omitting the key. This keeps a relocated/fresh install working: the
        # installer and hand-editors may blank these to drop the author's absolute paths,
        # and Path("") would otherwise resolve to "." (e.g. database_path="." would make
        # sqlite3.connect fail on first run).
        def _path_or_default(key: str, default: Path) -> Path:
            raw = str(data.get(key) or "").strip()
            if not raw:
                return default
            resolved = resolve_config_path(raw, self.data_root)
            return Path(resolved) if resolved else default

        self.archive_root = _path_or_default("archive_root", self.data_root / "Archive")
        self.calibration_library_root = _path_or_default(
            "calibration_library_root", self.data_root / "CalibrationLibrary"
        )
        self.database_path = _path_or_default(
            "database_path", self.data_root / "vyvar.sqlite3"
        )
        self.gaia_db_path = resolve_config_path(
            str(data.get("gaia_db_path", "") or "").strip(),
            self.data_root,
        )
        self.astrometry_index_dir = resolve_config_path(
            str(data.get("astrometry_index_dir", "") or "").strip(),
            self.data_root,
        )
        self.objects_db_path = resolve_config_path(
            str(data.get("objects_db_path", self.objects_db_path) or self.objects_db_path or "").strip(),
            self.data_root,
        )

        # Table-driven keys: a value that is not a number / not true-false keeps the default,
        # a number outside its range is clamped; both with a warning naming the key.
        for key in BOOL_KEYS:
            if key in data:
                b = _coerce_bool(key, data[key])
                if b is not None:
                    setattr(self, key, b)
        for key, (lo, hi) in PARAM_RANGES.items():
            if key in data:
                kind = _PARAM_KINDS[key]
                v = _coerce_number(key, data[key], kind, lo, hi)
                if v is not None:
                    setattr(self, key, v)

        # Rules between keys.
        if self.annulus_outer_fwhm <= self.annulus_inner_fwhm:
            logging.warning("config.json: annulus_outer_fwhm must exceed annulus_inner_fwhm; inner + 1 is used")
            self.annulus_outer_fwhm = self.annulus_inner_fwhm + 1.0
        if int(self.phase01_comparison_n_comp_max) < int(self.phase01_comparison_n_comp_min):
            self.phase01_comparison_n_comp_max = int(self.phase01_comparison_n_comp_min)
        if int(self.comp_trust_min_comps) > int(self.phase01_comparison_n_comp_max):
            self.comp_trust_min_comps = int(self.phase01_comparison_n_comp_max)

        # Structured and text keys.
        _voos = data.get("vsx_out_of_scope_types", self.vsx_out_of_scope_types)
        if _voos is None:
            self.vsx_out_of_scope_types = []
        elif isinstance(_voos, str):
            self.vsx_out_of_scope_types = [p.strip() for p in _voos.split(",") if p.strip()]
        elif isinstance(_voos, (list, tuple)):
            self.vsx_out_of_scope_types = [str(p).strip() for p in _voos if str(p).strip()]
        else:
            self.vsx_out_of_scope_types = []
        _obn = data.get("observer_name")
        if _obn is None or str(_obn).strip() == "":
            _obn = self.observer_name
        self.observer_name = str(_obn or "").strip() or "Unknown Observer"
        self.observer_code = str(data.get("observer_code", self.observer_code) or "").strip()
        _ffm = data.get("aavso_filter_map", {})
        if isinstance(_ffm, dict):
            self.aavso_filter_map = {
                str(k).strip().upper(): str(v).strip()
                for k, v in _ffm.items()
                if str(k).strip() and str(v).strip()
            }
        else:
            self.aavso_filter_map = {}
        self.epsf_targets = str(data.get("epsf_targets", self.epsf_targets) or "ALL").strip() or "ALL"
        _act = str(data.get("apply_color_term", self.apply_color_term) or "auto").strip().lower()
        if _act in _TRUE_WORDS:
            self.apply_color_term = "on"
        elif _act in _FALSE_WORDS:
            self.apply_color_term = "off"
        else:
            self.apply_color_term = "auto"
        _k2m = str(data.get("k2_mode", self.k2_mode) or "literature").strip().lower()
        self.k2_mode = "off" if _k2m in _FALSE_WORDS or _k2m == "none" else "literature"
        _k2o = data.get("k2_defaults_bprp", self.k2_defaults_bprp)
        if isinstance(_k2o, dict):
            parsed: dict[str, float] = {}
            for k, v in _k2o.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    parsed[str(k)] = fv
            self.k2_defaults_bprp = parsed
        # comp_color_tiers: colour classes 1-3. Older files carry a 4th row (it only mirrored
        # comp_max_delta_bprp) and a per-row weight "w" (never used: Broeg weights) - both are dropped.
        _cct_raw = data.get("comp_color_tiers")
        if isinstance(_cct_raw, list) and _cct_raw:
            _rows: list[dict[str, float]] = []
            for _row in _cct_raw[:3]:
                try:
                    _rows.append({"bprp": max(0.02, min(5.0, float(_row["bprp"])))})
                except (KeyError, TypeError, ValueError):
                    _rows = []
                    break
            if len(_rows) == 3:
                self.comp_color_tiers = _rows
            else:
                logging.warning("config.json: comp_color_tiers needs 3 rows {\"bprp\": limit}; defaults used")

        # Run-time values (not in config.json).
        self.qc_preprocess_workers = int(
            recommended_vyvar_parallel_workers(reserve_ram_gb=float(self.per_frame_mp_reserve_ram_gb))
        )
        # SETTINGS-2026-10: the site is not stored in config.json; the default site is the
        # LOCATION row with IS_DEFAULT = 1 (Database Explorer), a run may choose another one.
        from database import get_default_observer_location_id  # noqa: PLC0415

        self.observer_location_id = get_default_observer_location_id(str(self.database_path))
        # WAVE-B STEP 5 (DELETE-DB-DUP): observer_lat/lon/alt_m/name are DB-authoritative
        # (draft LOCATION via observer_location_id). They are no longer loaded from or saved to
        # config.json; the fields remain as run-time hydrated mirrors (block below) with the
        # NaN (no site) when the DB has none. Science reads the draft LOCATION row, not cfg.
        if self.observer_location_id > 0:
            try:
                from database import get_observer_location_by_id

                loc = get_observer_location_by_id(
                    str(self.database_path), int(self.observer_location_id)
                )
                if loc is not None:
                    # DB LOCATION is authoritative (WAVE-B STEP 5): overwrite the mirror
                    # unconditionally when a row is found.
                    self.observer_location_name = str(loc.get("name") or "")
                    self.observer_lat = float(loc.get("lat", 0.0) or 0.0)
                    self.observer_lon = float(loc.get("lon", 0.0) or 0.0)
                    self.observer_alt_m = float(loc.get("alt_m", 0.0) or 0.0)
            except (sqlite3.Error, TypeError, ValueError) as exc:
                logging.getLogger(__name__).warning(
                    "Observer location DB hydrate failed (location_id=%s); "
                    "no observing site until one is chosen: %s",
                    self.observer_location_id,
                    exc,
                )

    # --- structured-key accessors (WAVE-B STEP 4) ---------------------------------- #
    def comp_tier_bprp_limits(self) -> list[float]:
        """|dBP-RP| colour limit per comp tier, row order, from ``comp_color_tiers``."""
        return [float(t.get("bprp", 0.0)) for t in self.comp_color_tiers]

    def to_json(self) -> dict[str, Any]:
        """The config.json keys (paths, BOOL_KEYS, PARAM_RANGES, structured and text keys)."""
        out: dict[str, Any] = {
            "archive_root": str(self.archive_root),
            "calibration_library_root": str(self.calibration_library_root),
            "database_path": str(self.database_path),
            "gaia_db_path": str(self.gaia_db_path or ""),
            "astrometry_index_dir": str(self.astrometry_index_dir or ""),
            "objects_db_path": str(self.objects_db_path or ""),
        }
        for key in BOOL_KEYS:
            out[key] = bool(getattr(self, key))
        for key in PARAM_RANGES:
            v = getattr(self, key)
            out[key] = int(v) if isinstance(v, int) else float(v)
        out.update(
            {
                "vsx_out_of_scope_types": list(self.vsx_out_of_scope_types),
                "observer_name": str(self.observer_name),
                "observer_code": str(self.observer_code),
                "aavso_filter_map": dict(self.aavso_filter_map),
                "epsf_targets": str(self.epsf_targets),
                "apply_color_term": str(self.apply_color_term),
                "k2_mode": str(self.k2_mode),
                "k2_defaults_bprp": dict(self.k2_defaults_bprp),
                "comp_color_tiers": [{"bprp": float(t.get("bprp", 0.0))} for t in self.comp_color_tiers],
            }
        )
        return out

    # Backward-compatible alias (some callers expect to_dict()).
    def to_dict(self) -> dict[str, Any]:
        return self.to_json()

    def ensure_base_dirs(self) -> None:
        """Create base directories required by file-first workflow."""
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.calibration_library_root.mkdir(parents=True, exist_ok=True)



#: int / float per PARAM_RANGES key, from the dataclass default (a value is loaded as that type).
_PARAM_KINDS: dict[str, type] = {
    f.name: type(f.default) for f in _dc_fields(AppConfig) if f.name in PARAM_RANGES
}
assert set(_PARAM_KINDS) == set(PARAM_RANGES), sorted(set(PARAM_RANGES) - set(_PARAM_KINDS))

def compute_field_density(n_masterstar_stars: int, chip_w_px: int, chip_h_px: int) -> float:
    """Return the field density in stars/Mpx."""
    mpx = (float(chip_w_px) * float(chip_h_px)) / 1_000_000.0
    if mpx <= 0 or n_masterstar_stars < 0:
        return 0.0
    return float(n_masterstar_stars) / float(mpx)


def classify_field_density(density: float, sparse_th: float, dense_th: float) -> str:
    """Return ``sparse`` / ``normal`` / ``dense``."""
    if density < float(sparse_th):
        return "sparse"
    if density <= float(dense_th):
        return "normal"
    return "dense"
