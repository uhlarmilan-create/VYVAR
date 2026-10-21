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
        "phase01_flux_col removed 2026-10-19 (CFG-REVIEW-2026-10): Phase 1 runs before RUN ePSF, so dao_flux was its only usable value"
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
    "frame_align_residual_gate_enabled": "ALIGN-VERIFY-2026-10: the circular Phase-2A alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
    "frame_align_residual_max_frac": "ALIGN-VERIFY-2026-10: the circular Phase-2A alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
    "frame_align_residual_min_keep_frames": "ALIGN-VERIFY-2026-10: the circular Phase-2A alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
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
        "phase01_match_radius_arcsec removed 2026-07; Phase 0 uses catalog_id identity join (PHASE0-IDENTITY-GATE)"
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
        "export_err_mode removed 2026-09-01 CONSOLIDATE-01D; exported err is always ERR-CALIB calibrated"
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
        "phase01_comparison_max_psf_chi2 removed 2026-10-10 (CLEAN-STEP5-2026-10); Phase-1 comparison selection never applied a PSF chi2 cut (proc CSVs are DAO-era)"
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
    """Jednotny pocet workerov pre QC, preprocess, combined, per-frame CSV (zaklad pred RAM stropom), alignment, calibrate MP.

    Berie minimum z CPU-heuristiky a odhadu podla volnej RAM (rovnaky odhad pamate ako pri per-frame exporte),
    aby jedna hodnota bola bezpecna v celom workflow.
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


@dataclass(slots=True)
class AppConfig:
    """Central application config.

    SQLite schema is intentionally not defined yet.
    """

    # src_py/config.py -> repo root is parent.parent; project_root anchors config.json,
    # Archive/, CalibrationLibrary/, vyvar.sqlite3, catalogs/ discovery (all at repo root).
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Calibration validity defaults (days)
    masterdark_validity_days: int = 90
    masterflat_validity_days: int = 200

    #: Field diameter floor in degrees (not read from config.json); the Gaia cone of a chip comes from
    #: FOCALLEN + PIXSIZE + NAXIS (``utils.catalog_cone_radius_deg_from_optics``).
    plate_solve_fov_deg: float = 1.0

    #: Max |DeltaT| ( degC) when matching library dark masters to light ``CCD_TEMP`` (``find_best_calibration_library_path``).
    calibration_master_ccd_temp_tolerance_c: float = 0.5

    #: Local Gaia DR3 SQLite (table ``gaia_dr3``), built by ``catalogs/build_gaia_catalog.py``
    #: (default output ``catalogs/vyvar_gaia_dr3.db``). Empty = no catalog (LIMITED MODE).
    gaia_db_path: str = ""
    #: PLATESOLVE-2026-10 blind plate-solving index (astro_index build); empty = next to the Gaia DB.
    astrometry_index_dir: str = ""

    #: HRD extreme-object online enrichment (Gaia TAP teff/logg for candidates only).
    hrd_online_enrich_enabled: bool = True
    #: HRD SIMBAD otype refinement for extreme candidates (additive label only).
    hrd_simbad_enrich_enabled: bool = True
    #: Max Stage-1 extreme candidates sent to online enrichment (1..100).
    hrd_enrich_max_candidates: int = 20
    #: Per-attempt Gaia TAP timeout (seconds) for HRD online enrichment.
    hrd_enrich_tap_timeout_s: float = 60.0  # HRD-ENRICH-2026-10 (was 20 s: timed out on draft 526)
    #: HRD parallax floor (mas) for M_G reliability; SNR is the primary quality filter.
    hrd_parallax_min_mas: float = 0.15
    #: HRD parallax SNR floor for M_G reliability.
    hrd_parallax_snr_min: float = 5.0
    #: Max table rows per Stage-2 category label (1..20).
    hrd_max_per_category: int = 3
    #: Min reserved Stage-1 enrich slots per candidate net (0..20); NSS net filled last.
    hrd_min_per_net: int = 4
    #: Include Gaia NSS binary candidates in HRD Stage-1/2 (default off; reversible flag).
    hrd_nss_category_enabled: bool = False
    #: Gaia DSC probability floor for HRD "likely" white-dwarf tier (clamp 0.5..1.0).
    hrd_dsc_confirm_prob: float = 0.90

    #: Objects catalog DB (tables ``vsx_data`` and ``exoplanet_data``; built by catalogs/objects_make.py).
    objects_db_path: str = "catalogs/vyvar_objects.db"
    #: VSX types to leave unmeasured (token-match; empty = inactive). See vsx_type_scope.py.
    vsx_out_of_scope_types: list[str] = field(default_factory=list)

    #: Per-detection exoplanet host cross-match tolerance (arcsec); informational label only.
    exoplanet_match_max_sep_arcsec: float = 3.0


    #: Faza 2A: ukladat PNG (lightcurve, cutout, field map). ``False`` = len CSV + summary; UI pouziva Plotly z CSV.
    save_lightcurve_png: bool = False
    #: LC-OUTLIER-01: enable isolated-spike + image-evidence + frame_qc flagging.
    lc_outlier_enabled: bool = True
    #: Isolated-spike residual threshold (|r| > n_sigma); statistical convention.
    lc_outlier_n_sigma: float = 5.0
    #: Adjacent same-sign residual threshold that protects flares/eclipses.
    lc_outlier_adjacent_sigma: float = 3.0
    #: Night MAD-sigma threshold for whole-frame QC (FWHM/elong/sky/ZP).
    lc_outlier_frame_qc_n_sigma: float = 5.0
    #: Image-evidence MAD-sigma vs same-frame similar-flux peers.
    lc_outlier_evidence_n_sigma: float = 5.0
    #: LC-FLAG-ERR-01: err_i > median(err) + n_sigma x 1.4826 x MAD(err).
    lc_high_err_nsigma: float = 5.0
    #: Post-Phase 2A trust flag (GREEN/YELLOW/RED); metadata + report/export notes only.
    trust_flag_enabled: bool = True
    #: LC-quality frame floors for ``classify_lc_quality`` (Phase 2A summary).
    lc_quality_min_frames: int = 20
    lc_quality_short_min_frames: int = 3
    lc_quality_min_normal_frac: float = 0.5
    #: Trust-only comp floor (RED below); Phase-1 selection uses ``phase01_comparison_n_comp_min``.
    comp_trust_min_comps: int = 5
    #: Minimum check-star epochs before trust scatter thresholds apply.
    check_star_min_epochs: int = 5
    #: Sparse-trust CI band thresholds (dev/results/specs/VYVAR_SPARSE_TRUST_SPEC.md).
    sparse_trust_T_green: float = 1.5
    sparse_trust_T_red: float = 4.0
    sparse_trust_X2_RED: float = 0.0004
    #: Artefact floor for check-star selection metric (comp_rms / p2p_rms).
    check_select_rms_floor: float = 1e-4
    #: Phase-1 comp selection: drop candidates with comp_rms below this (isolated_bin artefact).
    comp_select_rms_floor: float = 1e-6
    # Export reports (AAVSO + VAR.ASTRO.CZ)
    observer_name: str = "Unknown Observer"
    observer_code: str = ""
    #: User overrides: filter/setup name (uppercase key) -> AAVSO FILT code (e.g. ``"MYLUM": "CV"``).
    aavso_filter_map: dict[str, str] = field(default_factory=dict)
    # Observer location - used for BJD, airmass, lunar context
    #: Run-time only (not in config.json): LOCATION.ID of the default site (IS_DEFAULT), or the site
    #: chosen for the current run; 0 = no default site in the DB.
    observer_location_id: int = 0
    observer_lat: float = 50.1121658  # degrees N
    observer_lon: float = 14.6982547  # degrees E
    observer_alt_m: float = 275.0  # metres above sea level
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
    #: EPSF-CHI2-LOCUS-01: n_sigma cut on residual from the per-frame (or night)
    #: log10(chi2)=a+k*log10(flux) locus. Statistical convention, not equipment.
    psf_chi2_locus_nsigma: float = 5.0
    #: Per calibrated light: subtract source-masked sigma-clipped polynomial sky surface in preprocess (0=off, default 2).
    preprocess_sky_surface_order: int = 2
    #: When True, bypass the VY_SKYSF idempotency guard and re-subtract sky surface in-place.
    preprocess_sky_surface_force_reapply: bool = False
    #: OSC: post-extraction NxN average binning on each channel plane (1-4; default 2). Total scale vs raw = 2 x N.
    osc_channel_binning: int = 2
    #: Minimum clean stars required to build the ePSF model.
    epsf_min_stars: int = 30
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
    detection_threshold_sigma: float = 4.5
    #: E.2: max DAO-vs-WCS centroid shift (x FWHM) before WCS pixel fallback on matched stars.
    dao_centroid_max_shift_fwhm: float = 1.0
    #: MASTERSTAR katalog: DAO FWHM z najlepsieho zdrojoveho snimku (``best_frame_fwhm_px``), nie median ``VY_FWHM`` v hlavicke.
    masterstar_use_best_frame_fwhm: bool = True
    #: MASTERSTAR DAO pass 2: local annulus sigma at Gaia seed positions (born-owned; INV-DET-FALSEFILL-01).
    masterstar_dao_pass2_sigma: float = 4.0
    #: MASTERSTAR DAO pass 2: max centroid offset [px] for targeted cutout acceptance (INV-DET-FALSEFILL-01 audit).
    masterstar_dao_pass2_center_tol_px: float = 2.0
    #: Pass1 spatial dedup radius [px] before pass2 (iter4 I6).
    masterstar_dao_pass1_dedup_px: float = 0.75
    #: Gaia-complete census + pass2 seed window: maximum G magnitude (default 15.0).
    #: FORCED_SEED admission (MASTERSTAR-GAIA-01): max COM centroid offset from propagated Gaia [px].
    masterstar_forced_seed_centroid_max_px: float = 2.0
    #: FORCED_SEED admission: minimum aperture SNR on local annulus background.
    masterstar_forced_seed_snr_min: float = 4.0
    #: FORCED_SEED / leftover promotion: greedy match radius for leftover detections to Gaia [px].
    masterstar_lock_leftover_radius_px: float = 3.0
    #: Lock-existing assignment: max shift from locked catalog_id xy to detection [px].
    masterstar_lock_pair_tol_px: float = 3.0
    #: Gaia-complete census: edge margin [px] for EDGE state (matches target_depth edge band).
    masterstar_gaia_census_edge_margin_px: float = 10.0
    #: Comp/CT pool: admit FORCED_SEED stars (default off; separate future decision).
    masterstar_forced_seed_comp_pool_enabled: bool = False
    #: DAO-GAIA-ERA-01 (D-A): match radius = k x astrometric residual p95 [px] (validated ~1.7 -> 3 px @ p95~1.78).
    masterstar_dao_match_radius_k: float = 1.7
    #: DAO-GAIA-ERA-01: centroid QA floor/cap when deriving pass2/seed tolerances from residual p95 [px].
    masterstar_dao_centroid_qa_floor_px: float = 1.0
    masterstar_dao_centroid_qa_cap_px: float = 3.0
    #: Empty-sky audit (INV-DET/SEED-FALSEFILL-01): max pure-noise false-accept rate at MS build
    #: (measured on the image mirrored about its background, 2*B - data).
    masterstar_dao_empty_sky_false_accept_max: float = 0.01
    masterstar_dao_empty_sky_target_n: int = 2200

    #: Paralelizmus (QC, preprocess, combined, per-frame CSV, alignment, calibrate MP): jedna hodnota
    #: pocitana v ``__post_init__``; nie v ``config.json``. Runtime override: ``VYVAR_PARALLEL_WORKERS`` alebo legacy env v pipeline.
    qc_preprocess_workers: int = 1
    #: Reserve this much RAM (GB) when capping paralelneho exportu katalogov cez ``psutil`` (nad ramec jednotneho ``_pw``).
    per_frame_mp_reserve_ram_gb: float = 1.5

    #: Frame alignment (DAO detection ladder): max brightest sources detected per frame (astroalign picks a subset).
    alignment_max_stars: int = 160
    #: Frame alignment (``astroalign`` triangle matching): brightest N stars used as control points (independent of detection count).
    alignment_max_control_points: int = 80

    #: DAOStarFinder FWHM (pixels) tuned for SIPS-like centroid search (aperture ~13 -> ~4-5 px FWHM).
    sips_dao_fwhm_px: float = 2.5

    #: Faza 0+1 - vyber porovnavacich hviezd (``photometry_core.select_comparison_stars_per_target``).
    #: No magnitude window and no maximum distance: candidates are ranked RMS -> colour -> distance.
    phase01_comparison_n_comp_min: int = 3
    phase01_comparison_n_comp_max: int = 8
    #: SNR-MIN-2026-10: faint limit for every star (targets, comparison and check stars):
    #: aperture SNR on MASTERSTAR below this -> not measured; a frame where the target SNR
    #: falls below it is not exported. Default 10 (sigma ~ 0.11 mag).
    snr_min: float = 10.0
    phase01_comparison_max_comp_rms: float = 0.1
    #: COMP-RMS-DEF-01-B: LOO mag MAD ceiling is k x photon_sigma(snr_ap_pixscaled).
    #: C3-0 (516 R2, 70 comps): p90(r)=3.67 -> k=5 (rule: 3<=p90<=5 => 5).
    comp_rms_loo_photon_k: float = 5.0
    phase01_comparison_min_dist_arcsec: float = 60.0
    phase01_comparison_min_frames_frac: float = 0.2
    #: Max |DeltaBP-RP| v efektivnom farebnom priestore (hard filter pri vybere comp).
    comp_max_delta_bprp: float = 0.79
    #: Comparison-star colour classes 1-3: rows {"bprp": |dBP-RP| limit (Gaia)}; the next rung is
    #: ``comp_max_delta_bprp`` and above it the red-target T4 fallback (CFG-REVIEW-2026-10: the 4th row
    #: only mirrored comp_max_delta_bprp).
    comp_color_tiers: list[dict[str, float]] = field(
        default_factory=lambda: [{"bprp": 0.15}, {"bprp": 0.30}, {"bprp": 0.55}]
    )
    #: Exponential contamination penalty in comp score: score *= exp(-k * contamination_idx).
    comp_contamination_penalty_k: float = 3.0
    phase01_plate_scale_arcsec_per_px: float = 1.3
    #: Plate scale (arcsec/px) for Phase 2A metadata, GS11, dilution; Set 1 default 1.3.
    plate_scale_arcsec_per_px: float = 1.3
    #: Faza 2A: minimum number of comps used in color-term fit before applying CT (``should_apply_color_term``).
    phase01_ct_min_comp: int = 7
    #: Faza 2A: apply BP-RP colour-term correction (``auto`` = on for B/V/Rc broadband, off for L/Clear).
    apply_color_term: str = "auto"
    #: A comparison candidate whose nearest catalogue neighbour is closer than this x FWHM is a blend and
    #: is not a comparison star (``photometry_comp._select_comps_by_rms_then_color``). Was snr_cog_isolation_fwhm.
    comp_isolation_fwhm: float = 3.0
    #: Second-order extinction: ``off`` | ``literature`` | ``fit_else_literature`` (fit path v2).
    k2_mode: str = "literature"
    #: Optional per-band k'' overrides (mag/airmass/BP-RP); empty dict uses ``k2_extinction`` converter.
    k2_defaults_bprp: dict[str, float] = field(default_factory=dict)
    #: Max CI width factor (hi/lo) to accept photon-transfer g_pt as authority.
    photon_transfer_ci_max_width_factor: float = 3.0
    #: Faza 2A: BP-RP tolerance (mag) when testing target vs comp range before applying CT (0 = strict).
    phase01_ct_extrapolation_tol: float = 0.0

    #: Faza 2A: exclude comparison stars with |linear slope| above this (mmag/hr) in stability check.
    comp_max_slope_mmag_hr: float = 5.0
    #: Minimum |slope|/stderr (sigma) to treat a post-common-mode residual slope as real (stability check).
    comp_slope_significance_k: float = 3.0
    #: COMP-CATOFF-2026-10: exclude (and replace from the reserve) a comparison candidate whose
    #: catalogue - measured magnitude differs from the median of the candidates by more than this
    #: (mag); 0 = off.
    comp_catalog_offset_max_mag: float = 0.20

    # GS11 - Flux dilution correction
    gs11_dilution_enabled: bool = False
    gs11_dilution_aperture_arcsec: float = 0.0
    gs11_dilution_mag_limit_delta: float = 5.0
    gs11_target_min_dilution: float = 0.50


    #: Minimum fraction of clean (unsaturated) frames required to measure a target
    #: (per-frame saturation is always on, SAT-LIN-2026-10). Clamped to [0.1, 1.0].
    per_frame_sat_min_clean_frac: float = 0.5

    #: CCD gain (e-/ADU) - used in noise model / SNR estimates.
    gain: float = 1.0
    #: CCD read noise (e-) - used in noise model.
    read_noise: float = 10.0

    #: Number of random empty apertures per frame/radius for ``sigma_bkg_ap`` (clamped 16..256).
    err_empty_apertures_n: int = 64
    #: Minimum valid empty apertures before Howell fallback for that frame.
    err_empty_apertures_min: int = 16

    #: MASTERSTAR: stack N frames and pick best N for ePSF/catalog build.
    masterstar_best_of_n: int = 10


    #: Aperture correction: reject comp stars with scatter above this (mag).
    #: Phase 1 comp gate: reject comps with FWHM > this factor x field median FWHM.
    #: Radius (px) for the comparison contamination index (neighbour flux within it / own flux);
    #: a soft column used by check-star selection, not a comparison gate.
    phase01_comparison_isolation_radius_px: float = 25.0
    #: Sensor frame dimensions in pixels (used when FITS NAXIS1/2 unavailable).
    frame_width_px: int = 2082
    frame_height_px: int = 1397

    #: Jednotny vnutorny okraj cipu (px) pre **celu Fazu 0+1**: aktivne premenne, porovnavacie hviezdy aj suspected.
    #: Hviezdy s ``x,y`` blizsie ako tento pocet pixelov od okraja referencneho pola sa neberu (zmiernuje artefakty
    #: pri zarovnani / posune pola / okrajoch). ``0`` = vypnute (cely cip). Predvolene 50 px.
    phase01_chip_interior_margin_px: int = 50


    #: ``True`` = stahovanie/analyza TESS FFI cez lightkurve (TessCut), UI + ``tess_runner`` + pipeline hook.
    #: ``False`` = vypnute - ziadne stahovanie; log ``[TESS] preskocene``. Zapnut: ``"tess_enabled": true`` v ``config.json``.
    tess_enabled: bool = False
    #: TESS-2026-10: largest neighbour light in the TESS aperture (fraction of the target's)
    #: for a RELIABLE period verdict. TESS pixels are 21 arcsec.
    tess_contam_max: float = 0.10

    #: COMP-POOL-01 Stage 2 / COMP-ADMIT-03: derived path now only drops known variables;
    #: scatter/colour/distance enter continuous weights (``comp_weights``).
    comp_pool_derived_admission: bool = True
    #: Plan/spatial comparison pool size. ``0`` = uncapped (COMP-POOL-01); legacy default was 150.
    comparison_stars_pool_n: int = 0
    #: FORCED-PHOT-01: inject MASTERSTAR pool-eligible stars missing from DAO each frame.
    forced_photometry_enabled: bool = True
    #: Extra pixel margin for in-footprint geometry gate on forced rows.
    forced_photometry_margin_px: float = 0.0

    #: Post-calibration QC metrics (HFR, stars, sky) on each calibrated light; measurements only.
    qc_after_calibrate_enabled: bool = True

    #: FITS QA dashboard: odvodzovat predvoleny FWHM limit z MAD (median + kxsigma_MAD).
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

        self.masterdark_validity_days = int(data.get("masterdark_validity_days", 90))
        self.masterflat_validity_days = int(data.get("masterflat_validity_days", 200))

        # ``plate_solve_fov_deg`` is no longer read from JSON - resolved from FITS + DB (see ``resolve_plate_solve_fov_deg_hint``).
        self.plate_solve_fov_deg = 1.0
        try:
            _ctol = float(
                data.get(
                    "calibration_master_ccd_temp_tolerance_c",
                    self.calibration_master_ccd_temp_tolerance_c,
                )
            )
            if math.isfinite(_ctol) and 0.01 <= _ctol <= 20.0:
                self.calibration_master_ccd_temp_tolerance_c = float(_ctol)
        except (TypeError, ValueError):
            pass

        self.gaia_db_path = resolve_config_path(
            str(data.get("gaia_db_path", "") or "").strip(),
            self.data_root,
        )
        self.astrometry_index_dir = resolve_config_path(
            str(data.get("astrometry_index_dir", "") or "").strip(),
            self.data_root,
        )

        self.hrd_online_enrich_enabled = bool(
            data.get("hrd_online_enrich_enabled", self.hrd_online_enrich_enabled)
        )
        self.hrd_simbad_enrich_enabled = bool(
            data.get("hrd_simbad_enrich_enabled", self.hrd_simbad_enrich_enabled)
        )
        try:
            self.hrd_enrich_max_candidates = int(
                data.get("hrd_enrich_max_candidates", self.hrd_enrich_max_candidates)
            )
        except (TypeError, ValueError):
            self.hrd_enrich_max_candidates = 20
        self.hrd_enrich_max_candidates = max(1, min(100, int(self.hrd_enrich_max_candidates)))
        try:
            self.hrd_enrich_tap_timeout_s = float(
                data.get("hrd_enrich_tap_timeout_s", self.hrd_enrich_tap_timeout_s)
            )
        except (TypeError, ValueError):
            self.hrd_enrich_tap_timeout_s = 60.0
        self.hrd_enrich_tap_timeout_s = max(5.0, min(120.0, float(self.hrd_enrich_tap_timeout_s)))
        try:
            self.hrd_parallax_min_mas = float(
                data.get("hrd_parallax_min_mas", self.hrd_parallax_min_mas)
            )
        except (TypeError, ValueError):
            self.hrd_parallax_min_mas = 0.15
        self.hrd_parallax_min_mas = max(0.0, min(10.0, float(self.hrd_parallax_min_mas)))
        try:
            self.hrd_parallax_snr_min = float(
                data.get("hrd_parallax_snr_min", self.hrd_parallax_snr_min)
            )
        except (TypeError, ValueError):
            self.hrd_parallax_snr_min = 5.0
        self.hrd_parallax_snr_min = max(1.0, min(20.0, float(self.hrd_parallax_snr_min)))
        try:
            self.hrd_max_per_category = int(
                data.get("hrd_max_per_category", self.hrd_max_per_category)
            )
        except (TypeError, ValueError):
            self.hrd_max_per_category = 3
        self.hrd_max_per_category = max(1, min(20, int(self.hrd_max_per_category)))
        try:
            self.hrd_min_per_net = int(data.get("hrd_min_per_net", self.hrd_min_per_net))
        except (TypeError, ValueError):
            self.hrd_min_per_net = 4
        self.hrd_min_per_net = max(0, min(20, int(self.hrd_min_per_net)))
        self.hrd_nss_category_enabled = bool(
            data.get("hrd_nss_category_enabled", self.hrd_nss_category_enabled)
        )
        try:
            self.hrd_dsc_confirm_prob = float(
                data.get("hrd_dsc_confirm_prob", self.hrd_dsc_confirm_prob)
            )
        except (TypeError, ValueError):
            self.hrd_dsc_confirm_prob = 0.90
        self.hrd_dsc_confirm_prob = max(0.5, min(1.0, float(self.hrd_dsc_confirm_prob)))


        self.objects_db_path = resolve_config_path(
            str(data.get("objects_db_path", self.objects_db_path) or self.objects_db_path or "").strip(),
            self.data_root,
        )
        _voos = data.get("vsx_out_of_scope_types", self.vsx_out_of_scope_types)
        if _voos is None:
            self.vsx_out_of_scope_types = []
        elif isinstance(_voos, str):
            self.vsx_out_of_scope_types = [p.strip() for p in _voos.split(",") if p.strip()]
        elif isinstance(_voos, (list, tuple)):
            self.vsx_out_of_scope_types = [str(p).strip() for p in _voos if str(p).strip()]
        else:
            self.vsx_out_of_scope_types = []
        try:
            _exo_sep = float(
                data.get("exoplanet_match_max_sep_arcsec", self.exoplanet_match_max_sep_arcsec)
            )
            if math.isfinite(_exo_sep):
                self.exoplanet_match_max_sep_arcsec = max(0.5, min(30.0, _exo_sep))
        except (TypeError, ValueError):
            self.exoplanet_match_max_sep_arcsec = 3.0

        try:
            self.per_frame_mp_reserve_ram_gb = float(
                data.get("per_frame_mp_reserve_ram_gb", self.per_frame_mp_reserve_ram_gb)
            )
            if not math.isfinite(self.per_frame_mp_reserve_ram_gb) or self.per_frame_mp_reserve_ram_gb < 0:
                self.per_frame_mp_reserve_ram_gb = 1.5
        except (TypeError, ValueError):
            self.per_frame_mp_reserve_ram_gb = 1.5

        _pw = int(
            recommended_vyvar_parallel_workers(reserve_ram_gb=float(self.per_frame_mp_reserve_ram_gb))
        )
        self.qc_preprocess_workers = _pw

        try:
            self.alignment_max_stars = max(
                10, min(5000, int(data.get("alignment_max_stars", self.alignment_max_stars)))
            )
        except (TypeError, ValueError):
            self.alignment_max_stars = 200
        try:
            self.alignment_max_control_points = max(
                12,
                min(
                    500,
                    int(data.get("alignment_max_control_points", self.alignment_max_control_points)),
                ),
            )
        except (TypeError, ValueError):
            self.alignment_max_control_points = 80
        # CFG-CLEAN 2026-10: single detection threshold (N x sky sigma) for every DAO detection.
        self.detection_threshold_sigma = resolve_detection_threshold_sigma(
            data.get("detection_threshold_sigma", self.detection_threshold_sigma)
        )
        try:
            self.sips_dao_fwhm_px = float(data.get("sips_dao_fwhm_px", self.sips_dao_fwhm_px))
            if not math.isfinite(self.sips_dao_fwhm_px) or self.sips_dao_fwhm_px <= 0:
                self.sips_dao_fwhm_px = 2.5
        except (TypeError, ValueError):
            self.sips_dao_fwhm_px = 2.5
        self.sips_dao_fwhm_px = max(1.0, min(8.0, float(self.sips_dao_fwhm_px)))

        self.qc_after_calibrate_enabled = bool(
            data.get("qc_after_calibrate_enabled", self.qc_after_calibrate_enabled)
        )

        self.auto_fwhm_enabled = bool(data.get("auto_fwhm_enabled", self.auto_fwhm_enabled))
        try:
            self.auto_fwhm_k_factor = float(data.get("auto_fwhm_k_factor", self.auto_fwhm_k_factor))
        except (TypeError, ValueError):
            self.auto_fwhm_k_factor = 3.0
        if not math.isfinite(float(self.auto_fwhm_k_factor)):
            self.auto_fwhm_k_factor = 3.0
        self.auto_fwhm_k_factor = max(1.0, min(4.0, float(self.auto_fwhm_k_factor)))


        self.save_lightcurve_png = bool(data.get("save_lightcurve_png", self.save_lightcurve_png))
        self.lc_outlier_enabled = bool(data.get("lc_outlier_enabled", self.lc_outlier_enabled))
        try:
            _los = float(data.get("lc_outlier_n_sigma", self.lc_outlier_n_sigma))
            self.lc_outlier_n_sigma = _los if math.isfinite(_los) and _los > 0 else 5.0
        except (TypeError, ValueError):
            self.lc_outlier_n_sigma = 5.0
        try:
            _loa = float(data.get("lc_outlier_adjacent_sigma", self.lc_outlier_adjacent_sigma))
            self.lc_outlier_adjacent_sigma = _loa if math.isfinite(_loa) and _loa > 0 else 3.0
        except (TypeError, ValueError):
            self.lc_outlier_adjacent_sigma = 3.0
        try:
            _lof = float(data.get("lc_outlier_frame_qc_n_sigma", self.lc_outlier_frame_qc_n_sigma))
            self.lc_outlier_frame_qc_n_sigma = _lof if math.isfinite(_lof) and _lof > 0 else 5.0
        except (TypeError, ValueError):
            self.lc_outlier_frame_qc_n_sigma = 5.0
        try:
            _loe = float(data.get("lc_outlier_evidence_n_sigma", self.lc_outlier_evidence_n_sigma))
            self.lc_outlier_evidence_n_sigma = _loe if math.isfinite(_loe) and _loe > 0 else 5.0
        except (TypeError, ValueError):
            self.lc_outlier_evidence_n_sigma = 5.0
        try:
            _lhe = float(data.get("lc_high_err_nsigma", self.lc_high_err_nsigma))
            self.lc_high_err_nsigma = _lhe if math.isfinite(_lhe) and _lhe > 0 else 5.0
        except (TypeError, ValueError):
            self.lc_high_err_nsigma = 5.0
        self.trust_flag_enabled = bool(data.get("trust_flag_enabled", self.trust_flag_enabled))
        # Export reports - prefer ``observer_name`` / ``observer_code``; fall back to legacy JSON key for the code.
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
        # SETTINGS-2026-10: the site is not stored in config.json; the default site is the
        # LOCATION row with IS_DEFAULT = 1 (Database Explorer), a run may choose another one.
        from database import get_default_observer_location_id  # noqa: PLC0415

        self.observer_location_id = get_default_observer_location_id(str(self.database_path))
        # WAVE-B STEP 5 (DELETE-DB-DUP): observer_lat/lon/alt_m/name are DB-authoritative
        # (draft LOCATION via observer_location_id). They are no longer loaded from or saved to
        # config.json; the fields remain as run-time hydrated mirrors (block below) with the
        # dataclass site defaults as fallback. Science reads the draft LOCATION row, not cfg.
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
                    "using dataclass site-default observer coordinates: %s",
                    self.observer_location_id,
                    exc,
                )
        self.epsf_auto_run = bool(data.get("epsf_auto_run", self.epsf_auto_run))
        self.epsf_targets = str(data.get("epsf_targets", self.epsf_targets) or "ALL").strip() or "ALL"
        self.epsf_export_auto = bool(data.get("epsf_export_auto", self.epsf_export_auto))
        try:
            _pln = float(data.get("psf_chi2_locus_nsigma", self.psf_chi2_locus_nsigma))
            self.psf_chi2_locus_nsigma = _pln if math.isfinite(_pln) and _pln > 0 else 5.0
        except (TypeError, ValueError):
            self.psf_chi2_locus_nsigma = 5.0
        try:
            _sso = int(data.get("preprocess_sky_surface_order", self.preprocess_sky_surface_order))
            self.preprocess_sky_surface_order = max(0, min(2, _sso))
        except (TypeError, ValueError):
            self.preprocess_sky_surface_order = 2
        self.preprocess_sky_surface_force_reapply = bool(
            data.get(
                "preprocess_sky_surface_force_reapply",
                self.preprocess_sky_surface_force_reapply,
            )
        )
        try:
            _ocb = int(data.get("osc_channel_binning", self.osc_channel_binning))
            self.osc_channel_binning = _ocb if _ocb in (1, 2, 3, 4) else 2
        except (TypeError, ValueError):
            self.osc_channel_binning = 2
        try:
            _ems = int(data.get("epsf_min_stars", self.epsf_min_stars))
            self.epsf_min_stars = max(10, _ems)
        except (TypeError, ValueError):
            self.epsf_min_stars = 30
        try:
            self.aperture_fwhm_factor = float(data.get("aperture_fwhm_factor", self.aperture_fwhm_factor))
            if not math.isfinite(self.aperture_fwhm_factor) or self.aperture_fwhm_factor <= 0:
                self.aperture_fwhm_factor = 2.75
        except (TypeError, ValueError):
            self.aperture_fwhm_factor = 2.75
        self.aperture_fwhm_factor = max(0.25, min(6.0, float(self.aperture_fwhm_factor)))
        try:
            _v = float(data.get("per_frame_sat_min_clean_frac", self.per_frame_sat_min_clean_frac))
            self.per_frame_sat_min_clean_frac = (
                _v if math.isfinite(_v) else 0.5
            )
        except (TypeError, ValueError):
            self.per_frame_sat_min_clean_frac = 0.5
        self.per_frame_sat_min_clean_frac = max(
            0.1, min(1.0, float(self.per_frame_sat_min_clean_frac))
        )
        # WAVE-B STEP 5 (DELETE-DB-DUP): gain (FITS EGAIN then DB EQUIPMENTS.GAIN_ADU) and
        # read_noise (DB EQUIPMENTS.READNOISE_E then FITS) are resolver-authoritative via
        # param_resolver at run time; no longer loaded from or saved to config.json. The fields
        # keep their dataclass defaults as last-resort fallbacks.
        try:
            self.err_empty_apertures_n = max(
                16,
                min(256, int(data.get("err_empty_apertures_n", self.err_empty_apertures_n))),
            )
        except (TypeError, ValueError):
            self.err_empty_apertures_n = 64
        try:
            self.err_empty_apertures_min = max(
                1,
                min(256, int(data.get("err_empty_apertures_min", self.err_empty_apertures_min))),
            )
        except (TypeError, ValueError):
            self.err_empty_apertures_min = 16
        try:
            self.masterstar_best_of_n = max(
                1,
                min(25, int(data.get("masterstar_best_of_n", self.masterstar_best_of_n))),
            )
        except (TypeError, ValueError):
            self.masterstar_best_of_n = 10
        try:
            self.phase01_ct_min_comp = max(
                2,
                min(30, int(data.get("phase01_ct_min_comp", self.phase01_ct_min_comp))),
            )
        except (TypeError, ValueError):
            self.phase01_ct_min_comp = 7
        _act = str(data.get("apply_color_term", self.apply_color_term) or "auto").strip().lower()
        if _act in ("1", "true", "yes", "on"):
            self.apply_color_term = "on"
        elif _act in ("0", "false", "no", "off"):
            self.apply_color_term = "off"
        else:
            self.apply_color_term = "auto"
        try:
            _iso = float(data.get("comp_isolation_fwhm", self.comp_isolation_fwhm))
            self.comp_isolation_fwhm = _iso if math.isfinite(_iso) and _iso > 0 else 3.0
        except (TypeError, ValueError):
            self.comp_isolation_fwhm = 3.0
        _k2m = str(data.get("k2_mode", self.k2_mode) or "literature").strip().lower()
        if _k2m in ("0", "false", "no", "off", "none"):
            self.k2_mode = "off"
        else:
            self.k2_mode = "literature"
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
        try:
            self.photon_transfer_ci_max_width_factor = max(
                1.0,
                float(
                    data.get(
                        "photon_transfer_ci_max_width_factor",
                        self.photon_transfer_ci_max_width_factor,
                    )
                ),
            )
        except (TypeError, ValueError):
            self.photon_transfer_ci_max_width_factor = 3.0
        try:
            self.phase01_ct_extrapolation_tol = max(
                0.0,
                float(data.get("phase01_ct_extrapolation_tol", self.phase01_ct_extrapolation_tol)),
            )
        except (TypeError, ValueError):
            self.phase01_ct_extrapolation_tol = 0.0
        try:
            self.comp_max_slope_mmag_hr = max(
                0.0,
                min(500.0, float(data.get("comp_max_slope_mmag_hr", self.comp_max_slope_mmag_hr))),
            )
        except (TypeError, ValueError):
            self.comp_max_slope_mmag_hr = 5.0
        try:
            self.comp_slope_significance_k = max(
                0.0,
                min(10.0, float(data.get("comp_slope_significance_k", self.comp_slope_significance_k))),
            )
        except (TypeError, ValueError):
            self.comp_slope_significance_k = 3.0
        try:
            self.comp_catalog_offset_max_mag = max(
                0.0,
                min(2.0, float(data.get("comp_catalog_offset_max_mag", self.comp_catalog_offset_max_mag))),
            )
        except (TypeError, ValueError):
            self.comp_catalog_offset_max_mag = 0.20
        try:
            self.annulus_inner_fwhm = float(data.get("annulus_inner_fwhm", self.annulus_inner_fwhm))
            self.annulus_outer_fwhm = float(data.get("annulus_outer_fwhm", self.annulus_outer_fwhm))
        except (TypeError, ValueError):
            self.annulus_inner_fwhm = 5.5
            self.annulus_outer_fwhm = 10.5
        if self.annulus_outer_fwhm <= self.annulus_inner_fwhm:
            self.annulus_outer_fwhm = self.annulus_inner_fwhm + 1.0

        try:
            self.dao_centroid_max_shift_fwhm = float(
                data.get("dao_centroid_max_shift_fwhm", self.dao_centroid_max_shift_fwhm)
            )
        except (TypeError, ValueError):
            self.dao_centroid_max_shift_fwhm = 1.0
        self.dao_centroid_max_shift_fwhm = max(0.1, min(5.0, float(self.dao_centroid_max_shift_fwhm)))

        def _f01(key: str, default: float, lo: float, hi: float) -> None:
            try:
                v = float(data.get(key, getattr(self, key)))
                if not math.isfinite(v):
                    raise ValueError
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, float(default))

        def _i01(key: str, default: int, lo: int, hi: int) -> None:
            try:
                v = int(data.get(key, getattr(self, key)))
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, int(default))

        _f01("comp_max_delta_bprp", 0.79, 0.0, 5.0)
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
        _f01("comp_contamination_penalty_k", 3.0, 0.0, 20.0)
        _f01("calibration_master_ccd_temp_tolerance_c", 0.5, 0.01, 20.0)
        self.gs11_dilution_enabled = bool(
            data.get("gs11_dilution_enabled", self.gs11_dilution_enabled)
        )
        _f01("gs11_dilution_aperture_arcsec", 0.0, 0.0, 120.0)
        _f01("gs11_dilution_mag_limit_delta", 5.0, 0.5, 15.0)
        _f01("gs11_target_min_dilution", 0.50, 0.01, 1.0)
        _i01("phase01_comparison_n_comp_min", 3, 2, 12)
        _i01("phase01_comparison_n_comp_max", 8, 3, 20)
        _f01("snr_min", 10.0, 3.0, 1000.0)
        if int(self.phase01_comparison_n_comp_max) < int(self.phase01_comparison_n_comp_min):
            self.phase01_comparison_n_comp_max = int(self.phase01_comparison_n_comp_min)
        _i01("comp_trust_min_comps", 5, 3, 20)
        if int(self.comp_trust_min_comps) > int(self.phase01_comparison_n_comp_max):
            self.comp_trust_min_comps = int(self.phase01_comparison_n_comp_max)
        _i01("lc_quality_min_frames", 20, 3, 500)
        _i01("lc_quality_short_min_frames", 3, 2, 100)
        _f01("lc_quality_min_normal_frac", 0.5, 0.1, 1.0)
        if int(self.lc_quality_short_min_frames) > int(self.lc_quality_min_frames):
            self.lc_quality_short_min_frames = int(self.lc_quality_min_frames)
        _i01("check_star_min_epochs", 5, 3, 50)
        _f01("sparse_trust_T_green", 1.5, 0.5, 10.0)
        _f01("sparse_trust_T_red", 4.0, 1.0, 20.0)
        _f01("sparse_trust_X2_RED", 0.0004, 0.0, 0.01)
        _f01("check_select_rms_floor", 1e-4, 0.0, 0.01)
        _f01("comp_select_rms_floor", 1e-6, 0.0, 0.01)
        _f01("phase01_comparison_max_comp_rms", 0.05, 0.01, 0.5)
        _f01("comp_rms_loo_photon_k", 5.0, 1.0, 50.0)
        _f01("phase01_comparison_min_dist_arcsec", 60.0, 0.0, 600.0)
        _f01("phase01_comparison_min_frames_frac", 0.3, 0.05, 0.95)
        # Plate-scale ceiling matches the runtime resolver clamp [0.1, 30.0] so that a
        # wide-field config value (e.g. 9.77"/px) survives load instead of being capped
        # at 5.0. phase01_* keeps lo=0.0 because 0.0 is the "auto / unset" sentinel
        # (consumed as ``phase01_plate_scale_arcsec_per_px or 1.3``); clamping it to 0.1
        # would turn "auto" into a real 0.1"/px scale.
        # WAVE-B STEP 5 (DELETE-DB-DUP): plate_scale_arcsec_per_px (WCS/CD then DB optics) and
        # phase01_plate_scale_arcsec_per_px (WCS-superseded) resolve at run time; no longer loaded
        # from or saved to config.json. Fields keep their dataclass defaults as fallbacks.
        _f01("exoplanet_match_max_sep_arcsec", 3.0, 0.5, 30.0)
        _f01("phase01_comparison_isolation_radius_px", 25.0, 1.0, 200.0)
        # WAVE-B STEP 3: frame_width_px / frame_height_px are INTERNAL. Frame geometry
        # resolves from FITS NAXIS1/2 at run time; the dataclass default is the only
        # fallback. They are no longer loaded from or saved to config.json. Warn once if a
        # legacy config.json still carries them.
        if "frame_width_px" in data or "frame_height_px" in data:
            logging.warning(
                "[DEPRECATED] frame_width_px/frame_height_px in config.json are ignored; "
                "frame dimensions resolve from FITS NAXIS at run time (WAVE-B STEP 3)."
            )

        _chip_m = data.get("phase01_chip_interior_margin_px")
        if _chip_m is None and "phase01_suspected_interior_margin_px" in data:
            _chip_m = data.get("phase01_suspected_interior_margin_px")
        if _chip_m is not None and _chip_m != "":
            try:
                self.phase01_chip_interior_margin_px = max(0, min(2000, int(_chip_m)))
            except (TypeError, ValueError):
                self.phase01_chip_interior_margin_px = 100


        self.comp_pool_derived_admission = bool(
            data.get("comp_pool_derived_admission", self.comp_pool_derived_admission)
        )
        try:
            self.comparison_stars_pool_n = int(
                data.get("comparison_stars_pool_n", self.comparison_stars_pool_n)
            )
        except (TypeError, ValueError):
            self.comparison_stars_pool_n = 0
        self.comparison_stars_pool_n = max(0, min(50000, int(self.comparison_stars_pool_n)))
        self.forced_photometry_enabled = bool(
            data.get("forced_photometry_enabled", self.forced_photometry_enabled)
        )
        try:
            self.forced_photometry_margin_px = float(
                data.get("forced_photometry_margin_px", self.forced_photometry_margin_px)
            )
        except (TypeError, ValueError):
            self.forced_photometry_margin_px = 0.0
        if not math.isfinite(float(self.forced_photometry_margin_px)) or float(self.forced_photometry_margin_px) < 0:
            self.forced_photometry_margin_px = 0.0

        self.tess_enabled = bool(data.get("tess_enabled", self.tess_enabled))
        _f01("tess_contam_max", 0.10, 0.0, 1.0)


    # --- structured-key accessors (WAVE-B STEP 4) ---------------------------------- #
    def comp_tier_bprp_limits(self) -> list[float]:
        """|dBP-RP| colour limit per comp tier, row order, from ``comp_color_tiers``."""
        return [float(t.get("bprp", 0.0)) for t in self.comp_color_tiers]



    def to_json(self) -> dict[str, Any]:
        return {
            "archive_root": str(self.archive_root),
            "calibration_library_root": str(self.calibration_library_root),
            "database_path": str(self.database_path),
            "masterdark_validity_days": int(self.masterdark_validity_days),
            "masterflat_validity_days": int(self.masterflat_validity_days),
            "calibration_master_ccd_temp_tolerance_c": float(
                self.calibration_master_ccd_temp_tolerance_c
            ),
            "gaia_db_path": str(self.gaia_db_path or ""),
            "astrometry_index_dir": str(self.astrometry_index_dir or ""),
            "hrd_online_enrich_enabled": bool(self.hrd_online_enrich_enabled),
            "hrd_simbad_enrich_enabled": bool(self.hrd_simbad_enrich_enabled),
            "hrd_enrich_max_candidates": int(self.hrd_enrich_max_candidates),
            "hrd_enrich_tap_timeout_s": float(self.hrd_enrich_tap_timeout_s),
            "hrd_parallax_min_mas": float(self.hrd_parallax_min_mas),
            "hrd_parallax_snr_min": float(self.hrd_parallax_snr_min),
            "hrd_max_per_category": int(self.hrd_max_per_category),
            "hrd_min_per_net": int(self.hrd_min_per_net),
            "hrd_nss_category_enabled": bool(self.hrd_nss_category_enabled),
            "hrd_dsc_confirm_prob": float(self.hrd_dsc_confirm_prob),
            "objects_db_path": str(self.objects_db_path or ""),
            "vsx_out_of_scope_types": list(self.vsx_out_of_scope_types),
            "exoplanet_match_max_sep_arcsec": float(self.exoplanet_match_max_sep_arcsec),
            "per_frame_mp_reserve_ram_gb": float(self.per_frame_mp_reserve_ram_gb),
            "alignment_max_stars": int(self.alignment_max_stars),
            "alignment_max_control_points": int(self.alignment_max_control_points),
            "detection_threshold_sigma": float(self.detection_threshold_sigma),
            "sips_dao_fwhm_px": float(self.sips_dao_fwhm_px),
            "qc_after_calibrate_enabled": bool(self.qc_after_calibrate_enabled),
            "auto_fwhm_enabled": bool(self.auto_fwhm_enabled),
            "auto_fwhm_k_factor": float(self.auto_fwhm_k_factor),
            "save_lightcurve_png": bool(self.save_lightcurve_png),
            "lc_outlier_enabled": bool(self.lc_outlier_enabled),
            "lc_outlier_n_sigma": float(self.lc_outlier_n_sigma),
            "lc_outlier_adjacent_sigma": float(self.lc_outlier_adjacent_sigma),
            "lc_outlier_frame_qc_n_sigma": float(self.lc_outlier_frame_qc_n_sigma),
            "lc_outlier_evidence_n_sigma": float(self.lc_outlier_evidence_n_sigma),
            "lc_high_err_nsigma": float(self.lc_high_err_nsigma),
            "trust_flag_enabled": bool(self.trust_flag_enabled),
            "lc_quality_min_frames": int(self.lc_quality_min_frames),
            "lc_quality_short_min_frames": int(self.lc_quality_short_min_frames),
            "lc_quality_min_normal_frac": float(self.lc_quality_min_normal_frac),
            "comp_trust_min_comps": int(self.comp_trust_min_comps),
            "check_star_min_epochs": int(self.check_star_min_epochs),
            "sparse_trust_T_green": float(self.sparse_trust_T_green),
            "sparse_trust_T_red": float(self.sparse_trust_T_red),
            "sparse_trust_X2_RED": float(self.sparse_trust_X2_RED),
            "check_select_rms_floor": float(self.check_select_rms_floor),
            "comp_select_rms_floor": float(self.comp_select_rms_floor),
            "observer_name": str(self.observer_name),
            "observer_code": str(self.observer_code),
            "aavso_filter_map": dict(self.aavso_filter_map),
            "epsf_auto_run": bool(self.epsf_auto_run),
            "epsf_targets": str(self.epsf_targets),
            "epsf_export_auto": bool(self.epsf_export_auto),
            "psf_chi2_locus_nsigma": float(self.psf_chi2_locus_nsigma),
            "preprocess_sky_surface_order": int(self.preprocess_sky_surface_order),
            "preprocess_sky_surface_force_reapply": bool(self.preprocess_sky_surface_force_reapply),
            "osc_channel_binning": int(self.osc_channel_binning),
            "epsf_min_stars": int(self.epsf_min_stars),
            "aperture_fwhm_factor": float(self.aperture_fwhm_factor),
            "per_frame_sat_min_clean_frac": float(self.per_frame_sat_min_clean_frac),
            "err_empty_apertures_n": int(self.err_empty_apertures_n),
            "err_empty_apertures_min": int(self.err_empty_apertures_min),
            "masterstar_best_of_n": int(self.masterstar_best_of_n),
            "phase01_comparison_isolation_radius_px": float(self.phase01_comparison_isolation_radius_px),
            "annulus_inner_fwhm": float(self.annulus_inner_fwhm),
            "annulus_outer_fwhm": float(self.annulus_outer_fwhm),
            "dao_centroid_max_shift_fwhm": float(self.dao_centroid_max_shift_fwhm),
            "comp_max_delta_bprp": float(self.comp_max_delta_bprp),
            "comp_color_tiers": [
                {"bprp": float(t.get("bprp", 0.0))}
                for t in self.comp_color_tiers
            ],
            "comp_contamination_penalty_k": float(self.comp_contamination_penalty_k),
            "phase01_comparison_n_comp_min": int(self.phase01_comparison_n_comp_min),
            "snr_min": float(self.snr_min),
            "phase01_comparison_n_comp_max": int(self.phase01_comparison_n_comp_max),
            "phase01_comparison_max_comp_rms": float(self.phase01_comparison_max_comp_rms),
            "comp_rms_loo_photon_k": float(self.comp_rms_loo_photon_k),
            "phase01_comparison_min_dist_arcsec": float(self.phase01_comparison_min_dist_arcsec),
            "phase01_comparison_min_frames_frac": float(self.phase01_comparison_min_frames_frac),
            "phase01_ct_min_comp": int(self.phase01_ct_min_comp),
            "apply_color_term": str(self.apply_color_term),
            "comp_isolation_fwhm": float(self.comp_isolation_fwhm),
            "k2_mode": str(self.k2_mode),
            "k2_defaults_bprp": dict(self.k2_defaults_bprp),
            "photon_transfer_ci_max_width_factor": float(self.photon_transfer_ci_max_width_factor),
            "phase01_ct_extrapolation_tol": float(self.phase01_ct_extrapolation_tol),
            "comp_max_slope_mmag_hr": float(self.comp_max_slope_mmag_hr),
            "comp_slope_significance_k": float(self.comp_slope_significance_k),
            "comp_catalog_offset_max_mag": float(self.comp_catalog_offset_max_mag),
            "gs11_dilution_enabled": bool(self.gs11_dilution_enabled),
            "gs11_dilution_aperture_arcsec": float(self.gs11_dilution_aperture_arcsec),
            "gs11_dilution_mag_limit_delta": float(self.gs11_dilution_mag_limit_delta),
            "gs11_target_min_dilution": float(self.gs11_target_min_dilution),
            "phase01_chip_interior_margin_px": int(self.phase01_chip_interior_margin_px),
            "tess_enabled": bool(self.tess_enabled),
            "tess_contam_max": float(self.tess_contam_max),
            "comp_pool_derived_admission": bool(self.comp_pool_derived_admission),
            "comparison_stars_pool_n": int(self.comparison_stars_pool_n),
            "forced_photometry_enabled": bool(self.forced_photometry_enabled),
            "forced_photometry_margin_px": float(self.forced_photometry_margin_px),
        }

    # Backward-compatible alias (some callers expect to_dict()).
    def to_dict(self) -> dict[str, Any]:
        return self.to_json()

    def ensure_base_dirs(self) -> None:
        """Create base directories required by file-first workflow."""
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.calibration_library_root.mkdir(parents=True, exist_ok=True)


def compute_field_density(n_masterstar_stars: int, chip_w_px: int, chip_h_px: int) -> float:
    """Vrati hustotu pola v hviezd/Mpx."""
    mpx = (float(chip_w_px) * float(chip_h_px)) / 1_000_000.0
    if mpx <= 0 or n_masterstar_stars < 0:
        return 0.0
    return float(n_masterstar_stars) / float(mpx)


def classify_field_density(density: float, sparse_th: float, dense_th: float) -> str:
    """Vrati ``sparse`` / ``normal`` / ``dense``."""
    if density < float(sparse_th):
        return "sparse"
    if density <= float(dense_th):
        return "normal"
    return "dense"
