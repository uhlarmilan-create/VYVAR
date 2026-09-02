# -*- coding: ascii -*-
"""Machine-checkable claims made by the FLOW doc (VYVAR_FLOW_CZ.pdf).

Every value here is asserted in the prose of VYVAR_FLOW_CZ.pdf. If a test
against this file goes red, the pipeline changed under the documentation:
update BOTH the builder prose AND this file, then regenerate the PDF.
"""
from __future__ import annotations

# Config keys with the exact defaults quoted in the FLOW doc (from config.json).
DOC_CONFIG_FACTS: dict[str, object] = {
    "comp_max_delta_bprp": 0.79,
    "phase01_comparison_n_comp_min": 3,
    "phase01_comparison_n_comp_max": 8,
    "comp_trust_min_comps": 3,
    "aperture_fwhm_factor": 1.35,
    "annulus_inner_fwhm": 2.7,
    "annulus_outer_fwhm": 5.2,
    "err_empty_apertures_n": 64,
    "sigma_sys_mag": {"4": 0.018},
    "k2_mode": "literature",
    "k2_ceiling": 0.1,
    "k2_fit_enabled": False,
    "temporal_binning_enabled": False,
    "pytics_enabled": True,
    "pytics_n_iter": 5,
    "psf_photometry_enabled": False,
    "per_frame_saturation_enabled": False,
    "epsf_min_stars": 30,
    "psf_chi2_threshold": 50.0,
    "calibration_master_ccd_temp_tolerance_c": 0.5,
    "masterdark_validity_days": 90,
    "masterflat_validity_days": 200,
    "field_density_sparse_threshold": 300.0,
    "field_density_dense_threshold": 1000.0,
    "verify_mag_limit": 14.0,
    "apply_color_term": "auto",
    "phase01_comparison_max_comp_rms": 0.1,
    "comp_max_slope_mmag_hr": 5.0,
    "sparse_trust_T_green": 1.5,
    "sparse_trust_T_red": 4.0,
    "sparse_trust_X2_RED": 0.0004,
    "variability_sigma_threshold": 2.3,
    "variability_vdi_z_threshold": 3.0,
    "preprocess_sky_surface_order": 2,
    "auto_fwhm_k_factor": 1.5,
    "masterstar_dao_threshold_sigma": 4.5,
    "exoplanet_match_max_sep_arcsec": 3.0,
}

# (relative file, symbol) pairs the FLOW doc names. Text scan only in tests.
DOC_FUNCTIONS: list[tuple[str, str]] = [
    ("src_py/photometry_core.py", "run_full_photometry_pipeline"),
    ("src_py/photometry_gate_helpers.py", "measure_empty_aperture_sigma_bkg"),
    ("src_py/photometry_core.py", "compute_aperture_correction"),
    ("src_py/photometry_core.py", "ensemble_normalize"),
    ("src_py/photometry_core.py", "compute_mag_calib_final"),
    ("src_py/photometry_comp.py", "build_global_comp_pool"),
    ("src_py/k2_extinction.py", "apply_k2_to_comp_mag_inst"),
    ("src_py/band_classify.py", "classify_photometric_band"),
    ("src_py/sigma_floor_core.py", "combine_production_err_rel"),
    ("src_py/psf_photometry.py", "build_epsf_model"),
    ("src_py/psf_photometry.py", "build_epsf_grid_model"),
    ("src_py/psf_photometry.py", "_psf_sandwich_flux_err"),
    ("src_py/crowding_index.py", "compute_crowding_index"),
    ("src_py/tess_verify.py", "_period_consensus"),
    ("src_py/check_star_kmag.py", "select_check_star"),
    ("src_py/trust_flag_core.py", "CompTrustThresholds"),
    ("src_py/calibration.py", "resample_master_to_light_binning"),
]

ANCHOR_ID = "draft_435"
