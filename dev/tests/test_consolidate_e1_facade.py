"""CONSOLIDATE-01E1: moved defs remain reachable through the facade."""

from __future__ import annotations

import photometry
import photometry_core
import pipeline


PIPELINE_E1 = (
    "extract_fits_metadata",
    "fits_metadata_from_primary_header",
    "scan_usb_folder",
    "generate_observation_hash",
    "observation_group_key_from_metadata",
    "log_lights_binning_from_headers_preflight",
    "run_quality_analysis",
    "list_best_processed_light_paths_for_masterstar",
    "_resolve_light_fits_for_quality_inspection",
    "resolve_masterstars_metadata_csv",
    "preprocess_sky_summary_from_df",
    "validate_comparison_ensemble_flatness",
)

PHOTOMETRY_E1: tuple[str, ...] = (
    "resolve_lc_time_base",
    "lc_time_axis_short_label",
    "_annulus_sky_subtracted_flux",
    "_sky_pp_from_annulus_image",
    "measure_empty_aperture_sigma_bkg",
    "measure_growth_curve_ee",
    "photometer_check_star_production_path",
    "_aperture_flux_sky_per_star",
    "_compute_fov_max_dist",
    "_phase2a_star_mag_lookup",
    "_assert_inv_err_sigma_acct_01",
    "_median_bkg_var_from_aligned_frames",
    "estimate_star_free_per_pixel_variance_adu2",
    "discover_aligned_science_fits",
    "_recompute_bjd_hjd_per_target",
    "_estimate_annulus_sky_pp",
    "_labbe_content_seed_from_header",
    "bkg_scale_ratio_empirical_over_howell",
    "_howell_bkg_variance_adu2",
    "scaled_sigma_bkg_ap_from_howell",
    "comp_quality_quality_strings",
    "_frame_quality_gate_select",
    "_resolve_star_flux_method",
    "_sigma_bkg_r_key",
    "_clamp_err_empty_apertures_n",
    "compute_setup_bkg_scale_r",
    "_normalize_err_background_mode",
    "_clamp_bkg_scale_r",
)


def test_e1_pipeline_facade_getattr() -> None:
    for name in PIPELINE_E1:
        obj = getattr(pipeline, name)
        assert callable(obj), name


def test_e1_extract_fits_metadata_patch_string_path() -> None:
    """risk_register string/getattr: tests patch pipeline.extract_fits_metadata."""
    assert pipeline.extract_fits_metadata is not None
    assert pipeline.extract_fits_metadata.__module__ == "fits_meta"


def test_e1_photometry_core_facade_getattr() -> None:
    for name in PHOTOMETRY_E1:
        obj = getattr(photometry_core, name)
        assert callable(obj), name


def test_e1_photometry_star_import_still_binds_all() -> None:
    for name in photometry_core.__all__:
        assert hasattr(photometry, name), name


def test_e1_annulus_patch_string_path() -> None:
    """risk_register patch-string: photometry_core._annulus_sky_subtracted_flux."""
    fn = photometry_core._annulus_sky_subtracted_flux
    assert callable(fn)
    assert fn.__module__ == "photometry_gate_helpers"
