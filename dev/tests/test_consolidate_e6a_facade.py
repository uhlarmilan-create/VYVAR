"""CONSOLIDATE-01E6a: moved defs remain reachable through the pipeline facade."""

from __future__ import annotations

import inspect

import pipeline
import pipeline_astrometry
import pipeline_catalog
import pipeline_preprocess


PIPELINE_E6A_PREPROCESS: tuple[str, ...] = (
    "_archive_raw_to_calibrated_light",
    "_load_raw_for_frame",
    "_load_raw_hdr_for_frame",
    "_partition_detrended_by_subfolder",
    "_qc_suggest_thresholds",
    "build_prefilter_rejected_map",
    "calibrated_paths_for_draft_apply_filters",
    "filter_files_by_qc_metrics_allowlist",
    "load_qc_metrics_status_by_path",
    "preprocess_calibrated_to_processed",
    "qc_enrich_calibrated_lights_in_place",
    "resolve_obs_file_to_processed_fits",
    "resolve_preprocess_target_coordinates",
)

PIPELINE_E6A_ASTROMETRY: tuple[str, ...] = (
    "_EPSF_SKIP_LOGGED",
    "_VYVAR_TIME_JD_CSV_COLS",
    "_apply_wcs_header_to_fits",
    "_assert_alignment_produced_fits",
    "_catalog_match_radius_px",
    "_dao_targeted_pass2_unmatched_gaia",
    "_ensure_parent_dirs_for_aligned_fits",
    "_equipment_saturate_adu_from_db",
    "_export_catalog_psf_st_fields",
    "_field_jump_empty_result",
    "_fill_masterstars_gaia_matched_bp_rp_from_local_db",
    "_finite_positive_adu",
    "_header_focal_length_mm",
    "_header_vy_fwhm_px",
    "_merge_astrometry_group_reports",
    "_merge_dao_pass1_pass2_tables",
    "_merge_platesolve_gaia_pairs_into_masterstars_df",
    "_merge_vsx_exoplanet_variable_targets",
    "_pass2_sibling_wcs_recovery",
    "_path_is_under_tree",
    "_path_segments_forbidden_for_masterstar_physical_source",
    "_photometry_mode_run_flags",
    "_pick_preferred_masterstar_basename_hit",
    "_pipeline_ui_info",
    "_plate_solve_input_bundle",
    "_query_vsx_local_frame_bbox",
    "_resolve_best_effort_path_under",
    "_resolve_focal_mm_for_plate_scale",
    "_run_osc_multi_group_alignment",
    "_safe_proc_name",
    "_sat_adu_from_draft_sat_diag",
    "_solve_wcs_external",
    "_sort_masterstar_paths_by_fwhm",
    "_strip_external_platesolve_header",
    "_sync_comparison_stars_across_setups",
    "_try_rescale_masterstar_linear_wcs_to_expected_plate_scale",
    "_update_masterstar_obs_file_status",
    "_vyvar_df_round_time_jd_for_csv",
    "_vyvar_df_to_csv",
    "_vyvar_open_database",
    "_vyvar_per_frame_csv_workers",
    "_wcs_field_center_radec_deg",
    "astrometry_align_and_build_masterstar",
    "build_masterstar_from_detrended",
    "compute_plate_scale_from_db",
    "detect_field_jumps",
    "draft_is_multi_group_obs",
    "draft_obs_group_count",
    "get_masterstar_candidate_rows",
    "get_masterstar_candidates",
    "resolve_masterstar_input_root",
    "resolve_plate_solve_fov_deg_hint",
    "select_comparison_stars_spatial_grid",
    "write_photometry_plan_files",
)

PIPELINE_E6A_CATALOG: tuple[str, ...] = (
    "_BATCH_E_N_EQUIV_LOGGED",
    "_EXPORT_PER_FRAME_WORKER_STATE",
    "_MASTERSTAR_ZONE_LOG_ONCE",
    "_MOFFAT_CHI2_LIMIT",
    "_airmass_from_altitude_deg",
    "_all_pix2world_icrs_deg",
    "_annotate_masterstars_flux_zones",
    "_apply_aperture_catalog_enhancements_from_st",
    "_apply_dao_centroid_wcs_guard",
    "_apply_exo_host_columns_to_proc_df",
    "_apply_wcs_tan_fragment_to_header",
    "_bin2d_mean",
    "_box_peak_max_adu",
    "_box_peaks_at_centroids",
    "_build_exoplanet_promotion_rows_from_masterstars",
    "_catalog_df_cap_brightest_by_mag",
    "_cfg_from_export_worker_state",
    "_chord_to_arcsec",
    "_compute_airmass_from_altaz",
    "_dao_auto_binning_factor",
    "_dao_convolved_background_rms_adu",
    "_dao_detection_threshold_adu",
    "_dao_noise_sigma_adu",
    "_dao_spatial_flux_cap_row_indices",
    "_dao_star_count_from_array",
    "_detect_empirical_clip_level_adu",
    "_effective_field_catalog_cone_radius_deg",
    "_epsf_fit_catalog_ids",
    "_epsf_target_catalog_ids",
    "_estimate_catalog_frame_hw",
    "_exo_host_annotation_arrays",
    "_export_first_icrs_center_radius",
    "_export_per_frame_disk_worker_task",
    "_export_per_frame_ram_worker_task",
    "_export_per_frame_run_catalog_core",
    "_extract_airmass_from_header",
    "_field_catalog_cone_meta_path",
    "_field_center_and_radius_from_wcs",
    "_fill_psf_catalog_columns",
    "_finalize_hybrid_bkg_fallback_sidecar",
    "_fits_header_first_positive_float",
    "_fits_header_vy_algn_aligned",
    "_gaia_catalog_cone_radius_optics_floor_deg",
    "_gaia_chip_xy_from_catalog",
    "_icrs_center_radius_from_hdr_data",
    "_icrs_deg_to_unitxyz",
    "_init_export_per_frame_worker",
    "_invalidate_field_catalog_cone_cache_if_needed",
    "_lock_matched_centroids_to_master_grid",
    "_masterstar_zone_linear_threshold",
    "_masterstar_zone_log_once",
    "_mean_bin2d_for_dao",
    "_pick_reference_frame_by_star_count",
    "_pixel_noise_sigma_pp_adu",
    "_prefetch_export_shared_catalog_for_process_pool",
    "_prefilter_dao_table_brightest",
    "_proc_catalog_keep_matched_rows_only",
    "_proc_deduplicate_matched_catalog_rows",
    "_proc_drop_unmatched_dao_rows",
    "_proc_rename_det_names_to_catalog_id",
    "_proc_sat_block_for_csv",
    "_query_exoplanet_local",
    "_query_gaia_local",
    "_query_vsx_local",
    "_resolve_masterstar_bg_sigma_adu",
    "_resolve_peak_saturation_limit_adu",
    "_sat_ctx_from_worker",
    "_saturated_core_plateau_vectorized",
    "_slice_exo_annotation",
    "_vectorized_star_saturation_columns",
    "_vyvar_cap_mp_workers_for_catalog",
    "_write_field_catalog_cone_meta",
    "build_ucac_catalog_kdtree",
    "detect_stars_match_master_reference",
    "find_qc_metrics_csv",
    "inv_sat_limit_peak_test_adu",
    "nearest_sky_nn_kdtree",
)

_FOLLOW_HOME = {
    "pipeline_astrometry": frozenset({"_plate_solve_input_bundle"}),
}


def _assert_facade(mod, names: tuple[str, ...]) -> None:
    follows = _FOLLOW_HOME.get(mod.__name__, frozenset())
    for name in names:
        obj = getattr(pipeline, name)
        home = getattr(mod, name)
        if name in follows:
            assert obj is not home, name
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            assert obj is home, name
            assert obj.__module__ == mod.__name__, name
            continue
        if isinstance(home, (dict, set, list)):
            assert obj is home, name
            continue
        # Imported bool/int/float/str/tuple are copy-bound. A later
        # `global` rebind in the home module (e.g. _BATCH_E_N_EQUIV_LOGGED)
        # diverges identity; presence on the facade is the contract.
        assert hasattr(pipeline, name), name


def test_e6a_preprocess_facade_getattr() -> None:
    _assert_facade(pipeline_preprocess, PIPELINE_E6A_PREPROCESS)


def test_e6a_astrometry_facade_getattr() -> None:
    _assert_facade(pipeline_astrometry, PIPELINE_E6A_ASTROMETRY)


def test_e6a_catalog_facade_getattr() -> None:
    _assert_facade(pipeline_catalog, PIPELINE_E6A_CATALOG)


def test_e6a_giants_stay_in_pipeline() -> None:
    assert pipeline.generate_masterstar_and_catalog.__module__ == "pipeline"
    assert pipeline.detect_stars_and_match_catalog.__module__ == "pipeline"
    assert pipeline.export_per_frame_catalogs.__module__ == "pipeline"
    assert pipeline._astrometry_align_impl_body.__module__ == "pipeline"
    assert pipeline.AstroPipeline.__module__ == "pipeline"


def test_e6a_export_init_arity_one() -> None:
    sig = inspect.signature(pipeline._init_export_per_frame_worker)
    assert list(sig.parameters) == ["state"]


def test_e6a_sat_limit_peak_source_stays() -> None:
    assert pipeline.SAT_LIMIT_PEAK_TEST_SOURCE.__class__ is str
    assert "INV-SAT-LIMIT" in pipeline.SAT_LIMIT_PEAK_TEST_SOURCE
    # Defined in pipeline.py next to SAT_LIMIT_CONTAINER_CLIP_ADU.
    # catalog must not re-export a second object (identity).
    cat_src = getattr(pipeline_catalog, "SAT_LIMIT_PEAK_TEST_SOURCE", None)
    assert cat_src is None or cat_src is pipeline.SAT_LIMIT_PEAK_TEST_SOURCE
