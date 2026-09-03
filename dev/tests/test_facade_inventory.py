"""Permanent facade inventory (CONSOLIDATE-01 E-FINAL).

Union of the per-wave e1..edead getattr-loop name lists. Per-wave history
lives in git. Scalar-rebind rule from bcead65: imported bool/int/float/str/tuple
are copy-bound; a later home-module global rebind diverges identity; presence
on the facade is the contract.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import photometry
import photometry_comp
import photometry_core
import photometry_exports
import photometry_gate_helpers
import photometry_lightcurve
import photometry_phase2a
import photometry_provenance
import photometry_shared
import pipeline
import pipeline_astrometry
import pipeline_calibrate
import pipeline_catalog
import pipeline_constants
import pipeline_preprocess
import pipeline_ui_helpers
import catalog_match
import frame_export
import masterstar_build
import astrometry_align
import phase01_run
import phase2a_state
import phase2a_target


# --- union name lists (from test_consolidate_e1..edead_facade) ---

PIPELINE_CALLABLE_E1 = (
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
    "_epsf_lc_catalog_ids",
    "_add_catalog_ids_from_csv",
)

PHOTOMETRY_CALLABLE_E1 = (
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
    "ensemble_member_ids",
    "apply_comp_w_rel_for_display",
    "_get_lc_psf_strict",
    "lc_has_finite_airmass",
    "_get_lc_adaptive_per_star",
    "load_epsf_metrics_for_draft",
)

PHOTOMETRY_E2_PROVENANCE = (
    "_resolve_git_provenance",
    "_build_pipeline_provenance_block",
    "classify_git_dirty_paths",
    "_porcelain_status_by_path",
    "_is_import_relevant_py_path",
    "_complete_config_snapshot",
    "_json_safe_snapshot_value",
    "merge_photometry_pipeline_meta",
)

PHOTOMETRY_E2_SHARED = (
    "enhance_catalog_dataframe_aperture_bpm",
    "_get_plate_scale_from_cfg",
    "finalize_hybrid_bkg_fallback_proc_dir",
    "stamp_vsx_known_variable_on_masterstars",
    "stress_test_relative_rms_from_sidecars",
    "stamp_masterstar_snr_columns",
    "compute_fwhm_gaussian_for_aperture_catalog",
    "_read_plate_scale_from_fits_path",
    "vsx_is_known_variable_top3_per_bin",
    "build_gs11_summary",
    "_cd_matrix_scale_arcsec_per_px",
    "_resolve_plate_scale_arcsec_per_px",
    "_fwhm_moment_at",
    "common_field_intersection_bbox_px",
    "recommended_aperture_by_color",
    "bad_columns_for_light_frame",
    "_safe_polyfit",
    "_get_lc_adaptive",
    "_target_display_name",
    "common_field_intersection_bbox_px_from_arrays",
    "_normalize_gaia_id",
    "_angular_distance_deg",
    "StressTestResult",
)

PHOTOMETRY_E3_COMP = (
    "select_comparison_stars_per_target",
    "select_active_targets",
    "build_global_comp_pool",
    "_select_comps_by_rms_then_color",
    "_write_suspected_variables",
    "_enrich_comp_bp_rp",
    "_enrich_active_targets_bp_rp",
    "_select_comps_tiered",
    "_enrich_target_bp_rp_from_gaia_db",
    "_batch_enrich_targets_bp_rp_from_gaia_db",
    "_refresh_variable_targets_xy",
    "_read_field_density_inputs",
    "_resolve_frame_hw_px_from_masterstar",
    "ensure_full_variable_targets_if_presel_stub",
    "_auto_repair_catalog_ids",
    "_warn_zero_compstars_edge",
    "_count_gate_passing_comps",
    "_attach_predicted_dilution_report",
    "_phase0_effective_frame_hw_px",
    "_select_comps_by_color_then_rms",
    "_dedupe_comp_pool_by_gaia_key",
    "_bprp_tier_ladder_for_selection",
    "_variable_targets_looks_like_ct_presel_stub",
    "_active_target_zone_flag",
    "_ensure_active_target_display_names",
    "_normalize_id_value",
    "_sid_int",
    "_bool_col",
    "_normalize_id_series",
)

PHOTOMETRY_E4_PHASE2A = (
    "run_phase2a",
    "measure_fwhm_from_masterstar",
    "read_flux_from_csv",
    "compute_aperture_correction",
    "_Phase2AState",
    "_ColorTermGroupFit",
    "parse_comp_quality_json_map",
    "democratic_detrend_lc",
    "detect_outliers",
    "apply_reporting_postprocess",
    "auto_export_variability_candidates_csv",
    "fit_color_term_c1",
    "should_apply_color_term",
    "resolve_apply_color_term",
    "save_field_map_png",
    "_photometric_error",
    "_sky_pp_for_photometric_error",
)

PHOTOMETRY_E4_LC = (
    "ensemble_normalize",
    "compute_mag_calib_final",
    "save_lightcurve_csv",
    "apply_color_term",
    "_coerce_bool_cell",
    "BlendMapEntry",
    "_get_lc",
    "_route_lc_per_frame_err",
    "_recompute_bjd_hjd_with_status",
    "run_sysrem_field",
)

PHOTOMETRY_EDEAD_SHARED = (
    "_finite_pixel_bbox_from_array",
    "_intersection_bbox_from_frame_bboxes",
    "_aperture_flux_sky_batch",
    "compute_per_frame_cog_correction",
)

PHOTOMETRY_EDEAD_GATE = (
    "_clamp_err_empty_apertures_min",
    "_robust_scatter_mad",
    "_build_star_exclusion_mask",
    "_canonicalize_star_xy",
    "_labbe_debug_dump_enabled",
    "_labbe_debug_dump_path",
    "_labbe_append_debug_record",
)

PIPELINE_EDEAD_UI = (
    "_quality_inspection_dao_metrics",
    "_estimate_fov_deg_from_fits_path",
    "_obs_fwhm_basename_map_from_db",
)

PIPELINE_E5 = (
    "SkySurfaceOrderConflictError",
    "_CALIB_MASTER_NB_UNSET",
    "_archive_preprocess_lights_root",
    "_archive_root_from_lights_root",
    "_available_system_ram_bytes",
    "_cal_diag_export_for_workers",
    "_cal_diag_session_from_export",
    "_calibrate_batch_process_one",
    "_calibrate_one_light_apply_masters_in_ram",
    "_calibrate_one_light_disk",
    "_calibration_flags",
    "_calibration_type_from_flags",
    "_cfg_calibration_library_native_binning",
    "_dao_star_table_mean_roundness",
    "_db_for_calibration_tasks",
    "_decide_preprocess_sky_action",
    "_effective_saturation_limit",
    "_estimate_dao_fwhm_guess",
    "_estimate_fov_deg_from_header",
    "_exposure_sec_from_header",
    "_filter_light_paths_maybe",
    "_fit_subtract_preprocess_sky_surface",
    "_fits_primary_pixel_count",
    "_half_flux_radius_in_cutout",
    "_has_usable_master_dark",
    "_has_valid_wcs",
    "_hdr_vy_cflag_str",
    "_header_vyskyord",
    "_infer_raw_light_path_for_calibrated",
    "_infer_sat_limit_from_bitpix",
    "_init_calibrate_batch_worker",
    "_inspection_jd_from_header",
    "_iter_light_fits",
    "_light_binning_from_path",
    "_log_calibration_io_preflight",
    "_log_calibration_metadata_diagnostic",
    "_match_and_crop_pair",
    "_mean_hfr_bright_stars_dao",
    "_moment_fwhm_elong_peak_at",
    "_obs_group_key_from_light_path",
    "_passthrough_lights_to_calibrated",
    "_perf10_lookup_qc",
    "_pick_light_for_metadata_diagnostic",
    "_pipeline_ui_error",
    "_post_calibration_qc_eval",
    "_qc_center_crop_for_stars",
    "_qc_enrich_calibrated_in_place",
    "_qc_enrich_one_frame",
    "_qc_fwhm_elongation",
    "_qc_pack_from_config",
    "_quality_inspection_dao_metrics_array",
    "_resolve_dark_path_for_light",
    "_resolve_draft_light_raw_path",
    "_robust_frame_fwhm_median",
    "_saturate_limit_adu_from_header",
    "_saturation_adu_for_cal_diag",
    "_strip_raw_linearity_header_keywords",
    "_sync_manifest_cal_stage_from_qc_row",
    "_sync_obs_calibration_state_with_retry",
    "_vy_calib_status_numeric",
    "_vyvar_calibrate_multiprocessing_enabled",
    "_vyvar_parallel_pool",
    "_vyvar_parallel_use_processes",
    "_vyvar_parallel_worker_count",
    "_vyvar_qc_preprocess_workers",
    "apply_perf10_dao_qc_to_obs_files",
    "calibrate_lights_to_calibrated",
    "draft_median_pointing_icrs_deg",
    "estimate_archive_memory_profile",
    "estimate_memory_from_fits_headers",
    "format_memory_bytes",
    "norm_fits_path_key",
    "run_draft_ram_calibration_qc_to_obs_files",
    "run_osc_channel_extraction_for_archive",
    "scan_calibrated_lights_pointing",
    "sync_obs_files_drift_arcmin_for_draft",
)

PIPELINE_E6A_PREPROCESS = (
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

PIPELINE_E6A_ASTROMETRY = (
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

PIPELINE_E6A_CATALOG = (
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

PIPELINE_PHYSICAL = frozenset({
    "_analyze_calibrated_qc_one",
    "analyze_calibrated_qc",
    "AstroPipeline",
})
PHOTOMETRY_PHYSICAL = frozenset({
    "compute_auto_fwhm_limit",
    "run_full_photometry_pipeline",
    "__getattr__",
})


def _assert_facade(facade, home, names: tuple[str, ...]) -> None:
    """bcead65: functions/classes/mutables must be identity; scalars presence-only."""
    for name in names:
        obj = getattr(facade, name)
        home_obj = getattr(home, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            assert obj is home_obj, name
            assert obj.__module__ == home.__name__, name
            continue
        if isinstance(home_obj, (dict, set, list)):
            assert obj is home_obj, name
            continue
        assert hasattr(facade, name), name


def test_facade_inventory_pipeline_homes() -> None:
    _assert_facade(pipeline, pipeline_calibrate, PIPELINE_E5)
    _assert_facade(pipeline, pipeline_preprocess, PIPELINE_E6A_PREPROCESS)
    _assert_facade(pipeline, pipeline_astrometry, PIPELINE_E6A_ASTROMETRY)
    _assert_facade(pipeline, pipeline_catalog, PIPELINE_E6A_CATALOG)
    _assert_facade(pipeline, pipeline_ui_helpers, PIPELINE_EDEAD_UI)
    for name in PIPELINE_CALLABLE_E1:
        assert callable(getattr(pipeline, name)), name


def test_facade_inventory_photometry_homes() -> None:
    _assert_facade(photometry_core, photometry_provenance, PHOTOMETRY_E2_PROVENANCE)
    _assert_facade(photometry_core, photometry_shared, PHOTOMETRY_E2_SHARED)
    _assert_facade(photometry_core, photometry_comp, PHOTOMETRY_E3_COMP)
    _assert_facade(photometry_core, photometry_phase2a, PHOTOMETRY_E4_PHASE2A)
    _assert_facade(photometry_core, photometry_lightcurve, PHOTOMETRY_E4_LC)
    _assert_facade(photometry_core, photometry_shared, PHOTOMETRY_EDEAD_SHARED)
    _assert_facade(photometry_core, photometry_gate_helpers, PHOTOMETRY_EDEAD_GATE)
    assert photometry_core._get_lc_star_method is photometry_exports._get_lc_star_method
    assert photometry_core._phase2a_prepare_shared_state is phase2a_state._phase2a_prepare_shared_state
    assert photometry_core._phase2a_process_one_target is phase2a_target._phase2a_process_one_target
    assert photometry_core.run_phase0_and_phase1 is phase01_run.run_phase0_and_phase1
    for name in PHOTOMETRY_CALLABLE_E1:
        assert callable(getattr(photometry_core, name)), name


def test_facade_inventory_giants_and_stays() -> None:
    assert pipeline.generate_masterstar_and_catalog is masterstar_build.generate_masterstar_and_catalog
    assert pipeline.detect_stars_and_match_catalog is catalog_match.detect_stars_and_match_catalog
    assert pipeline.export_per_frame_catalogs is frame_export.export_per_frame_catalogs
    assert pipeline._astrometry_align_impl_body is astrometry_align._astrometry_align_impl_body
    assert pipeline.AstroPipeline.__module__ == "pipeline"
    assert pipeline.analyze_calibrated_qc.__module__ == "pipeline"
    assert pipeline._analyze_calibrated_qc_one.__module__ == "pipeline"
    assert photometry_core.run_full_photometry_pipeline.__module__ == "photometry_core"
    assert photometry_core.compute_auto_fwhm_limit.__module__ == "photometry_core"
    assert photometry_core.select_active_targets is photometry_comp.select_active_targets


def test_facade_inventory_star_import() -> None:
    for name in photometry_core.__all__:
        assert hasattr(photometry, name), name
    assert photometry.run_phase0_and_phase1 is photometry_core.run_phase0_and_phase1
    assert photometry.select_active_targets is photometry_core.select_active_targets


def test_facade_inventory_constants_leaf() -> None:
    assert pipeline.SAT_LIMIT_PEAK_TEST_SOURCE is pipeline_constants.SAT_LIMIT_PEAK_TEST_SOURCE
    cat_src = getattr(pipeline_catalog, "SAT_LIMIT_PEAK_TEST_SOURCE", None)
    assert cat_src is None or cat_src is pipeline.SAT_LIMIT_PEAK_TEST_SOURCE


def test_facade_inventory_init_arities() -> None:
    sig = inspect.signature(pipeline._init_calibrate_batch_worker)
    assert list(sig.parameters) == ["_md_s", "native_b", "cal_diag_blob"]
    sig2 = inspect.signature(pipeline._init_export_per_frame_worker)
    assert list(sig2.parameters) == ["state"]


def test_facade_physical_def_ast_guard() -> None:
    """Facades have no lambdas, no injects, and only the sanctioned physical defs."""
    root = Path(pipeline.__file__).resolve().parent
    _assert_facade_ast(
        root / "pipeline.py",
        allowed_defs=PIPELINE_PHYSICAL,
        allow_getattr=False,
    )
    _assert_facade_ast(
        root / "photometry_core.py",
        allowed_defs=PHOTOMETRY_PHYSICAL,
        allow_getattr=True,
    )


def _assert_facade_ast(
    path: Path,
    *,
    allowed_defs: frozenset[str],
    allow_getattr: bool,
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "__getattr__" and allow_getattr:
                continue
            assert node.name in allowed_defs, f"{path.name} physical def {node.name!r} not sanctioned"
        if isinstance(node, ast.Assign):
            for val in (node.value,):
                if isinstance(val, ast.Lambda):
                    raise AssertionError(f"{path.name} has a module-level lambda")
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute):
                    raise AssertionError(f"{path.name} inject assign {ast.unparse(tgt)}")
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and fn.id == "setattr":
                raise AssertionError(f"{path.name} setattr inject")
            if isinstance(fn, ast.Attribute) and fn.attr == "setattr":
                raise AssertionError(f"{path.name} setattr inject")
