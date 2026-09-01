# CONSOLIDATE-01E0 proposed module map

Measure only. Facades `pipeline.py` and `photometry_core.py` stay and re-export.
Facade removal is E-final, a separate decision. Stage boundaries win over graph aesthetics.
Line cap 4000. Base 5b1068d.

## Table

| module | stage | n_defs | lines | over_cap | note |
| --- | --- | --- | --- | --- | --- |
| `pipeline_import.py` | import | 20 | 661 |  | from pipeline.py |
| `photometry_calibrate.py` | calibration | 1 | 39 |  | from photometry_core.py |
| `pipeline_calibrate.py` | calibration | 76 | 4121 | YES | from pipeline.py |
| `pipeline_astrometry__generate_masterstar_and_catalog.py` | astrometry-MASTERSTAR | 1 | 2540 |  | from pipeline.py; single def 2540 lines (>=800). Function body split is a later E-task if still over cap. |
| `pipeline_astrometry__detect_stars_and_match_catalog.py` | astrometry-MASTERSTAR | 1 | 1372 |  | from pipeline.py; single def 1372 lines (>=800). Function body split is a later E-task if still over cap. |
| `pipeline_astrometry__export_per_frame_catalogs.py` | astrometry-MASTERSTAR | 1 | 1101 |  | from pipeline.py; single def 1101 lines (>=800). Function body split is a later E-task if still over cap. |
| `pipeline_astrometry__astrometry_align_impl_body.py` | astrometry-MASTERSTAR | 1 | 1049 |  | from pipeline.py; single def 1049 lines (>=800). Function body split is a later E-task if still over cap. |
| `pipeline_astrometry.py` | astrometry-MASTERSTAR | 65 | 3995 |  | from pipeline.py; stage over 4000; packed by LPA clusters, product stage kept |
| `pipeline_astrometry_2.py` | astrometry-MASTERSTAR | 73 | 3687 |  | from pipeline.py; stage over 4000; packed by LPA clusters, product stage kept |
| `photometry_comp__run_phase0_and_phase1.py` | phase0+1 comp selection | 1 | 1078 |  | from photometry_core.py; single def 1078 lines (>=800). Function body split is a later E-task if still over cap. |
| `photometry_comp.py` | phase0+1 comp selection | 29 | 3017 |  | from photometry_core.py |
| `photometry_shared.py` | photometry-shared | 32 | 1936 |  | from photometry_core.py |
| `photometry_phase2a__phase2a_process_one_target.py` | phase2a photometry | 1 | 1611 |  | from photometry_core.py; single def 1611 lines (>=800). Function body split is a later E-task if still over cap. |
| `photometry_phase2a__phase2a_prepare_shared_state.py` | phase2a photometry | 1 | 940 |  | from photometry_core.py; single def 940 lines (>=800). Function body split is a later E-task if still over cap. |
| `photometry_phase2a.py` | phase2a photometry | 63 | 3995 |  | from photometry_core.py; stage over 4000; packed by LPA clusters, product stage kept |
| `photometry_phase2a_2.py` | phase2a photometry | 46 | 2584 |  | from photometry_core.py; stage over 4000; packed by LPA clusters, product stage kept |
| `photometry_epsf_hooks.py` | ePSF hooks | 2 | 198 |  | from photometry_core.py |
| `pipeline_epsf_hooks.py` | ePSF hooks | 2 | 50 |  | from pipeline.py |
| `photometry_exports.py` | exports-reports | 5 | 94 |  | from photometry_core.py |
| `photometry_ui_helpers.py` | UI-only | 2 | 24 |  | from photometry_core.py |
| `pipeline_ui_helpers.py` | UI-only | 5 | 216 |  | from pipeline.py |
| `photometry_gate_helpers.py` | gate-only | 25 | 797 |  | from photometry_core.py |
| `pipeline_gate_helpers.py` | gate-only | 1 | 146 |  | from pipeline.py |
| `photometry_dead.py` | unreachable | 15 | 448 |  | from photometry_core.py |
| `pipeline_dead.py` | unreachable | 12 | 454 |  | from pipeline.py |

## Defs per module (names)

### `pipeline_import.py` (20 defs, 661 lines)

Source files: pipeline.py

`fits_metadata_from_primary_header`, `extract_fits_metadata`, `scan_usb_folder`, `_enrich_calibration_metadata_from_header`, `_apply_draft_combined_to_pipeline_meta`, `generate_observation_hash`, `_merge_equipment_pixel_into_metadata`, `log_lights_binning_from_headers_preflight`, `_recompute_effective_pixel_from_physical`, `_log_effective_pixel_pitch`, `_summarize_lights_binning_from_headers`, `_fits_pixel_raw_to_micrometres`, `observation_group_key_from_metadata`, `_header_pick_first`, `_safe_filter_token`, `_parse_fits_binning_int`, `_valid_bayerpat_from_header`, `_fits_meta_dec_deg`, `_fits_meta_ra_deg`, `_focal_mm_plausible`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:AstroPipeline`
- `caller-in-other-module:_analyze_calibrated_qc_one`
- `caller-in-other-module:_calibrate_one_light_apply_masters_in_ram`
- `caller-in-other-module:_calibrate_one_light_disk`
- `caller-in-other-module:_obs_group_key_from_light_path`
- `caller-in-other-module:_pick_light_for_metadata_diagnostic`
- `caller-in-other-module:_plate_solve_input_bundle`
- `caller-in-other-module:_qc_enrich_one_frame`
- `caller-in-other-module:_resolve_focal_mm_for_plate_scale`
- `caller-in-other-module:calibrate_lights_to_calibrated`
- `caller-in-other-module:run_draft_ram_calibration_qc_to_obs_files`
- `caller-in-other-module:scan_calibrated_lights_pointing`
- `caller-in-other-module:validate_comparison_ensemble_flatness`

Who imports it (external files that call these defs today):
- `dev/tests/test_calibration_library_match.py`
- `dev/tests/test_except_fix2_top10.py`
- `dev/tests/test_except_fix3.py`
- `src_py/app.py`
- `src_py/importer.py`
- `src_py/night_run.py`

### `photometry_calibrate.py` (1 defs, 39 lines)

Source files: photometry_core.py

`compute_auto_fwhm_limit`

Imports it will need (same-file/cross-file callees outside this module):

Who imports it (external files that call these defs today):
- `dev/tests/test_export_parity_01.py`
- `src_py/night_run.py`
- `src_py/photometry_report.py`
- `src_py/ui_quality_dashboard.py`

### `pipeline_calibrate.py` (76 defs, 4121 lines)

Source files: pipeline.py

`AstroPipeline`, `calibrate_lights_to_calibrated`, `run_draft_ram_calibration_qc_to_obs_files`, `_calibrate_one_light_apply_masters_in_ram`, `_qc_enrich_calibrated_in_place`, `scan_calibrated_lights_pointing`, `_qc_enrich_one_frame`, `_fit_subtract_preprocess_sky_surface`, `_calibrate_one_light_disk`, `_robust_frame_fwhm_median`, `_calibrate_batch_process_one`, `apply_perf10_dao_qc_to_obs_files`, `_passthrough_lights_to_calibrated`, `run_osc_channel_extraction_for_archive`, `estimate_archive_memory_profile`, `_quality_inspection_dao_metrics_array`, `_post_calibration_qc_eval`, `_mean_hfr_bright_stars_dao`, `sync_obs_files_drift_arcmin_for_draft`, `_moment_fwhm_elong_peak_at`, `_sync_obs_calibration_state_with_retry`, `_effective_saturation_limit`, `_filter_light_paths_maybe`, `_inspection_jd_from_header`, `_qc_fwhm_elongation`, `_vyvar_parallel_worker_count`, `_log_calibration_io_preflight`, `_qc_pack_from_config`, `estimate_memory_from_fits_headers`, `_estimate_fov_deg_from_header`, `_sync_manifest_cal_stage_from_qc_row`, `draft_median_pointing_icrs_deg`, `_decide_preprocess_sky_action`, `_resolve_draft_light_raw_path`, `_infer_sat_limit_from_bitpix`, `_strip_raw_linearity_header_keywords`, `_init_calibrate_batch_worker`, `_calibration_flags`, `_dao_star_table_mean_roundness`, `_half_flux_radius_in_cutout`, `_saturation_adu_for_cal_diag`, `_resolve_dark_path_for_light`, `_archive_preprocess_lights_root`, `_log_calibration_metadata_diagnostic`, `_fits_primary_pixel_count`, `_pipeline_ui_error`, `_qc_center_crop_for_stars`, `_db_for_calibration_tasks`, `_perf10_lookup_qc`, `_saturate_limit_adu_from_header`, `_calibration_type_from_flags`, `_exposure_sec_from_header`, `format_memory_bytes`, `_infer_raw_light_path_for_calibrated`, `_cal_diag_session_from_export`, `_match_and_crop_pair`, `_pick_light_for_metadata_diagnostic`, `_cfg_calibration_library_native_binning`, `_estimate_dao_fwhm_guess`, `_archive_root_from_lights_root`, `_available_system_ram_bytes`, `_vy_calib_status_numeric`, `_vyvar_parallel_pool`, `_hdr_vy_cflag_str`, `_header_vyskyord`, `norm_fits_path_key`, `_obs_group_key_from_light_path`, `_cal_diag_export_for_workers`, `_light_binning_from_path`, `_vyvar_calibrate_multiprocessing_enabled`, `_vyvar_parallel_use_processes`, `_iter_light_fits`, `_vyvar_qc_preprocess_workers`, `SkySurfaceOrderConflictError`, `_has_usable_master_dark`, `_has_valid_wcs`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_analyze_calibrated_qc_one`
- `caller-in-other-module:_astrometry_align_impl_body`
- `caller-in-other-module:_estimate_fov_deg_from_fits_path`
- `caller-in-other-module:_export_per_frame_run_catalog_core`
- `caller-in-other-module:_load_raw_for_frame`
- `caller-in-other-module:_load_raw_hdr_for_frame`
- `caller-in-other-module:_quality_inspection_dao_metrics`
- `caller-in-other-module:_vyvar_per_frame_csv_workers`
- `caller-in-other-module:analyze_calibrated_qc`
- `caller-in-other-module:astrometry_align_and_build_masterstar`
- `caller-in-other-module:build_masterstar_from_detrended`
- `caller-in-other-module:build_prefilter_rejected_map`
- `caller-in-other-module:calibrated_paths_for_draft_apply_filters`
- `caller-in-other-module:detect_stars_and_match_catalog`
- `caller-in-other-module:detect_stars_match_master_reference`
- `caller-in-other-module:export_per_frame_catalogs`
- `caller-in-other-module:filter_files_by_qc_metrics_allowlist`
- `caller-in-other-module:generate_masterstar_and_catalog`
- `caller-in-other-module:load_qc_metrics_status_by_path`
- `caller-in-other-module:qc_enrich_calibrated_lights_in_place`
- `caller-in-other-module:resolve_preprocess_target_coordinates`
- `caller-in-other-module:run_quality_analysis`
- `pipeline.py:_qc_suggest_thresholds`
- `pipeline.py:_safe_filter_token`
- `pipeline.py:_summarize_lights_binning_from_headers`
- `pipeline.py:_valid_bayerpat_from_header`
- `pipeline.py:analyze_calibrated_qc`
- `pipeline.py:extract_fits_metadata`
- `pipeline.py:fits_metadata_from_primary_header`
- `pipeline.py:log_lights_binning_from_headers_preflight`
- `pipeline.py:observation_group_key_from_metadata`
- `pipeline.py:preprocess_calibrated_to_processed`

Who imports it (external files that call these defs today):
- `dev/scripts/archive/diag/_debug_fk343.py`
- `dev/scripts/archive/draft_runs/_complete_draft341_photometry.py`
- `dev/scripts/audit_stage3_part0b_rebuild.py`
- `dev/scripts/chiandh_allfilters_overnight.py`
- `dev/scripts/chiandh_continue375_solve.py`
- `dev/scripts/chiandh_inject_platesolve_phot.py`
- `dev/scripts/chiandh_resume_draft369.py`
- `dev/scripts/dy_peg_night_run_bvr.py`
- `dev/scripts/post453_preprocess_bench.py`
- `dev/scripts/post453_ui_startup_measure.py`
- `dev/scripts/qatar8_night_run_v.py`
- `dev/scripts/session_baseline_check.py`
- `dev/scripts/t3_validate_sky_surface_429.py`
- `dev/tests/test_cal_diag_v2_gate.py`
- `dev/tests/test_cal_stage_gate.py`
- `dev/tests/test_database_sqlite_threading.py`
- `dev/tests/test_except_fix2_top10.py`
- `dev/tests/test_masterstar_zone_classifier.py`
- `dev/tests/test_osc1_extraction.py`
- `dev/tests/test_preprocess_sky_surface.py`
- `dev/tests/test_robust_frame_fwhm.py`
- `dev/tests/test_sibling_wcs_recovery.py`
- `dev/tests/test_skipproc_qc_allowlist.py`
- `dev/tests/test_skysf_double_guard.py`
- `dev/tools/batch_e_physical_recut.py`
- `dev/tools/closure_batch_b_d52_mechanism.py`
- `dev/tools/inv_cal01_validate.py`
- `dev/tools/inv_cal02_validate.py`
- `dev/tools/sat_diag_dry_run.py`
- `src_py/app.py`

### `pipeline_astrometry__generate_masterstar_and_catalog.py` (1 defs, 2540 lines)

Source files: pipeline.py

`generate_masterstar_and_catalog`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_astrometry_align_impl_body`
- `caller-in-other-module:_pass2_sibling_wcs_recovery`
- `caller-in-other-module:_run_osc_multi_group_alignment`
- `photometry_core.py:_fwhm_moment_at`
- `photometry_core.py:bad_columns_for_light_frame`
- `photometry_core.py:common_field_intersection_bbox_px`
- `photometry_core.py:measure_fwhm_from_masterstar`
- `photometry_core.py:merge_photometry_pipeline_meta`
- `photometry_core.py:recommended_aperture_by_color`
- `photometry_core.py:stamp_vsx_known_variable_on_masterstars`
- `photometry_core.py:stress_test_relative_rms_from_sidecars`
- `photometry_core.py:vsx_is_known_variable_top3_per_bin`
- `pipeline.py:_annotate_masterstars_flux_zones`
- `pipeline.py:_effective_field_catalog_cone_radius_deg`
- `pipeline.py:_effective_saturation_limit`
- `pipeline.py:_equipment_saturate_adu_from_db`
- `pipeline.py:_fill_masterstars_gaia_matched_bp_rp_from_local_db`
- `pipeline.py:_has_valid_wcs`
- `pipeline.py:_invalidate_field_catalog_cone_cache_if_needed`
- `pipeline.py:_merge_platesolve_gaia_pairs_into_masterstars_df`
- `pipeline.py:_path_is_under_tree`
- `pipeline.py:_path_segments_forbidden_for_masterstar_physical_source`
- `pipeline.py:_plate_solve_input_bundle`
- `pipeline.py:_resolve_best_effort_path_under`
- `pipeline.py:_sat_adu_from_draft_sat_diag`
- `pipeline.py:_sync_comparison_stars_across_setups`
- `pipeline.py:_try_rescale_masterstar_linear_wcs_to_expected_plate_scale`
- `pipeline.py:_update_masterstar_obs_file_status`
- `pipeline.py:_vyvar_df_to_csv`
- `pipeline.py:_vyvar_open_database`
- `pipeline.py:build_masterstar_from_detrended`
- `pipeline.py:compute_plate_scale_from_db`
- `pipeline.py:detect_stars_and_match_catalog`
- `pipeline.py:draft_is_multi_group_obs`
- `pipeline.py:draft_median_pointing_icrs_deg`
- `pipeline.py:get_masterstar_candidate_rows`
- `pipeline.py:get_masterstar_candidates`
- `pipeline.py:resolve_masterstar_input_root`
- `pipeline.py:resolve_obs_file_to_processed_fits`
- `pipeline.py:resolve_plate_solve_fov_deg_hint`
- ... 1 more

Who imports it (external files that call these defs today):
- `dev/scripts/chiandh_allfilters_overnight.py`
- `dev/scripts/chiandh_continue375_solve.py`
- `dev/tests/test_pipeline_meta_provenance.py`
- `src_py/app.py`
- `src_py/pipeline.py`

### `pipeline_astrometry__detect_stars_and_match_catalog.py` (1 defs, 1372 lines)

Source files: pipeline.py

`detect_stars_and_match_catalog`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_export_per_frame_run_catalog_core`
- `caller-in-other-module:export_per_frame_catalogs`
- `caller-in-other-module:generate_masterstar_and_catalog`
- `photometry_core.py:stamp_masterstar_snr_columns`
- `pipeline.py:_all_pix2world_icrs_deg`
- `pipeline.py:_apply_wcs_tan_fragment_to_header`
- `pipeline.py:_box_peaks_at_centroids`
- `pipeline.py:_catalog_df_cap_brightest_by_mag`
- `pipeline.py:_catalog_match_radius_px`
- `pipeline.py:_chord_to_arcsec`
- `pipeline.py:_dao_auto_binning_factor`
- `pipeline.py:_dao_convolved_background_rms_adu`
- `pipeline.py:_dao_noise_sigma_adu`
- `pipeline.py:_dao_spatial_flux_cap_row_indices`
- `pipeline.py:_detect_empirical_clip_level_adu`
- `pipeline.py:_effective_field_catalog_cone_radius_deg`
- `pipeline.py:_effective_saturation_limit`
- `pipeline.py:_exo_host_annotation_arrays`
- `pipeline.py:_gaia_chip_xy_from_catalog`
- `pipeline.py:_icrs_deg_to_unitxyz`
- `pipeline.py:_mean_bin2d_for_dao`
- `pipeline.py:_prefilter_dao_table_brightest`
- `pipeline.py:_proc_rename_det_names_to_catalog_id`
- `pipeline.py:_proc_sat_block_for_csv`
- `pipeline.py:_query_exoplanet_local`
- `pipeline.py:_query_gaia_local`
- `pipeline.py:_query_vsx_local`
- `pipeline.py:_resolve_peak_saturation_limit_adu`
- `pipeline.py:_slice_exo_annotation`
- `pipeline.py:_vectorized_star_saturation_columns`
- `pipeline.py:_vyvar_df_to_csv`
- `pipeline.py:_write_field_catalog_cone_meta`
- `pipeline.py:build_ucac_catalog_kdtree`
- `pipeline.py:nearest_sky_nn_kdtree`
- `pipeline.py:resolve_plate_solve_fov_deg_hint`

Who imports it (external files that call these defs today):
- `dev/scripts/audit_stage3_part2b_threshold_sweep.py`

### `pipeline_astrometry__export_per_frame_catalogs.py` (1 defs, 1101 lines)

Source files: pipeline.py

`export_per_frame_catalogs`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_astrometry_align_impl_body`
- `pipeline.py:_apply_aperture_catalog_enhancements_from_st`
- `pipeline.py:_apply_exo_host_columns_to_proc_df`
- `pipeline.py:_effective_field_catalog_cone_radius_deg`
- `pipeline.py:_epsf_fit_catalog_ids`
- `pipeline.py:_equipment_saturate_adu_from_db`
- `pipeline.py:_estimate_catalog_frame_hw`
- `pipeline.py:_export_catalog_psf_st_fields`
- `pipeline.py:_export_per_frame_disk_worker_task`
- `pipeline.py:_export_per_frame_ram_worker_task`
- `pipeline.py:_extract_airmass_from_header`
- `pipeline.py:_field_catalog_cone_meta_path`
- `pipeline.py:_fill_psf_catalog_columns`
- `pipeline.py:_finalize_hybrid_bkg_fallback_sidecar`
- `pipeline.py:_has_valid_wcs`
- `pipeline.py:_init_export_per_frame_worker`
- `pipeline.py:_invalidate_field_catalog_cone_cache_if_needed`
- `pipeline.py:_prefetch_export_shared_catalog_for_process_pool`
- `pipeline.py:_proc_catalog_keep_matched_rows_only`
- `pipeline.py:_proc_deduplicate_matched_catalog_rows`
- `pipeline.py:_proc_drop_unmatched_dao_rows`
- `pipeline.py:_query_gaia_local`
- `pipeline.py:_query_vsx_local`
- `pipeline.py:_resolve_draft_light_raw_path`
- `pipeline.py:_vyvar_cap_mp_workers_for_catalog`
- `pipeline.py:_vyvar_df_to_csv`
- `pipeline.py:_vyvar_per_frame_csv_workers`
- `pipeline.py:_write_field_catalog_cone_meta`
- `pipeline.py:build_ucac_catalog_kdtree`
- `pipeline.py:detect_stars_and_match_catalog`
- `pipeline.py:detect_stars_match_master_reference`
- `pipeline.py:find_qc_metrics_csv`
- `pipeline.py:resolve_plate_solve_fov_deg_hint`

Who imports it (external files that call these defs today):
- `dev/scripts/archive/draft_runs/_complete_draft341_photometry.py`
- `dev/scripts/archive/verify/_reexport_phase2a_draft307.py`
- `dev/scripts/archive/verify/_todo8_epsf_final_verify_draft321.py`
- `dev/scripts/archive/verify/_todo8_epsf_verify_draft321.py`
- `dev/scripts/bingain_acceptance_run.py`
- `dev/scripts/pilot_palomar7_deep_gaia_ab.py`
- `dev/scripts/pilot_palomar7_epsf_phot364.py`
- `dev/scripts/pilot_palomar7_part2c_364.py`
- `dev/scripts/run_draft_000244_align_and_photometry.py`
- `dev/tests/test_pre_cal_proc_csv_naming_e2e.py`

### `pipeline_astrometry__astrometry_align_impl_body.py` (1 defs, 1049 lines)

Source files: pipeline.py

`_astrometry_align_impl_body`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_pass2_sibling_wcs_recovery`
- `caller-in-other-module:_run_osc_multi_group_alignment`
- `caller-in-other-module:astrometry_align_and_build_masterstar`
- `pipeline.py:_apply_wcs_header_to_fits`
- `pipeline.py:_assert_alignment_produced_fits`
- `pipeline.py:_ensure_parent_dirs_for_aligned_fits`
- `pipeline.py:_equipment_saturate_adu_from_db`
- `pipeline.py:_finalize_hybrid_bkg_fallback_sidecar`
- `pipeline.py:_has_valid_wcs`
- `pipeline.py:_photometry_mode_run_flags`
- `pipeline.py:_pick_reference_frame_by_star_count`
- `pipeline.py:_pipeline_ui_error`
- `pipeline.py:_pipeline_ui_info`
- `pipeline.py:_solve_wcs_external`
- `pipeline.py:_vyvar_open_database`
- `pipeline.py:_vyvar_parallel_worker_count`
- `pipeline.py:_wcs_field_center_radec_deg`
- `pipeline.py:compute_plate_scale_from_db`
- `pipeline.py:estimate_archive_memory_profile`
- `pipeline.py:export_per_frame_catalogs`
- `pipeline.py:generate_masterstar_and_catalog`
- `pipeline.py:resolve_plate_solve_fov_deg_hint`
- `pipeline.py:write_photometry_plan_files`

Who imports it (external files that call these defs today):
- `dev/tests/test_astrometry_fault_isolation.py`

### `pipeline_astrometry.py` (65 defs, 3995 lines)

Source files: pipeline.py

`write_photometry_plan_files`, `_solve_wcs_external`, `astrometry_align_and_build_masterstar`, `select_comparison_stars_spatial_grid`, `_pass2_sibling_wcs_recovery`, `_run_osc_multi_group_alignment`, `detect_field_jumps`, `_try_rescale_masterstar_linear_wcs_to_expected_plate_scale`, `_plate_solve_input_bundle`, `_query_vsx_local_frame_bbox`, `get_masterstar_candidate_rows`, `_update_masterstar_obs_file_status`, `_merge_vsx_exoplanet_variable_targets`, `resolve_plate_solve_fov_deg_hint`, `_fill_masterstars_gaia_matched_bp_rp_from_local_db`, `_sync_comparison_stars_across_setups`, `_merge_platesolve_gaia_pairs_into_masterstars_df`, `_resolve_focal_mm_for_plate_scale`, `resolve_masterstar_input_root`, `resolve_obs_file_to_processed_fits`, `compute_plate_scale_from_db`, `_merge_astrometry_group_reports`, `_photometry_mode_run_flags`, `_vyvar_df_to_csv`, `_partition_detrended_by_subfolder`, `_wcs_field_center_radec_deg`, `filter_files_by_qc_metrics_allowlist`, `_equipment_saturate_adu_from_db`, `_path_segments_forbidden_for_masterstar_physical_source`, `_pipeline_ui_info`, `load_qc_metrics_status_by_path`, `_apply_wcs_header_to_fits`, `_header_focal_length_mm`, `draft_obs_group_count`, `_export_catalog_psf_st_fields`, `build_prefilter_rejected_map`, `_sat_adu_from_draft_sat_diag`, `_field_jump_empty_result`, `_finite_positive_adu`, `_vyvar_df_round_time_jd_for_csv`, `_assert_alignment_produced_fits`, `_vyvar_per_frame_csv_workers`, `get_masterstar_candidates`, `_vyvar_open_database`, `_ensure_parent_dirs_for_aligned_fits`, `draft_is_multi_group_obs`, `calibrated_paths_for_draft_apply_filters`, `qc_enrich_calibrated_lights_in_place`, `resolve_preprocess_target_coordinates`, `_qc_suggest_thresholds`, `preprocess_calibrated_to_processed`, `_archive_raw_to_calibrated_light`, `_load_raw_for_frame`, `_load_raw_hdr_for_frame`, `build_masterstar_from_detrended`, `_resolve_best_effort_path_under`, `_sort_masterstar_paths_by_fwhm`, `_strip_external_platesolve_header`, `_pick_preferred_masterstar_basename_hit`, `_header_vy_fwhm_px`, `_safe_proc_name`, `_path_is_under_tree`, `_dao_targeted_pass2_unmatched_gaia`, `_merge_dao_pass1_pass2_tables`, `_catalog_match_radius_px`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:AstroPipeline`
- `caller-in-other-module:_annotate_masterstars_flux_zones`
- `caller-in-other-module:_apply_exo_host_columns_to_proc_df`
- `caller-in-other-module:_astrometry_align_impl_body`
- `caller-in-other-module:_export_per_frame_run_catalog_core`
- `caller-in-other-module:_prefetch_export_shared_catalog_for_process_pool`
- `caller-in-other-module:_resolve_light_fits_for_quality_inspection`
- `caller-in-other-module:_solve_wcs_astrometry_net`
- `caller-in-other-module:_solve_wcs_solve_field_cli`
- `caller-in-other-module:detect_stars_and_match_catalog`
- `caller-in-other-module:detect_stars_match_master_reference`
- `caller-in-other-module:export_per_frame_catalogs`
- `caller-in-other-module:generate_masterstar_and_catalog`
- `caller-in-other-module:get_auto_fov`
- `caller-in-other-module:list_best_processed_light_paths_for_masterstar`
- `photometry_core.py:common_field_intersection_bbox_px`
- `photometry_core.py:common_field_intersection_bbox_px_from_arrays`
- `pipeline.py:_all_pix2world_icrs_deg`
- `pipeline.py:_archive_preprocess_lights_root`
- `pipeline.py:_astrometry_align_impl_body`
- `pipeline.py:_build_exoplanet_promotion_rows_from_masterstars`
- `pipeline.py:_effective_field_catalog_cone_radius_deg`
- `pipeline.py:_filter_light_paths_maybe`
- `pipeline.py:_focal_mm_plausible`
- `pipeline.py:_qc_enrich_calibrated_in_place`
- `pipeline.py:_resolve_draft_light_raw_path`
- `pipeline.py:_vyvar_parallel_worker_count`
- `pipeline.py:draft_median_pointing_icrs_deg`
- `pipeline.py:extract_fits_metadata`
- `pipeline.py:find_qc_metrics_csv`
- `pipeline.py:generate_masterstar_and_catalog`
- `pipeline.py:norm_fits_path_key`

Who imports it (external files that call these defs today):
- `dev/scripts/chiandh_inject_platesolve_phot.py`
- `dev/scripts/diagnose_epsf_quality_364.py`
- `dev/scripts/diagnose_psf_elongation_362.py`
- `dev/scripts/palomar7_continue367_bgr.py`
- `dev/scripts/pilot_palomar7_continue364.py`
- `dev/scripts/pilot_palomar7_deep_gaia_ab.py`
- `dev/scripts/session_baseline_check.py`
- `dev/scripts/test_border_bbox.py`
- `dev/scripts/validate_exo_as_target_422.py`
- `dev/scripts/verify_adaptive_faint_targets_364.py`
- `dev/scripts/verify_adaptive_wiring_364.py`
- `dev/tests/test_astrometry_fault_isolation.py`
- `dev/tests/test_border_ram_handoff.py`
- `dev/tests/test_except_fix2_top10.py`
- `dev/tests/test_exoplanet_local_match.py`
- `dev/tests/test_exoplanet_promotion_restore.py`
- `dev/tests/test_exoplanet_variable_targets_merge.py`
- `dev/tests/test_export_parity_01.py`
- `dev/tests/test_field_run_findings.py`
- `dev/tests/test_g1_f003_alignment_pixel_fallback.py`
- `dev/tests/test_inv_match_identity_01.py`
- `dev/tests/test_invariants_p2.py`
- `dev/tests/test_masterstar_gaia_01.py`
- `dev/tests/test_masterstar_obs_group.py`
- `dev/tests/test_pre_calibrated_run.py`
- `dev/tests/test_sibling_wcs_recovery.py`
- `dev/tests/test_skipproc_qc_allowlist.py`
- `dev/tests/test_skysf_double_guard.py`
- `dev/tools/sat_diag_dry_run.py`
- `src_py/app.py`

### `pipeline_astrometry_2.py` (73 defs, 3687 lines)

Source files: pipeline.py

`_export_per_frame_run_catalog_core`, `_fill_psf_catalog_columns`, `_build_exoplanet_promotion_rows_from_masterstars`, `_apply_exo_host_columns_to_proc_df`, `_query_gaia_local`, `_query_vsx_local`, `_prefetch_export_shared_catalog_for_process_pool`, `_epsf_target_catalog_ids`, `_query_exoplanet_local`, `_exo_host_annotation_arrays`, `_gaia_catalog_cone_radius_optics_floor_deg`, `_invalidate_field_catalog_cone_cache_if_needed`, `_effective_field_catalog_cone_radius_deg`, `_field_center_and_radius_from_wcs`, `_proc_deduplicate_matched_catalog_rows`, `_export_per_frame_ram_worker_task`, `_export_first_icrs_center_radius`, `_epsf_fit_catalog_ids`, `_icrs_center_radius_from_hdr_data`, `_write_field_catalog_cone_meta`, `_estimate_catalog_frame_hw`, `_init_export_per_frame_worker`, `_vyvar_cap_mp_workers_for_catalog`, `build_ucac_catalog_kdtree`, `_export_per_frame_disk_worker_task`, `_catalog_df_cap_brightest_by_mag`, `_sat_ctx_from_worker`, `_proc_catalog_keep_matched_rows_only`, `_proc_drop_unmatched_dao_rows`, `_fits_header_first_positive_float`, `_cfg_from_export_worker_state`, `_field_catalog_cone_meta_path`, `detect_stars_match_master_reference`, `_lock_matched_centroids_to_master_grid`, `_dao_spatial_flux_cap_row_indices`, `_vectorized_star_saturation_columns`, `_resolve_peak_saturation_limit_adu`, `_box_peaks_at_centroids`, `_apply_dao_centroid_wcs_guard`, `_pixel_noise_sigma_pp_adu`, `_saturated_core_plateau_vectorized`, `_gaia_chip_xy_from_catalog`, `_dao_convolved_background_rms_adu`, `nearest_sky_nn_kdtree`, `_dao_detection_threshold_adu`, `_detect_empirical_clip_level_adu`, `_dao_noise_sigma_adu`, `_fits_header_vy_algn_aligned`, `_proc_rename_det_names_to_catalog_id`, `_all_pix2world_icrs_deg`, `_mean_bin2d_for_dao`, `_apply_wcs_tan_fragment_to_header`, `_proc_sat_block_for_csv`, `_prefilter_dao_table_brightest`, `_icrs_deg_to_unitxyz`, `_chord_to_arcsec`, `_dao_auto_binning_factor`, `_slice_exo_annotation`, `_annotate_masterstars_flux_zones`, `_resolve_masterstar_bg_sigma_adu`, `_masterstar_zone_linear_threshold`, `_masterstar_zone_log_once`, `_compute_airmass_from_altaz`, `_extract_airmass_from_header`, `_airmass_from_altitude_deg`, `_apply_aperture_catalog_enhancements_from_st`, `_dao_star_count_from_array`, `_pick_reference_frame_by_star_count`, `_bin2d_mean`, `_finalize_hybrid_bkg_fallback_sidecar`, `find_qc_metrics_csv`, `_box_peak_max_adu`, `inv_sat_limit_peak_test_adu`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_astrometry_align_impl_body`
- `caller-in-other-module:_star_saturation_flags`
- `caller-in-other-module:_try_rescale_masterstar_linear_wcs_to_expected_plate_scale`
- `caller-in-other-module:astrometry_align_and_build_masterstar`
- `caller-in-other-module:detect_stars_and_match_catalog`
- `caller-in-other-module:export_per_frame_catalogs`
- `caller-in-other-module:generate_masterstar_and_catalog`
- `caller-in-other-module:write_photometry_plan_files`
- `photometry_core.py:compute_fwhm_gaussian_for_aperture_catalog`
- `photometry_core.py:enhance_catalog_dataframe_aperture_bpm`
- `photometry_core.py:finalize_hybrid_bkg_fallback_proc_dir`
- `pipeline.py:_dao_targeted_pass2_unmatched_gaia`
- `pipeline.py:_effective_saturation_limit`
- `pipeline.py:_finite_positive_adu`
- `pipeline.py:_has_valid_wcs`
- `pipeline.py:_load_raw_for_frame`
- `pipeline.py:_load_raw_hdr_for_frame`
- `pipeline.py:_vyvar_df_to_csv`
- `pipeline.py:detect_stars_and_match_catalog`
- `pipeline.py:resolve_plate_solve_fov_deg_hint`

Who imports it (external files that call these defs today):
- `dev/scripts/audit_stage3_part0b_rebuild.py`
- `dev/scripts/audit_stage3_part2_threshold_sweep.py`
- `dev/scripts/audit_stage3_part2b_threshold_sweep.py`
- `dev/scripts/diag_428_coord_forensics_v4.py`
- `dev/scripts/diag_428_coord_forensics_v5.py`
- `dev/scripts/diag_428_unmatched_sep.py`
- `dev/scripts/pilot_palomar7_deep_gaia_ab.py`
- `dev/scripts/verify_adaptive_faint_targets_364.py`
- `dev/scripts/verify_adaptive_wiring_364.py`
- `dev/tests/test_batch_e_recut.py`
- `dev/tests/test_c3_comp_rms_loo.py`
- `dev/tests/test_dao_convolved_threshold_option_b.py`
- `dev/tests/test_dao_sigma_pp_estimator.py`
- `dev/tests/test_epsf_psf_merge.py`
- `dev/tests/test_err_background_empirical.py`
- `dev/tests/test_except_fix2_top10.py`
- `dev/tests/test_exoplanet_local_match.py`
- `dev/tests/test_exoplanet_promotion_restore.py`
- `dev/tests/test_field_run_findings.py`
- `dev/tests/test_g1_f003_alignment_pixel_fallback.py`
- `dev/tests/test_geo.py`
- `dev/tests/test_master_grid_photometry.py`
- `dev/tests/test_masterstar_zone_classifier.py`
- `dev/tests/test_obsloc_null_island.py`
- `dev/tests/test_pfs_semantics_01.py`
- `dev/tests/test_proc_catalog_dedupe.py`
- `dev/tests/test_proc_dedupe_catalog_id.py`
- `dev/tests/test_skipproc_qc_allowlist.py`
- `dev/tests/validation/recover.py`
- `dev/tools/dao_phys_measure.py`

### `photometry_comp__run_phase0_and_phase1.py` (1 defs, 1078 lines)

Source files: photometry_core.py

`run_phase0_and_phase1`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:run_full_photometry_pipeline`
- `photometry_core.py:_angular_distance_deg`
- `photometry_core.py:_attach_predicted_dilution_report`
- `photometry_core.py:_batch_enrich_targets_bp_rp_from_gaia_db`
- `photometry_core.py:_enrich_target_bp_rp_from_gaia_db`
- `photometry_core.py:_normalize_id_series`
- `photometry_core.py:_normalize_id_value`
- `photometry_core.py:_phase0_effective_frame_hw_px`
- `photometry_core.py:_read_field_density_inputs`
- `photometry_core.py:_refresh_variable_targets_xy`
- `photometry_core.py:_resolve_frame_hw_px_from_masterstar`
- `photometry_core.py:_resolve_plate_scale_arcsec_per_px`
- `photometry_core.py:_write_suspected_variables`
- `photometry_core.py:build_global_comp_pool`
- `photometry_core.py:build_gs11_summary`
- `photometry_core.py:merge_photometry_pipeline_meta`
- `photometry_core.py:select_active_targets`
- `photometry_core.py:select_comparison_stars_per_target`

Who imports it (external files that call these defs today):
- `dev/scripts/_rerun_phase01_draft365.py`
- `dev/scripts/archive/draft_runs/_gs11_validate_step_b.py`
- `dev/scripts/archive/draft_runs/_rerun_phase01_draft343.py`
- `dev/scripts/archive/verify/_smoke_run_draft287.py`
- `dev/scripts/m67_continue368_gr_merge_sandbox.py`
- `dev/scripts/run_draft_000244_align_and_photometry.py`
- `dev/tools/comp_assign_01_phase1.py`
- `src_py/photometry_core.py`

### `photometry_comp.py` (29 defs, 3017 lines)

Source files: photometry_core.py

`select_comparison_stars_per_target`, `select_active_targets`, `build_global_comp_pool`, `_select_comps_by_rms_then_color`, `_write_suspected_variables`, `_enrich_comp_bp_rp`, `_enrich_active_targets_bp_rp`, `_select_comps_tiered`, `_enrich_target_bp_rp_from_gaia_db`, `_batch_enrich_targets_bp_rp_from_gaia_db`, `_refresh_variable_targets_xy`, `_read_field_density_inputs`, `_resolve_frame_hw_px_from_masterstar`, `ensure_full_variable_targets_if_presel_stub`, `_auto_repair_catalog_ids`, `_warn_zero_compstars_edge`, `_count_gate_passing_comps`, `_attach_predicted_dilution_report`, `_phase0_effective_frame_hw_px`, `_select_comps_by_color_then_rms`, `_dedupe_comp_pool_by_gaia_key`, `_bprp_tier_ladder_for_selection`, `_variable_targets_looks_like_ct_presel_stub`, `_active_target_zone_flag`, `_ensure_active_target_display_names`, `_normalize_id_value`, `_sid_int`, `_bool_col`, `_normalize_id_series`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:run_full_photometry_pipeline`
- `caller-in-other-module:run_phase0_and_phase1`
- `photometry_core.py:_normalize_gaia_id`
- `photometry_core.py:_safe_polyfit`
- `photometry_core.py:_target_display_name`
- `photometry_core.py:merge_photometry_pipeline_meta`
- `pipeline.py:write_photometry_plan_files`

Who imports it (external files that call these defs today):
- `dev/scripts/archive/draft_runs/_gs11_phase1_only.py`
- `dev/scripts/comp_pool_r_forensic.py`
- `dev/tests/test_c3_comp_rms_loo.py`
- `dev/tests/test_comp_determinism_synthetic.py`
- `dev/tests/test_comp_rms_gate_authoritative.py`
- `dev/tests/test_f428_fixbatch.py`
- `dev/tests/test_forced_phot_and_weights.py`
- `dev/tests/test_frame_hw_from_naxis.py`
- `dev/tests/test_gate_regime_01.py`
- `dev/tests/test_osc2_wcs_photometry.py`
- `dev/tests/test_phase0_identity_gate.py`
- `dev/tests/test_phase1_duplicate_comp_pool.py`
- `dev/tests/test_photometry_core.py`
- `dev/tests/test_post451_part_b.py`
- `dev/tests/test_select_active_targets_excludes_unmatched_vsx.py`
- `dev/tests/test_target_depth_01.py`
- `dev/tests/test_vsx_out_of_scope_types.py`
- `dev/tools/comp_admit_03_measure.py`
- `dev/tools/docs_pdf/flow_doc_facts.py`
- `src_py/comp_selection_per_target.py`
- `src_py/photometry_core.py`
- `src_py/pinned_ensembles.py`

### `photometry_shared.py` (32 defs, 1936 lines)

Source files: photometry_core.py

`enhance_catalog_dataframe_aperture_bpm`, `run_full_photometry_pipeline`, `_get_plate_scale_from_cfg`, `finalize_hybrid_bkg_fallback_proc_dir`, `stamp_vsx_known_variable_on_masterstars`, `stress_test_relative_rms_from_sidecars`, `stamp_masterstar_snr_columns`, `compute_fwhm_gaussian_for_aperture_catalog`, `_read_plate_scale_from_fits_path`, `_resolve_git_provenance`, `vsx_is_known_variable_top3_per_bin`, `build_gs11_summary`, `_build_pipeline_provenance_block`, `_cd_matrix_scale_arcsec_per_px`, `_resolve_plate_scale_arcsec_per_px`, `_fwhm_moment_at`, `common_field_intersection_bbox_px`, `recommended_aperture_by_color`, `bad_columns_for_light_frame`, `_complete_config_snapshot`, `_safe_polyfit`, `merge_photometry_pipeline_meta`, `_get_lc_adaptive`, `classify_git_dirty_paths`, `_target_display_name`, `common_field_intersection_bbox_px_from_arrays`, `_json_safe_snapshot_value`, `_is_import_relevant_py_path`, `_porcelain_status_by_path`, `_normalize_gaia_id`, `_angular_distance_deg`, `StressTestResult`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_build_csv_lookup`
- `caller-in-other-module:_comp_maps_from_comparison_stars_csv`
- `caller-in-other-module:_dedupe_comp_pool_by_gaia_key`
- `caller-in-other-module:_ensure_active_target_display_names`
- `caller-in-other-module:_group_comp_mag_inst_from_proc_csvs`
- `caller-in-other-module:_load_blend_worklist`
- `caller-in-other-module:_lookup_star_in_csv`
- `caller-in-other-module:_measured_aperture_from_proc_cache`
- `caller-in-other-module:_per_frame_sat_flags_for_catalog_id`
- `caller-in-other-module:_phase2a_finalize_exports`
- `caller-in-other-module:_phase2a_prepare_shared_state`
- `caller-in-other-module:_phase2a_process_one_target`
- `caller-in-other-module:_phase2a_star_mag_lookup`
- `caller-in-other-module:_phase2a_write_summary`
- `caller-in-other-module:_resolve_photometric_aperture_px_for_gs11`
- `caller-in-other-module:_write_suspected_variables`
- `caller-in-other-module:apply_per_frame_saturation_to_active_targets`
- `caller-in-other-module:build_global_comp_pool`
- `caller-in-other-module:build_rms_mag_model`
- `caller-in-other-module:compute_aperture_correction`
- `caller-in-other-module:compute_optimal_apertures`
- `caller-in-other-module:democratic_detrend_lc`
- `caller-in-other-module:fit_color_term_c1`
- `caller-in-other-module:photometer_check_star_production_path`
- `caller-in-other-module:read_flux_from_csv`
- `caller-in-other-module:run_phase0_and_phase1`
- `caller-in-other-module:run_phase2a`
- `caller-in-other-module:save_field_map_png`
- `caller-in-other-module:select_active_targets`
- `photometry_core.py:_aperture_flux_sky_batch`
- `photometry_core.py:_assert_inv_err_sigma_acct_01`
- `photometry_core.py:_clamp_err_empty_apertures_min`
- `photometry_core.py:_clamp_err_empty_apertures_n`
- `photometry_core.py:_coerce_bool_cell`
- `photometry_core.py:_compute_fov_max_dist`
- `photometry_core.py:_finite_pixel_bbox_from_array`
- `photometry_core.py:_intersection_bbox_from_frame_bboxes`
- `photometry_core.py:_labbe_content_seed_from_header`
- `photometry_core.py:_resolve_frame_hw_px_from_masterstar`
- `photometry_core.py:_sigma_bkg_r_key`
- ... 11 more

Who imports it (external files that call these defs today):
- `dev/scripts/anchor435_protocol_v2.py`
- `dev/scripts/anchor_pair_run.py`
- `dev/scripts/anchor_recut_sigma_proof.py`
- `dev/scripts/archive/draft_runs/_complete_draft341_photometry.py`
- `dev/scripts/archive/draft_runs/_gs11_aperture_diagnostic.py`
- `dev/scripts/audit_stage3_part0b_rebuild.py`
- `dev/scripts/audit_stage3_part0c_cohort_delta.py`
- `dev/scripts/audit_stage3_part0d_delta_forensics.py`
- `dev/scripts/audit_stage3_part1b_check_chi2.py`
- `dev/scripts/audit_stage3_part1c_robust_chi2.py`
- `dev/scripts/audit_stage3_part2_threshold_sweep.py`
- `dev/scripts/audit_stage3_part2b_threshold_sweep.py`
- `dev/scripts/bingain_patch_sigma_bkg.py`
- `dev/scripts/chiandh_allfilters_overnight.py`
- `dev/scripts/chiandh_continue375_solve.py`
- `dev/scripts/chiandh_continue_bvr_phot.py`
- `dev/scripts/chiandh_ct_dump_bvr_375.py`
- `dev/scripts/chiandh_inject_platesolve_phot.py`
- `dev/scripts/chiandh_resume_draft369.py`
- `dev/scripts/draft_426_regen.py`
- `dev/scripts/force_aperture_r_px.py`
- `dev/scripts/m67_continue368_bgr_phot.py`
- `dev/scripts/m67_continue368_gr_merge_sandbox.py`
- `dev/scripts/m67_continue368_gr_sandbox.py`
- `dev/scripts/palomar7_continue367_bgr.py`
- `dev/scripts/palomar7_photometry_180_bgr.py`
- `dev/scripts/pilot_palomar7_continue364.py`
- `dev/scripts/pilot_palomar7_deep_gaia_ab.py`
- `dev/scripts/pilot_palomar7_epsf_phot364.py`
- `dev/scripts/pilot_palomar7_rerun_phot364.py`

### `photometry_phase2a__phase2a_process_one_target.py` (1 defs, 1611 lines)

Source files: photometry_core.py

`_phase2a_process_one_target`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:photometer_check_star_production_path`
- `caller-in-other-module:run_phase2a`
- `photometry_core.py:_Phase2AState`
- `photometry_core.py:_ac_summary_fields`
- `photometry_core.py:_angular_distance_deg`
- `photometry_core.py:_append_ct_prototype_row`
- `photometry_core.py:_check_color_term_extrapolation`
- `photometry_core.py:_color_term_cat_inst_scatter_pair`
- `photometry_core.py:_combine_err_with_ensemble_scatter_keyed`
- `photometry_core.py:_ct_prototype_enabled`
- `photometry_core.py:_draft_dir_from_phase2a_paths`
- `photometry_core.py:_ensemble_scatter_by_source_file`
- `photometry_core.py:_err_budget_components_keyed`
- `photometry_core.py:_exclude_err_scatter_unmatched_epochs`
- `photometry_core.py:_get_comp_bjd_series`
- `photometry_core.py:_get_lc`
- `photometry_core.py:_load_adaptive_blend_map`
- `photometry_core.py:_measured_aperture_from_proc_cache`
- `photometry_core.py:_normalize_gaia_id`
- `photometry_core.py:_phase2a_skip_empty_comps_target`
- `photometry_core.py:_recompute_bjd_hjd_with_status`
- `photometry_core.py:_resolve_photometric_aperture_px_for_gs11`
- `photometry_core.py:_route_lc_per_frame_err`
- `photometry_core.py:_target_display_name`
- `photometry_core.py:apply_color_term`
- `photometry_core.py:apply_reporting_postprocess`
- `photometry_core.py:check_comparison_stability`
- `photometry_core.py:compute_aperture_correction`
- `photometry_core.py:compute_lc_flux_method`
- `photometry_core.py:compute_lc_rms_ooe`
- `photometry_core.py:ct_ensemble_reference_maps`
- `photometry_core.py:democratic_detrend_lc`
- `photometry_core.py:ensemble_normalize`
- `photometry_core.py:fit_color_term_c1`
- `photometry_core.py:pytics_iterative_weights`
- `photometry_core.py:read_flux_from_csv`
- `photometry_core.py:save_cutout_png`
- `photometry_core.py:save_lightcurve_csv`
- `photometry_core.py:save_lightcurve_png`
- `photometry_core.py:save_target_field_map_png`
- ... 2 more

Who imports it (external files that call these defs today):
- `dev/tests/test_phase2a_saturated_skip.py`

### `photometry_phase2a__phase2a_prepare_shared_state.py` (1 defs, 940 lines)

Source files: photometry_core.py

`_phase2a_prepare_shared_state`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:run_phase2a`
- `photometry_core.py:_ColorTermGroupFit`
- `photometry_core.py:_Phase2AState`
- `photometry_core.py:_build_csv_lookup`
- `photometry_core.py:_build_phase2a_resolved_facts`
- `photometry_core.py:_compute_frame_align_residuals`
- `photometry_core.py:_compute_group_color_term_fit`
- `photometry_core.py:_draft_dir_from_phase2a_paths`
- `photometry_core.py:_ensure_group_comp_pool_csv`
- `photometry_core.py:_frame_align_residual_gate_select`
- `photometry_core.py:_normalize_gaia_id`
- `photometry_core.py:_phase2a_attempt_k2_night_fit`
- `photometry_core.py:_phase2a_cache_columns`
- `photometry_core.py:_phase2a_coerce_skip_photometry`
- `photometry_core.py:_record_align_residuals_to_report`
- `photometry_core.py:_require_comparison_stars_per_target_schema`
- `photometry_core.py:_resolve_phase2a_equipment_id`
- `photometry_core.py:_resolve_plate_scale_arcsec_per_px`
- `photometry_core.py:_sat_limit_peak_adu`
- `photometry_core.py:apply_per_frame_saturation_to_active_targets`
- `photometry_core.py:compute_optimal_apertures`
- `photometry_core.py:evaluate_cog_night_apcorr_gate`
- `photometry_core.py:measure_fwhm_from_masterstar`
- `photometry_core.py:read_flux_from_csv`
- `photometry_core.py:resolve_apply_color_term`
- `photometry_core.py:save_field_map_png`
- `pipeline.py:inv_sat_limit_peak_test_adu`

Who imports it (external files that call these defs today):
- `dev/scripts/audit_stage3_part1b_check_chi2.py`
- `dev/scripts/audit_stage3_part1c_robust_chi2.py`
- `dev/tools/wide_err_step0_checkstar.py`
- `dev/tools/wide_err_w1w2.py`
- `dev/tools/wide_error_budget_diag.py`

### `photometry_phase2a.py` (63 defs, 3995 lines)

Source files: photometry_core.py

`_phase2a_finalize_exports`, `auto_export_variability_candidates_csv`, `read_flux_from_csv`, `measure_fwhm_from_masterstar`, `compute_aperture_correction`, `save_field_map_png`, `_edge_ok_from_masterstar_pipeline`, `_phase2a_write_summary`, `_phase2a_proc_column_requirements`, `classify_lc_quality`, `_compute_frame_align_residuals`, `compute_optimal_apertures`, `build_lc_quality_summary`, `_ensure_group_comp_pool_csv`, `_lookup_star_in_csv`, `_build_csv_lookup`, `resolve_variable_targets_csv`, `_propagate_phase2a_skip_reason_to_active`, `_frame_align_residual_gate_select`, `resolve_apply_color_term`, `_measured_aperture_from_proc_cache`, `_resolve_phase2a_equipment_id`, `_photometric_error_with_bkg_mode`, `_howell_variance_adu2`, `_phase2a_empirical_sigma_bkg_ap`, `_resolve_photometric_aperture_px_for_gs11`, `parse_comp_quality_json_map`, `_photometric_error`, `_record_align_residuals_to_report`, `_sky_pp_for_photometric_error`, `_require_comparison_stars_per_target_schema`, `_phase2a_coerce_skip_photometry`, `expected_rms_from_model`, `_phase2a_cache_columns`, `_proc_stem`, `_sat_limit_peak_adu`, `run_phase2a`, `democratic_detrend_lc`, `_compute_group_color_term_fit`, `_build_phase2a_dynamic_params`, `_phase2a_attempt_k2_night_fit`, `fit_color_term_c1`, `should_apply_color_term`, `_Phase2AState`, `build_rms_mag_model`, `_phase2a_compute_lunar_context`, `_phase2a_observer_location_dict`, `_group_comp_mag_inst_from_proc_csvs`, `_sky_surface_meta_from_qc`, `_phase2a_resolve_field_center_ra_dec`, `_comp_maps_from_comparison_stars_csv`, `_group_comp_mag_inst_from_flux_matrix`, `_median_sky_from_phase2a_csv_cache`, `_phase2a_collect_session_jd_values`, `_ColorTermGroupFit`, `_draft_dir_from_phase2a_paths`, `_obs_group_filter_key`, `detect_outliers`, `apply_reporting_postprocess`, `empirical_feature_mask_mag`, `_preserve_nondetection_flags_helper`, `_target_row_is_vsx_known_variable`, `_mad_sigma_or_std_floor`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_phase2a_prepare_shared_state`
- `caller-in-other-module:_phase2a_process_one_target`
- `caller-in-other-module:finalize_hybrid_bkg_fallback_proc_dir`
- `caller-in-other-module:photometer_check_star_production_path`
- `caller-in-other-module:run_full_photometry_pipeline`
- `photometry_core.py:_coerce_bool_cell`
- `photometry_core.py:_normalize_gaia_id`
- `photometry_core.py:_phase2a_prepare_shared_state`
- `photometry_core.py:_phase2a_process_one_target`
- `photometry_core.py:_safe_polyfit`
- `photometry_core.py:_target_display_name`
- `photometry_core.py:build_gs11_summary`
- `photometry_core.py:merge_photometry_pipeline_meta`
- `pipeline.py:find_qc_metrics_csv`
- `pipeline.py:write_photometry_plan_files`

Who imports it (external files that call these defs today):
- `dev/scripts/archive/draft_runs/_gs11_revalidate_draft342.py`
- `dev/scripts/archive/draft_runs/_gs11_validate_step_b.py`
- `dev/scripts/archive/verify/_alg2_savgol_verify_draft321.py`
- `dev/scripts/archive/verify/_alg3_tempbin_verify_draft321.py`
- `dev/scripts/archive/verify/_alg4_democratic_verify_draft321.py`
- `dev/scripts/archive/verify/_alg5_pytics_verify_draft321.py`
- `dev/scripts/archive/verify/_phase2a_only_draft307.py`
- `dev/scripts/archive/verify/_reexport_phase2a_draft307.py`
- `dev/scripts/archive/verify/_smoke_phase2a_draft298.py`
- `dev/scripts/archive/verify/_smoke_run_draft287.py`
- `dev/scripts/archive/verify/_todo44_verify_draft321.py`
- `dev/scripts/archive/verify/_todo8_epsf_final_verify_draft321.py`
- `dev/scripts/archive/verify/_todo8_epsf_verify_draft321.py`
- `dev/scripts/audit_stage3_part1_measure.py`
- `dev/scripts/backfill_check_kmag_sidecars.py`
- `dev/scripts/bin4_sigma_forensics.py`
- `dev/scripts/bingain_acceptance_run.py`
- `dev/scripts/bingain_err_decompose.py`
- `dev/scripts/chi2_sigma_gate.py`
- `dev/scripts/chiandh_allfilters_overnight.py`
- `dev/scripts/chiandh_ct_dump_bvr_375.py`
- `dev/scripts/comp_pool_r_forensic.py`
- `dev/scripts/ct_bgr_summary.py`
- `dev/scripts/force_aperture_r_px.py`
- `dev/scripts/forced_photometry_grouper_adaptive_364.py`
- `dev/scripts/labbe_det_phase2a_double_run.py`
- `dev/scripts/m67_continue368_gr_merge_sandbox.py`
- `dev/scripts/m67_continue368_gr_sandbox.py`
- `dev/scripts/m67_ct_gate_diagnosis.py`
- `dev/scripts/reexport_draft_aavso.py`

### `photometry_phase2a_2.py` (46 defs, 2584 lines)

Source files: photometry_core.py

`save_lightcurve_csv`, `ensemble_normalize`, `save_target_field_map_png`, `temporal_bin_comp_lc`, `save_lightcurve_png`, `savgol_detrend_lc`, `pytics_iterative_weights`, `_recompute_bjd_hjd_with_status`, `apply_color_term`, `save_cutout_png`, `_combine_err_with_ensemble_scatter_keyed`, `_color_term_cat_inst_scatter_pair`, `_check_color_term_extrapolation`, `_load_blend_worklist`, `compute_mag_calib_final`, `_err_budget_components_keyed`, `compute_lc_rms_ooe`, `_ensemble_scatter_by_source_file`, `_phase2a_skip_empty_comps_target`, `_phase2a_empty_comp_summary_row`, `_route_lc_per_frame_err`, `_ac_summary_fields`, `_append_ct_prototype_row`, `_exclude_err_scatter_unmatched_epochs`, `ct_ensemble_reference_maps`, `_get_comp_bjd_series`, `BlendMapEntry`, `_load_adaptive_blend_map`, `_ct_prototype_enabled`, `decide_target_saturation_policy`, `apply_per_frame_saturation_to_active_targets`, `_per_frame_sat_flags_for_catalog_id`, `compute_lc_flux_method`, `evaluate_cog_night_apcorr_gate`, `_resolve_pfs_peak_test`, `pfs_rescue_eligible`, `_frame_has_usable_cog`, `_coerce_bool_cell`, `_keep_recorded_skip_reason`, `check_comparison_stability`, `_common_mode_detrend_comp_lc`, `_comp_lc_frame_ensemble_residual`, `run_sysrem_field`, `_build_phase2a_resolved_facts`, `_fits_header_facts`, `_get_lc`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_get_lc_adaptive`
- `caller-in-other-module:_get_lc_psf_strict`
- `caller-in-other-module:_get_lc_star_method`
- `caller-in-other-module:_phase2a_prepare_shared_state`
- `caller-in-other-module:_phase2a_process_one_target`
- `caller-in-other-module:_recompute_bjd_hjd_per_target`
- `caller-in-other-module:read_flux_from_csv`
- `caller-in-other-module:run_full_photometry_pipeline`
- `photometry_core.py:_normalize_gaia_id`
- `pipeline.py:inv_sat_limit_peak_test_adu`

Who imports it (external files that call these defs today):
- `dev/scripts/backfill_check_kmag_sidecars.py`
- `dev/scripts/bin4_sigma_forensics.py`
- `dev/scripts/chi2_sigma_gate.py`
- `dev/scripts/chiandh_allfilters_overnight.py`
- `dev/scripts/chiandh_ct_dump_bvr_375.py`
- `dev/scripts/comp_pool_r_forensic.py`
- `dev/scripts/ct_bgr_summary.py`
- `dev/scripts/epsf_ac_01.py`
- `dev/scripts/epsf_pin_census_01.py`
- `dev/scripts/k2_cohort_run.py`
- `dev/scripts/m67_ct_gate_diagnosis.py`
- `dev/scripts/select_constant_calibrators.py`
- `dev/scripts/sigma_floor_attribution.py`
- `dev/scripts/sigma_newton_run.py`
- `dev/scripts/sigma_prov_forensic.py`
- `dev/scripts/sigma_sem_cause.py`
- `dev/scripts/smoke_psf_gated_364.py`
- `dev/scripts/verify_adaptive_faint_targets_364.py`
- `dev/scripts/verify_adaptive_wiring_364.py`
- `dev/scripts/verify_method_report_separation.py`
- `dev/tests/test_alg_functions.py`
- `dev/tests/test_apcorr_mixedframe_night_gate.py`
- `dev/tests/test_batch_d_audit_closure.py`
- `dev/tests/test_comp_quality_json.py`
- `dev/tests/test_comp_stability.py`
- `dev/tests/test_ensemble_normalize_no_zp_clip.py`
- `dev/tests/test_except_fix_top10.py`
- `dev/tests/test_field_map_no_catalog_only.py`
- `dev/tests/test_forced_phot_and_weights.py`
- `dev/tests/test_g1_f003_alignment_pixel_fallback.py`

### `photometry_epsf_hooks.py` (2 defs, 198 lines)

Source files: photometry_core.py

`load_epsf_metrics_for_draft`, `_annulus_sky_subtracted_flux`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:measure_empty_aperture_sigma_bkg`
- `photometry_core.py:_sky_pp_from_annulus_image`

Who imports it (external files that call these defs today):
- `dev/scripts/forced_photometry_fixed_vs_free.py`
- `dev/tests/test_except_fix_top10.py`
- `dev/tests/validation/a9_core.py`
- `dev/tests/validation/recover.py`
- `dev/tests/validation/v3d_fine_scale.py`
- `dev/tools/pre_impl_01_measure.py`
- `src_py/photometry_core.py`
- `src_py/photometry_report.py`
- `src_py/psf_neighbor_sub.py`
- `src_py/psf_photometry.py`
- `src_py/ui_epsf_dashboard.py`

### `pipeline_epsf_hooks.py` (2 defs, 50 lines)

Source files: pipeline.py

`_epsf_lc_catalog_ids`, `_add_catalog_ids_from_csv`

Imports it will need (same-file/cross-file callees outside this module):

Who imports it (external files that call these defs today):
- `dev/scripts/verify_adaptive_faint_targets_364.py`
- `dev/tests/test_epsf_science_set.py`

### `photometry_exports.py` (5 defs, 94 lines)

Source files: photometry_core.py

`ensemble_member_ids`, `apply_comp_w_rel_for_display`, `_get_lc_psf_strict`, `lc_has_finite_airmass`, `_get_lc_adaptive_per_star`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_get_lc_star_method`
- `photometry_core.py:_coerce_bool_cell`
- `photometry_core.py:_get_lc_star_method`
- `photometry_core.py:_resolve_star_flux_method`
- `photometry_core.py:comp_quality_quality_strings`

Who imports it (external files that call these defs today):
- `dev/scripts/backfill_check_kmag_sidecars.py`
- `dev/scripts/select_constant_calibrators.py`
- `dev/tests/test_get_lc_psf_strict.py`
- `dev/tools/wide_error_budget_diag.py`
- `src_py/check_star_kmag.py`
- `src_py/export_reports.py`
- `src_py/method_lc_output.py`
- `src_py/photometry_report.py`

### `photometry_ui_helpers.py` (2 defs, 24 lines)

Source files: photometry_core.py

`resolve_lc_time_base`, `lc_time_axis_short_label`

Imports it will need (same-file/cross-file callees outside this module):

Who imports it (external files that call these defs today):
- `src_py/export_reports.py`
- `src_py/photometry_report.py`
- `src_py/ui_aperture_photometry.py`

### `pipeline_ui_helpers.py` (5 defs, 216 lines)

Source files: pipeline.py

`run_quality_analysis`, `list_best_processed_light_paths_for_masterstar`, `_resolve_light_fits_for_quality_inspection`, `resolve_masterstars_metadata_csv`, `preprocess_sky_summary_from_df`

Imports it will need (same-file/cross-file callees outside this module):
- `pipeline.py:_archive_raw_to_calibrated_light`
- `pipeline.py:_estimate_fov_deg_from_fits_path`
- `pipeline.py:_obs_fwhm_basename_map_from_db`
- `pipeline.py:_path_segments_forbidden_for_masterstar_physical_source`
- `pipeline.py:_quality_inspection_dao_metrics`
- `pipeline.py:_sort_masterstar_paths_by_fwhm`
- `pipeline.py:_vyvar_open_database`
- `pipeline.py:draft_median_pointing_icrs_deg`
- `pipeline.py:resolve_masterstar_input_root`
- `pipeline.py:sync_obs_files_drift_arcmin_for_draft`

Who imports it (external files that call these defs today):
- `dev/tests/test_skysf_double_guard.py`
- `src_py/app.py`
- `src_py/photometry_report.py`
- `src_py/ui_aperture_photometry.py`
- `src_py/ui_components.py`
- `src_py/ui_quality_dashboard.py`

### `photometry_gate_helpers.py` (25 defs, 797 lines)

Source files: photometry_core.py

`measure_empty_aperture_sigma_bkg`, `measure_growth_curve_ee`, `photometer_check_star_production_path`, `_aperture_flux_sky_per_star`, `_compute_fov_max_dist`, `_phase2a_star_mag_lookup`, `_assert_inv_err_sigma_acct_01`, `_median_bkg_var_from_aligned_frames`, `estimate_star_free_per_pixel_variance_adu2`, `discover_aligned_science_fits`, `_recompute_bjd_hjd_per_target`, `_estimate_annulus_sky_pp`, `_labbe_content_seed_from_header`, `bkg_scale_ratio_empirical_over_howell`, `_howell_bkg_variance_adu2`, `scaled_sigma_bkg_ap_from_howell`, `comp_quality_quality_strings`, `_frame_quality_gate_select`, `_resolve_star_flux_method`, `_sigma_bkg_r_key`, `_clamp_err_empty_apertures_n`, `compute_setup_bkg_scale_r`, `_normalize_err_background_mode`, `_sky_pp_from_annulus_image`, `_clamp_bkg_scale_r`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_annulus_sky_subtracted_flux`
- `caller-in-other-module:_aperture_flux_sky_batch`
- `caller-in-other-module:_get_lc_adaptive_per_star`
- `caller-in-other-module:apply_comp_w_rel_for_display`
- `caller-in-other-module:enhance_catalog_dataframe_aperture_bpm`
- `caller-in-other-module:finalize_hybrid_bkg_fallback_proc_dir`
- `caller-in-other-module:run_full_photometry_pipeline`
- `photometry_core.py:_Phase2AState`
- `photometry_core.py:_annulus_sky_subtracted_flux`
- `photometry_core.py:_build_star_exclusion_mask`
- `photometry_core.py:_canonicalize_star_xy`
- `photometry_core.py:_clamp_err_empty_apertures_min`
- `photometry_core.py:_labbe_append_debug_record`
- `photometry_core.py:_normalize_gaia_id`
- `photometry_core.py:_phase2a_process_one_target`
- `photometry_core.py:_recompute_bjd_hjd_with_status`
- `photometry_core.py:_robust_scatter_mad`
- `photometry_core.py:compute_per_frame_cog_correction`

Who imports it (external files that call these defs today):
- `dev/scripts/archive/verify/_todo44_verify_draft321.py`
- `dev/scripts/audit_stage3_part1b_check_chi2.py`
- `dev/scripts/audit_stage3_part1c_robust_chi2.py`
- `dev/scripts/bingain_err_decompose.py`
- `dev/scripts/bingain_patch_sigma_bkg.py`
- `dev/scripts/m67_continue368_gr_merge_sandbox.py`
- `dev/scripts/sky_gradient_sky_plane_361_362.py`
- `dev/tests/test_comp_quality_json.py`
- `dev/tests/test_err_background_empirical.py`
- `dev/tests/test_f431_labbe_provenance.py`
- `dev/tests/test_frame_quality_gate.py`
- `dev/tests/test_geo.py`
- `dev/tests/test_labbe_det_determinism.py`
- `dev/tests/test_psf_lc_routing.py`
- `dev/tests/test_time_base_flag.py`
- `dev/tools/cog_draft512_measure.py`
- `dev/tools/docs_pdf/flow_doc_facts.py`
- `dev/tools/impl_01_measure.py`
- `dev/tools/impl_02_part_a_bkg.py`
- `dev/tools/impl_02_rebuild_table.py`
- `dev/tools/q1_xval_matched_run.py`
- `dev/tools/sky_clip_510_impact.py`
- `dev/tools/wide_err_a2b.py`
- `dev/tools/wide_err_e3.py`
- `dev/tools/wide_err_step0_checkstar.py`
- `dev/tools/wide_err_w1w2.py`
- `dev/tools/wide_error_budget_diag.py`

### `pipeline_gate_helpers.py` (1 defs, 146 lines)

Source files: pipeline.py

`validate_comparison_ensemble_flatness`

Imports it will need (same-file/cross-file callees outside this module):
- `pipeline.py:extract_fits_metadata`

Who imports it (external files that call these defs today):
- `dev/tests/test_except_fix2_top10.py`

### `photometry_dead.py` (15 defs, 448 lines)

Source files: photometry_core.py

`compute_per_frame_cog_correction`, `_aperture_flux_sky_batch`, `_build_star_exclusion_mask`, `_canonicalize_star_xy`, `_median_bkg_var_adu2_per_px_from_proc_cache`, `_finite_pixel_bbox_from_array`, `_intersection_bbox_from_frame_bboxes`, `_star_mag_for_aperture_sizing`, `_robust_scatter_mad`, `_labbe_append_debug_record`, `_labbe_debug_dump_enabled`, `_get_lc_star_method`, `_labbe_debug_dump_path`, `_clamp_err_empty_apertures_min`, `_is_broadband_photometric_filter`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:_get_lc_adaptive_per_star`
- `caller-in-other-module:common_field_intersection_bbox_px`
- `caller-in-other-module:common_field_intersection_bbox_px_from_arrays`
- `caller-in-other-module:enhance_catalog_dataframe_aperture_bpm`
- `caller-in-other-module:estimate_star_free_per_pixel_variance_adu2`
- `caller-in-other-module:measure_empty_aperture_sigma_bkg`
- `caller-in-other-module:measure_growth_curve_ee`
- `caller-in-other-module:stamp_masterstar_snr_columns`
- `photometry_core.py:_aperture_flux_sky_per_star`
- `photometry_core.py:_get_lc`
- `photometry_core.py:_get_lc_psf_strict`
- `photometry_core.py:_sky_pp_from_annulus_image`

Who imports it (external files that call these defs today):

### `pipeline_dead.py` (12 defs, 454 lines)

Source files: pipeline.py

`_solve_wcs_solve_field_cli`, `get_auto_fov`, `analyze_calibrated_qc`, `_solve_wcs_astrometry_net`, `_saturated_core_plateau`, `_star_saturation_flags`, `_analyze_calibrated_qc_one`, `_obs_fwhm_basename_map_from_db`, `_quality_inspection_dao_metrics`, `_frame_gain_readnoise_for_error_map`, `_estimate_fov_deg_from_fits_path`, `_per_frame_noise_error_map`

Imports it will need (same-file/cross-file callees outside this module):
- `caller-in-other-module:AstroPipeline`
- `caller-in-other-module:list_best_processed_light_paths_for_masterstar`
- `caller-in-other-module:run_quality_analysis`
- `pipeline.py:_apply_wcs_header_to_fits`
- `pipeline.py:_box_peak_max_adu`
- `pipeline.py:_estimate_fov_deg_from_header`
- `pipeline.py:_filter_light_paths_maybe`
- `pipeline.py:_iter_light_fits`
- `pipeline.py:_qc_fwhm_elongation`
- `pipeline.py:_quality_inspection_dao_metrics_array`
- `pipeline.py:_safe_filter_token`
- `pipeline.py:_vyvar_parallel_pool`
- `pipeline.py:_vyvar_qc_preprocess_workers`
- `pipeline.py:estimate_memory_from_fits_headers`
- `pipeline.py:format_memory_bytes`
- `pipeline.py:resolve_plate_solve_fov_deg_hint`

Who imports it (external files that call these defs today):

## photometry_core phase2a vs comp-selection (do not auto-merge)

- phase0+1 symbols: 30, lines 4095
- phase2a symbols: 111, lines 9130
- shared (both stages): 32, lines 1936
- directed call weight comp->phase2a: 0
- directed call weight phase2a->comp: 0
- directed call weight comp->shared: 11 / shared->comp: 3
- directed call weight phase2a->shared: 27 / shared->phase2a: 4

These two stay separate proposed modules even if they call each other.
Shared names go to `photometry_shared.py` in the table if that module exists.

## Facades

- `pipeline.py` re-exports every name moved out of it.
- `photometry_core.py` re-exports every name moved out of it (and `photometry.py` star-import stays).
- Spawn MP workers listed in the risk register must remain importable as `pipeline.<name>`.

## Architect notes (measure, not a task list)

Direct phase0+1 <-> phase2a call weight is **0**. They only meet in `photometry_shared.py`
(comp->shared 11, phase2a->shared 27). Do not merge those two stages because they share helpers.

Graph LPA found **no** remaining weak cluster (cross > internal after merge). The 50 LPA
communities are not the module cut: stage boundaries win. LPA is used only to pack leftovers
after peeling defs >= 800 lines.

`pipeline_calibrate.py` is 4121 lines (YES over ~4000). Least-bad: keep one module. The class
`AstroPipeline` (507 lines) can stay in the `pipeline.py` facade instead if E1 wants the
calibrate module under 4000.

Giant single-def modules cannot go under 4000 without splitting the function body
(`generate_masterstar_and_catalog` 2540, `_phase2a_process_one_target` 1611, ...). Marked.

`pipeline_dead.py` / `photometry_dead.py` are static-unreachable from the night-run seeds
(legacy solve-field/astrometry.net helpers, QC helpers only called from UI methods). Confirm
before deleting in a later task; this task moves nothing.

E-final (facade removal) is a separate decision. Suggested extraction order for E1..En:
small isolated modules first (import, ePSF hooks, exports), then shared photometry, then
comp, then phase2a (helpers before the two giant defs), then calibrate (MP spawn risk),
then astrometry leftovers, giant astrometry defs last. G-EPSF when the ePSF graph is touched.

