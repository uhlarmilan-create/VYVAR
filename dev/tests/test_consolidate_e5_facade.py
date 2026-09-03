"""CONSOLIDATE-01E5: moved defs remain reachable through the pipeline facade."""

from __future__ import annotations

import pipeline
import pipeline_calibrate


PIPELINE_E5: tuple[str, ...] = (
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


def test_e5_pipeline_facade_getattr() -> None:
    for name in PIPELINE_E5:
        obj = getattr(pipeline, name)
        home = getattr(pipeline_calibrate, name)
        assert obj is home, name
        if name == "_CALIB_MASTER_NB_UNSET":
            assert obj is pipeline_calibrate._CALIB_MASTER_NB_UNSET
        else:
            assert obj.__module__ == "pipeline_calibrate", name


def test_e5_astropipeline_stays() -> None:
    assert pipeline.AstroPipeline.__module__ == "pipeline"


def test_e5_init_arity_three() -> None:
    import inspect

    sig = inspect.signature(pipeline._init_calibrate_batch_worker)
    assert list(sig.parameters) == ["_md_s", "native_b", "cal_diag_blob"]


def test_e5_sky_surface_identity() -> None:
    assert (
        pipeline.SkySurfaceOrderConflictError
        is pipeline_calibrate.SkySurfaceOrderConflictError
    )
