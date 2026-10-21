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
import phase01_run
import phase2a_state
import phase2a_target


# --- union name lists (from test_consolidate_e1..edead_facade) ---

PIPELINE_CALLABLE_E1 = (
    "extract_fits_metadata",
    "fits_metadata_from_primary_header",
    "generate_observation_hash",
    "log_lights_binning_from_headers_preflight",
    "_resolve_light_fits_for_quality_inspection",
    "resolve_masterstars_metadata_csv",
)

PHOTOMETRY_CALLABLE_E1 = (
    "resolve_lc_time_base",
    "_annulus_sky_subtracted_flux",
    "measure_empty_aperture_sigma_bkg",
    "_aperture_flux_sky_per_star",
    "_assert_inv_err_sigma_acct_01",
    "_labbe_content_seed_from_header",
    "bkg_scale_ratio_empirical_over_howell",
    "_howell_bkg_variance_adu2",
    "scaled_sigma_bkg_ap_from_howell",
    "comp_quality_quality_strings",
    "_sigma_bkg_r_key",
    "_clamp_err_empty_apertures_n",
    "compute_setup_bkg_scale_r",
    "_clamp_bkg_scale_r",
    "apply_comp_w_rel_for_display",
    "lc_has_finite_airmass",
)

PHOTOMETRY_E2_PROVENANCE = (
    "_resolve_git_provenance",
    "_build_pipeline_provenance_block",
    "classify_git_dirty_paths",
    "_is_import_relevant_py_path",
    "_complete_config_snapshot",
    "merge_photometry_pipeline_meta",
)

PHOTOMETRY_E2_SHARED = (
    "enhance_catalog_dataframe_aperture",
    "_get_plate_scale_from_cfg",
    "finalize_hybrid_bkg_fallback_proc_dir",
    "stamp_vsx_known_variable_on_masterstars",
    "stress_test_relative_rms_from_sidecars",
    "stamp_masterstar_snr_columns",
    "compute_fwhm_gaussian_for_aperture_catalog",
    "_read_plate_scale_from_fits_path",
    "vsx_is_known_variable_top3_per_bin",
    "build_gs11_summary",
    "_resolve_plate_scale_arcsec_per_px",
    "_fwhm_moment_at",
    "common_field_intersection_bbox_px",
    "recommended_aperture_by_color",
    "_safe_polyfit",
    "_target_display_name",
    "common_field_intersection_bbox_px_from_arrays",
    "_normalize_gaia_id",
    "StressTestResult",
)

PHOTOMETRY_E3_COMP = (
    "select_comparison_stars_per_target",
    "select_active_targets",
    "build_global_comp_pool",
    "_select_comps_by_rms_then_color",
    "_enrich_comp_bp_rp",
    "_resolve_frame_hw_px_from_masterstar",
    "ensure_full_variable_targets_if_presel_stub",
    "_warn_zero_compstars_edge",
    "_count_gate_passing_comps",
    "_dedupe_comp_pool_by_gaia_key",
    "_bprp_tier_ladder_for_selection",
    "_variable_targets_looks_like_ct_presel_stub",
    "_normalize_id_value",
    "_bool_col",
    "_normalize_id_series",
)

PHOTOMETRY_E4_PHASE2A = (
    "run_phase2a",
    "measure_fwhm_from_masterstar",
    "read_flux_from_csv",
    "_Phase2AState",
    "parse_comp_quality_json_map",
    "detect_outliers",
    "apply_reporting_postprocess",
    "fit_color_term_c1",
    "should_apply_color_term",
    "resolve_apply_color_term",

    "_photometric_error",
    "_sky_pp_for_photometric_error",
)

PHOTOMETRY_E4_LC = (
    "ensemble_normalize",
    "save_lightcurve_csv",
    "apply_color_term",
    "_recompute_bjd_hjd_with_status",
)

PHOTOMETRY_EDEAD_SHARED = (
)

PHOTOMETRY_EDEAD_GATE = (
)

PIPELINE_EDEAD_UI = (
)

PIPELINE_E5 = (
    "SkySurfaceOrderConflictError",
    "_cal_diag_export_for_workers",
    "_cal_diag_session_from_export",
    "_calibrate_batch_process_one",
    "_calibrate_one_light_disk",
    "_db_for_calibration_tasks",
    "_decide_preprocess_sky_action",
    "_effective_saturation_limit",
    "_filter_light_paths_maybe",
    "_fit_subtract_preprocess_sky_surface",
    "_has_valid_wcs",
    "_infer_sat_limit_from_bitpix",
    "_init_calibrate_batch_worker",
    "_iter_light_fits",
    "_light_binning_from_path",
    "_log_calibration_io_preflight",
    "_match_and_crop_pair",
    "_obs_group_key_from_light_path",
    "_pipeline_ui_error",
    "_qc_enrich_calibrated_in_place",
    "_qc_enrich_one_frame",
    "_qc_fwhm_elongation",
    "_qc_pack_from_config",
    "_saturate_limit_adu_from_header",
    "_saturation_adu_for_cal_diag",
    "_sync_obs_calibration_state_with_retry",
    "_vyvar_calibrate_multiprocessing_enabled",
    "_vyvar_parallel_pool",
    "_vyvar_qc_preprocess_workers",
    "apply_perf10_dao_qc_to_obs_files",
    "calibrate_lights_to_calibrated",
    "estimate_archive_memory_profile",
    "estimate_memory_from_fits_headers",
    "format_memory_bytes",
    "norm_fits_path_key",
    "run_draft_ram_calibration_qc_to_obs_files",
    "run_osc_channel_extraction_for_archive",
    "scan_calibrated_lights_pointing",
)

PIPELINE_E6A_PREPROCESS = (
    "_archive_raw_to_calibrated_light",
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
    "_dao_targeted_pass2_unmatched_gaia",
    "_equipment_saturate_adu_from_db",
    "_merge_vsx_exoplanet_variable_targets",
    "_pass2_sibling_wcs_recovery",
    "_path_segments_forbidden_for_masterstar_physical_source",
    "_plate_solve_input_bundle",
    "_query_vsx_local_frame_bbox",
    "_resolve_best_effort_path_under",
    "_vyvar_df_to_csv",
    "astrometry_align_and_build_masterstar",
    "draft_is_multi_group_obs",
    "draft_obs_group_count",
    "get_masterstar_candidates",
    "resolve_masterstar_input_root",
    "select_comparison_stars_spatial_grid",
    "write_photometry_plan_files",
)

PIPELINE_E6A_CATALOG = (
    "_EXPORT_PER_FRAME_WORKER_STATE",
    "_MASTERSTAR_ZONE_LOG_ONCE",
    "_annotate_masterstars_flux_zones",
    "_apply_dao_centroid_wcs_guard",
    "_box_peak_max_adu",
    "_build_exoplanet_promotion_rows_from_masterstars",
    "_cfg_from_export_worker_state",
    "_compute_airmass_from_altaz",
    "_dao_auto_binning_factor",
    "_dao_convolved_background_rms_adu",
    "_dao_detection_threshold_adu",
    "_dao_noise_sigma_adu",
    "_detect_empirical_clip_level_adu",
    "_effective_field_catalog_cone_radius_deg",
    "_exo_host_annotation_arrays",
    "_extract_airmass_from_header",
    "_finalize_hybrid_bkg_fallback_sidecar",
    "_init_export_per_frame_worker",
    "_lock_matched_centroids_to_master_grid",
    "_masterstar_zone_linear_threshold",
    "_mean_bin2d_for_dao",
    "_pixel_noise_sigma_pp_adu",
    "_prefilter_dao_table_brightest",
    "_proc_deduplicate_matched_catalog_rows",
    "_query_exoplanet_local",
    "_query_gaia_local",
    "_query_vsx_local",
    "_resolve_peak_saturation_limit_adu",
    "detect_stars_match_master_reference",
    "find_qc_metrics_csv",
    "inv_sat_limit_peak_test_adu",
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
    assert photometry_core._phase2a_prepare_shared_state is phase2a_state._phase2a_prepare_shared_state
    assert photometry_core._phase2a_process_one_target is phase2a_target._phase2a_process_one_target
    assert photometry_core.run_phase0_and_phase1 is phase01_run.run_phase0_and_phase1
    for name in PHOTOMETRY_CALLABLE_E1:
        assert callable(getattr(photometry_core, name)), name


def test_facade_inventory_giants_and_stays() -> None:
    assert pipeline.generate_masterstar_and_catalog is masterstar_build.generate_masterstar_and_catalog
    assert pipeline.detect_stars_and_match_catalog is catalog_match.detect_stars_and_match_catalog
    assert pipeline.export_per_frame_catalogs is frame_export.export_per_frame_catalogs
    assert pipeline.AstroPipeline.__module__ == "pipeline"
    assert pipeline.analyze_calibrated_qc.__module__ == "pipeline"
    assert pipeline._analyze_calibrated_qc_one.__module__ == "pipeline"
    assert photometry_core.run_full_photometry_pipeline.__module__ == "photometry_core"
    assert photometry_core.select_active_targets is photometry_comp.select_active_targets


def test_facade_inventory_star_import() -> None:
    for name in photometry_core.__all__:
        assert hasattr(photometry, name), name
    assert photometry.run_phase0_and_phase1 is photometry_core.run_phase0_and_phase1
    assert photometry.select_active_targets is photometry_core.select_active_targets


def test_facade_inventory_constants_leaf() -> None:
    assert pipeline.N_STARS_DIAG_K is pipeline_constants.N_STARS_DIAG_K
    assert pipeline.N_STARS_DIAG_K == 5.0
    cat_src = getattr(pipeline_catalog, "SAT_LIMIT_PEAK_TEST_SOURCE", None)
    assert cat_src is None or cat_src is pipeline_constants.SAT_LIMIT_PEAK_TEST_SOURCE


def test_facade_inventory_init_arities() -> None:
    sig = inspect.signature(pipeline._init_calibrate_batch_worker)
    assert list(sig.parameters) == ["native_b", "cal_diag_blob"]
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
