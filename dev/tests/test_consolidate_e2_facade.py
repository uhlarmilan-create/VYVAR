"""CONSOLIDATE-01E2: moved defs remain reachable through the facade."""

from __future__ import annotations

import photometry
import photometry_core
import photometry_provenance


PHOTOMETRY_E2_PROVENANCE: tuple[str, ...] = (
    "_resolve_git_provenance",
    "_build_pipeline_provenance_block",
    "classify_git_dirty_paths",
    "_porcelain_status_by_path",
    "_is_import_relevant_py_path",
    "_complete_config_snapshot",
    "_json_safe_snapshot_value",
    "merge_photometry_pipeline_meta",
)


def test_e2_provenance_facade_getattr() -> None:
    for name in PHOTOMETRY_E2_PROVENANCE:
        obj = getattr(photometry_core, name)
        assert callable(obj), name
        assert obj.__module__ == "photometry_provenance", name


def test_e2_merge_in_star_import_not_required() -> None:
    """merge is not in __all__; facade getattr is the contract."""
    assert hasattr(photometry_core, "merge_photometry_pipeline_meta")
    assert photometry_core.merge_photometry_pipeline_meta is (
        photometry_provenance.merge_photometry_pipeline_meta
    )


def test_e2_resolve_git_follow_proxy() -> None:
    """Facade re-exports the real def; test_f431 patches photometry_provenance."""
    assert photometry_core._resolve_git_provenance is photometry_provenance._resolve_git_provenance


PHOTOMETRY_E2_SHARED: tuple[str, ...] = (
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

# C-D: production entry stays in the facade. E4 moved run_phase2a and
# measure_fwhm_from_masterstar to photometry_phase2a.py.
PHOTOMETRY_E2_STAY: tuple[str, ...] = (
    "run_full_photometry_pipeline",
)


def test_e2_shared_facade_getattr() -> None:
    import photometry_shared as ps

    for name in PHOTOMETRY_E2_SHARED:
        obj = getattr(photometry_core, name)
        assert getattr(ps, name) is obj, name
        if name == "StressTestResult":
            assert obj.__module__ == "photometry_shared"
        else:
            assert callable(obj), name
            assert obj.__module__ == "photometry_shared", name


def test_e2_c_d_entries_stay_in_facade() -> None:
    for name in PHOTOMETRY_E2_STAY:
        obj = getattr(photometry_core, name)
        assert callable(obj), name
        assert obj.__module__ == "photometry_core", name


def test_e2_star_import_binds_shared_all_names() -> None:
    for name in (
        "StressTestResult",
        "_get_lc_adaptive",
        "common_field_intersection_bbox_px",
        "enhance_catalog_dataframe_aperture_bpm",
        "recommended_aperture_by_color",
        "vsx_is_known_variable_top3_per_bin",
        "run_full_photometry_pipeline",
    ):
        assert hasattr(photometry, name), name
        assert getattr(photometry, name) is getattr(photometry_core, name)
