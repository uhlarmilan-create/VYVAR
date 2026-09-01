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
)

PHOTOMETRY_E1: tuple[str, ...] = (
    "resolve_lc_time_base",
    "lc_time_axis_short_label",
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
