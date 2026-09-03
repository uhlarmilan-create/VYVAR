"""CONSOLIDATE-01 E-DEAD: moved defs remain reachable through the facade."""
from __future__ import annotations

import photometry_core
import photometry_exports
import photometry_gate_helpers
import photometry_shared
import pipeline
import pipeline_ui_helpers


PHOTOMETRY_EDEAD_SHARED: tuple[str, ...] = (
    "_finite_pixel_bbox_from_array",
    "_intersection_bbox_from_frame_bboxes",
    "_aperture_flux_sky_batch",
    "compute_per_frame_cog_correction",
)

PHOTOMETRY_EDEAD_GATE: tuple[str, ...] = (
    "_clamp_err_empty_apertures_min",
    "_robust_scatter_mad",
    "_build_star_exclusion_mask",
    "_canonicalize_star_xy",
    "_labbe_debug_dump_enabled",
    "_labbe_debug_dump_path",
    "_labbe_append_debug_record",
)


def test_edead_shared_facade_getattr() -> None:
    for name in PHOTOMETRY_EDEAD_SHARED:
        obj = getattr(photometry_core, name)
        home = getattr(photometry_shared, name)
        assert obj is home, name
        assert obj.__module__ == "photometry_shared", name
        assert callable(obj), name


def test_edead_exports_facade_getattr() -> None:
    obj = photometry_core._get_lc_star_method
    home = photometry_exports._get_lc_star_method
    assert obj is home
    assert obj.__module__ == "photometry_exports"
    assert callable(obj)


def test_edead_gate_facade_getattr() -> None:
    for name in PHOTOMETRY_EDEAD_GATE:
        obj = getattr(photometry_core, name)
        home = getattr(photometry_gate_helpers, name)
        assert obj is home, name
        assert obj.__module__ == "photometry_gate_helpers", name
        assert callable(obj), name


PIPELINE_EDEAD_UI: tuple[str, ...] = (
    "_quality_inspection_dao_metrics",
    "_estimate_fov_deg_from_fits_path",
    "_obs_fwhm_basename_map_from_db",
)


def test_edead_ui_facade_getattr() -> None:
    for name in PIPELINE_EDEAD_UI:
        obj = getattr(pipeline, name)
        home = getattr(pipeline_ui_helpers, name)
        assert obj is home, name
        assert obj.__module__ == "pipeline_ui_helpers", name
        assert callable(obj), name


def test_edead_qc_stays_in_pipeline() -> None:
    assert pipeline.analyze_calibrated_qc.__module__ == "pipeline"
    assert pipeline._analyze_calibrated_qc_one.__module__ == "pipeline"
    assert pipeline.AstroPipeline.__module__ == "pipeline"
