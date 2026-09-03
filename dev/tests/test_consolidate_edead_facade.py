"""CONSOLIDATE-01 E-DEAD: moved defs remain reachable through the facade."""
from __future__ import annotations

import photometry_core
import photometry_gate_helpers
import photometry_shared


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


def test_edead_gate_facade_getattr() -> None:
    for name in PHOTOMETRY_EDEAD_GATE:
        obj = getattr(photometry_core, name)
        home = getattr(photometry_gate_helpers, name)
        assert obj is home, name
        assert obj.__module__ == "photometry_gate_helpers", name
        assert callable(obj), name
