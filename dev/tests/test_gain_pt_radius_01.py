"""GAIN-PT-RADIUS-01: leftover dynamic_params must not set PT radius."""
from __future__ import annotations

from gain_photon_transfer import (
    PHOTON_TRANSFER_APERTURE_R_PX,
    PHOTON_TRANSFER_APERTURE_SOURCE,
    legacy_pt_aperture_from_leftover_dynamic_params,
    resolve_photon_transfer_aperture_r_px,
)


def test_legacy_hole_leftover_meta_overrides_pin() -> None:
    """Fire-proof (a): pre-fix behavior - leftover aperture_r_px=2.499 wins."""
    leftover = {"aperture_r_px": 2.499, "fwhm_px": 3.3, "gain": 0.7925}
    assert legacy_pt_aperture_from_leftover_dynamic_params(leftover) == 2.499
    assert (
        legacy_pt_aperture_from_leftover_dynamic_params(
            {"aperture_r_px": 2.499},
            default_r_px=4.0,
        )
        == 2.499
    )


def test_resolve_pt_radius_ignores_leftover_meta() -> None:
    """Fire-proof (b): pinned 4.0 regardless of leftover / CV-like meta."""
    leftover = {"aperture_r_px": 2.499, "fwhm_px": 3.3, "gain": 0.7925}
    r, src = resolve_photon_transfer_aperture_r_px(leftover)
    assert r == PHOTON_TRANSFER_APERTURE_R_PX
    assert r == 4.0
    assert src == PHOTON_TRANSFER_APERTURE_SOURCE
    assert src == "pinned_sky_dominated_4px"


def test_resolve_pt_radius_cv_like_leftover_fixture() -> None:
    """CV-like leftover meta (small scatter radius) must not pull PT off 4.0."""
    cv_like = {
        "aperture_r_px": 1.999,
        "fwhm_px": 3.3014,
        "gain": 0.7925,
        "density_class": "dense",
    }
    assert legacy_pt_aperture_from_leftover_dynamic_params(cv_like) == 1.999
    r, src = resolve_photon_transfer_aperture_r_px(cv_like)
    assert r == 4.0
    assert src == "pinned_sky_dominated_4px"


def test_resolve_pt_radius_none_meta() -> None:
    r, src = resolve_photon_transfer_aperture_r_px(None)
    assert r == 4.0
    assert src == "pinned_sky_dominated_4px"
