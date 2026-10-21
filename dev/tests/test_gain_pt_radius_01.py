"""GAIN-PT-RADIUS-01: the photon-transfer radius is pinned (sky-dominated r = 4 px)."""
from __future__ import annotations

from gain_photon_transfer import PHOTON_TRANSFER_APERTURE_R_PX, PHOTON_TRANSFER_APERTURE_SOURCE


def test_pt_radius_is_pinned() -> None:
    assert PHOTON_TRANSFER_APERTURE_R_PX == 4.0
    assert PHOTON_TRANSFER_APERTURE_SOURCE == "pinned_sky_dominated_4px"
