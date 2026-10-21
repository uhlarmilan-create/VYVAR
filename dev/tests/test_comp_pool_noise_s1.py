"""COMP-POOL-01 Stage 1: smoke tests for noise-curve helpers."""
from __future__ import annotations

import math

import numpy as np

from comp_pool_noise import (
    predicted_sigma_mag_phot,
    _robust_scatter_mag,
    instrumental_mag_from_flux,
)


def test_predicted_sigma_decreases_with_flux():
    s_faint = predicted_sigma_mag_phot(1.0e3, sky_adu=1500.0, area_px=50.0, gain=3.17, read_noise_e=7.6)
    s_bright = predicted_sigma_mag_phot(1.0e5, sky_adu=1500.0, area_px=50.0, gain=3.17, read_noise_e=7.6)
    assert math.isfinite(s_faint) and math.isfinite(s_bright)
    assert s_bright < s_faint


def test_robust_scatter_mad_on_gaussian():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 0.01, size=200)
    sc = _robust_scatter_mag(x)
    assert 0.005 < sc["mad_sigma"] < 0.02


def test_instrumental_mag_roundtrip_scale():
    m1 = instrumental_mag_from_flux(1000.0, zp=25.0)
    m2 = instrumental_mag_from_flux(10000.0, zp=25.0)
    assert abs((m1 - m2) - 2.5) < 1e-9
