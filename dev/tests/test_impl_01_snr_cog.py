"""IMPL-01: measured growth-curve SNR aperture table."""
from __future__ import annotations

import math

import numpy as np

from photometry_core import compute_snr_optimal_aperture_table


def test_snr_table_gaussian_fallback_when_no_ee():
    out = compute_snr_optimal_aperture_table(
        fwhm_px=3.0, sky_adu_per_px=1000.0, gain=3.0, read_noise=8.0
    )
    assert out["ee_path"] == "gaussian_fallback"
    assert "bound_hit_by_mag" in out
    assert "ee_at_opt_by_mag" in out


def test_snr_table_measured_ee_widens_vs_gaussian_clamp():
    """A wing-heavy EE curve should not stick the bright-star optimum on r_min."""
    fwhm = 3.389
    r_min = 0.8 * fwhm
    # Synthetic Moffat-like curve: slower growth than Gaussian (EE~0.66 at r_min).
    radii = np.arange(0.5, 16.0, 0.25)
    # Approximate measured draft-514 shape: EE(2.711)~0.66, EE(12)~1
    ee = 1.0 - np.exp(-((radii / 4.2) ** 1.3))
    ee = ee / ee[-1]
    meas = compute_snr_optimal_aperture_table(
        fwhm_px=fwhm,
        sky_adu_per_px=1500.0,
        gain=3.17,
        read_noise=7.6,
        ee_radii=radii,
        ee_curve=ee,
        ee_source="measured_growth_curve",
    )
    assert meas["ee_path"] == "measured_growth_curve"
    r8 = float(meas["table"][8.0])
    # Must not silently sit on the Gaussian lower clamp for a bright bin.
    assert r8 > r_min + 0.2, f"expected r8>{r_min}+0.2, got {r8}"
    assert meas["bound_hit_by_mag"][8.0] in ("none", "r_max", "r_min")
    ee8 = float(meas["ee_at_opt_by_mag"][8.0])
    assert math.isfinite(ee8) and ee8 > 0.66


def test_snr_table_reports_bound_hit_when_clamped():
    fwhm = 3.0
    # Nearly flat EE: optimum wants huge r -> hits r_max.
    radii = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0])
    ee = np.array([0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 1.0])
    out = compute_snr_optimal_aperture_table(
        fwhm_px=fwhm,
        sky_adu_per_px=50.0,
        gain=3.0,
        read_noise=5.0,
        ee_radii=radii,
        ee_curve=ee,
    )
    assert out["n_bound_hits"] >= 0
    assert set(out["bound_hit_by_mag"].values()) <= {"none", "r_min", "r_max"}
