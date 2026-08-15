"""IMPL-03 unit tests: ZP calibration + scatter curve helpers."""
from __future__ import annotations

import math

import numpy as np

from aperture_scatter_select import (
    calibrate_snr_zero_point_from_fluxes,
    classify_scatter_curve_shape,
    flat_aperture_table_from_radius,
    split_selection_holdout,
)
from photometry_core import compute_snr_optimal_aperture_table


def test_zp_calibration_recovers_injected_zp():
    rng = np.random.default_rng(3)
    zp_true = 22.5
    mags = np.linspace(8.0, 14.0, 40)
    ftot = 10.0 ** ((zp_true - mags) / 2.5)
    # Aperture at EE=0.7
    flux = ftot * 0.7 * (1.0 + 0.01 * rng.normal(size=mags.size))
    r = np.full_like(mags, 3.0)
    ee_r = np.array([1.0, 3.0, 8.0])
    ee_c = np.array([0.3, 0.7, 1.0])
    cal = calibrate_snr_zero_point_from_fluxes(mags, flux, r, ee_radii=ee_r, ee_curve=ee_c)
    assert cal["ok"]
    assert abs(float(cal["zero_point"]) - zp_true) < 0.15


def test_snr_table_bright_opt_shrinks_with_calibrated_zp():
    """IMPL-03 Item 3: ZP=25 inflates bright r_opt vs draft-calibrated ZP."""
    fwhm = 5.195
    radii = np.arange(0.5, 14.0, 0.5)
    # Rough measured EE shape
    ee = 1.0 - np.exp(-((radii / 3.5) ** 1.2))
    ee = ee / ee[-1]
    common = dict(
        fwhm_px=fwhm,
        sky_adu_per_px=1919.0,
        gain=3.17,
        read_noise=7.6,
        bkg_var_adu2_per_px=1873.0,
        ee_radii=radii,
        ee_curve=ee,
        ee_source="measured_growth_curve",
    )
    hi = compute_snr_optimal_aperture_table(**common, zero_point=25.0)
    lo = compute_snr_optimal_aperture_table(**common, zero_point=22.5)
    assert float(hi["table"][8.0]) > float(lo["table"][8.0])
    assert abs(float(hi["zero_point"]) - 25.0) < 1e-9
    assert abs(float(lo["zero_point"]) - 22.5) < 1e-9


def test_split_selection_holdout_disjoint():
    ids = [f"s{i}" for i in range(20)]
    a, b = split_selection_holdout(ids, seed=1)
    assert set(a).isdisjoint(set(b))
    assert len(a) + len(b) == 20


def test_classify_flat_min():
    r = np.arange(2.0, 10.0, 0.5)
    s = np.full(r.shape, 20.0)
    s[4:10] = 19.5  # broad flat valley
    assert classify_scatter_curve_shape(r, s) == "flat_min"


def test_flat_aperture_table_scatter_mode():
    tab = flat_aperture_table_from_radius(6.5, fwhm_px=5.0)
    assert tab["selection_criterion"] == "scatter"
    assert tab["fixed_radius_px"] == 6.5
    assert math.isclose(float(tab["table"][10.0]), 6.5)
