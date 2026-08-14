"""NOISE-FLOOR-01: flatness criterion and complete Howell terms."""
from __future__ import annotations

import math

import pandas as pd

from comp_pool_noise import (
    FLATNESS_CRITERION,
    FLATNESS_MIN_BINS,
    find_flat_scatter_range,
    predicted_sigma_mag_phot,
    predicted_sigma_mag_phot_complete,
)


def test_flatness_criterion_stated():
    assert "upper limit" in FLATNESS_CRITERION.lower() or "N-R0" in FLATNESS_CRITERION
    assert FLATNESS_MIN_BINS >= 3


def test_flatness_no_range_when_rising():
    rows = []
    for i, sc in enumerate([0.0074, 0.0070, 0.0093, 0.0100, 0.0114]):
        rows.append(
            {
                "mag_center": 8.25 + 0.5 * i,
                "n": 12,
                "scatter_median": sc,
                "usable": True,
            }
        )
    np_curve = pd.DataFrame(rows)
    flat = find_flat_scatter_range(np_curve)
    assert flat["flat"] is False
    assert flat["upper_limit_scatter"] == 0.0070
    assert "no_flat_range" in flat["result"]


def test_flatness_measured_when_three_flat_bins():
    rows = []
    for i, sc in enumerate([0.0070, 0.0071, 0.0072, 0.0090]):
        rows.append(
            {
                "mag_center": 8.25 + 0.5 * i,
                "n": 20,
                "scatter_median": sc,
                "usable": True,
            }
        )
    flat = find_flat_scatter_range(pd.DataFrame(rows))
    assert flat["flat"] is True
    assert flat["n_bins"] == 3


def test_complete_model_sky_factor_increases_sigma():
    kwargs = dict(flux_adu=5.0e4, sky_adu=1500.0, area_px=36.55, gain=3.17, read_noise_e=7.6)
    s0 = predicted_sigma_mag_phot(**kwargs)
    s1, terms = predicted_sigma_mag_phot_complete(
        **kwargs, n_sky_px=2000.0, dark_e_per_px=0.0, digitization_adu=1.0, correlated_pixel_factor=1.0
    )
    assert terms["sky_factor"] > 1.0
    assert s1 >= s0
    assert math.isclose(terms["sky_factor"], 1.0 + 36.55 / 2000.0, rel_tol=1e-6)


def test_complete_model_corr_factor_inflates_extended_not_source_alone():
    kwargs = dict(flux_adu=5.0e4, sky_adu=1500.0, area_px=36.55, gain=3.17, read_noise_e=7.6)
    s1, t1 = predicted_sigma_mag_phot_complete(**kwargs, correlated_pixel_factor=1.0)
    s2, t2 = predicted_sigma_mag_phot_complete(**kwargs, correlated_pixel_factor=1.5)
    assert t2["var_source"] == t1["var_source"]
    assert s2 > s1
