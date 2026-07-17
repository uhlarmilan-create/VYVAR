"""Tests for wide_slope_noise_core pure helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wide_slope_noise_core import (
    analytic_slope_se,
    apply_affine_xy,
    attainable_flat_drift_slope,
    attainable_neighbor_slope,
    bootstrap_slope_se,
    brightness_tertile_slices,
    centroid_cutout_detector,
    contamination_fraction,
    excess_variance_by_tertile,
    fwhm_sensitivity,
    fwhm_to_sigma,
    gaussian_aperture_overlap,
    invert_affine_2x3,
    neighbor_sensitivity_mag_per_fwhm_px,
    p4_noise_consistency_check,
    sigma_slope_pt_mmag,
    slope_se_pair,
    star_drift_metrics,
    track_detector_positions,
)


def test_analytic_slope_se_bright_regime_hand_formula() -> None:
    """Bright tertile: err~0.015, N=139, SD(X)~0.2-0.3 -> SE ~ 0.004-0.006."""
    rng = np.random.default_rng(0)
    n = 139
    am = np.linspace(1.0, 1.6, n) + rng.normal(0, 0.02, n)
    err = np.full(n, 0.015)
    out = analytic_slope_se(am, err)
    assert out["n"] == n
    assert 0.17 <= out["sd_x"] <= 0.35
    assert out["median_err"] == pytest.approx(0.015)
    assert 0.003 <= out["se_analytic"] <= 0.008
    assert out["hand_formula_se"] == pytest.approx(out["se_analytic"], rel=0.15)


def test_analytic_slope_se_faint_regime_hand_formula() -> None:
    """Faint tertile: err~0.086, N=139, SD(X)~0.2-0.3 -> SE ~ 0.024-0.036."""
    rng = np.random.default_rng(1)
    n = 139
    am = np.linspace(1.0, 1.6, n) + rng.normal(0, 0.02, n)
    err = np.full(n, 0.086)
    out = analytic_slope_se(am, err)
    assert 0.020 <= out["se_analytic"] <= 0.045
    assert out["se_analytic"] > 0.010  # must exceed bright-regime SE


def test_brightness_tertile_slices_lower_mag_is_bright() -> None:
    mags = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    slices = brightness_tertile_slices(mags)
    bright = slices[0]
    faint = slices[2]
    assert bright[0] == "bright"
    assert faint[0] == "faint"
    bright_stars = [m for m in mags if bright[1] <= m < bright[2]]
    faint_stars = [m for m in mags if faint[1] <= m < faint[2]]
    assert max(bright_stars) < min(faint_stars)


def test_excess_variance_tertile_mag_ranges_correct_orientation() -> None:
    stars = []
    for i, mag in enumerate(np.linspace(9.0, 15.0, 30)):
        stars.append({"mag_g": mag, "b_X": 0.05 * ((-1) ** i), "se_use": 0.04})
    rows = excess_variance_by_tertile(stars, n_bootstrap=100, seed=2)
    by_label = {r["tertile"]: r for r in rows}
    assert by_label["bright"]["mag_max"] < by_label["faint"]["mag_min"]


def test_excess_variance_noise_dominated_synthetic() -> None:
    stars = [{"mag_g": 10 + i * 0.1, "b_X": 0.01 * ((-1) ** i), "se_use": 0.05} for i in range(30)]
    rows = excess_variance_by_tertile(stars, n_bootstrap=200, seed=2)
    assert rows
    assert all(r["excess_variance"] < 0.01 for r in rows)


def test_bootstrap_slope_se_hand_line() -> None:
    rng = np.random.default_rng(0)
    am = np.linspace(1.0, 1.4, 40)
    err = np.full(40, 0.01)
    mags = 0.05 * (am - am.mean()) + rng.normal(0, 0.002, 40)
    se = bootstrap_slope_se(mags, am, err, n_draws=500, seed=1)
    assert math.isfinite(se)
    assert se < 0.02


def test_star_drift_metrics_positive_corr() -> None:
    am = np.linspace(1.0, 1.5, 20)
    x = 100 + 50 * (am - am.min()) / (am.max() - am.min())
    y = 200 + 20 * (am - am.min()) / (am.max() - am.min())
    m = star_drift_metrics(x, y, am)
    assert m["drift_span_px"] > 40
    assert m["drift_x_corr"] > 0.9


def test_fwhm_sensitivity_known_partial() -> None:
    am = np.linspace(1.0, 1.3, 30)
    fwhm = 3.0 + 0.1 * am
    mags = 0.2 * fwhm + 0.01 * am
    out = fwhm_sensitivity(mags, fwhm, am)
    assert out["fwhm_sens"] == pytest.approx(0.2, abs=0.05)


def test_attainable_flat_drift_slope() -> None:
    v = attainable_flat_drift_slope(100.0, 0.8, eps_flat=0.01)
    assert v == pytest.approx(0.8, abs=1e-9)


def test_slope_se_pair_uses_analytic_not_wls_residual() -> None:
    am = np.linspace(1.0, 1.5, 25)
    err = np.full(25, 0.015)
    mags = 0.03 * (am - am.mean())
    out = slope_se_pair(mags, am, err, bootstrap_draws=200, seed=3)
    assert math.isfinite(out["se_propagated"])
    assert math.isfinite(out["se_bootstrap"])
    assert out["se_use"] >= out["se_propagated"]
    assert out["se_propagated"] > 1e-4  # not the tiny WLS residual artifact


def test_affine_roundtrip_detector_drift() -> None:
    x = np.array([10.0, 20.0, 30.0])
    y = np.array([100.0, 110.0, 120.0])
    a, b, tx, c, d, ty = 1.02, 0.01, 5.0, -0.01, 0.98, -3.0
    xp, yp = apply_affine_xy(x, y, a=a, b=b, tx=tx, c=c, d=d, ty=ty)
    ai, bi, txi, ci, di, tyi = invert_affine_2x3(a, b, tx, c, d, ty)
    xr, yr = apply_affine_xy(xp, yp, a=ai, b=bi, tx=txi, c=ci, d=di, ty=tyi)
    assert xr == pytest.approx(x, rel=1e-9, abs=1e-9)
    assert yr == pytest.approx(y, rel=1e-9, abs=1e-9)


def test_track_detector_positions_synthetic_gaussian() -> None:
    """Synthetic star blob chain: recovered positions follow injected drift."""
    rng = np.random.default_rng(4)
    images = []
    cx, cy = 64.0, 64.0
    for i in range(5):
        img = rng.normal(0, 1, (128, 128))
        ix = int(round(cx)) + i * 3
        iy = int(round(cy)) + i * 2
        yg, xg = np.ogrid[:128, :128]
        img += 200.0 * np.exp(-((xg - ix) ** 2 + (yg - iy) ** 2) / (2 * 2.0**2))
        images.append(img)
    xs, ys = track_detector_positions(images, 64.0, 64.0, half=24, fwhm=3.0)
    assert xs[-1] - xs[0] > 10
    assert ys[-1] - ys[0] > 5


def test_centroid_cutout_detector_on_blob() -> None:
    rng = np.random.default_rng(5)
    img = rng.normal(0, 1, (64, 64))
    cx, cy = 32.5, 28.0
    yg, xg = np.ogrid[:64, :64]
    img += 150.0 * np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * 1.8**2))
    x_out, y_out = centroid_cutout_detector(img, 30.0, 26.0, half=16)
    assert abs(x_out - cx) < 1.5
    assert abs(y_out - cy) < 1.5


def test_p4_noise_consistency_fail_on_inflated_rms() -> None:
    out = p4_noise_consistency_check(29.0, sigma_r_ref_mmag=5.5)
    assert out["passed"] is False
    assert out["ratio"] > 5.0


def test_gaussian_aperture_overlap_on_axis_matches_analytic() -> None:
    fwhm = 4.0
    r_ap = 3.818
    sigma = fwhm_to_sigma(fwhm)
    expected = 1.0 - math.exp(-0.5 * (r_ap / sigma) ** 2)
    got = gaussian_aperture_overlap(0.0, r_ap, fwhm, grid_n=64)
    assert got == pytest.approx(expected, rel=0.02)


def test_gaussian_aperture_overlap_single_neighbor_hand() -> None:
    """Neighbor at sep=2*sigma, faint flux -- overlap decreases from on-axis."""
    fwhm = 4.0
    r_ap = 3.818
    sigma = fwhm_to_sigma(fwhm)
    on_axis = gaussian_aperture_overlap(0.0, r_ap, fwhm, grid_n=64)
    off = gaussian_aperture_overlap(2.0 * sigma, r_ap, fwhm, grid_n=64)
    assert off < on_axis
    assert off > 0.0


def test_contamination_fraction_single_neighbor() -> None:
    fwhm = 4.0
    r_ap = 3.818
    sigma = fwhm_to_sigma(fwhm)
    sep = 2.0 * sigma
    o = gaussian_aperture_overlap(sep, r_ap, fwhm, grid_n=64)
    target_g = 10.0
    neighbor_g = 12.0
    flux_ratio = 10 ** (-0.4 * (neighbor_g - target_g))
    expected = flux_ratio * o
    got = contamination_fraction(
        [{"sep_px": sep, "g_mag": neighbor_g}],
        r_ap_px=r_ap,
        fwhm_px=fwhm,
        target_g_mag=target_g,
    )
    assert got == pytest.approx(expected, rel=0.01)


def test_sigma_slope_pt_hand_bright_tertile() -> None:
    # sqrt(0.00269) * 0.056 * 1000 ~ 2.905 mmag
    val = sigma_slope_pt_mmag(0.00269, 0.056)
    assert val == pytest.approx(2.905, abs=0.05)


def test_sigma_slope_pt_hand_faint_tertile() -> None:
    # sqrt(0.00850) * 0.056 * 1000 ~ 5.164 mmag
    val = sigma_slope_pt_mmag(0.00850, 0.056)
    assert val == pytest.approx(5.164, abs=0.05)


def test_neighbor_sensitivity_sign() -> None:
    """More contamination at larger FWHM -> positive S if fc_p90 > fc_med."""
    s = neighbor_sensitivity_mag_per_fwhm_px(0.02, 0.01, 8.0, 4.0)
    assert s > 0
    b = attainable_neighbor_slope(s, 0.5)
    assert b == pytest.approx(0.5 * s, rel=1e-9)
