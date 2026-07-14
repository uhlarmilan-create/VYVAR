"""Tests for k2_cohort_core pure helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from k2_cohort_core import (
    benjamini_hochberg_fdr,
    check_k2_internal_consistency,
    extract_cell_report_stats,
    k2_eff_ci95,
    k2_eff_honest_uncertainties,
    k2_priority_verdict,
    lag1_autocorrelation,
    photon_weighted_airmass_slope,
    spearman_min_n_for_power,
    weighted_linear_regression,
)


def test_benjamini_hochberg_hand_computed() -> None:
    # Three p-values; smallest p=0.01 -> q=0.03, reject at q=0.05
    adj = benjamini_hochberg_fdr([0.01, 0.04, 0.20], q=0.05)
    assert len(adj) == 3
    assert adj[0]["reject"] is True
    assert adj[0]["q_value"] <= 0.05
    assert adj[2]["reject"] is False


def test_weighted_linear_regression_perfect_line() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])  # y = 1 + 2x
    w = np.ones(3)
    fit = weighted_linear_regression(x, y, w)
    assert fit["slope"] == pytest.approx(2.0, abs=1e-9)
    assert fit["intercept"] == pytest.approx(1.0, abs=1e-9)


def test_photon_weighted_airmass_slope_lever_arm_gate() -> None:
    mags = np.array([0.0, 0.01, -0.01])
    am = np.array([1.0, 1.05, 1.08])  # range 0.08 < 0.15
    err = np.array([0.01, 0.01, 0.01])
    out = photon_weighted_airmass_slope(mags, am, err, min_airmass_range=0.15)
    assert out["excluded_lever_arm"] is True
    assert math.isnan(out["b_X"])


def test_photon_weighted_airmass_slope_positive_slope() -> None:
    # detrended mags increase with airmass
    am = np.linspace(1.0, 1.5, 10)
    mags = 0.02 * (am - am.mean()) + np.random.default_rng(0).normal(0, 0.001, 10)
    err = np.full(10, 0.01)
    out = photon_weighted_airmass_slope(mags, am, err, min_airmass_range=0.15)
    assert out["excluded_lever_arm"] is False
    assert out["b_X"] == pytest.approx(0.02, abs=0.01)


def test_lag1_autocorrelation_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lag1_autocorrelation(y) == pytest.approx(1.0, abs=1e-9)


def test_spearman_min_n_for_power_near_literature() -> None:
    n = spearman_min_n_for_power(rho_alt=0.4, alpha=0.05, power=0.8)
    assert 40 <= n <= 55


def test_k2_priority_verdict_up() -> None:
    cells = [{
        "cell_key": "wide_CLEAR",
        "excluded": False,
        "n_t1": 50,
        "t1_fdr": {"rho": -0.35, "q_value": 0.01, "reject": True, "expected_sign": -1.0},
        "t2_fdr": {"rho": 0.1, "q_value": 0.5, "reject": False},
    }]
    v = k2_priority_verdict(cells)
    assert v["verdict"] == "UP"


def test_k2_priority_verdict_unchanged_underpowered() -> None:
    cells = [{
        "cell_key": "wide_CLEAR",
        "excluded": False,
        "n_t1": 12,
        "t1_fdr": {"rho": 0.05, "q_value": 0.9, "reject": False, "expected_sign": -1.0},
        "t2_fdr": {"rho": 0.05, "q_value": 0.9, "reject": False},
    }]
    v = k2_priority_verdict(cells)
    assert v["verdict"] == "UNCHANGED"


def test_k2_priority_verdict_unchanged_mixed_power_null() -> None:
    """Verbatim rule: DOWN needs each tested cell >=80% power, not only power-adequate subset."""
    cells = [
        {
            "cell_key": "wide_CLEAR",
            "excluded": False,
            "n_t1": 147,
            "t1_fdr": {"rho": -0.01, "q_value": 0.9, "reject": False, "expected_sign": None},
            "t2_fdr": {"rho": -0.19, "q_value": 0.11, "reject": False},
        },
        {
            "cell_key": "Newton_g",
            "excluded": False,
            "n_t1": 23,
            "t1_fdr": {"rho": -0.04, "q_value": 0.9, "reject": False, "expected_sign": -1.0},
            "t2_fdr": {"rho": -0.14, "q_value": 0.8, "reject": False},
        },
    ]
    v = k2_priority_verdict(cells)
    assert v["verdict"] == "UNCHANGED"
    assert "wide_CLEAR" in v["tested_cells"]
    assert "Newton_g" in v["tested_cells"]


def test_k2_eff_ci95_half_width() -> None:
    ci = k2_eff_ci95(-0.04, 1.0e-6)
    assert ci["ci_half_width"] == pytest.approx(1.96e-6, rel=1e-6)
    assert ci["ci_lo"] == pytest.approx(-0.04 - 1.96e-6, rel=1e-6)


def test_extract_cell_report_stats_from_summary_shape() -> None:
    rng = np.random.default_rng(0)
    stars = [
        {
            "colour_offset_signed": float(xi),
            "b_X": float(-0.04 * xi + rng.normal(0, 0.05)),
            "b_X_se": 0.002,
            "t1_lever_excluded": False,
        }
        for xi in rng.normal(0, 0.2, 30)
    ]
    cell = {
        "cell_key": "wide_CLEAR",
        "stars": stars,
        "t1_fdr": {"q_value": 0.88},
        "t2_fdr": {"rho": -0.19, "q_value": 0.11},
        "spearman_power_rho0.4": 0.999,
    }
    stats = extract_cell_report_stats(cell, n_bootstrap=200, seed=0)
    assert stats["bootstrap_authoritative"] is True
    assert math.isfinite(float(stats["bootstrap_ci_lo"]))
    assert math.isfinite(float(stats["chi2_red"]))
    assert float(stats["chi2_red"]) > 1.0


def test_check_k2_internal_consistency_flags_stop() -> None:
    warnings = check_k2_internal_consistency(-0.04, 1.0e-6, -0.013)
    assert warnings
    assert "INTERNAL_INCONSISTENCY" in warnings[0]


def test_check_k2_internal_consistency_silent_when_coherent() -> None:
    warnings = check_k2_internal_consistency(-0.04, 0.05, -0.35)
    assert warnings == []


def _overdispersed_stars(
    rng: np.random.Generator,
    *,
    n: int,
    true_slope: float,
    scatter: float,
    err: float,
) -> list[dict[str, float | bool]]:
    x = rng.normal(0.0, 0.25, n)
    y = true_slope * x + rng.normal(0.0, scatter, n)
    return [
        {
            "colour_offset_signed": float(xi),
            "b_X": float(yi),
            "b_X_se": float(err),
            "t1_lever_excluded": False,
        }
        for xi, yi in zip(x, y, strict=False)
    ]


def test_overdispersed_null_bootstrap_coverage() -> None:
    """Naive photon-weight WLS SE fails; bootstrap null coverage ~95%."""
    boot_covers = 0
    naive_covers = 0
    n_trials = 50
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        stars = _overdispersed_stars(
            rng, n=60, true_slope=0.0, scatter=0.09, err=0.002,
        )
        honest = k2_eff_honest_uncertainties(stars, n_bootstrap=400, seed=trial)
        if honest["bootstrap_ci_lo"] <= 0.0 <= honest["bootstrap_ci_hi"]:
            boot_covers += 1
        naive_ci = k2_eff_ci95(float(honest["k2_eff"]), float(honest["se_naive"]))
        if naive_ci["ci_lo"] <= 0.0 <= naive_ci["ci_hi"]:
            naive_covers += 1
    assert boot_covers / n_trials >= 0.75
    assert naive_covers / n_trials <= 0.25


def test_overdispersed_known_slope_bootstrap_finds_truth() -> None:
    rng = np.random.default_rng(42)
    stars = _overdispersed_stars(rng, n=80, true_slope=0.05, scatter=0.08, err=0.001)
    honest = k2_eff_honest_uncertainties(stars, n_bootstrap=600, seed=42)
    assert honest["se_naive"] < 0.01
    assert honest["bootstrap_ci_lo"] <= 0.05 <= honest["bootstrap_ci_hi"]
