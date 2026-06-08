"""Tests for LC quality classification and mag-dependent RMS model."""

from __future__ import annotations

import math

import numpy as np

from photometry_core import (
    build_lc_quality_summary,
    build_rms_mag_model,
    classify_lc_quality,
    expected_rms_from_model,
)


def _classify(**kwargs: object) -> str:
    defaults = {
        "zone_flag": "linear",
        "lc_rms": 0.01,
        "lc_median_mag": 12.0,
        "n_frames": 139,
        "n_normal_frames": 139,
        "lunar_risk": "LOW",
        "rms_model_coeffs": None,
    }
    defaults.update(kwargs)
    return classify_lc_quality(
        zone_flag=str(defaults["zone_flag"]),
        lc_rms=float(defaults["lc_rms"]),  # type: ignore[arg-type]
        lc_median_mag=float(defaults["lc_median_mag"]),  # type: ignore[arg-type]
        n_frames=int(defaults["n_frames"]),  # type: ignore[arg-type]
        n_normal_frames=int(defaults["n_normal_frames"]),  # type: ignore[arg-type]
        lunar_risk=str(defaults["lunar_risk"]),
        rms_model_coeffs=defaults.get("rms_model_coeffs"),  # type: ignore[arg-type]
    )


def test_saturated_wins():
    assert (
        _classify(
            zone_flag="saturated",
            n_frames=5,
            n_normal_frames=0,
            lc_rms=999.0,
        )
        == "saturated"
    )


def test_no_data_low_frames():
    assert _classify(n_frames=10, n_normal_frames=10) == "no_data"


def test_no_data_low_normal_frac():
    assert _classify(n_frames=100, n_normal_frames=30) == "no_data"


def test_noisy_zone():
    assert _classify(zone_flag="noisy2") == "noisy"


def test_noisy_rms_model():
    coeffs = np.array([-0.35, -1.2])
    mag = 12.0
    expected = expected_rms_from_model(mag, coeffs)
    assert _classify(
        zone_flag="linear",
        lc_rms=5.0 * expected,
        lc_median_mag=mag,
        rms_model_coeffs=coeffs,
        lunar_risk="LOW",
    ) == "noisy"


def test_noisy_moon():
    coeffs = np.array([-0.35, -1.2])
    mag = 12.0
    expected = expected_rms_from_model(mag, coeffs)
    assert (
        _classify(
            zone_flag="linear",
            lc_rms=5.0 * expected,
            lc_median_mag=mag,
            rms_model_coeffs=coeffs,
            lunar_risk="HIGH",
        )
        == "noisy_moon"
    )


def test_good():
    coeffs = np.array([-0.35, -1.2])
    mag = 12.0
    expected = expected_rms_from_model(mag, coeffs)
    assert (
        _classify(
            zone_flag="linear",
            lc_rms=expected,
            lc_median_mag=mag,
            n_frames=139,
            n_normal_frames=139,
            rms_model_coeffs=coeffs,
        )
        == "good"
    )


def test_noisy1_with_ok_rms_is_good():
    coeffs = np.array([-0.35, -1.2])
    mag = 12.0
    expected = expected_rms_from_model(mag, coeffs)
    assert (
        _classify(
            zone_flag="noisy1",
            lc_rms=expected,
            lc_median_mag=mag,
            rms_model_coeffs=coeffs,
        )
        == "good"
    )


def test_no_model_no_rms_flag():
    assert _classify(zone_flag="linear", lc_rms=0.5, rms_model_coeffs=None) == "good"


def test_rms_model_fit():
    rows = []
    for mag in np.linspace(8.0, 16.0, 15):
        # Fainter stars → higher RMS (Poisson-like, positive slope in log space)
        rms = 10 ** (0.35 * mag - 1.2)
        rows.append(
            {
                "zone_flag": "linear",
                "lc_rms": float(rms),
                "lc_median_mag": float(mag),
            }
        )
    fit = build_rms_mag_model(rows, min_stars=10)
    assert fit is not None
    coeffs, mags_used = fit
    assert len(mags_used) >= 10
    assert float(coeffs[0]) > 0.0


def test_rms_model_too_few_stars():
    rows = [
        {"zone_flag": "linear", "lc_rms": 0.01, "lc_median_mag": 10.0 + i * 0.1}
        for i in range(5)
    ]
    assert build_rms_mag_model(rows, min_stars=10) is None


def test_quality_summary_counts():
    rows = [
        {"lc_quality_flag": "good"},
        {"lc_quality_flag": "good"},
        {"lc_quality_flag": "noisy"},
        {"lc_quality_flag": "saturated"},
        {"lc_quality_flag": "noisy_moon"},
    ]
    coeffs = np.array([0.17, -3.43])
    s = build_lc_quality_summary(
        rows,
        rms_model_coeffs=coeffs,
        rms_model_n_stars=42,
        rms_noisy_k=3.0,
    )
    assert s["good"] == 2
    assert s["noisy"] == 1
    assert s["saturated"] == 1
    assert s["noisy_moon"] == 1
    assert s["no_data"] == 0
    assert s["total"] == 5
    assert s["available"] is True
    assert abs(s["rms_model_slope"] - 0.17) < 1e-6
    assert abs(s["rms_model_intercept"] - (-3.43)) < 1e-6
    assert s["rms_model_n_stars"] == 42
    assert s["rms_noisy_k"] == 3.0


def test_quality_summary_missing_column():
    rows = [{"zone_flag": "linear", "lc_rms": 0.01} for _ in range(4)]
    s = build_lc_quality_summary(rows, rms_model_coeffs=None)
    assert s["available"] is False
    assert s["total"] == 0
    assert s["good"] == 0
    assert s["noisy"] == 0
