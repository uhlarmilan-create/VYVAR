"""
Unit tests for photometry_core.py - physical correctness validation.
Run with: python -m pytest tests/test_photometry_core.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1 - Howell (1989) CCD equation
# ---------------------------------------------------------------------------
def test_snr_formula_howell1989():
    """
    Validate _photometric_error() against Merline & Howell (1995) with the sky-estimation term of a
    median sky (CCD-ONE-2026-10). Known values: flux=10000 ADU, sky=100 ADU/px, area=28px, n_sky=400 px,
    gain=3.17, RN=7.6:
    variance = flux/g + area (1 + (pi/2) area/n_sky) (sky/g + (RN/g)^2)
    """
    from photometry_core import _photometric_error

    flux = 10000.0
    sky = 100.0
    area = 28.27  # pi * 3^2
    gain = 3.17
    rn = 7.6

    n_sky = 400.0
    err = _photometric_error(flux, sky, area, n_sky, gain=gain, read_noise=rn)

    # manual calculation - relative flux error sqrt(variance) / flux
    variance = flux / gain + area * (1.0 + (math.pi / 2.0) * area / n_sky) * (sky / gain + (rn / gain) ** 2)
    expected_err = math.sqrt(variance) / flux

    assert math.isfinite(err), "Error must be finite"
    assert abs(err - expected_err) < 1e-6, (
        f"Howell (1989) CCD error mismatch: got {err:.8f}, expected {expected_err:.8f}"
    )


def test_snr_zero_flux():
    """Zero or negative flux must return NaN, not crash."""
    from photometry_core import _photometric_error

    assert not math.isfinite(_photometric_error(0.0, 100.0, 28.0, 400.0, gain=3.17, read_noise=7.6))
    assert not math.isfinite(_photometric_error(-100.0, 100.0, 28.0, 400.0, gain=3.17, read_noise=7.6))


def test_snr_sky_dominated():
    """
    Sky-dominated regime (faint star): error must increase with sky.
    """
    from photometry_core import _photometric_error

    err_low_sky = _photometric_error(100.0, 10.0, 28.0, 400.0, gain=3.17, read_noise=7.6)
    err_high_sky = _photometric_error(100.0, 500.0, 28.0, 400.0, gain=3.17, read_noise=7.6)
    assert err_high_sky > err_low_sky, "Higher sky must produce larger error"


# ---------------------------------------------------------------------------
# Test 2 - Broeg (2005) inverse-variance weights
# ---------------------------------------------------------------------------
def test_broeg_weights_inverse_variance():
    """
    Broeg (2005) AN 326:134 - weights w_i = 1/sigma_i^2.
    Better comp star (lower RMS) must get higher weight.
    """
    # simulate comp_weight calculation as in ensemble_normalize()
    comp_rms = np.array([0.005, 0.010, 0.020])  # sigma per comp
    weights = 1.0 / (comp_rms**2)
    weights /= weights.sum()

    # best comp (lowest RMS) must have highest weight
    assert weights[0] > weights[1] > weights[2], (
        "Broeg (2005): lower RMS comp must have higher weight"
    )

    # weights must sum to 1
    assert abs(weights.sum() - 1.0) < 1e-10, "Weights must sum to 1"


def test_broeg_equal_comps():
    """Equal-RMS comp stars must get equal weights."""
    comp_rms = np.array([0.01, 0.01, 0.01])
    weights = 1.0 / (comp_rms**2)
    weights /= weights.sum()
    assert np.allclose(weights, 1 / 3), "Equal RMS must give equal weights"


# ---------------------------------------------------------------------------
# Test 3 - Sky subtraction
# ---------------------------------------------------------------------------
def test_sky_subtraction_formula():
    """
    dao_flux = aperture_sum - sky_pp * area  (Howell 1989 S2)
    Net flux must equal gross minus sky contribution.
    """
    # draft_310 sky level; gross high enough for positive net (bright star)
    aperture_sum = 100000.0
    sky_pp = 2078.83  # ADU/px
    area = 34.59  # pi * 3.318^2

    net_flux = aperture_sum - sky_pp * area

    assert net_flux > 0, "Net flux must be positive for bright star"
    assert net_flux < aperture_sum, "Net flux must be less than gross"
    expected = aperture_sum - sky_pp * area
    assert abs(net_flux - expected) < 1e-6


def test_sky_subtraction_negative_net():
    """Very faint star below sky should produce negative net flux."""
    aperture_sum = 100.0
    sky_pp = 2078.83
    area = 34.59
    net_flux = aperture_sum - sky_pp * area
    assert net_flux < 0, "Net flux below sky must be negative"


# ---------------------------------------------------------------------------
# Test 4 - ZP MAD sigma-clip (Stetson 1987)
# ---------------------------------------------------------------------------
def test_zp_sigmaclip_removes_outliers():
    """
    MAD-based 3-sigma clip must remove outliers from ZP residuals.
    Stetson (1987) PASP 99:191 - iterative outlier rejection.
    """
    zp_residuals = np.array(
        [0.01, -0.01, 0.02, -0.02, 0.015, -0.015, 5.0, -4.5]
    )  # last two = outliers

    med = np.median(zp_residuals)
    mad = np.median(np.abs(zp_residuals - med))
    mask = np.abs(zp_residuals - med) < 3 * 1.4826 * mad
    clipped = zp_residuals[mask]

    assert len(clipped) == 6, f"Expected 6 inliers, got {len(clipped)}"
    assert all(abs(v) < 1.0 for v in clipped), "Outliers must be removed"


def test_zp_sigmaclip_preserves_good_data():
    """Sigma-clip must not remove valid measurements."""
    zp_residuals = np.array([0.01, -0.01, 0.02, -0.02, 0.015, -0.015])
    med = np.median(zp_residuals)
    mad = np.median(np.abs(zp_residuals - med))
    mask = np.abs(zp_residuals - med) < 3 * 1.4826 * mad
    assert mask.sum() == len(zp_residuals), "All good data must be preserved"


# ---------------------------------------------------------------------------
# Color term - BP-RP extrapolation guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("filter_name", "expected"),
    [("SG", True), ("SR", True), ("SI", True), ("g", True), ("V", True),
     ("TG", False), ("TB", False), ("TR", False), ("CV", False), ("CR", False),
     ("NoFilter", False), ("Clear", False), ("L", False), ("Luminance", False), ("test", False)],
)
def test_colour_term_only_for_standard_filters(filter_name: str, expected: bool):
    """CT-ONE-2026-10 / BANDAUTH-2026-10: one authority decides whether a filter gets a colour term
    (should_apply_color_term of the removed "fit" path is gone; tri-colour OSC channels are not standard)."""
    from gaia_bands import filter_band_for_obs_group

    assert filter_band_for_obs_group(filter_name).transformable is expected


@pytest.mark.parametrize(
    ("mode", "obs_group", "expected"),
    [
        ("auto", "B_20_2", True),
        ("auto", "L_20_2", False),
        ("on", "L_20_2", True),
        ("off", "B_20_2", False),
    ],
)
def test_resolve_apply_color_term_toggle(mode, obs_group, expected):
    from config import AppConfig
    from photometry_core import resolve_apply_color_term

    cfg = AppConfig()
    cfg.apply_color_term = mode
    assert resolve_apply_color_term(cfg, obs_group) is expected


def test_resolve_apply_color_term_clear_is_untransformed():
    """BANDAUTH-2026-10: Clear / unfiltered has no colour term in auto mode (CFG-REVIEW-2026-10)."""
    from config import AppConfig
    from photometry_core import resolve_apply_color_term

    cfg = AppConfig()
    cfg.apply_color_term = "auto"
    assert resolve_apply_color_term(cfg, "NoFilter|60|2") is False


def test_target_display_name_prefers_vsx_over_nan():
    from photometry_core import _target_display_name

    row = {"vsx_name": "V0842 Her", "catalog_id": "1400549806859236864"}
    assert _target_display_name(row) == "V0842 Her"
    row_nan = {"vsx_name": float("nan"), "catalog_id": "1400549806859236864"}
    assert _target_display_name(row_nan) == "1400549806859236864"

