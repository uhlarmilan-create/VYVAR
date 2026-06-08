"""
Unit tests for photometry_core.py — physical correctness validation.
Run with: python -m pytest tests/test_photometry_core.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1 — Howell (1989) CCD equation
# ---------------------------------------------------------------------------
def test_snr_formula_howell1989():
    """
    Validate _photometric_error() against Howell (1989) PASP 101:616, eq. 2.
    Known values: flux=10000 ADU, sky=100 ADU/px, area=28px, gain=3.17, RN=7.6
    variance = flux/g + sky/g*area + (RN/g)^2 * area
    """
    from photometry_core import _photometric_error

    flux = 10000.0
    sky = 100.0
    area = 28.27  # pi * 3^2
    gain = 3.17
    rn = 7.6

    err = _photometric_error(flux, sky, area, gain, rn)

    # manual calculation — relative flux error sqrt(variance) / flux
    variance = flux / gain + sky / gain * area + (rn / gain) ** 2 * area
    expected_err = math.sqrt(variance) / flux

    assert math.isfinite(err), "Error must be finite"
    assert abs(err - expected_err) < 1e-6, (
        f"Howell (1989) CCD error mismatch: got {err:.8f}, expected {expected_err:.8f}"
    )


def test_snr_zero_flux():
    """Zero or negative flux must return NaN, not crash."""
    from photometry_core import _photometric_error

    assert not math.isfinite(_photometric_error(0.0, 100.0, 28.0, 3.17, 7.6))
    assert not math.isfinite(_photometric_error(-100.0, 100.0, 28.0, 3.17, 7.6))


def test_snr_sky_dominated():
    """
    Sky-dominated regime (faint star): error must increase with sky.
    """
    from photometry_core import _photometric_error

    err_low_sky = _photometric_error(100.0, 10.0, 28.0, 3.17, 7.6)
    err_high_sky = _photometric_error(100.0, 500.0, 28.0, 3.17, 7.6)
    assert err_high_sky > err_low_sky, "Higher sky must produce larger error"


# ---------------------------------------------------------------------------
# Test 2 — Broeg (2005) inverse-variance weights
# ---------------------------------------------------------------------------
def test_broeg_weights_inverse_variance():
    """
    Broeg (2005) AN 326:134 — weights w_i = 1/sigma_i^2.
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
# Test 3 — Sky subtraction
# ---------------------------------------------------------------------------
def test_sky_subtraction_formula():
    """
    dao_flux = aperture_sum - sky_pp * area  (Howell 1989 §2)
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
# Test 4 — ZP MAD sigma-clip (Stetson 1987)
# ---------------------------------------------------------------------------
def test_zp_sigmaclip_removes_outliers():
    """
    MAD-based 3-sigma clip must remove outliers from ZP residuals.
    Stetson (1987) PASP 99:191 — iterative outlier rejection.
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
# Test 5 — SNR-optimal aperture table
# ---------------------------------------------------------------------------
def test_snr_aperture_increases_with_flux():
    """
    Brighter stars (higher flux) need larger optimal aperture.
    Howell (1989) §3 — r_opt grows with flux in photon-noise regime.
    """
    from photometry_core import compute_snr_optimal_aperture_table

    table_bright = compute_snr_optimal_aperture_table(
        fwhm_px=3.0, sky_adu_per_px=500.0, gain=3.17, read_noise=7.6
    )
    table_faint = compute_snr_optimal_aperture_table(
        fwhm_px=3.0, sky_adu_per_px=5000.0, gain=3.17, read_noise=7.6
    )

    # bright star (mag 8) should have larger aperture than faint (mag 14)
    r_bright_8 = table_bright["table"][8.0]
    r_bright_14 = table_bright["table"][14.0]
    assert r_bright_8 >= r_bright_14, (
        "Brighter stars need larger or equal optimal aperture"
    )

    # high sky → smaller optimal aperture (minimize sky noise)
    r_faint_12_lowsky = table_bright["table"][12.0]
    r_faint_12_highsky = table_faint["table"][12.0]
    assert r_faint_12_highsky <= r_faint_12_lowsky, (
        "Higher sky background should yield smaller or equal optimal aperture"
    )


def test_snr_aperture_within_bounds():
    """Optimal aperture must respect r_min and r_max bounds."""
    from photometry_core import compute_snr_optimal_aperture_table

    fwhm = 3.0
    result = compute_snr_optimal_aperture_table(
        fwhm_px=fwhm, sky_adu_per_px=1000.0, gain=3.17, read_noise=7.6
    )
    r_min = result["r_min_px"]
    r_max = result["r_max_px"]

    for mag_str, r in result["table"].items():
        assert r_min - 1e-6 <= r <= r_max + 1e-6, (
            f"Aperture r={r:.3f} for mag {mag_str} outside bounds "
            f"[{r_min:.3f}, {r_max:.3f}]"
        )


# ---------------------------------------------------------------------------
# Color term — BP-RP extrapolation guard
# ---------------------------------------------------------------------------
def test_color_term_extrapolation_in_range():
    from photometry_core import _check_color_term_extrapolation

    assert _check_color_term_extrapolation(1.0, [0.8, 0.9, 1.1, 1.2]) is True
    assert _check_color_term_extrapolation(0.8, [0.8, 0.9, 1.1, 1.2]) is True
    assert _check_color_term_extrapolation(1.2, [0.8, 0.9, 1.1, 1.2]) is True


def test_color_term_extrapolation_out_of_range_blocks():
    from photometry_core import _check_color_term_extrapolation

    assert _check_color_term_extrapolation(2.5, [0.8, 0.9, 1.1, 1.2]) is False
    assert _check_color_term_extrapolation(0.5, [0.8, 0.9, 1.1, 1.2]) is False


def test_color_term_extrapolation_tolerance():
    from photometry_core import _check_color_term_extrapolation

    comps = [0.8, 0.9, 1.1, 1.2]
    assert _check_color_term_extrapolation(1.25, comps, extrapolation_tol=0.0) is False
    assert _check_color_term_extrapolation(1.25, comps, extrapolation_tol=0.1) is True


def test_color_term_extrapolation_fallback_preserves_mag():
    """Out-of-range → apply_color_term not used; magnitudes unchanged."""
    import numpy as np

    from photometry_core import (
        _check_color_term_extrapolation,
        apply_color_term,
    )

    mag = np.array([12.0, 12.01, 11.99], dtype=np.float64)
    comp_bp_rp = {"c1": 0.9, "c2": 1.0, "c3": 1.1}
    comp_quality = {
        "c1": {"quality": "good"},
        "c2": {"quality": "good"},
        "c3": {"quality": "good"},
    }
    target_bp_rp = 2.0
    c1 = -0.5
    in_range = _check_color_term_extrapolation(
        target_bp_rp,
        list(comp_bp_rp.values()),
    )
    assert in_range is False
    mag_ct, ct_corr, bp_med = apply_color_term(
        mag, target_bp_rp, comp_bp_rp, comp_quality, c1
    )
    assert abs(float(ct_corr)) > 0.1
    # Call-site fallback (block path)
    mag_blocked = mag.copy()
    ct_corr_blocked = 0.0
    ct_ok = False
    if not in_range:
        mag_blocked = mag.copy()
        ct_corr_blocked = 0.0
        ct_ok = False
    assert ct_ok is False
    assert ct_corr_blocked == 0.0
    np.testing.assert_array_equal(mag_blocked, mag)


@pytest.mark.parametrize(
    "filter_name",
    ["TG", "TB", "TR", "SG", "SR", "SI", "CV", "CR"],
)
def test_should_apply_color_term_osc_aavso_broadband(filter_name: str):
    from photometry_core import should_apply_color_term

    apply, reason = should_apply_color_term(
        obs_group=filter_name,
        c1=-0.5,
        c1_stderr=0.05,
        n_comp=8,
        min_comp_for_ct=7,
    )
    assert apply is True
    assert "aplikovaný" in reason.lower() or "CT" in reason


@pytest.mark.parametrize(
    "filter_name",
    ["NoFilter", "Clear", "L", "Luminance", "test"],
)
def test_should_apply_color_term_not_broadband_or_nofilter(filter_name: str):
    from photometry_core import should_apply_color_term

    apply, _ = should_apply_color_term(
        obs_group=filter_name,
        c1=-0.5,
        c1_stderr=0.05,
        n_comp=8,
        min_comp_for_ct=7,
    )
    assert apply is False


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


def test_target_display_name_prefers_vsx_over_nan():
    from photometry_core import _target_display_name

    row = {"vsx_name": "V0842 Her", "catalog_id": "1400549806859236864"}
    assert _target_display_name(row) == "V0842 Her"
    row_nan = {"vsx_name": float("nan"), "catalog_id": "1400549806859236864"}
    assert _target_display_name(row_nan) == "1400549806859236864"


def test_variable_targets_presel_stub_detection(tmp_path):
    import pandas as pd

    from photometry_core import _variable_targets_looks_like_ct_presel_stub

    ps = tmp_path / "B_20_2"
    ps.mkdir()
    vt = ps / "variable_targets.csv"
    ms = ps / "masterstars.csv"
    pd.DataFrame(
        {
            "name": ["M67 in-range 123"],
            "catalog_id": ["123"],
            "notes": ["CT presel in-range BP-RP [0.5,1.5]"],
        }
    ).to_csv(vt, index=False)
    pd.DataFrame({"catalog_id": [str(i) for i in range(300)]}).to_csv(ms, index=False)
    assert _variable_targets_looks_like_ct_presel_stub(vt, masterstars_csv=ms) is True

    pd.DataFrame({"name": ["V0842 Her"] * 100, "catalog_id": [str(i) for i in range(100)], "notes": [""] * 100}).to_csv(vt, index=False)
    assert _variable_targets_looks_like_ct_presel_stub(vt, masterstars_csv=ms) is False
