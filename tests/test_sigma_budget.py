"""Unit tests for diagnostic sigma budget (Howell + Osborn scintillation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from photometry_core import _photometric_error
from scripts.chi2_sigma_gate import reduced_chi2_constant
from sigma_budget import (
    howell_sigma,
    scintillation_sigma,
    total_sigma,
    SIGMA_VARIANT_HOWELL_ONLY,
    SIGMA_VARIANT_HOWELL_SCINT_FULL,
)


def test_howell_sigma_matches_production():
    flux, sky, area = 10000.0, 100.0, 28.27
    gain, rn = 3.17, 7.6
    expected = _photometric_error(flux, sky, area, gain=gain, read_noise=rn)
    assert howell_sigma(flux, sky, area, gain=gain, read_noise=rn) == pytest.approx(expected)


def test_scintillation_osborn_paper_scale_sanity():
    """Larger aperture -> lower scintillation variance (D^-4/3)."""
    v200 = scintillation_sigma(telescope_diameter_m=0.2, airmass=1.5, exposure_s=60.0, altitude_m=275.0) ** 2
    v300 = scintillation_sigma(telescope_diameter_m=0.3, airmass=1.5, exposure_s=60.0, altitude_m=275.0) ** 2
    assert v200 > v300
    ratio = v200 / v300
    assert abs(ratio - (0.3 / 0.2) ** (4.0 / 3.0)) < 0.05


def test_total_sigma_quadrature_variants():
    kwargs = dict(
        flux=5000.0,
        sky_pp=50.0,
        area=100.0,
        telescope_diameter_m=0.2,
        airmass=1.5,
        exposure_s=60.0,
        altitude_m=250.0,
    )
    h_only, sh, _ = total_sigma(**kwargs, variant=SIGMA_VARIANT_HOWELL_ONLY)
    full, sh2, ss_full = total_sigma(**kwargs, variant=SIGMA_VARIANT_HOWELL_SCINT_FULL)
    assert h_only == pytest.approx(sh)
    assert sh2 == pytest.approx(sh)
    assert full == pytest.approx(math.sqrt(sh**2 + ss_full**2))


def test_chi2_near_one_with_correct_sigmas():
    rng = np.random.default_rng(0)
    sig = 0.01
    mags = rng.normal(10.0, sig, 80)
    sigmas = np.full(80, sig)
    _, dof, chi2_dof, _ = reduced_chi2_constant(mags, sigmas)
    assert dof == 79
    assert 0.6 < chi2_dof < 1.4


def test_chi2_inflated_when_sigma_too_small():
    rng = np.random.default_rng(1)
    sig = 0.012
    mags = rng.normal(10.0, sig, 50)
    _, _, ok, _ = reduced_chi2_constant(mags, np.full(50, sig))
    _, _, bad, _ = reduced_chi2_constant(mags, np.full(50, sig * 0.4))
    assert bad > ok
