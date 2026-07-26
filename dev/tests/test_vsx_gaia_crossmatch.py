"""Unit tests for VSX->Gaia density-aware cross-match (MATCHER-FIX / MATCHER-FIX-2 / MATCHER-FIX-3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from vsx_gaia_crossmatch import (
    VsxGaiaCrossmatchDegenerateError,
    _acceptance_radius_arcsec,
    _chance_contamination_fraction,
    _check_fit_degeneracy,
    fit_rayleigh_mixture_from_separations,
    fit_sigma_q_from_separations,
    match_vsx_to_gaia_density_aware,
)


def _rayleigh_sample(rng: np.random.Generator, sigma: float, n: int) -> np.ndarray:
    uu = rng.random(n)
    return sigma * np.sqrt(-2.0 * np.log(1.0 - uu + 1e-12))


def test_degenerate_fit_guard_raises_on_chance_scale() -> None:
    """Synthetic separations at chance scale must fail with degeneracy message."""
    rng = np.random.default_rng(42)
    rho = 200.0  # deg^-2
    mean_nn = 3600.0 / (2.0 * math.sqrt(rho))
    seps = np.abs(rng.normal(mean_nn, mean_nn * 0.15, 80))

    with pytest.raises(VsxGaiaCrossmatchDegenerateError, match="degenerate"):
        fit_rayleigh_mixture_from_separations(seps, rho)


def test_two_population_mixture_recovers_injected_sigmas_and_w() -> None:
    """Synthetic two-Rayleigh true-match mixture must recover sigma_n, sigma_b, w."""
    rng = np.random.default_rng(11)
    sn_true = 0.25
    sb_true = 1.5
    w_true = 0.75
    q_true = 0.92
    rho = 80.0
    n_total = 600

    n_true = int(q_true * n_total)
    n_n = int(w_true * n_true)
    n_b = n_true - n_n
    true_seps = np.concatenate(
        [
            _rayleigh_sample(rng, sn_true, n_n),
            _rayleigh_sample(rng, sb_true, n_b),
        ]
    )
    mean_nn_deg = 1.0 / (2.0 * math.sqrt(rho))
    chance_seps = []
    while len(chance_seps) < n_total - len(true_seps):
        chance_seps.append(float(rng.exponential(mean_nn_deg) * 3600.0))
    seps = np.concatenate([true_seps, np.asarray(chance_seps[: n_total - len(true_seps)])])
    rng.shuffle(seps)

    fit = fit_rayleigh_mixture_from_separations(seps, rho)
    assert abs(fit.sigma_narrow_arcsec - sn_true) < 0.35
    assert fit.sigma_broad_arcsec >= sb_true * 0.25
    assert fit.sigma_broad_arcsec <= sb_true * 4.0
    assert abs(fit.w_fit - w_true) < 0.35
    assert abs(fit.q_fit - q_true) < 0.20


def test_acceptance_radius_from_contamination_budget() -> None:
    """r_max = sqrt(0.01 / (pi * rho)) in arcsec."""
    assert abs(_acceptance_radius_arcsec(706.0) - 7.64) < 0.05
    assert abs(_acceptance_radius_arcsec(133.0) - 17.6) < 0.2
    assert abs(_acceptance_radius_arcsec(10000.0) - 2.03) < 0.05


def test_counterpart_at_18_arcsec_accepted_at_anchor_density() -> None:
    """Regression FIX-3: ~1.8\" counterparts must pass at rho ~706 deg^-2 (reliability must not gate)."""
    rng = np.random.default_rng(20260726)
    rho = 706.0
    n_gaia = int(rho * 21.4)
    area = n_gaia / rho
    ra0, dec0 = 150.0, 45.0
    ga_ra = ra0 + rng.uniform(-2.0, 2.0, n_gaia)
    ga_dec = dec0 + rng.uniform(-2.0, 2.0, n_gaia)

    n_vsx = 30
    pick = rng.choice(n_gaia, size=n_vsx, replace=False)
    vsx_ra = np.empty(n_vsx)
    vsx_dec = np.empty(n_vsx)
    for i, j in enumerate(pick):
        off = 1.8 if i < 15 else float(rng.uniform(0.05, 0.4))
        c = SkyCoord(ra=ga_ra[j] * u.deg, dec=ga_dec[j] * u.deg).directional_offset_by(
            float(rng.uniform(0, 360)) * u.deg, off * u.arcsec
        )
        vsx_ra[i] = c.ra.deg
        vsx_dec[i] = c.dec.deg
    cids = np.array([str(i) for i in range(n_gaia)])

    rows, diag = match_vsx_to_gaia_density_aware(
        vsx_ra,
        vsx_dec,
        cids,
        ga_ra,
        ga_dec,
        field_area_deg2=area,
    )
    accepted_18 = [r for r in rows[:15] if r.accepted]
    assert len(accepted_18) >= 14
    assert diag.r_max_arcsec > 7.0
    assert diag.expected_contamination_fraction <= 0.01 + 1e-6


def test_counterpart_at_15_arcsec_accepted_at_anchor_density() -> None:
    """Regression: legitimate ~1.5\" counterparts must pass at rho ~706 deg^-2."""
    rng = np.random.default_rng(2026)
    rho = 706.0
    n_gaia = int(rho * 21.4)
    area = n_gaia / rho
    ra0, dec0 = 150.0, 45.0
    ga_ra = ra0 + rng.uniform(-2.0, 2.0, n_gaia)
    ga_dec = dec0 + rng.uniform(-2.0, 2.0, n_gaia)

    n_vsx = 40
    pick = rng.choice(n_gaia, size=n_vsx, replace=False)
    vsx_ra = np.empty(n_vsx)
    vsx_dec = np.empty(n_vsx)
    for i, j in enumerate(pick):
        off = 1.5 if i < 20 else float(rng.uniform(0.05, 0.4))
        c = SkyCoord(ra=ga_ra[j] * u.deg, dec=ga_dec[j] * u.deg).directional_offset_by(
            float(rng.uniform(0, 360)) * u.deg, off * u.arcsec
        )
        vsx_ra[i] = c.ra.deg
        vsx_dec[i] = c.dec.deg
    cids = np.array([str(i) for i in range(n_gaia)])

    rows, diag = match_vsx_to_gaia_density_aware(
        vsx_ra,
        vsx_dec,
        cids,
        ga_ra,
        ga_dec,
        field_area_deg2=area,
    )
    accepted_15 = [r for r in rows[:20] if r.accepted]
    assert len(accepted_15) >= 18
    assert diag.expected_contamination_fraction <= 0.01


def test_match_accepts_high_q_field_with_low_contamination() -> None:
    """End-to-end match on injected 1:1 counterparts yields astrometric-scale sigmas."""
    rng = np.random.default_rng(99)
    n_vsx = 40
    area = 4.0  # deg^2
    ra0, dec0 = 180.0, 30.0
    ga_ra = ra0 + np.linspace(-0.8, 0.8, n_vsx) + rng.normal(0, 0.001, n_vsx)
    ga_dec = dec0 + np.linspace(-0.8, 0.8, n_vsx) + rng.normal(0, 0.001, n_vsx)
    sigma_true = 1.8
    offsets = _rayleigh_sample(rng, sigma_true, n_vsx)
    pa = rng.uniform(0, 2 * math.pi, n_vsx)
    vsx_c = SkyCoord(ra=ga_ra * u.deg, dec=ga_dec * u.deg)
    vsx_c = vsx_c.directional_offset_by(pa * u.rad, offsets * u.arcsec)
    vsx_ra = vsx_c.ra.deg
    vsx_dec = vsx_c.dec.deg
    cids = np.array([str(i) for i in range(n_vsx)])

    rows, diag = match_vsx_to_gaia_density_aware(
        vsx_ra,
        vsx_dec,
        cids,
        ga_ra,
        ga_dec,
        field_area_deg2=area,
    )
    assert diag.converged
    assert diag.q_fit > 0.5
    assert diag.sigma_broad_arcsec < 30.0
    assert diag.n_accepted >= n_vsx * 0.75
    assert diag.expected_contamination_fraction <= 0.01


def test_degeneracy_guard_threshold() -> None:
    rho = 133.0
    thr = 0.25 * 3600.0 / math.sqrt(rho)
    with pytest.raises(VsxGaiaCrossmatchDegenerateError):
        _check_fit_degeneracy(thr + 1.0, rho)
    _check_fit_degeneracy(thr - 1.0, rho)  # no raise


def test_chance_contamination_formula() -> None:
    seps = np.array([3.0, 10.0])
    rho = 133.0
    got = _chance_contamination_fraction(seps, rho)
    exp = float(np.mean([rho * math.pi * (s / 3600.0) ** 2 for s in seps]))
    assert abs(got - exp) < 1e-12


def test_legacy_fit_sigma_q_wrapper() -> None:
    rng = np.random.default_rng(3)
    seps = _rayleigh_sample(rng, 1.0, 50)
    sigma, q = fit_sigma_q_from_separations(seps, 30.0)
    assert sigma > 0
    assert 0 < q <= 1


def test_no_acceptance_when_mixture_degenerate() -> None:
    """Pure chance separations must refuse via degeneracy guard."""
    rng = np.random.default_rng(123)
    rho = 300.0
    mean_nn = 3600.0 / (2.0 * math.sqrt(rho))
    seps = np.abs(rng.normal(mean_nn, mean_nn * 0.1, 30))
    with pytest.raises(VsxGaiaCrossmatchDegenerateError):
        fit_rayleigh_mixture_from_separations(seps, rho)
