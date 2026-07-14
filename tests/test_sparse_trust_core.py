"""Unit tests for sparse_trust_core (hand-computed + synthetic S1)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from sparse_trust_core import (
    SparseTrustConfig,
    check_model_ratio_ci,
    comp_stability_test,
    compute_sparse_trust_stats,
    diff_variance,
    photon_corrected_excess,
    sigma_zp_per_epoch,
    triangulate_variances,
    trust_band,
)


def test_triangulation_hand_computed() -> None:
    # s2_KC1=0.04, s2_KC2=0.09, s2_C1C2=0.01
    tri = triangulate_variances(0.04, 0.09, 0.01)
    assert tri.sig2_K == pytest.approx(0.06)
    assert tri.sig2_C1 == pytest.approx(0.0)
    assert tri.sig2_C2 == pytest.approx(0.03)
    assert tri.triangulation_clipped is True


def test_photon_corrected_excess_clip() -> None:
    x2, clipped = photon_corrected_excess(0.01, 0.02)
    assert x2 == 0.0
    assert clipped is True
    x2b, clipped_b = photon_corrected_excess(0.05, 0.02)
    assert x2b == pytest.approx(0.03)
    assert clipped_b is False


def test_check_model_ratio_ci_n25() -> None:
    n = 25
    v_obs = 0.0004
    v_model = 0.0004
    ci = check_model_ratio_ci(v_obs, v_model, n)
    assert ci.R == pytest.approx(1.0)
    df = n - 1
    assert ci.R_lo == pytest.approx(1.0 * df / stats.chi2.ppf(0.975, df))
    assert ci.R_hi == pytest.approx(1.0 * df / stats.chi2.ppf(0.025, df))


def test_trust_band_green_red_yellow() -> None:
    cfg = SparseTrustConfig()
    g, _ = trust_band(R_hi=1.2, R_lo=0.8, stability_p=0.05, x2_pair_mag2=0.0, n_comps=2, cfg=cfg)
    assert g == "GREEN"
    r, flags = trust_band(R_hi=2.0, R_lo=4.5, stability_p=0.5, x2_pair_mag2=0.0, n_comps=2, cfg=cfg)
    assert r == "RED"
    assert "R_lo>=" in flags[0]
    y, _ = trust_band(R_hi=2.0, R_lo=1.0, stability_p=0.005, x2_pair_mag2=0.0, n_comps=2, cfg=cfg)
    assert y == "YELLOW"
    y1, f1 = trust_band(R_hi=1.0, R_lo=0.5, stability_p=0.5, x2_pair_mag2=0.0, n_comps=1, cfg=cfg)
    assert y1 == "YELLOW"
    assert "single_comp" in f1


def test_sigma_zp_per_epoch_two_comp() -> None:
    n = 5
    flux = np.array([[100.0] * n, [200.0] * n])
    phot = np.array([[0.01] * n, [0.008] * n])
    x2 = np.array([0.0001, 0.0001])
    sig = sigma_zp_per_epoch(flux, phot, x2)
    assert np.all(np.isfinite(sig))
    # w = [1/3, 2/3]; var = (1/9)*(0.01^2+0.0001) + (4/9)*(0.008^2+0.0001)
    w0, w1 = 1.0 / 3.0, 2.0 / 3.0
    expected = math.sqrt(w0 * w0 * (0.01**2 + 0.0001) + w1 * w1 * (0.008**2 + 0.0001))
    assert sig[0] == pytest.approx(expected, rel=1e-6)


def test_comp_stability_test_photon_only() -> None:
    rng = np.random.default_rng(42)
    n = 30
    p1 = rng.uniform(0.005, 0.015, n)
    p2 = rng.uniform(0.005, 0.015, n)
    noise = rng.normal(0, 0.003, n)
    m1 = 12.0 + noise
    m2 = 12.1 + noise * 0.9
    s2 = diff_variance(m1, m2)
    tri = triangulate_variances(
        diff_variance(m1, m1 + 0.1),
        diff_variance(m1, m2),
        s2,
    )
    stab = comp_stability_test(
        s2,
        p1,
        p2,
        sig2_C1_hat=tri.sig2_C1,
        sig2_C2_hat=tri.sig2_C2,
        pbar2_C1=float(np.mean(p1 * p1)),
        pbar2_C2=float(np.mean(p2 * p2)),
    )
    assert math.isfinite(stab.p_value)
    assert stab.p_value > 0.001


@pytest.mark.slow
@pytest.mark.parametrize("n_epochs", [15, 25, 139])
def test_s1_synthetic_triangulation_coverage(n_epochs: int) -> None:
    """S1: triangulation recovers injected sigmas within CI in >= 93% of trials."""
    rng = np.random.default_rng(n_epochs)
    n_trials = 500
    ok = 0
    clip = 0
    for _ in range(n_trials):
        sig_k = rng.uniform(0.003, 0.015)
        sig_c1 = rng.uniform(0.003, 0.012)
        sig_c2 = rng.uniform(0.003, 0.012)
        pb_k = rng.uniform(0.00001, 0.00005)
        pb_c1 = rng.uniform(0.00001, 0.00005)
        pb_c2 = rng.uniform(0.00001, 0.00005)
        x2_k = max(sig_k**2 - pb_k, 0)
        x2_c1 = max(sig_c1**2 - pb_c1, 0)
        x2_c2 = max(sig_c2**2 - pb_c2, 0)
        n = n_epochs
        m_k = rng.normal(12.0, math.sqrt(x2_k + pb_k), n)
        m_c1 = rng.normal(11.5, math.sqrt(x2_c1 + pb_c1), n)
        m_c2 = rng.normal(11.8, math.sqrt(x2_c2 + pb_c2), n)
        s2_kc1 = diff_variance(m_k, m_c1)
        s2_kc2 = diff_variance(m_k, m_c2)
        s2_c1c2 = diff_variance(m_c1, m_c2)
        tri = triangulate_variances(s2_kc1, s2_kc2, s2_c1c2)
        if tri.triangulation_clipped:
            clip += 1
        # chi2 CI on sig_K (variance domain)
        hat = tri.sig2_K
        if not math.isfinite(hat):
            continue
        df = n - 1
        lo = hat * df / stats.chi2.ppf(0.975, df)
        hi = hat * df / stats.chi2.ppf(0.025, df)
        if lo <= sig_k**2 <= hi:
            ok += 1
    rate = ok / n_trials
    assert rate >= 0.93, f"n={n_epochs}: coverage {rate:.3f}, clip_rate={clip/n_trials:.3f}"


def test_compute_sparse_trust_stats_smoke() -> None:
    n = 20
    rng = np.random.default_rng(0)
    m_k = 12.0 + rng.normal(0, 0.005, n)
    m_c1 = 11.5 + rng.normal(0, 0.004, n)
    m_c2 = 11.8 + rng.normal(0, 0.004, n)
    kmag = m_k - 0.5 * (m_c1 + m_c2) + 11.65
    phot = {cid: np.full(n, 0.01) for cid in ("c1", "c2")}
    phot["__check__"] = np.full(n, 0.012)
    stats_out = compute_sparse_trust_stats(
        kmag=kmag,
        m_K=m_k,
        comp_mags={"c1": m_c1, "c2": m_c2},
        comp_photon_mag=phot,
        sigma_sys_mag=0.018,
        n_comps=2,
    )
    assert stats_out.check_sparse is True
    assert stats_out.n_epochs == n
    assert math.isfinite(stats_out.trust_R)
