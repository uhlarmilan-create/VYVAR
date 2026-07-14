"""Tests for SPARSE-CHECK-POOL external K sourcing (Amendment 1)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from check_star_kmag import (
    evaluate_k_colour_caveat,
    select_external_check_star,
)
from sparse_trust_core import (
    check_model_ratio_ci,
    compute_sparse_trust_stats,
    detrend_kmag_airmass,
    trust_band,
    two_star_model_ratio_ci,
)


def _pool_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"catalog_id": "100", "mag": 12.0, "p2p_rms": 0.02, "bp_rp": 0.5, "status": "good"},
            {"catalog_id": "200", "mag": 12.1, "p2p_rms": 0.03, "bp_rp": 1.5, "status": "good"},
            {"catalog_id": "300", "mag": 14.0, "p2p_rms": 0.01, "bp_rp": 0.6, "status": "good"},
        ]
    )


def test_external_k_excludes_ensemble_member() -> None:
    sel = select_external_check_star(
        _pool_df(),
        ensemble_ids={"100", "200"},
        target_mag=12.05,
        target_bprp=0.5,
        ensemble_bprp_median=1.0,
    )
    assert sel is not None
    assert str(sel.row["catalog_id"]) == "300"


def test_external_k_brightness_proximity_tiebreak_p2p() -> None:
    df = pd.DataFrame(
        [
            {"catalog_id": "400", "mag": 12.00, "p2p_rms": 0.04, "bp_rp": 0.55},
            {"catalog_id": "500", "mag": 12.00, "p2p_rms": 0.02, "bp_rp": 0.56},
        ]
    )
    sel = select_external_check_star(
        df, ensemble_ids=set(), target_mag=12.0, target_bprp=0.5, ensemble_bprp_median=0.5,
    )
    assert sel is not None
    assert str(sel.row["catalog_id"]) == "500"


def test_tier_excluded_source_when_colour_window_fails() -> None:
    df = pd.DataFrame(
        [
            {"catalog_id": "200", "mag": 12.0, "p2p_rms": 0.02, "bp_rp": 1.5, "status": "good"},
            {"catalog_id": "300", "mag": 12.1, "p2p_rms": 0.01, "bp_rp": 0.6, "status": "good"},
        ]
    )
    sel = select_external_check_star(
        df,
        ensemble_ids=set(),
        target_mag=12.0,
        target_bprp=0.5,
        ensemble_bprp_median=0.55,
    )
    assert sel is not None
    assert str(sel.row["catalog_id"]) == "200"
    assert sel.k_source == "tier_excluded"
    assert sel.k_tier_excluded is True


def test_k_colour_caveat_trigger() -> None:
    assert evaluate_k_colour_caveat(0.9, colour_window=0.79, airmass_range=0.25) is True
    assert evaluate_k_colour_caveat(0.5, colour_window=0.79, airmass_range=0.25) is False
    assert evaluate_k_colour_caveat(0.9, colour_window=0.79, airmass_range=0.1) is False


def test_n1_two_star_r_hand_computed() -> None:
    n = 30
    rng = np.random.default_rng(7)
    p_k = np.full(n, 0.01)
    p_c1 = np.full(n, 0.012)
    floor = 0.018
    m_k = 12.0 + rng.normal(0, 0.02, n)
    m_c1 = 11.5 + rng.normal(0, 0.015, n)
    raw, _ = two_star_model_ratio_ci(m_k, m_c1, p_k, p_c1, sigma_sys_mag=floor)
    d = m_k - m_c1
    v_obs = float(np.var(d, ddof=1))
    v_model = float(np.mean(p_k * p_k + p_c1 * p_c1 + floor * floor))
    expect = check_model_ratio_ci(v_obs, v_model, n)
    assert raw.R == pytest.approx(expect.R, rel=1e-6)
    assert raw.R_lo == pytest.approx(expect.R_lo, rel=1e-5)


def test_n1_band_capped_yellow_with_r_numbers() -> None:
    band, flags = trust_band(
        R_hi=5.0, R_lo=3.0, stability_p=float("nan"), x2_pair_mag2=float("nan"), n_comps=1,
    )
    assert band == "YELLOW"
    assert "single_comp" in flags


def test_n2_external_k_end_to_end_synthetic() -> None:
    n = 25
    rng = np.random.default_rng(11)
    m_k = 12.0 + rng.normal(0, 0.006, n)
    m_c1 = 11.5 + rng.normal(0, 0.005, n)
    m_c2 = 11.8 + rng.normal(0, 0.005, n)
    kmag = m_k - 0.5 * (m_c1 + m_c2) + 11.65
    phot = {"c1": np.full(n, 0.01), "c2": np.full(n, 0.01), "__check__": np.full(n, 0.01)}
    stats_out = compute_sparse_trust_stats(
        kmag=kmag,
        m_K=m_k,
        comp_mags={"c1": m_c1, "c2": m_c2},
        comp_photon_mag=phot,
        sigma_sys_mag=0.018,
        n_comps=2,
    )
    assert math.isfinite(stats_out.trust_R)
    assert math.isfinite(stats_out.trust_R_lo)
    assert math.isfinite(stats_out.trust_R_hi)
    assert stats_out.n_epochs == n


def test_detrend_kmag_changes_series() -> None:
    n = 20
    am = np.linspace(1.1, 1.4, n)
    km = 0.05 * am + np.random.default_rng(0).normal(0, 0.001, n)
    dt = detrend_kmag_airmass(km, am)
    assert float(np.std(dt)) < float(np.std(km))
