"""Tests for SPARSE-CHECK-POOL external K sourcing (Amendment 1)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from check_star_kmag import build_comp_photon_mag_from_frames
from sparse_trust_core import (
    check_model_ratio_ci,
    compute_sparse_trust_stats,
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


def test_n1_two_star_r_hand_computed() -> None:
    n = 30
    rng = np.random.default_rng(7)
    p_k = np.full(n, 0.01)
    p_c1 = np.full(n, 0.012)
    floor = 0.018
    m_k = 12.0 + rng.normal(0, 0.02, n)
    m_c1 = 11.5 + rng.normal(0, 0.015, n)
    raw = two_star_model_ratio_ci(m_k, m_c1, p_k, p_c1, sigma_sys_mag=floor)
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


def test_build_comp_photon_matches_production_err() -> None:
    """The sidecar photon sigma is the row's production err (CCD-ONE-2026-10: no recomputation with a
    default gain / read noise when the row has none - such a row has no photon sigma)."""
    import math

    import pandas as pd
    from mag_constants import MAG_ERR_SCALE

    row = pd.Series(
        {
            "catalog_id": "999",
            "source_file": "proc_test.csv",
            "dao_flux": 1_949_529.187,
            "sigma_bkg_ap": 174.983269,
            "aperture_r_px": 5.0,
            "sky_adu_per_px_annulus": 1200.0,
        }
    )
    prod = 1.234e-3
    mag = build_comp_photon_mag_from_frames(
        pd.DataFrame([row]), ["999"], ["proc_test.csv"], )["999"][0]
    assert math.isnan(float(mag))

    row_with_err = row.copy()
    row_with_err["err"] = prod
    mag2 = build_comp_photon_mag_from_frames(
        pd.DataFrame([row_with_err]), ["999"], ["proc_test.csv"], )["999"][0]
    assert float(mag2) / MAG_ERR_SCALE == pytest.approx(prod, rel=1e-12)

