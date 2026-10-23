"""GS11 Step B - target dilution correction (dilution is not a comparison-star cut, COMP-ADMIT-03)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from config import AppConfig
from dilution import apply_target_dilution_to_mag_calib, compute_dilution_factor


def _cfg_gs11_on(**overrides: object) -> AppConfig:
    c = AppConfig()
    c.gs11_dilution_enabled = True
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def test_target_correction_applied() -> None:
    """GS11-2026-10: blend light is removed in flux and the true amplitude comes back."""
    cfg = _cfg_gs11_on()
    m_true = np.array([10.0, 10.5, 11.0])  # 1.0 mag eclipse
    f_c = 10 ** (-0.4 * 11.0)  # neighbour 1 mag fainter than the target at maximum
    m_obs = -2.5 * np.log10(10 ** (-0.4 * m_true) + f_c)
    assert (m_obs[2] - m_obs[0]) < 0.75  # amplitude compressed by the blend
    d = 10 ** (-0.4 * 10.0) / (10 ** (-0.4 * 10.0) + f_c)
    dil = {"dilution_factor": d, "contam_flux_total": f_c, "n_neighbors": 1}
    out, res = apply_target_dilution_to_mag_calib(m_obs, dil, cfg, target_cid="T")
    assert np.allclose(out, m_true, atol=1e-9)
    assert res["dilution_applied"] is True
    assert res["dilution_delta_mag_max"] > res["dilution_delta_mag_min"] > 0


def test_target_correction_skipped_too_low() -> None:
    cfg = _cfg_gs11_on()
    mag = np.array([10.0, 10.1])
    dil = {"dilution_factor": 0.30, "dilution_delta_mag": 1.0, "n_neighbors": 2}
    out, _ = apply_target_dilution_to_mag_calib(mag, dil, cfg, target_cid="T")
    assert np.allclose(out, mag)


def test_dilution_disabled_no_effect() -> None:
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    with patch("dilution.query_gaia_neighbors", return_value=[{"g_mag": 10.0, "sep_arcsec": 0.0}]):
        r = compute_dilution_factor(0.0, 0.0, 10.0, 4.0, "dummy.db")
    assert r["dilution_factor"] < 1.0
    mag = np.array([10.0])
    out2, res = apply_target_dilution_to_mag_calib(mag, r, cfg)
    assert np.allclose(out2, mag)
    assert res["dilution_factor"] == r["dilution_factor"]


def test_gs11_summary_keys() -> None:
    from photometry_core import build_gs11_summary

    rows = [
        {
            "dilution_factor": 0.95,
            "dilution_delta_mag": 0.02,
            "gs11_aperture_arcsec": 4.5,
        },
        {
            "dilution_factor": 0.30,
            "dilution_delta_mag": 0.5,
            "gs11_aperture_arcsec": 3.9,
        },
    ]
    s = build_gs11_summary(rows, _cfg_gs11_on(), plate_scale_arcsec=1.3)
    assert 3.0 <= s["aperture_arcsec"] <= 5.0
    for key in (
        "enabled",
        "aperture_arcsec",
        "targets_corrected",
        "targets_skipped_low_d",
        "median_correction_mmag",
        "max_correction_mmag",
    ):
        assert key in s
    assert s["enabled"] is True
    assert s["targets_corrected"] == 1
    assert s["targets_skipped_low_d"] == 1


def test_gs11_pdf_disabled_message() -> None:
    from photometry_report import gs11_report_lines

    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    lines = gs11_report_lines({"gs11_summary": {"enabled": False}}, cfg)
    assert any("disabled" in ln.lower() for ln in lines)


def test_gs11_dilution_from_summary() -> None:
    """GS11-2026-10: per-target D reaches the PDF target page and AAVSO NOTES only when GS11 is on,
    the correction was applied and the blend is significant (D < 0.99)."""
    import pandas as pd

    from dilution import gs11_dilution_from_summary

    cfg = _cfg_gs11_on()
    got = gs11_dilution_from_summary(pd.Series({"dilution_factor": 0.95, "dilution_delta_mag": 0.056}), cfg)
    assert got == (0.95, 0.056)
    assert gs11_dilution_from_summary(pd.Series({"dilution_factor": 0.995}), cfg) is None
    assert gs11_dilution_from_summary(
        pd.Series({"dilution_factor": 0.9, "gs11_dilution_skipped": True}), cfg) is None
    off = AppConfig()
    off.gs11_dilution_enabled = False
    assert gs11_dilution_from_summary(pd.Series({"dilution_factor": 0.9}), off) is None
    # dict rows (PDF star_data) work as well
    assert gs11_dilution_from_summary({"dilution_factor": 0.8, "dilution_delta_mag": 0.24}, cfg) == (0.8, 0.24)
