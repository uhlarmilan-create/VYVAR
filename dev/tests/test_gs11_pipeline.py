"""GS11 Step B - pipeline integration tests (comp filter + target correction)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from config import AppConfig
from comp_selection_per_target import _apply_comp_metric_hard_filters
from dilution import apply_target_dilution_to_mag_calib, compute_dilution_factor


def _cfg_gs11_on(**overrides: object) -> AppConfig:
    c = AppConfig()
    c.gs11_dilution_enabled = True
    c.gs11_comp_max_dilution = 0.90
    c.gs11_comp_suspect_dilution = 0.98
    c.gs11_target_min_dilution = 0.50
    c.gs11_dilution_mag_limit_delta = 5.0
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def test_comp_hard_reject_low_dilution() -> None:
    """COMP-ADMIT-03: dilution is not an admission cut."""
    flux = {"c1": [1.0, 1.0, 1.0, 1.0, 1.0]}
    dil = {"c1": {"dilution_factor": 0.85, "dilution_delta_mag": 0.1}}
    notes: dict[str, str] = {}
    out, rej = _apply_comp_metric_hard_filters(
        flux,
        {},
        {},
        {},
        {},
        {},
        [],
        {},
        {},
        target_cid="T1",
        edge_bad_frame_frac_max=0.1,
        max_psf_chi2=float("inf"),
        max_fwhm_factor=float("inf"),
        dilution_map=dil,
        cfg=_cfg_gs11_on(),
        comp_quality_notes=notes,
    )
    assert "c1" in out
    assert "c1" not in rej


def test_comp_suspect_medium_dilution() -> None:
    """COMP-ADMIT-03: medium dilution no longer writes suspect notes via hard filter."""
    flux = {"c2": [1.0, 1.0, 1.0, 1.0, 1.0]}
    dil = {"c2": {"dilution_factor": 0.95, "dilution_delta_mag": 0.02}}
    notes: dict[str, str] = {}
    out, rej = _apply_comp_metric_hard_filters(
        flux,
        {},
        {},
        {},
        {},
        {},
        [],
        {},
        {},
        target_cid="T1",
        edge_bad_frame_frac_max=0.1,
        max_psf_chi2=float("inf"),
        max_fwhm_factor=float("inf"),
        dilution_map=dil,
        cfg=_cfg_gs11_on(),
        comp_quality_notes=notes,
    )
    assert "c2" in out
    assert "c2" not in rej
    assert "c2" not in notes


def test_comp_passes_high_dilution() -> None:
    flux = {"c3": [1.0, 1.0, 1.0, 1.0, 1.0]}
    dil = {"c3": {"dilution_factor": 0.99, "dilution_delta_mag": 0.001}}
    notes: dict[str, str] = {}
    out, rej = _apply_comp_metric_hard_filters(
        flux,
        {},
        {},
        {},
        {},
        {},
        [],
        {},
        {},
        target_cid="T1",
        edge_bad_frame_frac_max=0.1,
        max_psf_chi2=float("inf"),
        max_fwhm_factor=float("inf"),
        dilution_map=dil,
        cfg=_cfg_gs11_on(),
        comp_quality_notes=notes,
    )
    assert "c3" in out
    assert "c3" not in rej
    assert "c3" not in notes


def test_target_correction_applied() -> None:
    cfg = _cfg_gs11_on()
    mag = np.array([10.0, 10.1, 10.0])
    dil = {"dilution_factor": 0.95, "dilution_delta_mag": 0.02, "n_neighbors": 1}
    out, _ = apply_target_dilution_to_mag_calib(mag, dil, cfg, target_cid="T")
    assert np.allclose(out, mag + 0.02)


def test_target_correction_skipped_too_low() -> None:
    cfg = _cfg_gs11_on()
    mag = np.array([10.0, 10.1])
    dil = {"dilution_factor": 0.30, "dilution_delta_mag": 1.0, "n_neighbors": 2}
    out, _ = apply_target_dilution_to_mag_calib(mag, dil, cfg, target_cid="T")
    assert np.allclose(out, mag)


def test_dilution_disabled_no_effect() -> None:
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    flux = {"c1": [1.0, 1.0, 1.0, 1.0, 1.0]}
    notes: dict[str, str] = {}
    out, rej = _apply_comp_metric_hard_filters(
        flux,
        {},
        {},
        {},
        {},
        {},
        [],
        {},
        {},
        target_cid="T1",
        edge_bad_frame_frac_max=0.1,
        max_psf_chi2=float("inf"),
        max_fwhm_factor=float("inf"),
        dilution_map=None,
        cfg=cfg,
        comp_quality_notes=notes,
    )
    assert "c1" in out
    with patch("dilution.query_gaia_neighbors", return_value=[{"g_mag": 10.0}]):
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
    s = build_gs11_summary(rows, _cfg_gs11_on(), comps_gs11_rejected=3, plate_scale_arcsec=1.3)
    assert 3.0 <= s["aperture_arcsec"] <= 5.0
    for key in (
        "enabled",
        "aperture_arcsec",
        "comps_gs11_rejected",
        "targets_corrected",
        "targets_skipped_low_d",
        "median_correction_mmag",
        "max_correction_mmag",
    ):
        assert key in s
    assert s["enabled"] is True
    assert s["comps_gs11_rejected"] == 3
    assert s["targets_corrected"] == 1
    assert s["targets_skipped_low_d"] == 1


def test_gs11_pdf_disabled_message() -> None:
    from photometry_report import gs11_report_lines

    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    lines = gs11_report_lines({"gs11_summary": {"enabled": False}}, cfg)
    assert any("disabled" in ln.lower() for ln in lines)


def test_aavso_notes_dilution() -> None:
    import pandas as pd

    from export_reports import _aavso_gs11_notes_suffix

    cfg = _cfg_gs11_on()
    row = pd.Series({"dilution_factor": 0.95})
    assert _aavso_gs11_notes_suffix(row, cfg) == "|GS11:D=0.950"
    row_ok = pd.Series({"dilution_factor": 0.995})
    assert _aavso_gs11_notes_suffix(row_ok, cfg) == ""
