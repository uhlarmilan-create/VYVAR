"""Unit tests for Gaia<->DAO reconciliation helper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from dao_reconcile import (
    BLEND_FWHM_FACTOR,
    blend_radius_px,
    compute_gaia_dao_reconcile,
    corrected_completeness_pct,
    decompose_undetected_cone,
    estimate_g_lim,
    is_blended_with_matched,
    reconcile_to_pipeline_meta,
)


def _minimal_wcs() -> WCS:
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 2000
    hdr["NAXIS2"] = 2000
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 45.0
    hdr["CRPIX1"] = 1000.0
    hdr["CRPIX2"] = 1000.0
    hdr["CDELT1"] = -0.001
    hdr["CDELT2"] = 0.001
    return WCS(hdr)


def test_estimate_g_lim_p95():
    mags = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    assert estimate_g_lim(mags, percentile=95) == pytest.approx(14.75)


def test_blend_radius_matches_crowding_convention():
    fwhm = 2.4
    assert blend_radius_px(fwhm) == pytest.approx(1.5 * fwhm)


def test_is_blended_edge_inclusive():
    matched = np.array([[100.0, 100.0]])
    r = blend_radius_px(2.0)
    assert is_blended_with_matched(100.0 + r, 100.0, matched, blend_r_px=r) is True
    assert is_blended_with_matched(100.0 + r + 0.01, 100.0, matched, blend_r_px=r) is False


def test_decompose_all_four_buckets():
    wcs = _minimal_wcs()
    fwhm = 2.0
    blend_r = blend_radius_px(fwhm)

    cone_rows = [
        {"catalog_id": "100", "ra_deg": 180.0, "dec_deg": 45.0, "mag": 12.0},
        {"catalog_id": "200", "ra_deg": 180.01, "dec_deg": 45.01, "mag": 13.0},
        {"catalog_id": "300", "ra_deg": 180.02, "dec_deg": 45.02, "mag": 16.0},
        {"catalog_id": "400", "ra_deg": 180.01002, "dec_deg": 45.01002, "mag": 13.5},
        {"catalog_id": "500", "ra_deg": 180.05, "dec_deg": 45.05, "mag": 13.2},
    ]
    cone = pd.DataFrame(cone_rows)

    mx, my = wcs.all_world2pix(
        [180.01, 180.03],
        [45.01, 45.03],
        0,
    )
    det = pd.DataFrame(
        [
            {"catalog_id": "100", "x": 1000.0, "y": 1000.0, "mag": 12.0},
            {"catalog_id": "200", "x": float(mx[0]), "y": float(my[0]), "mag": 13.0},
            {"catalog_id": "", "x": 500.0, "y": 500.0, "mag": np.nan},
        ]
    )

    g_lim = 14.0
    labeled, counts = decompose_undetected_cone(cone, det, g_lim=g_lim, fwhm_px=fwhm, wcs=wcs)
    buckets = set(labeled["_bucket"])
    assert buckets == {"matched", "below_limit", "blended", "genuinely_missed"}
    assert counts["n_gaia_matched"] == 2
    assert counts["n_gaia_below_limit"] == 1
    assert counts["n_gaia_blended"] == 1
    assert counts["n_gaia_missed"] == 1
    assert blend_r == pytest.approx(BLEND_FWHM_FACTOR * fwhm)


def test_corrected_completeness_excludes_below_limit_and_blended():
    pct = corrected_completeness_pct(3970, 12)
    assert pct == pytest.approx(100.0 * 3970 / 3982, rel=1e-4)


def test_compute_gaia_dao_reconcile_synthetic():
    wcs = _minimal_wcs()
    cone = pd.DataFrame(
        {
            "catalog_id": [str(i) for i in range(1, 11)],
            "ra_deg": np.linspace(179.99, 180.01, 10),
            "dec_deg": np.full(10, 45.0),
            "mag": [11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 16.0],
        }
    )
    det = pd.DataFrame(
        {
            "catalog_id": ["1", "2", "3", "4", "5", ""],
            "x": [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 200.0],
            "y": [1000.0] * 6,
            "mag": [11.0, 11.5, 12.0, 12.5, 13.0, np.nan],
            "flux": [1000, 900, 800, 700, 600, 50],
            "peak_dao": [100, 90, 80, 70, 60, 500],
        }
    )
    report = compute_gaia_dao_reconcile(cone, det, fwhm_px=2.0, wcs=wcs, g_lim_percentile=95)
    assert report["g_lim_est"] is not None
    assert report["n_gaia_matched"] == 5
    assert report["n_dao_unmatched"] == 1
    assert report["gaia_dao_completeness_pct"] is not None
    assert report["gaia_dao_completeness_raw_pct"] == pytest.approx(50.0)
    meta = reconcile_to_pipeline_meta(report)
    assert meta["gaia_dao_completeness_pct"] == report["gaia_dao_completeness_pct"]
    assert meta["gaia_dao_completeness_raw_pct"] == report["gaia_dao_completeness_raw_pct"]
    assert "n_gaia_missed" in meta


def test_reconcile_to_pipeline_meta_keys():
    report = {
        "g_lim_est": 14.1,
        "n_gaia_matched": 100,
        "n_gaia_below_limit": 90000,
        "n_gaia_blended": 500,
        "n_gaia_missed": 3,
        "gaia_dao_completeness_pct": 97.09,
        "gaia_dao_completeness_raw_pct": 0.1,
        "n_dao_unmatched": 7,
        "g_lim_stats": {"percentile": 95.0},
        "blend_radius_px": 3.6,
        "blend_radius_arcsec": 35.2,
    }
    meta = reconcile_to_pipeline_meta(report)
    assert meta == {
        "g_lim_est": 14.1,
        "n_gaia_matched": 100,
        "n_gaia_below_limit": 90000,
        "n_gaia_blended": 500,
        "n_gaia_missed": 3,
        "gaia_dao_completeness_pct": 97.09,
        "gaia_dao_completeness_raw_pct": 0.1,
        "n_dao_unmatched": 7,
        "g_lim_percentile": 95.0,
        "blend_radius_px": 3.6,
        "blend_radius_arcsec": 35.2,
    }


def test_collinearity_classifies_satellite_trail(tmp_path: Path):
    wcs = _minimal_wcs()
    cone = pd.DataFrame(
        {
            "catalog_id": ["1", "2", "3", "4"],
            "ra_deg": [180.0, 180.001, 180.002, 180.003],
            "dec_deg": [45.0, 45.0, 45.0, 45.0],
            "mag": [12.0, 12.5, 13.0, 13.5],
        }
    )
    det = pd.DataFrame(
        {
            "catalog_id": ["", "", "", "1"],
            "x": [100.0, 200.0, 300.0, 1000.0],
            "y": [100.0, 200.0, 300.0, 1000.0],
            "mag": [np.nan, np.nan, np.nan, 12.0],
            "peak_dao": [10, 10, 10, 80],
            "flux": [5, 5, 5, 900],
        }
    )
    report = compute_gaia_dao_reconcile(cone, det, fwhm_px=2.0, wcs=wcs)
    ud = report["unmatched_dao"]
    assert ud["n_dao_unmatched"] == 3
    assert ud["collinearity"]["consistent_with_line"] is True
    assert ud["n_artifact_candidates"] >= 3
