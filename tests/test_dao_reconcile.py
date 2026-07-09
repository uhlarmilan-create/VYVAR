"""Unit tests for Gaia<->DAO reconciliation helper (R-2 footprint + Fleming fit)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from dao_reconcile import (
    ReferencePopulationMismatch,
    apply_footprint_filter,
    bin_completeness_curve,
    blend_radius_px,
    check_reference_population_consistency,
    completeness_50_pct,
    decompose_reference_population,
    fit_fleming_completeness,
    fleming_completeness,
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


def _synthetic_reference(wcs: WCS, n: int = 40) -> pd.DataFrame:
    ra_c = np.linspace(179.995, 180.005, n)
    dec_c = np.linspace(44.995, 45.005, n)
    xp, yp = wcs.all_world2pix(ra_c, dec_c, 0)
    return pd.DataFrame(
        {
            "catalog_id": [f"{1000000000000000000 + i}" for i in range(n)],
            "ra_deg": ra_c,
            "dec_deg": dec_c,
            "mag": np.linspace(10.0, 16.0, n),
            "_x_pix": xp,
            "_y_pix": yp,
        }
    )


def test_fleming_completeness_half_at_g50():
    g50, sig = 14.0, 0.5
    assert fleming_completeness(g50, g50, sig) == pytest.approx(0.5, abs=1e-6)
    assert fleming_completeness(g50 - 2 * sig, g50, sig) > 0.9
    assert fleming_completeness(g50 + 2 * sig, g50, sig) < 0.1


def test_blend_radius_matches_crowding_convention():
    fwhm = 2.4
    assert blend_radius_px(fwhm) == pytest.approx(1.5 * fwhm)


def test_footprint_off_frame_bucket():
    wcs = _minimal_wcs()
    fwhm = 2.0
    margin = 2.0 * fwhm
    ref = _synthetic_reference(wcs, n=20)
    ref, n_off = apply_footprint_filter(ref, wcs, 2000, 2000, fwhm_px=fwhm, edge_margin_fwhm=2.0)
    assert n_off >= 0
    in_frame = ref.loc[ref["_in_frame"]]
    assert (in_frame["_x_pix"] >= margin).all()
    assert (in_frame["_x_pix"] < 2000 - margin).all()


def test_bin_completeness_curve():
    wcs = _minimal_wcs()
    ref = _synthetic_reference(wcs, n=20)
    ref, _ = apply_footprint_filter(ref, wcs, 2000, 2000, fwhm_px=2.0)
    ref_in = ref.loc[ref["_in_frame"]].copy()
    matched_ids = set(ref_in["catalog_id"].iloc[:10].astype(str))
    bins = bin_completeness_curve(ref_in, matched_ids, bin_width=0.5)
    assert bins
    assert all("completeness_frac" in b for b in bins)
    bright_bins = [b for b in bins if b["bin_center"] < 12.0]
    assert bright_bins and bright_bins[0]["completeness_frac"] == pytest.approx(1.0)


def test_fleming_fit_on_synthetic_bins():
    bins = []
    for center, frac in [(10.5, 1.0), (11.5, 1.0), (12.5, 0.95), (13.5, 0.7), (14.5, 0.35), (15.5, 0.1)]:
        bins.append(
            {
                "bin_lo": center - 0.25,
                "bin_hi": center + 0.25,
                "bin_center": center,
                "n_ref": 200,
                "n_matched": int(200 * frac),
                "completeness_frac": frac,
            }
        )
    fit = fit_fleming_completeness(bins)
    assert fit.fit_method in ("fleming1995_erf", "interpolation")
    assert math.isfinite(fit.g_lim_50)
    assert 12.0 < fit.g_lim_50 < 16.0


def test_source_id_exact_match_population_check():
    wcs = _minimal_wcs()
    ref = _synthetic_reference(wcs, n=5)
    det = pd.DataFrame(
        {
            "catalog_id": [ref["catalog_id"].iloc[0], ref["catalog_id"].iloc[1], ""],
            "x": [1000.0, 1010.0, 50.0],
            "y": [1000.0, 1010.0, 50.0],
            "phot_g_mean_mag": [10.5, 11.0, np.nan],
        }
    )
    check_reference_population_consistency(det, ref)
    det_bad = det.copy()
    det_bad.loc[0, "catalog_id"] = "9999999999999999999"
    with pytest.raises(ReferencePopulationMismatch):
        check_reference_population_consistency(det_bad, ref)


def test_decompose_reference_all_buckets():
    wcs = _minimal_wcs()
    fwhm = 2.0
    ref = _synthetic_reference(wcs, n=8)
    ref, _ = apply_footprint_filter(ref, wcs, 2000, 2000, fwhm_px=fwhm)
    ref.loc[ref.index[0], "mag"] = 18.0
    det = pd.DataFrame(
        {
            "catalog_id": [ref["catalog_id"].iloc[1], ref["catalog_id"].iloc[2], ""],
            "x": [float(ref["_x_pix"].iloc[1]), float(ref["_x_pix"].iloc[2]), 500.0],
            "y": [float(ref["_y_pix"].iloc[1]), float(ref["_y_pix"].iloc[2]), 500.0],
        }
    )
    g_lim_50 = 14.0
    labeled, counts = decompose_reference_population(ref, det, g_lim_50=g_lim_50, fwhm_px=fwhm)
    buckets = set(labeled["_bucket"])
    assert "matched" in buckets
    assert counts["n_gaia_matched"] >= 2


def test_completeness_50_headline():
    pct = completeness_50_pct(900, 100)
    assert pct == pytest.approx(90.0)


def test_reconcile_to_pipeline_meta_r2_keys():
    report = {
        "g_lim_50": 14.2,
        "g_lim_90": 15.1,
        "fit_method": "fleming1995_erf",
        "completeness_curve": [{"bin_center": 12.0, "n_ref": 10, "n_matched": 8, "completeness_frac": 0.8}],
        "n_ref_in_frame": 500,
        "n_gaia_matched": 400,
        "n_gaia_off_frame": 50,
        "n_gaia_below_limit": 30,
        "n_gaia_blended": 10,
        "n_gaia_missed": 5,
        "gaia_dao_completeness_pct": 95.0,
        "gaia_dao_completeness_raw_pct": 3.5,
        "n_dao_unmatched": 12,
        "unmatched_dao": {"n_now_matched_to_faint": 3},
        "blend_radius_px": 3.6,
        "methodology": "footprint_reference_fleming1995",
    }
    meta = reconcile_to_pipeline_meta(report)
    assert meta["g_lim_50"] == 14.2
    assert meta["g_lim_90"] == 15.1
    assert meta["n_gaia_off_frame"] == 50
    assert meta["fit_method"] == "fleming1995_erf"
    assert meta["n_dao_matched_to_faint"] == 3


def test_is_blended_edge_inclusive():
    matched = np.array([[100.0, 100.0]])
    r = blend_radius_px(2.0)
    assert is_blended_with_matched(100.0 + r, 100.0, matched, blend_r_px=r) is True
