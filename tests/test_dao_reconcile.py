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
    apply_limit_censoring,
    bin_completeness_curve,
    blend_radius_px,
    check_reference_population_consistency,
    completeness_50_pct,
    decompose_reference_population,
    fit_fleming_completeness,
    fleming_completeness,
    is_blended_with_matched,
    reconcile_to_pipeline_meta,
    resolve_effective_match_depth,
    split_missed_by_g90,
    CENSOR_MARGIN_MAG,
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


def test_limit_censoring_when_fit_exceeds_reference_depth():
    lim = apply_limit_censoring(19.25, 17.5, label="G_lim_50")
    assert lim.censored is True
    assert lim.value_g == pytest.approx(17.5)
    assert lim.raw_fit_g == pytest.approx(19.25)
    assert "censored" in lim.display
    assert "19.25" not in lim.display


def test_limit_not_censored_when_fit_within_depth():
    lim = apply_limit_censoring(14.97, 17.5, label="G_lim_50")
    assert lim.censored is False
    assert lim.value_g == pytest.approx(14.97)


def test_censoring_synthetic_high_at_reference_edge():
    """Curve still high at reference edge -> censored flag, no extrapolated G surfaced."""
    bins = []
    for center, frac in [(13.0, 0.55), (14.0, 0.72), (15.0, 0.85), (16.0, 0.94), (17.0, 0.98), (17.4, 0.99)]:
        bins.append(
            {
                "bin_lo": center - 0.25,
                "bin_hi": center + 0.25,
                "bin_center": center,
                "n_ref": 100,
                "n_matched": int(100 * frac),
                "completeness_frac": frac,
            }
        )
    fit = fit_fleming_completeness(bins)
    depth = 17.5
    lim50 = apply_limit_censoring(fit.g_lim_50, depth, label="G_lim_50")
    assert float(fit.g_lim_50) > depth - CENSOR_MARGIN_MAG
    assert lim50.censored is True
    assert lim50.value_g == pytest.approx(depth)
    assert lim50.raw_fit_g == pytest.approx(float(fit.g_lim_50))
    assert "censored" in lim50.display


def test_split_missed_below_g90_and_fadezone():
    labeled = pd.DataFrame(
        {
            "_bucket": ["genuinely_missed"] * 3,
            "_mag": [12.0, 13.5, 14.5],
        }
    )
    split = split_missed_by_g90(
        labeled,
        g_lim_90=14.0,
        g_lim_50=15.0,
        g_lim_90_censored=False,
        reference_depth_g=17.5,
    )
    assert split["n_missed_below_g90"] == 2
    assert split["n_missed_fadezone"] == 1


def test_resolve_effective_match_depth_masterstar_default():
    md = resolve_effective_match_depth({}, is_masterstar=True)
    assert md["match_depth"] == pytest.approx(18.0)
    assert "18.0" in md["match_depth_source"]


def test_flat_curve_no_crossing_censored_not_median():
    """Flat completeness to reference edge -> no_crossing, not median-bin fabrication."""
    bins = []
    for center, frac in [
        (10.0, 1.0),
        (12.0, 1.0),
        (14.0, 1.0),
        (16.0, 1.0),
        (17.25, 0.988),
    ]:
        bins.append(
            {
                "bin_lo": center - 0.25,
                "bin_hi": center + 0.25,
                "bin_center": center,
                "n_ref": 50,
                "n_matched": int(50 * frac),
                "completeness_frac": frac,
            }
        )
    fit = fit_fleming_completeness(bins)
    assert fit.fit_method == "no_crossing"
    assert fit.g_lim_50 is None
    assert fit.no_crossing_50 is True
    depth = 17.5
    lim50 = apply_limit_censoring(fit.g_lim_50, depth, label="G_lim_50", no_crossing=True)
    assert lim50.censored is True
    assert lim50.value_g == pytest.approx(depth)
    assert lim50.raw_fit_g is None
    assert "no crossing" in lim50.display
    assert "13" not in lim50.display


def test_degenerate_two_bin_input_flagged():
    bins = [
        {
            "bin_lo": 10.0,
            "bin_hi": 10.5,
            "bin_center": 10.25,
            "n_ref": 20,
            "n_matched": 10,
            "completeness_frac": 0.5,
        },
        {
            "bin_lo": 11.0,
            "bin_hi": 11.5,
            "bin_center": 11.25,
            "n_ref": 20,
            "n_matched": 18,
            "completeness_frac": 0.9,
        },
    ]
    fit = fit_fleming_completeness(bins)
    assert fit.fit_method == "degenerate"
    assert fit.g_lim_50 is not None
    assert math.isfinite(float(fit.g_lim_50))


def test_reconcile_to_pipeline_meta_r2b_keys():
    report = {
        "g_lim_50": 17.5,
        "g_lim_90": 17.5,
        "g_lim_50_raw_fit": 19.25,
        "g_lim_50_censored": True,
        "g_lim_90_censored": True,
        "g_lim_50_display": ">= 17.5 (censored)",
        "completeness_50_label": "measured to G <= 17.5",
        "match_depth": 18.0,
        "match_depth_source": "MASTERSTAR default",
        "n_missed_below_g90": 5,
        "n_missed_fadezone": 12,
        "fit_method": "fleming1995_erf",
        "completeness_curve": [],
        "n_ref_in_frame": 500,
        "n_gaia_matched": 400,
        "n_gaia_off_frame": 50,
        "n_gaia_below_limit": 30,
        "n_gaia_blended": 10,
        "n_gaia_missed": 17,
        "gaia_dao_completeness_pct": 95.0,
        "n_dao_unmatched": 12,
        "unmatched_dao": {"n_now_matched_to_faint": 3},
        "blend_radius_px": 3.6,
        "methodology": "footprint_reference_fleming1995",
    }
    meta = reconcile_to_pipeline_meta(report)
    assert meta["g_lim_50_censored"] is True
    assert meta["n_missed_below_g90"] == 5
    assert meta["match_depth"] == 18.0
