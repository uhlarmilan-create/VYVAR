"""Unit tests for Gaia<->DAO reconciliation (R-2 footprint) against the one frame depth (FRAME-DEPTH-2026-10)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from dao_reconcile import ReferencePopulationMismatch, apply_footprint_filter, bin_completeness_curve, blend_radius_px, check_reference_population_consistency, classify_dao_only_dataframe, completeness_50_pct, ConfirmableDepthResult, DAO_ONLY_CLASS_AMBIGUOUS_DEPTH, DAO_ONLY_CLASS_ARTIFACT_NEGATIVE, DAO_ONLY_CLASS_BEYOND_CATALOGUE, DAO_ONLY_CLASS_INDETERMINATE, DAO_ONLY_CLASS_UNMATCHED_IN_RANGE, derive_confirmable_depth_g, decompose_reference_population, fit_fleming_completeness, fit_instrumental_flux_to_g, fleming_completeness, frame_depth_limits, FluxToGFitResult, format_dao_only_census_log, reconcile_to_pipeline_meta, sigma_g_row, split_missed_by_g90


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
        "methodology": "footprint_reference_frame_depth",
    }
    meta = reconcile_to_pipeline_meta(report)
    assert meta["g_lim_50"] == 14.2
    assert meta["g_lim_90"] == 15.1
    assert meta["n_gaia_off_frame"] == 50
    assert meta["fit_method"] == "fleming1995_erf"
    assert meta["n_dao_matched_to_faint"] == 3


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


def test_frame_depth_limits_one_depth():
    lim = frame_depth_limits({
        "g50": 13.88, "sigma_mag": 0.57, "g_lim_frame": 15.59, "g_lim": 15.57, "db_depth_g": 17.5,
        "method": "completeness_fit", "censored": False,
    })
    assert lim["g_lim_50"] == pytest.approx(13.88)
    assert lim["g_lim_90"] == pytest.approx(13.88 - 1.2815516 * 0.57, abs=1e-6)
    assert lim["catalogue_depth_g"] == 17.5 and lim["match_depth"] == 15.57
    assert lim["censored"] is False and "detector" not in lim["g_lim_50_display"]
    cen = frame_depth_limits({"g50": 18.2, "g90": 17.6, "sigma_mag": 0.4, "censored": True,
                              "method": "detector_estimate", "db_depth_g": 17.5, "g_lim": 17.5})
    assert cen["g_lim_90"] == 17.6 and cen["censored"] is True
    assert "detector estimate" in cen["g_lim_90_display"]
    empty = frame_depth_limits(None)
    assert empty["g_lim_50"] is None and empty["catalogue_depth_g"] is None


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
        "g_lim_50_censored": True,
        "g_lim_90_censored": True,
        "g_lim_50_display": ">= 17.5 (censored)",
        "completeness_50_label": "measured to G <= 17.5",
        "match_depth": 17.5,
        "match_depth_source": "frame_depth (detector_estimate)",
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
        "methodology": "footprint_reference_frame_depth",
    }
    meta = reconcile_to_pipeline_meta(report)
    assert meta["g_lim_50_censored"] is True
    assert meta["n_missed_below_g90"] == 5
    assert meta["match_depth"] == 17.5
    assert "g_lim_50_raw_fit" not in meta and "g_lim_est" not in meta


def test_fit_instrumental_flux_to_g_median_zp():
    matched = pd.DataFrame(
        {
            "flux": [1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
            "phot_g_mean_mag": [12.0, 11.25, 10.5, 9.75, 9.0],
        }
    )
    fit = fit_instrumental_flux_to_g(matched)
    assert fit.ok
    assert fit.zp is not None
    assert fit.n_matched == 5
    assert fit.residual_mad is not None
    assert fit.residual_mad < 0.01


def test_derive_confirmable_depth_g_min_winner():
    depth = derive_confirmable_depth_g(catalogue_depth_g=17.5, matching_depth_g=15.57)
    assert depth.depth_resolvable is True
    assert depth.confirmable_depth_g == pytest.approx(15.57)
    assert depth.winning_input == "matching_depth_g"
    depth = derive_confirmable_depth_g(catalogue_depth_g=17.5, matching_depth_g=17.5)
    assert depth.confirmable_depth_g == pytest.approx(17.5)


def test_derive_confirmable_depth_unresolvable_without_catalogue_depth():
    depth = derive_confirmable_depth_g(catalogue_depth_g=None, matching_depth_g=18.0)
    assert depth.depth_resolvable is False
    assert depth.confirmable_depth_g is None


def test_sigma_g_row_uses_rms_not_fleming():
    sg = sigma_g_row(zp_residual_rms=0.431, snr=50.0)
    assert sg == pytest.approx(math.hypot(0.431, 1.0857362 / 50.0), rel=1e-6)
    assert sg < 0.5


def test_dao_only_classification_buckets():
    depth = ConfirmableDepthResult(17.5, "catalogue_depth_g", {"catalogue_depth_g": 17.5}, True)
    fit = FluxToGFitResult(True, 22.0, 10, 0.05, 0.431, "median_zp")
    df = pd.DataFrame(
        {
            "source_type": ["DAO_ONLY"] * 5,
            "flux": [-1.0, 50000.0, 10.0, 50000.0, 50000.0],
            "flux_ap": [-1.0, 50000.0, 10.0, 50000.0, 50000.0],
            "peak_max_adu": [1.0, 500.0, 500.0, 50.0, 500.0],
            "likely_saturated": [False, False, False, False, False],
            "edge_safe_10px": [True, True, True, True, True],
        }
    )
    out, meta = classify_dao_only_dataframe(
        df,
        depth=depth,
        flux_fit=fit,
        frame_noise_adu=10.0,
    )
    classes = list(out["dao_only_class"])
    assert classes[0] == DAO_ONLY_CLASS_ARTIFACT_NEGATIVE
    assert DAO_ONLY_CLASS_UNMATCHED_IN_RANGE in classes
    assert DAO_ONLY_CLASS_BEYOND_CATALOGUE in classes
    assert meta["confirmable_depth_g"] == pytest.approx(17.5)


def test_dao_only_depth_unresolvable_all_indeterminate():
    from dao_reconcile import annotate_dao_only_magnitude_classes

    df = pd.DataFrame(
        {
            "source_type": ["DAO_ONLY", "GAIA_MATCHED"],
            "flux": [100.0, 1000.0],
            "flux_ap": [100.0, 1000.0],
            "phot_g_mean_mag": [np.nan, 12.0],
            "catalog_id": ["", "123"],
        }
    )
    out, meta = annotate_dao_only_magnitude_classes(
        df, gaia_db_path="/fake/path", frame_depth={"g_lim": 15.0}, gaia_db_identity={}
    )
    assert meta["depth_resolvable"] is False
    assert out.loc[out["source_type"] == "DAO_ONLY", "dao_only_class"].iloc[0] == DAO_ONLY_CLASS_INDETERMINATE


def test_format_dao_only_census_log():
    meta = {
        "n_dao_only": 10,
        "confirmable_depth_g": 17.5,
        "confirmable_depth_winner": "catalogue_depth_g",
        "depth_resolvable": True,
        "counts": {
            DAO_ONLY_CLASS_ARTIFACT_NEGATIVE: 2,
            DAO_ONLY_CLASS_UNMATCHED_IN_RANGE: 5,
            DAO_ONLY_CLASS_AMBIGUOUS_DEPTH: 1,
            DAO_ONLY_CLASS_BEYOND_CATALOGUE: 1,
            DAO_ONLY_CLASS_INDETERMINATE: 1,
        },
    }
    line = format_dao_only_census_log(meta, n_total=100)
    assert "artifact_negative=2" in line
    assert "unmatched_in_range=5" in line
    assert "confirmable_depth G=17.50" in line
    assert "informational, not a gate" in line


def test_census_log_includes_unmeasurable_and_zp_rms():
    meta = {
        "confirmable_depth_g": 17.5,
        "confirmable_depth_winner": "catalogue_depth_g",
        "depth_resolvable": True,
        "counts": {DAO_ONLY_CLASS_UNMATCHED_IN_RANGE: 3},
        "n_dao_only": 3,
        "flux_fit": {"residual_rms": 0.837},
        "sigma_g_unmeasurable_fraction": 0.5,
        "sigma_g_unmeasurable_threshold_mag": 1.0,
    }
    line = format_dao_only_census_log(meta, n_total=50)
    assert "flux-to-G RMS=0.837 mag" in line
    assert "unmeasurable(sigma_g>1.0)=0.500" in line


def test_reconcile_to_pipeline_meta_includes_dao_only_class():
    report = {
        "g_lim_50": 14.2,
        "n_dao_unmatched": 12,
        "completeness_curve": [],
        "dao_only_class_meta": {
            "counts": {
                "artifact_negative": 2,
                "unmatched_in_range": 8,
                "ambiguous_depth": 1,
                "beyond_catalogue": 0,
                "indeterminate": 1,
            },
            "confirmable_depth_g": 17.5,
            "confirmable_depth_winner": "catalogue_depth_g",
            "catalogue_depth_g": 17.5,
            "flux_fit": {"zp": 22.0, "residual_mad": 0.05, "residual_rms": 0.431},
            "gaia_db_identity": {"fingerprint_sha256": "abc", "row_count": 100},
        },
    }
    meta = reconcile_to_pipeline_meta(report)
    assert meta["dao_only_n_artifact_negative"] == 2
    assert meta["dao_only_n_unmatched_in_range"] == 8
    assert meta["confirmable_depth_g"] == 17.5
    assert meta["catalogue_depth_g"] == 17.5


@pytest.mark.slow
def test_draft501_artifact_negative_count():
    """draft_501: artifact_negative DAO_ONLY rows == DAO_ONLY rows with non-positive aperture flux (G04-24)."""
    from config import AppConfig
    from pathlib import Path

    cfg = AppConfig()
    csv_path = (
        Path(cfg.archive_root)
        / "Drafts"
        / "draft_000501"
        / "platesolve"
        / "V_60_2"
        / "masterstars_full_match.csv"
    )
    if not csv_path.is_file():
        pytest.skip(f"missing {csv_path}")
    from dao_reconcile import annotate_dao_only_magnitude_classes

    df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
    if "source_type" not in df.columns:
        cid = df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        df["source_type"] = pd.Series(["DAO_ONLY"] * len(df)).where(cid.eq(""), "GAIA_MATCHED")
    _out, meta = annotate_dao_only_magnitude_classes(
        df,
        gaia_db_path=str(cfg.gaia_db_path),
        # FRAME-DEPTH-2026-10 signature (was effective_match_depth / cone_query_mag_limit = 18)
        frame_depth={"g_lim": 18.0, "db_depth_g": 18.0},
        frame_noise_adu=10.0,
    )
    n_neg = int((meta.get("counts") or {}).get(DAO_ONLY_CLASS_ARTIFACT_NEGATIVE, 0))
    # G04-24: the class follows the aperture flux (was 142 on the DAO flux sign)
    _dao = df["source_type"].astype(str).eq("DAO_ONLY")
    assert n_neg == int((pd.to_numeric(df.loc[_dao, "flux_ap"], errors="coerce") <= 0).sum())


def test_membership_rows_are_not_detections():
    """Catalogue-membership rows carry a Gaia id but were not detected by DAO: they never count
    as matched in the completeness decomposition (they made the final catalogue 100 % complete)."""
    from dao_reconcile import _detected_catalog_id_set

    det = pd.DataFrame({
        "catalog_id": ["1", "2", "3", ""],
        "source_state": ["DETECTED_P1", "DETECTED_P2", "catalog_membership", "DAO_ONLY"],
        "x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0],
    })
    assert _detected_catalog_id_set(det) == {"1", "2"}
    assert _detected_catalog_id_set(det.drop(columns=["source_state"])) == {"1", "2", "3"}
