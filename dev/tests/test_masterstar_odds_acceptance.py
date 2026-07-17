"""MASTERSTAR odds-based acceptance gate (replaces hard recovery-fraction reject)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.wcs import WCS

from vyvar_platesolver import (
    _compute_masterstar_catalog_recovery,
    _masterstar_solve_acceptance,
    _sibling_false_alarm_p,
)


def _make_tan_wcs(ra: float, dec: float, scale_arcsec: float = 2.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [512.0, 512.0]
    w.wcs.crval = [float(ra), float(dec)]
    w.wcs.cd = np.array(
        [
            [-scale_arcsec / 3600.0, 0.0],
            [0.0, scale_arcsec / 3600.0],
        ]
    )
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _grid(ra: float, dec: float, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = _make_tan_wcs(ra, dec)
    xs = np.linspace(50, 970, int(math.sqrt(n)))
    ys = np.linspace(50, 970, int(math.sqrt(n)))
    gx, gy = np.meshgrid(xs, ys)
    ra_c, de_c = w.all_pix2world(gx.ravel(), gy.ravel(), 0)
    px, py = w.all_world2pix(ra_c, de_c, 0)
    return (
        np.asarray(ra_c, dtype=np.float64),
        np.asarray(de_c, dtype=np.float64),
        np.asarray(px, dtype=np.float64),
        np.asarray(py, dtype=np.float64),
    )


def test_draft_406_dense_field_odds_accepts_low_fraction():
    """Dense field: low recovery % but many tight matches - odds accepts; legacy fraction rejects."""
    ra, de, xs, ys = _grid(150.0, 45.0, n=196)
    w = _make_tan_wcs(150.0, 45.0)
    rec = _compute_masterstar_catalog_recovery(
        w, ra, de, xs, ys, naxis1=1024, naxis2=1024, qa_px=15.0, tight_px=2.5
    )
    n_tight = 175
    n_det = 250
    rec["n_matched_tight"] = n_tight
    rec["n_detections_used"] = n_det
    rec["n_cat_in_frame"] = 2838
    rec["catalog_recovery_tight"] = n_tight / 2838.0
    rec["catalog_recovery_tight_gate"] = n_tight / n_det
    rec["quadrants_with_match"] = 4
    area = 1024 * 1024
    p_one = min(1.0, 2838 * math.pi * 2.5**2 / area)
    rec["expected_random"] = float(n_det) * p_one
    rec["false_alarm_p"] = _sibling_false_alarm_p(
        n_tight,
        n_det,
        2838,
        1024,
        1024,
        r_px=2.5,
    )
    odds = _masterstar_solve_acceptance(
        accept_mode="odds",
        catalog_recovery_tight=float(rec["catalog_recovery_tight"]),
        catalog_recovery_tight_gate=float(rec["catalog_recovery_tight_gate"]),
        n_matched_tight=int(rec["n_matched_tight"]),
        n_det=int(rec["n_detections_used"]),
        n_cat_in_frame=2838,
        quadrants_with_match=int(rec.get("quadrants_with_match", 4)),
        expected_random=rec["expected_random"],
        false_alarm_p=rec["false_alarm_p"],
        dist_benign=False,
        centre_rms=5.72,
        edge_rms=1.55,
        recovery_min=0.65,
        matched_floor=30,
        centre_rms_max=1.2,
        hint_sep_deg=0.03,
        hint_sep_limit=0.15,
        fov_diameter_deg=2.0,
        crowded_n_cat_min=800,
    )
    legacy = _masterstar_solve_acceptance(
        accept_mode="fraction",
        catalog_recovery_tight=float(rec["catalog_recovery_tight"]),
        catalog_recovery_tight_gate=float(rec["catalog_recovery_tight_gate"]),
        n_matched_tight=int(rec["n_matched_tight"]),
        dist_benign=False,
        centre_rms=5.72,
        edge_rms=1.55,
        recovery_min=0.65,
        matched_floor=40,
        centre_rms_max=1.2,
        hint_sep_deg=0.03,
        hint_sep_limit=0.15,
        fov_diameter_deg=2.0,
    )
    assert odds["masterstar_verified"] is True
    assert float(rec["catalog_recovery_tight"]) < 0.10
    assert float(rec["catalog_recovery_tight_gate"]) >= 0.70
    assert legacy["masterstar_verified"] is False
    assert odds["quality_flag_primary"] in ("crowded", "blurred", "ok")


def test_wrong_field_odds_rejects():
    """Random/wrong overlap - matches near chance must fail."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 1023, 80)
    ys = rng.uniform(0, 1023, 80)
    w = _make_tan_wcs(10.0, 10.0)
    ra = np.linspace(9.5, 10.5, 400)
    de = np.linspace(9.5, 10.5, 400)
    rec = _compute_masterstar_catalog_recovery(
        w, ra, de, xs, ys, naxis1=1024, naxis2=1024, qa_px=15.0, tight_px=2.5
    )
    res = _masterstar_solve_acceptance(
        accept_mode="odds",
        catalog_recovery_tight=float(rec["catalog_recovery_tight"]),
        catalog_recovery_tight_gate=float(rec["catalog_recovery_tight_gate"]),
        n_matched_tight=int(rec["n_matched_tight"]),
        n_det=int(rec["n_detections_used"]),
        n_cat_in_frame=int(rec["n_cat_in_frame"]),
        quadrants_with_match=int(rec.get("quadrants_with_match", 0)),
        expected_random=rec.get("expected_random"),
        false_alarm_p=rec.get("false_alarm_p"),
        dist_benign=True,
        centre_rms=1.0,
        edge_rms=1.0,
        recovery_min=0.65,
        matched_floor=30,
        centre_rms_max=1.2,
        hint_sep_deg=5.0,
        hint_sep_limit=0.15,
        fov_diameter_deg=2.0,
    )
    assert res["masterstar_verified"] is False


def test_sparse_field_odds_accepts():
    ra, de, xs, ys = _grid(120.0, 30.0, n=100)
    w = _make_tan_wcs(120.0, 30.0)
    rec = _compute_masterstar_catalog_recovery(
        w, ra, de, xs, ys, naxis1=1024, naxis2=1024, qa_px=15.0, tight_px=2.5
    )
    res = _masterstar_solve_acceptance(
        accept_mode="odds",
        catalog_recovery_tight=float(rec["catalog_recovery_tight"]),
        catalog_recovery_tight_gate=float(rec["catalog_recovery_tight_gate"]),
        n_matched_tight=int(rec["n_matched_tight"]),
        n_det=int(rec["n_detections_used"]),
        n_cat_in_frame=int(rec["n_cat_in_frame"]),
        quadrants_with_match=int(rec.get("quadrants_with_match", 0)),
        expected_random=rec.get("expected_random"),
        false_alarm_p=rec.get("false_alarm_p"),
        dist_benign=True,
        centre_rms=0.8,
        edge_rms=0.9,
        recovery_min=0.65,
        matched_floor=30,
        centre_rms_max=1.2,
        hint_sep_deg=0.01,
        hint_sep_limit=0.15,
        fov_diameter_deg=2.0,
    )
    assert res["masterstar_verified"] is True
    assert res["quality_flag_primary"] == "ok"


def test_hint_sep_does_not_reject_odds_verified():
    res = _masterstar_solve_acceptance(
        accept_mode="odds",
        catalog_recovery_tight=0.5,
        catalog_recovery_tight_gate=0.7,
        n_matched_tight=80,
        n_det=100,
        n_cat_in_frame=120,
        quadrants_with_match=4,
        expected_random=2.0,
        false_alarm_p=1e-12,
        dist_benign=True,
        centre_rms=0.9,
        edge_rms=1.0,
        recovery_min=0.65,
        matched_floor=30,
        centre_rms_max=1.2,
        hint_sep_deg=0.5,
        hint_sep_limit=0.15,
        fov_diameter_deg=2.0,
    )
    assert res["masterstar_verified"] is True
    assert res["hint_sep_warn"] is True
    assert res["hint_sep_bad_hard"] is False
