"""Sibling-WCS Pass 2 recovery unit tests (ported from sandbox T1–T5 + invariants)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from config import AppConfig
from vyvar_platesolver import (
    FILTER_EFFECTIVE_WAVELENGTH_NM,
    _sibling_adopt_and_confirm,
    _sibling_apply_bulk_shift_crpix,
    _sibling_best_bulk_shift,
    _sibling_match_metrics,
    _sibling_odds_confirmed,
    filter_code_from_setup_name,
    pick_sibling_donor_filter,
)


def _thresholds(**kw) -> dict[str, float | int]:
    base = {
        "min_matched": 40,
        "rms_max_px": 2.0,
        "min_quadrants": 3,
        "stack_n": 10,
    }
    base.update(kw)
    return base


def _make_tan_wcs(ra: float, dec: float, scale_arcsec: float = 2.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [256.0, 256.0]
    w.wcs.crval = [float(ra), float(dec)]
    w.wcs.cd = np.array(
        [
            [-scale_arcsec / 3600.0, 0.0],
            [0.0, scale_arcsec / 3600.0],
        ]
    )
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _grid_catalog(wcs: WCS, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(40, 470, int(math.sqrt(n)))
    ys = np.linspace(40, 470, int(math.sqrt(n)))
    gx, gy = np.meshgrid(xs, ys)
    ra, de = wcs.all_pix2world(gx.ravel(), gy.ravel(), 0)
    return np.asarray(ra, dtype=np.float64), np.asarray(de, dtype=np.float64)


def _detections_from_wcs(
    wcs: WCS, ra_cat: np.ndarray, de_cat: np.ndarray, *, noise_px: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    px, py = wcs.all_world2pix(ra_cat, de_cat, 0)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    if noise_px > 0:
        rng = np.random.default_rng(42)
        px = px + rng.normal(0, noise_px, size=px.shape)
        py = py + rng.normal(0, noise_px, size=py.shape)
    return px, py


def test_filter_code_from_setup_name():
    assert filter_code_from_setup_name("g_60_4") == "g"
    assert filter_code_from_setup_name("z_90_4") == "z"
    assert filter_code_from_setup_name("") == ""
    assert filter_code_from_setup_name("(root)") == ""


def test_pick_donor_spectral_nearest_for_z():
    verified = {"g", "i", "r"}
    assert pick_sibling_donor_filter("z", verified) == "i"
    lam = FILTER_EFFECTIVE_WAVELENGTH_NM
    assert abs(lam["i"] - lam["z"]) < abs(lam["r"] - lam["z"])


def test_pick_donor_single_available():
    assert pick_sibling_donor_filter("z", {"r"}) == "r"
    assert pick_sibling_donor_filter("z", set()) is None


def test_bulk_shift_improves_offset():
    w_true = _make_tan_wcs(120.0, 30.0, scale_arcsec=2.0)
    ra, de = _grid_catalog(w_true, n=100)
    xs, ys = _detections_from_wcs(w_true, ra, de)
    w_donor = w_true.deepcopy()
    w_donor.wcs.crpix[0] += 1.0
    w_donor.wcs.crpix[1] -= 0.8
    thr = _thresholds()
    before = _sibling_match_metrics(w_donor, ra, de, xs, ys, 512, 512, thresholds=thr)
    w_best, bulk, after = _sibling_best_bulk_shift(
        w_donor, ra, de, xs, ys, 512, 512, thresholds=thr
    )
    assert int(after.get("n_matched_tight") or 0) >= int(before.get("n_matched_tight") or 0)
    if not before.get("confirmed"):
        assert bulk.get("applied") is True or after.get("confirmed")
    assert float(after.get("median_dpx") or 99) <= float(before.get("median_dpx") or 99) + 0.01


def test_t1_far_donor_bulk_shift_confirms():
    """T1 — adopt offset donor WCS; bulk-shift recenters on detections."""
    w_true = _make_tan_wcs(150.0, 45.0, scale_arcsec=1.5)
    ra, de = _grid_catalog(w_true, n=144)
    xs, ys = _detections_from_wcs(w_true, ra, de, noise_px=0.15)
    w_g = w_true.deepcopy()
    w_g.wcs.crpix[0] += 2.0
    w_g.wcs.crpix[1] += 1.5
    res = _sibling_adopt_and_confirm(
        w_g, ra, de, xs, ys, 512, 512, thresholds=_thresholds(min_matched=30)
    )
    assert res["confirmed"] is True
    assert int(res["after"]["n_matched_tight"]) >= 30


def test_t2_order_independent_same_donor():
    """T2 — donor choice depends only on spectral distance, not processing order."""
    verified = {"g", "i", "r"}
    picks = [pick_sibling_donor_filter("z", verified) for _ in range(5)]
    assert len(set(picks)) == 1
    assert picks[0] == "i"


def test_t4_single_donor_used():
    assert pick_sibling_donor_filter("z", {"g"}) == "g"


def test_t5_flip_guard_zero_matches():
    """T5 — flipped WCS yields near-zero matches -> not confirmed."""
    w_true = _make_tan_wcs(100.0, 20.0)
    ra, de = _grid_catalog(w_true, n=64)
    xs, ys = _detections_from_wcs(w_true, ra, de)
    w_flip = w_true.deepcopy()
    w_flip.wcs.crval[0] += 5.0
    w_flip.wcs.crval[1] += 5.0
    res = _sibling_adopt_and_confirm(
        w_flip, ra, de, xs, ys, 512, 512, thresholds=_thresholds()
    )
    assert res["confirmed"] is False
    assert int(res["after"].get("n_matched_tight") or 0) < 10


def test_odds_gate_requires_quadrants_and_false_alarm():
    m_ok = {
        "n_matched_tight": 45,
        "rms_px": 1.2,
        "quadrants_with_match": 3,
        "false_alarm_p": 1e-12,
    }
    assert _sibling_odds_confirmed(m_ok, min_matched=40, rms_max_px=2.0, min_quadrants=3)
    m_bad = dict(m_ok, false_alarm_p=0.5)
    assert not _sibling_odds_confirmed(m_bad, min_matched=40, rms_max_px=2.0, min_quadrants=3)


def test_pass2_disabled_is_noop():
    from pipeline import _pass2_sibling_wcs_recovery

    cfg = AppConfig()
    cfg.masterstar_sibling_recovery_enabled = False
    reports = [{"observation_group_key": "g_60_4"}]
    skipped = [{"gkey": "z_90_4", "setup": "z_90_4"}]
    jobs = [
        {"gkey": "g_60_4", "platesolve_dir": "/x/g", "detrended_root": "/d/g", "files": []},
        {"gkey": "z_90_4", "platesolve_dir": "/x/z", "detrended_root": "/d/z", "files": []},
    ]
    out_rep, out_sk = _pass2_sibling_wcs_recovery(
        reports=list(reports),
        skipped=list(skipped),
        job_list=jobs,
        align_kw={"archive_path": Path("/tmp"), "app_config": cfg},
    )
    assert out_rep == reports
    assert out_sk == skipped


def test_single_filter_job_list_skips_pass2():
    from pipeline import _pass2_sibling_wcs_recovery

    cfg = AppConfig()
    reports: list[dict] = []
    skipped = [{"gkey": "z_90_4", "setup": "z_90_4"}]
    jobs = [{"gkey": "z_90_4", "platesolve_dir": "/x/z", "detrended_root": "/d/z", "files": []}]
    out_rep, out_sk = _pass2_sibling_wcs_recovery(
        reports=reports,
        skipped=skipped,
        job_list=jobs,
        align_kw={"archive_path": Path("/tmp"), "app_config": cfg},
    )
    assert out_sk == skipped


def test_crpix_shift_sign_variants():
    w = _make_tan_wcs(10.0, 10.0)
    w2 = _sibling_apply_bulk_shift_crpix(w, 3.0, -2.0, sx=-1, sy=1)
    assert w2.wcs.crpix[0] == pytest.approx(w.wcs.crpix[0] - 3.0)
    assert w2.wcs.crpix[1] == pytest.approx(w.wcs.crpix[1] - 2.0)


def test_sibling_recovered_header_route():
    """Recovered MASTERSTAR carries VY_CRT=sibling_recovered for skip-solve branch."""
    from pipeline import _has_valid_wcs

    w = _make_tan_wcs(120.0, 30.0)
    hdr = w.to_header()
    hdr["VY_CRT"] = ("sibling_recovered", "test")
    hdr["VY_SODD"] = (46, "test")
    assert str(hdr.get("VY_CRT", "")).strip().lower() == "sibling_recovered"
    assert _has_valid_wcs(hdr)
