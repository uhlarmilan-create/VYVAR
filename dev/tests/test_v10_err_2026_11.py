"""VYVAR 1.0 closing items, tag err (Milan approved 2026-11-04).

Item 2 - LC-PRECISION-ERR-2026-11: the headline precision is the per-point error incl. the night floor (MERR);
         the point-to-point scatter is a secondary, labelled short-term number.
Item 7 - CHECK-ROBUST-2026-11: the check-star scatter is 1.4826 x MAD (one computation), not the standard deviation.
Item 8 - K2-RELIABLE-2026-11: k'' only with a cited value and slope, and only when the colour difference is known.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src_py", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------------------- item 7
def _check_series(n: int = 60, sig: float = 0.01, seed: int = 3) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    km = 11.0 + rng.normal(0.0, sig, n)
    km[17] += 0.5  # one bad frame (cosmic ray / satellite / cloud edge)
    return km, [f"f{i:03d}" for i in range(n)]


def test_item7_robust_sigma_is_mad_scale_no_clipping():
    from stats_core import MAD_TO_SIGMA, robust_sigma

    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert robust_sigma(v) == pytest.approx(MAD_TO_SIGMA * 1.0)
    assert np.isnan(robust_sigma([1.0])) and np.isnan(robust_sigma([np.nan, 2.0]))
    rng = np.random.default_rng(0)
    assert robust_sigma(rng.normal(0, 0.02, 20000)) == pytest.approx(0.02, rel=0.03)


def test_item7_check_star_scatter_robust_to_one_outlier_frame():
    """A check star with one large outlier frame: robust scatter ~ Gaussian sigma, std inflated (old value)."""
    from check_star_kmag import check_star_stats

    sig = 0.01
    km, sf = _check_series(sig=sig)
    sc = pd.DataFrame({"source_file": sf, "bjd": np.arange(km.size, dtype=float), "kmag": km, "kmag_err": sig})
    st = check_star_stats(sc, sf)
    old = float(np.std(km, ddof=1))
    assert old > 5.0 * sig  # the old ddof=1 std: one frame makes the check look 6x worse
    assert st.scatter == pytest.approx(sig, rel=0.25)
    assert st.n == km.size


def test_item7_trust_decision_follows_robust_scatter(tmp_path: Path):
    """Trust fallback (no kmag_err): absolute 0.02 / 0.05 mag limits. One outlier frame no longer makes RED."""
    from trust_flag_core import _CHECK_HARD_LO, check_star_scatter

    sig = 0.01
    km, sf = _check_series(sig=sig)
    lcd = tmp_path / "photometry" / "lightcurves"
    lcd.mkdir(parents=True)
    pd.DataFrame({"source_file": sf, "bjd": np.arange(km.size, dtype=float), "kmag": km}).to_csv(
        lcd / "check_kmag_123.csv", index=False)
    pd.DataFrame({"source_file": sf, "jd": np.arange(km.size, dtype=float), "bjd": np.arange(km.size, dtype=float),
                  "mag_calib": 12.0, "delta_mag": 0.1, "err": 0.005, "flag": "normal"}).to_csv(
        lcd / "lightcurve_123.csv", index=False)
    (lcd / "comp_quality_123.json").write_text(json.dumps({"band_system": "V", "check_star": {"catalog_id": "9"}}))
    chk, n = check_star_scatter(tmp_path / "photometry", "123")
    assert n == km.size and chk < 0.02
    assert float(np.std(km, ddof=1)) >= _CHECK_HARD_LO  # the old std crossed the hard limit
    from trust_flag_core import CompTrustThresholds, classify_warnings

    kw = dict(n_clean=8, lc_quality="good", thresholds=CompTrustThresholds.from_bounds(5, 8), n_frames=km.size,
              n_check=n, check_min_epochs=5)
    hard, soft = classify_warnings(check_scatter=chk, **kw)
    assert not [w for w in hard + soft if "check-star scatter" in w]
    hard_old, _ = classify_warnings(check_scatter=float(np.std(km, ddof=1)), **kw)
    assert [w for w in hard_old if "check-star scatter" in w]  # what the old std decided


def test_item7_check_ratio_uses_the_same_robust_scale():
    """check_star_chi2's ratio is robust_sigma of the normalised residuals (one implementation)."""
    import inspect

    import trust_flag_core

    src = inspect.getsource(trust_flag_core.check_star_chi2)
    assert "robust_sigma(z)" in src and "np.median(np.abs(z" not in src


# ---------------------------------------------------------------------------------------- item 2
def test_item2_target_page_headline_is_the_error_p2p_secondary():
    """Bright star: err incl. night floor 0.010, p2p 0.004 -> headline row carries the error, p2p is second and
    labelled as short-term scatter with its ratio."""
    from target_page import PRECISION_LABEL, SHORT_SCATTER_LABEL, precision_facts

    rows = precision_facts(0.010, 0.004)
    assert [a for a, _ in rows] == [PRECISION_LABEL, SHORT_SCATTER_LABEL]
    assert "error" in PRECISION_LABEL.lower() and "short-term" in SHORT_SCATTER_LABEL.lower()
    assert rows[0][1].startswith("0.0100 mag") and "night floor" in rows[0][1]
    assert rows[1][1].startswith("0.0040 mag frame-to-frame") and "0.40 x error" in rows[1][1]
    assert precision_facts(float("nan"), 0.004)[0][1] == "-"
    assert precision_facts(0.01, float("nan"))[1][1] == "-"


def test_item2_target_page_facts_use_precision_facts():
    """The PDF and UI facts list takes the two rows from precision_facts (no 'Precision (p2p)' headline)."""
    import inspect

    import target_page

    src = inspect.getsource(target_page.TargetPages)
    assert "*precision_facts(_merr, _p2p)" in src and "Precision (p2p)" not in src


def test_item2_report_run_summary_and_table_order():
    from photometry_report import (RUN_PRECISION_LABEL, RUN_SHORT_SCATTER_LABEL, SUMMARY_TABLE_HEADER,
                                   SUMMARY_TABLE_TAIL, median_target_error, run_precision_rows)

    rows = run_precision_rows(0.012, 0.005)
    assert rows == [(RUN_PRECISION_LABEL, "0.0120"), (RUN_SHORT_SCATTER_LABEL, "0.0050")]
    assert "error" in RUN_PRECISION_LABEL.lower() and "short-term" in RUN_SHORT_SCATTER_LABEL.lower()
    assert SUMMARY_TABLE_TAIL.index("merr_med") < SUMMARY_TABLE_TAIL.index("lc_precision_p2p")
    assert SUMMARY_TABLE_HEADER == {"merr_med": "MERR", "lc_precision_p2p": "p2p"}
    st = {"1": {"merr_med": 0.01}, "2": {"merr_med": 0.03}, "3": {"merr_med": float("nan")}, "9": {"merr_med": 9.0}}
    assert median_target_error(st, ["1", "2", "3"]) == pytest.approx(0.02)
    assert np.isnan(median_target_error(st, ["3", "4"]))


# ---------------------------------------------------------------------------------------- item 8
@pytest.mark.parametrize("tok", ["u", "U", "SU", "sloan_u", "u'"])
def test_item8_u_and_johnson_u_have_no_k2(tok):
    from k2_extinction import K2_REASON_NO_SLOPE_U, k2_for_token

    val, reason = k2_for_token(tok)
    assert val is None and reason == K2_REASON_NO_SLOPE_U and "no citable colour slope" in reason


@pytest.mark.parametrize("tok,cited", [
    ("g", "Smith et al. 2002"), ("r", "Smith et al. 2002"), ("B", "AAVSO PEP Guide 3.5.4"),
])
def test_item8_applied_bands_carry_a_cited_value_and_slope(tok, cited):
    from k2_extinction import k2_for_token

    val, reason = k2_for_token(tok)
    assert val is not None and val < 0 and cited in reason and "Jordi 2010" in reason


@pytest.mark.parametrize("tok,word", [
    ("i", "no citable colour slope"), ("z", "no citable colour slope"), ("V", "negligible"), ("R", "negligible"),
    ("I", "negligible"), ("CV", "uncorrected"), ("TG", "uncorrected"), ("NoFilter", "uncorrected"),
    ("G", "ambiguous"), ("J", "no literature k''"), ("H", "no literature k''"),
])
def test_item8_none_bands_have_a_reason(tok, word):
    from k2_extinction import resolve_k2

    v, src, reason = resolve_k2(None, f"{tok}_60_2")
    assert src.value == "none" and np.isnan(v) and reason.startswith("none - ")
    if word == "ambiguous":
        assert "ambiguous" in reason or "not a standard band" in reason
    else:
        assert word in reason


def test_item8_every_standard_band_resolves_without_error():
    """Every band of the one filter table has a k'' rule (cited or none with a reason); no KeyError."""
    import band_classify
    from k2_extinction import k2_for_token

    names = set(band_classify._SINGLE_LETTER_BAND) | set(band_classify._BAND_NAMES)
    for n in names:
        val, reason = k2_for_token(n)
        assert reason and (val is None) == reason.startswith("none - "), n


def test_item8_colour_reference_rules():
    from k2_extinction import K2_REASON_NO_WEIGHTS, K2_REASON_TARGET_COLOUR, k2_colour_reference

    col = {"a": 0.8, "b": 1.0, "c": 1.2}
    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    ref, f, why = k2_colour_reference(0.9, col, w)
    assert why == "" and f == 0.0 and ref == pytest.approx(0.5 * 0.8 + 0.3 * 1.0 + 0.2 * 1.2)
    ref, f, why = k2_colour_reference(float("nan"), col, w)  # target colour unknown
    assert np.isnan(ref) and why == K2_REASON_TARGET_COLOUR
    ref, f, why = k2_colour_reference(0.9, {"a": 0.8, "b": 1.0, "c": float("nan")}, w)  # one comp colourless
    assert np.isnan(ref) and f == pytest.approx(0.2) and "20.0 %" in why and why.startswith("none - ")
    ref, f, why = k2_colour_reference(0.9, col, {})
    assert np.isnan(ref) and why == K2_REASON_NO_WEIGHTS


def _run_tgt_k2(target_bp: float, comp_bp: list[float], monkeypatch, *, k2: float = -0.0169):
    """Drive aperture_photometry_target._tgt_k2 (the night-run / GUI k'' step) with a fixed airmass 1..2."""
    from types import SimpleNamespace

    import aperture_photometry_target as apt
    import photometry_lightcurve

    n = 10
    am = np.linspace(1.0, 2.0, n)
    monkeypatch.setattr(photometry_lightcurve, "target_airmass", lambda *a, **k: am)
    state = SimpleNamespace(k2_bprp=k2, k2_source="literature_default", k2_reason="cited band rule",
                            site_ok=False, site_lat=0.0, site_lon=0.0, site_alt=0.0, target_time_grid=None)
    ids = ["100", "200"]
    comps = pd.DataFrame({"catalog_id": ids, "bp_rp": comp_bp})
    frames = pd.DataFrame({"catalog_id": "1", "jd": 2460000.0 + np.arange(n) / 100.0, "airmass": am})
    out = apt._tgt_k2(_calibrated=True, _cfg=None, all_frames=frames, comp_weight_map={"100": 0.5, "200": 0.5},
                      delta_mag=np.zeros(n), mag_calib=np.full(n, 12.0), state=state,
                      target_bp_rp_by_cid={"1": target_bp}, target_cid="1", target_comps=comps,
                      target_row={"ra_deg": 10.0, "dec_deg": 20.0})
    (_am, _cbp, delta_mag, k2_coeff, k2_ref, wfrac, reason, k2_src, k2_val, mag_calib, _tbp) = out
    return am, delta_mag, mag_calib, k2_coeff, k2_ref, wfrac, reason, k2_src, k2_val


def test_item8_target_step_applies_with_known_colours(monkeypatch):
    am, dm, mc, coeff, ref, wfrac, reason, src, val = _run_tgt_k2(1.5, [0.8, 1.0], monkeypatch)
    assert src == "literature_default" and reason == "cited band rule" and wfrac == 0.0
    assert ref == pytest.approx(0.9) and coeff == pytest.approx(-0.0169 * 0.6)
    assert np.allclose(dm, 0.0169 * 0.6 * am) and np.allclose(mc - dm, 12.0)


@pytest.mark.parametrize("target_bp,comp_bp,word", [
    (float("nan"), [0.8, 1.0], "target BP-RP unknown"),
    (1.5, [0.8, float("nan")], "50.0 % of the ensemble weight on comps without BP-RP"),
])
def test_item8_target_step_not_applied_without_colour(monkeypatch, caplog, target_bp, comp_bp, word):
    import logging

    with caplog.at_level(logging.WARNING):
        am, dm, mc, coeff, ref, wfrac, reason, src, val = _run_tgt_k2(target_bp, comp_bp, monkeypatch)
    assert src == "none" and np.all(dm == 0.0) and np.all(mc == 12.0)
    assert np.isnan(coeff) and np.isnan(ref) and np.isnan(val)
    assert word in reason and reason.startswith("none - ")
    assert any(word in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_item8_g_slope_is_the_cited_jordi_table5_row():
    """Jordi et al. 2010 Table 5 GBP-GRP(g-r) = 0.3482 + 1.3463 x - 0.0310 x^2 + 0.0067 x^3: the slope at the
    FGK anchor g-r = 0.48 is 1 / P'(0.48) = 0.757 (the former 1.054 came from coefficients not in the paper)."""
    from k2_extinction import SLOPE_BV_PER_BPRP, SLOPE_GR_PER_BPRP, SLOPE_RI_PER_BPRP, k2_for_token

    def p(x, a, b, c, d):
        return a + b * x + c * x * x + d * x ** 3

    h = 1e-5
    for slope, row, x0 in ((SLOPE_GR_PER_BPRP, (0.3482, 1.3463, -0.0310, 0.0067), 0.48),
                           (SLOPE_RI_PER_BPRP, (0.4649, 2.3969, -0.2976, 0.0207), 0.15),
                           (SLOPE_BV_PER_BPRP, (0.0981, 1.4290, -0.0269, 0.0061), 0.58)):
        numeric = 2 * h / (p(x0 + h, *row) - p(x0 - h, *row))
        assert slope == pytest.approx(numeric, rel=1e-6)
    assert SLOPE_GR_PER_BPRP == pytest.approx(0.757, abs=5e-4)
    assert k2_for_token("g")[0] == pytest.approx(-0.016 * SLOPE_GR_PER_BPRP) and "Table 5" in k2_for_token("g")[1]
