"""Step 7 (T17) comparison-star items, Milan 2026-10-31: G07-02, G07-26, G08-41, G08-46."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import AppConfig
from photometry_comp import COMP_SIGMA_REF_COL, _select_comps_by_rms_then_color


def _pool(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "catalog_id": [str(1000 + i) for i in range(n)],
        "comp_rms": rng.uniform(0.003, 0.012, n),
        COMP_SIGMA_REF_COL: np.full(n, 0.004),
        "bp_rp": rng.uniform(0.3, 2.0, n),
        "_dist_deg": rng.uniform(0, 0.3, n),
        "_nn_dist_fwhm": np.full(n, 10.0),
    })


# --- G07-02: one floor for both branches -------------------------------------------------------

def test_g0702_admitted_count_is_zero_or_in_range_for_any_target_colour():
    cfg = AppConfig()
    nmin, nmax = int(cfg.phase01_comparison_n_comp_min), int(cfg.phase01_comparison_n_comp_max)
    rng = np.random.default_rng(1)
    seen_nan_admitted = False
    for _ in range(200):
        df = _pool(rng, int(rng.integers(1, 12)))
        for tb in (float(rng.uniform(0.3, 2.0)), float("nan")):
            s = _select_comps_by_rms_then_color(df, tb, nmin, nmax, 0.6, cfg=cfg)
            n = len(s)
            assert n == 0 or nmin <= n <= nmax, (n, tb, len(df))
            if n == 0:
                assert s.attrs.get("no_comps_reason") == "too_few_comps"
            if not np.isfinite(tb) and n > 0:
                seen_nan_admitted = True
                assert s.attrs.get("colour_unknown") is True
    assert seen_nan_admitted


def test_g0702_nan_colour_uses_whole_pool_ranked_by_comp_rms():
    cfg = AppConfig()
    rng = np.random.default_rng(3)
    df = _pool(rng, 12)
    s = _select_comps_by_rms_then_color(df, float("nan"), 3, 8, 0.6, cfg=cfg)
    want = df.sort_values("comp_rms")["catalog_id"].head(8).tolist()
    assert s["catalog_id"].tolist() == want
    assert s["_delta_bprp_abs"].isna().all()  # |dC| not defined without a target colour


def test_g0702_nan_colour_with_two_comps_is_no_comps(caplog):
    cfg = AppConfig()
    df = _pool(np.random.default_rng(5), 2)
    with caplog.at_level(logging.WARNING):
        s = _select_comps_by_rms_then_color(df, float("nan"), 3, 8, 0.6, cfg=cfg)
    assert s.empty and s.attrs["no_comps_reason"] == "too_few_comps"
    assert "too_few_comps" in caplog.text


def test_g0702_sel_note_and_trust_soft_note():
    from comp_selection_per_target import _color_rms_sel_note
    from trust_flag_core import TARGET_COLOUR_UNKNOWN_SOFT, CompTrustThresholds, classify_warnings

    cfg = AppConfig()
    df = _pool(np.random.default_rng(7), 6)
    s = _select_comps_by_rms_then_color(df, float("nan"), 3, 8, 0.6, cfg=cfg)
    assert _color_rms_sel_note(s, cfg) == "colour_unknown"
    th = CompTrustThresholds.from_bounds(5, 8)
    _, soft = classify_warnings(n_clean=6, check_scatter=float("nan"), lc_quality="ok", thresholds=th,
                                n_check=50, target_colour_unknown=True)
    assert TARGET_COLOUR_UNKNOWN_SOFT in soft
    _, soft0 = classify_warnings(n_clean=6, check_scatter=float("nan"), lc_quality="ok", thresholds=th,
                                 n_check=50)
    assert TARGET_COLOUR_UNKNOWN_SOFT not in soft0


# --- G07-26: one comp S/N gate -----------------------------------------------------------------

def _snr_pool() -> pd.DataFrame:
    return pd.DataFrame({"catalog_id": ["15", "25"], "snr_ap_pixscaled": [15.0, 25.0]})


def test_g0726_comp_snr_gate_is_max_of_snr_min_and_20():
    from photometry_comp import apply_comp_quality_gates, comp_snr_gate

    assert comp_snr_gate(10.0) == 20.0 and comp_snr_gate(30.0) == 30.0
    out10 = apply_comp_quality_gates(_snr_pool(), comp_peak_sat_frac=0.8, snr_min=10.0)
    assert out10["catalog_id"].tolist() == ["25"]
    out30 = apply_comp_quality_gates(_snr_pool(), comp_peak_sat_frac=0.8, snr_min=30.0)
    assert out30.empty


def test_g0726_comp_snr_gate_fails_loud_without_column():
    import pytest

    from photometry_comp import apply_comp_quality_gates

    with pytest.raises(ValueError, match="snr_ap_pixscaled"):
        apply_comp_quality_gates(pd.DataFrame({"catalog_id": ["1"]}), comp_peak_sat_frac=0.8, snr_min=10.0)


def test_g0726_d3_does_not_cut_on_snr():
    from d3_comparison_candidacy import apply_d3_comparison_candidacy

    df = pd.DataFrame({"source_state": ["DETECTED_P1"], "vy_identity_gate": ["pass"],
                       "gaia_dao_resid_px": [0.1], "snr_ap_pixscaled": [5.0]})
    m, _ = apply_d3_comparison_candidacy(df, fwhm_dao_px=2.0, solve_rms_px=None)
    assert m.tolist() == [True]


def test_g0726_ens_to_target_noise_is_median_ratio():
    from aperture_photometry_target import ens_to_target_noise

    sem = np.array([0.003, 0.004, 0.020, np.nan])
    ph = np.array([0.003, 0.003, 0.010, 0.01])
    sc = np.array([0.004, 0.0, 0.0, 0.0])
    r = ens_to_target_noise(sem, ph, sc)
    assert abs(r - float(np.median([0.003 / 0.005, 0.004 / 0.003, 2.0]))) < 1e-12
    assert np.isnan(ens_to_target_noise([np.nan], [0.01], [0.0]))


# --- G08-41: one clean independent check star suffices ------------------------------------------

def test_g0841_one_independent_clean_candidate_is_the_check_star():
    from check_star_kmag import select_check_star

    ens = [str(100 + i) for i in range(8)]
    rows = [{"catalog_id": c, "status": "good", "comp_rms": 0.002 + 0.0001 * i, "comp_tier": 1}
            for i, c in enumerate(ens)]
    rows.append({"catalog_id": "999", "status": "good", "comp_rms": 0.01, "comp_tier": 1})
    row = select_check_star(pd.DataFrame(rows), ensemble_ids=set(ens), cfg=AppConfig())
    assert row is not None and str(row["catalog_id"]) == "999"


def test_g0841_no_independent_candidate_gives_none():
    from check_star_kmag import select_check_star

    rows = [{"catalog_id": str(100 + i), "status": "good", "comp_rms": 0.002, "comp_tier": 1} for i in range(3)]
    assert select_check_star(pd.DataFrame(rows), ensemble_ids={"100", "101", "102"}, cfg=AppConfig()) is None


# --- G08-46: colour-term coefficient uncertainty is a zero-point term ---------------------------

def _ct_data(seed: int, n: int = 300, noise: float = 0.02):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.3, 2.0, n)
    y = 0.149 * (c - 1.0) + rng.normal(0, noise, n)
    return rng, c, y


def test_g0846_covariance_sigma_matches_bootstrap():
    from color_transform import _robust_poly, fit_colour_term

    rng, c, y = _ct_data(11)
    f = fit_colour_term(c, y, max_order=1)
    assert f is not None and f.order == 1 and f.cov is not None
    se = f.coeff_stderr()[0]
    boot = []
    for _ in range(300):
        i = rng.integers(0, len(c), len(c))
        boot.append(_robust_poly(c[i] - f.c_ref, y[i], 1)[0][1])
    assert abs(se / float(np.std(boot)) - 1.0) < 0.2


def test_g0846_correction_sigma_order1_and_colour_term():
    from aperture_photometry_target import ct_zero_point_sigma
    from color_transform import fit_colour_term

    _, c, y = _ct_data(12)
    f = fit_colour_term(c, y, max_order=1)
    cc, ww = [0.6, 0.9, 1.4], [1.0, 2.0, 1.0]
    xt = 1.3 - f.c_ref
    xbar = sum(w * (ci - f.c_ref) for ci, w in zip(cc, ww)) / sum(ww)
    s = f.correction_sigma(1.3, cc, ww)
    assert abs(s - abs(xt - xbar) * f.coeff_stderr()[0]) < 1e-12
    bp = {"a": 0.6, "b": 0.9, "c": 1.4}
    cat = {"a": 12.0, "b": 12.5, "c": 13.0}
    w = {"a": 1.0, "b": 2.0, "c": 1.0}
    r0 = ct_zero_point_sigma(f, 1.3, float("nan"), "unavailable: test", bp, cat, w)
    assert r0["sigma_zp_ct_colour_mag"] is None and r0["target_bp_rp_err_source"] == "unavailable: test"
    assert abs(r0["sigma_zp_ct_mag"] - s) < 1e-6
    r1 = ct_zero_point_sigma(f, 1.3, 0.03, "gaia_db:bp_rp_error", bp, cat, w)
    assert abs(r1["sigma_zp_ct_colour_mag"] - abs(f.coeffs[0]) * 0.03) < 1e-6
    assert abs(r1["sigma_zp_ct_mag"] - float(np.hypot(s, abs(f.coeffs[0]) * 0.03))) < 1e-6


def test_g0846_per_point_error_has_no_colour_term_input():
    import inspect

    from aperture_photometry_target import _tgt_time_and_errors

    params = inspect.signature(_tgt_time_and_errors).parameters
    assert "c1_stderr" not in params and "ct_corr" not in params


def test_g0846_gaia_bp_rp_error_unavailable_or_from_flux_errors(tmp_path):
    import math
    import sqlite3

    from catalog_db import query_local_gaia_bp_rp_error

    db = tmp_path / "g.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, bp_rp REAL, g_flux_error_rel REAL)")
    con.execute("INSERT INTO gaia_dr3 VALUES (123456789012345, 1.0, 0.001)")
    con.commit()
    con.close()
    v, src = query_local_gaia_bp_rp_error(db, "123456789012345")
    assert math.isnan(v) and src.startswith("unavailable")
    db2 = tmp_path / "g2.db"
    con = sqlite3.connect(db2)
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, bp_rp REAL, "
                "bp_flux_error_rel REAL, rp_flux_error_rel REAL)")
    con.execute("INSERT INTO gaia_dr3 VALUES (123456789012345, 1.0, 0.003, 0.004)")
    con.commit()
    con.close()
    v2, src2 = query_local_gaia_bp_rp_error(db2, "123456789012345")
    assert src2 == "gaia_db:bp_rp_flux_error_rel" and abs(v2 - 1.0857362047581294 * 0.005) < 1e-9
