"""GSPC-EXP-2026-10: experimental GSPC zero-point helper (offline, cache only)."""

from __future__ import annotations

import math

import photometry_core  # noqa: F401  (import order: avoids circular import)
from gspc_zp import CACHE_NAME, fetch_gspc, gspc_mag, gspc_query, read_cache, sigma_c_star

HDR = ("source_id,g,bp_rp,ruwe,c_star,g_sdss_mag,g_sdss_relerr,g_sdss_flag,"
       "bp_n_transits,bp_n_blended_transits,bp_n_contaminated_transits\n")
# rows from the Gaia archive (draft 525 comps), 2026-10-11
ROWS = (
    "2172293994744599040,14.865019,2.278222,0.9969774,-0.0026278496,16.843708,0.0063797487,1,34,0,0\n"
    "2172312342844990976,14.356422,2.282032,1.4008132,0.02644515,16.293331,0.0069977264,1,33,33,0\n"
    "2172294991177191680,11.913214,2.2993708,1.0076019,-0.005092144,13.956538,0.005186415,0,35,0,0\n"
    "2172314129551187456,14.60451,2.257307,2.162606,0.015936017,16.507355,0.0069370554,1,30,0,0\n"
)


def _cache(tmp_path):
    (tmp_path / CACHE_NAME).write_text(HDR + ROWS)
    return read_cache(tmp_path / CACHE_NAME)


def test_accept_and_reject_reasons(tmp_path) -> None:
    c = _cache(tmp_path)
    m, s, why = gspc_mag(c["2172293994744599040"], "g")
    assert why == "" and abs(m - 16.843708) < 1e-9
    # photon error 1.0857*0.00638 = 6.9 mmag (+) 10 mmag floor
    assert abs(s - math.hypot(1.0857 * 0.0063797487, 0.010)) < 1e-12
    assert gspc_mag(c["2172312342844990976"], "g")[2] == "XP spectrum blended"  # 33/33 transits
    assert gspc_mag(c["2172294991177191680"], "g")[2] == "flag 0 (extrapolation)"
    assert gspc_mag(c["2172294991177191680"], "g", flag_policy="any")[2] == ""
    assert gspc_mag(c["2172314129551187456"], "g", ruwe_max=1.4)[2] == "ruwe"
    assert gspc_mag(c["2172314129551187456"], "g")[2] == ""  # ruwe cut is optional


def test_low_sn_rejected(tmp_path) -> None:
    c = _cache(tmp_path)
    r = dict(c["2172293994744599040"], g_sdss_relerr="0.05")  # S/N 20
    assert gspc_mag(r, "g")[2] == "S/N <= 30"


def test_c_star_sigma_riello() -> None:
    # Riello et al. 2021 Eq. 18: c0 + c1 G^m (c0 0.0059898, c1 8.817481e-12, m 7.618399)
    assert abs(sigma_c_star(12.0) - (0.0059898 + 8.817481e-12 * 12.0 ** 7.618399)) < 1e-15
    assert 0.0074 < sigma_c_star(12.0) < 0.0076
    assert sigma_c_star(18.0) > sigma_c_star(15.0) > sigma_c_star(12.0)


def _full_cache(tmp_path):
    from gspc_zp import expected_columns

    cols = expected_columns()
    row = {c: "" for c in cols}
    row.update(source_id="2172293994744599040", g="14.865019", bp_rp="2.278222", g_sdss_mag="16.843708",
               g_sdss_relerr="0.0063797487", g_sdss_flag="1", bp_n_transits="34", bp_n_blended_transits="0")
    (tmp_path / CACHE_NAME).write_text(",".join(cols) + "\n" + ",".join(row[c] for c in cols) + "\n")


def test_fetch_offline_uses_cache_only(tmp_path) -> None:
    _full_cache(tmp_path)
    got = fetch_gspc(["2172293994744599040", "1234"], tmp_path, offline=True)
    assert set(got) == {"2172293994744599040"}


def test_stale_cache_is_not_used(tmp_path) -> None:
    # cache from an older query (fewer columns) must be rebuilt, never appended to (review finding)
    _cache(tmp_path)
    assert fetch_gspc(["2172293994744599040"], tmp_path, offline=True) == {}


def test_query_text() -> None:
    q = gspc_query([1, 2], bands=("g",))
    assert "x.g_sdss_flux_error/x.g_sdss_flux AS g_sdss_relerr" in q
    assert "b_jkc" in gspc_query([1]) and "i_jkc" in gspc_query([1])
    assert "gaiadr3.xp_summary" in q and "IN (1,2)" in q


# --- GSPC-ZP-2026-10: integration pieces ----------------------------------------------------
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src_py"


def test_gspc_zero_point_matches_ensemble_estimator(tmp_path) -> None:
    from gspc_zp import gspc_zero_point

    rows = _cache(tmp_path)
    w = {"2172293994744599040": 0.5, "2172314129551187456": 0.3, "2172294991177191680": 0.2,
         "2172312342844990976": 0.1}
    lv = {"2172293994744599040": -8.0, "2172314129551187456": -8.3, "2172294991177191680": -10.9,
          "2172312342844990976": -8.5}
    r = gspc_zero_point(w, lv, rows, "g")
    # accepted: the two flag-1 unblended comps; flag-0 and blended are rejected
    assert r["n_comp"] == 2
    assert r["rejected"] == {"flag 0 (extrapolation)": 1, "XP spectrum blended": 1}
    exp = (0.5 * (16.843708 + 8.0) + 0.3 * (16.507355 + 8.3)) / 0.8
    assert abs(r["C"] - exp) < 1e-9
    inst = (0.5 * -8.0 + 0.3 * -8.3 + 0.2 * -10.9 + 0.1 * -8.5) / 1.1
    assert abs(r["inst_level"] - inst) < 1e-9
    assert r["sigma_data"] is None  # < 3 comps


def test_gspc_zero_point_too_few(tmp_path) -> None:
    from gspc_zp import gspc_zero_point

    r = gspc_zero_point({"2172293994744599040": 1.0}, {"2172293994744599040": -8.0}, _cache(tmp_path), "g")
    assert r["n_comp"] == 1 and math.isnan(r["C"])


def test_note_texts() -> None:
    from gaia_bands import zero_point_note_text

    gz = {"comp_bp_rp_median": 1.64, "comp_bp_rp_min": 2.13, "comp_bp_rp_max": 2.40,
          "galactic_b_deg": 1.34, "n_comp": 3, "C": 1.0, "sigma_data": 0.017, "diff_vs_std_mag": 0.045}
    a = zero_point_note_text("A", "g", gz)
    assert a.startswith("No comparison star in the ensemble has a valid g magnitude from Gaia DR3 G, BP-RP")
    assert "BP-RP <= 2.0" in a and "BP-RP 2.13-2.40" in a and "ZP=GaiaDR3-GSPC" in a and "MTYPE=DIF" not in a
    a1 = zero_point_note_text("A", "g", dict(gz, n_comp=1))  # too few GSPC comps -> differential
    # REPORT-AUDIT-2026-10: differential -> VarAstro only; the note names why GSPC is missing
    assert "VarAstro only" in a1 and "ZP=GaiaDR3-GSPC" not in a1
    assert "1 comparison star with an accepted Gaia DR3 synthetic g magnitude (at least 2 needed" in a1
    b = zero_point_note_text("B", "g", gz)
    assert "median BP-RP 1.64" in b and "b = +1.3 deg" in b and "+0.045 mag" in b
    assert b.endswith("The differential light curve is not affected.")
    s = zero_point_note_text("B", "g", gz, short=True)
    assert s.startswith("ZPNOTE=red comps BP-RP 1.64 at b=+1.3deg; Gaia->g zero point uncertain")
    assert zero_point_note_text("", "g", gz) == ""
    blue = dict(gz, comp_bp_rp_median=1.1)
    assert "are red" not in zero_point_note_text("B", "g", blue)
    assert zero_point_note_text("B", "g", blue, short=True).startswith("ZPNOTE=comps BP-RP 1.10")


def test_lc_csv_has_mag_gspc(tmp_path) -> None:
    from photometry_lightcurve import save_lightcurve_csv
    import pandas as pd

    n = 3
    z = np.zeros(n)
    p = tmp_path / "lc.csv"
    save_lightcurve_csv(p, bjd=z + 2460000.5, hjd=z, jd=z, airmass=z + 1.2, mag_inst=z, mag_calib=z + 13.0,
                        delta_mag=z + 0.5, err=z + 0.01, aperture_r_px=z + 5.0, flags=["normal"] * n,
                        source_files=["f"] * n, mag_gspc=z + 15.5)
    d = pd.read_csv(p)
    assert list(d["mag_gspc"]) == [15.5, 15.5, 15.5] and list(d["mag_calib"]) == [13.0] * 3


def test_colour_term_default_auto() -> None:
    from config import AppConfig

    assert AppConfig().apply_color_term == "auto"


def test_k2_on_delta_mag_and_mag_calib() -> None:
    """K2-ONE-2026-10 (D4): k'' goes into delta_mag always and into mag_calib when calibrated, so
    mag_calib - delta_mag stays the zero point (and the PSF strip, which subtracts k2_delta_mag)."""
    s = (SRC / "aperture_photometry_target.py").read_text(encoding="utf-8")
    i_cal = s.index('_calibrated = _band_system != "NONE"')
    i_k2 = s.index("delta_mag, _, k2_source_rows = apply_k2_per_frame(")
    assert i_cal < i_k2
    assert "mag_calib, _, _ = apply_k2_per_frame(" in s
    assert "ensemble_bp_rp(comp_bp_rp, comp_weight_map)" in s
    assert s.count('_calibrated = _band_system != "NONE"') == 1


def test_ensemble_colour_is_weighted() -> None:
    from k2_extinction import ensemble_bp_rp

    assert ensemble_bp_rp({"a": 0.5, "b": 1.5, "c": float("nan")}, {"a": 3.0, "b": 1.0, "c": 5.0}) == 0.75


def test_ids_never_via_float() -> None:
    # 19-digit ids above 2**61 are not exact in float64 (review finding)
    big = "5937173300407375616"
    assert str(int(float(big))) != big
    q = gspc_query([big])
    assert big in q
    s = (SRC / "aperture_photometry_state.py").read_text(encoding="utf-8")
    assert "str(int(float(x)))" not in s and "normalize_gaia_source_id" in s


def test_bad_cache_and_empty_flag(tmp_path) -> None:
    (tmp_path / CACHE_NAME).write_text("garbage\nnot,a,cache\n")
    assert read_cache(tmp_path / CACHE_NAME) == {}
    c = _cache(tmp_path)
    r = dict(c["2172293994744599040"], g_sdss_flag="")
    assert gspc_mag(r, "g")[2] == "flag 0 (extrapolation)"  # no exception


def test_c_std_recomputation_equals_ensemble_zero_point() -> None:
    """mag_calib - delta_mag = C + inst_level with C, inst_level rebuilt from the Broeg weights
    and Honeycutt levels exactly as aperture_photometry_target does (basis of mag_gspc)."""
    from photometry_lightcurve import ensemble_normalize

    rng = np.random.default_rng(7)
    n = 50
    z = rng.normal(0, 0.05, n)
    comps = {f"{i}": -9.0 + 0.4 * i + z + rng.normal(0, 0.004 * (i + 1), n) for i in range(5)}
    cat = {"0": 12.1, "1": 12.5, "2": float("nan"), "3": 13.3, "4": 13.7}
    t = -8.5 + z + 0.1 * np.sin(np.arange(n) / 6)
    w, lv = {}, {}
    mc, dm, _ = ensemble_normalize(t, comps, cat, {c: {"quality": "good"} for c in comps},
                                   n_comp_min=3, weights_out=w, levels_out=lv)
    ok = [c for c in w if np.isfinite(cat[c])]
    c_std = sum(w[c] * (cat[c] - lv[c]) for c in ok) / sum(w[c] for c in ok)
    inst = sum(w[c] * lv[c] for c in w) / sum(w.values())
    d = (mc - dm) - (c_std + inst)
    assert np.nanmax(np.abs(d)) < 1e-9
