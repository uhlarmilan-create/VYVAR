"""Fixes 2026-10-11: k2_source rows after epoch exclusion, low_snr flag applied,
single-comparison check star KMAG on the catalogue scale."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import photometry_core  # noqa: F401  (import order: avoids circular import)
from check_star_kmag import compute_check_ensemble_mag_calib
from config import AppConfig

SRC = Path(__file__).resolve().parents[2] / "src_py"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling test helpers


def test_k2_source_is_a_per_target_summary_value() -> None:
    # OUTPUT-REVIEW-2026-10 step 6: k'' provenance moved from every LC row to the summary
    s = (SRC / "phase2a_target.py").read_text(encoding="utf-8")
    assert '"k2_source": k2_source_target' in s
    assert "k2_source=" not in s[s.index("save_lightcurve_csv(\n        lc_csv"):s.index("# EPSF-LC-LOG-01")]


def test_low_snr_applied_after_flagging() -> None:
    s = (SRC / "phase2a_target.py").read_text(encoding="utf-8")
    i_pp = s.index("apply_reporting_postprocess(")
    i_fix = s.index("FIX-LOWSNR-2026-10")
    assert i_fix > i_pp
    assert 'out_flags[_i] = "low_snr"' in s
    from lc_outlier import EXPORT_EXCLUDE_FLAGS

    assert "low_snr" in EXPORT_EXCLUDE_FLAGS


def _single(cat: dict, allow_relative: bool):
    rng = np.random.default_rng(2)
    n = 40
    z = rng.normal(0, 0.05, n)
    lc = {"c1": -9.0 + z + rng.normal(0, 0.004, n), "9": -8.0 + z + rng.normal(0, 0.004, n)}
    return compute_check_ensemble_mag_calib(
        "9", ["c1"], lc, cat, {"c1": {"quality": "good"}},
        cfg=AppConfig(), allow_relative=allow_relative,
    )


def test_single_comp_check_kmag_on_catalogue_scale() -> None:
    r = _single({"c1": 12.0}, allow_relative=False)
    assert r is not None and abs(float(np.nanmedian(r.kmag)) - 13.0) < 0.01  # not ~1.0
    assert _single({"c1": float("nan")}, allow_relative=False) is None
    rel = _single({"c1": float("nan")}, allow_relative=True)
    assert rel is not None and abs(float(np.nanmedian(rel.kmag)) - 1.0) < 0.01


def test_fitsqa_all_nan_qc_does_not_crash() -> None:
    s = (Path(__file__).resolve().parents[2] / "src_py" / "photometry_report.py").read_text(encoding="utf-8")
    assert "if wsum <= 0.0 or not np.isfinite(d).any():" in s


def test_spike_residual_in_mag_units() -> None:
    from lc_outlier import isolated_spike_mask

    n = 41
    m = np.zeros(n)
    m[20] = 0.0475  # 0.0475 mag / (1.0857 * 0.01) = 4.37 sigma < 5 -> no spike (old: 4.75 / ... )
    e = np.full(n, 0.01)  # relative flux error
    sp, r, _ = isolated_spike_mask(m, e, np.arange(n) * 0.003)
    assert not sp[20] and abs(r[20] - 0.0475 / (1.0857362 * 0.01)) < 1e-3
    m[20] = 0.06  # 5.53 sigma -> spike
    assert isolated_spike_mask(m, e, np.arange(n) * 0.003)[0][20]


def test_fitsqa_skips_all_nan_column() -> None:
    import photometry_report as pr

    qc = pd.DataFrame({"FWHM_PX": [3.0, 5.0], "ELONGATION": [1.1, 1.2], "STAR_COUNT": [900, 1000],
                       "SKY_LEVEL": [np.nan, np.nan]})
    hdr = {"VY_FWHM": 5.0, "VY_ELONG": 1.2, "VY_NSTAR": 1000, "VY_QCBG": 300.0}
    r = pr._PhotometryReportBuilder._match_qc_row_by_vy_header_metrics(None, qc, hdr)
    assert r is not None and r["FWHM_PX"] == 5.0


def test_reported_aperture_is_per_target_radius() -> None:
    s = (Path(__file__).resolve().parents[2] / "src_py" / "phase2a_target.py").read_text(encoding="utf-8")
    i = s.index("FIX-APREPORT-2026-10")
    assert "if _pt_active:\n        _measured_ap = float(\"nan\")" in s[i:i + 600]


def test_aperture_tab_is_informational_only() -> None:
    s = (Path(__file__).resolve().parents[2] / "src_py" / "ui_aperture_photometry.py").read_text(encoding="utf-8")
    assert "RUN Aperture Photometry\"" not in s and "run_ui_night_photometry" not in s
    assert "run_phase2a" not in s and "UI-INFO-2026-10" in s


def test_export_rows_differential_and_all_flagged() -> None:
    from export_reports import _select_export_lc_rows

    lc = pd.DataFrame({"jd": [1.0, 2.0, 3.0], "bjd": [1.0, 2.0, 3.0], "mag_calib": [np.nan] * 3,
                       "mag_calib_final": [np.nan] * 3, "delta_mag": [0.4, 0.5, 0.45],
                       "flag": ["normal", "low_snr", "normal"]})
    assert _select_export_lc_rows(lc).empty  # no standard magnitude
    d = _select_export_lc_rows(lc, differential=True)
    assert list(d["mag_calib"]) == [0.4, 0.45]  # delta_mag, low_snr dropped
    lc["flag"] = "low_snr"
    assert _select_export_lc_rows(lc, differential=True).empty  # FIX-EXPORTALLBAD: never export flagged rows


def test_note_a_short_and_offline() -> None:
    from gaia_bands import zero_point_note_text

    gz = {"comp_bp_rp_min": 2.13, "comp_bp_rp_max": 2.40, "status": "unavailable (URLError: timed out)"}
    s = zero_point_note_text("A", "g", gz, short=True)
    assert s.startswith("ZPNOTE=not calibrated (differential)") and s.endswith("GSPC unavailable")
    assert "Gaia archive not reachable: URLError: timed out" in zero_point_note_text("A", "g", gz)
    assert "not reachable" not in zero_point_note_text("A", "g", dict(gz, status="ok (3 of 3)"))


def test_star_page_m3_helpers(tmp_path) -> None:
    import star_page_m3 as m3

    assert m3.target_display_name(pd.Series({"vsx_name": "", "name": "TOI-1234", "exo_host_name": "TOI-1234",
                                             "catalog_id": "5"})) == "TOI-1234"
    assert m3.target_display_name({"vsx_name": "", "name": "5", "catalog_id": "5"}) == "Gaia DR3 5"
    assert m3.short_status({"quality": "suspect", "note": "slope=5.1 mmag/hr (6.8sigma; kept COMP-ADMIT-03)"}) \
        == "suspect* 5.1 mmag/h"
    assert m3.ra_dec_text(320.370026, 51.815449) == "21:21:28.81 +51:48:55.6"
    n = 30
    lc = pd.DataFrame({"bjd": 2461316.3 + np.arange(n) * 0.01, "jd": 2461316.296 + np.arange(n) * 0.01,
                       "delta_mag": np.sin(np.arange(n) / 5), "err": np.full(n, 0.01), "flag": ["normal"] * (n - 1) + ["low_snr"]})
    lc.to_csv(tmp_path / "lc.csv", index=False)
    pd.DataFrame({"bjd": lc.bjd, "kmag": 0.9 + 0.01 * np.cos(np.arange(n))}).to_csv(tmp_path / "k.csv", index=False)
    assert m3.plot_lc_check(tmp_path / "lc.csv", tmp_path / "o.png", check_csv=tmp_path / "k.csv", uncalibrated=True).is_file()
    f = m3.lc_facts(tmp_path / "lc.csv")
    assert f["n_total"] == n and f["n_flagged"] == 1 and f["date_ut"] == "2026-10-02"


def test_gspc_standard_rule() -> None:
    from gspc_zp import gspc_standard_available

    gz = {"C": 24.8, "n_comp": 3}
    assert gspc_standard_available(gz) and gspc_standard_available(gz, [np.nan, 15.5])
    assert not gspc_standard_available(gz, [np.nan, np.nan])  # LC without GSPC values -> differential
    assert not gspc_standard_available(dict(gz, n_comp=1)) and not gspc_standard_available({"C": None, "n_comp": 3})


def test_trust_coverage_and_blend() -> None:
    from trust_flag_core import CompTrustThresholds, classify_warnings, coverage_insufficient

    assert coverage_insufficient(5, 122) and coverage_insufficient(23, 122) and not coverage_insufficient(25, 122)
    assert not coverage_insufficient(2, 2) and coverage_insufficient(4, 10)
    th = CompTrustThresholds.from_bounds(3, 8)
    kw = dict(n_clean=8, check_scatter=0.01, lc_quality="good", thresholds=th, n_check=100, check_min_epochs=5)
    h, s = classify_warnings(n_frames=122, n_export=5, **kw)
    assert any("insufficient data: 5 of 122" in x for x in h)
    h, s = classify_warnings(n_frames=122, n_export=50, **kw)
    assert not h and any("only 50 of 122" in x for x in s)
    h, s = classify_warnings(n_frames=122, n_export=122, blend_frac=0.12, **kw)
    assert any("blended" in x for x in h)


def test_comp_smart_replaces_suspects() -> None:
    from comp_smart import smart_select

    rng = np.random.default_rng(3)
    n = 120
    bjd = 2461160.3 + np.arange(n) * (6.0 / 24 / n)  # 6 h
    common = rng.normal(0, 0.01, n)
    lc = {f"P{i}": 10.0 + common + rng.normal(0, 0.003, n) for i in range(8)}
    for c in ("P4", "P7"):  # night-long trends 30 mmag/h -> suspect
        lc[c] = lc[c] + 0.030 * (bjd - bjd[0]) * 24
    for i in range(4):
        lc[f"R{i}"] = 10.5 + common + rng.normal(0, 0.003, n)
    kw = dict(max_comp_p2p=0.1, max_comp_slope_mmag_hr=5.0, comp_slope_significance_k=3.0,
              common_mode_detrend=True)
    prim = [f"P{i}" for i in range(8)]
    final, log = smart_select(prim, [f"R{i}" for i in range(4)], lc, {c: bjd for c in lc},
                              n_comp_min=3, n_comp_max=8, stability_kwargs=kw)
    assert set(log["replaced"]) == {"P4", "P7"} and log["added"] == ["R0", "R1"] and len(final) == 8
    assert not log["kept_suspect"]
    # only 2 clean in total -> the least suspect is kept to reach 3, marked
    lc2 = {c: lc[c] + (0.03 if c not in ("P0", "P1") else 0) * (bjd - bjd[0]) * 24 * (1 + int(c[1]) / 10) for c in prim}
    final2, log2 = smart_select(prim, [], lc2, {c: bjd for c in lc2}, n_comp_min=3, n_comp_max=8, stability_kwargs=kw)
    assert len(final2) == 3 and len(log2["kept_suspect"]) == 1


def test_hrd_table_fits_page_and_wraps(tmp_path) -> None:
    """HRD-PAGE-2026-10: Paragraph cells, widths sum to USE_W, d falls back to GSP-Phot."""
    from test_hrd_details import _minimal_report_builder

    b = _minimal_report_builder(tmp_path)
    rows = []
    for i in range(17):
        rows.append({"category": "Hot luminous (likely) candidate", "catalog_id": f"15946{i:014d}",
                     "simbad_id": "V* VERY LONG SIMBAD NAME 123456 (EclBin); likely candidate note",
                     "sp_type_raw": "K2III+F5V", "mag_g": "12.34", "abs_mag_g": "-1.20", "bp_rp": "1.234",
                     "teff": "4500", "logg": "2.10", "dist_pc": "\u2014" if i % 2 else "512.3",
                     "dist_gspphot": "830", "src": "gaia+simbad", "_empty_field": False})
    top = pd.DataFrame(rows)
    data, widths = b._hrd_table_rows(top)
    assert abs(sum(widths) - b.USE_W) < 1e-6
    assert len(data) == 18 and len(data[0]) == len(widths)
    txt = [p.text for p in data[2]]
    assert "830 (GSP)" in txt and "512.3" in [p.text for p in data[1]]
    t = b.Table(data, colWidths=widths, repeatRows=1)
    w, h = t.wrap(b.USE_W, 10 * b.PAGE_H)
    assert w <= b.USE_W + 1e-6
    for row in data:
        for p, cw in zip(row, widths):
            pw, _ph = p.wrap(cw - 4, 9999)
            assert pw <= cw - 4 + 1e-6


def test_check_star_report_facts(tmp_path) -> None:
    """CHECK-FACTS-2026-10: full id, G, BP-RP, KMAG meas vs catalogue, scatter vs expected."""
    import json

    from target_page import TargetPages

    b = TargetPages(tmp_path, summary_df=pd.DataFrame(), at_df=pd.DataFrame(), comp_df=pd.DataFrame(), cfg=None)
    b.lc_dir.mkdir(parents=True, exist_ok=True)
    kc = "1593035088421948672"
    pd.DataFrame({"check_catalog_id": [kc] * 4, "kmag": [10.19, 10.20, 10.21, 10.20],
                  "kmag_err": [0.004] * 4}).to_csv(b.lc_dir / "check_kmag_77.csv", index=False)
    (b.lc_dir / "comp_quality_77.json").write_text(json.dumps({"comp_band_mag": {kc: 10.180}}))
    pd.DataFrame({"catalog_id": [kc], "phot_g_mean_mag": [10.51], "bp_rp": [0.88]}).to_csv(
        b.platesolve_dir / "masterstars_full_match.csv", index=False)
    b.uncalibrated = lambda cid: False
    out = b.check_star("77")
    assert out["kcid"] == kc and out["g"] == "10.51" and out["bp_rp"] == "0.88"
    assert out["kmag"] == "10.200" and out["kref"] == "10.180" and out["kdelta"] == "+0.020"
    assert out["err_med"] == "0.0040" and out["scatter_ratio"] == "1.8"
    b.uncalibrated = lambda cid: True
    b._chk.clear()
    out2 = b.check_star("77")
    assert out2["kref"] == "" and "not calibrated" in out2["kmag"]


def test_psf_local_magnitude_matched(tmp_path) -> None:
    """PSF-LOCAL-2026-10: faint normal stars are not 'broad' (moment bias cancels with
    same-magnitude references); a broad faint star is scaled."""
    from astropy.io import fits

    from aperture_pertarget import psf_scale_for_targets

    rng = np.random.default_rng(3)
    H = W = 900
    img = rng.normal(1000.0, 12.0, (H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    sig = 3.0 / 2.3548
    refs = []
    k = 0
    for mag in np.repeat(np.arange(9.0, 14.5, 0.5), 12):
        x, y = 60 + (k % 26) * 30, 60 + (k // 26) * 30
        k += 1
        amp = 30000.0 * 10 ** (-0.4 * (mag - 9.0))
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))
        refs.append((float(x), float(y), float(mag)))
    # targets: a normal faint star (one of the refs) and a broad faint star (1.6x sigma)
    xb, yb = 800.0, 820.0
    amp_b = 30000.0 * 10 ** (-0.4 * 4.5) / 1.6 ** 2
    img += amp_b * np.exp(-((xx - xb) ** 2 + (yy - yb) ** 2) / (2 * (1.6 * sig) ** 2))
    p = tmp_path / "ms.fits"
    fits.writeto(p, img.astype(np.float32))
    xn, yn, mn = refs[-3]
    scale, ratio = psf_scale_for_targets(
        p, {"normal": (xn, yn), "broad": (xb, yb)}, refs, 3.0,
        target_mag={"normal": mn, "broad": 13.5})
    assert "normal" not in scale and abs(ratio["normal"] - 1.0) < 0.15
    assert "broad" in scale and ratio["broad"] > 1.3
    # no references of the target's magnitude -> not judged
    s2, r2 = psf_scale_for_targets(p, {"broad": (xb, yb)}, refs, 3.0, target_mag={"broad": 18.0})
    assert s2 == {} and r2 == {}


def test_psf_scaled_grid_measures_flux(tmp_path) -> None:
    """PSF-LOCAL-2026-10: a non-round scale (1.329475) is measured (was NaN: rounding mismatch)
    and the larger aperture collects more of a broad star's flux."""
    from astropy.io import fits

    from aperture_pertarget import measure_night_grid, series_from_grid

    H = W = 120
    yy, xx = np.mgrid[0:H, 0:W]
    sig = 1.6 * 3.0 / 2.3548
    img = 1000.0 + 5000.0 * np.exp(-((xx - 60.3) ** 2 + (yy - 60.7) ** 2) / (2 * sig ** 2))
    hdr = fits.Header()
    hdr["VY_FWHM"] = 3.0
    fits.writeto(tmp_path / "f1.fits", img.astype(np.float32), hdr)
    pd.DataFrame({"catalog_id": ["111"], "x": [60.3], "y": [60.7]}).to_csv(tmp_path / "proc_f1.csv", index=False)
    g0 = measure_night_grid(frames_dir=tmp_path, catalog_ids=["111"], f_grid=[1.0], fwhm_night_px=3.0)
    g1 = measure_night_grid(frames_dir=tmp_path, catalog_ids=["111"], f_grid=[1.0], fwhm_night_px=3.0,
                            scale_by_cid={"111": 1.329475})
    f0 = series_from_grid(g0, "111", 1.0, ["f1"])[0]
    f1 = series_from_grid(g1, "111", 1.0, ["f1"])[0]
    assert np.isfinite(f0) and np.isfinite(f1) and f1 > 1.1 * f0


def test_comp_smart_verifies_final_set(monkeypatch) -> None:
    """COMP-SMART-2026-10: a comp good among all candidates but suspect in the final set is
    replaced (the per-target stability test runs on the final set)."""
    import photometry_lightcurve

    from comp_smart import smart_select

    def fake(comp_lc, **_kw):
        ids = list(comp_lc)
        return {c: ({"quality": "suspect", "slope_mmag_hr": 7.0} if (c == "B" and len(ids) <= 3) else {"quality": "good"})
                for c in ids}

    monkeypatch.setattr(photometry_lightcurve, "check_comparison_stability", fake)
    lc = {c: np.zeros(5) for c in "ABCD"}
    final, log = smart_select(["A", "B", "C"], ["D"], lc, {c: np.arange(5.0) for c in lc},
                              n_comp_min=2, n_comp_max=3, stability_kwargs={})
    assert final == ["A", "C", "D"] and log["replaced"] == ["B"] and log["added"] == ["D"]
    assert "final set" in log["verdict"]["B"] and not log["kept_suspect"]
    # no reserve left and already at n_comp_min -> kept and marked
    final2, log2 = smart_select(["A", "B"], [], lc, {c: np.arange(5.0) for c in lc},
                                n_comp_min=2, n_comp_max=3, stability_kwargs={})
    assert final2 == ["A", "B"] and log2["kept_suspect"] == ["B"]


def test_reserve_does_not_change_primary_comps() -> None:
    """COMP-SMART-2026-10 (review): the admitted comps are those of the unchanged n_comp_max
    selection; the reserve is a separate, larger selection minus the primary set."""
    from test_comp_determinism_synthetic import _TARGET_ID, _build_synthetic_dataset

    from photometry_comp import select_comparison_stars_per_target
    from phase01_run import _split_comp_reserve

    target, ms, paths, cache = _build_synthetic_dataset()
    ms = ms.copy()
    ms["bp_rp"] = [0.95 + 0.006 * ((i * 7) % 30) for i in range(len(ms))]  # colour spread: ladder matters
    target = ms.loc[ms["catalog_id"] == _TARGET_ID].iloc[0]
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    kw = dict(csv_cache=cache, cfg=cfg, chip_fw=800, chip_fh=600, chip_interior_margin_px=0,
              min_dist_arcsec=30.0, n_comp_max=5,
              variable_target_catalog_ids=frozenset({_TARGET_ID}))
    base = select_comparison_stars_per_target(target, ms, paths, **kw)
    both = select_comparison_stars_per_target(target, ms, paths, n_reserve=8, **kw)
    prim, resv = _split_comp_reserve(both)
    assert list(prim["catalog_id"]) == list(base["catalog_id"])
    pd.testing.assert_frame_equal(prim.reset_index(drop=True), base.reset_index(drop=True), check_like=True)
    assert len(resv) > 0 and not set(resv["catalog_id"]) & set(prim["catalog_id"])
    assert list(resv["reserve_rank"]) == sorted(resv["reserve_rank"])


def test_hrd_unavailable_no_blank_page(tmp_path) -> None:
    """FIX-BLANKPAGE-2026-10: a section that starts on a fresh page must not emit an empty page."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader

    from test_hrd_details import _minimal_report_builder

    b = _minimal_report_builder(tmp_path)
    out = tmp_path / "x.pdf"
    c = canvas.Canvas(str(out), pagesize=landscape(A4))
    c.drawString(50, 50, "previous section")
    c.showPage()  # previous section ended its page
    b._report_hrd_unavailable_page(c, "test")
    c.save()
    pages = PdfReader(str(out)).pages
    assert len(pages) == 2 and all((p.extract_text() or "").strip() for p in pages)


def test_night_sigma_sys_from_comps_recovers_floor() -> None:
    """SIGMA-SYS-NIGHT-2026-10: synthetic comps with photon noise + 4 mmag white systematic +
    a common transparency drift -> floor ~4 mmag; without systematic -> ~0."""
    from sigma_floor_core import night_sigma_sys_from_comps

    rng = np.random.default_rng(11)
    nf, nc, g, rn = 120, 25, 2.0, 5.0
    fw = np.full(nf, 3.0)
    k, area = 1.25, np.pi * (1.25 * 3.0) ** 2
    common = 0.3 * np.sin(np.linspace(0, 3, nf))  # transparency, cancels in the ensemble
    for sys_mmag in (4.0, 0.0):
        flux, sky = {}, {}
        for i in range(nc):
            F0 = 2e5 * 10 ** (-0.4 * 0.1 * i)
            n_sky = np.pi * ((5.2 * 3.0) ** 2 - (2.7 * 3.0) ** 2)
            var = F0 / g + area * (1 + area / n_sky) * (200.0 / g + (rn / g) ** 2)
            ph = 1.0857 * np.sqrt(var) / F0
            m = -2.5 * np.log10(F0) + common + rng.normal(0, ph, nf) + rng.normal(0, sys_mmag / 1000, nf)
            flux[str(i)] = 10 ** (-0.4 * m); sky[str(i)] = np.full(nf, 200.0)
        r = night_sigma_sys_from_comps(flux, sky, fw, k=k, gain=g, read_noise=rn,
                                       annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2)
        assert r["n_comps"] >= 5
        assert abs(1000 * r["sigma_sys_mag"] - sys_mmag) < 1.2, r
