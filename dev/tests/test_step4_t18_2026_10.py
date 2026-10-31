"""Step 4 T18 (code review 2): report / export / provenance consistency."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------- G11-36 / NEW-3: texts from constants
def _ctx():
    from citations import RunCitationContext

    return RunCitationContext(annulus_fwhm=(2.7, 5.2), k_grid=(0.6, 1.0), zp_band_label="g")


def test_methods_sky_line_names_the_source_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEW-3: the local-sky line describes SKY-SRC-MASK (sources masked), built from its constants."""
    import sky_estimation
    from citations import build_methods_matrix_lines

    sky = next(ln for ln in build_methods_matrix_lines(_ctx()) if ln.startswith("local sky"))
    assert "sources detected at 5 x MAD sigma" in sky and f"{sky_estimation.SKY_MIN_PIX} sky pixels" in sky
    monkeypatch.setattr(sky_estimation, "SKY_SRC_NSIGMA", 4.0)
    sky = next(ln for ln in build_methods_matrix_lines(_ctx()) if ln.startswith("local sky"))
    assert "detected at 4 x MAD sigma" in sky


def test_variability_and_transient_texts_follow_their_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """G11-36: every number of the variability / transient method texts is its code constant."""
    import transient_field
    import variability_field
    from photometry_report import transient_method_text, variability_method_text

    meta = {"n_frames": 80, "n_stars": 500, "z_threshold": 4.1, "n_gates_ok": 300}
    t = variability_method_text(meta, snr_min=10.0)
    assert "at least 90 %" in t and "probability 1 %" in t and "<= 1 %" in t and "SNR >= 10" in t
    monkeypatch.setattr(variability_field, "COVERAGE_MIN", 0.75)
    monkeypatch.setattr(variability_field, "FAP_FIELD", 0.002)
    t = variability_method_text(meta, snr_min=7.0)
    assert "at least 75 %" in t and "probability 0.2 %" in t and "SNR >= 7" in t
    tm = {"n_frames": 80, "n_candidates": 0, "n_sources": 3}
    t = transient_method_text(tm)
    assert ">= 3 consecutive" in t and "G < 8 " in t and "2.5 FWHM" in t
    monkeypatch.setattr(transient_field, "N_CONSEC_MIN", 4)
    monkeypatch.setattr(transient_field, "BRIGHT_G", 7.5)
    monkeypatch.setattr(variability_field, "BRIGHTENED_NSIGMA", 4.0)
    t = transient_method_text(tm)
    assert ">= 4 consecutive" in t and "G < 7.5 " in t and ">= 4 sigma above Gaia" in t


def test_no_hard_coded_mag_scale_in_report_texts() -> None:
    """G11-36: 2.5/ln10 comes from mag_constants, not 1.0857 literals, in export and report texts."""
    for name in ("export_reports.py", "photometry_report.py", "star_page_m3.py", "citations.py"):
        src = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"1\.0857|1085\.7", src), name
    src = (SRC / "photometry_report.py").read_text(encoding="utf-8")
    assert "< 5 common stars" not in src and "MIN_COMMON_STARS" in src


# ---------------------------------------------------------------- one export plan (G11-03/12/13/20, G11-17)
def _lc(n: int = 20, *, airmass: bool = True, t0: float = 2461316.30) -> pd.DataFrame:
    t = t0 + 0.01 * np.arange(n)
    df = pd.DataFrame({
        "source_file": [f"proc_{i:03d}.csv" for i in range(n)], "jd": t - 0.0005, "bjd": t,
        "time_base": "BJD_TDB", "mag_calib": 12.0 + 0.01 * np.sin(np.arange(n)), "delta_mag": 0.1,
        "err": np.full(n, 0.01), "flag": ["normal"] * n, "aperture_r_px": 5.0,
        "airmass": np.linspace(1.2, 1.4, n) if airmass else np.nan,
    })
    return df


def _export_one(tmp_path: Path, lc: pd.DataFrame, *, obs_group: str = "V_60_2", vsx_type: str = "EA",
                manifest: dict | None = None, cid: str = "123") -> dict:
    from config import AppConfig
    from export_reports import export_lightcurve_reports
    from gaia_bands import filter_band_for_obs_group

    phot = tmp_path / "photometry"
    (phot / "lightcurves").mkdir(parents=True, exist_ok=True)
    lc.to_csv(phot / "lightcurves" / f"lightcurve_{cid}.csv", index=False)
    return export_lightcurve_reports(
        phot / "lightcurves_reports", pd.Series({"vsx_name": f"STAR {cid}", "catalog_id": cid, "vsx_type": vsx_type}),
        lc, pd.DataFrame(), pd.Series({"band_system": filter_band_for_obs_group(obs_group).zp_band, "trust": "GREEN",
                                       "trust_reason": "ok"}),
        cfg=AppConfig(), obs_group=obs_group, lc_dir=phot / "lightcurves", manifest=manifest)


def _data_lines(p: Path) -> list[str]:
    return [ln for ln in p.read_text(encoding="ascii").splitlines() if ln and not ln.startswith("#")]


def test_one_row_rule_for_aavso_varastro_pdf_and_trust(tmp_path: Path) -> None:
    """G11-12 / G11-20: err <= 0, NaN err / time and excluded flags drop the same epochs everywhere:
    AAVSO rows == VarAstro rows == PDF "exported" == trust count."""
    import star_page_m3 as m3
    from trust_flag_core import lc_epoch_counts

    lc = _lc(24)
    lc.loc[2, "err"] = 0.0          # VarAstro accepted it, AAVSO did not
    lc.loc[3, "err"] = -0.01
    lc.loc[4, "err"] = np.nan
    lc.loc[5, "bjd"] = np.nan       # AAVSO wrote it (jd finite), VarAstro did not
    lc.loc[6, "flag"] = "low_snr"   # excluded flag
    lc.loc[7, "flag"] = "high_err"  # kept, named in NOTES
    paths = _export_one(tmp_path, lc)
    n_a, n_v = len(_data_lines(paths["aavso"])), len(_data_lines(paths["varastro"]))
    lc_csv = tmp_path / "photometry" / "lightcurves" / "lightcurve_123.csv"
    facts = m3.lc_facts(lc_csv)
    n_trust, n_all = lc_epoch_counts(tmp_path / "photometry", "123")
    assert n_a == n_v == facts["n_exported"] == n_trust == 24 - 5
    assert facts["n_excluded"] == 5 and n_all == 24 and facts["n_noted"] == 1
    d = m3.lc_plot_data(lc, None, uncalibrated=False)
    assert int(d["good"].sum()) == n_a


def test_export_fact_is_what_the_exporter_wrote(tmp_path: Path) -> None:
    """G11-03: the page's Export fact comes from the manifest - a filter without an AAVSO code writes no AAVSO
    file and the page says so (it said 'AAVSO STD'); a non-EB target gets no VarAstro file."""
    from export_reports import export_fact_text, export_manifest_path
    from target_page import TargetPages

    man: dict = {}
    p1 = _export_one(tmp_path, _lc(), obs_group="I_60_2", manifest=man, cid="1")
    p2 = _export_one(tmp_path, _lc(), obs_group="V_60_2", vsx_type="RRAB", manifest=man, cid="2")
    assert "aavso" not in p1 and "varastro" in p1 and "aavso" in p2 and "varastro" not in p2
    export_manifest_path(tmp_path / "photometry").write_text(json.dumps({"targets": man}), encoding="utf-8")
    pages = TargetPages(tmp_path, summary_df=pd.DataFrame(), at_df=pd.DataFrame(), comp_df=pd.DataFrame(), cfg=None)
    for cid, paths in (("1", p1), ("2", p2)):
        txt = export_fact_text(pages.export_record(cid))
        assert ("AAVSO no" in txt) == ("aavso" not in paths), txt
        assert ("VarAstro no" in txt) == ("varastro" not in paths), txt
    assert "has no AAVSO code" in export_fact_text(pages.export_record("1"))
    assert "not exported" in export_fact_text(pages.export_record("3"))


def _group(tmp_path: Path, lcs: dict[str, pd.DataFrame], comp_quality: dict[str, dict] | None = None,
           comps: dict[str, list[str]] | None = None) -> Path:
    from config import AppConfig
    from export_reports import export_group_reports

    phot = tmp_path / "photometry"
    (phot / "lightcurves").mkdir(parents=True, exist_ok=True)
    for cid, lc in lcs.items():
        lc.to_csv(phot / "lightcurves" / f"lightcurve_{cid}.csv", index=False)
        cq = {"band_system": "V", **((comp_quality or {}).get(cid) or {})}
        (phot / "lightcurves" / f"comp_quality_{cid}.json").write_text(json.dumps(cq), encoding="utf-8")
    at = pd.DataFrame({"catalog_id": list(lcs), "vsx_name": [f"S{c}" for c in lcs], "vsx_type": "RRAB"})
    summ = pd.DataFrame({"catalog_id": list(lcs), "n_frames": 20, "band_system": "V", "trust": "GREEN",
                         "trust_reason": "ok"})
    comp_index = {c: pd.DataFrame({"catalog_id": ids}) for c, ids in (comps or {}).items()}
    export_group_reports(phot, obs_group="V_60_2", cfg=AppConfig(), at_df=at, summary_df=summ, comp_index=comp_index)
    return phot


def test_excluded_comp_not_in_ensemble_without_weights(tmp_path: Path) -> None:
    """G11-13: the batch passes the excluded comps to the exporter; without Broeg weights NCOMP drops them."""
    phot = _group(tmp_path, {"7": _lc()}, comp_quality={"7": {"11": "good", "12": "good", "13": "excluded"}},
                  comps={"7": ["11", "12", "13"]})
    rows = _data_lines(next((phot / "lightcurves_reports" / "aavso").glob("*.txt")))
    assert all("NCOMP=2" in r for r in rows) and "13 V" not in rows[0]
    man = json.loads((phot / "lightcurves_reports" / "export_manifest.json").read_text(encoding="utf-8"))
    assert man["targets"]["7"]["aavso"] and man["targets"]["7"]["n_rows"] == 20


def test_airmass_citation_is_decided_once_per_setup(tmp_path: Path) -> None:
    """G11-17: the first target without airmass gets Kasten & Young like the others (was loop-order dependent)."""
    phot = _group(tmp_path, {"1": _lc(airmass=False), "2": _lc()})
    heads = {p.name: p.read_text(encoding="ascii") for p in (phot / "lightcurves_reports" / "aavso").glob("*.txt")}
    assert len(heads) == 2 and all("Kasten" in h for h in heads.values())


def test_file_date_tag_is_the_earliest_epoch_in_utc(tmp_path: Path) -> None:
    """G11-19: rows not in time order -> the tag is the UTC date of the earliest epoch."""
    lc = _lc(20, t0=2461316.40)  # 2026-10-02 21:36 UT onwards ...
    lc.loc[0, ["jd", "bjd"]] = [2461317.0, 2461317.0005]  # ... first row the next UT day
    paths = _export_one(tmp_path, lc)
    assert "_20261002" in paths["aavso"].name and "_20261002" in paths["varastro"].name
    from export_reports import utc_jd_to_datestr_yyyymmdd

    assert utc_jd_to_datestr_yyyymmdd(2461316.4999) == "20261002"  # UTC, no TDB shift


def test_check_star_numbers_are_one_computation(tmp_path: Path) -> None:
    """G11-11: VarAstro header scatter == PDF / UI fact == plot text == trust value (exported epochs, ddof=1)."""
    from target_page import TargetPages
    from trust_flag_core import check_star_scatter
    import star_page_m3 as m3

    lc = _lc(20)
    lc.loc[3, "flag"] = "saturated"  # this epoch is not exported - its check point is not counted
    kc = "555"
    lcd = tmp_path / "photometry" / "lightcurves"
    lcd.mkdir(parents=True)
    km = 11.0 + 0.01 * np.cos(np.arange(20))
    km[3] = 11.5
    sc = pd.DataFrame({"source_file": lc["source_file"], "bjd": lc["bjd"], "kmag": km, "kmag_err": 0.004})
    sc.to_csv(lcd / "check_kmag_123.csv", index=False)
    (lcd / "comp_quality_123.json").write_text(json.dumps({"band_system": "V", "check_star": {"catalog_id": kc}}))
    paths = _export_one(tmp_path, lc)
    head = paths["varastro"].read_text(encoding="ascii")
    m = re.search(r"scatter ([\d.]+) mag", head)
    expect = float(np.std(np.delete(km, 3), ddof=1))
    assert m and float(m.group(1)) == pytest.approx(expect, abs=6e-5)
    assert check_star_scatter(tmp_path / "photometry", "123")[0] == pytest.approx(expect, abs=1e-12)
    pages = TargetPages(tmp_path, summary_df=pd.DataFrame(), at_df=pd.DataFrame(), comp_df=pd.DataFrame(), cfg=None)
    pages.uncalibrated = lambda cid: False
    assert pages.check_star("123")["scatter"] == f"{expect:.4f} mag"
    d = m3.lc_plot_data(lc, sc, uncalibrated=False)
    assert d["check_text"] == f"check star scatter {expect:.3f} mag" and len(d["kr"]) == 19


def test_comp_weight_column_is_the_notes_weight() -> None:
    """G11-38: the PDF / UI w column is the Broeg weight as written to the AAVSO NOTES (sum 1), not w / w_max."""
    import star_page_m3 as m3

    cq = {"broeg_weights": {"1": 0.5, "2": 0.3, "3": 0.2}}
    rows, _, _ = m3.comp_rows(pd.DataFrame({"catalog_id": ["1", "2", "3"], "mag": 12.0, "bp_rp": 0.8}), cq, band="V")
    assert [r[7] for r in rows] == ["0.500", "0.300", "0.200"]
    assert "sum to 1" in m3.COMP_FOOT


def test_ui_figure_and_pdf_share_the_numbers() -> None:
    """G14-17: the UI figure has no own copy of the export split / check strip / level shift."""
    src = (SRC / "ui_aperture_photometry.py").read_text(encoding="utf-8")
    assert "lc_plot_data(" in src and "EXPORT_EXCLUDE_FLAGS" not in src and "np.nanstd(kr)" not in src
    src = (SRC / "ui_epsf_dashboard.py").read_text(encoding="utf-8")
    assert "psf_level_shift(" in src and 'np.median(both["psf_mag"] - both["ap_mag"])' not in src
    from epsf_lc_compare import psf_level_shift

    df = pd.DataFrame({"ap_ok": [True, True, False], "psf_ok": [True, True, True], "ap_mag": [1.0, 1.1, 9.0],
                       "psf_mag": [1.3, 1.4, 0.0]})
    assert psf_level_shift(df) == pytest.approx(0.3)


# ---------------------------------------------------------------- error model text (G08-32, G11-10, NEW-4)
def test_err_model_lists_exactly_the_combined_terms() -> None:
    """G08-32: the published error model names the four terms combine_production_err_rel adds, no c4."""
    import math

    from sigma_floor_core import ERR_MODEL_TERMS, ERR_MODEL_TEXT, combine_production_err_rel

    assert ERR_MODEL_TERMS == ("photon", "ensemble", "scintillation", "systematic_floor")
    for t in ("err_photon", "err_ensemble", "err_scint", "sigma_sys"):
        assert t in ERR_MODEL_TEXT
    # each term enters in quadrature
    e = combine_production_err_rel(0.003, 1e-9, sigma_sys_mag=0.0, sigma_scint_mag=0.0)
    e2 = combine_production_err_rel(0.003, 1e-9, sigma_sys_mag=0.0, sigma_scint_mag=0.004)
    assert e2 > e and math.isclose(e2 ** 2 - e ** 2, (0.004 / 1.0857362047581294) ** 2, rel_tol=1e-6)
    src = (SRC / "aperture_photometry.py").read_text(encoding="utf-8")
    assert '"c4_correction"' not in src and "ERR_MODEL_TEXT" in src and '"scintillation": {' in src
    for name in ("sigma_floor_core.py", "photometry_lightcurve.py"):
        assert "c4 correction" not in (SRC / name).read_text(encoding="utf-8"), name


def test_floor_sentence_from_the_run_meta() -> None:
    """G11-10: config floor read from dynamic_params; NEW-4: night sentence names the recorded method."""
    from photometry_report import sigma_floor_method_line

    cfg_meta = {"sigma_sys_night": {"applied": "config"}, "dynamic_params": {"sigma_floor": {"sigma_sys_mag": 0.004}}}
    assert "per-rig systematic floor (4.0 mmag" in sigma_floor_method_line(cfg_meta)
    night = {"sigma_sys_night": {"applied": "night", "sigma_sys_mag": 0.0021, "n_comps": 34,
                                 "method": "comps LOO total scatter - (photon + scintillation + reference)"}}
    line = sigma_floor_method_line(night)
    assert "2.1 mmag" in line and "34 comparison stars" in line and "LOO total scatter" in line
    assert "scatter minus photon noise" not in line
    assert sigma_floor_method_line({}) is None


# ---------------------------------------------------------------- names and format (G11-18, G11-37)
def test_ra_dec_seconds_never_reach_60() -> None:
    """G11-37: RA 01:00:59.9999 printed '01:00:60.00'; rounding now carries into the minutes."""
    import star_page_m3 as m3

    assert m3.ra_dec_text(15.0 * (1 + 59.9999 / 3600.0), 10.0 + 59.99 / 3600.0) == "01:01:00.00 +10:01:00.0"
    rng = np.random.default_rng(1)
    for ra, de in zip(rng.uniform(0, 360, 2000), rng.uniform(-90, 90, 2000), strict=True):
        a, d = m3.ra_dec_text(float(ra), float(de)).split()
        assert float(a.split(":")[2]) < 60 and int(a.split(":")[1]) < 60 and int(a.split(":")[0]) < 24
        assert float(d.split(":")[2]) < 60 and int(d.split(":")[1]) < 60
    assert m3.ra_dec_text(320.370026, 51.815449) == "21:21:28.81 +51:48:55.6"
    assert m3.ra_dec_text(10.0, -0.5).endswith("-00:30:00.0")


def test_header_names_transliterated_not_question_marks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """G11-18: observer / site names with diacritics become ASCII letters, not '?'."""
    from config import AppConfig
    from export_reports import ascii_text, export_lightcurve_reports

    assert ascii_text("D\u00e1blice Uhl\u00e1r \u0160t\u011bp\u00e1nek") == "Dablice Uhlar Stepanek"
    phot = tmp_path / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    (phot / "pipeline_meta.json").write_text(json.dumps({"observer_location": {
        "name": "D\u00e1blice", "lat": 50.14, "lon": 14.47, "alt_m": 280.0}}), encoding="utf-8")
    cfg = AppConfig()
    monkeypatch.setattr(cfg, "observer_name", "Milan Uhl\u00e1r", raising=False)
    paths = export_lightcurve_reports(
        phot / "lightcurves_reports", pd.Series({"vsx_name": "STAR", "catalog_id": "1", "vsx_type": "EA"}), _lc(),
        pd.DataFrame(), pd.Series({"band_system": "V", "trust": "GREEN", "trust_reason": "ok"}), cfg=cfg,
        obs_group="V_60_2", lc_dir=phot / "lightcurves")
    head = paths["varastro"].read_text(encoding="ascii")
    assert "Milan Uhlar" in head and "# Site: Dablice" in head and "?" not in head.split("# Site:")[1].split("\n")[0]
    assert "# Site: Dablice" in paths["aavso"].read_text(encoding="ascii")


# ---------------------------------------------------------------- PDF cover / pages (G11-02, G11-35, G11-17 PDF)
def test_variability_cover_rows_from_the_meta(tmp_path: Path) -> None:
    """G11-02: cover counts = variability_meta.json (500 analysed, 3 candidates), no VDI anywhere."""
    from photometry_report import variability_cover_rows

    (tmp_path / "variability_meta.json").write_text(json.dumps(
        {"status": "ok", "n_stars": 500, "n_gates_ok": 420, "n_candidates": 3, "n_known_detected": 2}))
    rows = dict(variability_cover_rows(tmp_path))
    assert rows["Stars analysed (variability)"] == "500" and rows["New variable-star candidates"] == "3"
    assert rows["Known variables recovered"] == "2"
    (tmp_path / "variability_meta.json").write_text(json.dumps({"status": "too few frames"}))
    assert variability_cover_rows(tmp_path) == [("Variability search", "not available (too few frames)")]
    src = (SRC / "photometry_report.py").read_text(encoding="utf-8")
    assert "VDI" not in src.replace("RMS+VDI", "") and "var_results" not in src


def test_targets_without_page_use_the_page_rule(tmp_path: Path) -> None:
    """G11-35: a coverage-RED target gets no page and is counted (was len(summary) - n_lc)."""
    from photometry_report import _PhotometryReportBuilder

    lcd = tmp_path / "photometry" / "lightcurves"
    lcd.mkdir(parents=True)
    good, red = _lc(30), _lc(30)
    red.loc[3:, "flag"] = "low_snr"  # 3 of 30 usable
    good.to_csv(lcd / "lightcurve_1.csv", index=False)
    red.to_csv(lcd / "lightcurve_2.csv", index=False)
    b = _PhotometryReportBuilder.__new__(_PhotometryReportBuilder)
    b.photometry_dir = tmp_path / "photometry"
    b.summary_df = pd.DataFrame({"_cid": ["1", "2", "3"], "catalog_id": ["1", "2", "3"], "n_frames": [30, 30, 0],
                                 "lc_rms": [0.01, 0.02, np.nan]})
    assert [bool(b._no_page_reason(r)) for _, r in b.summary_df.iterrows()] == [False, True, True]
    assert b._n_targets_without_page() == 2
    src = (SRC / "photometry_report.py").read_text(encoding="utf-8")
    assert "use_airmass=setup_uses_airmass(self.lc_dir)" in src  # G11-17: PDF citations like the exports


# ---------------------------------------------------------------- ePSF provenance (G09-14, G09-18, NEW-5)
def test_internal_psf_header_has_no_closed_item() -> None:
    """G09-14: the internal PSF LC header does not cite the closed EPSF-SHAPE-01."""
    import psf_internal_lc as p

    for line in (p.TRUST_NOTE_LINE1, p.TRUST_NOTE_LINE2):
        assert "EPSF-SHAPE" not in line and "untrusted" not in line and "=" not in line


def test_epsf_stage_writes_invariants_and_locus_to_pipeline_meta(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """G09-18: the merge job's invariant records and the night locus reach photometry/pipeline_meta.json."""
    from types import SimpleNamespace

    from epsf_stage import EpsfStagePaths, run_epsf_stage
    from invariants_runtime import inv_check

    def _fake_merge(**kw):
        meta = kw["pipeline_meta"]
        assert isinstance(meta, dict)
        inv_check(meta, "INV-PSF-ADDITIVE-01", True, policy="FAIL", detail="frame=a ok")
        inv_check(meta, "INV-PSF-FRAME-01", False, policy="WARN", detail="3 of 10 frames zero-ok")
        meta["psf_chi2_locus_night"] = {"n_sigma": 5.0, "n_rewritten": 4}
        return {"written": 1, "frames_total": 1, "epsf_job_summary": {}}

    monkeypatch.setattr("epsf_psf_merge.run_epsf_psf_merge_job", _fake_merge)
    phot = tmp_path / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    (phot / "active_targets.csv").write_text("name,vsx_type,catalog_id,zone_flag,catalog,skip_photometry\n"
                                            "T,EW,111,linear,VSX,False\n", encoding="ascii")
    (phot / "comparison_stars_per_target.csv").write_text("catalog_id,target_catalog_id\n222,111\n", encoding="ascii")
    (phot / "lightcurves" / "lightcurve_111.csv").write_text("bjd,delta_mag\n", encoding="ascii")
    (phot / "pipeline_meta.json").write_text(json.dumps({"observer_location": {"name": "x"}}), encoding="utf-8")
    run_epsf_stage(EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path, photometry_dir=phot),
                   cfg=SimpleNamespace(vsx_out_of_scope_types=[]), do_build=False, do_lc=False, draft_id=1)
    meta = json.loads((phot / "pipeline_meta.json").read_text(encoding="utf-8"))
    assert meta["psf_chi2_locus_night"]["n_rewritten"] == 4 and meta["observer_location"]["name"] == "x"
    assert meta["epsf_invariants"]["INV-PSF-ADDITIVE-01"]["n_checked"] == 1
    assert meta["epsf_invariants"]["INV-PSF-FRAME-01"]["failed"] == ["3 of 10 frames zero-ok"]


def test_gui_and_night_run_call_the_epsf_stage_alike() -> None:
    """NEW-5: both callers pass the photometry dir explicitly (the GUI's RUN ePSF is the
    ``task_jobs.task_run_epsf`` job since G14-04); no target selection (S6-FIX EPSF-SCOPE-2026-10)."""
    gui = (SRC / "task_jobs.py").read_text(encoding="utf-8")
    nr = (SRC / "night_run.py").read_text(encoding="utf-8")
    a = gui[gui.index("out = run_epsf_stage("):][:700]
    n = nr[nr.index("epsf_out = run_epsf_stage("):][:900]
    for body in (a, n):
        assert "photometry_dir=" in body and "targets=" not in body and "pipeline_meta" not in body


# ---------------------------------------------------------------- names (G11-22, G11-25)
def test_field_map_names_like_the_exports() -> None:
    """G11-22: the per-target map is titled with target_display_name (NaN vsx_name -> exoplanet host)."""
    import field_map

    titles: list[str] = []

    class _Canvas:
        def save(self, path, title, draw):
            titles.append(title)
            return path

    m = field_map.TargetFieldMap.__new__(field_map.TargetFieldMap)
    m.canvas = _Canvas()
    m.save(Path("x.jpg"), pd.Series({"vsx_name": np.nan, "exo_host_name": "TOI-1234", "catalog_id": "5"}),
           pd.DataFrame())
    assert "TOI-1234" in titles[0] and "nan" not in titles[0]
    src = (SRC / "field_map.py").read_text(encoding="utf-8")
    assert 'target_row.get("vsx_name"' not in src and "_target_display_name" not in src


def test_nan_names_do_not_cite_vsx() -> None:
    """G11-25: [NaN, 'Gaia DR3 1'] has no VSX name (astype(str) made NaN 'nan' -> VSX cited)."""
    from citations import _targets_use_vsx_names

    assert _targets_use_vsx_names(pd.DataFrame({"vsx_name": [np.nan, "Gaia DR3 1"]})) is False
    assert _targets_use_vsx_names(pd.DataFrame({"vsx_name": [np.nan, "1400549806859236864"]})) is False
    assert _targets_use_vsx_names(pd.DataFrame({"vsx_name": [np.nan, "BO CVn"]})) is True
