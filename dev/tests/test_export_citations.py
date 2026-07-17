"""Tests for export header citations and observer location lines."""

from __future__ import annotations

from config import AppConfig
from citations import build_run_citation_context, load_citations_bib
from export_reports import (
    _append_aavso_observer_location_lines,
    _vyvar_export_citation_lines,
)


def _citation_text(cfg: AppConfig | None = None, **kw) -> str:
    return "".join(_vyvar_export_citation_lines(cfg, **kw))


def _aavso_header_snippet(cfg: AppConfig) -> str:
    lines: list[str] = ["#TYPE=Extended\n", "#OBSCODE=TEST\n"]
    _append_aavso_observer_location_lines(lines, cfg)
    return "".join(lines)


def test_citations_bib_has_required_keys() -> None:
    bib = load_citations_bib()
    required = {
        "broeg2005",
        "collins2017",
        "honeycutt1992",
        "howell1989",
        "stetson1987",
        "gaia2023",
        "lindegren2021",
        "eastman2010",
        "watson2006",
        "riello2021",
        "anderson2000",
        "moffat1969",
        "astier2013",
        "lacroix2025",
        "guy2010",
        "mighell1999",
        "pont2006",
        "tamuz2005",
        "savitzky1964",
        "aigrain2004",
        "hippke2024",
        "marconi2026",
        "vanderplas2018",
        "lomb1976",
        "scargle1982",
        "kovacs2002",
        "stellingwerf1978",
        "seager2003",
        "ciardi2015",
        "photutils",
        "astropy2022",
        "astroquery2019",
        "numpy2020",
        "scipy2020",
        "lightkurve2018",
        "henden_kaitchuck1982",
        "aavso_ccd_guide",
        "sokolovsky2017",
        "vonneumann1941",
        "barbary2016",
        "bertin1996",
    }
    missing = required - set(bib)
    assert not missing, f"Missing CITATIONS.bib keys: {sorted(missing)}"


def test_citation_lines_core_always_present() -> None:
    cfg = AppConfig()
    text = _citation_text(cfg)
    assert "Broeg" in text
    assert "Collins" in text or "AstroImageJ" in text
    assert "Howell" in text
    assert "Honeycutt" in text
    assert text.count("Honeycutt") == 1
    assert "Stetson" in text
    assert "Gaia Collaboration" in text
    assert "Lindegren" in text
    assert "Eastman" in text
    assert "photutils" in text
    assert "Astropy Collaboration" in text
    assert "Ginsburg" in text or "astroquery" in text
    assert "Harris" in text or "NumPy" in text
    assert "Virtanen" in text or "SciPy" in text


def test_citation_lines_vsx_when_configured() -> None:
    from pathlib import Path

    cfg = AppConfig()
    vsx = Path(__file__).resolve().parents[2] / "VSX" / "vyvar_vsx_local.db"
    if not vsx.is_file():
        return
    cfg.vsx_local_db_path = str(vsx)
    text = _citation_text(cfg)
    assert "Watson" in text


def test_citation_lines_psf_conditional() -> None:
    off = AppConfig()
    off.psf_photometry_enabled = False
    off.psf_adaptive_enabled = False
    on = AppConfig()
    on.psf_photometry_enabled = True
    assert "Anderson" not in _citation_text(off)
    assert "Moffat" not in _citation_text(off)
    assert "Anderson" in _citation_text(on)
    assert "Moffat" in _citation_text(on)
    assert "Astier" in _citation_text(on)
    assert "Lacroix" in _citation_text(on)


def test_citation_lines_sysrem_conditional() -> None:
    off = AppConfig()
    off.sysrem_enabled = False
    on = AppConfig()
    on.sysrem_enabled = True
    assert "Tamuz" not in _citation_text(off)
    assert "Tamuz" in _citation_text(on)


def test_citation_lines_data_quality_gate_conditional() -> None:
    off = AppConfig()
    off.comp_qa_enabled = False
    off.trust_flag_enabled = False
    on = AppConfig()
    text_off = _citation_text(off)
    text_on = _citation_text(on)
    assert "DATA-QUALITY GATE" not in text_off
    assert "Sokolovsky" not in text_off
    assert "Barbary" not in text_off
    assert "DATA-QUALITY GATE" in text_on
    assert "Sokolovsky" in text_on
    assert "von Neumann" in text_on
    assert "Barbary" not in text_on
    assert "Bertin" not in text_on


def test_citation_lines_common_mode_stability_detrend_no_duplicate_honeycutt() -> None:
    cfg = AppConfig()
    off = build_run_citation_context(cfg, pipeline_meta={"common_mode_stability_detrend": False})
    on = build_run_citation_context(cfg, pipeline_meta={"common_mode_stability_detrend": True})
    off_text = "".join(_vyvar_export_citation_lines(cfg, run_ctx=off))
    on_text = "".join(_vyvar_export_citation_lines(cfg, run_ctx=on))
    assert "Honeycutt" in off_text
    assert "Honeycutt" in on_text
    assert off_text.count("Honeycutt") == 1
    assert on_text.count("Honeycutt") == 1


def test_citation_lines_iterative_comp_clip_conditional() -> None:
    cfg_off = AppConfig()
    cfg_off.comp_sparse_fallback_enabled = False
    cfg_off.comp_iterative_clip_enabled = False
    off = build_run_citation_context(cfg_off, pipeline_meta={"comp_sparse_fallback_used": False})
    on_cfg = AppConfig()
    on_cfg.comp_sparse_fallback_enabled = True
    on = build_run_citation_context(on_cfg, pipeline_meta={"comp_sparse_fallback_used": True})
    off_text = "".join(_vyvar_export_citation_lines(cfg_off, run_ctx=off))
    on_text = "".join(_vyvar_export_citation_lines(on_cfg, run_ctx=on))
    assert "Gilliland" not in off_text
    assert "Gilliland" in on_text
    assert "Burdanov" in on_text


def test_citation_lines_democratic_conditional() -> None:
    off = AppConfig()
    off.democratic_detrend_enabled = False
    on = AppConfig()
    on.democratic_detrend_enabled = True
    assert "Hippke" not in _citation_text(off)
    assert "Hippke" in _citation_text(on)


def test_citation_lines_period_conditional() -> None:
    cfg = AppConfig()
    off = build_run_citation_context(cfg, period_analysis=False)
    on = build_run_citation_context(cfg, period_analysis=True)
    assert "VanderPlas" not in "".join(_vyvar_export_citation_lines(cfg, run_ctx=off))
    assert "VanderPlas" in "".join(_vyvar_export_citation_lines(cfg, run_ctx=on))


def test_aavso_header_has_location() -> None:
    cfg = AppConfig()
    cfg.observer_lat = 50.075
    cfg.observer_lon = 14.418
    cfg.observer_alt_m = 400.0
    hdr = _aavso_header_snippet(cfg)
    assert "#LATITUDE=" in hdr
    assert "#LONGITUDE=" in hdr
    assert "#ELEVATION=" in hdr


def test_aavso_header_no_location_when_zero() -> None:
    cfg = AppConfig()
    cfg.observer_lat = 0.0
    cfg.observer_lon = 0.0
    cfg.observer_alt_m = 0.0
    hdr = _aavso_header_snippet(cfg)
    assert "#LATITUDE" not in hdr
    assert "#LONGITUDE" not in hdr
    assert "#ELEVATION" not in hdr
