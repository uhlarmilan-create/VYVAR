"""Tests for export header citations, methods matrix, and observer location lines."""

from __future__ import annotations

from config import AppConfig
from citations import (
    build_methods_matrix_lines,
    build_run_citation_context,
    citation_line,
    emit_export_citation_lines,
    emit_pdf_methods_sections,
    load_citations_bib,
    plain_ascii_citation_text,
)
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
    load_citations_bib.cache_clear()
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
        "jordi2010",
        "smith2002",
    }
    missing = required - set(bib)
    assert not missing, f"Missing CITATIONS.bib keys: {sorted(missing)}"


def test_collins_and_jordi_bib_venues() -> None:
    load_citations_bib.cache_clear()
    bib = load_citations_bib()
    c = bib["collins2017"]
    assert "Stassun" in c["author"]
    assert "Hessman" in c["author"]
    assert "Astronomical Journal" in c["journal"]
    assert c["volume"] == "153"
    assert "77" in c["pages"]
    j = bib["jordi2010"]
    assert j["volume"] == "523"
    assert "A48" in j["pages"]
    assert "523, A48" in citation_line("jordi2010", bib=bib)
    assert "AJ 153, 77" in citation_line("collins2017", bib=bib)


def test_slim_export_has_matrix_not_core_blocks() -> None:
    cfg = AppConfig()
    text = _citation_text(cfg)
    assert "METHODS MATRIX (this run):" in text
    assert "ensemble flux-sum: ON" in text
    assert "Full algorithm references: SUMMARY MEASURE REPORT (PDF)" in text
    assert "[CORE]" not in text
    assert "[CATALOGS & TIME]" not in text
    assert "[SOFTWARE]" not in text
    assert "[FIELD ASTROPHYSICS (HRD)]" not in text
    assert "ALGORITHMS & REFERENCES" not in text


def test_methods_matrix_reflects_flags() -> None:
    off = AppConfig()
    off.pytics_enabled = False
    off.sysrem_enabled = False
    off.savgol_detrend_enabled = False
    off.democratic_detrend_enabled = False
    off.temporal_binning_enabled = False
    off.gs11_dilution_enabled = False
    off.psf_photometry_enabled = False
    off.psf_adaptive_enabled = False
    off.aperture_correction_enabled = False
    off.cog_aperture_correction_enabled = False
    off.per_frame_saturation_enabled = False
    off.trust_flag_enabled = False
    off.apply_color_term = "off"
    off.k2_mode = "off"
    off.err_background_mode = "howell"
    ctx_off = build_run_citation_context(off)
    mat_off = "\n".join(build_methods_matrix_lines(ctx_off))
    assert "PyTICS: OFF" in mat_off
    assert "SysRem: OFF" in mat_off
    assert "k2: OFF" in mat_off
    assert "color term: OFF" in mat_off
    assert "PSF branch: OFF" in mat_off
    assert "trust gate: OFF" in mat_off
    assert "empirical background mode: OFF" in mat_off
    assert "ensemble flux-sum: ON" in mat_off

    on = AppConfig()
    on.pytics_enabled = True
    on.sysrem_enabled = True
    on.apply_color_term = "auto"
    on.k2_mode = "literature"
    on.psf_photometry_enabled = True
    on.err_background_mode = "empirical"
    ctx_on = build_run_citation_context(on)
    mat_on = "\n".join(build_methods_matrix_lines(ctx_on))
    assert "PyTICS: ON" in mat_on
    assert "SysRem: ON" in mat_on
    assert "color term: ON (auto)" in mat_on
    assert "k2: ON (" in mat_on
    assert "PSF branch: ON" in mat_on
    assert "empirical background mode: ON" in mat_on


def test_slim_methods_citations_conditional_only() -> None:
    off = AppConfig()
    off.sysrem_enabled = False
    off.pytics_enabled = False
    off.temporal_binning_enabled = False
    off.democratic_detrend_enabled = False
    off.savgol_detrend_enabled = False
    off.k2_mode = "off"
    off.comp_sparse_fallback_enabled = False
    off.comp_iterative_clip_enabled = False
    off_text = _citation_text(off)
    assert "Tamuz" not in off_text
    assert "[METHODS - this run]" not in off_text

    on = AppConfig()
    on.sysrem_enabled = True
    on_text = _citation_text(on)
    assert "[METHODS - this run]" in on_text
    assert "Tamuz" in on_text


def test_pdf_keeps_full_citation_blocks() -> None:
    cfg = AppConfig()
    ctx = build_run_citation_context(cfg)
    sections = emit_pdf_methods_sections(ctx)
    titles = [t for t, _ in sections]
    assert any("Matrix" in t for t in titles)
    joined = "\n".join(t + "\n" + "\n".join(items) for t, items in sections)
    assert "Broeg" in joined
    assert "Collins" in joined or "AstroImageJ" in joined
    assert "Gaia Collaboration" in joined or "Gaia" in joined


def test_no_backslash_in_emitted_header_lines() -> None:
    load_citations_bib.cache_clear()
    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.psf_photometry_enabled = True
    text = _citation_text(cfg)
    for line in text.splitlines():
        assert "\\" not in line, line
    assert "tot_C_cnts" in citation_line("collins2017")
    assert "\\" not in plain_ascii_citation_text(r"tot\_C\_cnts \ensuremath\Delta mag")


def test_citation_lines_vsx_when_configured() -> None:
    from pathlib import Path

    cfg = AppConfig()
    vsx = Path(__file__).resolve().parents[2] / "VSX" / "vyvar_vsx_local.db"
    if not vsx.is_file():
        return
    cfg.vsx_local_db_path = str(vsx)
    # VSX is catalogs block - slim export omits it; PDF retains it.
    ctx = build_run_citation_context(cfg)
    pdf = "\n".join(x for _, items in emit_pdf_methods_sections(ctx) for x in items)
    assert "Watson" in pdf


def test_citation_lines_psf_conditional_pdf() -> None:
    off = AppConfig()
    off.psf_photometry_enabled = False
    off.psf_adaptive_enabled = False
    on = AppConfig()
    on.psf_photometry_enabled = True
    off_pdf = "\n".join(
        x for _, items in emit_pdf_methods_sections(build_run_citation_context(off)) for x in items
    )
    on_pdf = "\n".join(
        x for _, items in emit_pdf_methods_sections(build_run_citation_context(on)) for x in items
    )
    assert "Anderson" not in off_pdf
    assert "Anderson" in on_pdf
    assert "Moffat" in on_pdf


def test_citation_lines_sysrem_conditional() -> None:
    off = AppConfig()
    off.sysrem_enabled = False
    on = AppConfig()
    on.sysrem_enabled = True
    assert "Tamuz" not in _citation_text(off)
    assert "Tamuz" in _citation_text(on)


def test_citation_lines_data_quality_gate_conditional_pdf() -> None:
    off = AppConfig()
    off.comp_qa_enabled = False
    off.trust_flag_enabled = False
    on = AppConfig()
    text_off = "\n".join(
        t + "\n" + "\n".join(items)
        for t, items in emit_pdf_methods_sections(build_run_citation_context(off))
    )
    text_on = "\n".join(
        t + "\n" + "\n".join(items)
        for t, items in emit_pdf_methods_sections(build_run_citation_context(on))
    )
    assert "Data-Quality Gate" not in text_off and "DATA-QUALITY" not in text_off
    assert "Sokolovsky" not in text_off
    assert "Sokolovsky" in text_on
    assert "von Neumann" in text_on


def test_citation_lines_iterative_comp_clip_conditional() -> None:
    cfg_off = AppConfig()
    cfg_off.comp_sparse_fallback_enabled = False
    cfg_off.comp_iterative_clip_enabled = False
    off = build_run_citation_context(cfg_off, pipeline_meta={"comp_sparse_fallback_used": False})
    on_cfg = AppConfig()
    on_cfg.comp_sparse_fallback_enabled = True
    on = build_run_citation_context(on_cfg, pipeline_meta={"comp_sparse_fallback_used": True})
    off_text = "".join(emit_export_citation_lines(off))
    on_text = "".join(emit_export_citation_lines(on))
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


def test_citation_lines_period_conditional_pdf() -> None:
    cfg = AppConfig()
    off = build_run_citation_context(cfg, period_analysis=False)
    on = build_run_citation_context(cfg, period_analysis=True)
    off_pdf = "\n".join(x for _, items in emit_pdf_methods_sections(off) for x in items)
    on_pdf = "\n".join(x for _, items in emit_pdf_methods_sections(on) for x in items)
    assert "VanderPlas" not in off_pdf
    assert "VanderPlas" in on_pdf


def test_obscode_warning_only_when_unset() -> None:
    """UMIA is a real configured code - no placeholder warning."""
    import export_reports as er
    import inspect

    src = inspect.getsource(er)
    assert "default placeholder" not in src
    assert "_AAVSO_OBSCODE_PLACEHOLDER" not in src
    assert "aavso_observer_code" in src
    assert "observer code not set" in src


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
