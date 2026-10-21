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
from export_reports import _vyvar_export_citation_lines


def _citation_text(cfg: AppConfig | None = None, **kw) -> str:
    return "".join(_vyvar_export_citation_lines(cfg, **kw))


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
        "anderson2006",
        "andersonking2006",
        "aigrain2004",
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
    assert "ensemble weights: ON (Broeg 2005 iterative)" in text
    assert "ensemble flux-sum" not in text  # EXPORT-2026-10: not what the pipeline does
    assert "Full algorithm description: SUMMARY MEASURE REPORT (PDF)" in text
    assert "[METHODS - this run]" in text and "Broeg" in text
    assert "[CORE]" not in text
    assert "[SOFTWARE]" not in text
    assert "[FIELD ASTROPHYSICS (HRD)]" not in text

def test_methods_matrix_reflects_flags() -> None:
    off = AppConfig()
    off.gs11_dilution_enabled = False
    off.trust_flag_enabled = False
    off.apply_color_term = "off"
    off.k2_mode = "off"
    mat_off = "\n".join(build_methods_matrix_lines(build_run_citation_context(off, obs_group="g_120_4")))
    for gone in ("PyTICS", "SavGol", "SysRem", "flux-sum", "iterative comp clip", "COG"):
        assert gone not in mat_off
    assert "second-order extinction k'': OFF" in mat_off
    assert "colour term: OFF (config)" in mat_off
    assert "PSF photometry: OFF" in mat_off
    assert "trust gate: OFF" in mat_off
    assert "zero point: Gaia DR3 -> Sloan g" in mat_off
    assert "sigma clipping / detrending / binning: OFF" in mat_off

    on = AppConfig()
    on.apply_color_term = "auto"
    on.k2_mode = "literature"
    mat_on = "\n".join(build_methods_matrix_lines(build_run_citation_context(on, obs_group="g_120_4", psf_used=True)))
    assert "colour term: ON (polynomial in BP-RP" in mat_on
    assert "second-order extinction k'': ON (" in mat_on
    assert "PSF photometry: ON" in mat_on
    clear = "\n".join(build_methods_matrix_lines(build_run_citation_context(on, obs_group="NoFilter_60_2")))
    assert "colour term: OFF (filter reported untransformed)" in clear
    assert "Clear on Johnson V zero point (CV)" in clear

def test_slim_methods_citations_conditional_only() -> None:
    off = AppConfig()
    off.k2_mode = "off"
    off_text = _citation_text(off)
    assert "Smith et al. (2002)" not in off_text
    on = AppConfig()
    on.k2_mode = "literature"
    on_text = _citation_text(on, obs_group="g_120_4")
    assert "Smith et al. (2002)" in on_text
    assert "Tamuz" not in on_text
    # REPORT-AUDIT-2026-10: no literature k'' for a clear filter -> k'' not applied, not cited
    assert "Smith et al. (2002)" not in _citation_text(on, obs_group="NoFilter_60_2")
    clear = "\n".join(build_methods_matrix_lines(build_run_citation_context(on, obs_group="NoFilter_60_2")))
    assert "second-order extinction k'': OFF (no literature k'' for this filter)" in clear

def test_pdf_keeps_full_citation_blocks() -> None:
    cfg = AppConfig()
    ctx = build_run_citation_context(cfg)
    sections = emit_pdf_methods_sections(ctx)
    titles = [t for t, _ in sections]
    assert any("Matrix" in t for t in titles)
    joined = "\n".join(t + "\n" + "\n".join(items) for t, items in sections)
    assert "Broeg" in joined and "Honeycutt" in joined
    assert "Moffat" in joined
    assert "Gaia Collaboration" in joined
    assert "Collins" not in joined  # flux-sum ensemble is not used

def test_no_backslash_in_emitted_header_lines() -> None:
    load_citations_bib.cache_clear()
    cfg = AppConfig()
    cfg.k2_mode = "literature"
    text = _citation_text(cfg)
    for line in text.splitlines():
        assert "\\" not in line, line
    assert "tot_C_cnts" in citation_line("collins2017")
    assert "\\" not in plain_ascii_citation_text(r"tot\_C\_cnts \ensuremath\Delta mag")


def test_citation_lines_vsx_when_configured() -> None:
    from pathlib import Path

    cfg = AppConfig()
    vsx = Path(__file__).resolve().parents[2] / "catalogs" / "vyvar_objects.db"
    if not vsx.is_file():
        return
    cfg.objects_db_path = str(vsx)
    # VSX is catalogs block - slim export omits it; PDF retains it.
    ctx = build_run_citation_context(cfg)
    pdf = "\n".join(x for _, items in emit_pdf_methods_sections(ctx) for x in items)
    assert "Watson" in pdf


def test_citation_lines_psf_conditional_pdf() -> None:
    off_pdf = "\n".join(
        x for _, items in emit_pdf_methods_sections(build_run_citation_context(AppConfig())) for x in items
    )
    on_pdf = "\n".join(
        x for _, items in emit_pdf_methods_sections(build_run_citation_context(AppConfig(), psf_used=True))
        for x in items
    )
    assert "Anderson" not in off_pdf
    assert "Anderson" in on_pdf
    assert "A&A 454, 1029" in on_pdf or "454, 1029" in on_pdf


def test_citation_lines_variability_search_always_cited() -> None:
    """VAR-2026-10: the field search always runs; COMP-QA-2026-10: no data-quality gate block."""
    for trust in (False, True):
        cfg = AppConfig()
        cfg.trust_flag_enabled = trust
        text = "\n".join(
            t + "\n" + "\n".join(items)
            for t, items in emit_pdf_methods_sections(build_run_citation_context(cfg))
        )
        assert "data-quality" not in text.lower()
        assert "variability search" in text.lower()
        assert "Sokolovsky" in text and "von Neumann" in text and "Azzalini" in text


def test_citation_lines_iterative_comp_clip_never_cited() -> None:
    # COMP-RULES-2026-10 removed the sparse fallback / iterative clip; never cite it.
    on = build_run_citation_context(AppConfig(), pipeline_meta={"comp_sparse_fallback_used": True})
    on_text = "".join(emit_export_citation_lines(on))
    assert "Gilliland" not in on_text
    assert "Burdanov" not in on_text

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

    from tests.cython_compat import skip_if_compiled

    skip_if_compiled(
        "export_reports",
        "inspect.getsource requires interpreted export_reports.py",
    )
    src = inspect.getsource(er)
    assert "default placeholder" not in src
    assert "_AAVSO_OBSCODE_PLACEHOLDER" not in src
    assert "observer_code" in src
    assert "observer code not set" in src




def test_missing_bibliography_or_key_fails_loudly(tmp_path) -> None:
    """EPSF-CLEAN-2026-10: never write 'Unknown (?)' citation lines into exports."""
    import pytest

    from citations import citation_line

    with pytest.raises(FileNotFoundError):
        load_citations_bib(str(tmp_path / "nope.bib"))
    with pytest.raises(KeyError):
        citation_line("no_such_key_1900", bib=load_citations_bib())
