"""HRD extreme-object details row payload and PDF section tests (TODO-12f)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas

from hrd_analysis import (
    _make_row,
    build_hrd_detail_line,
    format_hrd_dsc_wd_p,
    format_hrd_sexagesimal,
    hrd_detail_header_name,
)
from photometry_report import _PhotometryReportBuilder


def _series(**kwargs) -> pd.Series:
    base = {
        "catalog_id": "458558784733311232",
        "phot_g_mean_mag": 12.5,
        "abs_mag_g": 11.0,
        "bp_rp": 0.2,
        "teff_gspphot": None,
        "logg_gspphot": None,
        "parallax": 0.4,
        "parallax_over_error": 8.0,
        "hrd_reliable": True,
        "ra_deg": 56.871,
        "dec_deg": 57.032,
        "x": 100.0,
        "y": 200.0,
        "simbad_main_id": "LAWD 12",
        "simbad_otype": "WD*",
        "simbad_sp_type": "DA2.3",
        "classprob_dsc_combmod_whitedwarf": 0.99994957447052,
        "enrich_source": "simbad",
        "_logg_source": "n/a",
    }
    base.update(kwargs)
    return pd.Series(base)


def test_make_row_dist_pc_only_when_reliable() -> None:
    reliable = _make_row(_series(), "White dwarf candidate", table_na=True)
    assert reliable["dist_pc"] == "2500.0"
    assert reliable["parallax_mas"] == "0.40"
    assert reliable["dsc_wd_p"] == "1"
    assert reliable["sp_type_raw"] == "DA2.3"
    assert reliable["otype_raw"] == "WD*"
    assert reliable["teff_source"] == "n/a"

    unreliable = _make_row(
        _series(hrd_reliable=False, parallax=0.4),
        "White dwarf candidate",
        table_na=True,
    )
    assert unreliable["dist_pc"] == "N/A"


def test_format_hrd_sexagesimal_known_coords() -> None:
    s = format_hrd_sexagesimal(56.871, 57.032)
    assert ":" in s
    assert "+57" in s


def test_build_hrd_detail_line_omits_empty_spt() -> None:
    row = _make_row(_series(simbad_sp_type="", simbad_otype=""), "White dwarf candidate", table_na=True)
    line = build_hrd_detail_line(row)
    assert "SpT=" not in line
    assert "otype=" not in line
    assert "Teff=N/A" in line
    assert "DSC WD p=" in line


def test_hrd_detail_header_prefers_simbad() -> None:
    row = _make_row(_series(), "White dwarf (DA2.3, SIMBAD)", ident="confirmed", table_na=True)
    assert hrd_detail_header_name(row) == "LAWD 12"
    row2 = _make_row(_series(simbad_main_id=""), "X", table_na=True)
    assert hrd_detail_header_name(row2).startswith("Gaia DR3")


def test_format_hrd_dsc_wd_p_two_sig() -> None:
    assert format_hrd_dsc_wd_p(0.99994957447052) == "1"
    assert format_hrd_dsc_wd_p(2.003e-12) == "2e-12"


def _minimal_report_builder(tmp_path: Path) -> _PhotometryReportBuilder:
    obs_group = "B_20_2"
    phot = tmp_path / "platesolve" / obs_group / "photometry"
    phot.mkdir(parents=True)
    pd.DataFrame({"catalog_id": ["1"], "lc_rms": [0.05]}).to_csv(
        phot / "photometry_summary.csv", index=False
    )
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Paragraph, Table, TableStyle

    return _PhotometryReportBuilder(
        draft_dir=tmp_path,
        obs_group=obs_group,
        output_pdf=tmp_path / "report.pdf",
        var_results=None,
        candidates=None,
        crossmatch_bullets=None,
        accepted_periods=None,
        variability_timestamp=None,
        report_draft_label=None,
        tess_results=None,
        report_title="Test",
        font_reg="Helvetica",
        font_bold="Helvetica-Bold",
        font_obl="Helvetica-Oblique",
        colors_mod=colors,
        cm_mod=cm,
        mm_mod=mm,
        landscape_fn=landscape,
        a4_size=A4,
        canvas_mod=canvas,
        image_reader_mod=ImageReader,
        table_mod=Table,
        table_style_mod=TableStyle,
        paragraph_mod=Paragraph,
        paragraph_style_mod=ParagraphStyle,
        ta_left_mod=TA_LEFT,
    )


def test_pdf_hrd_extreme_details_smoke(tmp_path: Path) -> None:
    builder = _minimal_report_builder(tmp_path)
    builder._verify_overflow = True
    row = _make_row(
        _series(
            catalog_id="458407464445792384",
            simbad_main_id="V* RS Per",
            simbad_sp_type="M3.5IabFe-1",
            simbad_otype="s*r",
            parallax=0.43,
            hrd_reliable=True,
            _logg_source="simbad_lumclass",
        ),
        "Red supergiant (M3.5IabFe-1, SIMBAD)",
        ident="confirmed",
        table_na=True,
    )
    top = pd.DataFrame([row])
    pdf_path = tmp_path / "hrd_details.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=landscape(A4))
    builder._report_hrd_extreme_details(c, top)
    c.save()
    assert pdf_path.is_file()
    assert builder.overflow_violation_count == 0
