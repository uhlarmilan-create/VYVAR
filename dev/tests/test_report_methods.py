"""Tests for method-keyed report path layout."""

from __future__ import annotations

from pathlib import Path

from report_methods import (
    aavso_export_path,
    active_report_methods,
    lc_csv_path,
    multi_method_reports_active,
    pdf_report_path,
    report_title,
    varastro_export_path,
)


class _Cfg:
    def __init__(self, psf: bool = False, adaptive: bool = False) -> None:
        self.psf_photometry_enabled = psf
        self.psf_adaptive_enabled = adaptive


def test_aperture_only_legacy_paths() -> None:
    cfg = _Cfg()
    methods = active_report_methods(cfg, have_psf_cols=False)
    assert methods == ["aperture"]
    assert not multi_method_reports_active(methods)

    lc = lc_csv_path(Path("/lc"), "123", "aperture")
    assert lc.name == "lightcurve_123.csv"

    aavso = aavso_export_path(Path("/r"), "Star", "20260430", "aperture", active_methods=methods)
    assert aavso.name == "Star_20260430.txt"

    pdf = pdf_report_path(Path("/d"), "Setup_1", "aperture", active_methods=methods)
    assert pdf.name.startswith("VYVAR_report_Setup_1_")
    assert "_psf" not in pdf.name


def test_multi_method_suffixed_paths() -> None:
    cfg = _Cfg(psf=True, adaptive=True)
    methods = active_report_methods(cfg, have_psf_cols=True)
    assert methods == ["aperture", "psf", "adaptive"]
    assert multi_method_reports_active(methods)

    assert lc_csv_path(Path("/lc"), "123", "psf").name == "lightcurve_123_psf.csv"
    assert (
        aavso_export_path(Path("/r"), "Star", "20260430", "psf", active_methods=methods).name
        == "Star_20260430_psf.txt"
    )
    assert (
        varastro_export_path(Path("/r"), "Star", "20260430", "adaptive", active_methods=methods).name
        == "Star_20260430_adaptive.txt"
    )
    pdf = pdf_report_path(Path("/d"), "Setup_1", "adaptive", active_methods=methods, date_str="20260531")
    assert pdf.name == "VYVAR_report_Setup_1_20260531_adaptive.pdf"


def test_report_title_suffix_only_when_multi() -> None:
    base = "VYVAR - Summary Measure Report"
    assert report_title(base, "aperture", active_methods=["aperture"]) == base
    assert report_title(base, "psf", active_methods=["aperture", "psf"]).endswith("[PSF photometry]")
