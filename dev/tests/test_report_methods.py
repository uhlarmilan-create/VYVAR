"""Product file layout (aperture photometry; CFG-REVIEW-2026-10)."""

from __future__ import annotations

from pathlib import Path

from report_methods import aavso_export_path, lc_csv_path, pdf_report_path, varastro_export_path


def test_paths() -> None:
    assert lc_csv_path(Path("/lc"), "123").name == "lightcurve_123.csv"
    assert lc_csv_path(Path("/lc"), "123", "psf").name == "lightcurve_123_psf.csv"
    assert aavso_export_path(Path("/r"), "Star", "20260430") == Path("/r/aavso/Star_20260430.txt")
    assert varastro_export_path(Path("/r"), "Star", "20260430") == Path("/r/varastro/Star_20260430.txt")
    pdf = pdf_report_path(Path("/d"), "Setup_1", date_str="20260531")
    assert pdf == Path("/d/platesolve/Setup_1/VYVAR_report_Setup_1_20260531.pdf")
