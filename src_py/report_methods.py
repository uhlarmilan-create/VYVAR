"""File layout of the light-curve and report products (aperture photometry; CFG-REVIEW-2026-10).

The published light curves, the AAVSO / VarAstro exports and the PDF report are aperture photometry; the ePSF
light curves are internal (``lightcurve_<id>_psf.csv``) plus the user export of ``epsf_export``.
"""

from __future__ import annotations

from pathlib import Path


def lc_csv_path(lc_dir: Path | str, catalog_id: str, method: str = "aperture") -> Path:
    cid = str(catalog_id).strip()
    base = Path(lc_dir)
    m = str(method or "aperture").strip().lower()
    if m == "aperture":
        return base / f"lightcurve_{cid}.csv"
    return base / f"lightcurve_{cid}_{m}.csv"


def aavso_export_path(reports_dir: Path | str, safe_name: str, date_tag: str) -> Path:
    return Path(reports_dir) / "aavso" / f"{safe_name}_{date_tag}.txt"


def varastro_export_path(reports_dir: Path | str, safe_name: str, date_tag: str) -> Path:
    return Path(reports_dir) / "varastro" / f"{safe_name}_{date_tag}.txt"


def pdf_report_path(draft_dir: Path | str, obs_group: str, *, date_str: str | None = None) -> Path:
    from datetime import datetime

    og = str(obs_group).strip()
    ds = str(date_str or datetime.today().strftime("%Y%m%d"))
    return Path(draft_dir) / "platesolve" / og / f"VYVAR_report_{og}_{ds}.pdf"
