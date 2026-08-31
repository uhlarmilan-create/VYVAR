"""Method-keyed report layout for AAVSO / VarAstro / PDF exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from config import AppConfig
except ImportError:  # pragma: no cover
    AppConfig = Any  # type: ignore[misc, assignment]

VALID_LC_METHODS = ("aperture", "psf", "adaptive")



def active_report_methods(
    cfg: AppConfig | None,
    *,
    have_psf_cols: bool = False,
) -> list[str]:
    """Methods with separate report/LC products for this run."""
    methods = ["aperture"]
    if cfg is None or not have_psf_cols:
        return methods
    if bool(getattr(cfg, "psf_photometry_enabled", False)):
        methods.append("psf")
    if bool(getattr(cfg, "psf_adaptive_enabled", False)):
        methods.append("adaptive")
    return methods


def multi_method_reports_active(methods: list[str]) -> bool:
    return len(methods) > 1


def lc_csv_path(lc_dir: Path | str, catalog_id: str, method: str = "aperture") -> Path:
    cid = str(catalog_id).strip()
    base = Path(lc_dir)
    m = str(method or "aperture").strip().lower()
    if m == "aperture":
        return base / f"lightcurve_{cid}.csv"
    return base / f"lightcurve_{cid}_{m}.csv"


def aavso_export_path(
    reports_dir: Path | str,
    safe_name: str,
    date_tag: str,
    method: str,
    *,
    active_methods: list[str] | None = None,
) -> Path:
    """AAVSO .txt path - aperture-only runs keep legacy filenames."""
    out = Path(reports_dir) / "aavso"
    m = str(method or "aperture").strip().lower()
    if m == "aperture" or not multi_method_reports_active(list(active_methods or ["aperture"])):
        return out / f"{safe_name}_{date_tag}.txt"
    return out / f"{safe_name}_{date_tag}_{m}.txt"


def varastro_export_path(
    reports_dir: Path | str,
    safe_name: str,
    date_tag: str,
    method: str,
    *,
    active_methods: list[str] | None = None,
) -> Path:
    out = Path(reports_dir) / "varastro"
    m = str(method or "aperture").strip().lower()
    if m == "aperture" or not multi_method_reports_active(list(active_methods or ["aperture"])):
        return out / f"{safe_name}_{date_tag}.txt"
    return out / f"{safe_name}_{date_tag}_{m}.txt"


def pdf_report_path(
    draft_dir: Path | str,
    obs_group: str,
    method: str,
    *,
    active_methods: list[str] | None = None,
    date_str: str | None = None,
) -> Path:
    from datetime import datetime

    d = Path(draft_dir)
    og = str(obs_group).strip()
    ds = str(date_str or datetime.today().strftime("%Y%m%d"))
    m = str(method or "aperture").strip().lower()
    if m == "aperture" or not multi_method_reports_active(list(active_methods or ["aperture"])):
        return d / "platesolve" / og / f"VYVAR_report_{og}_{ds}.pdf"
    return d / "platesolve" / og / f"VYVAR_report_{og}_{ds}_{m}.pdf"


def report_title(base_title: str, method: str, *, active_methods: list[str] | None = None) -> str:
    m = str(method or "aperture").strip().lower()
    if m == "aperture" or not multi_method_reports_active(list(active_methods or ["aperture"])):
        return str(base_title)
    label = {"psf": "PSF", "adaptive": "Adaptive"}.get(m, m.title())
    return f"{base_title} [{label} photometry]"


def software_method_label(method: str) -> str:
    m = str(method or "aperture").strip().lower()
    return {"aperture": "aperture", "psf": "PSF", "adaptive": "adaptive"}.get(m, m)
