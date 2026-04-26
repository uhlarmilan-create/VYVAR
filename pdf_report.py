"""
PDF variability-aware photometry report (wrapper around photometry_report).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from photometry_report import generate_photometry_report

logger = logging.getLogger(__name__)


def generate_report(
    photometry_dir: str,
    setup_name: str,
    draft_id: str,
    *,
    var_results: dict[str, Any] | None = None,
    candidates: list[str] | None = None,
    crossmatch_bullets: dict[str, str] | None = None,
    accepted_periods: dict[str, Any] | None = None,
    variability_timestamp: str | None = None,
) -> str | None:
    """
    Build ``report_{setup_name}.pdf`` under ``photometry_dir``.

    ``photometry_dir`` must be ``.../Drafts/draft_XXX/platesolve/{setup}/photometry``.
    Returns absolute path string, or None if reportlab / data missing.
    """
    pdir = Path(str(photometry_dir or "").strip()).resolve()
    if not pdir.is_dir():
        logger.warning("pdf_report: photometry_dir not a directory: %s", pdir)
        return None
    draft_dir = pdir.parent.parent
    out = pdir / f"report_{setup_name}.pdf"
    lbl = str(draft_id or "").strip() or draft_dir.name
    try:
        path = generate_photometry_report(
            draft_dir,
            str(setup_name),
            out,
            var_results=var_results,
            candidates=candidates,
            crossmatch_bullets=crossmatch_bullets,
            accepted_periods=accepted_periods,
            variability_timestamp=variability_timestamp,
            report_draft_label=lbl,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("pdf_report.generate_report failed: %s", exc)
        return None
    if path is None:
        return None
    return str(Path(path).resolve())
