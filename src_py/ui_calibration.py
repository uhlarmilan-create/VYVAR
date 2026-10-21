"""Streamlit UI helpers for calibration / smart-import (equipment header, master binning, multi-obs status)."""

from __future__ import annotations

from typing import Any

import pandas as pd



_PLAN_STATUS_LABEL = {
    "OK": "OK",
    "MISSING_DARK": "no valid DARK",
    "MISSING_FLAT": "no valid FLAT",
    "MISSING_DARK_AND_FLAT": "no valid DARK / FLAT",
    "NO_CAMERA_SELECTED": "select camera + telescope",
    "PRECALIBRATED": "used as is (pre-calibrated / uncalibrated)",
}


def build_calibration_plan_dataframe(plan: Any) -> pd.DataFrame:
    """CAL-2026-10 calibration plan table: one row per LIGHT group, which master DARK /
    FLAT RUN VYVAR will use and where it comes from (LIGHT folder or library)."""
    cp = getattr(plan, "calibration_plan", None) or {}
    _from = {"lights": "LIGHT folder", "library": "library"}
    rows: list[dict[str, Any]] = []
    for g in cp.get("groups", []):
        notes: list[str] = []
        for k in ("dark", "flat"):
            if int(g.get(f"{k}_rebin") or 1) > 1:
                notes.append(f"{k} 1x1->{g[f'{k}_rebin']}x{g[f'{k}_rebin']}")
        if g.get("flat") and g.get("flat_offset_handling"):
            notes.append(f"flat offset: {g['flat_offset_handling']}")
        notes += [m for m in (g.get("messages") or []) if m]
        rows.append(
            {
                "LIGHT group": f"{g.get('filter')} | {g.get('exposure_s'):g} s | bin{g.get('binning')}",
                "Frames": g.get("n_lights"),
                "Status": _PLAN_STATUS_LABEL.get(str(g.get("status")), str(g.get("status"))),
                "DARK": g.get("dark") or "-",
                "DARK from": _from.get(str(g.get("dark_from") or ""), "") if g.get("dark") else "",
                "FLAT": g.get("flat") or "-",
                "FLAT from": _from.get(str(g.get("flat_from") or ""), "") if g.get("flat") else "",
                "Note": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows)
