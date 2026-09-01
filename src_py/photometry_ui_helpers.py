"""Moved from photometry_core.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

import pandas as pd

from photometry_core import (
    TIME_BASE_BJD_TDB,
    TIME_BASE_COL,
    TIME_BASE_JD_FALLBACK,
)

def resolve_lc_time_base(lc_df: pd.DataFrame) -> str:
    """Return the single ``time_base`` value for an LC dataframe."""
    if lc_df is None or getattr(lc_df, "empty", True):
        raise ValueError("empty LC")
    if TIME_BASE_COL not in lc_df.columns:
        raise ValueError("time_base column absent")
    raw = lc_df[TIME_BASE_COL].astype(str).str.strip()
    raw = raw[raw.str.lower().ne("nan") & (raw != "")]
    if raw.empty:
        raise ValueError("time_base column empty")
    uniq = set(raw.unique())
    if len(uniq) > 1:
        raise ValueError(f"mixed time_base values: {sorted(uniq)}")
    tb = uniq.pop()
    if tb not in (TIME_BASE_BJD_TDB, TIME_BASE_JD_FALLBACK):
        raise ValueError(f"unknown time_base: {tb!r}")
    return tb

def lc_time_axis_short_label(time_base: str) -> str:
    """Human-readable LC time axis label from ``time_base``."""
    if time_base == TIME_BASE_BJD_TDB:
        return "BJD (TDB)"
    if time_base == TIME_BASE_JD_FALLBACK:
        return "JD (fallback)"
    return "time"
