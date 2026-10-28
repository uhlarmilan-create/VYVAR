"""Moved from photometry_core.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

import math
import pandas as pd
from gaia_catalog_id import normalize_gaia_source_id
from comp_table import comp_quality_quality_strings



def lc_has_finite_airmass(lc_df: pd.DataFrame) -> bool:
    """True when the LC carries at least one finite airmass value."""
    if lc_df is None or getattr(lc_df, "empty", True) or "airmass" not in lc_df.columns:
        return False
    am = pd.to_numeric(lc_df["airmass"], errors="coerce")
    return bool(am.notna().any())

def apply_comp_w_rel_for_display(
    comp_df: pd.DataFrame,
    quality_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add ``w_rel`` = comp_weight / max(comp_weight) over **non-excluded** comps only.

    Aperture-photometry ``excluded`` stars remain in the table for transparency but get ``w_rel=0``.
    """
    df = comp_df.copy()
    if df.empty or "comp_weight" not in df.columns:
        return df
    w = pd.to_numeric(df["comp_weight"], errors="coerce")
    excluded = pd.Series(False, index=df.index)
    qmap = comp_quality_quality_strings(quality_map) if quality_map else {}
    if qmap and "catalog_id" in df.columns:
        for i, row in df.iterrows():
            cid = normalize_gaia_source_id(row.get("catalog_id"))
            if cid and str(qmap.get(cid, "")).strip().lower() == "excluded":
                excluded.loc[i] = True
    w_use = w.mask(excluded)
    w_max = float(w_use.max()) if w_use.notna().any() else float("nan")
    if math.isfinite(w_max) and w_max > 0:
        df["w_rel"] = (w / w_max).round(3)
    else:
        df["w_rel"] = float("nan")
    df.loc[excluded, "w_rel"] = 0.0
    return df


