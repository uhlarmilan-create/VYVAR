"""Moved from photometry_core.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from gaia_catalog_id import normalize_gaia_source_id
from photometry_gate_helpers import _resolve_star_flux_method, comp_quality_quality_strings

from photometry_core import (
    _coerce_bool_cell,
)

def _get_lc_star_method(cid: str, all_frames: pd.DataFrame, star_method: str) -> np.ndarray:
    """Inst mag for one star using a fixed method for all frames (NaN if PSF missing)."""
    from photometry_lightcurve import _get_lc  # noqa: PLC0415
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if str(star_method).strip().lower() != "psf":
        return _get_lc(cid, all_frames)
    return _get_lc_psf_strict(cid, all_frames)


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

    Phase-2A ``excluded`` stars remain in the table for transparency but get ``w_rel=0``.
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

def ensemble_member_ids(
    comp_quality: dict[str, dict],
    comp_rms_map: dict[str, float] | None = None,
    *,
    n_comp_min: int = 3,
    n_comp_max: int = 10,
) -> set[str]:
    """Catalog ids selected for Phase-2A ``ensemble_normalize`` (check-star must be outside)."""
    comp_rms_map = comp_rms_map or {}
    p2p_thr = float("nan")
    for q in comp_quality.values():
        t = q.get("p2p_threshold")
        if t is not None and math.isfinite(float(t)):
            p2p_thr = float(t)
            break
    usable_all = [
        cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")
    ]
    usable_sorted = sorted(
        usable_all,
        key=lambda c: (
            0 if comp_quality[c].get("quality") == "good" else 1,
            float(comp_rms_map.get(c, float("inf"))),
            str(c),
        ),
    )
    selected: list[str] = []
    for cid in usable_sorted:
        if len(selected) >= int(n_comp_max):
            break
        p2p = float(comp_quality[cid].get("rms_p2p", float("nan")))
        if (
            len(selected) < int(n_comp_min)
            or (math.isfinite(p2p_thr) and math.isfinite(p2p) and p2p < p2p_thr)
            or not math.isfinite(p2p_thr)
        ):
            selected.append(cid)
    return {str(c) for c in selected[: int(n_comp_max)]}

def _get_lc_psf_strict(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    """PSF-only inst mag: NaN when PSF flux unavailable or AC not applied (no aperture fallback)."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if "psf_flux" not in sub.columns or "psf_fit_ok" not in sub.columns:
        return np.full(len(sub), float("nan"), dtype=float)
    psf_flux = pd.to_numeric(sub["psf_flux"], errors="coerce").to_numpy(dtype=float)
    psf_ok = sub["psf_fit_ok"].map(_coerce_bool_cell).to_numpy(dtype=bool)
    if "psf_ac_applied" in sub.columns:
        ac_ok = sub["psf_ac_applied"].map(_coerce_bool_cell).to_numpy(dtype=bool)
    else:
        ac_ok = np.zeros(len(sub), dtype=bool)
    psf_mag = np.where(
        psf_ok & np.isfinite(psf_flux) & (psf_flux > 0) & ac_ok,
        -2.5 * np.log10(psf_flux),
        np.nan,
    )
    return np.asarray(psf_mag, dtype=float)

def _get_lc_adaptive_per_star(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    """Adaptive LC: one method per star applied consistently across all frames."""
    sm = _resolve_star_flux_method(cid, all_frames)
    return _get_lc_star_method(cid, all_frames, sm)
