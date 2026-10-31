"""Global RMS computation for the comparison pool (one pass across frames).

Kept separate from ``photometry_core`` because of size and circular imports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stats_core import coerce_bool_series

from gaia_catalog_id import normalize_gaia_source_id as _norm_id_val  # GAIA-ID-2026-10: one exact normaliser
from pipeline_constants import LINEARITY_FRAC

LOGGER = logging.getLogger(__name__)


def comp_frame_stats(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    *,
    comp_peak_sat_frac: float,
    flux_col: str = "dao_flux",
) -> dict[str, dict[str, int]]:
    """PHASE1-ONE-PASS-2026-10: per comparison candidate, over all per-frame catalogs:
    ``n_frames`` (frames with a positive flux), ``n_peak`` (frames with a peak and a limit) and
    ``n_over`` (frames whose raw peak >= comp_peak_sat_frac x SAT or flagged likely_saturated;
    COMP-SAT-ONE-2026-10).
    One pass for the whole field (before: a field pass whose filters only fed a diagnostic
    column, plus a second pass per target whose filters did not change the selection)."""
    out: dict[str, dict[str, int]] = {str(c): {"n_frames": 0, "n_peak": 0, "n_over": 0} for c in cand_ids}
    if not cand_ids:
        return out
    for csv_path in per_frame_csv_paths:  # counts: the frame order does not enter (G07-31)
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        name_col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
        if name_col is None:
            continue
        sub = df[df[name_col].astype(str).isin(cand_ids)]
        if sub.empty:
            continue
        ids = sub[name_col].astype(str)
        fcol = flux_col if flux_col in sub.columns else ("dao_flux" if "dao_flux" in sub.columns else None)
        if fcol is not None:
            ok_f = pd.to_numeric(sub[fcol], errors="coerce").gt(0)
            for cid, n in ids[ok_f].value_counts().items():
                out[str(cid)]["n_frames"] += int(n)
        flagged = (
            coerce_bool_series(sub["likely_saturated"])
            if "likely_saturated" in sub.columns else pd.Series(False, index=sub.index)
        )
        over = flagged.copy()
        if "peak_max_adu" in sub.columns and "saturate_limit_adu_85pct" in sub.columns:
            pk = pd.to_numeric(sub["peak_max_adu"], errors="coerce")
            # saturate_limit_adu_85pct == LINEARITY_FRAC x SAT
            lim = pd.to_numeric(sub["saturate_limit_adu_85pct"], errors="coerce") * (
                float(comp_peak_sat_frac) / float(LINEARITY_FRAC))
            ok_p = pk.notna() & lim.gt(0)
            for cid, n in ids[ok_p].value_counts().items():
                out[str(cid)]["n_peak"] += int(n)
            over = over | (ok_p & (pk >= lim))
        for cid, n in ids[over].value_counts().items():
            out[str(cid)]["n_over"] += int(n)
    return out


def _lookup_rms_map(rms_map: dict[str, float], keys: pd.Series) -> pd.Series:
    _norm_rms: dict[str, float] = {}
    _norm_key: dict[str, str] = {}
    for rk, rv in sorted(rms_map.items(), key=lambda kv: (float(kv[1]), str(kv[0]))):
        nk = _norm_id_val(rk)
        if nk not in _norm_key or str(rk) < str(_norm_key[nk]):
            _norm_key[nk] = rk
            _norm_rms[nk] = float(rv)

    def _lookup(k: str) -> float:
        if k in rms_map:
            return float(rms_map[k])
        nk = _norm_id_val(k)
        return _norm_rms.get(nk, float("nan"))

    return keys.map(_lookup)


def attach_comp_rms_to_pool_rows(
    pool: pd.DataFrame,
    rms_map: dict[str, float],
    *,
    id_col: str,
    frames_basis: str = "all_loadable",
) -> pd.DataFrame:
    """Attach gated ``comp_rms`` (= LOO mag MAD on the light-curve aperture grid)."""
    if pool.empty:
        return pool
    out = pool.copy()
    keys = out[id_col].astype(str).str.strip()
    loo = _lookup_rms_map(rms_map, keys) if rms_map else pd.Series(np.nan, index=out.index)
    out["comp_rms_loo_mag"] = loo
    out["comp_rms"] = loo
    out["comp_rms_frames_basis"] = str(frames_basis)
    return out.reset_index(drop=True)
