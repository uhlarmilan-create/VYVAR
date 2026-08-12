"""Shared per-frame flux normalization for comp RMS (Phase 1 + global pool).

Differential photometry comp scatter is measured on frame-to-frame relative flux,
normalized by a stable reference per frame.  The reference must come from the
full matched catalog in that frame (one row per ``catalog_id``), not from a
per-target candidate subset whose membership varies by target and config.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

_MAD_SIGMA_SCALE = 1.4826


def _dedupe_score_series(df: pd.DataFrame, flux_col: str) -> pd.Series:
    if "peak_max_adu" in df.columns:
        score = pd.to_numeric(df["peak_max_adu"], errors="coerce")
    elif "dao_flux" in df.columns:
        score = pd.to_numeric(df["dao_flux"], errors="coerce")
    else:
        score = pd.to_numeric(df.get(flux_col), errors="coerce")
    return score.fillna(-1.0)


def dedupe_catalog_rows(df: pd.DataFrame, *, id_col: str, flux_col: str) -> pd.DataFrame:
    """Keep one row per non-empty ``id_col`` (brightest detection wins)."""
    if df is None or df.empty or id_col not in df.columns:
        return df
    out = df.copy()
    ids = out[id_col].fillna("").astype(str).str.strip()
    matched = ids.ne("") & ~ids.str.lower().isin({"nan", "none"})
    if not bool(matched.any()):
        return out
    out = (
        out.assign(_dedupe_score=_dedupe_score_series(out, flux_col))
        .sort_values("_dedupe_score", ascending=False, kind="mergesort")
        .drop_duplicates(subset=[id_col], keep="first")
        .drop(columns=["_dedupe_score"])
    )
    return out.reset_index(drop=True)


def matched_catalog_flux_rows(df: pd.DataFrame, *, flux_col: str) -> pd.DataFrame:
    """Positive-flux rows with a non-empty ``catalog_id`` (matched catalog only)."""
    pos = df[pd.to_numeric(df[flux_col], errors="coerce").gt(0)].copy()
    if pos.empty:
        return pos
    if "catalog_id" not in pos.columns:
        return pos
    cid = pos["catalog_id"].fillna("").astype(str).str.strip()
    pos = pos[cid.ne("") & ~cid.str.lower().isin({"nan", "none"})]
    if pos.empty:
        return pos
    return dedupe_catalog_rows(pos, id_col="catalog_id", flux_col=flux_col)


def build_frame_bin_medians(
    df: pd.DataFrame,
    *,
    flux_col: str,
    mag_bin_width: float = 0.5,
) -> tuple[dict[int, float], float, bool]:
    """Return (bin_medians, frame_median, used_mag_bins) for one frame CSV.

    ``bin_medians[b]`` is the median ``flux_col`` among matched-catalog stars in
    magnitude bin ``b`` (floor(mag / mag_bin_width)).  When ``mag`` is absent,
    ``frame_median`` is the median over the full deduped matched catalog.
    """
    norm_src = matched_catalog_flux_rows(df, flux_col=flux_col)
    if norm_src.empty:
        return {}, float("nan"), False

    mag_col = "mag" if "mag" in norm_src.columns else None
    if mag_col:
        work = norm_src.copy()
        work["_mag_num"] = pd.to_numeric(work[mag_col], errors="coerce")
        work["_mag_bin"] = (work["_mag_num"] / float(mag_bin_width)).apply(
            lambda x: int(x) if math.isfinite(x) else -1
        )
        bin_meds: dict[int, float] = {}
        for b, grp in work.groupby("_mag_bin"):
            bmed = float(grp[flux_col].median())
            if math.isfinite(bmed) and bmed > 0:
                bin_meds[int(b)] = bmed
        if not bin_meds:
            return {}, float("nan"), True
        return bin_meds, float("nan"), True

    frame_med = float(norm_src[flux_col].median())
    if not math.isfinite(frame_med) or frame_med <= 0:
        return {}, float("nan"), False
    return {}, frame_med, False


def norm_med_for_bin(b: int, bin_meds: dict[int, float], bin_keys: np.ndarray) -> float:
    bi = int(b)
    if bi in bin_meds:
        return float(bin_meds[bi])
    if len(bin_keys) == 0:
        return float("nan")
    ck = int(bin_keys[int(np.argmin(np.abs(bin_keys - bi)))])
    return float(bin_meds[ck])


def assign_relative_flux(
    sub: pd.DataFrame,
    *,
    flux_col: str,
    bin_meds: dict[int, float],
    frame_med: float,
    mag_bin_width: float = 0.5,
    id_col: str,
) -> pd.DataFrame:
    """Attach ``_raw_flux``, ``_norm_med``, ``_rel`` to candidate rows (deduped)."""
    if sub.empty:
        return sub
    work = dedupe_catalog_rows(sub.copy(), id_col=id_col, flux_col=flux_col)
    work["_raw_flux"] = pd.to_numeric(work[flux_col], errors="coerce")
    if bin_meds:
        if "mag" in work.columns:
            work["_mag_num"] = pd.to_numeric(work["mag"], errors="coerce")
            work["_mag_bin"] = (work["_mag_num"] / float(mag_bin_width)).apply(
                lambda x: int(x) if math.isfinite(x) else -1
            )
        else:
            work["_mag_bin"] = -1
        _bin_keys = np.fromiter(bin_meds.keys(), dtype=np.int64)
        work["_norm_med"] = work["_mag_bin"].map(lambda b: norm_med_for_bin(b, bin_meds, _bin_keys))
    else:
        work["_norm_med"] = float(frame_med)
    work["_rel"] = work["_raw_flux"] / pd.to_numeric(work["_norm_med"], errors="coerce")
    return work


def mad_sigma(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if int(v.size) < 3:
        return float(np.std(v)) if int(v.size) > 0 else float("inf")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad > 0:
        return float(_MAD_SIGMA_SCALE * mad)
    std = float(np.std(v))
    return std if std > 0 else float("inf")


def robust_comp_rms(
    rel_flux: list[float] | np.ndarray,
    *,
    clip_sigma: float = 5.0,
    min_keep: int = 3,
    min_flux_frac: float = 0.75,
) -> float:
    """Intrinsic scatter of a comp star about its own median (MAD; no frame rejection).

    Definition (differential photometry, zero-clipping policy 2026-08-12):
    1. Input: frame-to-frame relative flux ``f_i = raw_i / norm_ref_i``.
    2. Keep every finite positive ``f_i`` (no flux-fraction drop, no sigma-clip).
    3. ``comp_rms = 1.4826 * median(|f_i - median(f)|)``.

    ``clip_sigma`` / ``min_flux_frac`` are accepted for call-site compatibility and ignored.
    Units match the historical gate (~0.01-0.05 = 1-5% fractional flux scatter).
    """
    _ = (clip_sigma, min_flux_frac)
    arr = np.asarray(rel_flux, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if int(arr.size) < min_keep:
        return float("nan")
    med = float(np.median(arr))
    resid = arr - med
    if not np.any(np.abs(resid) > 1e-9):
        return 0.0
    rms = mad_sigma(resid)
    return float(rms) if math.isfinite(rms) else float("nan")
