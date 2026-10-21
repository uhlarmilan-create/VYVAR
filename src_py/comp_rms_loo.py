"""Leave-one-out differential-mag RMS for comparison candidacy (COMP-RMS-DEF-01-B).

Gated statistic is MAD*1.4826 of mag(star) - mag(median of other pool stars)
on loadable proc frames. No clipping. The old mag-bin relative-flux MAD is
``comp_relflux_mad`` (diagnostic only).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from comp_pool_rms import _norm_id_val, sort_per_frame_csv_paths
from stats_core import _flux_to_mag

LOGGER = logging.getLogger(__name__)

LN10_OVER_2P5 = 1.0857362047581294
COMP_RMS_LOO_PHOTON_K_DEFAULT = 5.0
COMP_RMS_FRAMES_BASIS = "all_loadable"


def mad_sigma_scale_or_zero(values: np.ndarray) -> float:
    """1.4826*MAD after dropping non-finite; n<3 returns nan; MAD==0 returns 0.0."""
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan")
    med = float(np.median(a))
    return float(1.4826 * np.median(np.abs(a - med)))


def _loo_median(v: np.ndarray) -> np.ndarray:
    """Median of all OTHER elements, for every element (vectorised, exact)."""
    v = np.asarray(v, dtype=np.float64)
    m = int(v.size)
    if m < 2:
        return np.full(m, np.nan)
    order = np.argsort(v, kind="mergesort")
    srt = v[order]
    rank = np.empty(m, dtype=np.int64)
    rank[order] = np.arange(m)
    n2 = m - 1

    def _pick(j: int) -> np.ndarray:
        # j-th element (0-based) of the sorted array with the element of rank r removed
        idx = np.where(j < rank, j, j + 1)
        return srt[idx]

    if n2 % 2 == 1:
        return _pick(n2 // 2)
    return 0.5 * (_pick(n2 // 2 - 1) + _pick(n2 // 2))


def compute_loo_mag_rms_map(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.3,
    strict_flux_col: bool = False,
) -> tuple[dict[str, float], str]:
    """Return ``{normalized_id: MAD*1.4826 of LOO dmag}`` and the frames-basis stamp.

    Pool is ``cand_ids`` excluding the star itself. Frames: every loadable proc
    CSV in ``csv_cache`` (no separate QC-admit list in the draft manifest).
    """
    if not cand_ids:
        return {}, COMP_RMS_FRAMES_BASIS
    norm_ids = sorted({_norm_id_val(x) for x in cand_ids if _norm_id_val(x)})
    if len(norm_ids) < 2:
        return {}, COMP_RMS_FRAMES_BASIS
    id_set = set(norm_ids)
    dmag_map: dict[str, list[float]] = {cid: [] for cid in norm_ids}
    n_frames = 0
    for csv_path in sort_per_frame_csv_paths(per_frame_csv_paths, csv_cache):
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        if strict_flux_col and flux_col not in df.columns:
            continue  # never mix apertures between frames
        actual_flux = flux_col if flux_col in df.columns else "dao_flux"
        if actual_flux not in df.columns:
            continue
        id_col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
        if id_col is None:
            continue
        work = df[[id_col, actual_flux]].copy()
        work["_nid"] = work[id_col].map(_norm_id_val)
        work["_flux"] = pd.to_numeric(work[actual_flux], errors="coerce")
        work = work[work["_nid"].isin(id_set) & work["_flux"].gt(0)]
        if work.empty:
            continue
        flux_by = work.groupby("_nid", sort=True)["_flux"].median()
        if len(flux_by) < 2:
            continue
        n_frames += 1
        cids_f = list(flux_by.index)
        mags = np.asarray([_flux_to_mag(float(f)) for f in flux_by.to_numpy()], dtype=np.float64)
        okm = np.isfinite(mags)
        if int(okm.sum()) < 2:
            continue
        cids_f = [c for c, o in zip(cids_f, okm, strict=False) if o]
        mags = mags[okm]
        loo = _loo_median(mags)
        for cid, mag_s, mag_c in zip(cids_f, mags, loo, strict=False):
            if math.isfinite(mag_c):
                dmag_map[cid].append(float(mag_s - mag_c))

    min_frames = max(3, int(n_frames * float(min_frames_frac)))
    out: dict[str, float] = {}
    for cid, vals in dmag_map.items():
        if len(vals) < min_frames:
            continue
        rms = mad_sigma_scale_or_zero(np.asarray(vals, dtype=np.float64))
        if math.isfinite(rms):
            out[cid] = rms
    LOGGER.info(
        "[COMP-RMS-LOO] frames_basis=%s n_loaded=%d n_stars=%d n_with_rms=%d",
        COMP_RMS_FRAMES_BASIS,
        n_frames,
        len(norm_ids),
        len(out),
    )
    return out, COMP_RMS_FRAMES_BASIS


def saturated_any_frame_ids(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
) -> dict[str, int]:
    """SAT-LIN-2026-10: ``{normalized_id: n_frames}`` for stars saturated on >= 1 frame.

    A frame counts when ``likely_saturated`` is true (raw peak >= LINEARITY_FRAC x SAT,
    set by sat_diag) or, without that column, ``peak_max_adu >= saturate_limit_adu_85pct``.
    A comparison star that leaves the linear range on any frame biases every ratio it
    enters, so it is not a comparison star for this night.
    """
    if not cand_ids:
        return {}
    id_set = {_norm_id_val(x) for x in cand_ids if _norm_id_val(x)}
    out: dict[str, int] = {}
    for csv_path in sort_per_frame_csv_paths(per_frame_csv_paths, csv_cache):
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        id_col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
        if id_col is None:
            continue
        if "likely_saturated" in df.columns:
            flag = df["likely_saturated"].map(
                lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes")
            ).to_numpy(dtype=bool)
        elif "peak_max_adu" in df.columns and "saturate_limit_adu_85pct" in df.columns:
            pk = pd.to_numeric(df["peak_max_adu"], errors="coerce").to_numpy(dtype=np.float64)
            lim = pd.to_numeric(df["saturate_limit_adu_85pct"], errors="coerce").to_numpy(dtype=np.float64)
            flag = np.isfinite(pk) & np.isfinite(lim) & (lim > 0) & (pk >= lim)
        else:
            continue
        if not flag.any():
            continue
        for v in df.loc[flag, id_col].tolist():
            nid = _norm_id_val(v)
            if nid in id_set:
                out[nid] = out.get(nid, 0) + 1
    return out


def choose_field_aperture_k(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    *,
    min_frames_frac: float = 0.3,
) -> tuple[float | None, dict[str, float], dict[float, float]]:
    """APERTURE-K-2026-10: field aperture k for comparison-star RMS.

    For every k of the light-curve aperture grid (proc columns ``flux_kNNN``) compute the
    LOO magnitude RMS of the pool; the field k is the one with the smallest median RMS over
    the stars measured at every k (ties within 1 % -> larger k, more robust to seeing).
    Returns (k, loo_map at k, {k: median rms}); (None, {}, {}) without grid columns.
    """
    from aperture_pertarget import DEFAULT_APERTURE_F_GRID  # noqa: PLC0415
    from photometry_shared import flux_k_column  # noqa: PLC0415

    have: set[float] | None = None
    for csv_path in per_frame_csv_paths:
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        ks = {k for k in DEFAULT_APERTURE_F_GRID if flux_k_column(k) in df.columns}
        have = ks if have is None else (have & ks)
    if not have:
        return None, {}, {}
    maps: dict[float, dict[str, float]] = {}
    for k in sorted(have):
        maps[k], _ = compute_loo_mag_rms_map(
            cand_ids,
            per_frame_csv_paths,
            csv_cache,
            flux_col=flux_k_column(k),
            min_frames_frac=min_frames_frac,
            strict_flux_col=True,
        )
    common = set.intersection(*(set(m) for m in maps.values())) if maps else set()
    if len(common) < 3:
        return None, {}, {}
    scores = {k: float(np.median([maps[k][c] for c in common])) for k in maps}
    best = min(scores.values())
    k_best = max(k for k, v in scores.items() if v <= best * 1.01)
    LOGGER.info(
        "[APERTURE-K] field k for comparison RMS = %.2f (median LOO rms by k: %s, n_stars=%d)",
        k_best,
        ", ".join(f"{k:.2f}:{v:.4f}" for k, v in sorted(scores.items())),
        len(common),
    )
    return float(k_best), maps[k_best], scores
