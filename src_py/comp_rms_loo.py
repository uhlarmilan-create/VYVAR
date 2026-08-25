"""Leave-one-out differential-mag RMS for comparison candidacy (COMP-RMS-DEF-01-B).

Gated statistic is MAD*1.4826 of mag(star) - mag(median of other pool stars)
on loadable proc frames. No clipping. The old mag-bin relative-flux MAD is
``comp_relflux_mad`` (diagnostic only).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from comp_pool_rms import _norm_id_val, sort_per_frame_csv_paths

LOGGER = logging.getLogger(__name__)

LN10_OVER_2P5 = 1.0857362047581294
COMP_RMS_LOO_PHOTON_K_DEFAULT = 5.0
COMP_RMS_FRAMES_BASIS = "all_loadable"


def photon_sigma_mag(snr_ap_pixscaled: Any) -> float:
    snr = float(pd.to_numeric(snr_ap_pixscaled, errors="coerce"))
    if not (math.isfinite(snr) and snr > 0):
        return float("nan")
    return LN10_OVER_2P5 / snr


def mad_sigma(values: np.ndarray) -> float:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan")
    med = float(np.median(a))
    return float(1.4826 * np.median(np.abs(a - med)))


def _flux_to_mag(flux: float) -> float:
    if not (math.isfinite(flux) and flux > 0):
        return float("nan")
    return -2.5 * math.log10(flux)


def compute_loo_mag_rms_map(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.3,
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
        actual_flux = flux_col if flux_col in df.columns else "flux"
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
        mag_by = {cid: _flux_to_mag(float(f)) for cid, f in flux_by.items()}
        for cid, mag_s in mag_by.items():
            if not math.isfinite(mag_s):
                continue
            others = [mag_by[o] for o in mag_by if o != cid and math.isfinite(mag_by[o])]
            if not others:
                continue
            mag_c = float(np.median(np.asarray(others, dtype=np.float64)))
            if not math.isfinite(mag_c):
                continue
            dmag_map[cid].append(mag_s - mag_c)

    min_frames = max(3, int(n_frames * float(min_frames_frac)))
    out: dict[str, float] = {}
    for cid, vals in dmag_map.items():
        if len(vals) < min_frames:
            continue
        rms = mad_sigma(np.asarray(vals, dtype=np.float64))
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


def loo_ceiling_mag(
    snr_ap_pixscaled: Any,
    *,
    k: float = COMP_RMS_LOO_PHOTON_K_DEFAULT,
    abs_max: float = 0.1,
) -> float:
    """Pass if loo_rms <= this value. Raises if SNR is missing/non-finite."""
    ph = photon_sigma_mag(snr_ap_pixscaled)
    if not math.isfinite(ph):
        raise ValueError(
            "INV-COMP-RMS-01: snr_ap_pixscaled missing or non-positive; "
            "refusing a default photon sigma"
        )
    k_eff = float(k) if math.isfinite(float(k)) and float(k) > 0 else COMP_RMS_LOO_PHOTON_K_DEFAULT
    abs_eff = float(abs_max) if math.isfinite(float(abs_max)) and float(abs_max) > 0 else 0.1
    return float(min(abs_eff, k_eff * ph))
