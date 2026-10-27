"""Plain deterministic statistics (no sigma-clip / kappa-sigma / outlier rejection).

Milan standing decision 2026-08-12: VYVAR photometry must not clip science data.
Use these helpers instead of ``astropy.stats.sigma_clipped_stats`` / ``sigma_clip``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def plain_mean_med_std(
    data: Any,
    *,
    mask: Any | None = None,
) -> tuple[float, float, float]:
    """Return (mean, median, sample_std) over finite unmasked samples.

    ``mask`` follows numpy/astropy convention: True = invalid / excluded.

    Warning: on a star field the sample ``std`` is dominated by sources and large-scale
    structure (scene variance), not sky noise. For a noise scale use
    :func:`sky_mad_sigma_adu` (SNR-GATE-01).
    """
    arr = np.asarray(data, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == arr.shape:
            arr = arr[~m]
        else:
            arr = arr.ravel()
    else:
        arr = arr.ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    med = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if int(arr.size) > 1 else 0.0
    if not math.isfinite(std):
        std = 0.0
    return mean, med, std


def sky_mad_sigma_adu(
    data: Any,
    *,
    mask: Any | None = None,
) -> tuple[float, float]:
    """Sky median and noise sigma (ADU) = (median, 1.4826 * MAD) of the unmasked pixels.

    No clipping. Stars are a minority of pixels, so median and MAD describe the sky.
    (DAO-NOISE-2026-10: the former version took the MAD of the pixels below the median
    only, which returns 0.59 sigma for Gaussian noise.)
    """
    arr = np.asarray(data, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        arr = arr[~m] if m.shape == arr.shape else arr.ravel()
    else:
        arr = arr.ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, 1.4826 * mad if math.isfinite(mad) and mad >= 0.0 else float("nan")
