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
    **_ignored: Any,
) -> tuple[float, float, float]:
    """Return (mean, median, sample_std) over finite unmasked samples.

    ``mask`` follows numpy/astropy convention: True = invalid / excluded.
    Extra kwargs (e.g. legacy ``sigma`` / ``maxiters``) are ignored.
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
