"""Named sky estimators for CircularAnnulus sites (CONSOLIDATE-01B A3).

Variants that exist in production; no private formulas at call sites.
Numeric identity is the gate (G2 era04_aperture bytes).
"""
from __future__ import annotations

import math
import numpy as np

from plain_stats import plain_mean_med_std


def sky_median_mask(
    data: np.ndarray | None = None,
    ann_img: np.ndarray | None = None,
    *,
    values: np.ndarray | None = None,
    min_pix: int = 1,
) -> float:
    """Production SKY-CLIP-01: plain median of annulus-mask pixels, no rejection.

    ``data`` + ``ann_img`` is the catalog path (mask image, ``ann_img > 0``).
    ``values`` is the PSF path (pre-extracted finite samples, ``min_pix=8``).
    """
    if values is not None:
        v = np.asarray(values, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size >= int(min_pix):
            return float(np.median(v))
        return float("nan")
    if data is None or ann_img is None:
        return float("nan")
    d = np.asarray(data)
    sky_pixels = d[np.asarray(ann_img) > 0]
    if sky_pixels.size >= 1:
        return float(np.median(sky_pixels))
    return float(np.nanmedian(d))


def sky_exact_mean(
    data: np.ndarray,
    x: float,
    y: float,
    *,
    r_in: float,
    r_out: float,
) -> float:
    """Exact photutils annulus sum / area (COG path when catalog sky is missing)."""
    from photutils.aperture import CircularAnnulus
    from photutils.aperture import aperture_photometry as _aphot

    xi, yi = float(x), float(y)
    if not (math.isfinite(xi) and math.isfinite(yi)):
        return float("nan")
    if not (math.isfinite(float(r_in)) and math.isfinite(float(r_out)) and float(r_out) > float(r_in)):
        return float("nan")
    d = np.asarray(data, dtype=np.float64)
    ann = CircularAnnulus([(xi, yi)], r_in=float(r_in), r_out=float(r_out))
    tab = _aphot(d, ann, method="exact")
    area = float(ann.area)
    s = float(tab["aperture_sum"][0])
    if area > 0 and math.isfinite(s):
        return s / area
    return float("nan")


def sky_clipped_mean_med_std(
    vals: np.ndarray,
    *,
    sigma: float = 3.0,
    maxiters: int = 2,
) -> tuple[float, float, float]:
    """Forced-seed 3-sigma clipped mean/median/std (plain_mean_med_std)."""
    return plain_mean_med_std(np.asarray(vals), sigma=float(sigma), maxiters=int(maxiters))
