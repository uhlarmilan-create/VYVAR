# -*- coding: ascii -*-
"""One implementation of on-sky / pixel distances for comparison-star ``_dist_deg``.

Phase-1 greedy selection (``comp_selection_per_target``) and the pin overlay
(``pinned_ensembles``) both persisted ``_dist_deg``. A third scalar haversine
lived in ``photometry_core`` (``math`` vs numpy). Full-chain vs photometry-only
then differed at ~1e-14 on a handful of CSV rows (SEL-GHOST-01 B3 T3-P5).

Persisted ``_dist_deg`` is rounded to ``DIST_DEG_QUANTUM`` (1e-9 deg). That
quantum is far below any selection or weight use; it is there because the
IEEE evaluation path (numpy vectorized vs math scalar, plus pandas CSV
round-trip) is path-dependent at ~1e-14 deg. Sentinel 999.0 (invalid
coords) is not rounded.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DIST_DEG_QUANTUM = 1e-9
DIST_DEG_INVALID = 999.0


def quantize_dist_deg(values: np.ndarray | float) -> np.ndarray | float:
    """Round finite distances to 1e-9 deg (9 decimal places); keep 999.0 and NaN."""
    scalar = isinstance(values, (float, int, np.floating, np.integer)) or (
        isinstance(values, np.ndarray) and values.ndim == 0
    )
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    out = np.empty(arr.shape, dtype=np.float64)
    flat = out.ravel()
    for i, x in enumerate(arr.ravel()):
        if not np.isfinite(x):
            flat[i] = np.nan
        elif float(x) >= 900.0:
            flat[i] = DIST_DEG_INVALID
        else:
            flat[i] = round(float(x), 9)
    if scalar:
        return float(out.ravel()[0])
    return out


def _dist_deg_csv_token(x: float) -> str:
    """Stable CSV token: 9 fractional digits, or empty/999.0."""
    if not np.isfinite(x):
        return ""
    if float(x) >= 900.0:
        return "999.0"
    return f"{round(float(x), 9):.9f}"


def persist_dist_deg_column(df: pd.DataFrame | None) -> None:
    """In-place: ``_dist_deg`` as 9-decimal strings so CSV bytes are path-independent."""
    if df is None or "_dist_deg" not in getattr(df, "columns", ()):
        return
    arr = pd.to_numeric(df["_dist_deg"], errors="coerce").to_numpy(dtype=np.float64)
    df["_dist_deg"] = [_dist_deg_csv_token(float(x)) for x in arr]


def pixel_distance_deg_vectorized(
    x_t: float,
    y_t: float,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    plate_scale_arcsec: float,
) -> np.ndarray:
    """Euclidean pixel distance converted to degrees via plate scale; invalid -> 999.0."""
    x2 = np.asarray(x_arr, dtype=np.float64)
    y2 = np.asarray(y_arr, dtype=np.float64)
    scale = float(plate_scale_arcsec)
    if not math.isfinite(scale) or scale <= 0:
        return np.full(x2.shape, DIST_DEG_INVALID, dtype=np.float64)
    _bad_xt = not math.isfinite(float(x_t))
    _bad_yt = not math.isfinite(float(y_t))
    bad = ~np.isfinite(x2) | ~np.isfinite(y2) | _bad_xt | _bad_yt
    dist_px = np.hypot(x2 - float(x_t), y2 - float(y_t))
    dist_deg = dist_px * scale / 3600.0
    return np.where(bad, DIST_DEG_INVALID, dist_deg)


def angular_distance_deg_vectorized(
    ra_t: float, dec_t: float, ra_arr: np.ndarray, dec_arr: np.ndarray
) -> np.ndarray:
    """Haversine distance (deg); invalid coords -> 999.0.

    This is the only on-sky formula used for ``_dist_deg``. Scalar callers
    go through :func:`angular_distance_deg`, which wraps this array path.
    """
    ra2 = np.asarray(ra_arr, dtype=np.float64)
    de2 = np.asarray(dec_arr, dtype=np.float64)
    ra1 = float(ra_t)
    de1 = float(dec_t)
    bad = (
        ~np.isfinite(ra2)
        | ~np.isfinite(de2)
        | (not math.isfinite(ra1))
        | (not math.isfinite(de1))
    )
    r1 = math.radians(ra1)
    d1 = math.radians(de1)
    r2 = np.radians(ra2)
    d2 = np.radians(de2)
    a = (
        np.sin((d2 - d1) / 2.0) ** 2
        + math.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2.0) ** 2
    )
    dist = np.degrees(2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.clip(a, 0.0, 1.0)))))
    return np.where(bad, DIST_DEG_INVALID, dist)


def angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Scalar haversine (deg); same formula as the vectorized path."""
    out = angular_distance_deg_vectorized(
        float(ra1),
        float(dec1),
        np.asarray([float(ra2)], dtype=np.float64),
        np.asarray([float(dec2)], dtype=np.float64),
    )
    return float(out[0])
