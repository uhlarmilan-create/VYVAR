# -*- coding: ascii -*-
"""One implementation of on-sky / pixel distances for comparison-star ``_dist_deg``.

Comparison-star selection (``comp_selection_per_target``) persists ``_dist_deg``. A second scalar haversine
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


#: decimal places of DIST_DEG_QUANTUM (9)
DIST_DEG_NDIGITS = int(round(-math.log10(DIST_DEG_QUANTUM)))


def quantize_dist_deg(values: np.ndarray | float) -> np.ndarray | float:
    """Round finite distances to DIST_DEG_QUANTUM (1e-9 deg, 9 decimal places); keep 999.0 and NaN.

    Python's correctly rounded ``round(x, 9)`` per value (G02-43: ``np.round`` rounds
    ``rint(x * 1e9) / 1e9`` and differs at half-quantum ties); the loop runs on a Python list."""
    scalar = isinstance(values, (float, int, np.floating, np.integer)) or (
        isinstance(values, np.ndarray) and values.ndim == 0
    )
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    nd = DIST_DEG_NDIGITS
    out = np.asarray(
        [
            (DIST_DEG_INVALID if x >= 900.0 else round(x, nd)) if math.isfinite(x) else math.nan
            for x in arr.ravel().tolist()
        ],
        dtype=np.float64,
    ).reshape(arr.shape)
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
