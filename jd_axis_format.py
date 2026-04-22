"""Julian date axis formatting (AIJ-style): subtract int(floor(min)) so ticks are small decimals.

Alternative AIJ style divides JD by 1e6 and appends ``M`` on tick labels; this module uses the
``Time scale - NNNNNNN`` convention (fractional part on the axis, full value in hover where wired).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Values clearly on full JD scale (not MJD ~60k, not phase 0–1, not frame index).
_JD_FULL_MIN = 2_000_000.0


def jd_series_relative(times: np.ndarray | pd.Series | list) -> tuple[np.ndarray, int | None]:
    """Return ``(times - offset, offset)`` with ``offset = int(floor(min(finite))))`` for full JD data.

    If there are no finite points or values are not on the full JD scale (max < 2_000_000),
    returns the original array (as float64) and ``None`` offset (no subtraction).
    """
    arr = np.asarray(pd.to_numeric(pd.Series(times), errors="coerce"), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, None
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi < _JD_FULL_MIN:
        return arr, None
    offset = int(np.floor(lo))
    return arr - float(offset), offset


def jd_axis_title(short_label: str, offset: int | None) -> str:
    """Axis title like ``BJD (TDB) - 2461097`` when ``offset`` is set (AIJ-style)."""
    if offset is None:
        return short_label
    return f"{short_label} - {offset}"
