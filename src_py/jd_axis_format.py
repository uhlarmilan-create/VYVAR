"""Julian date axis formatting (AIJ-style): subtract int(floor(min)) so ticks are small decimals.

Alternative AIJ style divides JD by 1e6 and appends ``M`` on tick labels; this module uses the
``Time scale - NNNNNNN`` convention (fractional part on the axis, full value in hover where wired).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Values clearly on full JD scale (not MJD ~60k, not phase 0-1, not frame index).
_JD_FULL_MIN = 2_000_000.0


def jd_offset(times: np.ndarray | pd.Series | list) -> int | None:
    """Axis offset ``int(floor(min(finite)))`` of full-JD times; None when there is no finite point or
    the values are not on the full JD scale (max < 2_000_000). G11-29: the one rule for every JD axis."""
    arr = np.asarray(pd.to_numeric(pd.Series(times), errors="coerce"), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or float(np.max(finite)) < _JD_FULL_MIN:
        return None
    return int(np.floor(float(np.min(finite))))


def jd_series_relative(times: np.ndarray | pd.Series | list) -> tuple[np.ndarray, int | None]:
    """Return ``(times - offset, offset)`` with the :func:`jd_offset` of full JD data; otherwise the
    original array (as float64) and ``None`` offset (no subtraction)."""
    arr = np.asarray(pd.to_numeric(pd.Series(times), errors="coerce"), dtype=float)
    offset = jd_offset(arr)
    if offset is None:
        return arr, None
    return arr - float(offset), offset


def jd_axis_title(short_label: str, offset: int | None) -> str:
    """Axis title like ``BJD (TDB) - 2461097`` when ``offset`` is set (AIJ-style)."""
    if offset is None:
        return short_label
    return f"{short_label} - {offset}"
