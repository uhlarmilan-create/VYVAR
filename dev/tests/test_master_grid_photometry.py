"""Tests for master-grid centroid lock at photometry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from pipeline import _lock_matched_centroids_to_master_grid


def test_lock_matched_centroids_to_master_grid_local_peak():
    arr = np.full((20, 20), 100.0, dtype=np.float64)
    arr[10, 12] = 5000.0  # bright peak offset from master ref (10, 10)
    master = pd.DataFrame({"x": [10.0], "y": [10.0]})
    x = np.array([11.0])
    y = np.array([11.0])
    matched = np.array([True])
    safe = np.array([0])
    xo, yo, n = _lock_matched_centroids_to_master_grid(
        arr, x, y, matched=matched, safe=safe, master_df=master, fwhm_px=2.5
    )
    assert n == 1
    assert float(xo[0]) == 12.0
    assert float(yo[0]) == 10.0
