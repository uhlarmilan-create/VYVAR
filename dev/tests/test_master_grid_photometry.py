"""Tests for master-grid centroid lock at photometry (IDENT-JUMP-01)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from pipeline import _lock_matched_centroids_to_master_grid


def test_lock_stays_on_master_when_neighbour_brighter():
    """Pre-fail under old 2.5-FWHM peak search; post-pass: stay on target.

    Synthetic aligned frame: faint target at master (10, 10), brighter neighbour
    14 px away, target undetected as a distinct peak -> position must remain on
    the target (master reference), not jump to the neighbour.
    """
    arr = np.full((40, 40), 100.0, dtype=np.float64)
    # Neighbour peak 14 px to the right (classic IDENT-JUMP distance).
    arr[10, 24] = 8000.0
    # Faint / absent target peak at master (sky only).
    master = pd.DataFrame({"x": [10.0], "y": [10.0]})
    x = np.array([11.0])
    y = np.array([11.0])
    matched = np.array([True])
    safe = np.array([0])
    xo, yo, n = _lock_matched_centroids_to_master_grid(
        arr, x, y, matched=matched, safe=safe, master_df=master, fwhm_px=5.4
    )
    assert n == 1
    assert abs(float(xo[0]) - 10.0) <= 1.0 + 1e-9
    assert abs(float(yo[0]) - 10.0) <= 1.0 + 1e-9
    assert float(xo[0]) != 24.0


def test_lock_allows_subpixel_peak_within_bound():
    """Bright peak 1 px from master may be taken (within refine_bound_px)."""
    arr = np.full((20, 20), 100.0, dtype=np.float64)
    arr[10, 11] = 5000.0
    master = pd.DataFrame({"x": [10.0], "y": [10.0]})
    x = np.array([10.0])
    y = np.array([10.0])
    matched = np.array([True])
    safe = np.array([0])
    xo, yo, n = _lock_matched_centroids_to_master_grid(
        arr, x, y, matched=matched, safe=safe, master_df=master, fwhm_px=2.5
    )
    assert n == 1
    assert float(xo[0]) == 11.0
    assert float(yo[0]) == 10.0
