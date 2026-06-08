"""Rig-prior gates and gnomonic triangle sides."""

from __future__ import annotations

import math

import numpy as np

from config import AppConfig
from vyvar_blind_solver import (
    _scale_ratio_accepts,
    _side_arcsec_flat,
    _side_arcsec_gnomonic,
    _triangle_sides_arcsec,
    _use_gnomonic_triangles,
)


def test_scale_ratio_gate() -> None:
    assert _scale_ratio_accepts(100.0, 100.0, scale_tol_frac=0.10)
    assert not _scale_ratio_accepts(100.0, 130.0, scale_tol_frac=0.10)


def test_gnomonic_side_lengths_positive() -> None:
    p0 = np.array([100.0, 100.0])
    p1 = np.array([900.0, 700.0])
    scale = 9.77
    flat = _side_arcsec_flat(p0, p1, plate_scale_arcsec_per_px=scale)
    gno = _side_arcsec_gnomonic(p0, p1, x_cen=1000.0, y_cen=750.0, plate_scale_arcsec_per_px=scale)
    assert flat > 100.0 and gno > 100.0


def test_gnomonic_enabled_for_wide_fov() -> None:
    cfg = AppConfig()
    assert _use_gnomonic_triangles(5.5, use_rig_prior=cfg.blind_use_rig_prior)
    assert not _use_gnomonic_triangles(1.1, use_rig_prior=True)


def test_triangle_sides_order() -> None:
    p0, p1, p2 = np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([5.0, 8.0])
    L1, L2, L3 = _triangle_sides_arcsec(
        p0, p1, p2, x_cen=5.0, y_cen=4.0, plate_scale_arcsec_per_px=1.3, use_gnomonic=False
    )
    assert L1 <= L2 <= L3
    assert L3 > 2.0 * 1.3
