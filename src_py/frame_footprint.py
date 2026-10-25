"""Data footprint of an aligned frame on the reference grid (EPSF-FOOTPRINT-2026-10, AP-FOOTPRINT-2026-10).

The aligner resamples every frame onto the MASTERSTAR grid and fills the reference pixels the
shifted frame does not cover with a constant (Brno 533 r_0199: x < 48 px). Photometry must not
measure stars or sky on that fill.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def has_alignment_transform(header: Any) -> bool:
    """True when the header carries the frame->reference transform ``VY_AM0..8``."""
    return header is not None and all(f"VY_AM{k}" in header for k in range(9))


def aligned_footprint(header: Any, shape: tuple[int, int]) -> np.ndarray:
    """True on the aligned-grid pixels that hold data of the frame.

    The frame's corners are mapped through the frame->reference transform ``VY_AM0..8``
    (ALIGN-MATRIX-2026-10); the pixels inside that quadrilateral, at least 1 px from its edges, are
    covered (checked on 533 r: first data column 30 / 48 px vs quadrilateral 29.4-33.8 / 47.8-51.7 px).
    Raises when the transform is missing.
    """
    try:
        mat = np.array([float(header[f"VY_AM{k}"]) for k in range(9)], dtype=np.float64).reshape(3, 3)
    except KeyError as exc:
        raise ValueError(f"aligned frame has no frame->reference transform ({exc.args[0]}) - re-run the alignment") from exc
    h, w = int(shape[0]), int(shape[1])
    wr, hr = int(header.get("NAXIS1", w)), int(header.get("NAXIS2", h))
    corners = np.array([[0, 0, 1], [wr - 1, 0, 1], [wr - 1, hr - 1, 1], [0, hr - 1, 1]], dtype=np.float64).T
    q = mat @ corners
    quad = (q[:2] / q[2]).T
    # SPEED-2026-10 (S2): broadcast row / column coordinates (same float operations as a full mgrid).
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    area = 0.5 * np.sum(quad[:, 0] * np.roll(quad[:, 1], -1) - np.roll(quad[:, 0], -1) * quad[:, 1])
    sgn = 1.0 if area > 0 else -1.0
    inside = np.ones((h, w), dtype=bool)
    for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=False):
        ex, ey = b - a
        inside &= sgn * (ex * (yy - a[1]) - ey * (xx - a[0])) / math.hypot(ex, ey) >= 1.0
    return inside


def footprint_bbox(header: Any, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Axis-aligned rectangle (x0, y0, x1, y1) inside the frame's data footprint on the aligned grid
    (BORDER-FOOTPRINT-2026-10). The frame's corners are mapped through ``VY_AM0..8``; the second
    smallest / second largest mapped x and y bound the rectangle, so it stays inside a slightly
    rotated (or 180 deg flipped) quadrilateral; 1 px inside its edges like :func:`aligned_footprint`.
    Raises when the transform is missing."""
    try:
        mat = np.array([float(header[f"VY_AM{k}"]) for k in range(9)], dtype=np.float64).reshape(3, 3)
    except KeyError as exc:
        raise ValueError(f"aligned frame has no frame->reference transform ({exc.args[0]}) - re-run the alignment") from exc
    h, w = int(shape[0]), int(shape[1])
    wr, hr = int(header.get("NAXIS1", w)), int(header.get("NAXIS2", h))
    corners = np.array([[0, 0, 1], [wr - 1, 0, 1], [wr - 1, hr - 1, 1], [0, hr - 1, 1]], dtype=np.float64).T
    q = mat @ corners
    xs = np.sort(q[0] / q[2])
    ys = np.sort(q[1] / q[2])
    return (
        max(0.0, float(xs[1]) + 1.0),
        max(0.0, float(ys[1]) + 1.0),
        min(float(w - 1), float(xs[2]) - 1.0),
        min(float(h - 1), float(ys[2]) - 1.0),
    )


def off_footprint(
    off: np.ndarray, xs: np.ndarray, ys: np.ndarray, r_core: float, r_in: float, r_out: float, *, square_core: bool = False
) -> np.ndarray:
    """True where the measuring core (fit box of half-size ``r_core`` when ``square_core``, else the
    aperture circle of radius ``r_core``) touches ``off`` pixels or under half of the sky annulus
    ``r_in``-``r_out`` lies on data."""
    n = len(xs)
    out = np.zeros(n, dtype=bool)
    if not off.any():
        return out
    h, w = off.shape
    r = int(math.ceil(max(r_out, r_core))) + 1
    for i in range(n):
        x, y = float(xs[i]), float(ys[i])
        if not (math.isfinite(x) and math.isfinite(y)):
            out[i] = True
            continue
        xi, yi = int(round(x)), int(round(y))
        yy, xx = np.mgrid[yi - r : yi + r + 1, xi - r : xi + r + 1]
        inb = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        offw = np.ones(yy.shape, dtype=bool)
        offw[inb] = off[yy[inb], xx[inb]]
        if square_core:
            core = (np.abs(xx - xi) <= r_core) & (np.abs(yy - yi) <= r_core)
        else:
            core = np.hypot(xx - x, yy - y) <= r_core
        if (core & offw).any():
            out[i] = True
            continue
        rr = np.hypot(xx - x, yy - y)
        ann = (rr >= r_in) & (rr <= r_out)
        out[i] = bool((ann & ~offw).sum() < 0.5 * ann.sum())
    return out
