"""Data footprint of an aligned frame on the reference grid (EPSF-FOOTPRINT-2026-10, AP-FOOTPRINT-2026-10).

The aligner resamples every frame onto the MASTERSTAR grid and fills the reference pixels the
shifted frame does not cover with a constant (Brno 533 r_0199: x < 48 px). Photometry must not
measure stars or sky on that fill.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def has_alignment_transform(header: Any) -> bool:
    """True when the header carries the frame->reference transform ``VY_AM0..8``."""
    return header is not None and all(f"VY_AM{k}" in header for k in range(9))


def stamp_alignment_transform(header: Any, registration: dict[str, Any], raw_shape: tuple[int, int]) -> None:
    """Write the frame->reference transform of an aligned frame into its header (every alignment
    method, OSC channel handoff included - G02-03): ``VY_AM0..8`` (row-major 3x3), ``VY_AMSRC``,
    ``VY_AMERR`` (max px error of VY_AM vs a polynomial registration; absent = VY_AM is exact) and the
    frame's own size ``VY_RAWNX`` / ``VY_RAWNY`` (G02-34: the aligned header's NAXIS is the
    reference size). Raises when the registration has no finite 3x3 matrix - an aligned frame
    without its transform cannot be footprint-checked."""
    mat = np.asarray((registration or {}).get("matrix") if isinstance(registration, dict) else [], dtype=np.float64)
    if mat.size != 9 or not np.all(np.isfinite(mat)):
        raise ValueError("aligned frame has no finite frame->reference matrix")
    for k, v in enumerate(mat.reshape(-1)):
        header[f"VY_AM{k}"] = (float(v), "frame->reference transform (row-major 3x3)")
    header["VY_AMSRC"] = (str(registration.get("method") or "astroalign")[:60], "transform source")
    err = registration.get("matrix_max_err_px")
    if err is not None:
        # ALIGN-POLY-2026-10: VY_AM is the projective approximation of the polynomial
        header["VY_AMERR"] = (float(err), "max px error of VY_AM vs the polynomial")
    elif "VY_AMERR" in header:
        del header["VY_AMERR"]
    header["VY_RAWNX"] = (int(raw_shape[1]), "frame width before alignment (px)")
    header["VY_RAWNY"] = (int(raw_shape[0]), "frame height before alignment (px)")


def _transform_raw_size_margin(header: Any, shape: tuple[int, int]) -> tuple[np.ndarray, int, int, float]:
    """VY_AM matrix, the frame's own size and the footprint margin (px).

    Raw size: ``VY_RAWNX`` / ``VY_RAWNY`` (G02-34); frames aligned before those keys existed carry
    only NAXIS1/2, which astropy rewrites to the aligned (reference) size - right only when the frame
    has the reference's size (one camera / binning, as every alignment before 2026-10-29).
    Margin: 1 px plus ``VY_AMERR`` (G02-24: the projective VY_AM of a polynomial registration is off by
    up to VY_AMERR px; the 1 px margin alone assumed that error < 1 px)."""
    try:
        mat = np.array([float(header[f"VY_AM{k}"]) for k in range(9)], dtype=np.float64).reshape(3, 3)
    except KeyError as exc:
        raise ValueError(f"aligned frame has no frame->reference transform ({exc.args[0]}) - re-run the alignment") from exc
    h, w = int(shape[0]), int(shape[1])
    if "VY_RAWNX" in header and "VY_RAWNY" in header:
        wr, hr = int(header["VY_RAWNX"]), int(header["VY_RAWNY"])
    else:
        wr, hr = int(header.get("NAXIS1", w)), int(header.get("NAXIS2", h))
    err = float(header.get("VY_AMERR", 0.0) or 0.0)
    if not (math.isfinite(err) and err >= 0.0):
        raise ValueError(f"VY_AMERR={header.get('VY_AMERR')!r} is not a finite error")
    return mat, wr, hr, 1.0 + err


def aligned_footprint(header: Any, shape: tuple[int, int]) -> np.ndarray:
    """True on the aligned-grid pixels that hold data of the frame.

    The frame's corners are mapped through the frame->reference transform ``VY_AM0..8``
    (ALIGN-MATRIX-2026-10); the pixels inside that quadrilateral, at least 1 px from its edges, are
    covered (checked on 533 r: first data column 30 / 48 px vs quadrilateral 29.4-33.8 / 47.8-51.7 px;
    plus VY_AMERR for a polynomial registration, see :func:`_transform_raw_size_margin`).
    Raises when the transform is missing.
    """
    mat, wr, hr, margin = _transform_raw_size_margin(header, shape)
    h, w = int(shape[0]), int(shape[1])
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
        inside &= sgn * (ex * (yy - a[1]) - ey * (xx - a[0])) / math.hypot(ex, ey) >= margin
    return inside


def footprint_bbox(header: Any, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Axis-aligned rectangle (x0, y0, x1, y1) inside the frame's data footprint on the aligned grid
    (BORDER-FOOTPRINT-2026-10). The frame's corners are mapped through ``VY_AM0..8``; the second
    smallest / second largest mapped x and y bound the rectangle, so it stays inside a slightly
    rotated (or 180 deg flipped) quadrilateral; inside its edges by the margin of
    :func:`aligned_footprint`. Raises when the transform is missing."""
    mat, wr, hr, margin = _transform_raw_size_margin(header, shape)
    h, w = int(shape[0]), int(shape[1])
    corners = np.array([[0, 0, 1], [wr - 1, 0, 1], [wr - 1, hr - 1, 1], [0, hr - 1, 1]], dtype=np.float64).T
    q = mat @ corners
    xs = np.sort(q[0] / q[2])
    ys = np.sort(q[1] / q[2])
    return (
        max(0.0, float(xs[1]) + margin),
        max(0.0, float(ys[1]) + margin),
        min(float(w - 1), float(xs[2]) - margin),
        min(float(h - 1), float(ys[2]) - margin),
    )


def off_footprint(
    off: np.ndarray, xs: np.ndarray, ys: np.ndarray, r_core: float, r_in: float, r_out: float, *, square_core: bool = False
) -> np.ndarray:
    """True where the measuring core (fit box of half-size ``r_core`` when ``square_core``, else the
    aperture circle of radius ``r_core``) touches ``off`` pixels or under half of the sky annulus
    ``r_in``-``r_out`` lies on data."""
    n = len(xs)
    out = np.zeros(n, dtype=bool)
    # NEW-2 (step-4 triage): no early return for a frame that covers the whole chip - pixels outside the
    # array are "off" as well, so the chip edge is tested on every frame
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


# --- moved from variability_field (LAYER-2026-10) ---


def _frame_footprints(frames_dir: Path, stems: list[str]) -> list[np.ndarray]:
    """Corners of every frame mapped onto the reference grid (VY_AM0..8 frame->ref)."""
    from astropy.io import fits  # noqa: PLC0415

    out = []
    for st in stems:
        fp = Path(frames_dir) / f"{st}.fits"
        if not fp.is_file():
            continue
        try:
            h = fits.getheader(fp)
            mat, w, hh, _margin = _transform_raw_size_margin(h, (int(h["NAXIS2"]), int(h["NAXIS1"])))
        except Exception:  # noqa: BLE001
            continue
        c = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, hh - 1, 1], [0, hh - 1, 1]], float).T
        q = mat @ c
        out.append((q[:2] / q[2]).T)
    return out


def _inside_all(x: np.ndarray, y: np.ndarray, r: float, quads: list[np.ndarray]) -> np.ndarray:
    """True where the circle (x, y, r) lies inside every quadrilateral."""
    ok = np.ones(x.size, bool)
    for q in quads:
        area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
        sgn = 1.0 if area > 0 else -1.0
        for a, b in zip(q, np.roll(q, -1, axis=0), strict=False):
            ex, ey = b - a
            ln = math.hypot(ex, ey)
            d = sgn * (ex * (y - a[1]) - ey * (x - a[0])) / ln  # signed distance, + inside
            ok &= d >= r
    return ok
