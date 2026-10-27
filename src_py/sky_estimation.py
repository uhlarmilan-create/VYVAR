"""Named sky estimators for CircularAnnulus sites (CONSOLIDATE-01B A3).

Variants that exist in production; no private formulas at call sites.
Numeric identity is the gate (G2 era04_aperture bytes).
"""
from __future__ import annotations

import numpy as np



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


# SKY-SRC-MASK-2026-10: local sky without the light of resolved stars.
# In a crowded field the annulus holds the wings of neighbouring stars; their light grows with the seeing
# (the annulus scales with the FWHM) and the plain median of all annulus pixels overestimates the sky. The
# error is a constant number of ADU per pixel, times the aperture area: a faint star loses a fixed flux and
# gets a magnitude-dependent airmass trend (Brno 534: -24 / -63 / -158 mmag per airmass at G 15-16 / 16-17
# / 17-18). Pixels of detected sources are excluded from the sky (photutils guide: mask sources rather than
# clip); the unresolved background stays in the sky, as it is in the aperture.
SKY_SRC_NSIGMA = 5.0
SKY_SRC_NPIXELS = 5


# the frame median and MAD sigma of the mask threshold are taken on every SKY_STATS_STEP-th pixel in x and y
# (0.6 Mpx of a 9.5 Mpx frame: same values to 2e-4, 100x faster)
SKY_STATS_STEP = 4


def disk_dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Binary dilation by a disk of ``radius`` px (exact; equals distance-to-mask <= radius).

    Row decomposition: the disk is a stack of horizontal segments, each a 1-D running maximum
    (``maximum_filter1d``) shifted in y - about 7x faster than ``binary_dilation`` with the disk.
    """
    from scipy.ndimage import maximum_filter1d  # noqa: PLC0415

    m = np.asarray(mask, dtype=bool)
    r = float(radius)
    rr = int(np.floor(r))
    h = m.shape[0]
    out = np.zeros_like(m)
    u = m.view(np.uint8)
    rows: dict[int, np.ndarray] = {}
    for dy in range(-rr, rr + 1):
        hw = int(np.floor(np.sqrt(max(r * r - dy * dy, 0.0))))
        if hw not in rows:
            rows[hw] = maximum_filter1d(u, size=2 * hw + 1, axis=1).astype(bool)
        row = rows[hw]
        if dy >= 0:
            out[dy:, :] |= row[: h - dy, :]
        else:
            out[: h + dy, :] |= row[-dy:, :]
    return out


def sky_source_mask(data: np.ndarray, fwhm_px: float, *, valid: np.ndarray | None = None) -> np.ndarray:
    """Pixels of detected sources, grown by one FWHM (True = not sky).

    Sources: photutils ``detect_sources`` on data minus the frame median, threshold ``SKY_SRC_NSIGMA`` x the
    MAD sigma of the frame, groups of >= ``SKY_SRC_NPIXELS`` connected pixels. No clipping.
    """
    from photutils.segmentation import detect_sources  # noqa: PLC0415

    d = np.asarray(data, dtype=np.float64)
    ok = np.isfinite(d) if valid is None else (np.isfinite(d) & np.asarray(valid, dtype=bool))
    v = d[::SKY_STATS_STEP, ::SKY_STATS_STEP][ok[::SKY_STATS_STEP, ::SKY_STATS_STEP]]
    if v.size == 0:
        return np.zeros(d.shape, dtype=bool)
    med = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - med)))
    if not (np.isfinite(sig) and sig > 0):
        return np.zeros(d.shape, dtype=bool)
    seg = detect_sources(np.where(ok, d - med, 0.0), SKY_SRC_NSIGMA * sig, n_pixels=SKY_SRC_NPIXELS)
    if seg is None:
        return np.zeros(d.shape, dtype=bool)
    grow = max(1.0, float(fwhm_px)) if np.isfinite(float(fwhm_px)) else 1.0
    return disk_dilate(seg.data > 0, grow)


def sky_image(data: np.ndarray, fwhm_px: float, *, covered: np.ndarray | None = None) -> np.ndarray:
    """The frame for local-sky medians: NaN off the aligned footprint and on detected sources."""
    d = np.asarray(data, dtype=np.float64)
    out = d.copy() if covered is None else np.where(np.asarray(covered, dtype=bool), d, np.nan)
    out[sky_source_mask(d, fwhm_px, valid=covered)] = np.nan
    return out


# A local sky needs at least SKY_MIN_PIX sky pixels; an annulus covered by the star's own halo or by neighbours
# (a saturated star: CV CVn, G 5.9) takes the sky pixels of a box of half-size SKY_BOX_FACTOR x r_out around it.
SKY_MIN_PIX = 30
SKY_BOX_FACTOR = 2.0


def annulus_sky_medians(sky: np.ndarray, pos: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """Local sky (ADU/px) per position: plain median of the finite ``sky`` pixels in the annulus.

    ``sky`` is ``sky_image`` (NaN off the footprint and on sources). Fewer than ``SKY_MIN_PIX`` such pixels:
    plain median of the finite pixels in the box of half-size ``SKY_BOX_FACTOR`` x r_out; still too few: NaN.
    """
    from photutils.aperture import CircularAnnulus  # noqa: PLC0415

    p = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    out = np.full(len(p), np.nan)
    if len(p) == 0:
        return out
    h, w = sky.shape
    half = SKY_BOX_FACTOR * float(r_out)
    for i, m in enumerate(CircularAnnulus(p, r_in=float(r_in), r_out=float(r_out)).to_mask(method="center")):
        v = np.asarray(m.get_values(sky), dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < SKY_MIN_PIX:
            x, y = p[i]
            box = sky[max(0, int(y - half)) : min(h, int(y + half) + 1), max(0, int(x - half)) : min(w, int(x + half) + 1)]
            v = box[np.isfinite(box)]
        if v.size >= SKY_MIN_PIX:
            out[i] = float(np.median(v))
    return out
