# -*- coding: ascii -*-
"""Satellite / aircraft trail detection on one frame (MASTERSTAR-TRAIL-2026-11).

Method: median Radon transform (MRT; Stark et al. 2022, "Improved Identification of Satellite
Trails in ACS/WFC Imaging Using a Modified Radon Transform", ACS ISR 2022-08): the MEDIAN of the
image along every straight line. A trail crossing the field lifts the median of its line; stars on a
line do not (they cover a small fraction of it), so no star mask is needed.

1. The frame is block-averaged to bins of ``TRAIL_BIN_FWHM`` x FWHM: the whole cross profile of a
   seeing-limited trail falls into one bin column when the line is aligned with the bins, and the
   cost no longer depends on the sensor size.
2. For angles 0..180 deg in steps that move the line end by <= 1 bin over the half diagonal, the
   image is rotated (nearest neighbour, no interpolation) and the median of every column is taken
   (lines shorter than ``TRAIL_MIN_LINE_FRAC`` of the longer frame side are not used).
3. The smooth background is removed (2-D running median of ``TRAIL_BASELINE_BINS`` bins: gradients,
   vignetting; a trail <= 4 bins wide does not move it); per angle a running median over the line
   offset of the same window is the baseline of the line medians; lines whose window leaves the
   frame are not used. Row and column medians are subtracted (CMOS read-out pattern: a detector row
   is a straight line; trails within ~0.4 deg of a detector axis are lost with it).
4. Significance S = (median - baseline) / (sqrt(pi/2) * sigma_bin / sqrt(n)) (sampling noise of a
   median of n Gaussian values), then normalised by its own robust spread over all lines of the
   frame (empirical null: Z = (S - median S) / (1.4826 MAD S)).
5. A frame has a trail when the largest Z >= ``TRAIL_Z_MIN``.

``TRAIL_Z_MIN`` was calibrated on 1057 frames of Milan's two rigs (QHY294 + 200 mm lens, 2x2,
5 nights; C5A-150M + 0.8 m Zdanice, 4x4, 2 nights g + r; 2026-04..10), each tested against a reference
frame of its group. Brno groups (386 frames, trails rare; every frame with Z >= 7 checked by eye):
all 13 frames with Z >= 13.7 show a trail, the frames at Z 7.8 and 8.0 do not (one is a residual of a
sky feature at 50 deg), the clean frames have Z p50 5.2, p95 6.4, p99 7.3. Home nights: visible trails,
including partial ones, from Z ~7.5 up to ~3000. With 7.5, about 0.5 % of the clean frames are skipped
(a false call only moves the MASTERSTAR to the next-ranked frame). Short partial trails (a corner segment) can stay below the threshold: the median
of a full line rises only when the trail covers most of it. In the home nights 25-35 % of the 60 s
frames have a trail. Without a reference frame the null is wider (clean frames up to Z ~10: CMOS
column defects and sky features fixed in the field); a group with one kept frame keeps that frame.
The result also carries the trail amplitude per pixel in units of the pixel noise (A, informational:
up to 2x low when the trail straddles two bins).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from stats_core import MAD_TO_SIGMA

#: Bin size in FWHM units (see module docstring, step 1).
TRAIL_BIN_FWHM = 3.5
#: Shortest line used, as a fraction of the longer frame side (corner lines are too short to judge).
TRAIL_MIN_LINE_FRAC = 0.3
#: Running-median window of the baseline, in bins (odd; a trail <= 4 bins wide does not move it).
TRAIL_BASELINE_BINS = 9
#: Line offsets (bins) a feature of the reference frame may be shifted by (frame QC keeps frames whose
#: stars match the reference within max(1.5 px, FWHM): well inside one bin of 3.5 FWHM; 2 bins margin).
TRAIL_SHIFT_BINS = 2
#: Detection threshold on the null-normalised line significance (calibration in the docstring).
TRAIL_Z_MIN = 7.5
#: Standard error of a median of n Gaussian values = sqrt(pi/2) * sigma / sqrt(n).
_MEDIAN_SE = math.sqrt(math.pi / 2.0)
#: Integral of a unit-peak Gaussian profile in FWHM units: sqrt(2 pi) / (2 sqrt(2 ln 2)).
_GAUSS_AREA_PER_FWHM = math.sqrt(2.0 * math.pi) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass
class TrailCheck:
    """Result of :func:`detect_trail` for one frame."""

    has_trail: bool
    z_max: float | None
    z_min_required: float
    angle_deg: float | None  # line direction in the binned frame, 0 = along x (columns after rotation)
    offset_bins: int | None
    amplitude_sigma: float | None  # trail amplitude per pixel / pixel noise
    bin_px: int
    n_lines: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("z_max", "amplitude_sigma"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        if d["angle_deg"] is not None:
            d["angle_deg"] = round(float(d["angle_deg"]), 3)
        return d


def trail_bin_px(fwhm_px: float) -> int:
    """Bin size of the trail test for a stellar FWHM (``TRAIL_BIN_FWHM`` x FWHM, at least 1 px)."""
    return max(1, int(round(TRAIL_BIN_FWHM * float(fwhm_px))))


def _block_mean(a: np.ndarray, b: int) -> np.ndarray:
    if b <= 1:
        return a.astype(np.float64, copy=False)
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def _line_medians(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(angles_deg, M[angle, offset], N[angle, offset]): median and pixel count of every line."""
    from scipy import ndimage  # noqa: PLC0415

    h, w = img.shape
    size = int(math.ceil(math.hypot(h, w)))
    n_ang = int(math.ceil(math.pi * size / 2.0))  # step 2/size rad: <= 1 bin drift over half the diagonal
    angles = np.linspace(0.0, 180.0, n_ang, endpoint=False)
    filled = np.zeros((size, size), dtype=np.float64)
    valid = np.zeros((size, size), dtype=np.float64)
    y0, x0 = (size - h) // 2, (size - w) // 2
    ok = np.isfinite(img)
    filled[y0:y0 + h, x0:x0 + w] = np.where(ok, img, 0.0)
    valid[y0:y0 + h, x0:x0 + w] = ok
    med = np.full((n_ang, size), np.nan)
    cnt = np.zeros((n_ang, size))
    for i, th in enumerate(angles):
        r = ndimage.rotate(filled, th, reshape=False, order=0, mode="constant", cval=0.0)
        v = ndimage.rotate(valid, th, reshape=False, order=0, mode="constant", cval=0.0) > 0.5
        n = v.sum(axis=0)
        cols = n > 0
        if cols.any():
            rr = np.where(v[:, cols], r[:, cols], np.nan)
            med[i, cols] = np.nanmedian(rr, axis=0)
        cnt[i] = n
    return angles, med, cnt


@dataclass
class LinePlane:
    """Line significance S[angle, offset] of one frame (see the module docstring, steps 1-4)."""

    angles: np.ndarray
    sig: np.ndarray  # S, NaN where the line is not usable
    excess: np.ndarray  # line median - baseline, in bin units (ADU)
    sigma_bin: float
    bin_px: int
    shape: tuple[int, int]
    note: str = ""


def line_plane(data: np.ndarray, fwhm_px: float) -> LinePlane:
    """Median Radon transform significance plane of ``data`` (2-D image, NaN allowed)."""
    from scipy import ndimage  # noqa: PLC0415

    a = np.asarray(data, dtype=np.float64)
    if a.ndim != 2 or not (math.isfinite(float(fwhm_px)) and fwhm_px > 0):
        raise ValueError(f"trail test: need a 2-D image and a positive FWHM (got {a.shape}, {fwhm_px})")
    b = trail_bin_px(fwhm_px)
    ok = np.isfinite(a)
    fill = float(np.median(a[ok])) if ok.any() else 0.0
    img = _block_mean(np.where(ok, a, fill), b)
    valid = _block_mean(ok.astype(np.float64), b) > 0.5
    empty = np.zeros((0, 0))
    if min(img.shape) < 2 * TRAIL_BASELINE_BINS:
        return LinePlane(np.zeros(0), empty, empty, float("nan"), b, a.shape,
                         note=f"frame too small for the trail test ({img.shape} bins of {b} px)")
    # smooth 2-D background off first (gradients, vignetting): a running median over
    # TRAIL_BASELINE_BINS x TRAIL_BASELINE_BINS bins is not moved by a trail <= 4 bins wide, and the
    # line medians of a flat field have no kinks where a line starts to meet another frame edge
    img = img - ndimage.median_filter(img, size=TRAIL_BASELINE_BINS, mode="nearest")
    img[~valid] = np.nan
    # CMOS / CCD read-out pattern (row and column offsets) off: a detector row is a perfect straight
    # line. Only a trail within atan(2 / frame side in bins) (~0.4 deg) of a detector axis fills more
    # than half of one row or column and is removed with it.
    with np.errstate(all="ignore"):
        img = img - np.nan_to_num(np.nanmedian(img, axis=1, keepdims=True))
        img = img - np.nan_to_num(np.nanmedian(img, axis=0, keepdims=True))
    level = float(np.nanmedian(img))
    sigma_bin = MAD_TO_SIGMA * float(np.nanmedian(np.abs(img - level)))
    if not (sigma_bin > 0):
        return LinePlane(np.zeros(0), empty, empty, float("nan"), b, a.shape, note="no pixel noise measurable")
    angles, med, cnt = _line_medians(img)
    good = cnt >= TRAIL_MIN_LINE_FRAC * max(img.shape)
    half = TRAIL_BASELINE_BINS // 2
    base = np.full_like(med, np.nan)
    for i in range(med.shape[0]):
        idx = np.flatnonzero(good[i])
        if idx.size < TRAIL_BASELINE_BINS:
            continue
        lo, hi = int(idx[0]), int(idx[-1])
        m = med[i, lo:hi + 1]
        m = np.where(np.isfinite(m), m, np.nanmedian(m))
        bl = ndimage.median_filter(m, size=TRAIL_BASELINE_BINS, mode="nearest")
        bl[:half] = np.nan
        bl[len(bl) - half:] = np.nan
        base[i, lo:hi + 1] = bl
    with np.errstate(invalid="ignore", divide="ignore"):
        excess = med - base
        sig = excess / (_MEDIAN_SE * sigma_bin / np.sqrt(np.maximum(cnt, 1.0)))
    sig[~good] = np.nan
    return LinePlane(angles, sig, excess, sigma_bin, b, a.shape)


def _persistent(ref: LinePlane, shape: tuple[int, int]) -> np.ndarray:
    """Positive line significance of the reference frame, widened by +-TRAIL_SHIFT_BINS offsets and
    +-1 angle step (frames of one group are aligned to a few pixels by frame QC's common-star match)."""
    from scipy import ndimage  # noqa: PLC0415

    r = np.where(np.isfinite(ref.sig), np.clip(ref.sig, 0.0, None), 0.0)
    if r.shape != shape:
        raise ValueError(f"trail test: reference plane {r.shape} != candidate {shape} (different frame size)")
    return ndimage.maximum_filter(r, size=(3, 2 * TRAIL_SHIFT_BINS + 1), mode="nearest")


def detect_trail(data: np.ndarray, fwhm_px: float, reference: LinePlane | None = None) -> TrailCheck:
    """Look for a straight trail crossing ``data`` (2-D image, NaN allowed); ``fwhm_px`` = stellar FWHM.

    ``reference``: the line plane of another frame of the same group (:func:`line_plane`). Lines that
    are significant there too are fixed on the sky or the detector (a dust lane edge, a diffraction
    spike, a column) and are taken out; a satellite trail is in one frame only."""
    pl = line_plane(data, fwhm_px)
    if pl.note:
        return TrailCheck(False, None, TRAIL_Z_MIN, None, None, None, pl.bin_px, 0, note=pl.note)
    stat = pl.sig - _persistent(reference, pl.sig.shape) if reference is not None and not reference.note else pl.sig
    sv = stat[np.isfinite(stat)]
    if sv.size == 0:
        return TrailCheck(False, None, TRAIL_Z_MIN, None, None, None, pl.bin_px, 0, note="no usable lines")
    s0 = float(np.median(sv))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(sv - s0)))
    if not (spread > 0):
        return TrailCheck(False, None, TRAIL_Z_MIN, None, None, None, pl.bin_px, int(sv.size),
                          note="flat line statistic")
    z = (stat - s0) / spread
    i, j = np.unravel_index(int(np.nanargmax(z)), z.shape)
    z_max = float(z[i, j])
    # excess of the bin mean / bin noise = A * (Gaussian profile area) in pixels  ->  A in sigma_pix
    amp = float(pl.excess[i, j] / pl.sigma_bin) / (_GAUSS_AREA_PER_FWHM * float(fwhm_px))
    return TrailCheck(
        has_trail=bool(z_max >= TRAIL_Z_MIN), z_max=z_max, z_min_required=TRAIL_Z_MIN,
        angle_deg=float(pl.angles[i]), offset_bins=int(j), amplitude_sigma=amp, bin_px=pl.bin_px,
        n_lines=int(sv.size), note="" if reference is not None else "no reference frame",
    )
