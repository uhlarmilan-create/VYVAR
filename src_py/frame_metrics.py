"""Per-frame QC metrics (QC-METRICS-2026-10, Milan 2026-10-09).

Universal, data-derived, no sigma clipping, no iterations, no fixed pixel scales:

* **Noise** sigma_pix = 1.4826 * median|x[i, j+1] - x[i, j]| / sqrt(2). Differences of
  neighbouring pixels cancel the sky level and gradients; stars occupy a small fraction
  of pixel pairs, so the median is not pulled by them (DER_SNR principle, Stoehr 2008).
  It includes sky photon noise, read noise and dark noise as they really are in the frame.
* **First FWHM guess** FWHM0 from the half-maximum radius of the brightest unsaturated
  local maxima above 5 sigma, walking out in 4 directions (C-Munipack / AIJ style).
  It sets the DAO kernel and all box / mask sizes (no fixed 4.5 px, no fixed box).
* **Detection** DAOStarFinder(FWHM0, 5 sigma_pix) - one fixed threshold, so the star
  count is comparable between frames.
* **Sky** median of the pixels outside a star mask (radius 3 x FWHM0) - one pass.
* **FWHM and elongation** elliptical Gaussian + constant fitted in a box of ~5 x FWHM0
  to isolated, unsaturated, bright stars (Siril / EXOTIC / photutils style);
  FWHM_i = 2.3548 * sqrt(sigma_x sigma_y), elongation_i = sigma_major / sigma_minor;
  frame values are medians.
* **Star list for transparency** aperture flux (r = 2.5 x FWHM, local sky = median of
  the 3-4 x FWHM annulus) of up to ``N_STARS_KEEP`` brightest unsaturated detections;
  frame_qc matches them to the group reference and derives the relative zero point.

"Unsaturated" means peak below 0.8 x the brightest pixel of the frame (data-derived,
no detector constant): if nothing saturates only the brightest few stars are skipped.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

DETECT_NSIGMA = 5.0
N_FWHM0_PEAKS = 50
N_FIT_STARS = 60
N_STARS_KEEP = 150
SAT_FRACTION_OF_MAX = 0.8
MAD_TO_SIGMA = 1.4826
GAUSS_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
#: A point spread function narrower than 1 px FWHM is not an optical image of a star
#: (hot pixel, cosmic ray hit). For a Gaussian of FWHM F the 4 neighbours of the peak
#: carry exp(-4 ln2 / F^2) of the peak: 2^-4 = 0.0625 at F = 1 px.
MIN_PSF_FWHM_PX = 1.0
MIN_NEIGHBOUR_RATIO = 2.0 ** (-4.0 / MIN_PSF_FWHM_PX**2)


def is_point_spread(img: np.ndarray, y: int, x: int, sky: float, sigma: float) -> bool:
    """True when the peak at (y, x) spreads into its 4 neighbours like an optical PSF.

    Hot pixels and most cosmic-ray hits leave the neighbours at sky level; a star of
    FWHM >= 1 px puts >= 6.25 % of the peak into each neighbour (DAOPHOT sharpness idea).
    The neighbours must also be significant (mean above 3 sigma / 2 of the sky noise).
    """
    ny, nx = img.shape
    if not (1 <= y < ny - 1 and 1 <= x < nx - 1):
        return False
    peak = float(img[y, x]) - sky
    nb = float(np.mean([img[y + 1, x], img[y - 1, x], img[y, x + 1], img[y, x - 1]])) - sky
    if not (math.isfinite(peak) and math.isfinite(nb)) or peak <= 0:
        return False
    return nb >= MIN_NEIGHBOUR_RATIO * peak and nb > 1.5 * sigma


def pixel_noise_sigma(img: np.ndarray) -> float | None:
    """sigma_pix from horizontal neighbour differences (no clipping)."""
    d = np.diff(np.asarray(img, dtype=np.float64), axis=1)
    d = d[np.isfinite(d)]
    if d.size < 100:
        return None
    s = MAD_TO_SIGMA * float(np.median(np.abs(d))) / math.sqrt(2.0)
    return s if s > 0 and math.isfinite(s) else None


def _hwhm_4dir(img: np.ndarray, y: int, x: int, sky: float, rmax: int) -> float | None:
    """Half width at half maximum from 4 directions (linear interpolation)."""
    ny, nx = img.shape
    peak = float(img[y, x]) - sky
    if not (peak > 0):
        return None
    half = 0.5 * peak
    out: list[float] = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        prev = peak
        for r in range(1, rmax + 1):
            yy, xx = y + dy * r, x + dx * r
            if not (0 <= yy < ny and 0 <= xx < nx):
                break
            v = float(img[yy, xx]) - sky
            if not math.isfinite(v):
                break
            if v <= half:
                out.append((r - 1) + (prev - half) / (prev - v) if prev > v else float(r))
                break
            prev = v
    if len(out) < 3:
        return None
    return float(np.mean(out))


def first_fwhm_guess(img: np.ndarray, sky: float, sigma: float) -> float | None:
    """FWHM0 = 2 x median HWHM of the brightest unsaturated local maxima above 5 sigma."""
    from photutils.detection import find_peaks  # noqa: PLC0415

    work = np.nan_to_num(np.asarray(img, dtype=np.float64) - sky, nan=0.0)
    tbl = find_peaks(work, threshold=DETECT_NSIGMA * sigma, box_size=3, border_width=2)
    if tbl is None or len(tbl) == 0:
        return None
    vmax = float(np.nanmax(img))
    pk = np.asarray(tbl["peak_value"], dtype=np.float64)
    ok = (pk + sky) < SAT_FRACTION_OF_MAX * vmax
    order = np.argsort(-pk[ok])[: N_FWHM0_PEAKS * 6]
    xs = np.asarray(tbl["x_peak"])[ok][order]
    ys = np.asarray(tbl["y_peak"])[ok][order]
    rmax = max(5, min(img.shape) // 10)
    hw: list[float] = []
    for x, y in zip(xs, ys):
        if not is_point_spread(img, int(y), int(x), sky, sigma):
            continue
        h = _hwhm_4dir(img, int(y), int(x), sky, rmax)
        if h is not None and h > 0:
            hw.append(h)
        if len(hw) >= N_FWHM0_PEAKS:
            break
    if len(hw) < 3:
        return None
    return 2.0 * float(np.median(hw))


def _fit_elliptical_gaussian(cut: np.ndarray, x0: float, y0: float, fwhm0: float) -> tuple[float, float, float, float] | None:
    """Fit A*exp(-Q/2) + c, Q from sigma_major/minor and angle. Returns (fwhm, elong, xc, yc)."""
    from scipy.optimize import least_squares  # noqa: PLC0415

    ny, nx = cut.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    m = np.isfinite(cut)
    if m.sum() < 25:
        return None
    z = cut[m]
    X, Y = xx[m].astype(np.float64), yy[m].astype(np.float64)
    c0 = float(np.median(np.concatenate([cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1]])))
    a0 = float(np.nanmax(cut)) - c0
    s0 = fwhm0 / GAUSS_FWHM

    def model(p: np.ndarray) -> np.ndarray:
        a, xc, yc, sa, sb, th, c = p
        ct, st = math.cos(th), math.sin(th)
        dx, dy = X - xc, Y - yc
        u = (dx * ct + dy * st) / sa
        v = (-dx * st + dy * ct) / sb
        return a * np.exp(-0.5 * (u * u + v * v)) + c

    p0 = np.array([a0, x0, y0, s0, s0, 0.0, c0])
    lo = np.array([0.0, 0.0, 0.0, 0.2 * s0, 0.2 * s0, -math.pi, -np.inf])
    hi = np.array([np.inf, nx - 1.0, ny - 1.0, 5.0 * s0, 5.0 * s0, math.pi, np.inf])
    try:
        r = least_squares(lambda p: model(p) - z, p0, bounds=(lo, hi), method="trf", max_nfev=200)
    except (ValueError, RuntimeError):
        return None
    if not r.success:
        return None
    a, xc, yc, sa, sb, _th, _c = r.x
    if a <= 0 or not (0 < xc < nx - 1 and 0 < yc < ny - 1):
        return None
    smaj, smin = max(sa, sb), min(sa, sb)
    if smin <= 0:
        return None
    return GAUSS_FWHM * math.sqrt(sa * sb), smaj / smin, float(xc), float(yc)


def measure_frame(img: np.ndarray) -> dict[str, Any]:
    """QC metrics of one calibrated image (or crop). Keys: noise_sigma, sky_background,
    fwhm0, fwhm_mean, elongation_mean, star_count, n_fit, qc_stars (list [x, y, flux]),
    qc_shape (ny, nx)."""
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry  # noqa: PLC0415
    from photutils.detection import DAOStarFinder  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    out: dict[str, Any] = {
        "noise_sigma": None, "sky_background": None, "fwhm0": None, "fwhm_mean": None,
        "elongation_mean": None, "star_count": 0, "n_fit": 0, "qc_stars": [], "qc_shape": list(np.shape(img)),
    }
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).any():
        return out
    sigma = pixel_noise_sigma(a)
    if sigma is None:
        return out
    out["noise_sigma"] = sigma
    sky0 = float(np.nanmedian(a))
    fwhm0 = first_fwhm_guess(a, sky0, sigma)
    if fwhm0 is None:
        out["sky_background"] = sky0
        return out
    out["fwhm0"] = fwhm0

    work = np.nan_to_num(a - sky0, nan=0.0)
    from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER  # noqa: PLC0415

    tbl = DAOStarFinder(fwhm=float(fwhm0), threshold=DETECT_NSIGMA * sigma,
                        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER)(work)
    n = 0 if tbl is None else len(tbl)
    out["star_count"] = int(n)

    # sky: median outside the star mask (radius 3 x FWHM0), one pass
    mask = np.zeros(a.shape, dtype=bool)
    if n:
        cx = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
        cy = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
        xs = np.asarray(tbl[cx], dtype=np.float64)
        ys = np.asarray(tbl[cy], dtype=np.float64)
        rr = int(math.ceil(3.0 * fwhm0))
        oy, ox = np.ogrid[-rr: rr + 1, -rr: rr + 1]
        disk = (ox * ox + oy * oy) <= rr * rr
        for x, y in zip(xs, ys):
            xi, yi = int(round(x)), int(round(y))
            y0, y1, x0, x1 = yi - rr, yi + rr + 1, xi - rr, xi + rr + 1
            sy0, sx0 = max(0, -y0), max(0, -x0)
            y0c, x0c = max(0, y0), max(0, x0)
            y1c, x1c = min(a.shape[0], y1), min(a.shape[1], x1)
            mask[y0c:y1c, x0c:x1c] |= disk[sy0: sy0 + (y1c - y0c), sx0: sx0 + (x1c - x0c)]
    free = (~mask) & np.isfinite(a)
    sky = float(np.median(a[free])) if free.sum() > 100 else sky0
    out["sky_background"] = sky
    if n == 0:
        return out

    vmax = float(np.nanmax(a))
    peak = np.asarray(tbl["peak"], dtype=np.float64) + sky0
    unsat = peak < SAT_FRACTION_OF_MAX * vmax
    tree = cKDTree(np.column_stack([xs, ys]))
    d2, _ = tree.query(np.column_stack([xs, ys]), k=2) if n > 1 else (np.full((n, 2), np.inf), None)
    isolated = d2[:, 1] > 3.0 * fwhm0
    half = int(math.ceil(2.5 * fwhm0))
    inside = (xs > half) & (xs < a.shape[1] - 1 - half) & (ys > half) & (ys < a.shape[0] - 1 - half)

    psf_like = np.array([is_point_spread(a, int(round(y)), int(round(x)), sky, sigma) for x, y in zip(xs, ys)],
                        dtype=bool)
    cand = np.where(unsat & isolated & inside & psf_like)[0]
    cand = cand[np.argsort(-peak[cand])][:N_FIT_STARS]
    fw: list[float] = []
    el: list[float] = []
    for i in cand:
        xi, yi = int(round(xs[i])), int(round(ys[i]))
        cut = a[yi - half: yi + half + 1, xi - half: xi + half + 1]
        r = _fit_elliptical_gaussian(cut, xs[i] - (xi - half), ys[i] - (yi - half), fwhm0)
        if r is not None and r[0] >= MIN_PSF_FWHM_PX:
            fw.append(r[0])
            el.append(r[1])
    out["n_fit"] = len(fw)
    if len(fw) >= 3:
        out["fwhm_mean"] = float(np.median(fw))
        out["elongation_mean"] = float(np.median(el))

    # star list for transparency (aperture scaled with this frame's FWHM)
    f_use = out["fwhm_mean"] or fwhm0
    keep = np.where(unsat & inside & psf_like)[0]
    keep = keep[np.argsort(-peak[keep])][:N_STARS_KEEP]
    if keep.size:
        pos = np.column_stack([xs[keep], ys[keep]])
        ap = CircularAperture(pos, r=2.5 * f_use)
        an = CircularAnnulus(pos, r_in=3.0 * f_use, r_out=4.0 * f_use)
        bkg = ApertureStats(np.nan_to_num(a, nan=sky), an).median
        ph = aperture_photometry(np.nan_to_num(a, nan=sky), ap)
        flux = np.asarray(ph["aperture_sum"], dtype=np.float64) - np.asarray(bkg, dtype=np.float64) * ap.area
        good = np.isfinite(flux) & (flux > 0)
        out["qc_stars"] = [[float(x), float(y), float(f)] for (x, y), f, g in zip(pos, flux, good) if g]
    return out
