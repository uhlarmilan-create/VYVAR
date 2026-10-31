"""BADPIX-REPAIR-2026-10: missing pixels of a calibrated frame filled from the frame's own PSF model.

A bad flat pixel (G01-29) is NaN in the calibrated frame. Left missing, the cubic alignment warp grows it
to a 4x4 block and every star whose aperture reached it was dropped (BO CVn 527: 27 points of TOI-3919,
none of them inside the chosen aperture). Practice elsewhere keeps such measurements: SExtractor
(MASK_TYPE=CORRECT) replaces a masked pixel by its mirror through the object centre, SEP by the aperture
mean, IRAF ``fixpix`` interpolates linearly before photometry.

The injection-recovery study (Brno 534, 10 frames g/r, 400 stars, 7164 cases per method; docs
VYVAR_DECISIONS BADPIX-REPAIR-2026-10) measured the flux error of a repaired aperture in units of the
star's photon noise: one missing pixel on the native grid - frame ePSF fit 0.12 sigma robust scatter
(95 % < 0.42 sigma, at the star centre < 0.72 sigma), Moffat 0.16, linear interpolation 0.17, mirror
0.22, aperture mean 0.49; the same pixel after the warp (4x4 block) 0.8 sigma even with the ePSF and
4-50 sigma with the others. So the repair runs on the calibrated frame, before alignment, and a pixel
gets the value the frame's PSF model predicts:

    value = local sky + sum of the ePSF models of the stars whose model reaches the pixel

* Sky: plain median of the 4-6 FWHM annulus of ``sky_image`` (sources masked) around the pixel - the
  local sky of the PSF photometry (``epsf_frame.fit_frame_psf``).
* Stars: DAO detections (``detection_threshold_sigma``) within reach of a missing pixel (model half-size
  plus a fit box, the neighbour reach of ``frame_fit_sources``); flux and position fitted together with
  their neighbours (photutils ``PSFPhotometry``, groups closer than two fit boxes) on the good pixels of
  the fit box; missing and non-linear pixels are masked (DAOPHOT "high good datum").
* ePSF: built from the frame's brightest isolated unsaturated stars (same builder as the MASTERSTAR ePSF:
  ``psf_photometry._epsf_build_imagepsf_from_stars``).

Nothing is filled silently: a pixel that a star's model reaches is left missing (the star is then
flagged ``edge_fail`` downstream, as before) when the frame has no linearity limit (``VY_CSAT``), fewer
than ``EPSF_MIN_BUILD_STARS`` build stars, or the fit of a star reaching it failed; the reason is
returned and logged. Every filled pixel is listed (``REPAIR_HDU`` extension of the calibrated frame), so a
measurement whose aperture holds one is annotated, not dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from badpix_sidecar import REPAIR_HDU

__all__ = ["REPAIR_HDU", "RepairResult", "calibrated_saturation_adu", "repair_missing_pixels", "repair_table_hdu"]

#: Build stars of the repair ePSF: the brightest isolated unsaturated detections, at most this many.
#: 2 x the ~50 stars per PSF of Anderson et al. (2006, A&A 454, 1029); the build-star noise of the model is
#: then well below the photon noise of one star.
BP_EPSF_BUILD_STARS_MAX = 100
#: photutils fit flags of a failed fit: 2 fit position outside the image, 8 not converged. (4, flux <= 0,
#: means no star - its model is removed; 16, covariance unavailable, leaves flux and position usable.)
_FIT_FAILED_FLAGS = 2 | 8


@dataclass
class RepairResult:
    data: np.ndarray
    n_missing: int = 0
    n_sky: int = 0
    n_psf: int = 0
    n_left: int = 0
    n_stars_fitted: int = 0
    n_build_stars: int = 0
    reason_left: str = ""
    xy_kind: list[tuple[int, int, str]] = field(default_factory=list)  # repaired pixels: x, y, 'S' / 'P'


def calibrated_saturation_adu(saturation_adu: float | None, pedestal_adu: float | None, flat_max: float | None) -> float | None:
    """Lowest calibrated level of a raw-saturated pixel: (SAT - pedestal) / max normalised flat.

    calibrated = (raw - dark) / flat_norm, so a pixel at the raw saturation level reaches its lowest
    calibrated value where the normalised flat is highest. None when a term is unknown.
    """
    vals = [saturation_adu, pedestal_adu, flat_max]
    if any(v is None or not math.isfinite(float(v)) for v in vals) or float(flat_max) <= 0:
        return None
    return (float(saturation_adu) - float(pedestal_adu)) / float(flat_max)


def _build_epsf(data: np.ndarray, cand: Any, fwhm_px: float, cutout: int) -> tuple[Any, int]:
    from photutils.psf import ImagePSF  # noqa: PLC0415

    from epsf_frame import OVERSAMPLING  # noqa: PLC0415
    from psf_photometry import _epsf_build_imagepsf_from_stars, _epsf_sky_subtracted_stars  # noqa: PLC0415
    from astropy.table import Table  # noqa: PLC0415

    stars = _epsf_sky_subtracted_stars(data, Table({"x": cand[:, 0], "y": cand[:, 1]}), int(cutout))
    built = _epsf_build_imagepsf_from_stars(stars, osamp=OVERSAMPLING, fwhm_px=float(fwhm_px))
    return ImagePSF(built["arr"], oversampling=OVERSAMPLING), int(len(stars))


def _star_models(d, missing, xy, peak, stellar, near, sky, bg, mx, my, *, fw, half, fsz, cutout, sat_cal_adu):
    """ePSF model image of the stars near missing pixels; (model, star_ok, n_build, n_fitted, reason).

    ``star_ok`` is False (scalar) when no model can be made - the model is then None (never read) -,
    or per missing pixel False where a star whose fit failed reaches it."""
    from astropy.table import Table  # noqa: PLC0415
    from photutils.psf import PSFPhotometry, SourceGrouper  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    from pipeline_constants import EPSF_MIN_BUILD_STARS, LINEARITY_FRAC  # noqa: PLC0415
    from sky_estimation import annulus_sky_medians  # noqa: PLC0415

    if sat_cal_adu is None:
        return None, False, 0, 0, "no linearity limit for the frame (VY_CSAT)"
    lin = LINEARITY_FRAC * float(sat_cal_adu)
    h, w = d.shape
    st = cKDTree(xy)
    n_close = np.array([len(st.query_ball_point(p, half)) - 1 for p in xy])
    inside = (xy[:, 0] >= half) & (xy[:, 0] < w - half) & (xy[:, 1] >= half) & (xy[:, 1] < h - half)
    d_miss, _ = cKDTree(np.c_[mx, my].astype(float)).query(xy)
    ok_b = stellar & inside & (n_close == 0) & (peak < lin) & (d_miss > half)
    cand = xy[ok_b][np.argsort(-peak[ok_b])[:BP_EPSF_BUILD_STARS_MAX]]
    if len(cand) < int(EPSF_MIN_BUILD_STARS):
        return None, False, int(len(cand)), 0, f"{len(cand)} ePSF build stars < {int(EPSF_MIN_BUILD_STARS)}"
    try:
        psf, n_build = _build_epsf(np.where(missing, np.nan, d), cand, fw, cutout)
        phot = PSFPhotometry(psf, (fsz, fsz), aperture_radius=1.5 * fw, grouper=SourceGrouper(min_separation=2.0 * fsz),
                             xy_bounds=(fw, fw), fitter_maxiters=1000)
        # xy_bounds: a fitted centre stays within one FWHM of its DAO centroid; 1000 iterations: the photutils
        # default (100) left joint fits of a bright star and a faint neighbour unconverged (Brno 534 g_0080)
        fxy = xy[near]
        lbkg = annulus_sky_medians(sky, fxy, 4.0 * fw, 6.0 * fw)
        init = Table({"id": np.arange(1, len(fxy) + 1), "x": fxy[:, 0], "y": fxy[:, 1],
                      "local_bkg": np.nan_to_num(lbkg, nan=bg)})
        r = phot(np.where(missing, bg, d), mask=missing | (d >= lin), init_params=init)
        model = np.asarray(phot.make_model_image(d.shape, include_local_bkg=False), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001  - the star pixels stay missing, with the reason
        return None, False, 0, 0, f"ePSF build / fit error: {type(exc).__name__}: {exc}"
    flags = np.asarray(r["flags"], int)
    flux = np.asarray(r["flux_fit"], float)
    # a source fitted with no positive flux is not a star: no light of its own, its model is removed
    for j in np.flatnonzero(~(flux > 0) & np.isfinite(flux)):
        x0, y0 = float(r["x_fit"][j]), float(r["y_fit"][j])
        if not (math.isfinite(x0) and math.isfinite(y0)):
            continue
        y_a, y_b = max(0, int(y0 - half)), min(d.shape[0], int(y0 + half) + 2)
        x_a, x_b = max(0, int(x0 - half)), min(d.shape[1], int(x0 + half) + 2)
        if y_b <= y_a or x_b <= x_a:
            continue  # its model lies off the frame
        yy, xx = np.mgrid[y_a:y_b, x_a:x_b]
        model[y_a:y_b, x_a:x_b] -= psf.evaluate(xx, yy, flux=float(flux[j]), x_0=x0, y_0=y0)
    # failed: the fit did not converge (8) or left the image (2)
    bad = ((flags & _FIT_FAILED_FLAGS) != 0) & (flux > 0)
    if not bad.any():
        return model, True, n_build, int(len(r)), ""
    # a missing pixel within the model reach of a star whose fit failed stays missing
    bxy = np.c_[np.asarray(r["x_fit"], float), np.asarray(r["y_fit"], float)][bad]
    bxy = np.where(np.isfinite(bxy), bxy, fxy[bad])
    dpx, _ = cKDTree(bxy).query(np.c_[mx, my].astype(float))
    return model, dpx > half, n_build, int(len(r)), f"PSF fit failed for {int(bad.sum())} star(s)"


def repair_missing_pixels(
    data: np.ndarray,
    *,
    fwhm_px: float,
    detection_sigma: float,
    sat_cal_adu: float | None,
) -> RepairResult:
    """Fill the missing (non-finite) pixels of a calibrated frame from the frame's PSF model (module doc)."""
    from photutils.detection import DAOStarFinder  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    from epsf_frame import fit_box_px  # noqa: PLC0415
    from psf_photometry import _EPSF_CUTOUT_FWHM, _to_odd_cutout  # noqa: PLC0415
    from sky_estimation import annulus_sky_medians, sky_image  # noqa: PLC0415

    d = np.asarray(data, dtype=np.float64)
    missing = ~np.isfinite(d)
    res = RepairResult(data=np.asarray(data))
    res.n_missing = int(missing.sum())
    if res.n_missing == 0:
        return res
    fw = float(fwhm_px)
    if not math.isfinite(fw) or fw <= 0:
        res.n_left, res.reason_left = res.n_missing, "no FWHM"
        return res
    my, mx = np.nonzero(missing)
    # identical-output memory cleanups (step 5): the repaired frame is kept in the input dtype (each
    # filled value is cast once, as the final astype did); temporaries are freed before the DAO pass
    # (the peak of the repair) and the detection image is shifted in place; no zero model images
    out = np.array(data, copy=True)
    sky = sky_image(np.where(missing, np.nan, d), fw)
    sky_px = annulus_sky_medians(sky, np.c_[mx, my].astype(float), 4.0 * fw, 6.0 * fw)

    cutout = _to_odd_cutout(int(round(fw * _EPSF_CUTOUT_FWHM)))
    fsz = fit_box_px(fw)
    half = cutout / 2.0
    reach = half + fsz / 2.0 + 1.0
    finite = d[~missing]
    bg = float(np.median(finite))
    sig = 1.4826 * float(np.median(np.abs(finite - bg)))
    del finite
    # Detection sees each missing pixel as the median of its finite 3x3 neighbours: masked, a missing
    # pixel at a star's peak hides the star from DAOStarFinder and the pixel would get the sky. The
    # filled peak is a dip, so a source within 2 px of a missing pixel is kept whatever its DAOFIND
    # sharpness / roundness; elsewhere those limits (Stetson 1987; photutils defaults) reject noise peaks,
    # whose free-position fits do not converge and would fail the fit of a real star in their group.
    det = np.where(missing, bg, d)
    for y, x in zip(my, mx, strict=True):
        nb = d[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        nb = nb[np.isfinite(nb)]
        if nb.size:
            det[y, x] = float(np.median(nb))
    det -= bg  # the DAO input det - bg, in place (same values)
    src = DAOStarFinder(fwhm=fw, threshold=float(detection_sigma) * sig, sharpness_range=None,
                        roundness_range=None)(det)
    del det
    if src is None or len(src) == 0:
        xy = np.zeros((0, 2))
        peak = np.zeros(0)
        stellar = np.zeros(0, dtype=bool)
    else:
        xcol = "x_centroid" if "x_centroid" in src.colnames else "xcentroid"
        ycol = "y_centroid" if "y_centroid" in src.colnames else "ycentroid"
        xy = np.c_[np.asarray(src[xcol], float), np.asarray(src[ycol], float)]
        peak = np.asarray(src["peak"], float) + bg
        stellar = ((np.asarray(src["sharpness"], float) >= 0.2) & (np.asarray(src["sharpness"], float) <= 1.0)
                   & (np.abs(np.asarray(src["roundness1"], float)) <= 1.0)
                   & (np.abs(np.asarray(src["roundness2"], float)) <= 1.0))
    miss_tree = cKDTree(np.c_[mx, my].astype(float))
    if len(xy):
        d_m, _ = miss_tree.query(xy)
        keep = stellar | (d_m <= 2.0)
        xy, peak, stellar = xy[keep], peak[keep], stellar[keep]
    near = np.zeros(len(xy), dtype=bool)
    if len(xy):
        dist, _ = miss_tree.query(xy)
        near = dist <= reach  # fitted (their model, or their neighbours', can reach a missing pixel)
    model = None  # read only where a fitted star reaches a missing pixel and the model exists
    star_ok: Any = True
    if near.any():
        model, star_ok, res.n_build_stars, res.n_stars_fitted, res.reason_left = _star_models(
            d, missing, xy, peak, stellar, near, sky, bg, mx, my, fw=fw, half=half, fsz=fsz, cutout=cutout,
            sat_cal_adu=sat_cal_adu,
        )
    # which missing pixels a fitted star's model reaches
    if len(xy) and near.any():
        dstar, _ = cKDTree(xy[near]).query(np.c_[mx, my].astype(float))
        star_px = dstar <= half
    else:
        star_px = np.zeros(len(mx), dtype=bool)
    ok = np.isfinite(sky_px) & (~star_px | star_ok)
    for i in np.flatnonzero(ok):
        out[my[i], mx[i]] = sky_px[i] + (model[my[i], mx[i]] if star_px[i] else 0.0)
        res.xy_kind.append((int(mx[i]), int(my[i]), "P" if star_px[i] else "S"))
    res.n_psf = int(np.count_nonzero(ok & star_px))
    res.n_sky = int(np.count_nonzero(ok & ~star_px))
    res.n_left = int(np.count_nonzero(~ok))
    if res.n_left and not res.reason_left:
        res.reason_left = "no local sky"
    res.data = out
    return res


def repair_table_hdu(res: RepairResult) -> Any:
    """BinTable listing the repaired pixels (x, y native 0-based; KIND 'S' sky, 'P' star model)."""
    from astropy.io import fits  # noqa: PLC0415

    xs = np.array([t[0] for t in res.xy_kind], dtype=np.int32)
    ys = np.array([t[1] for t in res.xy_kind], dtype=np.int32)
    ks = np.array([t[2] for t in res.xy_kind], dtype="S1")
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="X", format="J", array=xs), fits.Column(name="Y", format="J", array=ys),
         fits.Column(name="KIND", format="1A", array=ks)],
        name=REPAIR_HDU,
    )
    return hdu

