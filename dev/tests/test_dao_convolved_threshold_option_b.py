"""R2 verification for DAO option B: convolved-image RMS threshold."""

from __future__ import annotations

import numpy as np
import pytest
from photutils.detection import DAOStarFinder
from photutils.detection.core import _StarFinderKernel
from scipy.ndimage import convolve, map_coordinates

from pipeline import (
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    _dao_convolved_background_rms_adu,
)

FWHM = 3.2
SIGMA_PIX = 46.0
N_SIGMA = 3.8
REL_ERR_EXPECT = 1.3604


def _white_noise_frame(rng: np.random.Generator, shape: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    return rng.normal(0.0, SIGMA_PIX, size=shape).astype(np.float32)


def _cubic_shift(frame: np.ndarray, shift: float) -> np.ndarray:
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.array([yy + shift, xx + shift])
    return map_coordinates(frame, coords, order=3, mode="nearest").astype(np.float32)


def _measured_conv_sigma(frame: np.ndarray, *, fwhm: float = FWHM) -> float:
    kernel = _StarFinderKernel(fwhm=fwhm)
    conv = convolve(frame, kernel.data, mode="nearest")
    return float(np.std(conv))


def _count_dao(data: np.ndarray, *, threshold_adu: float, scale_threshold: bool) -> int:
    finder = DAOStarFinder(
        fwhm=FWHM,
        threshold=float(threshold_adu),
        scale_threshold=scale_threshold,
        n_brightest=5000,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = finder(data)
    return 0 if tbl is None else int(len(tbl))


def _achieved_significance(data: np.ndarray, threshold_adu: float, *, scale_threshold: bool) -> float:
    from astropy.stats import sigma_clipped_stats

    kernel = _StarFinderKernel(fwhm=FWHM)
    conv = convolve(data, kernel.data, mode="nearest")
    _, _, sigma_conv = sigma_clipped_stats(conv, sigma=3.0, maxiters=3)
    if scale_threshold:
        eff_thr = threshold_adu * float(kernel.rel_err)
    else:
        eff_thr = threshold_adu
    return eff_thr / sigma_conv if sigma_conv > 0 else float("nan")


def test_white_noise_convolved_rms_matches_rel_err() -> None:
    rng = np.random.default_rng(0)
    frame = _white_noise_frame(rng)
    rms_conv, rel_err = _dao_convolved_background_rms_adu(frame, fwhm_px=FWHM)
    assert rel_err == pytest.approx(REL_ERR_EXPECT, rel=0.01)
    assert rms_conv / SIGMA_PIX == pytest.approx(rel_err, rel=0.02)


@pytest.mark.parametrize("shift", [0.25, 0.50])
def test_resampled_frame_pixel_rel_err_breaks(shift: float) -> None:
    rng = np.random.default_rng(1)
    frame = _cubic_shift(_white_noise_frame(rng), shift)
    sigma_pix = float(np.std(frame))
    rms_conv, rel_err = _dao_convolved_background_rms_adu(frame, fwhm_px=FWHM)
    # White-noise shortcut sigma_conv ~ sigma_pix * rel_err fails on resampled frames.
    ratio = rms_conv / (sigma_pix * rel_err)
    assert ratio != pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize("shift", [0.0, 0.25, 0.50])
def test_option_b_nominal_significance_on_white_and_resampled(shift: float) -> None:
    rng = np.random.default_rng(2 + int(shift * 100))
    frame = _white_noise_frame(rng)
    if shift > 0:
        frame = _cubic_shift(frame, shift)
    rms_conv, _ = _dao_convolved_background_rms_adu(frame, fwhm_px=FWHM)
    thr = N_SIGMA * rms_conv
    sig = _achieved_significance(frame, thr, scale_threshold=False)
    assert sig == pytest.approx(N_SIGMA, rel=0.05)
