"""DAO-FALSE-2026-10: false-detection fraction of a DAO run from the negated image."""
from __future__ import annotations

import numpy as np
from photutils.detection import DAOStarFinder

from pipeline_catalog import DAO_FALSE_FRACTION_WARN, _dao_convolved_background_rms_adu, dao_false_detection_fraction
from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER


def _field(seed: int, n_stars: int, size: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 10.0, (size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_stars):
        x0, y0 = rng.uniform(10, size - 10, 2)
        img += (rng.uniform(20, 400) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.3 ** 2))).astype(np.float32)
    return img


def _run(img: np.ndarray, n_sigma: float) -> dict:
    thr = n_sigma * _dao_convolved_background_rms_adu(img, fwhm_px=3.0)[0]
    tbl = DAOStarFinder(fwhm=3.0, threshold=thr, scale_threshold=False, **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER)(img)
    xy = np.c_[tbl["x_centroid"], tbl["y_centroid"]] if tbl is not None else []
    return dao_false_detection_fraction(img, fwhm_px=3.0, threshold_adu=thr, star_xy=xy)


def test_pure_noise_is_all_false() -> None:
    r = _run(_field(1, 0), 3.0)
    assert r["n_detected"] > 50
    assert 0.6 < r["false_fraction"] < 1.6


def test_side_lobes_of_negated_stars_are_not_counted() -> None:
    r = _run(_field(2, 300), 4.5)
    # every raw negated detection here is a side lobe next to a bright star (measured 2.8-5.8 px)
    assert r["n_negative_all"] > 10
    assert r["n_negative_far"] <= 3
    assert r["false_fraction"] < DAO_FALSE_FRACTION_WARN


def test_no_detections_gives_nan_fraction() -> None:
    r = dao_false_detection_fraction(np.zeros((50, 50), np.float32), fwhm_px=3.0, threshold_adu=5.0, star_xy=[])
    assert r["n_negative_all"] == 0 and np.isnan(r["false_fraction"])
