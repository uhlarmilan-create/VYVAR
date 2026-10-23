"""SPEED-2026-10 (after step 7): exact shortcuts must give the same numbers as the slow path."""
from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder

from masterstar_gaia_accounting import _pass2_catalog
from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER


def _image(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 5.0, (21, 21))
    yy, xx = np.mgrid[0:21, 0:21]
    for x0, y0, a in ((10.3, 9.7, 200.0), (4.2, 15.1, 60.0)):
        img += a * np.exp(-0.5 * (((xx - x0) / 1.3) ** 2 + ((yy - y0) / 1.3) ** 2))
    return img


def test_pass2_catalog_equals_daostarfinder_table() -> None:
    img = _image()
    for fwhm, thr in ((3.1, 20.0), (3.1, 45.0), (2.4, 20.0)):
        tab = DAOStarFinder(fwhm=fwhm, threshold=thr, n_brightest=None,
                            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER)(img)
        cat = _pass2_catalog(img, fwhm, thr)
        assert (tab is None) == (cat is None)
        if tab is None:
            continue
        for col in ("x_centroid", "y_centroid", "flux", "peak"):
            np.testing.assert_array_equal(np.asarray(tab[col]), np.asarray(getattr(cat, col)))


def test_roundtrip_check_passes_for_a_plain_shift() -> None:
    from vyvar_alignment_frame import _roundtrip_needed

    w1 = WCS(naxis=2)
    w1.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w1.wcs.crval = [240.0, 50.0]
    w1.wcs.crpix = [500.0, 400.0]
    w1.wcs.cdelt = [-0.0027, 0.0027]
    w2 = w1.deepcopy()
    w2.wcs.crpix = [503.5, 398.0]
    assert _roundtrip_needed(w1, w2, (800, 1000)) is False
