"""SKY-SRC-MASK-2026-10 (Milan 2026-10-27, B): local sky = plain median of annulus pixels without detected sources.

Pre-fail (88d6368): the plain median of the whole annulus took the wings of neighbours and of the star itself,
so the sky was too high and the star too faint by a fixed number of ADU per pixel x aperture area.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src_py") not in sys.path:
    sys.path.insert(0, str(ROOT / "src_py"))

import pipeline  # noqa: E402,F401  (import order of the production package)
from photutils.aperture import CircularAnnulus  # noqa: E402

from sky_estimation import SKY_SRC_NSIGMA, sky_image, sky_source_mask  # noqa: E402


def _moffat(shape, x0, y0, flux, fwhm, beta=2.5):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    return flux * (beta - 1) / (np.pi * a * a) * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / a**2) ** (-beta)


def _crowded_frame(seed: int = 0, fwhm: float = 6.0, sky: float = 100.0):
    rng = np.random.default_rng(seed)
    shape = (300, 300)
    img = np.full(shape, sky) + rng.normal(0, 5.0, shape)
    for _ in range(120):  # resolved neighbours
        img += _moffat(shape, rng.uniform(0, 300), rng.uniform(0, 300), rng.uniform(2e3, 4e4), fwhm)
    img += _moffat(shape, 150, 150, 2e5, fwhm)  # the measured star
    return img, fwhm, sky


def _annulus_median(img, fwhm):
    m = CircularAnnulus((150, 150), r_in=2.7 * fwhm, r_out=5.2 * fwhm).to_mask(method="center")
    v = m.get_values(img)
    return float(np.median(v[np.isfinite(v)]))


def test_plain_annulus_median_is_biased_high_masked_is_not():
    img, fwhm, sky = _crowded_frame()
    plain = _annulus_median(img, fwhm)
    masked = _annulus_median(sky_image(img, fwhm), fwhm)
    assert plain - sky > 1.0  # pre-fix bias: wings of the star and neighbours
    assert abs(masked - sky) < 0.5 * (plain - sky)


def test_mask_respects_footprint_and_no_clipping_constants():
    img, fwhm, _ = _crowded_frame(seed=1)
    covered = np.ones(img.shape, bool)
    covered[:, :20] = False
    s = sky_image(img, fwhm, covered=covered)
    assert np.isnan(s[:, :20]).all()
    assert SKY_SRC_NSIGMA == 5.0
    assert sky_source_mask(np.full((50, 50), 7.0), 3.0).sum() == 0  # flat frame: nothing detected
    src = (ROOT / "src_py" / "sky_estimation.py").read_text(encoding="ascii")
    assert "sigma_clip" not in src and "SigmaClip" not in src


def test_psf_local_sky_uses_the_same_estimator():
    src = (ROOT / "src_py" / "epsf_frame.py").read_text(encoding="ascii")
    code = "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))
    assert "MedianBackground(" not in code and "LocalBackground(" not in code  # photutils default: 3-sigma clip
    assert "sky_image(" in src and '"local_bkg": lbkg' in src


def test_annulus_covered_by_a_bright_halo_uses_the_box():
    """A saturated star whose own halo covers the annulus (CV CVn, G 5.9) keeps a local sky (box fallback)."""
    from sky_estimation import SKY_MIN_PIX, annulus_sky_medians

    rng = np.random.default_rng(3)
    img = np.full((400, 400), 100.0) + rng.normal(0, 5.0, (400, 400))
    img += _moffat(img.shape, 200, 200, 5e7, 2.3)
    fwhm = 2.3
    s = sky_image(img, fwhm)
    ann = CircularAnnulus((200, 200), r_in=2.7 * fwhm, r_out=5.2 * fwhm).to_mask(method="center").get_values(s)
    assert np.isfinite(ann).sum() < SKY_MIN_PIX  # the annulus is all source
    v = annulus_sky_medians(s, np.array([[200.0, 200.0]]), 2.7 * fwhm, 5.2 * fwhm)[0]
    plain = float(np.median(
        CircularAnnulus((200, 200), r_in=2.7 * fwhm, r_out=5.2 * fwhm).to_mask(method="center").get_values(img)))
    assert np.isfinite(v) and abs(v - 100.0) < 0.1 * abs(plain - 100.0)  # far below the halo-biased plain median
