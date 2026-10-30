"""EPSF-UPGRADE-2026-10: ePSF build on photutils >= 3.1.1 with sky-subtracted build stars.

Regression guards for the two causes of the diverging MASTERSTAR ePSF (draft 535, edge/peak 1.9):
the build stars carried a flux that included the sky, and photutils 3.0 never updated that
normalization (photutils #2386). Plus truth recovery on synthetic stars and the NaN case that the
removed build guard (INV-EPSF-BUILD-GUARD-01) used to work around (photutils #2423).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import photutils
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.table import Table
from packaging.version import Version
from photutils.psf import CircularGaussianPRF, EPSFStar, EPSFStars, ImagePSF

from psf_photometry import _epsf_build_imagepsf_from_stars, _epsf_sky_subtracted_stars

FIXTURE = Path(__file__).parent / "fixtures" / "epsf_build_stars_535.npz"


def _edge_over_peak(a: np.ndarray) -> float:
    edge = max(np.abs(a[:, 0]).max(), np.abs(a[0]).max(), np.abs(a[:, -1]).max(), np.abs(a[-1]).max())
    return float(edge / a.max())


def _synthetic_field(fwhm: float, n: int, *, seed: int, sky: float = 0.0, nan_first: bool = False):
    """Isolated pixel-integrated Gaussian stars on a grid, Poisson + read noise."""
    rng = np.random.default_rng(seed)
    g = int(np.ceil(np.sqrt(n)))
    step = 40
    side = step * g + step
    yy, xx = np.mgrid[0:side, 0:side]
    img = np.zeros((side, side))
    xs, ys, fl = [], [], []
    for i in range(n):
        cx = step + step * (i % g) + rng.uniform(-0.5, 0.5)
        cy = step + step * (i // g) + rng.uniform(-0.5, 0.5)
        f = 10 ** rng.uniform(4.5, 5.3)
        img += CircularGaussianPRF(flux=f, x_0=cx, y_0=cy, fwhm=fwhm)(xx, yy)
        xs.append(cx)
        ys.append(cy)
        fl.append(f)
    img = rng.poisson(img + sky + 100.0).astype(float) - 100.0 + rng.normal(0.0, 5.0, img.shape)
    if nan_first:
        img[int(round(ys[0])), int(round(xs[0])) + 1] = np.nan
    return img, Table([xs, ys], names=("x", "y")), np.array(fl)


def test_photutils_is_the_fixed_release() -> None:
    assert Version(photutils.__version__) >= Version("3.1.1")


def test_build_stars_are_sky_subtracted_with_matching_flux() -> None:
    img, cat, flux = _synthetic_field(3.0, 4, seed=7, sky=500.0)
    stars = _epsf_sky_subtracted_stars(img, cat, 19)
    for star, f in zip(stars, flux, strict=True):
        assert abs(float(np.median(star.data[0]))) < 30.0  # sky gone from the border
        assert star.flux == pytest.approx(float(np.sum(star.data)), rel=1e-6)
        # noise: sky Poisson in 361 px + border-median sky error x 361 px ~ 2.5 % of the faintest star;
        # with the raw cutout the flux was ~ f + 361 * 500 (x 2.5 .. x 6)
        assert star.flux == pytest.approx(f, rel=0.08)


def test_masterstar_535_build_is_bounded_and_converged() -> None:
    z = np.load(FIXTURE)
    stars = EPSFStars([EPSFStar(d, cutout_center=tuple(c)) for d, c in zip(z["data"], z["cutout_center"], strict=True)])
    built = _epsf_build_imagepsf_from_stars(
        stars, osamp=2, fwhm_px=float(z["fwhm_px"])
    )
    arr = built["arr"].astype(float)
    assert _edge_over_peak(arr) < 0.02  # was 1.9 (photutils 3.0 + sky-in-flux stars)
    assert built["epsf_sum_native"] == pytest.approx(1.0, abs=1e-6)
    assert built["build_converged"] is True
    assert built["smoothing"].startswith("auto:")
    assert built["build_iterations"] and built["build_iterations"][-1]["converged"] is True
    assert 0.9 < built["qc"]["epsf_vs_input_fwhm_ratio"] < 1.3


def test_synthetic_truth_recovery() -> None:
    fwhm, osamp = 3.0, 2
    img, cat, _ = _synthetic_field(fwhm, 64, seed=11)
    built = _epsf_build_imagepsf_from_stars(
        _epsf_sky_subtracted_stars(img, cat, 19), osamp=osamp, fwhm_px=fwhm
    )
    arr = built["arr"].astype(float)
    ny, nx = arr.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    truth = CircularGaussianPRF(flux=1.0, x_0=(nx - 1) / 2 / osamp, y_0=(ny - 1) / 2 / osamp, fwhm=fwhm)(
        xx / osamp, yy / osamp
    )
    assert np.max(np.abs(arr - truth)) / truth.max() < 0.03

    model = ImagePSF(arr, oversampling=osamp)
    for x0, y0 in ((9.2, 9.7), (9.5, 9.5), (9.9, 9.1)):
        yy2, xx2 = np.mgrid[0:19, 0:19]
        star = CircularGaussianPRF(flux=1.0e5, x_0=x0, y_0=y0, fwhm=fwhm)(xx2, yy2)
        m = model.copy()
        m.x_0, m.y_0, m.flux = 9.0, 9.0, float(star.sum())
        fit = TRFLSQFitter()(m, xx2, yy2, star)
        assert fit.flux.value == pytest.approx(1.0e5, rel=0.01)
        assert abs(fit.x_0.value - x0) < 0.02 and abs(fit.y_0.value - y0) < 0.02


def test_nan_inside_a_fit_region_does_not_abort_the_build() -> None:
    img, cat, _ = _synthetic_field(2.0, 49, seed=3, nan_first=True)
    built = _epsf_build_imagepsf_from_stars(
        _epsf_sky_subtracted_stars(img, cat, 15), osamp=2, fwhm_px=2.0
    )
    assert np.isfinite(built["arr"]).all()
