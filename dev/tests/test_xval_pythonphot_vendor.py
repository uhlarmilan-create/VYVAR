# -*- coding: ascii -*-
"""Smoke test: vendored PythonPhot recovers a synthetic Gaussian flux."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_DEV = Path(__file__).resolve().parents[1]
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))


def _daoerf_star(ny: int, nx: int, x0: float, y0: float, height: float, sigma: float) -> np.ndarray:
    from xval_pythonphot.vendor import daoerf

    yy, xx = np.mgrid[0:ny, 0:nx]
    a = np.array([height, x0, y0, sigma, sigma], dtype=np.float64)
    f, _pder = daoerf.daoerf(xx.astype(np.float64), yy.astype(np.float64), a)
    return np.asarray(f, dtype=np.float64).reshape(ny, nx)


def test_pythonphot_vendor_recovers_synthetic_gaussian_flux(tmp_path: Path) -> None:
    from xval_pythonphot.vendor import aper, getpsf, pkfit_norecenter

    ny, nx = 96, 96
    sky = 40.0
    sigma = 2.0
    height = 800.0
    stars_xy = [(20.0, 20.0), (48.0, 22.0), (74.0, 28.0), (24.0, 70.0), (70.0, 72.0)]
    image = np.full((ny, nx), sky, dtype=np.float64)
    for x, y in stars_xy:
        image += _daoerf_star(ny, nx, x, y, height, sigma)

    xpos = np.array([p[0] for p in stars_xy], dtype=np.float64)
    ypos = np.array([p[1] for p in stars_xy], dtype=np.float64)
    zp = 25.0
    mag, _me, fl, _fe, skyv, _se, _bf, _out = aper.aper(
        image,
        xpos,
        ypos,
        phpadu=1.0,
        apr=8.0,
        zeropoint=zp,
        skyrad=[16.0, 22.0],
        badpix=[-1000.0, 1.0e7],
        exact=True,
        verbose=False,
    )
    mag1 = np.asarray(mag, dtype=np.float64).reshape(-1)[: len(xpos)]
    sky1 = np.asarray(skyv, dtype=np.float64).reshape(-1)[: len(xpos)]
    flux_aper = np.asarray(fl, dtype=np.float64).reshape(-1)[: len(xpos)]
    assert np.all(np.isfinite(mag1)), "aper failed on synthetic stars"

    psf_path = tmp_path / "synth_psf.fits"
    gauss, psf, psfmag = getpsf.getpsf(
        image,
        xpos,
        ypos,
        mag1,
        sky1,
        5.0,
        1.0,
        np.arange(len(xpos)),
        10.0,
        4.0,
        str(psf_path),
        zeropoint=zp,
        verbose=False,
    )
    assert psf_path.is_file()
    assert np.all(np.isfinite(np.asarray(gauss)))

    pk = pkfit_norecenter.pkfit_class(image, gauss, psf, 5.0, 1.0)
    x_t, y_t = stars_xy[-1]
    sky_t = float(sky1[-1])
    scale0 = float(10.0 ** (-0.4 * (float(mag1[-1]) - float(psfmag))))
    errmag, chi, sharp, niter, scale = pk.pkfit_norecenter(scale0, x_t, y_t, sky_t, 4.0)
    assert int(niter) != -1, f"pkfit singular niter={niter}"
    flux = float(scale) * 10.0 ** (0.4 * (zp - float(psfmag)))
    true_flux = float(flux_aper[-1])
    rel = abs(flux - true_flux) / true_flux
    assert rel < 0.001, (
        f"flux recover {flux:.3f} vs aper {true_flux:.3f} rel={rel:.4e} "
        f"chi={chi} sharp={sharp} niter={niter} errmag={errmag} scale={scale}"
    )
