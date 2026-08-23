"""FD-A: full CCD variance model for PSF fit weights (EPSF-BRIGHT-01 Phase 3)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from photutils.psf import ImagePSF

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psf_photometry import (  # noqa: E402
    _psf_fit_error_cutout,
    _psf_fit_error_cutout_full_ccd,
    _psf_sandwich_flux_err,
    _psf_variance_per_px_adu2,
    psf_photometry_stars,
)


def _make_gaussian_epsf(n: int = 25, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    g = np.exp(-0.5 * ((xx - c) ** 2 + (yy - c) ** 2) / sigma**2)
    g /= g.sum()
    return g.astype(np.float64)


def test_full_variance_gain_propagation() -> None:
    f = np.array([1000.0, 100.0, 10.0])
    sky, rn, g_lo, g_hi = 50.0, 9.0, 0.5, 2.0
    var_lo = _psf_variance_per_px_adu2(f, sky_per_px=sky, gain=g_lo, read_noise_e=rn)
    var_hi = _psf_variance_per_px_adu2(f, sky_per_px=sky, gain=g_hi, read_noise_e=rn)
    assert np.all(var_hi < var_lo)
    assert var_lo[0] > var_lo[1] > var_lo[2]


def test_sky_only_vs_full_error_map_bright_star() -> None:
    psf = _make_gaussian_epsf()
    model = ImagePSF(psf, oversampling=1)
    shape = psf.shape
    flux = 50_000.0
    sky, gain, rn = 300.0, 0.637, 9.0
    err_sky = _psf_fit_error_cutout(shape, sky_per_px=sky, gain=gain, read_noise_e=rn)
    err_full = _psf_fit_error_cutout_full_ccd(
        shape,
        psf_model=model,
        flux_init=flux,
        x_0=(shape[1] - 1) / 2.0,
        y_0=(shape[0] - 1) / 2.0,
        sky_per_px=sky,
        gain=gain,
        read_noise_e=rn,
    )
    assert float(np.max(err_full)) > float(np.max(err_sky))
    core = err_full > err_sky * 1.5
    assert int(core.sum()) >= 1


def test_perfect_bright_star_reduced_chi2_near_unity(tmp_path: Path) -> None:
    """Noiseless inject-and-recover: full variance model yields honest reduced chi2."""
    psf_arr = _make_gaussian_epsf(n=25, sigma=2.2)
    epsf_path = tmp_path / "masterstar_epsf.fits"
    meta_path = tmp_path / "masterstar_epsf_meta.json"
    fits.PrimaryHDU(data=psf_arr.astype(np.float32)).writeto(epsf_path, overwrite=True)
    meta_path.write_text(
        '{"cutout_size": 25, "oversampling": 1, "fwhm_px": 6.0}',
        encoding="utf-8",
    )

    model = ImagePSF(psf_arr, oversampling=1)
    cutout_size = 25
    c = (cutout_size - 1) / 2.0
    flux = 40_000.0
    sky = 300.0
    gain = 0.637
    rn = 9.0

    yy, xx = np.mgrid[0:cutout_size, 0:cutout_size]
    xg = xx.astype(np.float64).ravel()
    yg = yy.astype(np.float64).ravel()
    npx = xg.size
    src = np.asarray(
        model.evaluate(
            xg,
            yg,
            np.full(npx, flux, dtype=np.float64),
            np.full(npx, c, dtype=np.float64),
            np.full(npx, c, dtype=np.float64),
        ),
        dtype=np.float64,
    ).reshape(cutout_size, cutout_size)
    frame = (sky + src).astype(np.float32)

    hdr = fits.Header()
    hdr["GAIN"] = gain
    hdr["RDNOISE"] = rn
    pos = pd.DataFrame([{"catalog_id": "bright", "name": "b", "x": float(c), "y": float(c)}])
    ref = np.array([flux])

    df = psf_photometry_stars(
        frame,
        hdr,
        pos,
        epsf_path,
        cutout_size=cutout_size,
        ref_fluxes=ref,
        apply_aperture_correction=False,
        grouper_enabled=False,
        quality_fallback_enabled=False,
        use_iterative=False,
    )
    row = df.iloc[0]
    assert row.get("psf_weight_mode") == "full_ccd"
    assert row.get("psf_err_mode") == "sandwich_full_ccd"
    chi2 = float(row["psf_chi2"])
    assert math.isfinite(chi2)
    assert chi2 < 5.0, f"expected honest chi2 for noiseless perfect fit, got {chi2}"
    assert bool(row["psf_fit_ok"])


def test_sandwich_flux_err_uses_full_variance() -> None:
    psf = _make_gaussian_epsf()
    model = ImagePSF(psf, oversampling=1)
    shape = psf.shape
    flux = 20_000.0
    cx = cy = (shape[1] - 1) / 2.0
    err = _psf_sandwich_flux_err(
        flux,
        model,
        cx,
        cy,
        shape,
        sky_per_px=300.0,
        gain=0.637,
        read_noise_e=9.0,
        fit_shape=(shape[0], shape[1]),
    )
    assert math.isfinite(err) and err > 0
