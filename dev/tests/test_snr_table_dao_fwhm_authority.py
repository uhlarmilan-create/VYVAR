"""A-1: SNR aperture table FWHM authority prefers per-frame DAO moment, not VY_FWHM_GAUSS."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from photometry_core import (
    precompute_and_save_snr_aperture_table_for_draft,
    resolve_fwhm_px_for_snr_aperture_table,
)
def _synthetic_star_field(shape: tuple[int, int] = (256, 256), *, n_stars: int = 40) -> np.ndarray:
    rng = np.random.default_rng(42)
    img = np.full(shape, 1200.0, dtype=np.float32)
    img += rng.normal(0.0, 8.0, size=shape).astype(np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for _ in range(n_stars):
        x0 = float(rng.integers(30, shape[1] - 30))
        y0 = float(rng.integers(30, shape[0] - 30))
        sigma = float(rng.uniform(1.1, 1.6))
        amp = float(rng.uniform(400.0, 1200.0))
        img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))
    return img.astype(np.float32)


def test_resolve_fwhm_skips_vy_fwhm_gauss_for_sizing(tmp_path: Path) -> None:
    ms = tmp_path / "MASTERSTAR.fits"
    data = _synthetic_star_field()
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["VY_FWHM_GAUSS"] = (2.2, "stack gauss - record only")
    hdu.header["VY_FWHM"] = (9.9, "dao header - not frame median")
    hdu.writeto(ms, overwrite=True)

    aligned = tmp_path / "Light_001.fits"
    fits.PrimaryHDU(data=data.copy()).writeto(aligned, overwrite=True)

    fw, prov = resolve_fwhm_px_for_snr_aperture_table(
        masterstar_fits_path=ms,
        masterstar_selection={},
        aligned_fits_paths=[aligned],
    )
    assert fw is not None
    assert math.isfinite(float(fw))
    assert prov.get("vy_fwhm_gauss_px") == pytest.approx(2.2)
    assert prov.get("fwhm_px_scope") == "per_draft_median_frame_dao_moment"
    assert abs(float(fw) - 2.2) > 0.2


def test_precompute_snr_table_writes_dao_provenance(tmp_path: Path) -> None:
    data = _synthetic_star_field()
    ms = tmp_path / "MASTERSTAR.fits"
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["VY_FWHM_GAUSS"] = (2.5, "record")
    hdu.writeto(ms, overwrite=True)
    aligned = tmp_path / "Light_001.fits"
    fits.PrimaryHDU(data=data.copy()).writeto(aligned, overwrite=True)

    out = precompute_and_save_snr_aperture_table_for_draft(
        tmp_path,
        masterstar_fits_path=ms,
        aligned_fits_paths=[aligned],
        fwhm_fallback_px=3.0,
        sky_fallback=100.0,
    )
    assert out is not None
    tbl = json.loads((tmp_path / "aperture_snr_table.json").read_text(encoding="utf-8"))
    assert tbl.get("fwhm_px_scope") == "per_draft_median_frame_dao_moment"
    assert tbl.get("vy_fwhm_gauss_px") == pytest.approx(2.5)
    assert float(tbl["fwhm_px"]) != pytest.approx(2.5)
