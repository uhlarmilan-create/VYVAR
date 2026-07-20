"""T3: preprocess order-2 sky surface subtract (shared calibrated->processed)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from pipeline import _fit_subtract_preprocess_sky_surface, _preprocess_calibrated_one


def _synthetic_gradient_with_stars(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
  h, w = shape
  yy, xx = np.mgrid[0:h, 0:w]
  grad = 1200.0 + 0.15 * xx.astype(np.float32) + 0.08 * yy.astype(np.float32)
  stars = np.zeros_like(grad)
  for cy, cx, amp in ((80, 90, 800.0), (170, 200, 1200.0), (120, 180, 600.0)):
    stars += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 2.0**2)))
  return (grad + stars).astype(np.float32)


def test_order0_bypass_leaves_frame_identical() -> None:
    data = _synthetic_gradient_with_stars()
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=0)
    assert stats["sky_surface_applied"] is False
    np.testing.assert_array_equal(out, data)


def test_order2_flattens_gradient_only_frame() -> None:
    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w]
    data = (1000.0 + 0.05 * xx + 0.03 * yy).astype(np.float32)
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=2)
    assert stats["sky_surface_applied"] is True
    assert stats["sky_surface_p2p_adu"] > 1.0
    assert not np.allclose(out, data)


def test_order2_preserves_star_fluxes() -> None:
    data = _synthetic_gradient_with_stars()
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=2)
    assert stats["sky_surface_applied"] is True
    assert stats["sky_surface_p2p_adu"] > 10.0

    # Star peaks remain well above local background.
    for cy, cx in ((80, 90), (170, 200), (120, 180)):
        peak = float(out[cy, cx])
        local = out[cy - 6 : cy + 7, cx - 6 : cx + 7]
        assert peak - float(np.nanmedian(local)) > 200.0


def test_preprocess_one_writes_surface_headers(tmp_path) -> None:
    cal_root = tmp_path / "calibrated" / "lights" / "NoFilter_60_2"
    proc_root = tmp_path / "processed" / "lights" / "NoFilter_60_2"
    cal_root.mkdir(parents=True)
    data = _synthetic_gradient_with_stars((128, 128))
    src = cal_root / "BO_CVn_Light_001.fits"
    fits.writeto(src, data, overwrite=True)

    row = _preprocess_calibrated_one(
        src,
        calibrated_root=cal_root.parents[1],
        processed_root=proc_root.parents[1],
        reject_fwhm_px=None,
        reject_elongation=None,
        inject_pointing_ra_deg=None,
        inject_pointing_dec_deg=None,
        inject_pointing_only_if_missing=True,
        sky_surface_order=2,
    )
    assert row["sky_surface_applied"] is True
    dst = proc_root / "proc_BO_CVn_Light_001.fits"
    assert dst.is_file()
    with fits.open(dst) as hdul:
        hdr = hdul[0].header
        assert int(hdr["VYSKYORD"]) == 2
        assert float(hdr["VYSKYP2P"]) > 0.0


def test_preprocess_one_order0_is_pixel_copy(tmp_path) -> None:
    cal_root = tmp_path / "calibrated" / "lights" / "NoFilter_60_2"
    proc_root = tmp_path / "processed" / "lights" / "NoFilter_60_2"
    cal_root.mkdir(parents=True)
    data = _synthetic_gradient_with_stars((64, 64))
    src = cal_root / "BO_CVn_Light_002.fits"
    fits.writeto(src, data, overwrite=True)

    row = _preprocess_calibrated_one(
        src,
        calibrated_root=cal_root.parents[1],
        processed_root=proc_root.parents[1],
        reject_fwhm_px=None,
        reject_elongation=None,
        inject_pointing_ra_deg=None,
        inject_pointing_dec_deg=None,
        inject_pointing_only_if_missing=True,
        sky_surface_order=0,
    )
    assert row.get("sky_surface_applied") is False
    dst = proc_root / "proc_BO_CVn_Light_002.fits"
    with fits.open(src) as h1, fits.open(dst) as h2:
        np.testing.assert_array_equal(h1[0].data, h2[0].data)
