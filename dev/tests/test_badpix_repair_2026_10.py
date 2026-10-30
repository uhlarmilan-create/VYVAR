"""BADPIX-REPAIR-2026-10: missing pixels filled from the frame's ePSF model before alignment."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.io import fits

from badpix_repair import REPAIR_HDU, calibrated_saturation_adu, repair_missing_pixels, repair_table_hdu
from badpix_sidecar import (
    load_repaired_sidecar,
    read_repair_table,
    repaired_in_aperture,
    repaired_xy_on_reference,
    write_repaired_sidecar,
)

FWHM = 3.2
SKY = 200.0
GAIN = 1.0


def _moffat(xx, yy, x0, y0, flux, fwhm=FWHM, beta=2.5):
    a = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    norm = (beta - 1.0) / (math.pi * a * a)
    return flux * norm * (1.0 + ((xx - x0) ** 2 + (yy - y0) ** 2) / (a * a)) ** (-beta)


def _field(seed=1, n_side=9, size=640):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    img = np.full((size, size), SKY)
    stars = []
    step = size / (n_side + 1)
    for i in range(n_side):
        for j in range(n_side):
            x0 = step * (i + 1) + rng.uniform(-3, 3)
            y0 = step * (j + 1) + rng.uniform(-3, 3)
            f = 10 ** rng.uniform(4.0, 5.3)
            img += _moffat(xx, yy, x0, y0, f)
            stars.append((x0, y0, f))
    img = rng.poisson(img * GAIN) / GAIN
    return img.astype(np.float32), stars


def _ap(img, x, y, r):
    from photutils.aperture import CircularAperture, aperture_photometry

    return float(aperture_photometry(img, CircularAperture((x, y), r), method="exact")["aperture_sum"][0]) - SKY * math.pi * r * r


def test_star_pixel_is_filled_from_the_psf_within_the_noise():
    img, stars = _field()
    work = img.copy()
    picks = []
    for x0, y0, f in stars[::3]:
        px, py = int(round(x0)) + 1, int(round(y0))
        work[py, px] = np.nan
        picks.append((x0, y0, f))
    res = repair_missing_pixels(work, fwhm_px=FWHM, detection_sigma=5.0, sat_cal_adu=1e9)
    assert res.n_left == 0 and res.n_psf == len(picks) and res.n_sky == 0
    assert np.isfinite(res.data).all()
    z = []
    for x0, y0, _f in picks:
        r = 1.0 * FWHM
        f_true, f_rep = _ap(img, x0, y0, r), _ap(res.data.astype(float), x0, y0, r)
        sig = math.sqrt(f_true / GAIN + math.pi * r * r * SKY / GAIN)
        z.append((f_rep - f_true) / sig)
    z = np.asarray(z)
    assert abs(np.median(z)) < 0.3
    assert np.max(np.abs(z)) < 1.5  # a pixel next to the centre, still within the photon noise


def test_sky_pixel_gets_the_local_sky_and_is_listed():
    img, _ = _field(seed=2)
    work = img.copy()
    work[5, 5] = np.nan  # far from any star (stars start at size/10)
    res = repair_missing_pixels(work, fwhm_px=FWHM, detection_sigma=5.0, sat_cal_adu=1e9)
    assert res.n_sky == 1 and res.n_psf == 0
    assert abs(float(res.data[5, 5]) - SKY) < 5.0
    assert res.xy_kind == [(5, 5, "S")]


def test_without_linearity_limit_star_pixels_stay_missing_with_reason():
    img, stars = _field(seed=3)
    work = img.copy()
    x0, y0, _ = stars[10]
    work[int(round(y0)), int(round(x0))] = np.nan
    work[5, 5] = np.nan
    res = repair_missing_pixels(work, fwhm_px=FWHM, detection_sigma=5.0, sat_cal_adu=None)
    assert res.n_left == 1 and res.n_sky == 1
    assert "VY_CSAT" in res.reason_left
    assert np.isnan(res.data[int(round(y0)), int(round(x0))])


def test_no_missing_pixels_returns_the_frame_unchanged():
    img, _ = _field(seed=4, n_side=3, size=200)
    res = repair_missing_pixels(img, fwhm_px=FWHM, detection_sigma=5.0, sat_cal_adu=1e9)
    assert res.n_missing == 0 and np.array_equal(res.data, img)


def test_calibrated_saturation_uses_the_highest_flat():
    assert calibrated_saturation_adu(65535.0, 35.0, 1.05) == pytest.approx((65535.0 - 35.0) / 1.05)
    assert calibrated_saturation_adu(None, 35.0, 1.05) is None


def test_repair_table_round_trip_and_reference_mapping(tmp_path):
    from badpix_repair import RepairResult

    res = RepairResult(data=np.zeros((4, 4)), xy_kind=[(1, 2, "P"), (3, 0, "S")])
    p = tmp_path / "f.fits"
    fits.HDUList([fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)), repair_table_hdu(res)]).writeto(p)
    with fits.open(p) as hdul:
        assert REPAIR_HDU in hdul
        tab = read_repair_table(hdul)
    assert tab.tolist() == [[1.0, 2.0, 1.0], [3.0, 0.0, 0.0]]
    hdr = fits.Header()
    for i, v in enumerate([1, 0, 10.5, 0, 1, -2.0, 0, 0, 1]):
        hdr[f"VY_AM{i}"] = v
    ref = repaired_xy_on_reference(tab, hdr)
    assert ref == [[11.5, 0.0, 1], [13.5, -2.0, 0]]
    write_repaired_sidecar(tmp_path, {"f.fits": ref})
    write_repaired_sidecar(tmp_path, {"g.fits": []})
    side = load_repaired_sidecar(tmp_path)
    assert set(side) == {"f", "g"} and side["g"].shape == (0, 3)
    assert repaired_in_aperture(side["f"], 11.5, 3.0, 2.5) == 1
    assert repaired_in_aperture(side["f"], 11.5, 6.0, 2.5) == 0


def test_missing_peak_pixel_does_not_hide_the_star():
    """A missing pixel at a star's peak must still be filled from that star's model, not with the sky."""
    img, stars = _field(seed=5)
    work = img.copy()
    picks = stars[1::4]
    for x0, y0, _f in picks:
        work[int(round(y0)), int(round(x0))] = np.nan
    res = repair_missing_pixels(work, fwhm_px=FWHM, detection_sigma=5.0, sat_cal_adu=1e9)
    assert res.n_psf == len(picks) and res.n_sky == 0 and res.n_left == 0
