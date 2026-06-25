"""F-HOWELL-3 Stage C: err sky term reads sky_adu_per_px_annulus with legacy fallback."""

from __future__ import annotations

import math

import pandas as pd

from photometry_core import (
    SKY_ADU_PER_PX_ANNULUS_COL,
    _photometric_error,
    _sky_pp_for_photometric_error,
)


def test_sky_pp_prefers_annulus_column() -> None:
    row = pd.Series(
        {
            SKY_ADU_PER_PX_ANNULUS_COL: 500.0,
            "noise_floor_adu": 650.0,
        }
    )
    assert _sky_pp_for_photometric_error(row) == 500.0


def test_sky_pp_falls_back_to_noise_floor() -> None:
    row = pd.Series({"noise_floor_adu": 499.3})
    assert _sky_pp_for_photometric_error(row) == 499.3


def test_sky_pp_missing_columns_returns_zero() -> None:
    assert _sky_pp_for_photometric_error(pd.Series({})) == 0.0


def test_err_uses_annulus_not_detection_floor() -> None:
    flux = 5000.0
    area = math.pi * 4.0**2
    gain, rn = 1.3, 1.3
    err_ann = _photometric_error(flux, 500.0, area, gain=gain, read_noise=rn)
    err_det = _photometric_error(flux, 650.0, area, gain=gain, read_noise=rn)
    assert err_ann < err_det
    row = pd.Series({SKY_ADU_PER_PX_ANNULUS_COL: 500.0, "noise_floor_adu": 650.0})
    err_via_row = _photometric_error(
        flux,
        _sky_pp_for_photometric_error(row),
        area,
        gain=gain,
        read_noise=rn,
    )
    assert err_via_row == err_ann
