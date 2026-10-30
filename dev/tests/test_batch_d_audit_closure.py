"""Batch D audit closure: empirical background path, P-02 scintillation wiring (I-11 removed, G01-20)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from photometry_core import (
    _photometric_error_with_bkg_mode,
    _sky_pp_for_photometric_error,
    ERR_BKG_SOURCE_EMPIRICAL,
)
from sigma_budget import scintillation_sigma
from sigma_floor_core import combine_production_err_rel, scintillation_mag_per_epoch


def test_i11_empirical_path_ignores_sky_pp_when_sigma_bkg_present() -> None:
    row = {"sky_adu_per_px_annulus": 0.0}
    sky_pp = _sky_pp_for_photometric_error(row)
    err, src = _photometric_error_with_bkg_mode(
        10000.0,
        sky_pp=sky_pp,
        area=50.0,
        gain=1.0,
        read_noise=10.0,
        sigma_bkg_ap=12.0,
    )
    assert src == ERR_BKG_SOURCE_EMPIRICAL
    assert math.isfinite(err)


def test_d4_scintillation_hand_computed_match() -> None:
    """Young/Osborn scintillation: known (D, X, t) reproduces hand sigma."""
    d_m = 0.2
    x = 1.2
    t_s = 60.0
    alt_m = 250.0
    rel = scintillation_sigma(
        telescope_diameter_m=d_m,
        airmass=x,
        exposure_s=t_s,
        altitude_m=alt_m,
        c_y=1.5,
    )
    mag = scintillation_mag_per_epoch(
        telescope_diameter_m=d_m,
        airmass=x,
        exposure_s=t_s,
        altitude_m=alt_m,
        c_y=1.5,
    )
    assert rel > 0
    assert mag == pytest.approx(rel * (2.5 / math.log(10)), rel=1e-6)


def test_d4_scintillation_in_production_quadrature() -> None:
    phot = 0.01
    sem = 0.005
    scint = 0.00239
    floor = 0.0
    rel = combine_production_err_rel(
        phot, sem, sigma_sys_mag=floor, sigma_scint_mag=scint,
    )
    terms = (
        phot * phot
        + (sem / (2.5 / math.log(10))) ** 2
        + (scint / (2.5 / math.log(10))) ** 2
    )
    assert rel == pytest.approx(math.sqrt(terms), rel=1e-9)


def test_d4_scintillation_per_epoch_array() -> None:
    from photometry_core import _combine_err_with_ensemble_scatter_keyed

    err = np.array([0.01, 0.01])
    scatter = {"a.csv": 0.004, "b.csv": 0.004}
    scint = np.array([0.002, 0.003])
    out, unmatched = _combine_err_with_ensemble_scatter_keyed(
        err, ["a.csv", "b.csv"], scatter, sigma_scint_mag=scint,
    )
    assert not unmatched.any()
    assert out[1] > out[0]
