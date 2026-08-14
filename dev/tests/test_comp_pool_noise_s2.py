"""COMP-POOL-01 Stage 2 unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from comp_pool_noise import (
    NoiseCurveFit,
    DerivedPoolThresholds,
    admit_pool_stars,
    derive_pool_thresholds,
    nonparametric_noise_curve,
    _mag_where_phot_equals_sys,
)


def _fit(**kwargs) -> NoiseCurveFit:
    base = dict(
        n_stars=10,
        gain_e_per_adu=3.17,
        read_noise_e=7.6,
        sky_adu_median=1500.0,
        aperture_area_px_median=36.0,
        zp_inst=22.0,
        sigma_sys_mag=0.01,
        sigma_sys_mag_err=0.001,
        chi2_red=1.0,
        n_fit=10,
        scint_mag_predicted=0.002,
        scint_rel_predicted=0.002,
        scint_airmass_used=1.1,
    )
    base.update(kwargs)
    return NoiseCurveFit(**base)


def test_mag_where_phot_equals_sys_finite():
    fit = _fit()
    m = _mag_where_phot_equals_sys(fit)
    assert m is not None and 8.0 < m < 16.0


def test_admit_rejects_catalogue_variable():
    fit = _fit()
    thr = DerivedPoolThresholds(
        faint_limit_g=12.0,
        faint_limit_snr_approx=80.0,
        bright_limit_g=None,
        bright_upturn_visible=False,
        default_lin_frac=0.85,
        detect_frac_min=0.9,
        detect_frac_rule="test",
        dilution_threshold=0.9,
        dilution_rule="test",
        stability_excess_mad=3.0,
        stability_excess_iqr=3.0,
        stability_excess_inv_eta=2.0,
        stability_rule="test",
        nonparametric_min_bin_n=8,
        nonparametric_usable_above_g=None,
    )
    stars = pd.DataFrame(
        [
            {
                "catalog_id": "1",
                "mag_g": 10.0,
                "detect_frac": 1.0,
                "scatter_mad": 0.01,
                "scatter_iqr": 0.01,
                "inv_eta": 0.5,
                "dilution_factor": 1.0,
                "flux_median": 1.0e5,
                "sky_median": 1500.0,
                "aperture_r_median": 3.4,
                "vsx_known_variable": True,
                "gaia_variable_flag": False,
            },
            {
                "catalog_id": "2",
                "mag_g": 10.0,
                "detect_frac": 1.0,
                "scatter_mad": 0.01,
                "scatter_iqr": 0.01,
                "inv_eta": 0.5,
                "dilution_factor": 1.0,
                "flux_median": 1.0e5,
                "sky_median": 1500.0,
                "aperture_r_median": 3.4,
                "vsx_known_variable": False,
                "gaia_variable_flag": False,
            },
        ]
    )
    dec = admit_pool_stars(stars, fit, thr)
    assert bool(dec.loc[dec.catalog_id == "1", "admit"].iloc[0]) is False
    assert bool(dec.loc[dec.catalog_id == "2", "admit"].iloc[0]) is True


def test_dilution_missing_does_not_reject():
    fit = _fit()
    thr = DerivedPoolThresholds(
        faint_limit_g=12.0,
        faint_limit_snr_approx=80.0,
        bright_limit_g=None,
        bright_upturn_visible=False,
        default_lin_frac=0.85,
        detect_frac_min=0.5,
        detect_frac_rule="test",
        dilution_threshold=0.95,
        dilution_rule="test",
        stability_excess_mad=5.0,
        stability_excess_iqr=5.0,
        stability_excess_inv_eta=5.0,
        stability_rule="test",
        nonparametric_min_bin_n=8,
        nonparametric_usable_above_g=None,
    )
    stars = pd.DataFrame(
        [
            {
                "catalog_id": "3",
                "mag_g": 10.0,
                "detect_frac": 1.0,
                "scatter_mad": 0.01,
                "scatter_iqr": 0.01,
                "inv_eta": 0.5,
                "dilution_factor": float("nan"),
                "flux_median": 1.0e5,
                "sky_median": 1500.0,
                "aperture_r_median": 3.4,
                "vsx_known_variable": False,
                "gaia_variable_flag": False,
            }
        ]
    )
    dec = admit_pool_stars(stars, fit, thr)
    assert bool(dec["admit"].iloc[0]) is True


def test_dilution_pileup_steps_percentile():
    rng = np.random.default_rng(0)
    n = 200
    dvals = np.ones(n)
    dvals[:30] = rng.uniform(0.5, 0.95, size=30)
    stars = pd.DataFrame(
        {
            "catalog_id": [str(i) for i in range(n)],
            "mag_g": rng.uniform(8, 12, size=n),
            "detect_frac": np.ones(n),
            "scatter_mad": np.full(n, 0.01),
            "scatter_iqr": np.full(n, 0.01),
            "inv_eta": np.full(n, 0.5),
            "dilution_factor": dvals,
            "flux_median": np.full(n, 1.0e5),
            "sky_median": np.full(n, 1500.0),
            "aperture_r_median": np.full(n, 3.4),
            "vsx_known_variable": np.zeros(n, dtype=bool),
            "gaia_variable_flag": np.zeros(n, dtype=bool),
        }
    )
    np_curve = nonparametric_noise_curve(stars)
    fit = _fit()
    thr = derive_pool_thresholds(stars, np_curve, fit)
    assert thr.dilution_threshold is not None
    assert thr.dilution_threshold < 0.999
    assert "p" in thr.dilution_rule
