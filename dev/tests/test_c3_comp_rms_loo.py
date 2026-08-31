# -*- coding: ascii -*-
"""COMP-RMS-DEF-01-B + ZONE-SAT-01 positive controls."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from comp_rms_loo import (
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    compute_loo_mag_rms_map,
    loo_ceiling_mag,
    mad_sigma_scale_or_zero,
    photon_sigma_mag,
)
from photometry_core import _select_comps_by_rms_then_color
from pipeline import _annotate_masterstars_flux_zones


def test_loo_mad_puts_4mag_selector_star_at_honest_scatter() -> None:
    """C3-5: old-statistic 4.3 with LOO 0.016 must pass the new gate."""
    n = 25
    pool = ["A", "B", "C", "D"]
    cache: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(0)
    for i in range(n):
        med = 10000.0
        rows = []
        for j, cid in enumerate(pool):
            if cid == "A":
                flux = med * (10 ** (-0.4 * (0.016 / 1.4826) * rng.normal()))
            else:
                flux = med * (1.0 + 0.001 * rng.normal())
            rows.append({"catalog_id": cid, "dao_flux": float(flux)})
        cache[f"f{i}"] = pd.DataFrame(rows)
    loo, basis = compute_loo_mag_rms_map(set(pool), list(cache.keys()), cache)  # type: ignore[arg-type]
    assert basis == "all_loadable"
    assert loo["A"] < 0.05
    snr = 80.0
    ceil = loo_ceiling_mag(snr, k=5.0, abs_max=0.1)
    assert loo["A"] <= ceil


def test_loo_187_fails_k5_photon_025() -> None:
    """C3-5: LOO 0.187 and photon 0.025 (k=5 -> 0.126) fails."""
    snr = 1.0857362047581294 / 0.025
    ceil = loo_ceiling_mag(snr, k=5.0, abs_max=0.1)
    assert abs(ceil - 0.1) < 1e-9 or abs(ceil - 0.125) < 0.01
    assert 0.187 > ceil


def test_selector_old_4p3_loo_016_passes() -> None:
    df = pd.DataFrame(
        [
            {
                "catalog_id": "BRIGHT",
                "bp_rp": 1.0,
                "comp_rms": 0.016,
                "_dist_deg": 0.1,
                "_nn_dist_fwhm": 5.0,
                "snr_ap_pixscaled": 40.0,
            },
            {
                "catalog_id": "FAINTNOISY",
                "bp_rp": 1.0,
                "comp_rms": 0.187,
                "_dist_deg": 0.1,
                "_nn_dist_fwhm": 5.0,
                "snr_ap_pixscaled": 1.0857362047581294 / 0.025,
            },
            {
                "catalog_id": "OK2",
                "bp_rp": 1.0,
                "comp_rms": 0.02,
                "_dist_deg": 0.2,
                "_nn_dist_fwhm": 5.0,
                "snr_ap_pixscaled": 40.0,
            },
            {
                "catalog_id": "OK3",
                "bp_rp": 1.0,
                "comp_rms": 0.03,
                "_dist_deg": 0.3,
                "_nn_dist_fwhm": 5.0,
                "snr_ap_pixscaled": 40.0,
            },
        ]
    )
    out = _select_comps_by_rms_then_color(
        df, target_bprp=1.0, n_comp_min=3, n_comp_max=8, max_delta_bprp=0.5
    )
    ids = set(out["catalog_id"].astype(str))
    assert "BRIGHT" in ids
    assert "FAINTNOISY" not in ids


def test_clipped_peak_zone_saturated() -> None:
    df = pd.DataFrame(
        {
            "flux": [730331.0],
            "peak_max_adu": [88781.5],
            "peak_dao": [50134.7],
        }
    )
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=40.0,
        equipment_saturate_adu=65535.0,
        saturate_limit_adu_fallback=65535.0,
        n_stack=1,
        saturate_limit_fraction=1.0,
        sigma_px=40.0,
        sky_median_adu=1400.0,
        dao_detection_n_equiv=4.5,
        empirical_clip_adu=None,
    )
    assert str(out.loc[0, "zone"]) == "saturated"
    assert str(out.loc[0, "zone_peak_column"]) == "peak_max_adu"


def test_photon_sigma_and_k_default() -> None:
    assert COMP_RMS_LOO_PHOTON_K_DEFAULT == 5.0
    assert abs(photon_sigma_mag(10.0) - 0.10857362047581294) < 1e-9
    assert math.isnan(photon_sigma_mag(float("nan")))
    assert abs(mad_sigma_scale_or_zero(np.array([1.0, 1.0, 1.0, 2.0])) - 0.0) < 1.0
