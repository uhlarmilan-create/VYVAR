# -*- coding: ascii -*-
"""COMP-RMS-DEF-01-B + ZONE-SAT-01 positive controls."""
from __future__ import annotations


import pandas as pd

from photometry_core import _select_comps_by_rms_then_color
from pipeline import _annotate_masterstars_flux_zones


def test_selector_old_4p3_loo_016_passes() -> None:
    df = pd.DataFrame(
        [
            {
                "catalog_id": "BRIGHT",
                "bp_rp": 1.0,
                "comp_rms": 0.016,
                "_dist_deg": 0.1,
                "_nn_dist_fwhm": 5.0,
                "comp_sigma_ref_fieldk": 1.0857362047581294 / 40.0,
            },
            {
                "catalog_id": "FAINTNOISY",
                "bp_rp": 1.0,
                "comp_rms": 0.187,
                "_dist_deg": 0.1,
                "_nn_dist_fwhm": 5.0,
                "comp_sigma_ref_fieldk": 0.025,
            },
            {
                "catalog_id": "OK2",
                "bp_rp": 1.0,
                "comp_rms": 0.02,
                "_dist_deg": 0.2,
                "_nn_dist_fwhm": 5.0,
                "comp_sigma_ref_fieldk": 1.0857362047581294 / 40.0,
            },
            {
                "catalog_id": "OK3",
                "bp_rp": 1.0,
                "comp_rms": 0.03,
                "_dist_deg": 0.3,
                "_nn_dist_fwhm": 5.0,
                "comp_sigma_ref_fieldk": 1.0857362047581294 / 40.0,
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


