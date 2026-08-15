"""TARGET-DEPTH-02: MASTERSTAR zone depth + noise skip."""
from __future__ import annotations

import numpy as np
import pandas as pd

from comp_pool_noise import derive_target_depth_from_masterstar
from photometry_core import select_active_targets


def test_masterstar_zone_depth_half_linear():
    rows = []
    for mag in np.arange(8.0, 16.0, 0.25):
        # fully linear through G14.4; half-linear cliff in 14.5-15.0; noise after
        if mag < 14.5:
            zone = "linear"
        elif mag < 15.0:
            zone = "linear" if (int(mag * 100) % 2 == 0) else "noise"
        else:
            zone = "noise"
        for i in range(12):
            rows.append(
                {
                    "phot_g_mean_mag": float(mag + 0.01 * i),
                    "zone": zone if mag < 14.5 or mag >= 15.0 else ("linear" if i < 7 else "noise"),
                    "vsx_known_variable": False,
                }
            )
    lim = derive_target_depth_from_masterstar(pd.DataFrame(rows), masterstar_n_combine=1)
    assert lim.mode == "masterstar_zone"
    assert lim.snr_scale_factor == 1.0
    assert lim.mag_offset == 0.0
    assert lim.target_depth_g is not None
    assert abs(float(lim.target_depth_g) - 15.0) < 1e-9


def test_masterstar_single_frame_factor_is_one():
    rows = [
        {"phot_g_mean_mag": 10.0 + 0.01 * i, "zone": "linear", "vsx_known_variable": False}
        for i in range(20)
    ] + [
        {"phot_g_mean_mag": 14.2 + 0.01 * i, "zone": "noise", "vsx_known_variable": False}
        for i in range(20)
    ]
    lim = derive_target_depth_from_masterstar(pd.DataFrame(rows), masterstar_n_combine=1)
    assert lim.masterstar_n_combine == 1
    assert lim.snr_scale_factor == 1.0


def test_select_active_targets_noise_sets_skip(tmp_path):
    ms = pd.DataFrame(
        [
            {
                "name": "100",
                "catalog_id": "100",
                "x": 100.0,
                "y": 100.0,
                "mag": 12.0,
                "zone": "noise",
                "b_v": 0.5,
                "bp_rp": 0.8,
                "snr50_ok": True,
            },
            {
                "name": "200",
                "catalog_id": "200",
                "x": 120.0,
                "y": 120.0,
                "mag": 11.0,
                "zone": "linear",
                "b_v": 0.5,
                "bp_rp": 0.8,
                "snr50_ok": True,
            },
        ]
    )
    vt = pd.DataFrame(
        [
            {
                "name": "n1",
                "vsx_name": "N1",
                "vsx_type": "RRAB",
                "vsx_period": 0.5,
                "priority": 1,
                "ra_deg": 10.0,
                "dec_deg": 20.0,
                "x": 100.0,
                "y": 100.0,
                "catalog_id": "100",
                "mag": 12.0,
                "gaia_match_source": "masterstars",
                "target_origin": "vsx_auto",
            },
            {
                "name": "n2",
                "vsx_name": "N2",
                "vsx_type": "RRAB",
                "vsx_period": 0.5,
                "priority": 1,
                "ra_deg": 10.1,
                "dec_deg": 20.1,
                "x": 120.0,
                "y": 120.0,
                "catalog_id": "200",
                "mag": 11.0,
                "gaia_match_source": "masterstars",
                "target_origin": "vsx_auto",
            },
        ]
    )
    ms_p = tmp_path / "ms.csv"
    vt_p = tmp_path / "vt.csv"
    ms.to_csv(ms_p, index=False)
    vt.to_csv(vt_p, index=False)
    # Avoid VSX auto filter complexity: pass as non-auto by not matching is_vsx_auto
    # Identity join still works for catalog_id present.
    out = select_active_targets(
        vt_p,
        ms_p,
        frame_w_px=400,
        frame_h_px=400,
        edge_margin_px=10,
        target_depth_g=None,
    )
    assert not out.empty
    by = out.set_index(out["catalog_id"].astype(str))
    assert bool(by.loc["100", "skip_photometry"]) is True
    assert str(by.loc["100", "skip_reason"]) == "zone_noise"
    assert bool(by.loc["200", "skip_photometry"]) is False
