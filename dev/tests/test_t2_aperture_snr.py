# -*- coding: ascii -*-
"""T2: MASTERSTAR snr is aperture SNR, not peak/sigma."""

from __future__ import annotations

import numpy as np
import pandas as pd

from photometry_core import stamp_masterstar_snr_columns


def test_snr_uses_flux_over_empirical_err_not_peak() -> None:
    df = pd.DataFrame(
        {
            "flux": [2781.54],
            "peak_dao": [157.70],
            "x": [10.0],
            "y": [10.0],
        }
    )
    out = stamp_masterstar_snr_columns(
        df,
        image=None,
        fwhm_dao_px=5.195,
        bg_sigma_adu=24.35,
        gain=1.0,
        aperture_fwhm_factor=1.9,
    )
    snr_peak = 157.70 / 24.35
    assert abs(float(out.loc[0, "snr_peak"]) - snr_peak) < 1e-6
    r_ap = 1.9 * 5.195
    area = np.pi * r_ap * r_ap
    sigma_bkg_ap = 24.35 * np.sqrt(area)
    err = np.sqrt(2781.54 / 1.0 + sigma_bkg_ap**2)
    expected = 2781.54 / err
    assert abs(float(out.loc[0, "snr"]) - expected) < 1e-6
    assert float(out.loc[0, "snr"]) != float(out.loc[0, "snr_peak"])


def test_snr_peak_not_gated_column() -> None:
    df = pd.DataFrame({"flux": [1.0e6], "peak_dao": [50.0]})
    out = stamp_masterstar_snr_columns(
        df, image=None, fwhm_dao_px=2.5, bg_sigma_adu=10.0, gain=1.0
    )
    assert "snr_peak" in out.columns
    assert float(out.loc[0, "snr_peak"]) == 5.0
    assert float(out.loc[0, "snr"]) > 10.0
