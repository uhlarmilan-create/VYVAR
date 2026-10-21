"""FORCED-SAT-2026-10: forced proc rows get the per-frame saturation test of the DAO rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from pipeline_catalog import stamp_forced_rows_saturation


def _frame() -> np.ndarray:
    rng = np.random.default_rng(1)
    img = 1000.0 + rng.normal(0, 5, (120, 120))
    img[58:63, 58:63] = 65535.0  # saturated core of a bright star at (60, 60)
    img[30, 30] += 3000.0  # faint star at (30, 30)
    return img


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "catalog_id": ["dao1", "forced_in_core", "forced_faint"],
        "x": [90.0, 61.0, 30.0], "y": [90.0, 61.0, 30.0],
        "ra_deg": [1.0, 1.0, 1.0], "dec_deg": [2.0, 2.0, 2.0],
        "forced_photometry": [False, True, True],
        "peak_max_adu": [1500.0, np.nan, np.nan],
        "likely_saturated": [False, np.nan, np.nan],
        "photometry_ok": [True, True, True],
        "peak_dao": [400.0, np.nan, np.nan],
    })


def test_forced_rows_get_peak_and_flag_dao_rows_untouched() -> None:
    df = _df()
    out, meta = stamp_forced_rows_saturation(df, _frame(), fits.Header(), peak_sat_limit_adu=58981.5)
    assert meta["n_forced"] == 2 and meta["n_likely_saturated"] == 1
    assert out.loc[0, list(df.columns)].equals(df.loc[0]), "DAO row must not change"
    assert out.loc[1, "peak_max_adu"] == 65535.0 and bool(out.loc[1, "likely_saturated"])
    assert not bool(out.loc[1, "photometry_ok"])
    assert 3000.0 < out.loc[2, "peak_max_adu"] < 5000.0 and not bool(out.loc[2, "likely_saturated"])
    assert np.isnan(out.loc[1, "peak_dao"]), "no DAO peak for a forced row"


def test_no_forced_rows_is_a_no_op() -> None:
    df = _df().iloc[:1]
    out, meta = stamp_forced_rows_saturation(df, _frame(), fits.Header(), peak_sat_limit_adu=58981.5)
    assert meta["n_forced"] == 0 and out is df
