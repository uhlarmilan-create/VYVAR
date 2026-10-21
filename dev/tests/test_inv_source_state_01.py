# -*- coding: ascii -*-
"""INV-SOURCE-STATE-01: DETECTED_Pn requires this row's own detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import AppConfig
from masterstar_gaia_accounting import (
    SOURCE_CATALOG_MEMBERSHIP,
    SOURCE_DETECTED_P1,
    enrich_masterstar_gaia_complete,
)


def test_inject_row_not_labelled_detected_when_vy_dao_pass_column_exists() -> None:
    """H-LABEL: column presence must not make a catalog inject DETECTED_*."""
    h, w = 80, 80
    data0 = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    data0 += (80.0 * np.exp(-((xx - 20.0) ** 2 + (yy - 20.0) ** 2) / (2.0 * 1.6 ** 2))).astype(
        np.float32
    )
    df = pd.DataFrame(
        {
            "x": [20.0, 50.0],
            "y": [20.0, 50.0],
            "catalog_id": ["1001", "1002"],
            "peak_dao": [80.0, np.nan],
            "vy_dao_pass": [1, np.nan],
            "vy_match_mode": ["", "catalog_membership"],
            "forced_photometry": [False, False],
            "source_state": ["", ""],
        }
    )
    gaia = pd.DataFrame(
        {
            "catalog_id": ["1001", "1002"],
            "x_gaia": [20.0, 50.0],
            "y_gaia": [20.0, 50.0],
            "g_mag": [12.0, 12.0],
        }
    )
    out, _census, _meta = enrich_masterstar_gaia_complete(
        df,
        data0=data0,
        gaia_on_chip=gaia,
        cfg=AppConfig(),
        wpx=w,
        h=h,
        fwhm_px=3.5,
        target_depth_g=15.0,
        catalog_derived_membership=True,
    )
    inj = out.loc[out["catalog_id"].astype(str) == "1002"].iloc[0]
    det = out.loc[out["catalog_id"].astype(str) == "1001"].iloc[0]
    assert str(det["source_state"]) == SOURCE_DETECTED_P1
    assert str(inj["source_state"]) not in {SOURCE_DETECTED_P1, "DETECTED_P2"}
    assert str(inj["source_state"]) == SOURCE_CATALOG_MEMBERSHIP
    assert str(inj["vy_match_mode"]) == "catalog_membership"
