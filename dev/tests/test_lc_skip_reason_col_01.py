# -*- coding: ascii -*-
"""D-LC-SKIP-MANIFEST-01 item 5: skip_reason column on science LC CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from photometry_core import TIME_BASE_BJD_TDB
from photometry_lightcurve import save_lightcurve_csv


def test_lc_csv_skip_reason_column(tmp_path: Path) -> None:
    n = 3
    path = tmp_path / "lightcurve_test.csv"
    save_lightcurve_csv(
        path,
        bjd=np.arange(n, dtype=float),
        hjd=np.arange(n, dtype=float),
        jd=np.arange(n, dtype=float),
        airmass=np.ones(n),
        is_flipped=None,
        mag_inst=np.zeros(n),
        mag_calib_raw=np.zeros(n),
        mag_calib=np.zeros(n),
        mag_calib_ct=None,
        mag_calib_ac=None,
        delta_mag=np.zeros(n),
        err=np.ones(n) * 0.01,
        aperture_r_px=np.full(n, 5.0),
        flags=[""] * n,
        source_files=[f"f{i}" for i in range(n)],
        method="aperture",
        time_base=TIME_BASE_BJD_TDB,
        skip_reason="",
    )
    df = pd.read_csv(path)
    assert "skip_reason" in df.columns
    got = df["skip_reason"].fillna("").astype(str).tolist()
    assert got == ["", "", ""]
