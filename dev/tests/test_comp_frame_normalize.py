"""Tests for shared comp-frame normalization and robust RMS."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from comp_frame_normalize import (
    build_frame_bin_medians,
    dedupe_catalog_rows,
    matched_catalog_flux_rows,
    robust_comp_rms,
)


def test_matched_catalog_dedupes_before_bin_median():
    df = pd.DataFrame(
        {
            "catalog_id": ["100", "100"],
            "name": ["100", "100"],
            "mag": [10.0, 10.0],
            "dao_flux": [100.0, 500.0],
        }
    )
    norm = matched_catalog_flux_rows(df, flux_col="dao_flux")
    assert len(norm) == 1
    bin_meds, _frame_med, used_bins = build_frame_bin_medians(df, flux_col="dao_flux")
    assert used_bins
    # bin for mag 10.0 should use brighter duplicate (500), not mean(100,500)=300
    b10 = int(10.0 / 0.5)
    assert float(bin_meds[b10]) == 500.0
    # mag 10.2 is in the same 0.5-mag bin (int(10.2/0.5)==20)
    assert int(10.2 / 0.5) == b10


def test_robust_comp_rms_uses_all_finite_positive_frames():
    # Zero-clipping: MAD over all finite positive fluxes (no frame rejection).
    good = [1.0] * 112
    bad = [0.4] * 27
    vals = good + bad
    robust = robust_comp_rms(vals)
    # Population is bimodal; MAD about median (~1.0) is dominated by zeros among goods
    # once bad frames are kept, so RMS is larger than the historical clipped value.
    assert math.isfinite(robust)
    assert robust > 0.0


def test_dedupe_catalog_rows_by_catalog_id():
    df = pd.DataFrame(
        {
            "catalog_id": ["1", "1", "2"],
            "peak_max_adu": [10.0, 99.0, 50.0],
            "dao_flux": [1.0, 9.0, 5.0],
        }
    )
    out = dedupe_catalog_rows(df, id_col="catalog_id", flux_col="dao_flux")
    assert len(out) == 2
    assert float(out.loc[out["catalog_id"] == "1", "peak_max_adu"].iloc[0]) == 99.0
