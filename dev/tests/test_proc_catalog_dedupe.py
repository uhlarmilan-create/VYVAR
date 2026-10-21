"""Tests for per-frame catalog_id dedupe at export."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from pipeline import _proc_deduplicate_matched_catalog_rows


def test_proc_deduplicate_matched_catalog_rows_keeps_brightest_peak():
    df = pd.DataFrame(
        {
            "catalog_id": ["100", "100", "200", ""],
            "name": ["100", "100", "200", "DET_0001"],
            "peak_max_adu": [100.0, 5000.0, 800.0, 50.0],
            "dao_flux": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = _proc_deduplicate_matched_catalog_rows(df)
    assert len(out) == 3
    row100 = out[out["catalog_id"] == "100"].iloc[0]
    assert float(row100["peak_max_adu"]) == 5000.0


def test_proc_deduplicate_fallback_without_peak_or_dao_flux():
    """IMPL-05 A: one-row frame lacking peak_max_adu and dao_flux must not crash.

    Draft 515 hit this when DAO returned an empty table and forced-phot inject
    supplied catalog rows without saturation/flux columns; out.get('flux') was
    None and pd.to_numeric(None) became numpy.float64 (no .fillna).
    """
    df = pd.DataFrame(
        {
            "catalog_id": ["1497145751650265600"],
            "name": ["1497145751650265600"],
            "x": [100.0],
            "y": [200.0],
        }
    )
    assert "peak_max_adu" not in df.columns
    assert "dao_flux" not in df.columns
    assert "flux" not in df.columns
    out = _proc_deduplicate_matched_catalog_rows(df)
    assert len(out) == 1
    assert str(out.iloc[0]["catalog_id"]) == "1497145751650265600"
