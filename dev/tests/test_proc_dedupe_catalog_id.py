"""DRAFT-514-TRIAGE A2: per-frame catalog_id must be unique after export path."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from pipeline import _proc_deduplicate_matched_catalog_rows  # noqa: E402


def test_proc_dedupe_keeps_one_row_per_catalog_id() -> None:
    df = pd.DataFrame(
        {
            "catalog_id": ["A", "A", "A", "B", ""],
            "peak_max_adu": [10.0, 30.0, 20.0, 5.0, 1.0],
            "x": [1.0, 1.0, 1.0, 2.0, 3.0],
            "y": [1.0, 1.0, 1.0, 2.0, 3.0],
            "flux": [10.0, 30.0, 20.0, 5.0, 1.0],
        }
    )
    out = _proc_deduplicate_matched_catalog_rows(df)
    matched = out[out["catalog_id"].astype(str).str.strip().ne("")]
    assert matched["catalog_id"].astype(str).value_counts().max() == 1
    # Highest peak kept for A
    a = matched[matched["catalog_id"] == "A"].iloc[0]
    assert float(a["peak_max_adu"]) == 30.0
