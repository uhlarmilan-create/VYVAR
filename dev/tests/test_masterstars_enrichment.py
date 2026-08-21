"""Tests for masterstars CSV enrichment (MS-SOURCES-RETIRE C2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from masterstars_enrichment import (
    ENRICHMENT_COLUMNS,
    merge_enrichment_into_masterstars_df,
    missing_comp_selection_enrichment_columns,
)


def test_merge_enrichment_columns_present() -> None:
    df = pd.DataFrame(
        {
            "catalog_id": ["GaiaDR3_1", "GaiaDR3_2"],
            "x": [1.0, 2.0],
        }
    )
    rows_ms = [
        {
            "source_id_gaia": "GaiaDR3_1",
            "is_safe_comp": 1,
            "exclusion_reason": "",
            "stress_rms": 0.01,
            "phot_category": "Clear_mag_12.0_col_1.00",
            "likely_nonlinear": 0,
            "on_bad_column": 0,
            "recommended_aperture": 4.5,
            "non_single_star": 0,
            "phot_variable_flag": "",
            "g_flux_error_rel": 0.002,
        }
    ]
    out = merge_enrichment_into_masterstars_df(df, rows_ms)
    for col in ENRICHMENT_COLUMNS:
        assert col in out.columns
    assert int(out.loc[out["catalog_id"] == "GaiaDR3_1", "is_safe_comp"].iloc[0]) == 1
    assert int(out.loc[out["catalog_id"] == "GaiaDR3_2", "is_safe_comp"].iloc[0]) == 0


def test_missing_enrichment_columns_detected() -> None:
    df = pd.DataFrame({"catalog_id": ["a"], "x": [1.0]})
    missing = missing_comp_selection_enrichment_columns(df)
    assert "likely_nonlinear" in missing
    assert "on_bad_column" in missing


def test_no_fetch_master_sources_in_plan_or_ui() -> None:
    root = Path(__file__).resolve().parents[2] / "src_py"
    for rel in ("pipeline.py", "ui_components.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "fetch_master_sources_for_draft" not in src
