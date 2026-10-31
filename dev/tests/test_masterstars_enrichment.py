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
            "likely_nonlinear": 1,
            "non_single_star": 0,
            "g_flux_error_rel": 0.002,
        }
    ]
    out = merge_enrichment_into_masterstars_df(df, rows_ms)
    for col in ENRICHMENT_COLUMNS:
        assert col in out.columns
    assert list(ENRICHMENT_COLUMNS) == ["likely_nonlinear", "non_single_star", "g_flux_error_rel"]
    assert int(out.loc[out["catalog_id"] == "GaiaDR3_1", "likely_nonlinear"].iloc[0]) == 1
    assert int(out.loc[out["catalog_id"] == "GaiaDR3_2", "likely_nonlinear"].iloc[0]) == 0


def test_missing_enrichment_columns_detected() -> None:
    df = pd.DataFrame({"catalog_id": ["a"], "x": [1.0]})
    missing = missing_comp_selection_enrichment_columns(df)
    assert "likely_nonlinear" in missing


def test_no_fetch_master_sources_in_plan_or_ui() -> None:
    root = Path(__file__).resolve().parents[2] / "src_py"
    for rel in ("pipeline.py",):
        src = (root / rel).read_text(encoding="utf-8")
        assert "fetch_master_sources_for_draft" not in src


def test_enrichment_by_identity_not_by_neighbour() -> None:
    """G04-09: a saturated identified star keeps likely_nonlinear=1 next to a DAO_ONLY row; NSS by catalog_id."""
    from masterstars_enrichment import enrich_masterstars_by_identity

    df = pd.DataFrame({"catalog_id": ["111", "", "222"], "peak_max_adu": [60000.0, 900.0, 59000.0]})
    cone = pd.DataFrame({"catalog_id": ["111", "222"], "non_single_star": [1, 0], "g_flux_error_rel": [0.001, 0.002]})
    out = enrich_masterstars_by_identity(df, cone, sat_thr=58981.5)
    assert list(out["likely_nonlinear"]) == [1, 0, 1]
    assert list(out["non_single_star"]) == [1, 0, 0]
    assert out["g_flux_error_rel"].iloc[0] == 0.001 and pd.isna(out["g_flux_error_rel"].iloc[1])
