"""File-backed enrichment columns for masterstars_full_match.csv (MS-SOURCES-RETIRE)."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Columns the comparison-star selection reads (CODE-REVIEW-2026-10 step 6: is_safe_comp, exclusion_reason,
# stress_rms, phot_category, recommended_aperture had no reader and are no longer written).
ENRICHMENT_COLUMNS: tuple[str, ...] = (
    "likely_nonlinear",
    "non_single_star",
    "phot_variable_flag",
    "g_flux_error_rel",
)


def normalize_catalog_id(raw: Any) -> str:
    return str(raw or "").strip()


def row_ms_to_enrichment_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "likely_nonlinear": int(row.get("likely_nonlinear") or 0),
        "non_single_star": int(row.get("non_single_star") or 0),
        "phot_variable_flag": str(row.get("phot_variable_flag") or "") or "",
        "g_flux_error_rel": row.get("g_flux_error_rel"),
    }


def enrichment_by_catalog_id(rows_ms: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows_ms:
        cid = normalize_catalog_id(row.get("source_id_gaia"))
        if not cid:
            continue
        out[cid] = row_ms_to_enrichment_dict(row)
    return out


def merge_enrichment_into_masterstars_df(
    df: pd.DataFrame,
    rows_ms: list[dict[str, Any]],
) -> pd.DataFrame:
    """Attach MAKE MASTERSTAR enrichment columns to the full masterstars CSV frame."""
    out = df.copy()
    by_cid = enrichment_by_catalog_id(rows_ms)
    cids = out.get("catalog_id", pd.Series([""] * len(out))).map(normalize_catalog_id)

    for col in ENRICHMENT_COLUMNS:
        default: Any = 0 if col in ("likely_nonlinear", "non_single_star") else ""
        if col == "g_flux_error_rel":
            default = float("nan")
        values = []
        for cid in cids:
            ent = by_cid.get(cid)
            if ent is None:
                values.append(default)
            else:
                values.append(ent.get(col, default))
        out[col] = values
    return out


def missing_comp_selection_enrichment_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ("likely_nonlinear",) if c not in df.columns]
