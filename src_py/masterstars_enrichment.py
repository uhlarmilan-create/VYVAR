"""File-backed enrichment columns for masterstars_full_match.csv (MS-SOURCES-RETIRE)."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Column names match lowercase CSV convention; values mirror former MASTER_SOURCES rows.
ENRICHMENT_COLUMNS: tuple[str, ...] = (
    "is_safe_comp",
    "exclusion_reason",
    "stress_rms",
    "phot_category",
    "likely_nonlinear",
    "on_bad_column",
    "recommended_aperture",
    "non_single_star",
    "phot_variable_flag",
    "g_flux_error_rel",
)

_GRID_QA_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "G_MAG": ("phot_g_mean_mag", "catalog_mag", "mag", "implied_g_mag"),
    "BP_RP": ("bp_rp",),
    "STRESS_RMS": ("stress_rms",),
    "IS_SAFE_COMP": ("is_safe_comp",),
    "PHOT_CATEGORY": ("phot_category",),
    "LIKELY_NONLINEAR": ("likely_nonlinear",),
    "ON_BAD_COLUMN": ("on_bad_column",),
}


def normalize_catalog_id(raw: Any) -> str:
    return str(raw or "").strip()


def row_ms_to_enrichment_dict(row: dict[str, Any]) -> dict[str, Any]:
    excl = row.get("exclusion_reason")
    return {
        "is_safe_comp": int(row.get("is_safe_comp") or 0),
        "exclusion_reason": (str(excl).strip() if excl is not None else "") or "",
        "stress_rms": row.get("stress_rms"),
        "phot_category": str(row.get("phot_category") or "") or "",
        "likely_nonlinear": int(row.get("likely_nonlinear") or 0),
        "on_bad_column": int(row.get("on_bad_column") or 0),
        "recommended_aperture": row.get("recommended_aperture"),
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
        default: Any = 0 if col in ("is_safe_comp", "likely_nonlinear", "on_bad_column", "non_single_star") else ""
        if col in ("stress_rms", "recommended_aperture", "g_flux_error_rel"):
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


def apply_common_field_bbox_exclusion(
    rows_ms: list[dict[str, Any]],
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> int:
    """Mark safe comps outside the common-field bbox excluded (mirrors former DB UPDATE)."""
    n = 0
    for row in rows_ms:
        if int(row.get("safe_override") or 0) == 1:
            continue
        if int(row.get("is_safe_comp") or 0) != 1:
            continue
        xm = row.get("x_master")
        ym = row.get("y_master")
        if xm is None or ym is None:
            row["is_safe_comp"] = 0
            row["exclusion_reason"] = "Out of common field"
            n += 1
            continue
        try:
            xf = float(xm)
            yf = float(ym)
        except (TypeError, ValueError):
            row["is_safe_comp"] = 0
            row["exclusion_reason"] = "Out of common field"
            n += 1
            continue
        if xf < x0 or xf > x1 or yf < y0 or yf > y1:
            row["is_safe_comp"] = 0
            row["exclusion_reason"] = "Out of common field"
            n += 1
    return n


def apply_stress_rms_to_rows_ms(
    rows_ms: list[dict[str, Any]],
    per_source_rms: dict[str, float],
    med_by_bin: dict[str, float],
) -> None:
    for row in rows_ms:
        sid = normalize_catalog_id(row.get("source_id_gaia"))
        if sid and sid in per_source_rms:
            row["stress_rms"] = float(per_source_rms[sid])
    for row in rows_ms:
        if int(row.get("safe_override") or 0) == 1:
            continue
        if int(row.get("is_safe_comp") or 0) != 1:
            continue
        b = str(row.get("phot_category") or "").strip()
        sid = normalize_catalog_id(row.get("source_id_gaia"))
        if not b or b not in med_by_bin or not sid or sid not in per_source_rms:
            continue
        if float(per_source_rms[sid]) > 1.5 * float(med_by_bin[b]):
            row["is_safe_comp"] = 0
            row["exclusion_reason"] = "Unstable"


def apply_vsx_variable_flags(rows_ms: list[dict[str, Any]], var_ids: set[str]) -> int:
    n = 0
    for row in rows_ms:
        if int(row.get("safe_override") or 0) == 1:
            continue
        sid = normalize_catalog_id(row.get("source_id_gaia"))
        if sid and sid in var_ids:
            row["is_var"] = 1
            row["is_safe_comp"] = 0
            row["exclusion_reason"] = "Variable"
            n += 1
    return n


def grid_qa_dataframe_from_masterstars_csv(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalize draft CSV columns for Photometric Grid QA heatmap."""
    if df.empty:
        return None
    out = df.copy()
    for target, sources in _GRID_QA_COLUMN_ALIASES.items():
        if target in out.columns:
            continue
        for src in sources:
            if src in out.columns:
                out[target] = out[src]
                break
    needed = ("G_MAG", "BP_RP", "IS_SAFE_COMP", "PHOT_CATEGORY")
    if not all(c in out.columns for c in needed):
        return None
    return out


def missing_comp_selection_enrichment_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ("likely_nonlinear", "on_bad_column") if c not in df.columns]
