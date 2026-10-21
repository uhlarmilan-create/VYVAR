"""FORCED-PHOT-01: measure comparison-pool members every frame at MASTERSTAR XY.

Detection (DAO) is unchanged for discovery/QC. After DAO+match, any force-eligible
MASTERSTAR member missing from the frame is injected at the locked MASTERSTAR
grid position (aligned frames) and aperture-measured. Geometry out-of-footprint
yields no flux (recorded). Low SNR is kept. Per-frame saturation is flagged and
kept as a row; ensemble treatment is explicit (see ensemble_normalize).

INV-COMP-MEMBERSHIP becomes enforceable: membership is decided once; presence in
proc CSV is no longer conditional on DAO.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FORCED_SOURCE_TYPE = "GAIA_MATCHED"  # keep TODO-13 keep-filter; mark via forced_photometry col
FORCED_PHOT_COL = "forced_photometry"
GEOMETRY_OK_COL = "geometry_ok"
GEOMETRY_REASON_COL = "geometry_miss_reason"
# Bounded refine: search radius = ceil(fwhm * bound_fwhm); recorded in meta/result.


def force_eligible_masterstar_mask(master_tab: pd.DataFrame) -> pd.Series:
    """Stars eligible for forced measurement (three gates only).

    Measurability: not saturated on MASTERSTAR.
    Known variables ARE eligible (targets; FORCED-TARGET-2026-10).
    Geometry checked per frame (footprint).

    COMP-WEIGHT / COMP-ADMIT corrections:
    - ``is_noisy`` is NOT a gate (DAO peak significance; imprecise, not invalid).
    """
    if master_tab is None or getattr(master_tab, "empty", True):
        return pd.Series(dtype=bool)
    n = len(master_tab)
    idx = master_tab.index

    def _b(col: str, default: bool = False) -> pd.Series:
        if col not in master_tab.columns:
            return pd.Series([default] * n, index=idx)
        return master_tab[col].fillna(default).astype(bool)

    zone = (
        master_tab["zone"].astype(str).str.strip().str.lower()
        if "zone" in master_tab.columns
        else pd.Series([""] * n, index=idx)
    )
    sat = _b("is_saturated") | _b("likely_saturated") | zone.eq("saturated")
    # FORCED-TARGET-2026-10 (Milan 2026-10-11): known variables (VSX / Gaia variable / NSS)
    # are the photometry TARGETS - they are measured in every frame at the MASTERSTAR position
    # like the comparison pool (AIJ / IRAF phot with a coordinate list). Before, they were
    # measured only when DAO detected and matched them (draft 526: 53 of 81 targets > 50 %
    # no_data). Comparison selection excludes variables on its own (comp_selection_per_target).
    # is_noisy intentionally omitted (COMP-ADMIT-03 review correction).
    ok = ~sat
    cid = master_tab.get("catalog_id", master_tab.get("name", pd.Series([""] * n, index=idx)))
    cid_s = cid.astype(str).str.strip()
    ok &= cid_s.ne("") & ~cid_s.str.lower().isin({"nan", "none"})
    if "x" in master_tab.columns and "y" in master_tab.columns:
        ok &= pd.to_numeric(master_tab["x"], errors="coerce").notna()
        ok &= pd.to_numeric(master_tab["y"], errors="coerce").notna()
    return ok


def _in_footprint(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    margin_px: float,
) -> tuple[bool, str]:
    if not (math.isfinite(x) and math.isfinite(y)):
        return False, "nonfinite_xy"
    m = float(max(0.0, margin_px))
    if x < m or y < m or x >= (float(width) - m) or y >= (float(height) - m):
        return False, "outside_aligned_footprint"
    return True, ""



def inject_forced_masterstar_rows(
    df: pd.DataFrame,
    master_tab: pd.DataFrame,
    *,
    image: np.ndarray | None = None,
    margin_px: float = 0.0,
    force_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inject missing force-eligible MASTERSTAR members into ``df``.

    Positions are MASTERSTAR ``(x,y)`` (aligned grid; no refine, POS-SUBPIX-2026-10).
    ``image`` only gives the frame footprint. Geometry misses get a row with
    ``geometry_ok=False`` and no finite flux coordinates for measurement skip,
    or are omitted from measurement while still recorded.

    Returns (new_df, meta).
    """
    meta: dict[str, Any] = {
        "n_injected": 0,
        "n_geometry_miss": 0,
        "n_already_present": 0,
    }
    if master_tab is None or getattr(master_tab, "empty", True):
        return df if df is not None else pd.DataFrame(), meta

    eligible = master_tab.loc[force_eligible_masterstar_mask(master_tab)].copy()
    if force_ids is not None:
        cid_e = eligible["catalog_id"].astype(str).str.strip()
        eligible = eligible.loc[cid_e.isin({str(x).strip() for x in force_ids})].copy()
    if eligible.empty:
        return df if df is not None else pd.DataFrame(), meta

    out = df.copy() if df is not None and len(df) else pd.DataFrame()
    if FORCED_PHOT_COL not in out.columns and len(out):
        out[FORCED_PHOT_COL] = False
    if GEOMETRY_OK_COL not in out.columns and len(out):
        out[GEOMETRY_OK_COL] = True
    if GEOMETRY_REASON_COL not in out.columns and len(out):
        out[GEOMETRY_REASON_COL] = ""

    present: set[str] = set()
    if len(out) and "catalog_id" in out.columns:
        present = {
            str(x).strip()
            for x in out["catalog_id"].tolist()
            if str(x).strip() and str(x).strip().lower() not in {"nan", "none"}
        }
    meta["n_already_present"] = int(
        sum(1 for c in eligible["catalog_id"].astype(str).str.strip() if c in present)
    )

    h_img = w_img = 0
    img = None
    if image is not None:
        img = np.asarray(image, dtype=np.float64)
        if img.ndim == 2 and img.size:
            h_img, w_img = int(img.shape[0]), int(img.shape[1])

    # Infer frame size from master extent if image missing.
    if (h_img <= 0 or w_img <= 0) and "x" in master_tab.columns and "y" in master_tab.columns:
        mx = pd.to_numeric(master_tab["x"], errors="coerce")
        my = pd.to_numeric(master_tab["y"], errors="coerce")
        if mx.notna().any() and my.notna().any():
            w_img = int(math.ceil(float(mx.max()) + 1))
            h_img = int(math.ceil(float(my.max()) + 1))

    new_rows: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        cid = str(row.get("catalog_id", "")).strip()
        if not cid or cid in present:
            continue
        x_ref = float(pd.to_numeric(row.get("x"), errors="coerce"))
        y_ref = float(pd.to_numeric(row.get("y"), errors="coerce"))
        geo_ok, geo_reason = _in_footprint(
            x_ref, y_ref, width=max(w_img, 1), height=max(h_img, 1), margin_px=margin_px
        )
        if not geo_ok:
            meta["n_geometry_miss"] = int(meta["n_geometry_miss"]) + 1
            # Record miss as a stub row (no usable xy for aperture -> NaN flux).
            stub = {
                "catalog_id": cid,
                "name": cid,
                "x": float("nan"),
                "y": float("nan"),
                "ra_deg": row.get("ra_deg", row.get("ra")),
                "dec_deg": row.get("dec_deg", row.get("dec")),
                "source_type": FORCED_SOURCE_TYPE,
                FORCED_PHOT_COL: True,
                GEOMETRY_OK_COL: False,
                GEOMETRY_REASON_COL: geo_reason,
                "flux": float("nan"),
                "dao_flux": float("nan"),
            }
            for col in (
                "zone",
                "is_saturated",
                "is_usable",
                "bp_rp",
                "phot_g_mean_mag",
                "catalog_mag",
                "vsx_known_variable",
            ):
                if col in row.index:
                    stub[col] = row.get(col)
            new_rows.append(stub)
            continue

        # POS-SUBPIX-2026-10: registered frame -> the MASTERSTAR sub-pixel position is the centre
        x_use, y_use = x_ref, y_ref

        nr: dict[str, Any] = {
            "catalog_id": cid,
            "name": cid,
            "x": float(x_use),
            "y": float(y_use),
            "ra_deg": row.get("ra_deg", row.get("ra")),
            "dec_deg": row.get("dec_deg", row.get("dec")),
            "source_type": FORCED_SOURCE_TYPE,
            FORCED_PHOT_COL: True,
            GEOMETRY_OK_COL: True,
            GEOMETRY_REASON_COL: "",
        }
        for col in (
            "zone",
            "is_saturated",
            "is_usable",
            "bp_rp",
            "phot_g_mean_mag",
            "catalog_mag",
            "edge_safe_10px",
            "snr50_ok",
            "vsx_known_variable",
            "non_single_star",
        ):
            if col in row.index:
                nr[col] = row.get(col)
        new_rows.append(nr)
        meta["n_injected"] = int(meta["n_injected"]) + 1

    if not new_rows:
        return out, meta

    add = pd.DataFrame(new_rows)
    if len(out) == 0:
        return add.reset_index(drop=True), meta
    # Align to union of columns without empty-NA concat warning.
    all_cols = list(dict.fromkeys(list(out.columns) + list(add.columns)))
    for c in all_cols:
        if c not in out.columns:
            if c == FORCED_PHOT_COL:
                out[c] = False
            elif c == GEOMETRY_OK_COL:
                out[c] = True
            elif c == GEOMETRY_REASON_COL:
                out[c] = ""
            else:
                out[c] = np.nan
        if c not in add.columns:
            if c == FORCED_PHOT_COL:
                add[c] = True
            elif c == GEOMETRY_OK_COL:
                add[c] = True
            elif c == GEOMETRY_REASON_COL:
                add[c] = ""
            else:
                add[c] = np.nan
    if FORCED_PHOT_COL in out.columns:
        out[FORCED_PHOT_COL] = out[FORCED_PHOT_COL].fillna(False).astype(bool)
    return pd.concat([out[all_cols], add[all_cols]], ignore_index=True), meta
