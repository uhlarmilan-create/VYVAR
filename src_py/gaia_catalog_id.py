"""
Canonical Gaia DR3 source_id normalization utilities for VYVAR.
All catalog_id normalization must route through normalize_gaia_source_id()
or normalize_gaia_source_id_series() defined here.
"""

from __future__ import annotations

import logging
import math
import re
from decimal import Decimal, InvalidOperation
from typing import AbstractSet, Any

import numpy as np
import pandas as pd

# Float64 precision loss prevention: Gaia source_id has 19 digits.
# Reading as str prevents silent truncation to ~15 significant digits.
# See: TODO-fix (float64 catalog_id precision loss) resolved 17.5.2026
VYVAR_CSV_DTYPE: dict[str, type] = {
    "catalog_id": str,
    "name": str,
    "target_catalog_id": str,
    "comp_catalog_id": str,
}

# Per-frame ``proc_*.csv``: always read Gaia IDs as str (never float64 inference).
GAIA_PROC_CSV_READ_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}


def read_vyvar_csv(path: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Canonical CSV reader for all VYVAR proc/catalog CSV files.
    Always reads catalog_id and name as str to prevent float64 precision loss.
    Pass additional kwargs to pd.read_csv (e.g. usecols, nrows).
    """
    extra_dtype = kwargs.pop("dtype", {}) or {}
    dtype = {**VYVAR_CSV_DTYPE, **extra_dtype}
    return pd.read_csv(path, dtype=dtype, **kwargs)


def normalize_gaia_source_id(val) -> str:
    """Return the decimal ID string or ``\"\"``; unifies int, float, ``4.62e+17``, quotes."""
    # Safe: normalize_gaia_source_id handles NaN/empty/scientific notation;
    #        non-numeric strings returned as-is - callers must validate.
    # Fast path (CODE-REVIEW-2026-10): ids read as text are already in normal form (the
    # general path below returns them unchanged); this is the case of nearly every call.
    if type(val) is str and val.isascii() and val.isdigit():
        return val
    if val is None:
        return ""
    if isinstance(val, dict):
        for key in ("source_id", "catalog_id", "id", "SOURCE_ID", "SOURCE_ID_GAIA"):
            if key in val and val[key] is not None:
                nested = normalize_gaia_source_id(val[key])
                if nested:
                    return nested
        return ""
    if isinstance(val, (list, tuple)):
        for item in val:
            nested = normalize_gaia_source_id(item)
            if nested:
                return nested
        return ""
    if isinstance(val, float) and not math.isfinite(val):
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, float) and math.isfinite(val):
        if val.is_integer() and abs(val) < 2**53:
            return str(int(val))
        try:
            return str(int(Decimal(str(val))))
        except (InvalidOperation, ValueError, OverflowError):
            pass
    s = str(val).strip().strip('"').strip("'")
    if not s or s.lower() == "nan":
        return ""
    if re.fullmatch(r"-?\d+", s):
        return s
    if "e" in s.lower():
        try:
            return str(int(Decimal(s)))
        except (InvalidOperation, ValueError, OverflowError):
            pass
    try:
        fv = float(s)
        if math.isfinite(fv):
            if fv.is_integer() and abs(fv) < 2**53:
                return str(int(fv))
            return str(int(Decimal(s)))
    except (TypeError, ValueError, OverflowError, InvalidOperation):
        pass
    return s


def normalize_gaia_source_id_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_gaia_source_id).astype(object)


def normalize_gaia_id_set(
    values: AbstractSet[Any] | list[Any] | tuple[Any, ...] | None,
    *,
    log_label: str = "catalog_id",
) -> frozenset[str]:
    """Build a hashable Gaia ID set; drop non-scalars with a logged warning."""
    if not values:
        return frozenset()
    out: set[str] = set()
    for raw in values:
        if isinstance(raw, (dict, list, tuple)):
            logging.warning(
                "[GAIA ID] Dropping non-scalar %s entry in %s: %r",
                type(raw).__name__,
                log_label,
                raw,
            )
            continue
        gid = normalize_gaia_source_id(raw)
        if gid:
            out.add(gid)
    return frozenset(out)


def masterstar_row_gaia_key(row: pd.Series) -> str:
    """Key for the cone join: ``name`` first if it looks like a Gaia source_id (CSV often breaks ``catalog_id`` as float)."""
    name_k = normalize_gaia_source_id(row.get("name"))
    if name_k and re.fullmatch(r"\d{12,22}", name_k):
        return name_k
    cat_k = normalize_gaia_source_id(row.get("catalog_id"))
    if cat_k:
        return cat_k
    return name_k


def _catalog_id_empty_to_blank(s: pd.Series) -> pd.Series:
    """NaN / ``nan`` -> ``\"\"`` for writing to CSV (``na_rep=\"\"``)."""
    out = s.map(normalize_gaia_source_id).astype(object)
    return out.where(out.map(lambda x: bool(str(x).strip())), "")


def catalog_id_series_for_masterstars_export(df: pd.DataFrame) -> pd.Series:
    """Normalise an existing nonempty ``catalog_id`` for CSV; never copy ``name`` onto an empty ID.

    INV-MATCH-IDENTITY-01: identity lives in ``catalog_id``. DET_->catalog_id renaming
    is ``_proc_rename_det_names_to_catalog_id`` (catalog_id onto name), never the reverse.
    """
    if "catalog_id" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return _catalog_id_empty_to_blank(df["catalog_id"])


def norm_id_or_empty(x: Any) -> str:
    """Normalize a catalog_id-like value to Gaia decimal string, or \"\" if empty/nan/none."""
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return normalize_gaia_source_id(s)
    except Exception:  # noqa: BLE001
        try:
            return str(int(float(s)))
        except (ValueError, TypeError):
            return s
