"""
Canonical Gaia DR3 source_id normalization utilities for VYVAR.
All catalog_id normalization must route through normalize_gaia_source_id()
or normalize_gaia_source_id_series() defined here.
"""

from __future__ import annotations

import logging
import math
import re
from typing import AbstractSet, Any

import numpy as np
import pandas as pd

# Gaia source_id has 19 digits: every id column of a VYVAR CSV is read as text (float64 keeps ~15-16 digits
# and one blank cell turns a whole int64 column into float64). One map for every reader (code review 2 D06).
VYVAR_CSV_DTYPE: dict[str, type] = {
    "catalog_id": str,
    "name": str,
    "target_catalog_id": str,
    "comp_catalog_id": str,
    "check_catalog_id": str,
    "source_id": str,
    "source_id_gaia": str,
    "vy_seed_catalog_id": str,
}

def read_vyvar_csv(path: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Canonical CSV reader for all VYVAR proc/catalog CSV files: every id column (``VYVAR_CSV_DTYPE``) as text.
    Other kwargs go to ``pd.read_csv``; a scalar ``dtype`` (e.g. ``str``) is kept as given, a dict is merged,
    and a column with a converter keeps its converter.
    """
    extra = kwargs.pop("dtype", None)
    if extra is not None and not isinstance(extra, dict):
        return pd.read_csv(path, dtype=extra, **kwargs)
    dtype = {**VYVAR_CSV_DTYPE, **(extra or {})}
    for col in kwargs.get("converters") or {}:
        dtype.pop(col, None)
    return pd.read_csv(path, dtype=dtype, **kwargs)


_ID_WARNED: set[str] = set()


def _refuse_id(val: Any, why: str) -> str:
    key = f"{type(val).__name__}:{why}"
    if key not in _ID_WARNED:
        _ID_WARNED.add(key)
        logging.warning("[GAIA ID] refused %r (%s) - Gaia ids are read and written as text", val, why)
    return ""


def normalize_gaia_source_id(val: Any) -> str:
    """Gaia source_id as its decimal text, or ``""`` (GAIA-ID-STRICT-2026-10, code review 2 G04-16).

    Accepted: ``int`` / numpy integer >= 0, ASCII digit strings (quotes and blanks stripped), and floats or
    float strings that hold an exact integer below 2**53. Everything else gives ``""``: floats of 19-digit ids
    (they lost their last digits), ``"4.6e+18"``, bools, bytes, non-ASCII digits (one WARNING per kind - a
    wrong id must never be fabricated) and names such as VSX names or DET_ labels (silently).
    """
    if type(val) is str and val.isascii() and val.isdigit():  # fast path: ids read as text
        return val
    if val is None or isinstance(val, (bool, np.bool_, bytes)):
        return "" if val is None else _refuse_id(val, "not an id type")
    if isinstance(val, (int, np.integer)):
        return str(int(val)) if int(val) >= 0 else _refuse_id(val, "negative")
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if not math.isfinite(f):
            return ""
        if f.is_integer() and 0 <= f < 2**53:
            return str(int(f))
        return _refuse_id(val, "float without the exact digits")
    if not isinstance(val, str):
        try:
            if pd.isna(val):
                return ""
        except (TypeError, ValueError):
            pass
        return _refuse_id(val, "not an id type")
    s = val.strip().strip('"').strip("'").strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    if s.isascii() and s.isdigit():
        return s
    m = re.fullmatch(r"([0-9]+)\.0*", s)
    if m and s.isascii() and int(m.group(1)) < 2**53:
        return str(int(m.group(1)))
    if re.fullmatch(r"[+-]?[0-9.]+([eE][+-]?[0-9]+)?", s):
        return _refuse_id(val, "number without the exact digits")
    return ""  # a name (VSX name, DET_ label), not an id


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
    """Identity key of a MASTERSTAR row: its ``catalog_id`` only (INV-MATCH-IDENTITY-01; ``name`` is a label)."""
    return normalize_gaia_source_id(row.get("catalog_id"))

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
    """Normalised Gaia id or ``""`` (same rule as ``normalize_gaia_source_id``)."""
    return normalize_gaia_source_id(x)
