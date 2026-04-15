"""Gaia DR3 ``source_id`` ako reťazec pre join a CSV (float64 / vedecká notácia v CSV kazí zhodu)."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd


def normalize_gaia_source_id(val) -> str:
    """Vráti desiatkový reťazec ID alebo ``\"\"``; zjednotí int, float, ``4.62e+17``, úvodzovky."""
    if val is None:
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
        return str(int(round(val)))
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
            return str(int(round(fv)))
    except (TypeError, ValueError, OverflowError):
        pass
    return s


def normalize_gaia_source_id_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_gaia_source_id).astype(object)


def masterstar_row_gaia_key(row: pd.Series) -> str:
    """Kľúč pre join s kuželom: najprv ``name`` ak vyzerá ako Gaia source_id (CSV často pokazí ``catalog_id`` floatom)."""
    name_k = normalize_gaia_source_id(row.get("name"))
    if name_k and re.fullmatch(r"\d{12,22}", name_k):
        return name_k
    cat_k = normalize_gaia_source_id(row.get("catalog_id"))
    if cat_k:
        return cat_k
    return name_k


def catalog_id_series_for_masterstars_export(df: pd.DataFrame) -> pd.Series:
    """Stĺpec ``catalog_id`` do CSV ako desiatkový reťazec; pri platnom číselnom ``name`` berie ID odtiaľ."""
    cid = normalize_gaia_source_id_series(df["catalog_id"])
    if "name" not in df.columns:
        return cid
    nk = df["name"].map(normalize_gaia_source_id)
    mask = nk.map(lambda x: bool(x and re.fullmatch(r"\d{12,22}", x)))
    out = cid.copy()
    out.loc[mask] = nk.loc[mask].astype(object)
    return out
