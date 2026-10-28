"""Guarded PSF sidecar writes: a PSF column merge may change psf_* columns only (EXPORT-PARITY).

LAYER-2026-10: shared by the ePSF merge and the chi2 locus finaliser, so neither imports the other.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from gaia_catalog_id import read_vyvar_csv


# --- moved from proc_frame_store (LAYER-2026-10) ---


def non_psf_columns_equal(before: pd.DataFrame, after: pd.DataFrame, *, rtol: float = 1e-9) -> bool:
    """True when the non-psf columns of two tables hold the same values.

    Numeric values compare with a relative tolerance (``rtol``) because the CSV writer keeps ~11
    significant digits: an exact digit fingerprint of a re-read file differs whenever a value sits
    on a rounding tie (533 g: flux_k125 2.5538216505083e-05 -> 2.5538216505e-05 failed a .10g
    fingerprint). Columns whose entries are all numbers on both sides compare numerically, whatever
    the dtype; the others (names, flags) and integer IDs longer than 15 digits (Gaia source_id, where
    a tolerance would hide a changed ID) compare as exact text; NaN / empty equal.
    """
    cols = non_psf_columns(before.columns)
    if cols != non_psf_columns(after.columns) or len(before) != len(after):
        return False
    for c in cols:
        ta, tb = _as_text(before[c]), _as_text(after[c])
        if ta == tb:
            continue
        if _looks_like_ids(ta) or _looks_like_ids(tb):
            return False
        va, vb = _as_numbers(before[c]), _as_numbers(after[c])
        if va is not None and vb is not None:
            if not bool(np.isclose(va, vb, rtol=rtol, atol=0.0, equal_nan=True).all()):
                return False
        else:
            return False
    return True


def guarded_psf_sidecar_write(
    sidecar_path: Path,
    after: pd.DataFrame,
    before: pd.DataFrame,
) -> None:
    """Write sidecar; INV-EXPORT-READ-ONLY-01 restores if non-psf columns change."""
    from proc_frame_store import _vyvar_df_to_csv

    pre_bytes = sidecar_path.read_bytes() if sidecar_path.is_file() else None
    try:
        _vyvar_df_to_csv(after, sidecar_path)
        on_disk = read_vyvar_csv(sidecar_path, low_memory=False)
        if not non_psf_columns_equal(before, on_disk):
            raise RuntimeError(
                "INV-EXPORT-READ-ONLY-01: non-psf columns changed; sidecar restored"
            )
    except Exception:
        if pre_bytes is not None:
            sidecar_path.write_bytes(pre_bytes)
        raise


# --- moved from epsf_psf_merge (LAYER-2026-10) ---


# Fit coordinates are PSF-only (EPSF-SHAPE-01-F F3); treated as PSF columns
# so INV-PSF-ADDITIVE-01 non-PSF byte-identity still holds.
_PSF_FIT_COORD_COLS = frozenset({"x_fit", "y_fit"})


def is_psf_column(name: str) -> bool:
    n = str(name)
    return n.startswith("psf_") or n in _PSF_FIT_COORD_COLS


def non_psf_columns(columns) -> list[str]:
    return [c for c in columns if not is_psf_column(c)]


def _as_numbers(s: pd.Series) -> np.ndarray | None:
    """Float values of a column whose every non-empty entry is a number (object columns from a
    pyarrow round-trip included), else None."""
    num = pd.to_numeric(s, errors="coerce")
    empty = s.isna() | (s.astype(str).str.strip() == "")
    if bool((num.isna() & ~empty).any()):
        return None
    return num.to_numpy(dtype=float)


def _looks_like_ids(text: list[str]) -> bool:
    """Integers longer than a float mantissa (Gaia source_id): exact text only."""
    return any(len(t.lstrip("-")) > 15 and t.lstrip("-").isdigit() for t in text)


def _as_text(s: pd.Series) -> list[str]:
    return ["" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in s.tolist()]
