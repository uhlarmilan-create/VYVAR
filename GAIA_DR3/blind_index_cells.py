"""Per-cell brightest-N capping for blind index (equal-area-ish sky grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_cell_bins(
    ra: np.ndarray,
    dec: np.ndarray,
    cell_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dec bands + RA bins scaled by cos(dec) (no extra dependencies)."""
    cell = float(cell_deg)
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError(f"cell_deg must be positive, got {cell_deg!r}")
    dec_bin = np.floor((dec + 90.0) / cell).astype(np.int64)
    cos_dec = np.cos(np.radians(dec))
    ra_width = cell / np.maximum(cos_dec, 0.1)
    ra_bin = np.floor(ra / ra_width).astype(np.int64)
    return dec_bin, ra_bin


def cap_brightest_per_cell(
    df: pd.DataFrame,
    *,
    cell_deg: float,
    stars_per_cell: int,
    mag_col: str = "g_mag",
) -> pd.DataFrame:
    """Keep brightest ``stars_per_cell`` rows per sky cell (smallest ``g_mag``)."""
    if df.empty:
        return df.copy()
    n = max(1, int(stars_per_cell))
    dec_bin, ra_bin = assign_cell_bins(
        df["ra"].to_numpy(dtype=np.float64),
        df["dec"].to_numpy(dtype=np.float64),
        cell_deg,
    )
    work = df.copy()
    work["_dec_bin"] = dec_bin
    work["_ra_bin"] = ra_bin
    capped = (
        work.sort_values(mag_col, ascending=True)
        .groupby(["_dec_bin", "_ra_bin"], sort=False)
        .head(n)
        .drop(columns=["_dec_bin", "_ra_bin"])
    )
    return capped.reset_index(drop=True)
