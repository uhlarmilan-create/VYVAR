"""Gaia DR3 proper-motion propagation (epoch 2016.0 -> observation epoch).

Moved from ``vyvar_platesolver`` (BLIND-SOLVE-2026-10 clean-up, Milan 2026-10-20). The local Gaia
DB is built without pmra/pmdec (GAIA-1), so rows normally carry no proper motion. Used by
vsx_gaia_crossmatch (G03-27: the row-list helper and the header-epoch helper had no production
caller). G05-56: one array implementation; the scalar form is a wrapper of it.
"""
from __future__ import annotations

import math

import numpy as np

GAIA_EPOCH = 2016.0


def apply_proper_motion_arrays(
    ra: np.ndarray,
    dec: np.ndarray,
    pmra: np.ndarray,
    pmdec: np.ndarray,
    obs_year: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate Gaia DR3 coordinates (deg; pm in mas/yr) from epoch 2016.0 to ``obs_year``.

    Rows with a non-finite position or proper motion, or within 1e-6 of a pole (cos dec), keep
    their position. ``cos dec`` uses ``math.cos(math.radians(dec))`` per row (numpy's vectorised
    cos may differ in the last bit on some CPUs); the rest is elementwise IEEE arithmetic in the
    order of the scalar formula, and ``np.remainder`` equals Python's float ``%``."""
    ra_a = np.asarray(ra, dtype=np.float64)
    dec_a = np.asarray(dec, dtype=np.float64)
    pmra_a = np.asarray(pmra, dtype=np.float64)
    pmdec_a = np.asarray(pmdec, dtype=np.float64)
    ra_out = ra_a.copy()
    dec_out = dec_a.copy()
    dt = float(obs_year) - float(GAIA_EPOCH)
    if not math.isfinite(dt) or abs(dt) < 1e-9:
        return ra_out, dec_out
    idx = np.flatnonzero(np.isfinite(ra_a) & np.isfinite(dec_a) & np.isfinite(pmra_a) & np.isfinite(pmdec_a))
    cos_dec = np.asarray([math.cos(math.radians(d)) for d in dec_a[idx].tolist()], dtype=np.float64)
    keep = np.isfinite(cos_dec) & (np.abs(cos_dec) >= 1e-6)
    idx, cos_dec = idx[keep], cos_dec[keep]
    delta_ra = (pmra_a[idx] / 3_600_000.0) * dt / cos_dec
    delta_dec = (pmdec_a[idx] / 3_600_000.0) * dt
    ra_out[idx] = np.remainder(ra_a[idx] + delta_ra, 360.0)
    dec_out[idx] = np.clip(dec_a[idx] + delta_dec, -90.0, 90.0)
    return ra_out, dec_out


def apply_proper_motion(
    ra: float,
    dec: float,
    pmra: float | None,
    pmdec: float | None,
    obs_year: float,
) -> tuple[float, float]:
    """Propagate Gaia DR3 coordinates from epoch 2016.0 to observation year (one star)."""
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except (TypeError, ValueError):
        return ra, dec
    if not (math.isfinite(ra_f) and math.isfinite(dec_f)):
        return ra, dec
    try:
        pmra_f = float(pmra) if pmra is not None else float("nan")
        pmdec_f = float(pmdec) if pmdec is not None else float("nan")
    except (TypeError, ValueError):
        return ra_f, dec_f
    r, d = apply_proper_motion_arrays(
        np.array([ra_f]), np.array([dec_f]), np.array([pmra_f]), np.array([pmdec_f]), obs_year
    )
    return float(r[0]), float(d[0])
