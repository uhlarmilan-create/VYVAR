"""Gaia DR3 proper-motion propagation (epoch 2016.0 -> observation epoch).

Moved from ``vyvar_platesolver`` (BLIND-SOLVE-2026-10 clean-up, Milan 2026-10-20). The local Gaia
DB is built without pmra/pmdec (GAIA-1), so rows normally carry no proper motion. Only the scalar
propagation used by vsx_gaia_crossmatch remains (G03-27: the row-list helper and the header-epoch
helper had no production caller).
"""
from __future__ import annotations

import math

GAIA_EPOCH = 2016.0


def apply_proper_motion(
    ra: float,
    dec: float,
    pmra: float | None,
    pmdec: float | None,
    obs_year: float,
) -> tuple[float, float]:
    """Propagate Gaia DR3 coordinates from epoch 2016.0 to observation year."""
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
    if not (math.isfinite(pmra_f) and math.isfinite(pmdec_f)):
        return ra_f, dec_f
    dt = float(obs_year) - float(GAIA_EPOCH)
    if not math.isfinite(dt) or abs(dt) < 1e-9:
        return ra_f, dec_f
    cos_dec = math.cos(math.radians(dec_f))
    if not math.isfinite(cos_dec) or abs(cos_dec) < 1e-6:
        return ra_f, dec_f
    delta_ra = (pmra_f / 3_600_000.0) * dt / cos_dec
    delta_dec = (pmdec_f / 3_600_000.0) * dt
    ra_corr = (ra_f + delta_ra) % 360.0
    dec_corr = max(-90.0, min(90.0, dec_f + delta_dec))
    return ra_corr, dec_corr
