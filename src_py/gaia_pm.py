"""Gaia DR3 proper-motion propagation (epoch 2016.0 -> observation epoch).

Moved from ``vyvar_platesolver`` (BLIND-SOLVE-2026-10 clean-up, Milan 2026-10-20). The local Gaia
DB is built without pmra/pmdec (GAIA-1), so rows normally carry no proper motion and nothing is
corrected; the function logs that once instead of pretending success.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from astropy.io import fits
from astropy.time import Time

GAIA_EPOCH = 2016.0
#: proper motions below this (mas/yr, either axis) are not applied
PM_CORRECTION_MIN_MASYR = 10.0

_PM_COLUMNS_UNAVAILABLE_LOGGED = False


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


def obs_year_from_header(header: fits.Header | None) -> float:
    """Best-effort observation year from FITS DATE-OBS, fallback to the current UTC year."""
    date_obs = str((header or {}).get("DATE-OBS", "")).strip()
    if date_obs:
        try:
            return float(Time(date_obs, format="isot", scale="utc").decimalyear)
        except Exception:  # noqa: BLE001
            # EXC-0590: T4 -- DATE-OBS isot parse fail -> next format in ladder (EXCEPT-BULK 2026-07-08)
            pass
        try:
            return float(Time(date_obs, format="fits", scale="utc").decimalyear)
        except Exception:  # noqa: BLE001
            # EXC-0591: T4 -- DATE-OBS fits parse fail -> now()-year fallback (EXCEPT-BULK 2026-07-08)
            pass
    return float(datetime.now(timezone.utc).year)


def _row_has_finite_pm(row: dict[str, Any]) -> bool:
    try:
        pmra = row.get("pmra")
        pmdec = row.get("pmdec")
        if pmra is None or pmdec is None:
            return False
        return math.isfinite(float(pmra)) and math.isfinite(float(pmdec))
    except (TypeError, ValueError):
        return False


def apply_pm_to_gaia_rows(rows: list[dict[str, Any]], *, obs_year: float) -> tuple[list[dict[str, Any]], int]:
    """Apply the proper-motion correction to Gaia rows where it is significant."""
    global _PM_COLUMNS_UNAVAILABLE_LOGGED  # noqa: PLW0603
    if rows and not any(_row_has_finite_pm(row) for row in rows):
        if not _PM_COLUMNS_UNAVAILABLE_LOGGED:
            logging.warning(
                "[PM] Gaia catalog has no finite pmra/pmdec; proper-motion correction skipped "
                "(local DR3 build omits PM per GAIA-1 - n_corrected=0 is not evidence of success)"
            )
            _PM_COLUMNS_UNAVAILABLE_LOGGED = True
    out: list[dict[str, Any]] = []
    n_corrected = 0
    for row in rows:
        rr = dict(row)
        if not _row_has_finite_pm(rr):
            out.append(rr)
            continue
        try:
            pmra_f = float(rr["pmra"])
            pmdec_f = float(rr["pmdec"])
        except (TypeError, ValueError):
            out.append(rr)
            continue
        if abs(pmra_f) > PM_CORRECTION_MIN_MASYR or abs(pmdec_f) > PM_CORRECTION_MIN_MASYR:
            ra_corr, dec_corr = apply_proper_motion(
                rr.get("ra", float("nan")),
                rr.get("dec", float("nan")),
                rr.get("pmra"),
                rr.get("pmdec"),
                obs_year=float(obs_year),
            )
            rr["ra"] = float(ra_corr)
            rr["dec"] = float(dec_corr)
            n_corrected += 1
        out.append(rr)
    return out, n_corrected
