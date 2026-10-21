"""
lunar_context.py - Lunar observing conditions for a VYVAR session.

Uses astropy ephemeris (built-in, no internet required).
All inputs/outputs are plain Python types (float, str, dict) -
no astropy objects leak outside this module.
"""

from __future__ import annotations

import astro_env  # noqa: F401  (IERS-OFF-2026-10: no IERS downloads)

import math
from typing import Any, Sequence

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, GeocentricTrueEcliptic, SkyCoord, get_body
from astropy.time import Time

__all__ = ["get_jd_midpoint", "get_lunar_context"]

_SYNODIC_MONTH_DAYS = 29.53


def _site(lat: float, lon: float, alt_m: float) -> EarthLocation:
    return EarthLocation(lat=float(lat) * u.deg, lon=float(lon) * u.deg, height=float(alt_m) * u.m)


def _moon_field_geometry(
    jd: float, ra_field: float, dec_field: float, lat: float, lon: float, alt_m: float
) -> tuple[float, float, float, float]:
    """Return (moon_ra_deg, moon_dec_deg, moon_alt_deg, separation_deg) seen from the site.

    REPORT-AUDIT-2026-10: the Moon is near; its direction depends on where it is seen from. The
    earlier code took the barycentric (ICRS) direction of the topocentric Moon (about 1 deg off) and
    compared it with the field; now Moon and field are both placed in the site's AltAz frame.
    """
    location = _site(lat, lon, alt_m)
    t = Time(float(jd), format="jd", scale="utc", location=location)
    frame = AltAz(obstime=t, location=location)
    moon = get_body("moon", t, location=location)  # topocentric GCRS
    moon_aa = moon.transform_to(frame)
    field_aa = SkyCoord(ra=float(ra_field) * u.deg, dec=float(dec_field) * u.deg, frame="icrs").transform_to(frame)
    return (
        float(moon.ra.to(u.deg).value),
        float(moon.dec.to(u.deg).value),
        float(moon_aa.alt.to(u.deg).value),
        float(moon_aa.separation(field_aa).to(u.deg).value),
    )


def _lunar_phase(jd: float) -> tuple[float, float]:
    """Return (illuminated_pct 0-100, age_days since new moon).

    Age from the Moon-Sun difference in ecliptic longitude (0..360 deg over a synodic month); the
    elongation used before gave the same age for a waxing and a waning Moon.
    """
    t = Time(float(jd), format="jd", scale="utc")
    moon = get_body("moon", t)
    sun = get_body("sun", t)
    elong_deg = float(sun.separation(moon).to(u.deg).value)
    illuminated_pct = (1.0 - math.cos(math.radians(elong_deg))) / 2.0 * 100.0
    ecl = GeocentricTrueEcliptic(equinox=t)
    dlon = float((moon.transform_to(ecl).lon - sun.transform_to(ecl).lon).to(u.deg).value) % 360.0
    age_days = dlon / 360.0 * _SYNODIC_MONTH_DAYS
    return float(illuminated_pct), float(age_days)


def _lunar_risk_from_metrics(
    lunar_phase_pct: float,
    lunar_separation_deg: float,
    lunar_altitude_deg: float,
) -> tuple[str, str]:
    """Evaluate risk label and reason (first matching rule wins)."""
    phase = float(lunar_phase_pct)
    sep = float(lunar_separation_deg)
    alt = float(lunar_altitude_deg)

    if alt < 0.0:
        risk = "LOW"
        reason = f"Moon below horizon ({alt:.0f} deg altitude)"
    elif phase < 10.0:
        risk = "LOW"
        reason = f"New moon ({phase:.0f}% illuminated)"
    elif (phase >= 10.0 and sep < 20.0) or (phase >= 40.0 and sep < 45.0):
        risk = "HIGH"
        reason = (
            f"Moon {sep:.0f} deg from field, {phase:.0f}% illuminated, {alt:.0f} deg altitude"
        )
    elif (phase >= 40.0 and sep < 90.0) or (phase >= 10.0 and sep < 45.0):
        risk = "MEDIUM"
        reason = (
            f"Moon {sep:.0f} deg from field, {phase:.0f}% illuminated, {alt:.0f} deg altitude"
        )
    else:
        risk = "LOW"
        reason = (
            f"Moon {sep:.0f} deg from field, {phase:.0f}% illuminated, {alt:.0f} deg altitude"
        )
    return risk, reason


def get_jd_midpoint(jd_array: Sequence[float]) -> float:
    """Return midpoint JD from array/list of JD values (ignores NaN)."""
    vals: list[float] = []
    for x in jd_array:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        raise ValueError("get_jd_midpoint: no finite JD values")
    return (min(vals) + max(vals)) / 2.0


def get_lunar_context(
    jd_mid: float,
    ra_field: float,
    dec_field: float,
    lat: float,
    lon: float,
    alt_m: float,
) -> dict[str, Any]:
    """Lunar observing context at session midpoint for a field center and observer site."""
    moon_ra, moon_dec, moon_alt, separation = _moon_field_geometry(jd_mid, ra_field, dec_field, lat, lon, alt_m)
    phase_pct, age_days = _lunar_phase(jd_mid)
    risk, reason = _lunar_risk_from_metrics(phase_pct, separation, moon_alt)
    return {
        "lunar_phase_pct": float(phase_pct),
        "lunar_separation_deg": float(separation),
        "lunar_altitude_deg": float(moon_alt),
        "lunar_age_days": float(age_days),
        "lunar_risk": str(risk),
        "lunar_risk_reason": str(reason),
        "moon_ra_deg": float(moon_ra),
        "moon_dec_deg": float(moon_dec),
    }
