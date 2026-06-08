"""
lunar_context.py — Lunar observing conditions for a VYVAR session.

Uses astropy ephemeris (built-in, no internet required).
All inputs/outputs are plain Python types (float, str, dict) —
no astropy objects leak outside this module.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time

__all__ = ["get_jd_midpoint", "get_lunar_context"]

_SYNODIC_MONTH_DAYS = 29.53


def _moon_position(jd: float, lat: float, lon: float, alt_m: float) -> tuple[float, float, float]:
    """Return (ra_deg, dec_deg, alt_deg) for the Moon at ``jd`` from the observer site."""
    location = EarthLocation(
        lat=float(lat) * u.deg,
        lon=float(lon) * u.deg,
        height=float(alt_m) * u.m,
    )
    t = Time(float(jd), format="jd", scale="utc", location=location)
    moon = get_body("moon", t)
    altaz_frame = AltAz(obstime=t, location=location)
    moon_aa = moon.transform_to(altaz_frame)
    ra_deg = float(moon.icrs.ra.to(u.deg).value)
    dec_deg = float(moon.icrs.dec.to(u.deg).value)
    alt_deg = float(moon_aa.alt.to(u.deg).value)
    return ra_deg, dec_deg, alt_deg


def _lunar_phase(jd: float) -> tuple[float, float]:
    """Return (illuminated_pct 0–100, age_days since new moon approximation)."""
    t = Time(float(jd), format="jd", scale="utc")
    moon = get_body("moon", t)
    sun = get_body("sun", t)
    elong_deg = float(sun.separation(moon).to(u.deg).value)
    elong_rad = math.radians(elong_deg)
    illuminated_pct = (1.0 - math.cos(elong_rad)) / 2.0 * 100.0
    age_days = elong_deg / (360.0 / _SYNODIC_MONTH_DAYS)
    return float(illuminated_pct), float(age_days)


def _angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in degrees (haversine)."""
    ra1_r = math.radians(float(ra1))
    dec1_r = math.radians(float(dec1))
    ra2_r = math.radians(float(ra2))
    dec2_r = math.radians(float(dec2))
    d_ra = ra2_r - ra1_r
    d_dec = dec2_r - dec1_r
    a = (
        math.sin(d_dec / 2.0) ** 2
        + math.cos(dec1_r) * math.cos(dec2_r) * math.sin(d_ra / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    return math.degrees(2.0 * math.asin(math.sqrt(a)))


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
        reason = f"Moon below horizon ({alt:.0f}° altitude)"
    elif phase < 10.0:
        risk = "LOW"
        reason = f"New moon ({phase:.0f}% illuminated)"
    elif phase >= 10.0 and sep < 20.0:
        risk = "HIGH"
        reason = (
            f"Moon {sep:.0f}° from field, {phase:.0f}% illuminated, {alt:.0f}° altitude"
        )
    elif phase >= 40.0 and sep < 45.0:
        risk = "HIGH"
        reason = (
            f"Moon {sep:.0f}° from field, {phase:.0f}% illuminated, {alt:.0f}° altitude"
        )
    elif phase >= 40.0 and sep < 90.0:
        risk = "MEDIUM"
        reason = (
            f"Moon {sep:.0f}° from field, {phase:.0f}% illuminated, {alt:.0f}° altitude"
        )
    elif phase >= 10.0 and sep < 45.0:
        risk = "MEDIUM"
        reason = (
            f"Moon {sep:.0f}° from field, {phase:.0f}% illuminated, {alt:.0f}° altitude"
        )
    else:
        risk = "LOW"
        reason = (
            f"Moon {sep:.0f}° from field, {phase:.0f}% illuminated, {alt:.0f}° altitude"
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
    moon_ra, moon_dec, moon_alt = _moon_position(jd_mid, lat, lon, alt_m)
    phase_pct, age_days = _lunar_phase(jd_mid)
    separation = _angular_separation(ra_field, dec_field, moon_ra, moon_dec)
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
