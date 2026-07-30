"""JD / HJD / BJD helpers for per-frame catalog metadata (mid-exposure times)."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import astropy.units as u
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta

from database import VyvarDatabase
from infolog import log_event

if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)

# Log each warning category at most once per interpreter session.
_WARNED_ONCE: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    log_event(message)


def _clamp_lat(v: float) -> float:
    return max(-90.0, min(90.0, float(v)))


def _clamp_lon(v: float) -> float:
    x = float(v)
    while x > 180.0:
        x -= 360.0
    while x < -180.0:
        x += 360.0
    return max(-180.0, min(180.0, x))


def _clamp_elev(v: float) -> float:
    return max(-500.0, min(9000.0, float(v)))


def _header_float(hdr: fits.Header, key: str) -> float | None:
    if key not in hdr:
        return None
    try:
        v = float(hdr[key])
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return None


def mid_exposure_jd(header: fits.Header) -> float | None:
    date_obs_log = header.get("DATE-OBS")
    try:
        raw = header.get("DATE-OBS")
        if raw is None:
            _warn_once(
                "date_obs_missing",
                "VYVAR time_utils: DATE-OBS missing - jd_mid / HJD / BJD columns unavailable for affected frames.",
            )
            return None
        s = str(raw).strip()
        if not s:
            _warn_once(
                "date_obs_empty",
                "VYVAR time_utils: DATE-OBS missing - jd_mid / HJD / BJD columns unavailable for affected frames.",
            )
            return None
        try:
            t_start = Time(s, format="isot", scale="utc")
        except Exception:  # noqa: BLE001
            if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
                t_start = Time(f"{s[:10]}T00:00:00", format="isot", scale="utc")
            else:
                return None

        # DATE-OBS often carries only the calendar date; TIME-OBS then holds UT of exposure start.
        if ("T" not in s.upper()) and len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
            to_raw = header.get("TIME-OBS")
            if to_raw is not None:
                to = str(to_raw).strip().replace(" ", "")
                if to:
                    try:
                        t_start = Time(f"{s[:10]}T{to}", format="isot", scale="utc")
                    except Exception as exc:  # noqa: BLE001
                        from except_fix_counters import get_except_fix_counters

                        get_except_fix_counters().timeobs_parse_fallback += 1
                        logger.error(
                            "[TIME] TIME-OBS parse failed DATE-OBS=%r TIME-OBS=%r; "
                            "jd will be DATE-only (midnight) for this frame: %s",
                            s,
                            to_raw,
                            exc,
                        )

        exptime = 0.0
        exptime_ok = False
        for key in ("EXPTIME", "EXPOSURE"):
            if key not in header:
                continue
            try:
                exptime = float(header[key])
                if math.isfinite(exptime) and exptime > 0.0:
                    exptime_ok = True
                    break
            except (TypeError, ValueError):
                continue
        if not exptime_ok:
            _warn_once(
                "exptime_missing_or_invalid",
                "VYVAR time_utils: EXPTIME/EXPOSURE missing or <=0 - jd_mid equals shutter-open "
                "(not mid-exposure); BJD/HJD may be offset by up to EXPTIME/2 for affected frames.",
            )
            exptime = 0.0

        t_mid = t_start + TimeDelta(exptime / 2.0 * u.s)
        return float(t_mid.jd)
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().jd_mid_compute_fail += 1
        logger.error(
            "[TIME] jd_mid computation failed for DATE-OBS=%r: %s",
            date_obs_log,
            exc,
        )
        return None


def compute_hjd_bjd(
    jd: float,
    ra_deg: float,
    dec_deg: float,
    site_lat: float,
    site_lon: float,
    site_elev_m: float = 0.0,
) -> tuple[float | None, float | None]:
    """Heliocentric / barycentric Julian Date at mid-exposure (astropy ``light_travel_time`` added to UTC / TDB).

    ``bjd_tdb_mid`` is the JD number of the TDB instant ``t.tdb + ltt_bary`` (geometric Roemer correction to the SSB).
    """
    try:
        location = EarthLocation(
            lat=site_lat * u.deg,
            lon=site_lon * u.deg,
            height=site_elev_m * u.m,
        )
        t = Time(jd, format="jd", scale="utc", location=location)
        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

        ltt_helio = t.light_travel_time(target, "heliocentric")
        hjd = float((t + ltt_helio).jd)

        ltt_bary = t.light_travel_time(target, "barycentric")
        bjd = float((t.tdb + ltt_bary).jd)

        return hjd, bjd
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().hjd_bjd_compute_fail += 1
        logger.error(
            "[TIME] HJD/BJD computation failed for jd=%r ra=%r dec=%r site=(%r,%r,%r): %s",
            jd,
            ra_deg,
            dec_deg,
            site_lat,
            site_lon,
            site_elev_m,
            exc,
        )
        return None, None


def resolve_observer_location(
    header: fits.Header,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
    cfg: AppConfig | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Observer site via the unified resolver.

    SITE rule (param_resolver.resolve_site): per-draft ``ID_LOCATION`` ->
    header ``SITELAT/LONG/ELEV`` -> config (flagged, never silent).  The draft
    location is the user's explicit per-observation choice and therefore wins
    over a (possibly stale) capture-software ``SITELAT``; config only applies as
    a flagged last resort.
    """
    from param_resolver import resolve_site  # local import avoids cycles

    site = resolve_site(header, db=db, draft_id=draft_id, cfg=cfg)
    if not site.ok:
        logger.debug("Observer location resolved from: unknown")
        return None, None, None
    logger.debug(
        "Observer location resolved from: %s (%.4f degN, %.4f degE, %.0fm)",
        site.source,
        site.lat,
        site.lon,
        site.elev or 0.0,
    )
    return site.lat, site.lon, site.elev


def _parse_objctradec(ra_s: str, dec_s: str) -> tuple[float | None, float | None]:
    try:
        ra_str = str(ra_s).strip()
        de_str = str(dec_s).strip()
        if not ra_str or not de_str:
            return None, None
        ra_deg = float(Angle(ra_str, unit=u.hourangle).to(u.deg).value)
        dec_deg = float(Angle(de_str, unit=u.deg).value)
        if math.isfinite(ra_deg) and math.isfinite(dec_deg):
            return ra_deg, dec_deg
    except Exception:  # noqa: BLE001
        pass
    return None, None


def resolve_target_coordinates(
    header: fits.Header,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> tuple[float | None, float | None]:
    ra = _header_float(header, "VYTARGRA")
    if ra is None:
        ra = _header_float(header, "VY_TARGRA")
    de = _header_float(header, "VYTARGDE")
    if de is None:
        de = _header_float(header, "VY_TARGDEC")
    if ra is not None and de is not None:
        return ra, de

    ra = _header_float(header, "RA")
    de = _header_float(header, "DEC")
    if ra is not None and de is not None:
        return ra, de

    if "OBJCTRA" in header and "OBJCTDEC" in header:
        ra, de = _parse_objctradec(str(header["OBJCTRA"]), str(header["OBJCTDEC"]))
        if ra is not None and de is not None:
            return ra, de

    if db is not None and draft_id is not None:
        try:
            cur = db.conn.execute(
                "SELECT CENTEROFFIELDRA, CENTEROFFIELDDE FROM OBS_DRAFT WHERE ID = ?;",
                (int(draft_id),),
            )
            row = cur.fetchone()
            if row is not None:
                r0, d0 = row[0], row[1]
                if r0 is not None and d0 is not None:
                    rf, df = float(r0), float(d0)
                    if math.isfinite(rf) and math.isfinite(df):
                        return rf, df
        except Exception:  # noqa: BLE001
            pass

    return None, None


def compute_time_columns(
    header: fits.Header,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, float | None]:
    jd = mid_exposure_jd(header)
    if jd is None:
        return {"jd_mid": None, "hjd_mid": None, "bjd_tdb_mid": None}

    lat, lon, elev = resolve_observer_location(header, db, draft_id, cfg=cfg)
    ra, dec = resolve_target_coordinates(header, db, draft_id)

    if None in (lat, lon, ra, dec):
        if lat is None or lon is None:
            _warn_once(
                "observer_location_incomplete",
                "VYVAR time_utils: observer site (SITELAT/SITELONG or LOCATION via draft) missing - "
                "jd_mid filled; hjd_mid / bjd_tdb_mid left empty until location is available.",
            )
        return {"jd_mid": jd, "hjd_mid": None, "bjd_tdb_mid": None}

    hjd, bjd = compute_hjd_bjd(jd, float(ra), float(dec), float(lat), float(lon), float(elev or 0.0))
    return {"jd_mid": jd, "hjd_mid": hjd, "bjd_tdb_mid": bjd}
