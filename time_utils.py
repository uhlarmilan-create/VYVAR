"""JD / HJD / BJD helpers for per-frame catalog metadata (mid-exposure times)."""

from __future__ import annotations

import math

import astropy.units as u
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta

from database import VyvarDatabase
from infolog import log_event

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
    try:
        raw = header.get("DATE-OBS")
        if raw is None:
            _warn_once(
                "date_obs_missing",
                "VYVAR time_utils: DATE-OBS missing — jd_mid / HJD / BJD columns unavailable for affected frames.",
            )
            return None
        s = str(raw).strip()
        if not s:
            _warn_once(
                "date_obs_empty",
                "VYVAR time_utils: DATE-OBS missing — jd_mid / HJD / BJD columns unavailable for affected frames.",
            )
            return None
        try:
            t_start = Time(s, format="isot", scale="utc")
        except Exception:
            if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
                t_start = Time(f"{s[:10]}T00:00:00", format="isot", scale="utc")
            else:
                return None

        exptime = 0.0
        for key in ("EXPTIME", "EXPOSURE"):
            if key not in header:
                continue
            try:
                exptime = float(header[key])
                if math.isfinite(exptime):
                    break
            except (TypeError, ValueError):
                continue

        t_mid = t_start + TimeDelta(exptime / 2.0 * u.s)
        return float(t_mid.jd)
    except Exception:
        return None


def compute_hjd_bjd(
    jd: float,
    ra_deg: float,
    dec_deg: float,
    site_lat: float,
    site_lon: float,
    site_elev_m: float = 0.0,
) -> tuple[float | None, float | None]:
    try:
        location = EarthLocation(
            lat=site_lat * u.deg,
            lon=site_lon * u.deg,
            height=site_elev_m * u.m,
        )
        t = Time(jd, format="jd", scale="utc", location=location)
        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

        ltt_helio = t.light_travel_time(target, "heliocentric")
        hjd = float((t.utc + ltt_helio).jd)

        ltt_bary = t.light_travel_time(target, "barycentric")
        bjd = float((t.tdb + ltt_bary).jd)

        return hjd, bjd
    except Exception:
        return None, None


def resolve_observer_location(
    header: fits.Header,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> tuple[float | None, float | None, float | None]:
    lat = _header_float(header, "SITELAT")
    lon = _header_float(header, "SITELONG")
    elev = _header_float(header, "SITEELEV")
    if lat is None:
        lat = _header_float(header, "OBSLAT")
    if lon is None:
        lon = _header_float(header, "OBSLONG")
    if elev is None:
        elev = _header_float(header, "OBSELEV")

    if lat is not None and lon is not None:
        try:
            return _clamp_lat(lat), _clamp_lon(lon), _clamp_elev(elev if elev is not None else 0.0)
        except Exception:
            pass

    if db is not None and draft_id is not None:
        try:
            cur = db.conn.execute(
                """
                SELECT l.LATITUDE, l.LONGITUDE, l.ALTITUDE
                FROM OBS_DRAFT d
                JOIN LOCATION l ON l.ID = d.ID_LOCATION
                WHERE d.ID = ?
                """,
                (int(draft_id),),
            )
            row = cur.fetchone()
            if row is not None:
                la, lo, al = float(row[0]), float(row[1]), row[2]
                el = float(al) if al is not None and math.isfinite(float(al)) else 0.0
                if math.isfinite(la) and math.isfinite(lo):
                    return _clamp_lat(la), _clamp_lon(lo), _clamp_elev(el)
        except Exception:
            pass

    return None, None, None


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
    except Exception:
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
        except Exception:
            pass

    return None, None


def compute_time_columns(
    header: fits.Header,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> dict[str, float | None]:
    jd = mid_exposure_jd(header)
    if jd is None:
        return {"jd_mid": None, "hjd_mid": None, "bjd_tdb_mid": None}

    lat, lon, elev = resolve_observer_location(header, db, draft_id)
    ra, dec = resolve_target_coordinates(header, db, draft_id)

    if None in (lat, lon, ra, dec):
        if lat is None or lon is None:
            _warn_once(
                "observer_location_incomplete",
                "VYVAR time_utils: observer site (SITELAT/SITELONG or LOCATION via draft) missing — "
                "jd_mid filled; hjd_mid / bjd_tdb_mid left empty until location is available.",
            )
        return {"jd_mid": jd, "hjd_mid": None, "bjd_tdb_mid": None}

    hjd, bjd = compute_hjd_bjd(jd, float(ra), float(dec), float(lat), float(lon), float(elev or 0.0))
    return {"jd_mid": jd, "hjd_mid": hjd, "bjd_tdb_mid": bjd}
