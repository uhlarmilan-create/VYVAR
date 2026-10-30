"""JD / HJD / BJD helpers (mid-exposure times).

Columns of the per-frame catalogues (proc_*.csv, :func:`compute_time_columns`):

* ``jd_mid``      - UTC JD of mid-exposure (the same for every star of the frame);
* ``hjd_mid``     - HJD for the FIELD CENTRE of the frame;
* ``bjd_tdb_mid`` - BJD_TDB for the FIELD CENTRE of the frame.

The field-centre columns are frame times (order and spacing of frames). A time that belongs to
one star (light curve, candidate, transient) is computed from ``jd_mid`` and the star's own
RA/Dec with :func:`light_time_corrected_jd` (G02-18, decision C6 2026-10-29; Eastman, Siverd &
Gaudi 2010): the barycentric light time differs by ~8.8 s per degree across the field.
"""

from __future__ import annotations

import astro_env  # noqa: F401  (IERS-OFF-2026-10: no IERS downloads)

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta

from database import VyvarDatabase
from fits_meta import exposure_start_utc, field_centre_radec
from infolog import log_event
from utils import exposure_seconds

if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)

# Log each warning category at most once per interpreter session.
_WARNED_ONCE: set[str] = set()


def _warn_once_infolog(key: str, message: str) -> None:
    """First-time infolog.log_event for ``key``. Not logger.warning."""
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    log_event(message)





def mid_exposure_jd(header: fits.Header) -> float | None:
    """UTC JD of mid-exposure: :func:`exposure_start_utc` + EXPTIME / 2 (None without a start time)."""
    try:
        t_start = exposure_start_utc(header)
        if t_start is None:
            return None
        exptime = exposure_seconds(header) or 0.0
        if not exptime > 0.0:
            _warn_once_infolog(
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
            header.get("DATE-OBS"),
            exc,
        )
        return None


def light_time_corrected_jd(
    jd_utc: Any,
    ra_deg: Any,
    dec_deg: Any,
    site: tuple[float | None, float | None, float | None],
) -> tuple[np.ndarray, np.ndarray]:
    """BJD_TDB and HJD of mid-exposure instants for arrays of stars (G02-18, decision C6).

    ``jd_utc``, ``ra_deg`` and ``dec_deg`` broadcast against each other with numpy rules, so one
    call gives a star x frame grid (``ra[:, None]``, ``dec[:, None]``, ``jd[None, :]``) while the
    ephemeris is evaluated once per frame. ``site`` = (lat deg, lon deg, height m) of the observer.

    BJD_TDB = (t.tdb + ltt_bary).jd and HJD = (t_utc + ltt_helio).jd with astropy
    ``Time.light_travel_time`` (Roemer delay towards the star's own direction; Eastman, Siverd &
    Gaudi 2010, PASP 122, 935). The correction differs by up to ~8.8 s per degree between two
    directions, so a per-star time needs that star's coordinates, not the field centre.

    Non-finite inputs give NaN in both outputs. An incomplete ``site`` raises ``ValueError``.
    """
    lat, lon = site[0], site[1]
    if lat is None or lon is None or not (math.isfinite(float(lat)) and math.isfinite(float(lon))):
        raise ValueError(f"light_time_corrected_jd: observer site incomplete: {site!r}")
    elev = float(site[2]) if site[2] is not None and math.isfinite(float(site[2])) else 0.0
    jd_in = np.asarray(jd_utc, dtype=float)
    ra_in = np.asarray(ra_deg, dtype=float)
    de_in = np.asarray(dec_deg, dtype=float)
    shape = np.broadcast_shapes(jd_in.shape, ra_in.shape, de_in.shape)
    # NaN never reaches astropy: placeholders, and those cells are NaN in the result
    jd_ok = np.isfinite(jd_in)
    pos_ok = np.isfinite(ra_in) & np.isfinite(de_in)
    location = EarthLocation(lat=float(lat) * u.deg, lon=float(lon) * u.deg, height=elev * u.m)
    t = Time(np.where(jd_ok, jd_in, 2451545.0), format="jd", scale="utc", location=location)
    target = SkyCoord(
        ra=np.where(pos_ok, ra_in, 0.0) * u.deg, dec=np.where(pos_ok, de_in, 0.0) * u.deg, frame="icrs"
    )
    ltt_bary = t.light_travel_time(target, "barycentric")
    ltt_helio = t.light_travel_time(target, "heliocentric")
    bjd = np.broadcast_to(np.asarray((t.tdb + ltt_bary).jd, dtype=float), shape).copy()
    hjd = np.broadcast_to(np.asarray((t + ltt_helio).jd, dtype=float), shape).copy()
    bad = ~(np.broadcast_to(jd_ok, shape) & np.broadcast_to(pos_ok, shape))
    bjd[bad] = np.nan
    hjd[bad] = np.nan
    return bjd, hjd


def star_times(
    jd_utc: Any,
    ra_deg: Any,
    dec_deg: Any,
    site: tuple[float | None, float | None, float | None] | None,
) -> tuple[np.ndarray, str]:
    """Time of every frame for every star: (stars x frames array, time column name).

    BJD_TDB from the frames' mid-exposure UTC JD and each star's own RA/Dec (one vectorised
    :func:`light_time_corrected_jd` call), name ``"bjd_tdb"``. Without a site: the UTC JD of
    mid-exposure (the same for all stars), name ``"jd_mid"``.
    """
    import pandas as pd  # noqa: PLC0415

    jd = np.atleast_1d(np.asarray(jd_utc, dtype=float))
    ra = np.asarray(pd.to_numeric(pd.Series(np.ravel(ra_deg)), errors="coerce"), dtype=float)
    dec = np.asarray(pd.to_numeric(pd.Series(np.ravel(dec_deg)), errors="coerce"), dtype=float)
    if site is None:
        return np.broadcast_to(jd, (ra.size, jd.size)).copy(), "jd_mid"
    if ra.size == 0:
        return np.empty((0, jd.size)), "bjd_tdb"
    bjd, _hjd = light_time_corrected_jd(jd[None, :], ra[:, None], dec[:, None], site)
    return bjd, "bjd_tdb"


def compute_hjd_bjd(
    jd: float,
    ra_deg: float,
    dec_deg: float,
    site_lat: float,
    site_lon: float,
    site_elev_m: float = 0.0,
) -> tuple[float | None, float | None]:
    """Heliocentric / barycentric Julian Date of one instant and one direction.

    Scalar form of :func:`light_time_corrected_jd` (the one formula). In the per-frame export the
    direction is the FIELD CENTRE (:func:`compute_time_columns`); per-star times use
    :func:`light_time_corrected_jd` with the star's own RA/Dec.
    """
    try:
        bjd, hjd = light_time_corrected_jd(jd, ra_deg, dec_deg, (site_lat, site_lon, site_elev_m))
        hjd_f, bjd_f = float(hjd), float(bjd)
        if not (math.isfinite(hjd_f) and math.isfinite(bjd_f)):
            raise ValueError("non-finite input")
        return hjd_f, bjd_f
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


def compute_time_columns(
    header: fits.Header,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, float | None]:
    """Frame time columns of the per-frame catalogue (proc CSV): ``jd_mid``, ``hjd_mid``, ``bjd_tdb_mid``.

    ``hjd_mid`` / ``bjd_tdb_mid`` are computed for the FIELD CENTRE of the frame
    (:func:`fits_meta.field_centre_radec`), the same value on every row of the frame. They are
    frame times (time order, frame spacing), NOT the time of an individual star: across a wide
    field the barycentric correction differs by ~8.8 s per degree (G02-18). A time published for
    a star is computed from ``jd_mid`` and the star's RA/Dec with :func:`light_time_corrected_jd`.
    """
    jd = mid_exposure_jd(header)
    if jd is None:
        return {"jd_mid": None, "hjd_mid": None, "bjd_tdb_mid": None}

    lat, lon, elev = resolve_observer_location(header, db, draft_id, cfg=cfg)
    ra, dec, _src = field_centre_radec(header, db=db, draft_id=draft_id)

    if None in (lat, lon, ra, dec):
        if lat is None or lon is None:
            _warn_once_infolog(
                "observer_location_incomplete",
                "VYVAR time_utils: observer site (SITELAT/SITELONG or LOCATION via draft) missing - "
                "jd_mid filled; hjd_mid / bjd_tdb_mid left empty until location is available.",
            )
        return {"jd_mid": jd, "hjd_mid": None, "bjd_tdb_mid": None}

    hjd, bjd = compute_hjd_bjd(jd, float(ra), float(dec), float(lat), float(lon), float(elev or 0.0))
    return {"jd_mid": jd, "hjd_mid": hjd, "bjd_tdb_mid": bjd}


# --- moved from pipeline_catalog (LAYER-2026-10) ---


def _airmass_from_altitude_deg(alt_deg: float) -> float:
    """Kasten & Young (1989) airmass from altitude in degrees."""
    alt_rad = math.radians(float(alt_deg))
    return 1.0 / (
        math.sin(alt_rad) + 0.50572 * (float(alt_deg) + 6.07995) ** (-1.6364)
    )


def run_site(photometry_dir: Path) -> tuple[float, float, float] | None:
    """Observer site (lat, lon, height m) the run's aperture photometry used for the target BJD
    (``pipeline_meta.json`` ``observer_location``); None when missing or unset (0/0)."""
    from param_resolver import is_null_island_coords  # noqa: PLC0415

    try:
        loc = json.loads((Path(photometry_dir) / "pipeline_meta.json").read_text(encoding="utf-8")).get("observer_location") or {}
    except (OSError, ValueError):
        return None
    try:
        lat, lon = float(loc["lat"]), float(loc["lon"])
        alt = float(loc.get("alt_m") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)) or is_null_island_coords(lat, lon):
        return None
    return lat, lon, alt if math.isfinite(alt) else 0.0
