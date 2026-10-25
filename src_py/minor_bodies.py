# -*- coding: ascii -*-
"""Minor planets and comets in a field, offline (TRANSIENT-2026-10, Milan 2026-10-25).

Orbits from the Minor Planet Center files the user keeps in ``catalogs/``:

* ``MPCORB.DAT`` or ``MPCORB.DAT.gz`` - osculating elements of the minor planets
  (https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz, format:
  https://www.minorplanetcenter.net/iau/info/MPOrbitFormat.html);
* ``CometEls.txt`` - elements of the comets
  (https://www.minorplanetcenter.net/iau/MPCORB/CometEls.txt, format:
  https://www.minorplanetcenter.net/iau/info/CometOrbitFormat.html).

Positions are two-body (Kepler) heliocentric orbits, J2000 ecliptic -> equatorial, light-time
corrected, topocentric (observer from the run's site), Earth from the astropy built-in
ephemeris. Without planetary perturbations the error grows with the distance from the epoch
of the elements - tens of arcseconds for main-belt asteroids a few months away, more for
near-Earth objects and comets: enough to name a moving object in the field, not for astrometry.

Brightness: minor planets H, G (Bowell et al. 1989); comets total magnitude
m = H + 5 log Delta + 2.5 G log r (the MPC comet "G" is the K1 / 2.5 slope).

Neither file present -> nothing is computed (the transient search still runs).
"""
from __future__ import annotations

import gzip
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MPCORB_NAMES = ("MPCORB.DAT.gz", "MPCORB.DAT")
COMETELS_NAME = "CometEls.txt"
#: Milan 2026-10-25: bodies to V 20 are enough
MINOR_BODY_MAG_LIMIT = 20.0
K_GAUSS = 0.01720209895  # rad / day
C_AU_PER_DAY = 173.1446327
OBLIQUITY_J2000 = math.radians(23.4392911)
_CACHE_VERSION = 1


# ---------------------------------------------------------------- file parsing
def _unpack_epoch(s: str) -> float:
    """MPC packed date (e.g. K2555) -> JD (TT, 0h)."""
    s = s.strip()
    cent = {"I": 1800, "J": 1900, "K": 2000, "L": 2100}[s[0]]
    year = cent + int(s[1:3])

    def _c(ch: str) -> int:
        return int(ch) if ch.isdigit() else ord(ch) - ord("A") + 10

    month, day = _c(s[3]), _c(s[4])
    return _jd(year, month, float(day))


def _jd(year: int, month: int, day: float) -> float:
    """Julian date of a Gregorian calendar date (day may carry a fraction)."""
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5


def _f(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_mpcorb(lines: Iterable[str]) -> pd.DataFrame:
    """MPCORB.DAT lines -> one row per minor planet (angles in degrees, a in au)."""
    rows: list[tuple] = []
    started = False
    for ln in lines:
        if not started:
            if ln.startswith("-----"):
                started = True
            elif len(ln) > 160 and ln[0:7].strip() and _f(ln[92:103]) > 0:
                started = True  # file without the header
            else:
                continue
            if ln.startswith("-----"):
                continue
        if len(ln) < 103 or not ln[0:7].strip():
            continue
        H = _f(ln[8:13])
        if not math.isfinite(H):
            continue
        try:
            ep = _unpack_epoch(ln[20:25])
        except (KeyError, ValueError, IndexError):
            continue
        G = _f(ln[14:19])
        name = ln[166:194].strip() if len(ln) > 166 else ""
        rows.append((ln[0:7].strip(), name or ln[0:7].strip(), H, G if math.isfinite(G) else 0.15, ep,
                     _f(ln[26:35]), _f(ln[37:46]), _f(ln[48:57]), _f(ln[59:68]), _f(ln[70:79]),
                     _f(ln[80:91]), _f(ln[92:103])))
    df = pd.DataFrame(rows, columns=["packed", "name", "H", "G", "epoch_jd", "M", "peri", "node",
                                     "incl", "e", "n", "a"])
    return df[np.isfinite(df[["M", "peri", "node", "incl", "e", "n", "a"]]).all(axis=1)].reset_index(drop=True)


def parse_cometels(lines: Iterable[str]) -> pd.DataFrame:
    """CometEls.txt lines -> one row per comet (perihelion time JD TT, q au, angles deg)."""
    rows: list[tuple] = []
    for ln in lines:
        if len(ln) < 100:
            continue
        try:
            y, mo, d = int(ln[14:18]), int(ln[19:21]), float(ln[22:29])
        except ValueError:
            continue
        q, e = _f(ln[30:39]), _f(ln[41:49])
        if not (math.isfinite(q) and q > 0 and math.isfinite(e) and 0 <= e < 10):
            continue
        name = ln[102:158].strip() or ln[0:12].strip()
        H, G = _f(ln[91:95]), _f(ln[96:100])
        rows.append((name, _jd(y, mo, d), q, e, _f(ln[51:59]), _f(ln[61:69]), _f(ln[71:79]),
                     H if math.isfinite(H) else 12.0, G if math.isfinite(G) else 4.0))
    df = pd.DataFrame(rows, columns=["name", "tp_jd", "q", "e", "peri", "node", "incl", "H", "G"])
    return df[np.isfinite(df[["peri", "node", "incl"]]).all(axis=1)].reset_index(drop=True)


def _open_text(p: Path):
    return gzip.open(p, "rt", encoding="ascii", errors="replace") if p.suffix == ".gz" else open(p, encoding="ascii", errors="replace")


def load_orbits(catalogs_dir: Path | str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    """(minor planets, comets, info). A parsed MPCORB is cached next to it (``.vyvar_cache.pkl``),
    keyed by the file's size and modification time."""
    d = Path(catalogs_dir)
    info: dict[str, Any] = {"catalogs_dir": str(d)}
    mp = None
    src = next((d / n for n in MPCORB_NAMES if (d / n).is_file()), None)
    if src is not None:
        st = src.stat()
        key = (_CACHE_VERSION, st.st_size, int(st.st_mtime))
        cache = src.with_name(src.name + ".vyvar_cache.pkl")
        try:
            obj = pd.read_pickle(cache) if cache.is_file() else None
            if obj is not None and obj.get("key") == key:
                mp = obj["df"]
        except Exception:  # noqa: BLE001
            mp = None
        if mp is None:
            with _open_text(src) as fh:
                mp = parse_mpcorb(fh)
            try:
                pd.to_pickle({"key": key, "df": mp}, cache)
            except OSError as exc:
                LOGGER.info("[MINOR-BODIES] cache not written: %s", exc)
        info["mpcorb"] = {"file": str(src), "n": int(len(mp))}
    else:
        info["mpcorb"] = None
    cm = None
    cf = d / COMETELS_NAME
    if cf.is_file():
        with _open_text(cf) as fh:
            cm = parse_cometels(fh)
        info["cometels"] = {"file": str(cf), "n": int(len(cm))}
    else:
        info["cometels"] = None
    return mp, cm, info


# ---------------------------------------------------------------- orbits
def _kepler_E(M: np.ndarray, e: np.ndarray) -> np.ndarray:
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(30):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.nanmax(np.abs(dE)) < 1e-12:
            break
    return E


def _hyper_H(M: np.ndarray, e: np.ndarray) -> np.ndarray:
    H = np.arcsinh(M / e)
    for _ in range(50):
        dH = (e * np.sinh(H) - H - M) / (e * np.cosh(H) - 1.0)
        H = H - dH
        if np.nanmax(np.abs(dH)) < 1e-12:
            break
    return H


def _plane_to_equatorial(r: np.ndarray, nu: np.ndarray, peri: np.ndarray, node: np.ndarray,
                         incl: np.ndarray) -> np.ndarray:
    """Heliocentric equatorial (ICRS axes) position, au, shape (n, 3)."""
    w, O, i = np.radians(peri), np.radians(node), np.radians(incl)
    u = w + nu
    x = r * (np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(i))
    y = r * (np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(i))
    z = r * (np.sin(u) * np.sin(i))
    ce, se = math.cos(OBLIQUITY_J2000), math.sin(OBLIQUITY_J2000)
    return np.c_[x, y * ce - z * se, y * se + z * ce]


def heliocentric_minor_planets(df: pd.DataFrame, jd_tt: float | np.ndarray) -> np.ndarray:
    """Positions of MPCORB rows at ``jd_tt`` (scalar or one value per row)."""
    M = np.radians(df["M"].to_numpy(float) + df["n"].to_numpy(float) * (np.asarray(jd_tt) - df["epoch_jd"].to_numpy(float)))
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi
    e = df["e"].to_numpy(float)
    a = df["a"].to_numpy(float)
    E = _kepler_E(M, e)
    r = a * (1.0 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2), np.sqrt(1.0 - e) * np.cos(E / 2))
    return _plane_to_equatorial(r, nu, df["peri"].to_numpy(float), df["node"].to_numpy(float), df["incl"].to_numpy(float))


def heliocentric_comets(df: pd.DataFrame, jd_tt: float | np.ndarray) -> np.ndarray:
    """Positions of CometEls rows (elliptic, parabolic, hyperbolic) at ``jd_tt``."""
    q, e = df["q"].to_numpy(float), df["e"].to_numpy(float)
    dt = np.asarray(jd_tt, dtype=float) - df["tp_jd"].to_numpy(float)
    dt = np.broadcast_to(dt, q.shape).astype(float)
    r = np.full(q.shape, np.nan)
    nu = np.full(q.shape, np.nan)
    ell = e < 1.0 - 1e-6
    par = np.abs(e - 1.0) <= 1e-6
    hyp = e > 1.0 + 1e-6
    if ell.any():
        a = q[ell] / (1.0 - e[ell])
        M = K_GAUSS / a**1.5 * dt[ell]
        M = np.mod(M + np.pi, 2 * np.pi) - np.pi
        E = _kepler_E(M, e[ell])
        r[ell] = a * (1.0 - e[ell] * np.cos(E))
        nu[ell] = 2.0 * np.arctan2(np.sqrt(1.0 + e[ell]) * np.sin(E / 2), np.sqrt(1.0 - e[ell]) * np.cos(E / 2))
    if par.any():  # Barker: s^3 + 3 s = W
        W = 3.0 * K_GAUSS / np.sqrt(2.0 * q[par] ** 3) * dt[par]
        s = 2.0 * np.sinh(np.arcsinh(W / 2.0) / 3.0)
        r[par] = q[par] * (1.0 + s * s)
        nu[par] = 2.0 * np.arctan(s)
    if hyp.any():
        a = q[hyp] / (e[hyp] - 1.0)
        M = K_GAUSS / a**1.5 * dt[hyp]
        H = _hyper_H(M, e[hyp])
        r[hyp] = a * (e[hyp] * np.cosh(H) - 1.0)
        nu[hyp] = 2.0 * np.arctan(np.sqrt((e[hyp] + 1.0) / (e[hyp] - 1.0)) * np.tanh(H / 2.0))
    return _plane_to_equatorial(r, nu, df["peri"].to_numpy(float), df["node"].to_numpy(float), df["incl"].to_numpy(float))


def observer_heliocentric(jd_utc: float, lat_deg: float | None, lon_deg: float | None,
                          height_m: float | None) -> tuple[np.ndarray, float]:
    """(heliocentric ICRS position of the observer in au, TT - UTC in days)."""
    import astropy.units as u  # noqa: PLC0415
    from astropy.coordinates import EarthLocation, get_body_barycentric  # noqa: PLC0415
    from astropy.time import Time  # noqa: PLC0415

    t = Time(jd_utc, format="jd", scale="utc")
    earth = get_body_barycentric("earth", t).get_xyz().to(u.au).value
    sun = get_body_barycentric("sun", t).get_xyz().to(u.au).value
    pos = np.asarray(earth - sun, dtype=float)
    if lat_deg is not None and lon_deg is not None and math.isfinite(lat_deg) and math.isfinite(lon_deg):
        loc = EarthLocation.from_geodetic(lon_deg * u.deg, lat_deg * u.deg, (height_m or 0.0) * u.m)
        g, _v = loc.get_gcrs_posvel(t)
        pos = pos + g.get_xyz().to(u.au).value
    return pos, float((t.tt.jd - t.utc.jd))


def _hg_mag(H: np.ndarray, G: np.ndarray, r: np.ndarray, delta: np.ndarray, R: float) -> np.ndarray:
    cosa = np.clip((r * r + delta * delta - R * R) / (2 * r * delta), -1, 1)
    ta = np.tan(np.arccos(cosa) / 2.0)
    p1 = np.exp(-3.33 * ta**0.63)
    p2 = np.exp(-1.87 * ta**1.22)
    return H + 5.0 * np.log10(r * delta) - 2.5 * np.log10(np.clip((1 - G) * p1 + G * p2, 1e-12, None))


@dataclass
class BodyPositions:
    kind: str  # "minor_planet" | "comet"
    name: np.ndarray
    ra_deg: np.ndarray  # (n_bodies, n_times)
    dec_deg: np.ndarray
    mag: np.ndarray  # (n_bodies, n_times)
    r_au: np.ndarray
    delta_au: np.ndarray


def _topocentric(fn, df: pd.DataFrame, jd_utc: np.ndarray, site: tuple) -> BodyPositions | None:
    n_t = len(jd_utc)
    ra = np.zeros((len(df), n_t))
    dec = np.zeros((len(df), n_t))
    dl = np.zeros((len(df), n_t))
    rr = np.zeros((len(df), n_t))
    Rs = np.zeros(n_t)
    for k, t in enumerate(jd_utc):
        obs, dtt = observer_heliocentric(float(t), *site)
        tt = float(t) + dtt
        p = fn(df, tt)
        for _ in range(2):  # light time
            delta = np.linalg.norm(p - obs, axis=1)
            p = fn(df, tt - delta / C_AU_PER_DAY)
        v = p - obs
        delta = np.linalg.norm(v, axis=1)
        ra[:, k] = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        dec[:, k] = np.degrees(np.arcsin(np.clip(v[:, 2] / delta, -1, 1)))
        dl[:, k] = delta
        rr[:, k] = np.linalg.norm(p, axis=1)
        Rs[k] = float(np.linalg.norm(obs))
    return ra, dec, dl, rr, Rs


def bodies_in_field(
    mp: pd.DataFrame | None,
    cm: pd.DataFrame | None,
    *,
    jd_utc: np.ndarray,
    ra_c: float,
    dec_c: float,
    radius_deg: float,
    site: tuple[float | None, float | None, float | None] = (None, None, None),
    mag_limit: float = MINOR_BODY_MAG_LIMIT,
) -> pd.DataFrame:
    """Bodies brighter than ``mag_limit`` within ``radius_deg`` of the field centre at any of
    ``jd_utc`` (one row per body and time: name, kind, jd, ra, dec, mag, r, delta, rate)."""
    jd_utc = np.atleast_1d(np.asarray(jd_utc, dtype=float))
    t_mid = float(np.median(jd_utc))
    span = float(jd_utc.max() - jd_utc.min()) if jd_utc.size else 0.0
    out = []
    for kind, df, fn in (("minor_planet", mp, heliocentric_minor_planets), ("comet", cm, heliocentric_comets)):
        if df is None or df.empty:
            continue
        # coarse pass at mid-night: a margin for motion (NEO up to ~1 deg / h near the Earth)
        ra, dec, dl, rr, Rs = _topocentric(fn, df, np.array([t_mid]), site)
        sep = _sep_deg(ra[:, 0], dec[:, 0], ra_c, dec_c)
        rate_margin = np.clip(1.0 / np.clip(dl[:, 0], 1e-3, None), 0.05, 30.0) * (span + 0.05)  # deg
        near = sep < radius_deg + rate_margin
        if kind == "minor_planet":
            m0 = _hg_mag(df["H"].to_numpy(float), df["G"].to_numpy(float), rr[:, 0], dl[:, 0], float(Rs[0]))
        else:
            m0 = df["H"].to_numpy(float) + 5 * np.log10(dl[:, 0]) + 2.5 * df["G"].to_numpy(float) * np.log10(rr[:, 0])
        near &= m0 <= mag_limit + 1.0
        sub = df.loc[near].reset_index(drop=True)
        if sub.empty:
            continue
        ra, dec, dl, rr, Rs = _topocentric(fn, sub, jd_utc, site)
        if kind == "minor_planet":
            mag = np.vstack([_hg_mag(sub["H"].to_numpy(float), sub["G"].to_numpy(float), rr[:, k], dl[:, k], float(Rs[k]))
                             for k in range(jd_utc.size)]).T
        else:
            mag = sub["H"].to_numpy(float)[:, None] + 5 * np.log10(dl) + 2.5 * sub["G"].to_numpy(float)[:, None] * np.log10(rr)
        sep = _sep_deg(ra, dec, ra_c, dec_c)
        keep = ((sep < radius_deg) & (mag <= mag_limit)).any(axis=1)
        for i in np.flatnonzero(keep):
            if jd_utc.size > 1:
                dra = (ra[i, -1] - ra[i, 0] + 180) % 360 - 180
                rate = math.hypot(dra * math.cos(math.radians(dec[i].mean())), dec[i, -1] - dec[i, 0]) * 3600 / (span * 24)
            else:
                rate = float("nan")
            for k in range(jd_utc.size):
                out.append({"name": str(sub.loc[i, "name"]), "kind": kind, "jd": float(jd_utc[k]),
                            "ra_deg": float(ra[i, k]), "dec_deg": float(dec[i, k]), "mag": float(mag[i, k]),
                            "r_au": float(rr[i, k]), "delta_au": float(dl[i, k]), "rate_arcsec_h": rate})
    return pd.DataFrame(out, columns=["name", "kind", "jd", "ra_deg", "dec_deg", "mag", "r_au", "delta_au", "rate_arcsec_h"])


def _sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = (np.radians(np.asarray(v, dtype=float)) for v in (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


__all__ = [
    "COMETELS_NAME",
    "MINOR_BODY_MAG_LIMIT",
    "MPCORB_NAMES",
    "bodies_in_field",
    "heliocentric_comets",
    "heliocentric_minor_planets",
    "load_orbits",
    "parse_cometels",
    "parse_mpcorb",
]
