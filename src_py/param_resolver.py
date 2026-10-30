"""Unified per-parameter provenance resolver (VYVAR).

ONE place that decides, per physical parameter, which source wins:

    * GAIN / READ NOISE (CAMERA-PROFILE-2026-10): header e-/ADU -> the camera
      profile (``camera_profile``: measured / manual points, the camera's default
      point, the catalog curve) at the header gain setting and the frame binning.
      Read noise: camera profile -> header card (fallback + cross-check).
    * SITE (lat/lon/elev): per-draft ID_LOCATION -> header SITELAT/LONG/ELEV ->
      config.  config is NOT a silent fallback: if neither the draft location nor
      the header resolves the site, the result is flagged ``ok=False`` so callers
      (and the Phase 4 poor-FITS prompt) can surface it instead of silently
      borrowing a config that may belong to a different session.

Each resolution returns a :class:`Resolved` carrying the winning ``source``, the
header keyword that won (if any), an ``ok`` flag, and any cross-check warnings.
Every resolution is logged so provenance is auditable.

The module is intentionally dependency-light (stdlib + duck-typed ``db``/``cfg``)
to avoid import cycles with ``database`` / ``config`` / ``time_utils``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Warn at most once per (category) per interpreter session to avoid log spam.
_WARNED_ONCE: set[str] = set()


def _warn_once_logger(key: str, message: str) -> None:
    """First-time logger.warning for ``key``. Not infolog.log_event."""
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    logger.warning(message)


# --------------------------------------------------------------------------- #
# Sanity ranges (lo, hi).  A value is valid iff lo < v <= hi and math.isfinite.
# These reject present-but-invalid headers (e.g. GAIN=0.0), not plausible-but-wrong ones.
# --------------------------------------------------------------------------- #
SANITY: dict[str, tuple[float, float]] = {
    "gain": (0.0, 100.0),            # e-/ADU
    "read_noise": (0.0, 200.0),      # e-
    "binning": (0.5, 16.0),          # integer 1..16
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
    "elev": (-500.0, 9000.0),
}

# Header keyword variant families (first finite + in-range wins).
HEADER_KEYS: dict[str, tuple[str, ...]] = {
    "gain": ("VY_EGAIN", "EGAIN", "GAIN"),
    "read_noise": ("VYRDNOIS", "RDNOISE", "READNOISE", "RDNOISEE", "RN"),
    "binning": ("XBINNING", "BINNING"),
    "lat": ("SITELAT", "OBSLAT", "OBSGEO-B"),
    "lon": ("SITELONG", "OBSLONG", "OBSGEO-L"),
    "elev": ("SITEELEV", "OBSELEV", "OBSGEO-H", "ALTITUDE"),
}

# Relative tolerance for header<->DB cross-check warnings (equipment-intrinsic).
CROSS_CHECK_RTOL = 0.05

_E_PER_ADU_COMMENT_MARKERS = ("e-/adu", "e/adu", "electron")
_GAIN_INDEX_COMMENTS = frozenset({"gain", "index", ""})


@dataclass
class Resolved:
    """Result of resolving one parameter."""

    value: Any = None
    source: str = "unresolved"      # header | measured | manual | equipment | catalog | db | config | unresolved
    key: str | None = None          # winning header keyword (if source == header)
    ok: bool = False
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # truthy iff successfully resolved
        return bool(self.ok)


@dataclass
class SiteResult:
    lat: float | None = None
    lon: float | None = None
    elev: float | None = None
    source: str = "unresolved"      # draft | header | config | unresolved
    ok: bool = False
    warnings: list[str] = field(default_factory=list)


# Null-island guard: |lat| and |lon| both below this -> UNRESOLVED (sanity bound, not user pref).
NULL_ISLAND_LAT_LON_THRESHOLD_DEG: float = 0.01


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _is_valid(param: str, v: float | None) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(f):
        return False
    lo, hi = SANITY.get(param, (-math.inf, math.inf))
    return lo < f <= hi


def _header_value(header: Any, param: str) -> tuple[float | None, str | None]:
    """First finite + in-range header value for ``param`` and its winning key."""
    if header is None:
        return None, None
    for key in HEADER_KEYS.get(param, ()):  # ordered family
        try:
            if key not in header:
                continue
            v = float(header[key])
        except (TypeError, ValueError):
            continue
        if _is_valid(param, v):
            return v, key
    return None, None


def _clamp_param_sanity(param: str, v: float) -> float:
    """Clamp ``v`` to ``SANITY[param]``. Not a generic lo/hi clamp."""
    lo, hi = SANITY.get(param, (-math.inf, math.inf))
    return max(lo, min(hi, float(v)))


def _binning_from_header(header: Any) -> int | None:
    """Integer binning from FITS ``XBINNING`` / ``BINNING`` when trustworthy.

    Returns ``None`` when binning cannot be resolved safely (missing, invalid,
    or asymmetric ``XBINNING`` vs ``YBINNING``).
    """
    if header is None:
        return None
    x_v, _ = _header_value(header, "binning")
    if x_v is None:
        return None
    try:
        x_bin = int(round(float(x_v)))
    except (TypeError, ValueError):
        return None
    if x_bin < 1:
        return None
    if "YBINNING" in header:
        try:
            y_bin = int(round(float(header["YBINNING"])))
        except (TypeError, ValueError):
            return None
        if y_bin < 1 or y_bin != x_bin:
            return None
    return x_bin


def _header_card_comment(header: Any, key: str) -> str:
    if header is None or key not in header:
        return ""
    try:
        return str(header.comments[key] or "")
    except (TypeError, AttributeError, KeyError):
        pass
    try:
        return str(header.cards[key].comment or "")
    except (TypeError, AttributeError, KeyError):
        return ""


def _header_gain_raw(header: Any) -> tuple[float | None, str | None, str]:
    """First finite EGAIN/GAIN value (including 0) and its comment."""
    if header is None:
        return None, None, ""
    for key in HEADER_KEYS["gain"]:
        if key not in header:
            continue
        try:
            v = float(header[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            return v, key, _header_card_comment(header, key)
    return None, None, ""


def _comment_indicates_e_per_adu(comment: str) -> bool:
    c = str(comment or "").lower()
    return any(m in c for m in _E_PER_ADU_COMMENT_MARKERS)


def _comment_indicates_gain_index(comment: str) -> bool:
    return str(comment or "").strip().lower() in _GAIN_INDEX_COMMENTS


def _is_gain_setting_index(
    header: Any,
    raw: float,
    comment: str,
    reference_gain: float | None,
) -> bool:
    """Header GAIN card without an explicit unit: True when it reads as a driver setting, not e-/ADU."""
    if _comment_indicates_gain_index(comment):
        return True
    if header is not None and ("READMODE" in header or "READOUTM" in header):
        return True
    if _is_valid("gain", reference_gain) and not math.isclose(
        float(raw), float(reference_gain), rel_tol=CROSS_CHECK_RTOL
    ):
        # Integer 0..100 without e-/ADU semantics is implausible as e-/ADU for this camera.
        ri = int(round(float(raw)))
        if float(raw) == ri and 0 <= ri <= 100:
            return True
    return False


def _profile_for(
    *,
    profile: Any,
    db: Any,
    equipment_id: int | None,
    cfg: Any,
    db_value: float | None,
    param: str,
) -> Any:
    """Camera profile: given, else from ``db.conn`` / ``cfg.database_path``, else a bare profile
    holding only ``db_value`` (the camera's default point, any setting)."""
    from camera_profile import CameraProfile, CurvePoint, load_profile, profile_from_connection  # noqa: PLC0415

    if profile is not None:
        return profile
    if equipment_id is not None and db_value is None:
        conn = getattr(db, "conn", None)
        if conn is not None:
            try:
                return profile_from_connection(conn, int(equipment_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[RESOLVE %s] camera profile from DB failed (equipment %s): %s", param, equipment_id, exc)
        db_path = getattr(cfg, "database_path", None) if cfg is not None else None
        if db_path:
            return load_profile(db_path, int(equipment_id))
    pts = ()
    if _is_valid(param, db_value):
        g = float(db_value) if param == "gain" else None
        rn = float(db_value) if param == "read_noise" else None
        pts = (CurvePoint(None, 1, g, rn, "equipment"),)
    return CameraProfile(equipment_id=int(equipment_id) if equipment_id is not None else None, points=pts)


def _header_setting(header: Any, profile: Any) -> tuple[float | None, str | None, bool]:
    """(gain setting from the header or None, header key, header states e-/ADU)."""
    raw, key, comment = _header_gain_raw(header)
    if raw is None:
        return None, None, False
    # G01-04: VYVAR's own effective gain stamp (OSC channel frames: VY_EGAIN and the EGAIN written with
    # it) is e-/ADU whatever the camera writes into GAIN, as VYRDNOIS is for the read noise
    if key == "VY_EGAIN" or (key == "EGAIN" and header.get("VYCHANNL")):
        return None, key, True
    kind = getattr(profile, "header_gain", None)
    if kind == "e_per_adu":
        return None, key, True
    if kind == "setting":
        return float(raw), key, False
    if _comment_indicates_e_per_adu(comment):
        return None, key, True
    from camera_profile import lookup  # noqa: PLC0415

    ref = lookup(profile, "gain", None, 1).value
    if _is_gain_setting_index(header, raw, comment, ref):
        return float(raw), key, False
    return None, key, True


def _frame_binning(header: Any, label: str) -> int:
    b = _binning_from_header(header)
    if b is None:
        if header is not None and len(header):
            _warn_once_logger(f"bin_unresolved_{label}", f"[RESOLVE {label}] binning unresolved - bin 1 assumed")
        return 1
    return int(b)


def resolve_gain(
    header: Any = None,
    *,
    db: Any = None,
    equipment_id: int | None = None,
    cfg: Any = None,
    db_value: float | None = None,
    profile: Any = None,
) -> Resolved:
    """Gain in e-/ADU (native ADC units) of the frame (CAMERA-PROFILE-2026-10):

    1. the header states e-/ADU (``EGAIN``, or ``GAIN`` with an e-/ADU comment, or a camera whose
       header GAIN is e-/ADU) -> used as is;
    2. the header states the driver gain setting -> the camera's curve at that setting and the frame's
       binning (measured > manual > the camera's default point > catalog);
    3. no setting in the header -> the camera's default setting;
    4. otherwise unresolved (callers stop with a message).
    """
    from camera_profile import lookup  # noqa: PLC0415

    prof = _profile_for(profile=profile, db=db, equipment_id=equipment_id, cfg=cfg, db_value=db_value, param="gain")
    res = Resolved()
    binning = _frame_binning(header, "gain")
    setting, hdr_key, is_e_per_adu = _header_setting(header, prof)
    raw, _, _ = _header_gain_raw(header)
    if is_e_per_adu and raw is not None and _is_valid("gain", raw):
        res.value, res.source, res.key, res.ok = float(raw), "header", hdr_key, True
        ref = lookup(prof, "gain", None, binning)
        if ref.value is not None and not math.isclose(float(raw), float(ref.value), rel_tol=CROSS_CHECK_RTOL):
            msg = (
                f"[RESOLVE gain] header {hdr_key}={raw:g} disagrees with the camera profile "
                f"{float(ref.value):g} ({ref.detail}, >{CROSS_CHECK_RTOL:.0%}); using header (session truth)"
            )
            res.warnings.append(msg)
            _warn_once_logger(f"xcheck_gain_header_wins_{prof.equipment_id}", msg)
        return res
    lk = lookup(prof, "gain", setting, binning)
    if lk.value is not None and _is_valid("gain", lk.value):
        res.value, res.source, res.ok = float(lk.value), lk.source, True
        res.key = f"{hdr_key}={setting:g}" if setting is not None and hdr_key else None
        if setting is not None and prof.default_setting is None and lk.source == "equipment":
            msg = (
                f"[RESOLVE gain] camera {prof.name or prof.equipment_id}: header gain setting {setting:g}, "
                f"no default setting stored - GAIN_ADU {lk.value:g} assumed to belong to it"
            )
            res.warnings.append(msg)
            _warn_once_logger(f"gain_default_setting_unknown_{prof.equipment_id}_{setting:g}", msg)
        logger.debug("[RESOLVE gain] -> %s (%s)", res.value, lk.detail)
        return res
    res.source, res.ok = "unresolved", False
    res.warnings.append(f"[RESOLVE gain] {lk.detail} (camera {prof.name or prof.equipment_id}, bin {binning})")
    return res


def resolve_read_noise(
    header: Any = None,
    *,
    db: Any = None,
    equipment_id: int | None = None,
    cfg: Any = None,
    db_value: float | None = None,
    profile: Any = None,
) -> Resolved:
    """Read noise in e- of the frame: the camera's curve at the header gain setting and binning (same
    order as the gain); a header read-noise card only when the camera has none (it cross-checks
    otherwise)."""
    from camera_profile import lookup  # noqa: PLC0415

    prof = _profile_for(
        profile=profile, db=db, equipment_id=equipment_id, cfg=cfg, db_value=db_value, param="read_noise"
    )
    res = Resolved()
    # VYVAR's own effective stamp (OSC channel frames) is the session truth, like VY_EGAIN for the gain.
    vy = _header_value(header, "read_noise")
    if vy[1] == "VYRDNOIS" and vy[0] is not None:
        res.value, res.source, res.key, res.ok = float(vy[0]), "header", "VYRDNOIS", True
        return res
    binning = _frame_binning(header, "read_noise")
    setting, _, _ = _header_setting(header, prof)
    lk = lookup(prof, "read_noise", setting, binning)
    hdr_v, hdr_key = _header_value(header, "read_noise")
    if lk.value is not None and _is_valid("read_noise", lk.value):
        res.value, res.source, res.ok = float(lk.value), lk.source, True
        if hdr_v is not None and not math.isclose(float(hdr_v), float(lk.value), rel_tol=CROSS_CHECK_RTOL):
            msg = (
                f"[RESOLVE read_noise] header {hdr_key}={hdr_v:g} disagrees with the camera profile "
                f"{float(lk.value):g} ({lk.detail}); using the camera profile"
            )
            res.warnings.append(msg)
            _warn_once_logger(f"xcheck_read_noise_{prof.equipment_id}", msg)
        return res
    if hdr_v is not None:
        res.value, res.source, res.key, res.ok = float(hdr_v), "header", hdr_key, True
        return res
    res.source, res.ok = "unresolved", False
    res.warnings.append(f"[RESOLVE read_noise] {lk.detail} (camera {prof.name or prof.equipment_id}, bin {binning})")
    return res


# --------------------------------------------------------------------------- #
# SITE: per-draft ID_LOCATION -> header SITELAT/LONG/ELEV -> config (flagged)
# --------------------------------------------------------------------------- #

def _draft_location(db: Any, draft_id: int | None) -> tuple[float, float, float] | None:
    if db is None or draft_id is None or not hasattr(db, "get_draft_location_id"):
        return None
    try:
        eff_loc_id = db.get_draft_location_id(int(draft_id))
        if eff_loc_id is None:
            return None
        loc_only = db.conn.execute(
            "SELECT LATITUDE, LONGITUDE, ALTITUDE FROM LOCATION WHERE ID = ?;",
            (int(eff_loc_id),),
        ).fetchone()
        if loc_only is None:
            return None
        la = float(loc_only[0])
        lo = float(loc_only[1])
        alt_raw = loc_only[2]
        al = float(alt_raw) if alt_raw is not None and math.isfinite(float(alt_raw)) else 0.0
    except Exception as exc:  # noqa: BLE001
        logging.warning('[EXC-0118] science path may skip or use stale defaults () / row = cur.fetchone() / except Exceptio...: %s', exc)
        return None
    if not (math.isfinite(la) and math.isfinite(lo)):
        return None
    return la, lo, al


def _header_site(header: Any) -> tuple[float, float, float] | None:
    lat, _ = _header_value(header, "lat")
    lon, _ = _header_value(header, "lon")
    if lat is None or lon is None:
        return None
    elev, _ = _header_value(header, "elev")
    return float(lat), float(lon), float(elev if elev is not None else 0.0)


def is_null_island_coords(
    lat: float,
    lon: float,
    *,
    threshold_deg: float = NULL_ISLAND_LAT_LON_THRESHOLD_DEG,
) -> bool:
    """True when the coordinates are no site: not finite (unset), or both within ``threshold_deg``
    of (0, 0)."""
    if not (math.isfinite(float(lat)) and math.isfinite(float(lon))):
        return True
    return abs(float(lat)) < float(threshold_deg) and abs(float(lon)) < float(threshold_deg)


def _draft_id_location(db: Any, draft_id: int | None) -> int | None:
    if db is None or draft_id is None:
        return None
    try:
        if hasattr(db, "get_draft_location_id"):
            return db.get_draft_location_id(int(draft_id))
        return None
    except Exception:  # noqa: BLE001
        return None


def _apply_null_island_guard(
    res: SiteResult,
    *,
    db: Any = None,
    draft_id: int | None = None,
) -> SiteResult:
    if not res.ok or res.lat is None or res.lon is None:
        return res
    if not is_null_island_coords(res.lat, res.lon):
        return res
    id_loc = _draft_id_location(db, draft_id)
    detail = f"({res.lat:.4f},{res.lon:.4f}) source={res.source}"
    if draft_id is not None:
        detail += f" draft_id={draft_id}"
    if id_loc is not None:
        detail += f" ID_LOCATION={id_loc}"
    msg = f"[RESOLVE site] null-island observer coordinates {detail} - treated as UNRESOLVED"
    logger.error(msg)
    _warn_once_logger("site_null_island", msg)
    res.warnings.append(msg)
    res.ok = False
    res.source = "unresolved"
    return res


def resolve_site(
    header: Any = None,
    *,
    db: Any = None,
    draft_id: int | None = None,
    cfg: Any = None,
    allow_config: bool = True,
) -> SiteResult:
    """Resolve observer site.

    Priority: per-draft ID_LOCATION -> header SITELAT/LONG/ELEV -> config.
    Config is used ONLY when ``allow_config`` and it is non-zero, and the result is
    flagged (``source='config'``) with a warning so it is never a silent fallback.
    If nothing resolves, ``ok=False`` and the caller should prompt (Phase 4).
    """
    res = SiteResult()

    drv = _draft_location(db, draft_id)
    if drv is not None:
        la, lo, el = drv
        res.lat, res.lon, res.elev = _clamp_param_sanity("lat", la), _clamp_param_sanity("lon", lo), _clamp_param_sanity("elev", el)
        res.source, res.ok = "draft", True
        # Cross-check header site if present.
        hsv = _header_site(header)
        if hsv is not None and (
            abs(hsv[0] - res.lat) > 0.01 or abs(hsv[1] - res.lon) > 0.01
        ):
            msg = (
                f"[RESOLVE site] draft ID_LOCATION ({res.lat:.4f},{res.lon:.4f}) disagrees with "
                f"header SITELAT/LONG ({hsv[0]:.4f},{hsv[1]:.4f}); using draft location"
            )
            res.warnings.append(msg)
            _warn_once_logger("xcheck_site", msg)
        logger.debug("[RESOLVE site] -> draft (%.4f,%.4f,%.0f)", res.lat, res.lon, res.elev or 0.0)
        return _apply_null_island_guard(res, db=db, draft_id=draft_id)

    hsv = _header_site(header)
    if hsv is not None:
        res.lat, res.lon, res.elev = _clamp_param_sanity("lat", hsv[0]), _clamp_param_sanity("lon", hsv[1]), _clamp_param_sanity("elev", hsv[2])
        res.source, res.ok = "header", True
        # REPORT-AUDIT-2026-10: a header with SITELAT/LONG but no elevation gave 0 m; the configured
        # site's altitude applies when the configured site is the same place (within 0.01 deg)
        if _header_value(header, "elev")[0] is None and cfg is not None:
            try:
                clat = float(getattr(cfg, "observer_lat", float("nan")))
                clon = float(getattr(cfg, "observer_lon", float("nan")))
                calt = float(getattr(cfg, "observer_alt_m", float("nan")))
            except (TypeError, ValueError):
                clat = clon = calt = 0.0
            if abs(clat - res.lat) <= 0.01 and abs(clon - res.lon) <= 0.01 and calt > 0:
                res.elev = _clamp_param_sanity("elev", calt)
        logger.debug("[RESOLVE site] -> header (%.4f,%.4f,%.0f)", res.lat, res.lon, res.elev or 0.0)
        return _apply_null_island_guard(res, db=db, draft_id=draft_id)

    if allow_config and cfg is not None:
        try:
            clat = float(getattr(cfg, "observer_lat", float("nan")))
            clon = float(getattr(cfg, "observer_lon", float("nan")))
            calt = float(getattr(cfg, "observer_alt_m", float("nan")))
        except (TypeError, ValueError):
            clat = clon = calt = 0.0
        if not is_null_island_coords(clat, clon):
            res.lat, res.lon, res.elev = _clamp_param_sanity("lat", clat), _clamp_param_sanity("lon", clon), _clamp_param_sanity("elev", calt)
            res.source, res.ok = "config", True
            _warn_once_logger(
                "site_config_fallback",
                "[RESOLVE site] no per-draft ID_LOCATION and no header SITELAT/LONG; "
                "using config observer location as a FLAGGED fallback - verify it belongs to this session.",
            )
            logger.debug("[RESOLVE site] -> config (flagged) (%.4f,%.4f,%.0f)", res.lat, res.lon, res.elev or 0.0)
            return _apply_null_island_guard(res, db=db, draft_id=draft_id)

    res.source, res.ok = "unresolved", False
    _warn_once_logger(
        "site_unresolved",
        "[RESOLVE site] observer site unresolved (no draft location, no header, no config) - "
        "BJD/HJD/airmass left empty; surface in poor-FITS prompt.",
    )
    return res
