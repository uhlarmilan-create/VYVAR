"""Unified per-parameter provenance resolver (VYVAR Phase 1).

ONE place that decides, per physical parameter, which source wins:

    * OBSERVATION-SPECIFIC (pointing, time, exptime, binning, filter, ccd-temp,
      plate-scale): HEADER / solved-WCS (valid) -> DB -> config.
    * EQUIPMENT-INTRINSIC gain: HEADER (e-/ADU or setting-index mapped) ->
      DB -> config.  Read-noise and other equipment-intrinsic params remain
      DB-SET (valid) -> HEADER (fallback + cross-check) -> config.
      (DB authority is required because plausible-but-wrong headers exist:
      draft 363 carries XPIXSIZE=10.0 µm while the real IMX457 pitch is 3.76 µm —
      a pure sanity range would accept 10.0; only the DB cross-check rejects it.)
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


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    logger.warning(message)


# --------------------------------------------------------------------------- #
# Sanity ranges (lo, hi).  A value is valid iff lo < v <= hi and math.isfinite.
# These reject present-but-invalid headers (e.g. GAIN=0.0) but NOT
# plausible-but-wrong ones (e.g. pixel=10.0); the DB cross-check handles those.
# --------------------------------------------------------------------------- #
SANITY: dict[str, tuple[float, float]] = {
    "gain": (0.0, 100.0),            # e-/ADU
    "read_noise": (0.0, 200.0),      # e-
    "pixel_um": (0.0, 50.0),         # micron (physical pitch)
    "focal_mm": (1.0, 120000.0),     # mm
    "saturation": (1.0, 1.0e7),      # ADU
    "binning": (0.5, 16.0),          # integer 1..16
    "exptime": (0.0, 1.0e5),         # seconds
    "plate_scale": (0.1, 30.0),      # arcsec/px
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
    "elev": (-500.0, 9000.0),
}

# Header keyword variant families (first finite + in-range wins).
HEADER_KEYS: dict[str, tuple[str, ...]] = {
    "gain": ("EGAIN", "GAIN"),
    "read_noise": ("RDNOISE", "READNOISE", "RDNOISEE", "RN"),
    "pixel_um": ("XPIXSZ", "PIXSIZE1", "XPIXSIZE", "PIXSIZE", "PIXSZLX"),
    "focal_mm": ("FOCALLEN", "FOCLEN", "TELFOCA"),
    "saturation": ("SATURATE", "MAXLIN", "ESATUR", "LINLIMIT", "MAXADU", "DATAMAX"),
    "binning": ("XBINNING", "BINNING"),
    "exptime": ("EXPTIME", "EXPOSURE"),
    "lat": ("SITELAT", "OBSLAT", "OBSGEO-B"),
    "lon": ("SITELONG", "OBSLONG", "OBSGEO-L"),
    "elev": ("SITEELEV", "OBSELEV", "OBSGEO-H", "ALTITUDE"),
}

# Relative tolerance for header<->DB cross-check warnings (equipment-intrinsic).
CROSS_CHECK_RTOL = 0.05

# QHY driver GAIN header is a setting index, not e-/ADU.  Map (equipment_id, setting) -> e-/ADU.
# QHY294PROM read mode 0, gain setting 0 -> 3.17 e-/ADU (matches EQUIPMENTS.GAIN_ADU eq 1).
GAIN_SETTING_INDEX_MAP: dict[int, dict[int, float]] = {
    1: {0: 3.17},  # QHY294MM
}

_E_PER_ADU_COMMENT_MARKERS = ("e-/adu", "e/adu", "electron")
_GAIN_INDEX_COMMENTS = frozenset({"gain", "index", ""})


@dataclass
class Resolved:
    """Result of resolving one parameter."""

    value: Any = None
    source: str = "unresolved"      # header | header_index_mapped | db | config | default | unresolved
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


def _clamp(param: str, v: float) -> float:
    lo, hi = SANITY.get(param, (-math.inf, math.inf))
    return max(lo, min(hi, float(v)))


# --------------------------------------------------------------------------- #
# EQUIPMENT-INTRINSIC: DB-set (valid) -> header (fallback + cross-check) -> config
# --------------------------------------------------------------------------- #
def _resolve_equipment_intrinsic(
    param: str,
    *,
    header: Any,
    db_value: float | None,
    cfg_value: float | None,
    log_label: str,
) -> Resolved:
    hdr_v, hdr_key = _header_value(header, param)
    db_ok = _is_valid(param, db_value)
    res = Resolved()

    if db_ok:
        res.value = float(db_value)  # type: ignore[arg-type]
        res.source = "db"
        res.ok = True
        # Cross-check against a valid header value (DB still wins, but warn).
        if hdr_v is not None and not math.isclose(
            float(hdr_v), float(db_value), rel_tol=CROSS_CHECK_RTOL  # type: ignore[arg-type]
        ):
            msg = (
                f"[RESOLVE {log_label}] header {hdr_key}={hdr_v:g} disagrees with "
                f"DB={float(db_value):g} (>{CROSS_CHECK_RTOL:.0%}); using DB (equipment-intrinsic authority)"
            )
            res.warnings.append(msg)
            _warn_once(f"xcheck_{param}", msg)
    elif hdr_v is not None:
        res.value = float(hdr_v)
        res.source = "header"
        res.key = hdr_key
        res.ok = True
        _warn_once(
            f"hdrfallback_{param}",
            f"[RESOLVE {log_label}] DB value missing/invalid; falling back to header {hdr_key}={hdr_v:g}",
        )
    elif _is_valid(param, cfg_value):
        res.value = float(cfg_value)  # type: ignore[arg-type]
        res.source = "config"
        res.ok = True
    else:
        res.source = "unresolved"
        res.ok = False

    logger.debug("[RESOLVE %s] -> %s (source=%s, key=%s)", log_label, res.value, res.source, res.key)
    return res


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


def _equipment_gain_header_units_e_per_adu(equipment_id: int | None, cfg: Any) -> bool:
    """Optional per-camera override: header GAIN is always e-/ADU (not a setting index)."""
    if cfg is None or equipment_id is None:
        return False
    flag = getattr(cfg, "gain_header_units_e_per_adu", None)
    if isinstance(flag, dict):
        try:
            return bool(flag.get(int(equipment_id), False))
        except (TypeError, ValueError):
            return False
    return bool(flag)


def _is_gain_setting_index(
    header: Any,
    raw: float,
    comment: str,
    db_value: float | None,
    *,
    is_e_per_adu: bool,
) -> bool:
    """True when the header GAIN/EGAIN card is a driver setting index, not e-/ADU."""
    if is_e_per_adu:
        return False
    if _comment_indicates_gain_index(comment):
        return True
    if header is not None and "READMODE" in header:
        return True
    if _is_valid("gain", db_value) and not math.isclose(
        float(raw), float(db_value), rel_tol=CROSS_CHECK_RTOL
    ):
        # Integer 0..100 without e-/ADU semantics is implausible as e-/ADU for this camera.
        ri = int(round(float(raw)))
        if float(raw) == ri and 0 <= ri <= 100:
            return True
    return False


def _map_gain_setting_index(
    setting_index: int,
    equipment_id: int | None,
    header: Any,
) -> float | None:
    _ = header  # reserved for future read-mode keyed tables
    if equipment_id is not None:
        per_eq = GAIN_SETTING_INDEX_MAP.get(int(equipment_id), {})
        if setting_index in per_eq:
            return float(per_eq[setting_index])
    return None


def _resolve_gain_header_first(
    *,
    header: Any,
    db_value: float | None,
    cfg_value: float | None,
    equipment_id: int | None,
    cfg: Any,
) -> Resolved:
    """Header-first gain: e-/ADU card -> setting-index map -> DB -> config."""
    res = Resolved()
    db_ok = _is_valid("gain", db_value)
    raw, hdr_key, comment = _header_gain_raw(header)

    if raw is not None:
        force_e_per_adu = _equipment_gain_header_units_e_per_adu(equipment_id, cfg)
        is_e_per_adu = force_e_per_adu or _comment_indicates_e_per_adu(comment)
        is_index = _is_gain_setting_index(
            header, raw, comment, db_value, is_e_per_adu=is_e_per_adu
        )

        if is_e_per_adu and not is_index and _is_valid("gain", raw):
            res.value = float(raw)
            res.source = "header"
            res.key = hdr_key
            res.ok = True
            if db_ok and not math.isclose(float(raw), float(db_value), rel_tol=CROSS_CHECK_RTOL):
                msg = (
                    f"[RESOLVE gain] header {hdr_key}={raw:g} disagrees with "
                    f"DB={float(db_value):g} (>{CROSS_CHECK_RTOL:.0%}); using header (session truth)"
                )
                res.warnings.append(msg)
                _warn_once("xcheck_gain_header_wins", msg)
            logger.debug("[RESOLVE gain] -> %s (source=%s, key=%s)", res.value, res.source, res.key)
            return res

        if is_index or (not is_e_per_adu and _comment_indicates_gain_index(comment)):
            setting = int(round(float(raw)))
            mapped = _map_gain_setting_index(setting, equipment_id, header)
            if mapped is not None and _is_valid("gain", mapped):
                res.value = float(mapped)
                res.source = "header_index_mapped"
                res.key = hdr_key
                res.ok = True
                logger.debug("[RESOLVE gain] -> %s (source=%s, key=%s, index=%d)", res.value, res.source, res.key, setting)
                return res
            if db_ok:
                msg = (
                    f"[RESOLVE gain] header GAIN index {setting} not in gain map; using DB base "
                    f"{float(db_value):g} e-/ADU"
                )
                res.warnings.append(msg)
                _warn_once(f"gain_index_unmapped_{equipment_id}_{setting}", msg)
                res.value = float(db_value)  # type: ignore[arg-type]
                res.source = "db"
                res.ok = True
                logger.debug("[RESOLVE gain] -> %s (source=%s, unmapped index)", res.value, res.source)
                return res

    if db_ok:
        res.value = float(db_value)  # type: ignore[arg-type]
        res.source = "db"
        res.ok = True
        logger.debug("[RESOLVE gain] -> %s (source=%s)", res.value, res.source)
        return res

    if _is_valid("gain", cfg_value):
        res.value = float(cfg_value)  # type: ignore[arg-type]
        res.source = "config"
        res.ok = True
        logger.debug("[RESOLVE gain] -> %s (source=%s)", res.value, res.source)
        return res

    res.source = "unresolved"
    res.ok = False
    logger.debug("[RESOLVE gain] -> unresolved")
    return res


def _db_cosmic(db: Any, equipment_id: int | None) -> tuple[float | None, float | None]:
    if db is None or equipment_id is None:
        return None, None
    try:
        g, rn = db.get_equipment_cosmic_params(int(equipment_id))
        return g, rn
    except Exception:  # noqa: BLE001
        return None, None


def resolve_gain(
    header: Any = None,
    *,
    db: Any = None,
    equipment_id: int | None = None,
    cfg: Any = None,
    db_value: float | None = None,
) -> Resolved:
    if db_value is None:
        db_value, _ = _db_cosmic(db, equipment_id)
    cfg_v = getattr(cfg, "gain", None) if cfg is not None else None
    return _resolve_gain_header_first(
        header=header,
        db_value=db_value,
        cfg_value=cfg_v,
        equipment_id=equipment_id,
        cfg=cfg,
    )


def resolve_read_noise(
    header: Any = None,
    *,
    db: Any = None,
    equipment_id: int | None = None,
    cfg: Any = None,
    db_value: float | None = None,
) -> Resolved:
    if db_value is None:
        _, db_value = _db_cosmic(db, equipment_id)
    cfg_v = getattr(cfg, "read_noise", None) if cfg is not None else None
    return _resolve_equipment_intrinsic(
        "read_noise", header=header, db_value=db_value, cfg_value=cfg_v, log_label="read_noise"
    )


def resolve_pixel_um(
    header: Any = None,
    *,
    db: Any = None,
    equipment_id: int | None = None,
    cfg: Any = None,
    db_value: float | None = None,
) -> Resolved:
    if db_value is None and db is not None and equipment_id is not None:
        try:
            db_value = db.get_equipment_pixel_size_um(int(equipment_id))
        except Exception:  # noqa: BLE001
            db_value = None
    return _resolve_equipment_intrinsic(
        "pixel_um", header=header, db_value=db_value, cfg_value=None, log_label="pixel_um"
    )


def resolve_focal_mm(
    header: Any = None,
    *,
    db: Any = None,
    equipment_id: int | None = None,
    telescope_id: int | None = None,
    cfg: Any = None,
    db_value: float | None = None,
) -> Resolved:
    if db_value is None and db is not None:
        try:
            if equipment_id is not None:
                db_value = db.get_equipment_focal_mm(int(equipment_id))
        except Exception:  # noqa: BLE001
            db_value = None
        if (db_value is None or not _is_valid("focal_mm", db_value)) and telescope_id is not None:
            try:
                db_value = db.get_telescope_focal_mm(int(telescope_id))
            except Exception:  # noqa: BLE001
                pass
    # FOCALLEN sometimes carried in metres for fast scopes -> normalise.
    res = _resolve_equipment_intrinsic(
        "focal_mm", header=_focal_header_in_mm(header), db_value=db_value, cfg_value=None, log_label="focal_mm"
    )
    return res


def _focal_header_in_mm(header: Any) -> Any:
    """Return a shallow header-like view where FOCALLEN in metres is scaled to mm."""
    if header is None:
        return None
    try:
        v = header.get("FOCALLEN")
        if v is not None:
            f = float(v)
            if 0.0 < f < 25.0:  # metres (e.g. 1.2 -> 1200 mm); 25mm lens is the smallest realistic
                patched = dict(header)
                patched["FOCALLEN"] = f * 1000.0
                return patched
    except (TypeError, ValueError):
        pass
    return header


def resolve_saturation(
    header: Any = None,
    *,
    db: Any = None,
    equipment_id: int | None = None,
    cfg: Any = None,
    db_value: float | None = None,
) -> Resolved:
    if db_value is None and db is not None and equipment_id is not None:
        try:
            db_value = db.get_equipment_saturation_adu(int(equipment_id))
        except Exception:  # noqa: BLE001
            db_value = None
    cfg_v = getattr(cfg, "saturate_limit_adu", None) if cfg is not None else None
    return _resolve_equipment_intrinsic(
        "saturation", header=header, db_value=db_value, cfg_value=cfg_v, log_label="saturation"
    )


# --------------------------------------------------------------------------- #
# OBSERVATION-SPECIFIC: header (valid) -> DB -> config
# --------------------------------------------------------------------------- #
def _resolve_observation_specific(
    param: str,
    *,
    header: Any,
    db_value: float | None,
    cfg_value: float | None,
    log_label: str,
) -> Resolved:
    hdr_v, hdr_key = _header_value(header, param)
    res = Resolved()
    if hdr_v is not None:
        res.value, res.source, res.key, res.ok = float(hdr_v), "header", hdr_key, True
    elif _is_valid(param, db_value):
        res.value, res.source, res.ok = float(db_value), "db", True  # type: ignore[arg-type]
    elif _is_valid(param, cfg_value):
        res.value, res.source, res.ok = float(cfg_value), "config", True  # type: ignore[arg-type]
    else:
        res.source, res.ok = "unresolved", False
    logger.debug("[RESOLVE %s] -> %s (source=%s, key=%s)", log_label, res.value, res.source, res.key)
    return res


def resolve_binning(header: Any = None, *, db_value: float | None = None, cfg: Any = None) -> Resolved:
    return _resolve_observation_specific(
        "binning", header=header, db_value=db_value, cfg_value=None, log_label="binning"
    )


def resolve_exptime(header: Any = None, *, db_value: float | None = None, cfg: Any = None) -> Resolved:
    return _resolve_observation_specific(
        "exptime", header=header, db_value=db_value, cfg_value=None, log_label="exptime"
    )


# --------------------------------------------------------------------------- #
# SITE: per-draft ID_LOCATION -> header SITELAT/LONG/ELEV -> config (flagged)
# --------------------------------------------------------------------------- #
def _draft_location(db: Any, draft_id: int | None) -> tuple[float, float, float] | None:
    if db is None or draft_id is None:
        return None
    try:
        cur = db.conn.execute(
            """
            SELECT l.LATITUDE, l.LONGITUDE, l.ALTITUDE
            FROM OBS_DRAFT d JOIN LOCATION l ON l.ID = d.ID_LOCATION
            WHERE d.ID = ?
            """,
            (int(draft_id),),
        )
        row = cur.fetchone()
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    try:
        la, lo = float(row[0]), float(row[1])
        al = float(row[2]) if row[2] is not None and math.isfinite(float(row[2])) else 0.0
    except (TypeError, ValueError):
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
        res.lat, res.lon, res.elev = _clamp("lat", la), _clamp("lon", lo), _clamp("elev", el)
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
            _warn_once("xcheck_site", msg)
        logger.debug("[RESOLVE site] -> draft (%.4f,%.4f,%.0f)", res.lat, res.lon, res.elev or 0.0)
        return res

    hsv = _header_site(header)
    if hsv is not None:
        res.lat, res.lon, res.elev = _clamp("lat", hsv[0]), _clamp("lon", hsv[1]), _clamp("elev", hsv[2])
        res.source, res.ok = "header", True
        logger.debug("[RESOLVE site] -> header (%.4f,%.4f,%.0f)", res.lat, res.lon, res.elev or 0.0)
        return res

    if allow_config and cfg is not None:
        try:
            clat = float(getattr(cfg, "observer_lat", 0.0) or 0.0)
            clon = float(getattr(cfg, "observer_lon", 0.0) or 0.0)
            calt = float(getattr(cfg, "observer_alt_m", 0.0) or 0.0)
        except (TypeError, ValueError):
            clat = clon = calt = 0.0
        if clat != 0.0 or clon != 0.0:
            res.lat, res.lon, res.elev = _clamp("lat", clat), _clamp("lon", clon), _clamp("elev", calt)
            res.source, res.ok = "config", True
            _warn_once(
                "site_config_fallback",
                "[RESOLVE site] no per-draft ID_LOCATION and no header SITELAT/LONG; "
                "using config observer location as a FLAGGED fallback — verify it belongs to this session.",
            )
            logger.debug("[RESOLVE site] -> config (flagged) (%.4f,%.4f,%.0f)", res.lat, res.lon, res.elev or 0.0)
            return res

    res.source, res.ok = "unresolved", False
    _warn_once(
        "site_unresolved",
        "[RESOLVE site] observer site unresolved (no draft location, no header, no config) — "
        "BJD/HJD/airmass left empty; surface in poor-FITS prompt.",
    )
    return res
