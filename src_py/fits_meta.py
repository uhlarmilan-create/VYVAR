"""Moved from pipeline.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import logging
import math
import os
from astropy.io import fits
from astropy.time import Time, TimeDelta
from database import VyvarDatabase
from calib_library import header_gain, header_offset, header_readout_mode, header_temp
from infolog import log_event
from optics import focal_mm_plausible, header_native_pixel_um, optics_hint
from utils import fits_binning_xy_from_header
import numbers
import re

import astropy.units as u
from astropy.coordinates import Angle

def _header_pick(header: Any, *keys: str, default: Any = None) -> Any:
    """First non-empty value of ``keys`` in a header (the one header-pick helper of this module)."""
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return default


def normalize_filter_name(text: Any) -> str:
    """FITS FILTER as used in observation-group keys and calibration matching (G02-42: the one
    normaliser): stripped; empty / unknown / none / nan -> ``NoFilter``."""
    t = str(text or "").strip()
    if not t or t.lower() in {"unknown", "none", "nan"}:
        return "NoFilter"
    return t


def observation_group_key(filter_name: Any, exposure_s: Any, binning: Any) -> str:
    """Observation subgroup key ``<FILTER>|<EXPTIME:g>|<XBINNING>`` - the ONE builder (G02-42).

    Used by the importer (manifest / VY_OBSG), the calibration step and the metadata helpers.
    """
    flt = normalize_filter_name(filter_name)
    try:
        e = float(exposure_s)
    except (TypeError, ValueError):
        e = 0.0
    b = max(1, int(binning or 1))
    return f"{flt}|{e:g}|{b}"


def observation_group_key_from_metadata(meta: dict[str, Any]) -> str:
    """:func:`observation_group_key` of an ``extract_fits_metadata`` dict."""
    return observation_group_key(meta.get("filter"), meta.get("exposure", 0.0), meta.get("binning", 1))


def _summarize_lights_binning_from_headers(paths: list[Path]) -> dict[str, Any]:
    """Count ``(XBINNING,YBINNING)`` read directly from each primary FITS header."""
    counts: dict[tuple[int, int], int] = {}
    samples: dict[tuple[int, int], Path] = {}
    errors = 0
    for fp in paths:
        try:
            with fits.open(fp, memmap=False) as hdul:
                xb, yb = fits_binning_xy_from_header(hdul[0].header)
            key = (int(xb), int(yb))
            counts[key] = counts.get(key, 0) + 1
            samples.setdefault(key, fp)
        except Exception:  # noqa: BLE001
            errors += 1
    return {"counts": counts, "samples": samples, "errors": errors}

def log_lights_binning_from_headers_preflight(
    paths: list[Path],
    *,
    context: str = "VYVAR",
) -> tuple[int, int] | None:
    """Log binning from FITS headers at pipeline start (before DB/cache overlay)."""
    if not paths:
        return None
    summary = _summarize_lights_binning_from_headers(paths)
    counts: dict[tuple[int, int], int] = summary.get("counts") or {}
    if not counts:
        if int(summary.get("errors") or 0) > 0:
            log_event(f"{context} - binning from headers: could not read any FITS.")
        return None
    log_event(f"{context} - binning from FITS headers ({len(paths)} files):")
    samples: dict[tuple[int, int], Path] = summary.get("samples") or {}
    for (xb, yb), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        sample = samples.get((xb, yb))
        sn = sample.name if sample is not None else "?"
        log_event(f"  {xb}x{yb}: {n}x (example: {sn})")
    if len(counts) > 1:
        log_event(
            f"{context} - WARNING: mixed binning modes in the batch - "
            "calibration / master match use the binning from each file's header."
        )
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    return best_key

# ---------------------------------------------------------------------------
# exposure start time - the ONE authority (G02-01, G02-16, G01-05, G02-49)
# ---------------------------------------------------------------------------
_TIME_LOG = logging.getLogger("time_utils")
_WARNED_ONCE: set[str] = set()

# FITS TIMESYS values (FITS 4.0 sec. 9.2.1) -> astropy scale. GPS is TAI - 19 s (handled below).
_TIMESYS_SCALES: dict[str, str] = {
    "UTC": "utc",
    "UT": "utc",
    "TT": "tt",
    "TDT": "tt",
    "TAI": "tai",
    "TDB": "tdb",
    "GPS": "tai",
}
_GPS_MINUS_TAI_S = -19.0

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_TIME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ]+(?P<time>\d{1,2}:\d{2}(?::\d{2}(?:\.\d*)?)?)\s*"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?$",
    re.IGNORECASE,
)


def _timesys_scale(header: fits.Header) -> tuple[str, float] | None:
    """(astropy scale, extra offset seconds) for the header TIMESYS; UTC when absent; None if unknown."""
    raw = header.get("TIMESYS")
    if raw is None or not str(raw).strip():
        return "utc", 0.0
    key = str(raw).strip().upper()
    scale = _TIMESYS_SCALES.get(key)
    if scale is None:
        return None
    return scale, (_GPS_MINUS_TAI_S if key == "GPS" else 0.0)


def _isot_with_offset(date_part: str, time_part: str, tz: str | None, scale: str) -> Time:
    """Time from ISO date + time; a numeric UTC offset (``+01:00``) is removed (``Z`` = +00:00)."""
    t = Time(f"{date_part}T{time_part}", format="isot", scale=scale)
    if tz and tz.upper() != "Z":
        sign = 1.0 if tz[0] == "+" else -1.0
        digits = tz[1:].replace(":", "")
        offset_s = sign * (int(digits[:2]) * 3600.0 + int(digits[2:4]) * 60.0)
        t = t - TimeDelta(offset_s * u.s)
    return t


def exposure_start_utc(header: fits.Header) -> Time | None:
    """Shutter-open instant (UTC) of a frame: the ONE authority for exposure start (G02-01, G02-16).

    Order:

    1. ``DATE-OBS`` with a time part: ISO with ``T`` or a space separator, optional fraction,
       ``Z`` or a numeric UTC offset (``+00:00``, ``+0100``).
    2. ``DATE-OBS`` date only + ``TIME-OBS`` / ``TIMEOBS`` / ``TIME``.
    3. ``MJD-OBS``, then ``JD-OBS`` / ``JD`` (taken as exposure start, like ``DATE-OBS``).
    4. ``DATE-OBS`` date only and nothing else: midnight, with an error log and the
       ``timeobs_parse_fallback`` counter (never silently).

    ``TIMESYS`` (UTC, UT, TT, TDT, TAI, TDB, GPS) is honoured; absent means UTC. An unknown TIMESYS
    or a ``DATE-OBS`` that carries a time but does not parse is logged and counted
    (``date_obs_parse_fail``) and the frame gets no time (None) unless an (M)JD keyword exists.
    The jd_mid of a frame is this instant + EXPTIME / 2 (:func:`time_utils.mid_exposure_jd`).
    """
    from except_fix_counters import get_except_fix_counters  # noqa: PLC0415

    ts = _timesys_scale(header)
    if ts is None:
        get_except_fix_counters().date_obs_parse_fail += 1
        _TIME_LOG.error("[TIME] unknown TIMESYS=%r: frame has no exposure time", header.get("TIMESYS"))
        return None
    scale, extra_s = ts

    def _to_utc(t: Time) -> Time:
        if extra_s:
            t = t + TimeDelta(extra_s * u.s)
        return t.utc

    raw = _header_pick(header, "DATE-OBS", "DATEOBS")
    s = str(raw).strip() if raw is not None else ""
    date_only: str | None = None
    if s:
        m = _DATE_TIME_RE.match(s)
        if m is not None:
            try:
                return _to_utc(_isot_with_offset(m.group("date"), m.group("time"), m.group("tz"), scale))
            except Exception as exc:  # noqa: BLE001
                get_except_fix_counters().date_obs_parse_fail += 1
                _TIME_LOG.error("[TIME] DATE-OBS=%r does not parse: %s", s, exc)
        elif _DATE_ONLY_RE.match(s):
            date_only = s
        else:
            get_except_fix_counters().date_obs_parse_fail += 1
            _TIME_LOG.error("[TIME] DATE-OBS=%r is not an ISO date / date-time", s)

    if date_only is not None:
        to_raw = _header_pick(header, "TIME-OBS", "TIMEOBS", "TIME")
        to = str(to_raw).strip().replace(" ", "") if to_raw is not None else ""
        if to:
            try:
                return _to_utc(Time(f"{date_only}T{to}", format="isot", scale=scale))
            except Exception as exc:  # noqa: BLE001
                get_except_fix_counters().timeobs_parse_fallback += 1
                _TIME_LOG.error(
                    "[TIME] TIME-OBS parse failed DATE-OBS=%r TIME-OBS=%r; "
                    "jd will be DATE-only (midnight) for this frame: %s",
                    s,
                    to_raw,
                    exc,
                )
                return _to_utc(Time(f"{date_only}T00:00:00", format="isot", scale=scale))

    mjd = _finite_header_float(_header_pick(header, "MJD-OBS", "MJD_OBS"))
    if mjd is not None:
        return _to_utc(Time(mjd, format="mjd", scale=scale))
    jd = _finite_header_float(_header_pick(header, "JD-OBS", "JD_OBS", "JD"))
    if jd is not None:
        return _to_utc(Time(jd, format="jd", scale=scale))

    if date_only is not None:
        get_except_fix_counters().timeobs_parse_fallback += 1
        _TIME_LOG.error(
            "[TIME] DATE-OBS=%r has no time part and no TIME-OBS / MJD-OBS / JD: "
            "jd will be DATE-only (midnight) for this frame",
            s,
        )
        return _to_utc(Time(f"{date_only}T00:00:00", format="isot", scale=scale))
    if not s and "date_obs_missing" not in _WARNED_ONCE:
        _WARNED_ONCE.add("date_obs_missing")
        log_event("VYVAR time: DATE-OBS missing - jd_mid / HJD / BJD columns unavailable for affected frames.")
    return None


def _finite_header_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def generate_observation_hash(db: VyvarDatabase, draft_id: int) -> str:
    """Deterministic processing hashtag: camera (equipment) + telescope + filter/exptime set + JD start.

    JD start is the minimum finite ``INSPECTION_JD`` among draft lights, else ``draft manifest.OBSERVATIONSTARTJD``.
    Filter/exptime signature is sorted unique ``(FILTER, EXPTIME)`` pairs from ``manifest files[]`` lights.
    """
    import hashlib

    drow = db.fetch_obs_draft_by_id(int(draft_id))
    if drow is None:
        raise ValueError(f"Draft id={int(draft_id)} not found")
    id_eq = int(drow.get("ID_EQUIPMENTS") or 0)
    id_tel = int(drow.get("ID_TELESCOPE") or 0)
    lights = db.fetch_draft_light_rows_for_quality(int(draft_id))
    pair_set: set[tuple[str, float]] = set()
    jds: list[float] = []
    for r in lights:
        flt = str(r.get("FILTER") or "").strip() or "None"
        ex_raw = r.get("EXPTIME")
        try:
            ex_f = float(ex_raw) if ex_raw is not None else 0.0
        except (TypeError, ValueError):
            ex_f = 0.0
        if math.isfinite(ex_f):
            pair_set.add((flt, round(ex_f, 6)))
        jd_raw = r.get("INSPECTION_JD")
        try:
            jf = float(jd_raw) if jd_raw is not None else float("nan")
            if math.isfinite(jf):
                jds.append(jf)
        except (TypeError, ValueError):
            pass
    pairs_sorted = "|".join(f"{a}:{b:.6f}" for a, b in sorted(pair_set))
    try:
        jd0 = float(drow.get("OBSERVATIONSTARTJD") or 0.0)
    except (TypeError, ValueError):
        jd0 = 0.0
    if jds:
        jd0 = min(jds)
    if not math.isfinite(jd0):
        jd0 = 0.0
    payload = f"{id_eq}|{id_tel}|{pairs_sorted}|{jd0:.8f}"
    digest = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    date_prefix = VyvarDatabase._jd_to_yyyymmdd(jd0)
    return f"{date_prefix}_{digest}"

def _merge_equipment_pixel_into_metadata(meta: dict[str, Any], db: VyvarDatabase, equipment_id: int) -> None:
    """If FITS native pixel is missing or nonsense, use ``EQUIPMENTS.PIXELSIZE`` [um, 1x1] x binning."""
    try:
        native = db.get_equipment_pixel_size_um(int(equipment_id))
    except Exception:  # noqa: BLE001
        return
    if native is None:
        return
    try:
        nv = float(native)
    except (TypeError, ValueError):
        return
    if not math.isfinite(nv) or nv <= 0 or nv > 300.0:
        return
    x_bin = max(1, int(meta.get("binning", 1) or 1))
    y_bin = max(1, int(meta.get("binning_y", x_bin) or x_bin))
    prev = meta.get("pixel_size_um_physical")
    # Universal scale: trust EQUIPMENTS.PIXELSIZE (UI/DB) whenever equipment is known; binning stays from FITS.
    meta["pixel_size_um_physical"] = float(nv)
    eff_x = float(nv) * float(x_bin)
    eff_y = float(nv) * float(y_bin)
    meta["pixel_size_um_header"] = (eff_x + eff_y) / 2.0
    meta["effective_pixel_um_plate_scale"] = float(nv) * float(x_bin)
    meta["pixel_size_um_source"] = "equipment_db"
    _eff = float(nv) * float(x_bin)
    if prev is not None:
        try:
            pv = float(prev)
            same = math.isfinite(pv) and abs(pv - float(nv)) < 1e-6
        except (TypeError, ValueError):
            same = False
        if not same:
            log_event(
                f"PIXEL: EQUIPMENTS.PIXELSIZE={nv:g} um x binning {x_bin}x{y_bin} -> effective {_eff:g} um "
                f"(preferred over FITS; previously {prev!r})."
            )
    else:
        log_event(
            f"PIXEL: EQUIPMENTS.PIXELSIZE={nv:g} um x binning {x_bin}x{y_bin} -> effective {_eff:g} um."
        )

def _recompute_effective_pixel_from_physical(meta: dict[str, Any]) -> None:
    """``effective_pixel_um_plate_scale`` = native [um] x XBINNING (int)."""
    phys = meta.get("pixel_size_um_physical")
    if phys is None:
        return
    try:
        p = float(phys)
    except (TypeError, ValueError):
        return
    if not math.isfinite(p) or p <= 0:
        return
    x_bin = max(1, int(meta.get("binning", 1) or 1))
    y_bin = max(1, int(meta.get("binning_y", x_bin) or x_bin))
    eff_x = p * float(x_bin)
    eff_y = p * float(y_bin)
    meta["pixel_size_um_header"] = (eff_x + eff_y) / 2.0
    meta["effective_pixel_um_plate_scale"] = float(p) * float(x_bin)

def _enrich_calibration_metadata_from_header(
    meta: dict[str, Any],
    header: fits.Header,
    *,
    db: VyvarDatabase | None,
    id_telescope: int | None = None,
) -> None:
    """Add ``focal_length``, ``pixel_size_raw``, ``pixel_um``, ``focal_length_source`` for diagnostics / UI."""

    def _to_f(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    r1 = _to_f(_header_pick(header, "PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0))
    r2 = _to_f(_header_pick(header, "PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0))
    parts: list[str] = []
    if r1 > 0:
        parts.append(f"PIX1={r1:g}")
    if r2 > 0:
        parts.append(f"PIX2={r2:g}")
    meta["pixel_size_raw"] = " | ".join(parts) if parts else "n/a"

    meta["pixel_um"] = meta.get("effective_pixel_um_plate_scale")

    # T08 optics module: TELESCOPE.FOCAL of the draft telescope -> header FOCALLEN (no typo rewrite,
    # no "first active telescope" guess without a telescope id)
    hint = optics_hint(header, db, telescope_id=id_telescope)
    meta["focal_length"] = hint.focal_mm
    meta["focal_length_source"] = hint.focal_source


def _apply_draft_combined_to_pipeline_meta(meta: dict[str, Any], comb: dict[str, Any]) -> None:
    """Overlay ``VyvarDatabase.get_combined_metadata`` onto ``extract_fits_metadata`` output."""
    fl = comb.get("focal_length_mm")
    if fl is not None:
        try:
            fv = float(fl)
            if focal_mm_plausible(fv):
                meta["focal_length"] = fv
                meta["focal_length_source"] = str(comb.get("focal_source") or "draft_combined")
        except (TypeError, ValueError):
            pass
    pn = comb.get("pixel_native_um")
    if pn is not None:
        try:
            pv = float(pn)
            if math.isfinite(pv) and 0 < pv <= 300.0:
                meta["pixel_size_um_physical"] = pv
        except (TypeError, ValueError):
            pass
    xb = max(1, int(comb.get("xbinning", 1) or 1))
    yb = max(1, int(comb.get("ybinning", xb) or xb))
    meta["binning"] = xb
    meta["binning_y"] = yb
    pe = comb.get("pixel_effective_um")
    if pe is not None:
        try:
            ev = float(pe)
            if math.isfinite(ev) and ev > 0:
                meta["effective_pixel_um_plate_scale"] = ev
                if meta.get("pixel_size_um_physical") is not None:
                    pph = float(meta["pixel_size_um_physical"])
                    meta["pixel_size_um_header"] = (pph * float(xb) + pph * float(yb)) / 2.0
                else:
                    meta["pixel_size_um_header"] = ev
                meta["pixel_um"] = ev
        except (TypeError, ValueError):
            pass
    sat = comb.get("saturate_adu")
    if sat is not None:
        try:
            sv = float(sat)
            if math.isfinite(sv) and sv > 0:
                meta["equipment_saturate_adu"] = sv
        except (TypeError, ValueError):
            pass

def _parse_fits_binning_int(raw: Any, default: int = 1) -> int:
    from utils import parse_fits_binning_int

    return parse_fits_binning_int(raw, default)

def _log_effective_pixel_pitch(meta: dict[str, Any], *, filepath: str = "") -> None:
    """Infolog: physical pixel, binning, effective pixel for plate-scale style calculations."""
    pixsz_from_header_or_db = meta.get("pixel_size_um_physical")
    binning_x = max(1, int(meta.get("binning", 1) or 1))
    binning_y = max(1, int(meta.get("binning_y", binning_x) or binning_x))
    effective_pixsz = meta.get("effective_pixel_um_plate_scale")
    tail = f" - {filepath}" if filepath else ""
    try:
        ps_s = f"{float(pixsz_from_header_or_db):.4g}" if pixsz_from_header_or_db is not None else "n/a"
        eff_s = f"{float(effective_pixsz):.4g}" if effective_pixsz is not None else "n/a"
        log_event(
            f"DEBUG: Physical pixel: {ps_s} um | Binning: {binning_x}x{binning_y} | "
            f"EFFECTIVE PIXEL FOR COMPUTATION: {eff_s} um{tail}"
        )
    except (TypeError, ValueError):
        pass

def fits_metadata_from_primary_header(
    header: fits.Header,
    *,
    force_physical_pixel_um: float | None = None,
) -> dict[str, Any]:
    """Build the same dict as :func:`extract_fits_metadata` from an already-loaded primary header.

    ``pixel_size_um_header`` is the **effective** on-sky pitch [um] (physical pixel from header x binning).
    ``pixel_size_um_physical`` is the native pitch before binning (after optional ``force_physical_pixel_um``).
    ``effective_pixel_um_plate_scale`` is native x XBINNING [um] for plate scale / solver hints.
    """

    # G02-16: the one exposure-start authority
    _t_start = exposure_start_utc(header)
    # 0.0 = no time (DB column); exposure_start_utc has already logged and counted the reason
    jd_start = float(_t_start.jd) if _t_start is not None else 0.0

    _raw_xb = _header_pick(header, "XBINNING", "BINNING", default=1)
    _raw_yb = _header_pick(header, "YBINNING", default=_raw_xb)
    x_bin = _parse_fits_binning_int(_raw_xb, 1)
    y_bin = _parse_fits_binning_int(_raw_yb, x_bin)

    # native (1x1) pitch: optics convention (XPIXSZ is the binned pitch, PIXSIZE1 the native one)
    _nx, _ny = header_native_pixel_um(header)
    _ps1 = float(_nx) if _nx is not None else 0.0
    _ps2 = float(_ny) if _ny is not None else 0.0

    _force = None
    if force_physical_pixel_um is not None:
        try:
            fv = float(force_physical_pixel_um)
            if math.isfinite(fv) and fv > 0:
                _force = fv
        except (TypeError, ValueError):
            _force = None
    if _force is not None:
        _ps1, _ps2 = _force, _force

    _physical_x = _ps1 if _ps1 > 0 else None
    _physical_y = _ps2 if _ps2 > 0 else None
    if _physical_x is not None and _physical_y is not None:
        _physical_mean = (_physical_x + _physical_y) / 2.0
    elif _physical_x is not None:
        _physical_mean = float(_physical_x)
    elif _physical_y is not None:
        _physical_mean = float(_physical_y)
    else:
        _physical_mean = None

    _eff_x = (_physical_x * float(x_bin)) if _physical_x is not None else None
    _eff_y = (_physical_y * float(y_bin)) if _physical_y is not None else None
    if _eff_x is not None and _eff_y is not None:
        _effective_um = (_eff_x + _eff_y) / 2.0
    elif _eff_x is not None:
        _effective_um = float(_eff_x)
    elif _eff_y is not None:
        _effective_um = float(_eff_y)
    else:
        _effective_um = None

    if _physical_mean is not None:
        _plate_eff = float(_physical_mean) * float(x_bin)
    else:
        _plate_eff = _effective_um

    _centre = field_centre_radec(header)
    return {
        "exposure": float(_header_pick(header, "EXPTIME", "EXPOSURE", default=0.0)),
        "filter": str(_header_pick(header, "FILTER", "FILT", default="NoFilter")),
        "binning": int(x_bin),
        "binning_y": int(y_bin),
        "fits_xbinning_raw": _raw_xb,
        "fits_ybinning_raw": _raw_yb,
        "naxis1": int(header.get("NAXIS1", 0) or 0),
        "naxis2": int(header.get("NAXIS2", 0) or 0),
        "pixel_size_um_physical": _physical_mean,
        "pixel_size_um_header": _effective_um,
        "effective_pixel_um_plate_scale": _plate_eff,
        "temp": header_temp(header),  # G01-08: None when unknown (one reader, calib_library)
        "gain": header_gain(header),  # G01-10 / G02-40: exact float, one reader (calib_library)
        "offset": header_offset(header),  # G01-09: camera settings of the mixed-settings check
        "readout_mode": header_readout_mode(header),
        # G02-17: the one field-centre order (field_centre_radec); 0.0 when nothing resolves (DB column)
        "ra": float(_centre[0]) if _centre[0] is not None else 0.0,
        "dec": float(_centre[1]) if _centre[1] is not None else 0.0,
        "jd_start": jd_start,
        "telescope": _header_pick(header, "TELESCOP", "SCOPE", default=None),
        "camera": _header_pick(header, "INSTRUME", "CAMERA", default=None),
        "bayerpat": _valid_bayerpat_from_header(header),
    }

def _valid_bayerpat_from_header(header: fits.Header) -> str | None:
    from osc_extract import valid_bayer_pattern_4

    return valid_bayer_pattern_4(str(header.get("BAYERPAT") or ""))

def extract_fits_metadata(
    filepath: Path | str,
    *,
    db: VyvarDatabase | None = None,
    force_physical_pixel_um: float | None = None,
    id_equipment: int | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Extract key metadata from FITS primary header.

    When ``db`` is set, uses ``FITS_HEADER_CACHE`` when ``FILE_PATH``, ``FILE_SIZE``, and ``MTIME_NS`` match
    the file on disk; otherwise reads the FITS header and refreshes the cache row.

    ``ra`` / ``dec`` are in decimal degrees. Sources (first match wins): OBJCTRA/OBJCTDEC,
    RA_OBJ/DEC_OBJ, TARGRA/TARGDEC, CENTRA/CENTDEC, RA/DEC, RAJ2000/DEJ2000, CAT-RA/CAT-DEC,
    CRVAL1/CRVAL2.
    Sexagesimal strings are accepted with colons or spaces (e.g. ``' 3 39 06.45'`` for RA).

    Returned keys:
    - exposure
    - filter
    - binning (XBINNING / BINNING)
    - binning_y (YBINNING, default same as binning)
    - naxis1, naxis2
    - pixel_size_um_physical (native pitch from header / DB merge)
    - pixel_size_um_header (**effective** pitch [um] = physical x binning for plate solve / WCS)
    - effective_pixel_um_plate_scale (native mean x XBINNING; solver / plate scale)
    - focal_length [mm]: ``FOCALLEN`` in header when present; else ``TELESCOPE.FOCAL``
    - focal_length_source, pixel_size_raw (raw numbers from the header), pixel_um (= effective pixel)
    - temp
    - ra
    - dec
    - jd_start
    - telescope
    - camera
    - bayerpat (FITS BAYERPAT when present, e.g. RGGB)

    Convention:     ``EQUIPMENTS.PIXELSIZE`` in the DB is the **native 1x1** pitch [um]; FITS cache
    ``PIXEL_UM`` stores **effective** pitch after binning.

    When ``draft_id`` is set and ``db`` is given, :meth:`VyvarDatabase.get_combined_metadata` overlays
    FITS+SQL focal/pixel (``XBINNING``-strict effective pixel) and ``EQUIPMENTS.SATURATE_ADU``.
    """
    fp = Path(filepath)
    id_telescope: int | None = None
    if db is not None and draft_id is not None:
        try:
            dr = db.fetch_obs_draft_by_id(int(draft_id)) or {}
            if dr.get("ID_TELESCOPE") is not None:
                id_telescope = int(dr["ID_TELESCOPE"])
        except Exception:  # noqa: BLE001
            id_telescope = None
    st: os.stat_result | None = None
    try:
        st = fp.stat()
    except OSError:
        st = None

    if db is not None and st is not None:
        cached = db.fits_header_cache_try_meta(fp)
        if cached is not None:
            meta = dict(cached)
            if id_equipment is not None:
                _merge_equipment_pixel_into_metadata(meta, db, int(id_equipment))
            _recompute_effective_pixel_from_physical(meta)
            with fits.open(fp, memmap=False) as hdul:
                _enrich_calibration_metadata_from_header(
                    meta,
                    hdul[0].header,
                    db=db,
                    id_telescope=id_telescope,
                )
            if draft_id is not None:
                comb = db.get_combined_metadata(fp, int(draft_id))
                _apply_draft_combined_to_pipeline_meta(meta, comb)
            return meta

    _fpu: float | None = None
    if force_physical_pixel_um is not None:
        try:
            v = float(force_physical_pixel_um)
            if v > 0 and math.isfinite(v):
                _fpu = v
        except (TypeError, ValueError):
            _fpu = None
    with fits.open(fp, memmap=False) as hdul:
        header = hdul[0].header
    meta = fits_metadata_from_primary_header(header, force_physical_pixel_um=_fpu)
    if db is not None and id_equipment is not None:
        _merge_equipment_pixel_into_metadata(meta, db, int(id_equipment))
    _recompute_effective_pixel_from_physical(meta)
    _enrich_calibration_metadata_from_header(
        meta,
        header,
        db=db,
        id_telescope=id_telescope,
    )
    if draft_id is not None and db is not None:
        comb = db.get_combined_metadata(fp, int(draft_id))
        _apply_draft_combined_to_pipeline_meta(meta, comb)
    _log_effective_pixel_pitch(meta, filepath=str(fp.name))

    if db is not None and st is not None:
        imagetyp = str(header.get("IMAGETYP") or header.get("FRAME") or header.get("IMTYPE") or "")
        do = header.get("DATE-OBS") or header.get("DATEOBS")
        date_obs = None if do in (None, "") else str(do)
        db.fits_header_cache_upsert_one(
            fp,
            file_size=int(st.st_size),
            mtime_ns=int(st.st_mtime_ns),
            meta=meta,
            imagetyp=imagetyp,
            date_obs=date_obs,
        )

    return meta


# ---------------------------------------------------------------------------
# pointing from the FITS header (moved from vyvar_platesolver, BLIND-SOLVE-2026-10)
# ---------------------------------------------------------------------------
_DECIMAL_HDR_RE = re.compile(r"^[-+]?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$")


def parse_header_ra_deg(value: Any) -> float | None:
    """RA in degrees from a header value: degrees, or HMS strings."""
    if value is None:
        return None
    if isinstance(value, numbers.Real):
        x = float(value)
        return x if math.isfinite(x) else None
    s = str(value).strip()
    if not s or s.upper() in {"NAN", "NONE"}:
        return None
    if _DECIMAL_HDR_RE.fullmatch(s):
        try:
            x = float(s)
            return x if math.isfinite(x) else None
        except ValueError:
            return None
    try:
        x = float(Angle(s, unit=u.hourangle).degree)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        pass
    try:
        x = float(Angle(s, unit=u.deg).degree)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def parse_header_dec_deg(value: Any) -> float | None:
    """Dec in degrees from a header value: degrees, or DMS strings."""
    if value is None:
        return None
    if isinstance(value, numbers.Real):
        x = float(value)
        return x if math.isfinite(x) else None
    s = str(value).strip()
    if not s or s.upper() in {"NAN", "NONE"}:
        return None
    if _DECIMAL_HDR_RE.fullmatch(s):
        try:
            x = float(s)
            return x if math.isfinite(x) else None
        except ValueError:
            return None
    try:
        x = float(Angle(s, unit=u.deg).degree)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


# G02-17: (RA key, Dec key) pairs of capture software, in this order; a pair is used only when both
# values parse (decimal degrees or sexagesimal strings - capture apps often write "HH MM SS" under "RA").
_POINTING_KEY_PAIRS: tuple[tuple[str, str], ...] = (
    ("RA", "DEC"),
    ("OBJRA", "OBJDEC"),
    ("TELRA", "TELDEC"),
    ("MNT_RA", "MNT_DEC"),
    ("MOUNT_RA", "MOUNT_DEC"),
    ("RA_OBJ", "DEC_OBJ"),
    ("TARGRA", "TARGDEC"),
    ("CENTRA", "CENTDEC"),
    ("RAJ2000", "DEJ2000"),
    ("CAT-RA", "CAT-DEC"),
)


def pointing_hint_from_header(header: fits.Header) -> tuple[float | None, float | None, str]:
    """Field-centre RA/Dec in degrees from the pointing keywords of a header (no WCS solution used).

    The hint chain of :func:`field_centre_radec` and the plate-solve hint (before a solution exists).
    Order: ``VYTARGRA/VYTARGDE`` (``VY_TARG``) -> the capture-software pairs of
    ``_POINTING_KEY_PAIRS`` (``RA_DEC``, sexagesimal accepted) -> ``OBJCTRA/OBJCTDEC``
    (``OBJCTRA_DEC_hms``) -> ``CRVAL1/2`` (``CRVAL``, weak). ``(None, None, "none")`` otherwise.
    """
    vy1 = header.get("VYTARGRA")
    vy2 = header.get("VYTARGDE")
    if vy1 is not None and vy2 is not None:
        r = parse_header_ra_deg(vy1)
        d = parse_header_dec_deg(vy2)
        if r is not None and d is not None:
            return r, d, "VY_TARG"
    for kra, kde in _POINTING_KEY_PAIRS:
        ra = parse_header_ra_deg(_header_pick(header, kra))
        dec = parse_header_dec_deg(_header_pick(header, kde))
        if ra is not None and dec is not None:
            return ra, dec, "RA_DEC"
    ra = parse_header_ra_deg(_header_pick(header, "OBJCTRA"))
    dec = parse_header_dec_deg(_header_pick(header, "OBJCTDEC"))
    if ra is not None and dec is not None:
        return ra, dec, "OBJCTRA_DEC_hms"
    ra = _finite_float(header.get("CRVAL1"))
    dec = _finite_float(header.get("CRVAL2"))
    if ra is not None and dec is not None:
        return ra, dec, "CRVAL"
    return None, None, "none"


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _wcs_centre_radec(header: fits.Header) -> tuple[float, float] | None:
    """Sky position of the image centre pixel of a solved celestial WCS (None without one)."""
    import warnings  # noqa: PLC0415

    from astropy.wcs import WCS, FITSFixedWarning  # noqa: PLC0415

    if not any(str(header.get(f"CTYPE{i}", "")).upper().startswith("RA") for i in (1, 2)):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        w = WCS(header).celestial
    if not w.has_celestial:
        return None
    nx = _finite_float(header.get("NAXIS1"))
    ny = _finite_float(header.get("NAXIS2"))
    if nx is not None and ny is not None and nx > 0 and ny > 0:
        sky = w.pixel_to_world_values((nx - 1.0) / 2.0, (ny - 1.0) / 2.0)
        ra, dec = float(sky[0]) % 360.0, float(sky[1])
    else:
        ra, dec = float(w.wcs.crval[0]) % 360.0, float(w.wcs.crval[1])
    if math.isfinite(ra) and math.isfinite(dec):
        return ra, dec
    return None


def _draft_centre_radec(db: Any, draft_id: int | None, header: fits.Header) -> tuple[float, float] | None:
    """The manifest centre of the frame's own setup (``center_by_setup``; ``frame_setup_name``
    of the header, ``setup_centre``); 0/0 counts as unset (S6-FIX CENTRE-PER-SETUP)."""
    if db is None or draft_id is None or not hasattr(db, "fetch_obs_draft_by_id"):
        return None
    try:
        row = db.fetch_obs_draft_by_id(int(draft_id))
    except Exception as exc:  # noqa: BLE001
        log_event(f"WARNING: field centre: draft {draft_id} manifest not read ({exc}) - draft centre unavailable")
        return None
    if not row:
        return None

    return setup_centre(row.get("CENTER_BY_SETUP"), frame_setup_name(header))


def field_centre_radec(
    header: fits.Header,
    *,
    db: Any = None,
    draft_id: int | None = None,
) -> tuple[float | None, float | None, str]:
    """The field centre of a frame (ICRS degrees) - the one authority (G02-17).

    Order (first that resolves):

    1. ``WCS``  - sky position of the centre pixel of a solved celestial WCS in the header
       (aligned frames, MASTERSTAR);
    2. the pointing keywords of :func:`pointing_hint_from_header`: ``VY_TARG`` -> ``RA_DEC``
       (capture-software pairs, decimal or sexagesimal) -> ``OBJCTRA_DEC_hms`` -> ``CRVAL``;
    3. ``DRAFT_CENTRE`` - the manifest centre of the frame's setup (``center_by_setup``; ``db`` + ``draft_id``).

    Returns ``(ra, dec, source)``; ``(None, None, "none")`` when nothing resolves. Used by the
    frame time columns (HJD/BJD of the field centre), the frame airmass, the lunar context and
    the raw-frame metadata.
    """
    wc = _wcs_centre_radec(header)
    if wc is not None:
        return wc[0], wc[1], "WCS"
    ra, dec, src = pointing_hint_from_header(header)
    if ra is not None and dec is not None:
        return ra, dec, src
    dc = _draft_centre_radec(db, draft_id, header)
    if dc is not None:
        return dc[0], dc[1], "DRAFT_CENTRE"
    return None, None, "none"


# --- moved from importer (LAYER-2026-10) ---


def observation_group_folder_name(group_key: str) -> str:
    """Filesystem-safe folder under ``lights`` (replaces ``|``)."""
    s = group_key.replace("|", "_").replace("/", "_").replace("\\", "_")
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in s)
    return s[:120] if len(s) > 120 else s


def setup_centre(centres: Mapping[str, Any] | None, setup: str | None) -> tuple[float, float] | None:
    """``(ra_deg, dec_deg)`` of ``setup`` in a ``center_by_setup`` map; None when the setup has none (0/0 = unset).

    Every setup is its own instrument, so no other setup's centre is used. The one exception is the same frames
    split into OSC channels: a light folder ``<base>`` and its platesolve channel setups ``<base>_<ch>`` are one
    camera and one pointing, so a channel setup and its base resolve to each other (exact name first, then the
    channels in ``osc_align.OSC_CHANNELS`` order)."""
    if not centres or not setup:
        return None
    from osc_extract import OSC_CHANNELS, parse_osc_channel_from_setup  # noqa: PLC0415 - leaf module (no import cycle)

    want_base = parse_osc_channel_from_setup(str(setup))[0]
    names = [str(setup)] + [want_base] + [f"{want_base}_{ch}" for ch in OSC_CHANNELS]
    for name in names:
        c = centres.get(name)
        if not isinstance(c, Mapping):
            continue
        try:
            ra, de = float(c.get("ra_deg")), float(c.get("de_deg"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(ra) and math.isfinite(de) and not (abs(ra) < 1e-9 and abs(de) < 1e-9):
            return ra, de
    return None


def frame_setup_name(header: Any) -> str:
    """Setup (light folder / platesolve directory) name of a frame from its header: ``VY_OBSG`` written by the
    calibration, else the observation group of FILTER / EXPTIME / XBINNING (``fits_meta.observation_group_key``)."""
    key = str(header.get("VY_OBSG") or "").strip()
    if not key:
        key = observation_group_key_from_metadata(fits_metadata_from_primary_header(header))
    return observation_group_folder_name(key)
