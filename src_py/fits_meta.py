"""Moved from pipeline.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import os
from astropy.io import fits
from astropy.time import Time
import pandas as pd
from config import AppConfig
from database import VyvarDatabase, _db_to_float as _to_float_db
from fits_suffixes import path_suffix_is_fits
from infolog import log_event
from utils import fits_binning_xy_from_header, normalize_telescope_focal_mm_for_plate_scale
from vyvar_platesolver import _fits_header_parse_dec_deg, _fits_header_parse_ra_deg

def _safe_filter_token(text: str) -> str:
    t = (text or "").strip()
    if not t or t.lower() in {"unknown", "none", "nan"}:
        return "NoFilter"
    return t

def observation_group_key_from_metadata(meta: dict[str, Any]) -> str:
    """Must match ``importer.observation_group_key`` (FILTER, EXPTIME, binning from FITS / metadata dict)."""
    flt = _safe_filter_token(str(meta.get("filter") or "NoFilter"))
    try:
        e = float(meta.get("exposure", 0.0))
    except (TypeError, ValueError):
        e = 0.0
    b = max(1, int(meta.get("binning", 1) or 1))
    return f"{flt}|{e:g}|{b}"

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
            log_event(f"{context} - binning z hlaviciek: nepodarilo sa precitat ziadny FITS.")
        return None
    log_event(f"{context} - binning z hlaviciek FITS ({len(paths)} suborov):")
    samples: dict[tuple[int, int], Path] = summary.get("samples") or {}
    for (xb, yb), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        sample = samples.get((xb, yb))
        sn = sample.name if sample is not None else "?"
        log_event(f"  {xb}x{yb}: {n}x (priklad: {sn})")
    if len(counts) > 1:
        log_event(
            f"{context} - POZOR: zmiesane binning rezimy v davke - "
            "kalibracia / master match pouzivaju binning z hlavicky kazdeho suboru."
        )
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    return best_key

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

def _fits_pixel_raw_to_micrometres(value: float) -> float:
    """Map raw FITS pixel-size keywords to **micrometres** (WCS often uses SI metres)."""
    if not math.isfinite(value) or value <= 0:
        return 0.0
    v = float(value)
    if v < 5e-5:
        return v * 1e6
    if v < 0.2:
        return v * 1000.0
    return v

def _focal_mm_plausible(mm: float) -> bool:
    return math.isfinite(mm) and 40.0 <= mm <= 120_000.0

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
                f"PIXEL: EQUIPMENTS.PIXELSIZE={nv:g} um x binning {x_bin}x{y_bin} -> efektivny {_eff:g} um "
                f"(preferovane pred FITS; predtym {prev!r})."
            )
    else:
        log_event(
            f"PIXEL: EQUIPMENTS.PIXELSIZE={nv:g} um x binning {x_bin}x{y_bin} -> efektivny {_eff:g} um."
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

def _header_pick_first(header: fits.Header, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return default

def _enrich_calibration_metadata_from_header(
    meta: dict[str, Any],
    header: fits.Header,
    *,
    db: VyvarDatabase | None,
    id_equipment: int | None,
    id_telescope: int | None = None,
) -> None:
    """Add ``focal_length``, ``pixel_size_raw``, ``pixel_um``, ``focal_length_source`` for diagnostics / UI."""

    def _to_f(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    r1 = _to_f(_header_pick_first(header, "PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0))
    r2 = _to_f(_header_pick_first(header, "PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0))
    parts: list[str] = []
    if r1 > 0:
        parts.append(f"PIX1={r1:g}")
    if r2 > 0:
        parts.append(f"PIX2={r2:g}")
    meta["pixel_size_raw"] = " | ".join(parts) if parts else "n/a"

    meta["pixel_um"] = meta.get("effective_pixel_um_plate_scale")

    focal_mm: float | None = None
    src = "none"
    # Universal optics: prefer DB (UI) over FITS FOCALLEN for plate scale / solver hints.
    if db is not None:
        telf = db.get_telescope_focal_mm(int(id_telescope) if id_telescope is not None else None)
        if telf is not None:
            focal_mm, _fx = normalize_telescope_focal_mm_for_plate_scale(float(telf))
            src = "telescope_focal"

    has_focallen = "FOCALLEN" in header and header["FOCALLEN"] not in (None, "", " ", "0", 0)
    if focal_mm is None and has_focallen:
        raw_fc = header["FOCALLEN"]
        log_event(f"DIAG: FITS FOCALLEN (SIPS / zapisovatel hlavicky) = {raw_fc!r}")
        try:
            v = float(raw_fc)
            mm0 = (v * 1000.0) if (math.isfinite(v) and v > 0 and v < 25.0) else v
        except (TypeError, ValueError):
            mm0 = None
        if mm0 is not None and math.isfinite(mm0) and mm0 > 0 and _focal_mm_plausible(float(mm0)):
            focal_mm, _fx = normalize_telescope_focal_mm_for_plate_scale(float(mm0))
            src = "fits_focallen"

    meta["focal_length"] = focal_mm
    meta["focal_length_source"] = src

def _apply_draft_combined_to_pipeline_meta(meta: dict[str, Any], comb: dict[str, Any]) -> None:
    """Overlay ``VyvarDatabase.get_combined_metadata`` onto ``extract_fits_metadata`` output."""
    fl = comb.get("focal_length_mm")
    if fl is not None:
        try:
            fv = float(fl)
            if math.isfinite(fv) and _focal_mm_plausible(fv):
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

def _fits_meta_ra_deg(value: Any) -> float:
    r = _fits_header_parse_ra_deg(value)
    return float(r) if r is not None else 0.0

def _fits_meta_dec_deg(value: Any) -> float:
    d = _fits_header_parse_dec_deg(value)
    return float(d) if d is not None else 0.0

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
            f"DEBUG: Fyzicky pixel: {ps_s} um | Binning: {binning_x}x{binning_y} | "
            f"EFEKTIVNY PIXEL PRE VYPOCET: {eff_s} um{tail}"
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

    def _pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in header and header[key] not in (None, ""):
                return header[key]
        return default

    jd_obs = _pick("JD", "JD-OBS", default=None)
    if jd_obs is not None:
        jd_start = _to_float_db(jd_obs, 0.0)
    else:
        mjd_obs = _pick("MJD-OBS", default=None)
        if mjd_obs is not None:
            jd_start = _to_float_db(mjd_obs, 0.0) + 2400000.5
        else:
            # Fallback: compute JD from exposure start time keywords.
            # DATE-OBS is typically the start of exposure in UTC (FITS standard).
            date_obs = _pick("DATE-OBS", "DATEOBS", default=None)
            time_obs = _pick("TIME-OBS", "TIMEOBS", default=None)
            jd_start = 0.0
            if date_obs is not None:
                dt_str = str(date_obs).strip()
                if time_obs is not None and "T" not in dt_str:
                    dt_str = f"{dt_str}T{str(time_obs).strip()}"
                try:
                    jd_start = float(Time(dt_str, format="isot", scale="utc").jd)
                except Exception:  # noqa: BLE001
                    jd_start = 0.0

    _raw_xb = _pick("XBINNING", "BINNING", default=1)
    _raw_yb = _pick("YBINNING", default=_raw_xb)
    x_bin = _parse_fits_binning_int(_raw_xb, 1)
    y_bin = _parse_fits_binning_int(_raw_yb, x_bin)

    _ps1 = _fits_pixel_raw_to_micrometres(_to_float_db(_pick("PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0)))
    _ps2 = _fits_pixel_raw_to_micrometres(_to_float_db(_pick("PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0)))

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

    return {
        "exposure": float(_pick("EXPTIME", "EXPOSURE", default=0.0)),
        "filter": str(_pick("FILTER", "FILT", default="NoFilter")),
        "binning": int(x_bin),
        "binning_y": int(y_bin),
        "fits_xbinning_raw": _raw_xb,
        "fits_ybinning_raw": _raw_yb,
        "naxis1": int(header.get("NAXIS1", 0) or 0),
        "naxis2": int(header.get("NAXIS2", 0) or 0),
        "pixel_size_um_physical": _physical_mean,
        "pixel_size_um_header": _effective_um,
        "effective_pixel_um_plate_scale": _plate_eff,
        "temp": float(_pick("CCD-TEMP", "SENSORTEMP", "SET-TEMP", default=0.0)),
        "gain": int(_pick("GAIN", "GAINER", "CCD-GAIN", default=0) or 0),
        "ra": _fits_meta_ra_deg(
            _pick(
                "OBJCTRA",
                "RA_OBJ",
                "TARGRA",
                "CENTRA",
                "RA",
                "RAJ2000",
                "CAT-RA",
                "CRVAL1",
                default=0.0,
            )
        ),
        "dec": _fits_meta_dec_deg(
            _pick(
                "OBJCTDEC",
                "DEC_OBJ",
                "TARGDEC",
                "CENTDEC",
                "DEC",
                "DEJ2000",
                "CAT-DEC",
                "CRVAL2",
                default=0.0,
            )
        ),
        "jd_start": jd_start,
        "telescope": _pick("TELESCOP", "SCOPE", default=None),
        "camera": _pick("INSTRUME", "CAMERA", default=None),
        "bayerpat": _valid_bayerpat_from_header(header),
    }

def _valid_bayerpat_from_header(header: fits.Header) -> str | None:
    from osc_extract import valid_bayer_pattern_4

    return valid_bayer_pattern_4(str(header.get("BAYERPAT") or ""))

def extract_fits_metadata(
    filepath: Path | str,
    *,
    db: VyvarDatabase | None = None,
    app_config: AppConfig | None = None,
    force_physical_pixel_um: float | None = None,
    id_equipment: int | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Extract key metadata from FITS primary header.

    When ``db`` is set, uses ``FITS_HEADER_CACHE`` when ``FILE_PATH``, ``FILE_SIZE``, and ``MTIME`` match
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
    - focal_length_source, pixel_size_raw (surove cisla z hlavicky), pixel_um (= effective pixel)
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
                    id_equipment=id_equipment,
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
        id_equipment=id_equipment,
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
            mtime=float(st.st_mtime),
            meta=meta,
            imagetyp=imagetyp,
            date_obs=date_obs,
        )

    return meta

def scan_usb_folder(path: Path | str) -> pd.DataFrame:
    """Recursively scan source tree and detect Lights/Darks/Flats by IMAGETYP/FRAME.

    This scan is folder-name agnostic and reports real folder paths.
    If a folder contains mixed types, it will be marked as Mixed, but files can still be
    sorted per-file by IMAGETYP downstream (importer).

    Output columns:
    - Folder Path
    - Type
    - File Count
    - Lights Count
    - Darks Count
    - Flats Count
    - Unknown Count
    - Detected Filters
    - Params
    """
    root = Path(path)
    rows: list[dict[str, Any]] = []

    _light_tokens = frozenset(
        {"light", "light frame", "lights", "object", "science"},
    )

    def _classify(text: str) -> str:
        t = (text or "").strip().lower()
        if "dark" in t:
            return "Darks"
        if "flat" in t:
            return "Flats"
        if t in _light_tokens or "light" in t:
            return "Lights"
        return "Unknown"

    def _fits_files(folder: Path) -> list[Path]:
        files: list[Path] = []
        seen: set[str] = set()
        for fp in folder.iterdir():
            if not fp.is_file():
                continue
            if not path_suffix_is_fits(fp):
                continue
            key = str(fp.resolve()).casefold()
            if key in seen:
                continue
            seen.add(key)
            files.append(fp)
        return sorted(files)

    if not root.exists() or not root.is_dir():
        return pd.DataFrame(columns=["Folder Path", "Type", "File Count", "Params"])

    # scan all subfolders, including root itself
    folders = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for folder in folders:
        files = _fits_files(folder)
        if not files:
            continue

        # classify per-file to detect mixed folders
        type_counts: dict[str, int] = {"Lights": 0, "Darks": 0, "Flats": 0, "Unknown": 0}
        filter_set: set[str] = set()
        for fp in files:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header
                imagetyp = str(hdr.get("IMAGETYP") or hdr.get("FRAME") or hdr.get("IMTYPE") or "")
                cls = _classify(imagetyp)
                flt = str(hdr.get("FILTER") or hdr.get("FILT") or "").strip()
                if flt:
                    filter_set.add(flt)
            except Exception:  # noqa: BLE001
                cls = "Unknown"
            type_counts[cls] = type_counts.get(cls, 0) + 1

        present = [k for k, v in type_counts.items() if v > 0 and k != "Unknown"]
        if len(present) == 1:
            detected = present[0]
        elif len(present) > 1:
            detected = "Mixed"
        else:
            detected = "Unknown"

        # params from first file
        first = files[0]
        try:
            with fits.open(first, memmap=False) as hdul:
                hdr = hdul[0].header
            exp = hdr.get("EXPTIME") or hdr.get("EXPOSURE") or 0
            gain = hdr.get("GAIN") or hdr.get("GAINER") or hdr.get("CCD-GAIN") or 0
            params = f"Exp={float(exp):g}s, Gain={int(gain)}"
        except Exception:  # noqa: BLE001
            params = ""

        rows.append(
            {
                "Folder Path": str(folder),
                "Type": detected,
                "File Count": int(len(files)),
                "Lights Count": int(type_counts.get("Lights", 0)),
                "Darks Count": int(type_counts.get("Darks", 0)),
                "Flats Count": int(type_counts.get("Flats", 0)),
                "Unknown Count": int(type_counts.get("Unknown", 0)),
                "Detected Filters": ", ".join(sorted(filter_set)) if filter_set else "",
                "Params": params,
            }
        )

    return pd.DataFrame(rows).sort_values(["Type", "Folder Path"], ignore_index=True)
