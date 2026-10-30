"""Session importer for VYVAR (file-first workflow)."""

from __future__ import annotations

import shutil
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval


from database import DraftTechnicalMetadataError, VyvarDatabase
from fits_suffixes import path_suffix_is_fits
from infolog import log_event
from fits_meta import (
    extract_fits_metadata,
    fits_metadata_from_primary_header,
    log_lights_binning_from_headers_preflight,
)
from optics import plate_scale_arcsec_per_pixel
from utils import fits_binning_xy_from_header
from fits_meta import observation_group_folder_name  # LAYER-2026-10: moved there
from fits_meta import observation_group_key  # G02-42: the one group-key builder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImportResult:
    draft_id: int | None
    observation_id: str | None
    lights_path: str
    flat_path: str
    archive_path: str
    warnings: list[str]


@dataclass(slots=True)
class SmartScanRow:
    type: str  # Lights | Darks | Flats
    status: str  # ok | missing | empty | raw | master | library | expired | draft
    count: int
    parameters: str




def _raw_ccd_temp_from_header(header: fits.Header) -> float | None:
    """Sensor temperature [C] or None (G01-08: the one reader, ``calib_library.header_temp``)."""
    from calib_library import header_temp  # noqa: PLC0415

    return header_temp(header)


def _calibration_light_temp_c(
    path: Path | str,
    db: VyvarDatabase | None = None,
) -> float | None:
    """Light CCD temperature for dark matching from raw ``CCD_TEMP``; ``None`` when unknown."""
    p = Path(path)
    if db is not None:
        try:
            st = p.stat()
        except OSError:
            st = None
        if st is not None:
            row = db.fits_header_cache_get_if_fresh(
                p, file_size=int(st.st_size), mtime=float(st.st_mtime)
            )
            if row is not None:
                ct = row["CCD_TEMP"]
                if ct is None:
                    return None
                try:
                    v = float(ct)
                except (TypeError, ValueError):
                    return None
                return v if math.isfinite(v) else None
    try:
        with fits.open(p, memmap=False) as hdul:
            return _raw_ccd_temp_from_header(hdul[0].header)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[EXC-0088] _raw_ccd_temp: unreadable FITS -> None -> dark registration later refused as 'temp unkn...: %s", exc)
        return None


@dataclass(slots=True)
class SmartImportPlan:
    source_root: str
    lights_files: list[str]
    dark_files: list[str]
    flat_files: list[str]
    metadata: dict[str, Any] | None
    scan_rows: list[SmartScanRow]
    masterflat_by_filter: dict[str, str | None]
    missing_flat_filters: list[str]
    quick_look: bool
    detected_filters: list[str]
    warnings: list[str]
    #: (FILTER|EXPTIME|BINNING) -> group detail dict (paths, scale, ...)
    observation_groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    masterflat_by_obs_key: dict[str, str | None] = field(default_factory=dict)
    dark_master_by_obs_key: dict[str, str | None] = field(default_factory=dict)
    #: groups without any FLAT (calibrated with the DARK only, flagged; decision A1) - information only
    missing_obs_keys: list[str] = field(default_factory=list)
    #: groups whose FLAT is a flagged substitute from another night -> its capture date (decision A1)
    flat_substitute_by_obs_key: dict[str, str] = field(default_factory=dict)
    #: CAL-2026-10 calibration plan (calib_plan.CalibrationPlan.to_dict()).
    calibration_plan: dict[str, Any] = field(default_factory=dict)


def _looks_like_master(fp: Path) -> bool:
    name = fp.name.upper()
    # CAL-2026-10: raw frames from capture software are often named DARK_* / FLAT_*
    # (e.g. N.I.N.A.), so only explicit master names or master headers count.
    if name.startswith(("MD_", "MF_", "MASTER")):
        return True
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        if "NCOMBINE" in hdr:
            return True
        hist = hdr.get("HISTORY")
        if hist and "MASTER" in str(hist).upper():
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


_LIGHT_IMAGETYP = frozenset(
    {
        "light",
        "light frame",
        "lights",
        "object",
        "science",
    }
)


def _classify_imagetyp(value: Any) -> str:
    t = str(value or "").strip().lower()
    if "dark" in t:
        return "dark"
    if "flat" in t:
        return "flat"
    if t in _LIGHT_IMAGETYP or "light" in t:
        return "light"
    return "unknown"


def _collect_fits_by_type(
    source_root: Path,
    db: VyvarDatabase | None = None,
) -> dict[str, list[Path]]:
    """Recursively collect FITS files and classify by IMAGETYP/FRAME per file.

    With ``db``, uses ``FITS_HEADER_CACHE`` and refreshes missing/stale rows in one transaction.
    """
    out: dict[str, list[Path]] = {"light": [], "dark": [], "flat": [], "unknown": []}
    seen: set[str] = set()
    pending_cache: list[tuple[Path, int, float, dict[str, Any], str, str | None]] = []
    _force_phys: float | None = None
    for fp in source_root.rglob("*"):
        if not fp.is_file():
            continue
        if not path_suffix_is_fits(fp):
            continue
        key = str(fp.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        cls = "unknown"
        try:
            st = fp.stat()
        except OSError:
            out.setdefault(cls, []).append(fp)
            continue
        row = None
        if db is not None:
            row = db.fits_header_cache_get_if_fresh(
                fp, file_size=int(st.st_size), mtime=float(st.st_mtime)
            )
        if row is not None:
            imagetyp = str(row["IMAGETYP"] or "")
            cls = _classify_imagetyp(imagetyp)
        else:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header
                imagetyp_raw = str(
                    hdr.get("IMAGETYP") or hdr.get("FRAME") or hdr.get("IMTYPE") or ""
                )
                cls = _classify_imagetyp(imagetyp_raw)
                if db is not None:
                    meta = fits_metadata_from_primary_header(hdr, force_physical_pixel_um=_force_phys)
                    do = hdr.get("DATE-OBS") or hdr.get("DATEOBS")
                    date_obs = None if do in (None, "") else str(do)
                    pending_cache.append(
                        (fp, int(st.st_size), float(st.st_mtime), meta, imagetyp_raw, date_obs)
                    )
            except Exception:  # noqa: BLE001
                cls = "unknown"
        out.setdefault(cls, []).append(fp)
    if db is not None and pending_cache:
        db.fits_header_cache_upsert_batch(pending_cache)
    for k in out:
        out[k] = sorted(out[k])
    return out


def _read_filter(fp: Path, db: VyvarDatabase | None = None) -> str:
    if db is not None:
        hit = db.fits_header_cache_try_filter(fp)
        if hit is not None:
            return hit
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        flt = str(hdr.get("FILTER") or hdr.get("FILT") or "").strip()
        if not flt or flt.strip().lower() in {"unknown", "none", "nan"}:
            return "NoFilter"
        return flt
    except Exception as exc:  # noqa: BLE001
        # EXC-0095 / EXCEPT-FIX-3 #1 (T1): an unreadable header is NOT evidence of an
        # unfiltered frame; returning "NoFilter" is a known filter-MISATTRIBUTION risk
        # (wrong flat group + wrong band_classify/CT/k2 routing), now surfaced loudly.
        # Contract unchanged. OPEN follow-up: a fail-closed import abort (see census EXC-0095).
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().importer_filter_read_fail += 1
        logger.error("[IMPORT] filter read failed for %s (returning NoFilter): %s", fp, exc)
        return "NoFilter"


def _sanitize_filter_folder(name: str) -> str:
    """Sanitize FILTER value into a Windows-safe folder name."""
    text = (name or "").strip()
    if not text:
        return "NoFilter"
    # Remove spaces and normalize common separators
    text = text.replace(" ", "")
    text = text.replace("/", "-").replace("\\", "-")
    # Windows forbidden characters: <>:"/\|?*
    forbidden = '<>:"\\|?*'
    for ch in forbidden:
        text = text.replace(ch, "_")
    # Avoid trailing dots/spaces (invalid on Windows)
    text = text.rstrip(" .")
    # Avoid Windows reserved device names
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    if text.upper() in reserved:
        text = f"_{text}"

    # Keep it short to avoid long paths; add stable suffix if truncated
    if len(text) > 24:
        import hashlib

        suffix = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:6]
        text = f"{text[:18]}_{suffix}"

    return text or "NoFilter"


def _safe_copy2(src: Path, dst: Path) -> None:
    """Copy with actionable error context (Windows path issues)."""
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"{exc.strerror} | src='{src}' dst='{dst}'",
        ) from exc


def _sanitize_windows_filename(name: str) -> str:
    """Sanitize a filename for Windows (keep extension if present)."""
    name = (name or "").strip()
    if not name:
        return "file.fits"
    forbidden = '<>:"\\|?*'
    for ch in forbidden:
        name = name.replace(ch, "_")
    # Avoid trailing dots/spaces
    name = name.rstrip(" .")
    # Avoid reserved device names (stem)
    stem, dot, ext = name.partition(".")
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    if stem.upper() in reserved:
        stem = f"_{stem}"
    if dot:
        return f"{stem}.{ext}"
    return stem


def _dst_path_with_length_limit(dst_dir: Path, original_name: str, *, max_path_len: int = 240) -> Path:
    """Return a destination path that stays within Windows path limits."""
    import hashlib

    safe_name = _sanitize_windows_filename(original_name)
    dst = dst_dir / safe_name
    if len(str(dst)) <= max_path_len:
        return dst

    stem = Path(safe_name).stem
    ext = Path(safe_name).suffix
    digest = hashlib.md5(safe_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    short_stem = stem[: max(8, 40 - len(digest))].rstrip(" ._")
    candidate = dst_dir / f"{short_stem}_{digest}{ext}"
    if len(str(candidate)) <= max_path_len:
        return candidate

    # last resort: very short
    return dst_dir / f"{digest}{ext or '.fits'}"


def _params_string(meta: dict[str, Any], *, include_filter: bool) -> str:
    exp = float(meta.get("exposure", 0.0))
    gain = float(meta.get("gain", 0.0) or 0.0)
    _t = meta.get("temp")
    temp_s = f"{float(_t):g}" if _t is not None else "unknown"
    binning = int(meta.get("binning", 1))
    flt = str(meta.get("filter", "Unknown"))
    base = f"Exp={exp:g}s, Gain={gain:g}G, Temp={temp_s}C, Bin={binning}"
    return f"{base}, Filter={flt}" if include_filter else base


def mixed_camera_settings_message(group_key: str, settings: dict[tuple[Any, Any, Any], int]) -> str | None:
    """CAMERA-PROFILE-2026-10: the lights of one observation group must share gain setting, offset and
    readout mode - darks, flats and the gain are matched per setting. None when they do."""
    if len(settings) <= 1:
        return None

    def _t(k: tuple[Any, Any, Any]) -> str:
        g, o, m = k
        parts = [f"gain {g:g}" if isinstance(g, (int, float)) else f"gain {g}"]
        if o is not None:
            parts.append(f"offset {o:g}" if isinstance(o, (int, float)) else f"offset {o}")
        if m not in (None, ""):
            parts.append(f"mode {m}")
        return " / ".join(parts)

    desc = ", ".join(f"{_t(k)}: {n} frames" for k, n in sorted(settings.items(), key=lambda kv: -kv[1]))
    return (
        f"Lights {group_key.replace('|', ' / ')} were taken with different camera settings ({desc}). "
        "Calibration masters and the gain are matched per setting: put each setting in its own folder "
        "and import them separately."
    )


def smart_scan_source(
    *,
    source_root: str | Path,
    calibration_library_root: str | Path,
    masterdark_validity_days: int,
    masterflat_validity_days: int,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
    # Dark master |dT| tolerance (deg C). Production callers pass
    # ``cfg.calibration_master_ccd_temp_tolerance_c``; the 0.5 literal is a documented
    # last-resort default for dev scripts/tests that do not thread config (WAVE-B STEP 1).
    calibration_master_ccd_temp_tolerance_c: float = 0.5,
    precalibrated: bool = False,
    masters_from: str = "lights",
    import_source_masters: bool = False,
) -> SmartImportPlan:
    """Scan source for lights/darks/flats and decide calibration paths (CAL-2026-10 plan).

    Masters are made only in the Calibration Library. Master files lying next to the
    LIGHTs are recognised; ``masters_from`` ("lights" default, or "library") decides
    which source is preferred. The scan itself writes nothing; with
    ``import_source_masters`` (RUN) the chosen LIGHT-folder masters are imported into
    the library and the plan points at the library copies.

    With ``db``, refreshes ``FITS_HEADER_CACHE`` (header metadata only) for fast rescans; no observation/draft writes.
    The temperature tolerance governs library dark selection; flat selection code is unchanged
    (it shares the same tolerance, a no-op at the 0.5 default).
    """
    root = Path(source_root)
    try:
        temp_tol = float(calibration_master_ccd_temp_tolerance_c)
    except (TypeError, ValueError):
        temp_tol = 0.5
    if not math.isfinite(temp_tol) or temp_tol < 0:
        temp_tol = 0.5

    scan_rows: list[SmartScanRow] = []
    warnings: list[str] = []

    if not root.exists() or not root.is_dir():
        scan_rows.append(SmartScanRow("Lights", "missing", 0, ""))
        return SmartImportPlan(
            source_root=str(root),
            lights_files=[],
            dark_files=[],
            flat_files=[],
            metadata=None,
            scan_rows=scan_rows,
            masterflat_by_filter={},
            missing_flat_filters=[],
            quick_look=False,
            detected_filters=[],
            warnings=warnings,
        )

    files_by_type = _collect_fits_by_type(root, db=db)
    lights_files = files_by_type.get("light", [])
    dark_files = files_by_type.get("dark", [])
    flat_files = files_by_type.get("flat", [])
    # Filters+binning combos from lights
    detected_filters = sorted(
        {flt for flt in (_read_filter(fp, db=db) for fp in lights_files) if flt}  # G01-36: no cap
    )

    if not lights_files:
        scan_rows.append(SmartScanRow("Lights", "missing", 0, ""))
        # still report what exists
        scan_rows.append(SmartScanRow("Darks", "raw" if dark_files else "missing", len(dark_files), ""))
        scan_rows.append(SmartScanRow("Flats", "raw" if flat_files else "missing", len(flat_files), ""))
        return SmartImportPlan(
            source_root=str(root),
            lights_files=[],
            dark_files=[str(p) for p in dark_files],
            flat_files=[str(p) for p in flat_files],
            metadata=None,
            scan_rows=scan_rows,
            masterflat_by_filter={},
            missing_flat_filters=[],
            quick_look=False,
            detected_filters=[],
            warnings=warnings,
        )

    log_lights_binning_from_headers_preflight(lights_files, context="Scan Source")
    first_light = lights_files[0]
    with fits.open(first_light, memmap=False) as _hd0:
        _hdr0 = _hd0[0].header
    _xb0, _yb0 = fits_binning_xy_from_header(_hdr0)
    log_event(
        f"Scan Source - first light in the list: {first_light.name} | "
        f"XBINNING={_hdr0.get('XBINNING')!r} YBINNING={_hdr0.get('YBINNING')!r} BINNING={_hdr0.get('BINNING')!r} "
        f"-> {int(_xb0)}x{int(_yb0)}"
    )
    metadata = extract_fits_metadata(first_light, db=db)
    scan_rows.append(
        SmartScanRow(
            "Lights",
            "ok",
            len(lights_files),
            _params_string(metadata, include_filter=True),
        )
    )


    calib_root = Path(calibration_library_root)

    observation_groups: dict[str, dict[str, Any]] = {}
    for fp in lights_files:  # G01-36: every LIGHT gets its plan group (no cap)
        try:
            meta_i = extract_fits_metadata(fp, db=db)
        except Exception as exc:  # noqa: BLE001
            # EXC-0102 / EXCEPT-FIX-3 #7: a light frame silently dropped from calibration
            # observation-group planning is now surfaced (still continues). See census EXC-0102.
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().importer_obs_group_meta_skip += 1
            logger.error("[IMPORT] obs-group metadata read failed for %s (frame skipped): %s", fp, exc)
            continue
        f_i = str(meta_i.get("filter", "")).strip()
        if not f_i or f_i.lower() in {"unknown", "none", "nan"}:
            f_i = "NoFilter"
        exp_i = float(meta_i.get("exposure", 0.0))
        b_i = max(1, int(meta_i.get("binning", 1) or 1))
        gk = observation_group_key(f_i, exp_i, b_i)
        if gk not in observation_groups:
            pu = meta_i.get("effective_pixel_um_plate_scale")
            foc = meta_i.get("focal_length")
            scale = None
            if pu is not None and foc is not None:
                try:
                    scale = plate_scale_arcsec_per_pixel(
                        pixel_pitch_um=float(pu), focal_length_mm=float(foc)
                    )
                except (TypeError, ValueError):
                    scale = None
            observation_groups[gk] = {
                "filter": f_i,
                "exposure_s": exp_i,
                "binning": b_i,
                "gain": float(meta_i.get("gain", 0.0) or 0.0),
                "temp": _calibration_light_temp_c(fp, db=db),
                "representative_light": str(fp),
                "light_paths": [],
                "plate_scale_arcsec_per_px": scale,
            }
        observation_groups[gk]["light_paths"].append(str(fp))
        # CAMERA-PROFILE-2026-10: gain setting / offset / readout mode of every light of the group
        _gs = observation_groups[gk].setdefault("camera_settings", {})
        _key = (meta_i.get("gain"), meta_i.get("offset"), meta_i.get("readout_mode"))
        _gs[_key] = _gs.get(_key, 0) + 1

    for gk, grp in observation_groups.items():
        _mixed = mixed_camera_settings_message(gk, grp.get("camera_settings") or {})
        if _mixed:
            grp["mixed_camera_settings"] = True
            warnings.append(_mixed)

    # CAL-2026-10 (dev/results/specs/VYVAR_CALIBRATION_SPEC.md): one plan for GUI and night runner.
    import calib_plan as _cp  # noqa: PLC0415

    _is_master = {fp: _looks_like_master(fp) for fp in list(dark_files) + list(flat_files)}  # one pass (G01-36)
    master_like_in_source = [fp for fp in list(dark_files) + list(flat_files) if _is_master[fp]]
    raw_darks = [fp for fp in dark_files if not _is_master[fp]]
    raw_flats = [fp for fp in flat_files if not _is_master[fp]]
    _light_instrume = str(_hdr0.get("INSTRUME") or "").strip()
    src_masters, _src_msgs = _cp.masters_next_to_lights(
        master_like_in_source,
        camera_id=id_equipments,
        telescope_id=id_telescope,
        light_instrume=_light_instrume,
    )
    warnings.extend(_src_msgs)
    if (raw_darks or raw_flats) and not precalibrated:
        warnings.append(
            f"Source contains {len(raw_darks)} raw dark(s) and {len(raw_flats)} raw flat(s); they are not "
            "used - build masters from them in the Calibration Library."
        )
    built: list[str] = []

    cal_plan = _cp.build_calibration_plan(
        observation_groups,
        db=db,
        camera_id=id_equipments,
        telescope_id=id_telescope,
        dark_validity_days=float(masterdark_validity_days),
        flat_validity_days=float(masterflat_validity_days),
        temp_tol=temp_tol,
        precalibrated=bool(precalibrated),
        built=built,
        warnings=list(warnings),
        source_masters=src_masters,
        masters_from=str(masters_from or "lights"),
    )
    if import_source_masters and not precalibrated and db is not None:
        for _m in _cp.import_planned_source_masters(
            cal_plan, db=db, library_root=calib_root
        ):
            if _m.startswith(("!", "[X]")):
                warnings.append(_m)
    masterflat_by_obs_key = cal_plan.flat_by_group()
    dark_master_by_obs_key = cal_plan.dark_by_group()
    missing_obs_keys = [gk for gk, g in cal_plan.groups.items() if g.flat is None]
    flat_substitute_by_obs_key = {gk: g.flat_substitute for gk, g in cal_plan.groups.items() if g.flat_substitute}
    masterflat_by_filter: dict[str, str | None] = {}
    for gk, g in cal_plan.groups.items():
        if g.flat is not None and masterflat_by_filter.get(g.filter_name) is None:
            masterflat_by_filter[g.filter_name] = g.flat.path
        masterflat_by_filter.setdefault(g.filter_name, None)
    missing_flat_filters = sorted({cal_plan.groups[k].filter_name for k in missing_obs_keys})

    n_dark_ok = sum(1 for g in cal_plan.groups.values() if g.dark is not None)
    n_groups = len(cal_plan.groups)
    def _row(kind: str, attr: str) -> SmartScanRow:
        used = [getattr(g, attr) for g in cal_plan.groups.values() if getattr(g, attr) is not None]
        srcs = sorted({getattr(g, f"{attr}_from") for g in cal_plan.groups.values() if getattr(g, attr) is not None})
        n_ok = len(used)
        why = sorted({getattr(g, f"{attr}_reason") for g in cal_plan.groups.values() if getattr(g, attr) is None})
        if precalibrated:
            st_ = "skipped"
        elif n_ok == 0:
            st_ = "expired" if any("expired" in w for w in why) else "missing"
        else:
            st_ = "+".join("LIGHT folder" if x == "lights" else "library" for x in srcs)
        files = sorted({Path(m.path).name for m in used})
        txt = f"{n_ok}/{n_groups} groups" + (f": {', '.join(files)}" if files else "")
        if why and not precalibrated:
            txt += f"; missing: {'; '.join(why[:2])}"
        return SmartScanRow(kind, st_, n_ok, txt)

    scan_rows.append(_row("Darks", "dark"))
    scan_rows.append(_row("Flats", "flat"))
    quick_look = (not precalibrated) and n_dark_ok == 0
    if quick_look:
        warnings.append("No valid master DARK for any group - LIGHTs will stay uncalibrated.")
    return SmartImportPlan(
        source_root=str(root),
        lights_files=[str(p) for p in lights_files],
        dark_files=[str(p) for p in dark_files],
        flat_files=[str(p) for p in flat_files],
        metadata=metadata,
        scan_rows=scan_rows,
        masterflat_by_filter=masterflat_by_filter,
        missing_flat_filters=missing_flat_filters,
        quick_look=quick_look,
        detected_filters=detected_filters,
        warnings=warnings,
        observation_groups=observation_groups,
        masterflat_by_obs_key=masterflat_by_obs_key,
        dark_master_by_obs_key=dark_master_by_obs_key,
        missing_obs_keys=missing_obs_keys,
        flat_substitute_by_obs_key=flat_substitute_by_obs_key,
        calibration_plan=cal_plan.to_dict(),
    )


def _first_object_name(lights_files: list[Path]) -> str:
    """The first non-empty OBJECT (or OBJNAME) of the light frames - the observed star."""
    for fp in lights_files[:20]:
        try:
            hdr = fits.getheader(fp)
        except Exception:  # noqa: BLE001
            continue
        for key in ("OBJECT", "OBJNAME"):
            val = str(hdr.get(key) or "").strip()
            if val:
                return val
    return ""


def _unique_import_dst(dst_dir: Path, fp: Path, used: dict[Path, Path], warnings: list[str]) -> Path:
    """G01-18: destination of one imported file, unique within this import. Two source files with the
    same name (e.g. ``a/Light_001.fits`` and ``b/Light_001.fits``) no longer overwrite each other: the
    second gets a short hash of its source folder in the name, and the import warns."""
    import hashlib

    src = Path(fp).resolve()
    dst = first = _dst_path_with_length_limit(dst_dir, fp.name)
    if dst in used and used[dst] != src:
        tag = hashlib.sha1(str(src.parent).encode("utf-8")).hexdigest()[:8]
        base, dot, ext = fp.name.partition(".")
        dst = _dst_path_with_length_limit(dst_dir, f"{base}__{tag}{dot}{ext}")
        if dst in used and used[dst] != src:
            raise ValueError(f"import: cannot give {src} a unique name in {dst_dir}")
        msg = (
            f"{fp.name}: another source file of this import has the same name ({used[first]}); "
            f"imported as {dst.name}"
        )
        warnings.append(msg)
        log_event(f"IMPORT WARNING: {msg}")
    used[dst] = src
    return dst


def _copy_light_into_draft(
    fp: Path, lights_root: Path, db: VyvarDatabase, used: dict[Path, Path], warnings: list[str]
) -> dict[str, Any]:
    """Copy one LIGHT into its group folder; the manifest evidence row.

    G01-32: the group key is built from the raw FILTER (the key of the plan, VY_OBSG and the calibration);
    only the folder name uses the sanitised filter (folder names unchanged)."""
    flt_raw = _read_filter(fp, db=db)
    flt = _sanitize_filter_folder(flt_raw)
    try:
        meta_l = extract_fits_metadata(fp, db=db)
        exp_l, b_l = float(meta_l.get("exposure", 0.0)), int(meta_l.get("binning", 1) or 1)
        sc_i = int(db.derive_scanning_id(meta_l))
    except Exception as exc:  # noqa: BLE001
        logger.error("[IMPORT] metadata read failed for %s (group %s|0|1): %s", fp, flt_raw, exc)
        exp_l, b_l, sc_i = 0.0, 1, None
    gk = observation_group_key(flt_raw, exp_l, b_l)
    dst_dir = lights_root / observation_group_folder_name(observation_group_key(flt, exp_l, b_l))
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = _unique_import_dst(dst_dir, fp, used, warnings)
    _safe_copy2(fp, dst_path)
    return {
        "file_path": str(dst_path),
        "imagetyp": "light",
        "filter": flt,
        "observation_group_key": gk,
        "id_scanning": sc_i,
    }


def smart_import_session(
    *,
    plan: SmartImportPlan,
    pipeline: Any,
    id_equipment: int,
    id_telescope: int,
    id_location: int | None = None,
    location_source: str | None = None,
    cfg: Any | None = None,
) -> ImportResult:
    """Perform import according to SmartImportPlan decision tree (DB write)."""
    _cfg = cfg if cfg is not None else getattr(pipeline, "config", None)
    from observer_location import resolve_observer_location_for_run

    _resolved = resolve_observer_location_for_run(
        pipeline.db.db_path,
        explicit_location_id=id_location,
        cfg=_cfg,
        source_hint=location_source,  # type: ignore[arg-type]
    )
    _id_loc = int(_resolved.location_id)

    if not plan.lights_files:
        raise FileNotFoundError("Missing 'lights' directory! Import aborted.")
    lights_files = [Path(p) for p in plan.lights_files]
    if not lights_files:
        raise FileNotFoundError("Missing 'lights' directory! Import aborted.")

    metadata = extract_fits_metadata(lights_files[0], db=pipeline.db)
    from osc_extract import validate_bayer_crosscheck

    _import_warnings: list[str] = []
    _jd_starts: list[float] = []
    for fp in lights_files:  # G01-36: every LIGHT (Bayer cross-check, start time)
        try:
            meta_x = extract_fits_metadata(fp, db=pipeline.db)
        except Exception:  # noqa: BLE001
            continue
        try:
            _jd_starts.append(float(meta_x.get("jd_start")))
        except (TypeError, ValueError):
            pass
        verdict, msg = validate_bayer_crosscheck(
            fits_bayerpat=meta_x.get("bayerpat"),
            equipment_bayermask=pipeline.db.get_equipment_bayermask(int(id_equipment)),
        )
        if verdict == "fail" and msg:
            raise ValueError(msg)
        if verdict == "warn" and msg and msg not in _import_warnings:
            _import_warnings.append(msg)
    if _import_warnings:
        plan.warnings.extend(_import_warnings)
    for gk, grp in (plan.observation_groups or {}).items():
        _mixed = mixed_camera_settings_message(gk, grp.get("camera_settings") or {})
        if _mixed:
            raise ValueError(_mixed)
    scanning_id = pipeline.db.derive_scanning_id(metadata)

    # Ingestion creates DRAFT only (Session ID created after astrometry).
    # PROCESSED-FIELD-NAME-2026-10: the folder is <TARGET>_<YYYYMMDD>_<NNN> - the target from the
    # FITS OBJECT (else the selected source folder), the evening the night began (first frame).
    _jd_ok = [j for j in _jd_starts if j > 2400000.0]
    draft_id = pipeline.db.create_draft(
        {
            "id_equipments": int(id_equipment),
            "id_telescope": int(id_telescope),
            "id_location": int(_id_loc),
            "id_scanning": int(scanning_id),
            "observation_start_jd": float(min(_jd_ok)) if _jd_ok else float(metadata["jd_start"]),
            "is_calibrated": not plan.quick_look,
            "object_name": _first_object_name(lights_files),
            "source_folder_name": Path(str(plan.source_root)).name if plan.source_root else "",
            "site_lon_deg": float(_resolved.lon),
        }
    )

    try:
        _comb = pipeline.db.get_combined_metadata(lights_files[0], int(draft_id))
        if _comb.get("focal_length_mm") is None or _comb.get("pixel_effective_um") is None:
            plan.warnings.append(str(DraftTechnicalMetadataError(int(draft_id))))
    except Exception:  # noqa: BLE001
        # EXC-0103: T3 -- guard suppresses appending a non-fatal DraftTechnicalMetadataError warning (EXCEPT-BULK-2 2026-07-08)
        pass

    from draft_provenance import draft_dir_for_id  # noqa: PLC0415

    archive_session = draft_dir_for_id(pipeline.config.archive_root, int(draft_id))
    # Partial draft if some filters missing masterflat
    if plan.quick_look:
        target = archive_session / "non_calibrated"
        lights_root = target / "lights"
        evidence: list[dict[str, Any]] = []
        _used_q: dict[Path, Path] = {}
        for fp in lights_files:
            evidence.append(
                _copy_light_into_draft(fp, lights_root, pipeline.db, _used_q, plan.warnings)
                | {"is_calibrated": 0, "calib_type": "RAW_NON_CALIBRATED"}
            )
        now_utc = datetime.now(timezone.utc)
        pipeline.db.update_draft_import_log(
            draft_id,
            lights_path=str(lights_root),
            calib_path=f"draft_non_calibrated={target}",
            imported_at=now_utc.isoformat(timespec="seconds"),
            import_warnings="\n".join(plan.warnings) if plan.warnings else None,
            is_calibrated=False,
            archive_path=str(archive_session),
        )
        pipeline.db.insert_draft_files(draft_id, evidence)
        return ImportResult(
            draft_id=draft_id,
            observation_id=None,
            lights_path=str(lights_root),
            flat_path="",
            archive_path=str(archive_session),
            warnings=plan.warnings,
        )

    # calibrated path: copy into Raw structure
    raw_dir = archive_session / "Raw"
    lights_root = raw_dir / "lights"
    darks_dst = raw_dir / "darks"
    flats_dst = raw_dir / "flats"
    lights_root.mkdir(parents=True, exist_ok=True)
    darks_dst.mkdir(parents=True, exist_ok=True)
    flats_dst.mkdir(parents=True, exist_ok=True)

    # G01-01 (decision A1): one lights tree per draft. Every group goes to Raw/lights, also a group without
    # a FLAT (the plan calibrates it with the DARK only and flags it); the former non_calibrated split left
    # the groups WITH masters uncalibrated.
    # Flexible sorting: even if mixed folders exist, we sort per-file by IMAGETYP
    evidence: list[dict[str, Any]] = []
    _used: dict[Path, Path] = {}
    for fp in lights_files:
        evidence.append(
            _copy_light_into_draft(fp, lights_root, pipeline.db, _used, plan.warnings)
            | {"is_calibrated": 1, "calib_type": "MASTER_PIPELINE"}
        )
    for fp in [Path(p) for p in plan.dark_files]:
        dst_path = _unique_import_dst(darks_dst, fp, _used, plan.warnings)
        _safe_copy2(fp, dst_path)
        evidence.append({"file_path": str(dst_path), "imagetyp": "dark", "filter": ""})
    for fp in [Path(p) for p in plan.flat_files]:
        dst_path = _unique_import_dst(flats_dst, fp, _used, plan.warnings)
        _safe_copy2(fp, dst_path)
        evidence.append(
            {
                "file_path": str(dst_path),
                "imagetyp": "flat",
                "filter": _sanitize_filter_folder(_read_filter(fp, db=pipeline.db)),
            }
        )

    now_utc = datetime.now(timezone.utc)
    pipeline.db.update_draft_import_log(
        draft_id,
        lights_path=str(lights_root),
        calib_path=(
            f"flat_by_filter={plan.masterflat_by_filter};"
            f"flat_by_obs_key={getattr(plan, 'masterflat_by_obs_key', {})};"
            f"dark_by_obs_key={getattr(plan, 'dark_master_by_obs_key', {})}"
        ),
        imported_at=now_utc.isoformat(timespec="seconds"),
        import_warnings="\n".join(plan.warnings) if plan.warnings else None,
        is_calibrated=True,
        archive_path=str(archive_session),
    )
    pipeline.db.insert_draft_files(draft_id, evidence)
    return ImportResult(
        draft_id=draft_id,
        observation_id=None,
        lights_path=str(lights_root),
        flat_path=";".join(
            [f"{k}={v}" for k, v in (getattr(plan, "masterflat_by_obs_key", None) or plan.masterflat_by_filter or {}).items()]
        ),
        archive_path=str(archive_session),
        warnings=plan.warnings,
    )


def quicklook_preview_png_bytes(fits_path: str | Path) -> bytes:
    """Generate an 8-bit stretched preview PNG bytes (ZScale) from a FITS image."""
    fp = Path(fits_path)
    with fits.open(fp, memmap=False) as hdul:
        data = _to_float32_frame(hdul[0].data)
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    scaled = np.clip((data - vmin) / (vmax - vmin + 1e-12), 0, 1)
    # negative (dark stars on white sky), as every field image of VYVAR
    img8 = (255 - scaled * 255).astype(np.uint8)

    # Encode via PIL if available; fallback to raw bytes not supported by st.image
    try:
        from PIL import Image  # type: ignore
        import io

        im = Image.fromarray(img8)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PIL is required for PNG preview generation.") from exc

# --- CalibrationLibrary master stacking (no sigma-clipping on the stack axis) ---
#
# **Master dark - plain average (mean):** Low-signal frames; sigma-clipping would wrongly reject
# real hot pixels and harm thermal-noise subtraction SNR. We use per-pixel ``nanmean`` over frames.
#
# **Master flat - plain median:** Per-pixel ``nanmedian`` ignores dust motes / glints while preserving
# vignetting structure. Median normalization is deferred to :func:`calibration.get_processed_master`
# (after resample to the light's binning) and flagged on disk with ``VYFLNRD=1``.


def _to_float32_frame(data: np.ndarray) -> np.ndarray:
    """Use a float32 view when already float32; copy only for integer / other dtypes."""
    a = np.asanyarray(data)
    if a.dtype == np.float32:
        return a
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    return np.asarray(a, dtype=np.float32)


