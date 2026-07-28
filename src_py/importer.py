"""Session importer for VYVAR (file-first workflow)."""

from __future__ import annotations

import os
import shutil
import tempfile
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval


from calibration import resolve_master_age, reset_master_age_mtime_warnings
from database import DraftTechnicalMetadataError, VyvarDatabase
from fits_suffixes import path_suffix_is_fits
from infolog import log_event
from pipeline import (
    extract_fits_metadata,
    fits_metadata_from_primary_header,
    log_lights_binning_from_headers_preflight,
)
from utils import fits_binning_xy_from_header, plate_scale_arcsec_per_pixel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImportResult:
    draft_id: int | None
    observation_id: str | None
    lights_path: str
    dark_path: str
    flat_path: str
    archive_path: str
    warnings: list[str]


@dataclass(slots=True)
class CalibrationStatus:
    kind: str
    path: str | None
    status: str  # ok | expired | missing
    last_modified_utc: str | None
    age_days: int | None
    validity_days: int
    message: str


@dataclass(slots=True)
class MasterGenerationResult:
    dark_master_path: str | None
    flat_master_path: str | None
    messages: list[str]


@dataclass(slots=True)
class SmartScanRow:
    type: str  # Lights | Darks | Flats
    status: str  # ok | missing | empty | raw | master | library | expired | draft
    count: int
    parameters: str
    details: str | None = None


def observation_group_key(filter_name: str, exposure_s: float, binning: int) -> str:
    """Unique observation subgroup: (FILTER, EXPTIME, XBINNING) - binning from FITS via metadata."""
    flt = _filter_name_for_calibration_library_flat(filter_name)
    try:
        e = float(exposure_s)
    except (TypeError, ValueError):
        e = 0.0
    b = max(1, int(binning))
    return f"{flt}|{e:g}|{b}"


def observation_group_folder_name(group_key: str) -> str:
    """Filesystem-safe folder under ``lights`` (replaces ``|``)."""
    s = group_key.replace("|", "_").replace("/", "_").replace("\\", "_")
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in s)
    return s[:120] if len(s) > 120 else s


_CCD_TEMP_HEADER_KEYS = ("CCD-TEMP", "SENSORTEMP", "SET-TEMP")


def _raw_ccd_temp_from_header(header: fits.Header) -> float | None:
    for key in _CCD_TEMP_HEADER_KEYS:
        if key in header and header[key] not in (None, ""):
            try:
                v = float(header[key])
            except (TypeError, ValueError):
                continue
            return v if math.isfinite(v) else None
    return None


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


def _scoped_library_row_matches(
    db: VyvarDatabase,
    fp: Path,
    *,
    kind: str,
    id_equipments: int,
    id_telescope: int,
) -> bool:
    row = db.get_calibration_library_row_by_path(fp)
    if row is None:
        return False
    if str(row.get("KIND", "")).strip().lower() != str(kind).strip().lower():
        return False
    try:
        row_eq = row.get("ID_EQUIPMENTS")
        row_tel = row.get("ID_TELESCOPE")
        if row_eq is None or row_tel is None:
            return False
        return int(row_eq) == int(id_equipments) and int(row_tel) == int(id_telescope)
    except (TypeError, ValueError):
        return False


def _calibration_library_search_roots(calibration_library_root: Path) -> list[Path]:
    """Prefer ``<library>/Masters`` then the library root (matches multi-obs master layout)."""
    roots: list[Path] = []
    seen: set[str] = set()
    for r in (calibration_library_root / "Masters", calibration_library_root):
        try:
            rp = r.resolve()
        except OSError:
            continue
        key = str(rp).casefold()
        if key in seen:
            continue
        seen.add(key)
        if rp.is_dir():
            roots.append(rp)
    return roots


@dataclass(slots=True)
class SmartImportPlan:
    source_root: str
    lights_files: list[str]
    dark_files: list[str]
    flat_files: list[str]
    lights_first_fits: str | None
    metadata: dict[str, Any] | None
    scan_rows: list[SmartScanRow]
    dark_master: str | None
    flat_master: str | None
    masterflat_by_filter: dict[str, str | None]
    masterflat_status: dict[str, str]
    missing_flat_filters: list[str]
    masterdark_status: str
    quick_look: bool
    detected_filters: list[str]
    warnings: list[str]
    #: (FILTER|EXPTIME|BINNING) -> group detail dict (paths, scale, ...)
    observation_groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    masterflat_by_obs_key: dict[str, str | None] = field(default_factory=dict)
    dark_master_by_obs_key: dict[str, str | None] = field(default_factory=dict)
    missing_obs_keys: list[str] = field(default_factory=list)
    flat_fallback_prompts: list[dict[str, Any]] = field(default_factory=list)


def _list_fits_files(folder: Path) -> list[Path]:
    """List FITS files in folder without duplicates (Windows-safe)."""
    out: list[Path] = []
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
        out.append(fp)
    return sorted(out)


def _is_empty_or_missing(folder: Path) -> bool:
    return (not folder.exists()) or (not folder.is_dir()) or (len(_list_fits_files(folder)) == 0)


def _mtime_utc(path: Path) -> datetime:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _parse_date_obs(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return None
    text = str(value).strip()
    # Common formats: 2026-03-27T21:15:03.123, 2026-03-27T21:15:03Z
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fits_capture_date_yyyymmdd(path: Path) -> str:
    """Get capture date YYYYMMDD from DATE-OBS across frames (earliest)."""
    try:
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header
        dt = _parse_date_obs(header.get("DATE-OBS") or header.get("DATEOBS"))
        if dt is None:
            return datetime.now(timezone.utc).strftime("%Y%m%d")
        return dt.strftime("%Y%m%d")
    except Exception as exc:  # noqa: BLE001
        # EXC-0089 / EXCEPT-FIX-3 #5 [BEHAVIOR CHANGE: better fallback]: an unreadable FITS
        # is now surfaced (was silent) and the draft is dated from the file mtime, which is
        # strictly closer to the true capture date than "today"; only fall back to now() if
        # even the mtime is unavailable. See docs/VYVAR_EXCEPT_CENSUS.md (EXC-0089).
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().importer_capture_date_fallback += 1
        logger.error("[IMPORT] capture-date read failed for %s: %s", path, exc)
        try:
            return _mtime_utc(path).strftime("%Y%m%d")
        except Exception:  # noqa: BLE001
            return datetime.now(timezone.utc).strftime("%Y%m%d")


def _earliest_capture_datetime_utc(files: list[Path]) -> datetime | None:
    """Earliest DATE-OBS across source files; fallback to earliest mtime."""
    if not files:
        return None
    best: datetime | None = None
    for fp in files:
        dt: datetime | None = None
        try:
            with fits.open(fp, memmap=True) as hdul:
                hdr = hdul[0].header
            dt = _parse_date_obs(hdr.get("DATE-OBS") or hdr.get("DATEOBS"))
        except Exception:  # noqa: BLE001
            dt = None
        if dt is None:
            try:
                dt = _mtime_utc(fp)
            except Exception:  # noqa: BLE001
                dt = None
        if dt is None:
            continue
        if best is None or dt < best:
            best = dt
    return best


def _filter_name_for_calibration_library_flat(flt: str | None) -> str:
    """Normalize FILTER for CALIBRATION_LIBRARY flat rows (must match import matching)."""
    s = str(flt or "").strip()
    if not s or s.lower() in {"unknown", "none", "nan"}:
        return "NoFilter"
    return s


def _master_path_scope_conflicts(
    db: VyvarDatabase | None,
    path: Path,
    *,
    id_equipments: int | None,
    id_telescope: int | None,
) -> bool:
    if db is None:
        return False
    try:
        return db.calibration_library_scope_conflicts(
            path, id_equipments, id_telescope
        )
    except Exception as exc:  # noqa: BLE001
        # EXC-0090 / EXCEPT-FIX-3 #3 [BEHAVIOR CHANGE: fail-open -> fail-closed]: a DB error
        # must NOT be read as "no conflict" (that could silently allow a cross-rig master to
        # register). Assume a conflict so the caller refuses/disambiguates the filename -- the
        # safe direction. See docs/VYVAR_EXCEPT_CENSUS.md (EXC-0090).
        # EXC-0091: T3 -- pure log_event guard (EXCEPT-BULK-2 2026-07-08)
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().calib_scope_conflict_check_fail += 1
        logger.error(
            "[CALIB-LIB] scope-conflict check failed for %s (assuming conflict): %s", path, exc
        )
        return True


def _scoped_master_filename(base_name: str, id_equipments: int | None, id_telescope: int | None) -> str:
    """Disambiguate library master filename when the same stack params belong to another set."""
    stem = base_name
    if stem.lower().endswith((".fits", ".fit", ".fts")):
        stem, ext = base_name.rsplit(".", 1)
        ext = f".{ext}"
    else:
        ext = ".fits"
    if id_equipments is not None and id_telescope is not None:
        return f"{stem}_eq{int(id_equipments)}_tel{int(id_telescope)}{ext}"
    return f"{stem}{ext}"


def _register_master_path_in_calibration_library(
    db: VyvarDatabase | None,
    *,
    kind: str,
    path: Path,
    ncombine: int | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> bool:
    """Register master in CALIBRATION_LIBRARY. Returns False if path belongs to another set."""
    if db is None:
        return True
    if id_equipments is None or id_telescope is None:
        try:
            log_event(
                f"CALIB LIB: refused registration without scope for {path}"
            )
        except Exception:  # noqa: BLE001
            pass
        return False
    try:
        meta = extract_fits_metadata(path, db=db)
        k = str(kind).strip().lower()
        if k not in ("dark", "flat"):
            return True
        flt = "" if k == "dark" else _filter_name_for_calibration_library_flat(str(meta.get("filter", "")))
        return db.register_calibration_library_entry(
            kind=k,
            file_path=path.resolve(),
            xbinning=int(meta.get("binning", 1) or 1),
            exptime=float(meta.get("exposure", 0.0)),
        ccd_temp=_calibration_light_temp_c(path, db=db) if k == "dark" else None,
            filter_name=flt,
            gain=int(meta.get("gain", 0) or 0),
            ncombine=ncombine,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
    except Exception as exc:  # noqa: BLE001
        # EXC-0092 / EXCEPT-FIX-3 #4: registration failure is now surfaced (was a silent False),
        # which otherwise leaves a master invisible to the library (duplicates / wrong matches
        # downstream). Contract unchanged (still returns False). See census EXC-0092.
        logging.warning('[EXC-0093] _looks_like_master: unreadable header -> False (candidate skipped; fail-closed but silent): %s', exc)
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().calib_library_register_fail += 1
        logger.error(
            "[CALIB-LIB] library registration failed (kind=%s) for %s: %s", kind, path, exc
        )
        return False


def _sanitize_token(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(" ", "")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    return text or "Unknown"


def _format_exp_seconds(exposure: float) -> str:
    if abs(exposure - round(exposure)) < 1e-6:
        return str(int(round(exposure)))
    return f"{exposure:.2f}".rstrip("0").rstrip(".")


def _format_temp(temp: float) -> str:
    # Keep sign, avoid extra dots in filename
    return f"{temp:.1f}".replace(".", "p")


def _format_temp_deg_for_name(temp: float) -> str:
    """Format temperature token for filenames like -15deg."""
    if abs(temp - round(temp)) < 1e-6:
        return f"{int(round(temp))}deg"
    return f"{temp:.1f}".rstrip("0").rstrip(".") + "deg"


def _looks_like_master(fp: Path) -> bool:
    name = fp.name.upper()
    if name.startswith(("MD_", "MF_", "MASTERDARK", "MASTERFLAT", "DARK_", "FLAT_")):
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


def _first_fits_in_dir(folder: Path) -> Path | None:
    files = _list_fits_files(folder)
    return files[0] if files else None


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


def _find_lights_subdirectory(session_root: Path) -> Path | None:
    """Return ``Lights`` / ``lights`` child if present (case-insensitive folder name)."""
    if not session_root.is_dir():
        return None
    return next(
        (d for d in session_root.iterdir() if d.is_dir() and d.name.lower() == "lights"),
        None,
    )


def _imaging_kind_for_file(fp: Path, db: VyvarDatabase | None) -> str:
    """Classify one FITS as light/dark/flat/unknown using header cache or primary header."""
    try:
        st = fp.stat()
    except OSError:
        return "unknown"
    if db is not None:
        row = db.fits_header_cache_get_if_fresh(fp, file_size=int(st.st_size), mtime=float(st.st_mtime))
        if row is not None:
            return _classify_imagetyp(str(row.get("IMAGETYP") or ""))
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        imagetyp_raw = str(
            hdr.get("IMAGETYP") or hdr.get("FRAME") or hdr.get("IMTYPE") or ""
        )
        return _classify_imagetyp(imagetyp_raw)
    except Exception as exc:  # noqa: BLE001
        # EXC-0094 / EXCEPT-FIX-3 #6: classification failure is surfaced (was silent); an
        # "unknown" frame is excluded from lights. Contract unchanged. See census EXC-0094.
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().importer_imagetyp_read_fail += 1
        logger.error("[IMPORT] IMAGETYP classification failed for %s: %s", fp, exc)
        return "unknown"


def _list_top_level_light_fits(source_dir: Path, db: VyvarDatabase | None = None) -> list[Path]:
    """FITS directly under ``source_dir`` whose IMAGETYP classifies as light (non-recursive)."""
    out: list[Path] = []
    if not source_dir.is_dir():
        return out
    for fp in sorted(source_dir.iterdir(), key=lambda p: str(p).casefold()):
        if not fp.is_file() or not path_suffix_is_fits(fp):
            continue
        if _imaging_kind_for_file(fp, db) == "light":
            out.append(fp)
    return out


def _resolve_session_lights(
    root: Path, *, db: VyvarDatabase | None = None
) -> tuple[Path, list[Path]]:
    """Resolve light FITS: prefer case-insensitive ``lights`` subfolder, else top-level light frames in ``root``.

    Returns:
        (container_dir, fits_paths) - ``container_dir`` is the ``lights`` folder or ``root`` for flat layout.

    Raises:
        FileNotFoundError: if neither yields light FITS.
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Session root missing or not a directory: {root}")
    lights_sub = _find_lights_subdirectory(root)
    if lights_sub is not None:
        files = _list_fits_files(lights_sub)
        if files:
            return lights_sub, files
    files = _list_top_level_light_fits(root, db=db)
    if files:
        return root, files
    raise FileNotFoundError(
        "No light frames: add a 'lights' folder with FITS, or place OBJECT/science/light "
        f"IMAGETYP FITS directly under: {root}"
    )


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


def _master_kind_matches(fp: Path, kind: str) -> bool:
    """Ensure we don't pick Dark as Flat (and vice versa) from the library."""
    kind = kind.lower()
    name = fp.name.upper()
    # Strong filename hints
    if kind == "dark" and ("DARK" in name or name.startswith("MD_")):
        return True
    if kind == "flat" and ("FLAT" in name or name.startswith("MF_")):
        return True

    # Header hints
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        imagetyp = str(hdr.get("IMAGETYP") or hdr.get("FRAME") or hdr.get("IMTYPE") or "").lower()
        hist = str(hdr.get("HISTORY") or "").lower()
    except Exception:  # noqa: BLE001
        imagetyp = ""
        hist = ""

    if kind == "dark" and ("dark" in imagetyp or "dark" in hist):
        return True
    if kind == "flat" and ("flat" in imagetyp or "flat" in hist):
        return True

    return False


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
    gain = int(meta.get("gain", 0))
    temp_s = f"{float(meta.get('temp', 0.0)):g}"
    binning = int(meta.get("binning", 1))
    flt = str(meta.get("filter", "Unknown"))
    base = f"Exp={exp:g}s, Gain={gain}G, Temp={temp_s}C, Bin={binning}"
    return f"{base}, Filter={flt}" if include_filter else base


def _find_matching_master_in_library(
    calibration_library_root: Path,
    *,
    kind: str,  # "dark" | "flat"
    exp: float,
    gain: int,
    binning: int,
    temp: float | None,
    flt: str | None,
    db: VyvarDatabase | None = None,
    search_roots: list[Path] | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
    temp_tolerance: float = 0.5,
) -> Path | None:
    if id_equipments is None or id_telescope is None:
        try:
            log_event(
                f"CALIB LIB: cannot match {kind} master - missing equipment/telescope scope"
            )
        except Exception:  # noqa: BLE001
            # EXC-0096: T3 -- pure log_event guard (EXCEPT-BULK-2 2026-07-08)
            pass
        return None
    eq_id = int(id_equipments)
    tel_id = int(id_telescope)
    if kind == "dark":
        if temp is None or not math.isfinite(float(temp)):
            try:
                log_event("CALIB LIB: cannot match dark - light CCD_TEMP unknown")
            except Exception:  # noqa: BLE001
                # EXC-0097: T3 -- pure log_event guard (EXCEPT-BULK-2 2026-07-08)
                pass
            return None
    flt_key = _filter_name_for_calibration_library_flat(flt) if kind == "flat" else ""
    if db is not None:
        try:
            hit = db.find_best_calibration_library_path(
                kind=kind,
                xbinning=int(binning),
                exptime=float(exp),
                ccd_temp=float(temp) if kind == "dark" else None,
                filter_name=flt_key,
                gain=int(gain),
                prefer_unbinned_master=True,
                id_equipments=eq_id,
                id_telescope=tel_id,
                temp_tolerance=float(temp_tolerance),
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning('[EXC-0098] master candidate metadata fail -> continue -> silently falls back to worse/older master: %s', exc)
            hit = None
        if hit:
            p_hit = Path(hit)
            if p_hit.is_file() and _master_kind_matches(p_hit, kind) and _looks_like_master(p_hit):
                return p_hit

    roots = search_roots if search_roots else [calibration_library_root]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.fits", "*.fit", "*.fts", "*.FITS", "*.FIT", "*.FTS"):
            candidates.extend(root.rglob(ext))
    seen_c: set[str] = set()
    cand_u: list[Path] = []
    for fp in candidates:
        try:
            ck = str(fp.resolve()).casefold()
        except OSError:
            continue
        if ck in seen_c:
            continue
        seen_c.add(ck)
        cand_u.append(fp)
    candidates = cand_u
    if not candidates:
        return None

    kind_upper = kind.upper()
    best: Path | None = None
    best_mtime = -1.0

    for fp in candidates:
        name = fp.name.upper()
        if kind_upper not in name and not name.startswith(("MD_", "MF_")):
            # still allow if header indicates master, but prefer name match
            pass

        try:
            meta = extract_fits_metadata(fp, db=db)
        except Exception:  # noqa: BLE001
            continue

        if db is None or not _scoped_library_row_matches(
            db,
            fp,
            kind=kind,
            id_equipments=eq_id,
            id_telescope=tel_id,
        ):
            continue

        if kind == "dark":
            if float(meta.get("exposure", -1.0)) != float(exp):
                continue
            master_temp = _calibration_light_temp_c(fp, db=db)
            if master_temp is None:
                continue
            if abs(float(master_temp) - float(temp)) > float(temp_tolerance):
                continue
        else:
            flt_norm = _filter_name_for_calibration_library_flat(flt)
            if str(meta.get("filter", "")).strip() != str(flt_norm).strip():
                continue

        if int(meta.get("gain", 0)) != int(gain):
            continue
        mb = int(meta.get("binning", 0) or 0)
        if mb != int(binning) and not (int(binning) > 1 and mb == 1):
            continue

        if not _looks_like_master(fp):
            continue
        if not _master_kind_matches(fp, kind):
            continue

        mtime = os.path.getmtime(fp)
        if mtime > best_mtime:
            best = fp
            best_mtime = mtime

    return best


def _reset_master_age_mtime_warnings() -> None:
    reset_master_age_mtime_warnings()


def _age_days(
    path: Path,
    *,
    warnings: list[str] | None = None,
) -> float | None:
    """Master age in days -- same clock as library UI (:func:`calibration.get_master_age_days`)."""
    try:
        info = resolve_master_age(path, warnings=warnings)
    except OSError:
        return None
    return float(info.age_days)


def _find_best_masterflat_for_filter(
    calibration_library_root: Path,
    *,
    flt: str,
    binning: int,
    validity_days: int,
    db: VyvarDatabase | None = None,
    exp: float | None = None,
    gain: int | None = None,
    temp: float | None = None,
    search_roots: list[Path] | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
    temp_tolerance: float = 0.5,
    warnings: list[str] | None = None,
) -> tuple[Path | None, str]:
    """Return best masterflat and a UI-friendly status string."""
    roots = search_roots or _calibration_library_search_roots(calibration_library_root)
    flt_norm = _filter_name_for_calibration_library_flat(flt)
    if (
        db is not None
        and gain is not None
        and id_equipments is not None
        and id_telescope is not None
    ):
        try:
            hit = db.find_best_calibration_library_path(
                kind="flat",
                xbinning=int(binning),
                exptime=float(exp) if exp is not None else 0.0,
                ccd_temp=None,
                filter_name=flt_norm,
                gain=int(gain),
                prefer_unbinned_master=True,
                id_equipments=id_equipments,
                id_telescope=id_telescope,
                temp_tolerance=float(temp_tolerance),
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning('[EXC-0099] flat candidate metadata fail -> continue -> same as above for flats: %s', exc)
            hit = None
        if hit:
            p_hit = Path(hit)
            if p_hit.is_file() and _master_kind_matches(p_hit, "flat") and _looks_like_master(p_hit):
                age = _age_days(p_hit, warnings=warnings)
                if age is not None and age <= validity_days:
                    return p_hit, f"MasterFlat (Filter {flt}): [OK] library DB ({int(age)} days old)"
                if age is not None:
                    return (
                        None,
                        f"MasterFlat (Filter {flt}): ! library DB expirovane ({int(age)} dni) - vygeneruj novy",
                    )
                return p_hit, f"MasterFlat (Filter {flt}): [OK] library DB"

    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.fits", "*.fit", "*.fts", "*.FITS", "*.FIT", "*.FTS"):
            candidates.extend(root.rglob(ext))
    seen_cf: set[str] = set()
    cand_cf: list[Path] = []
    for fp in candidates:
        try:
            ck = str(fp.resolve()).casefold()
        except OSError:
            continue
        if ck in seen_cf:
            continue
        seen_cf.add(ck)
        cand_cf.append(fp)
    candidates = cand_cf
    best_fresh: tuple[Path, float] | None = None  # (path, age_days)
    best_any: tuple[Path, float] | None = None

    for fp in candidates:
        try:
            meta = extract_fits_metadata(fp, db=db)
        except Exception:  # noqa: BLE001
            continue
        if str(meta.get("filter", "")).strip() != str(flt).strip():
            continue
        mb = int(meta.get("binning", -1))
        if mb != int(binning) and not (int(binning) > 1 and mb == 1):
            continue
        if not _looks_like_master(fp):
            continue
        if not _master_kind_matches(fp, "flat"):
            continue
        age = _age_days(fp, warnings=warnings)
        if age is None:
            continue
        age_i = int(age)
        if best_any is None or age < best_any[1]:
            best_any = (fp, age)
        if age <= validity_days:
            if best_fresh is None or age < best_fresh[1]:
                best_fresh = (fp, age)

    if best_fresh:
        fp, age = best_fresh
        return fp, f"MasterFlat (Filter {flt}): [OK] found ({int(age)} days old)"
    if best_any:
        _fp, age = best_any
        return (
            None,
            f"MasterFlat (Filter {flt}): ! len expirovany master ({int(age)} dni) - vygeneruj novy",
        )
    return None, f"MasterFlat (Filter {flt}): [X] MISSING!"


def _write_master_to_library(
    *,
    kind: str,  # "dark" | "flat"
    files: list[Path],
    calibration_library_root: Path,
    target_binning: int | None = None,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> Path:
    if kind not in ("dark", "flat"):
        raise ValueError("kind must be 'dark' or 'flat'")
    if not files:
        raise ValueError("No files to build master from.")

    meta0 = extract_fits_metadata(files[0], db=db)
    exp_s = _format_exp_seconds(float(meta0["exposure"]))
    gain = int(meta0.get("gain", 0))
    temp_token = _format_temp_deg_for_name(float(meta0["temp"]))
    raw_binning = int(meta0["binning"])
    # CalibrationLibrary stores a **native** stack of calibration frames. Software resampling to match
    # light XBINNING happens only in :func:`calibration.get_processed_master` at calibrate time.
    _ = target_binning  # deprecated: do not bin-down masters in the library to match lights
    binning = raw_binning
    flt = _sanitize_token(meta0.get("filter", "Unknown"))
    date_yyyymmdd = min(_fits_capture_date_yyyymmdd(fp) for fp in files)

    type_token = "Dark" if kind == "dark" else "Flat"
    filter_token = "Dark" if kind == "dark" else flt

    filename = f"{type_token}_{exp_s}s_{filter_token}_{gain}G_{temp_token}_Bin{binning}_{date_yyyymmdd}.fits"
    out_root = calibration_library_root
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / filename

    # Reuse only when the on-disk file is not registered to a different equipment/telescope set.
    if out_path.exists():
        if _master_path_scope_conflicts(
            db,
            out_path,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        ):
            out_path = out_root / _scoped_master_filename(
                filename, id_equipments, id_telescope
            )
        else:
            return out_path

    files_for_stack = files
    if kind == "dark":
        master, header = _generate_master_dark(files_for_stack)
    else:
        master, header = _generate_master_flat(files_for_stack)

    # Binning keywords reflect the **calibration frames** used (prefer 1x1 for full-resolution masters).
    header["BINNING"] = binning
    header["XBINNING"] = binning
    header["YBINNING"] = binning
    header["VY_MBLIB"] = (
        1,
        "VYVAR: native stack in CalibrationLibrary; resample to light XBINNING at calibrate",
    )

    if kind == "dark":
        header["HISTORY"] = "VYVAR: Master Dark (mean stack)"
    else:
        header["HISTORY"] = "VYVAR: Master Flat (median stack; norm at calibrate)"
        header["VYFLNRD"] = (
            1,
            "Median normalization deferred to calibrate after resample to light binning",
        )
    dt_src = _earliest_capture_datetime_utc(files)
    if dt_src is not None:
        dt_iso = dt_src.strftime("%Y-%m-%dT%H:%M:%S")
        header["DATE-OBS"] = (dt_iso, "Earliest source raw capture time (UTC)")
        header["VY_CDATE"] = (dt_iso, "VYVAR source capture datetime (UTC)")
    header["NCOMBINE"] = len(files)
    fits.writeto(out_path, master.astype(np.float32), header=header, overwrite=True)
    if kind == "dark":
        try:
            from config import AppConfig
            from photometry import write_dark_bpm_json

            _sig = float(AppConfig().bpm_dark_mad_sigma)
            _nb = int(header.get("XBINNING") or header.get("BINNING") or binning)
            write_dark_bpm_json(out_path, master, mad_sigma=_sig, native_binning=_nb)
        except Exception as exc:  # noqa: BLE001
            # EXC-0100 / EXCEPT-FIX-3 #2 (T1): the BPM sidecar stays best-effort (the master
            # dark is still created), but the failure is now loud -- otherwise photometry runs
            # without the bad-pixel map with no trace. See census EXC-0100.
            logging.warning("[EXC-0101] sample metadata fail -> None ('no existing master') -> duplicate master creation: %s", exc)
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().dark_bpm_sidecar_write_fail += 1
            logger.error("[IMPORT] dark BPM sidecar write failed for %s: %s", out_path, exc)
    _register_master_path_in_calibration_library(
        db,
        kind=kind,
        path=out_path,
        ncombine=int(header.get("NCOMBINE", len(files))),
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    return out_path


def _find_existing_master_for_raw_set(
    calibration_library_root: Path,
    *,
    kind: str,  # "dark" | "flat"
    sample_file: Path,
    target_binning: int | None = None,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> Path | None:
    """Avoid duplicate master creation: find existing master in library by params."""
    try:
        meta0 = extract_fits_metadata(sample_file, db=db)
    except Exception:  # noqa: BLE001
        return None

    exp = float(meta0.get("exposure", 0.0))
    gain = int(meta0.get("gain", 0))
    raw_binning = int(meta0.get("binning", 1))
    _ = target_binning
    binning = raw_binning
    temp = _calibration_light_temp_c(sample_file, db=db)
    flt = str(meta0.get("filter", "")).strip() if kind == "flat" else None

    cr = Path(calibration_library_root)
    return _find_matching_master_in_library(
        cr,
        kind=kind,
        exp=exp,
        gain=gain,
        binning=binning,
        temp=temp,
        flt=flt,
        db=db,
        search_roots=_calibration_library_search_roots(cr),
        id_equipments=id_equipments,
        id_telescope=id_telescope,
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
) -> SmartImportPlan:
    """Scan source for lights/darks/flats and decide calibration paths.

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
    _reset_master_age_mtime_warnings()

    if not root.exists() or not root.is_dir():
        scan_rows.append(SmartScanRow("Lights", "missing", 0, ""))
        return SmartImportPlan(
            source_root=str(root),
            lights_files=[],
            dark_files=[],
            flat_files=[],
            lights_first_fits=None,
            metadata=None,
            scan_rows=scan_rows,
            dark_master=None,
            flat_master=None,
            masterflat_by_filter={},
            masterflat_status={},
            missing_flat_filters=[],
            masterdark_status="MasterDark: n/a (invalid source root)",
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
        {flt for flt in (_read_filter(fp, db=db) for fp in lights_files[:500]) if flt}
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
            lights_first_fits=None,
            metadata=None,
            scan_rows=scan_rows,
            dark_master=None,
            flat_master=None,
            masterflat_by_filter={},
            masterflat_status={},
            missing_flat_filters=[],
            masterdark_status="MasterDark: n/a (no Lights)",
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
        f"Scan Source - prvy light v zozname: {first_light.name} | "
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

    exp = float(metadata["exposure"])
    gain = int(metadata.get("gain", 0))
    binning = int(metadata["binning"])
    temp = _calibration_light_temp_c(first_light, db=db)
    flt = str(metadata.get("filter", "Unknown"))

    calib_root = Path(calibration_library_root)
    cal_roots = _calibration_library_search_roots(calib_root)

    observation_groups: dict[str, dict[str, Any]] = {}
    for fp in lights_files[:8000]:
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
                "gain": int(meta_i.get("gain", 0)),
                "temp": _calibration_light_temp_c(fp, db=db),
                "representative_light": str(fp),
                "light_paths": [],
                "plate_scale_arcsec_per_px": scale,
            }
        observation_groups[gk]["light_paths"].append(str(fp))

    masterflat_by_obs_key: dict[str, str | None] = {}
    masterflat_status: dict[str, str] = {}
    dark_master_by_obs_key: dict[str, str | None] = {}
    missing_obs_keys: list[str] = []
    for gk, g in sorted(observation_groups.items(), key=lambda x: x[0]):
        fp_best, status = _find_best_masterflat_for_filter(
            calib_root,
            flt=g["filter"],
            binning=int(g["binning"]),
            validity_days=masterflat_validity_days,
            db=db,
            exp=float(g["exposure_s"]),
            gain=int(g["gain"]),
            temp=g["temp"],
            search_roots=cal_roots,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
            temp_tolerance=temp_tol,
            warnings=warnings,
        )
        masterflat_by_obs_key[gk] = str(fp_best) if fp_best is not None else None
        masterflat_status[gk] = status
        if fp_best is None:
            missing_obs_keys.append(gk)

        d_found = _find_matching_master_in_library(
            calib_root,
            kind="dark",
            exp=float(g["exposure_s"]),
            gain=int(g["gain"]),
            binning=int(g["binning"]),
            temp=g["temp"],
            flt=None,
            db=db,
            search_roots=cal_roots,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
            temp_tolerance=temp_tol,
        )
        if d_found is not None:
            age_d = _age_days(d_found, warnings=warnings)
            if age_d is not None and age_d > masterdark_validity_days:
                dark_master_by_obs_key[gk] = None
                warnings.append(
                    f"MasterDark pre skupinu {gk} je expirovany ({int(age_d)} dni) - vygeneruj novy."
                )
            else:
                dark_master_by_obs_key[gk] = str(d_found)
        else:
            dark_master_by_obs_key[gk] = None

    masterflat_by_filter: dict[str, str | None] = {}
    for gk, g in observation_groups.items():
        fln = g["filter"]
        pth = masterflat_by_obs_key.get(gk)
        if fln not in masterflat_by_filter or (pth is not None and masterflat_by_filter[fln] is None):
            masterflat_by_filter[fln] = pth

    missing_flat_filters: list[str] = sorted(
        {observation_groups[k]["filter"] for k in missing_obs_keys}
    )

    flat_fallback_prompts: list[dict[str, Any]] = []
    for gk in missing_obs_keys:
        g = observation_groups[gk]
        label = f"{g['filter']}-{g['exposure_s']:g}s"
        alts: list[str] = []
        for ok, og in observation_groups.items():
            if ok == gk or masterflat_by_obs_key.get(ok) is None:
                continue
            if int(og["binning"]) != int(g["binning"]):
                continue
            if abs(float(og["exposure_s"]) - float(g["exposure_s"])) > 1e-6:
                continue
            if og["filter"] == g["filter"]:
                continue
            alts.append(ok)
        msg = (
            f"Observation {label} has no Master Flat. "
            "Use a flat from another filter, or skip?"
        )
        flat_fallback_prompts.append(
            {
                "group_key": gk,
                "label": label,
                "message_sk": msg,
                "alternatives": alts,
            }
        )
        warnings.append(msg)

    def _scan_cal(kind: str, files: list[Path]) -> tuple[SmartScanRow, str | None, bool]:
        # Priority: raw on source > master on source > library fallback > draft
        if files:
            any_master = any(_looks_like_master(fp) for fp in files[: min(3, len(files))])
            any_raw = any(not _looks_like_master(fp) for fp in files[: min(3, len(files))])
            if any_raw:
                return (
                    SmartScanRow(kind.title() + "s", "raw", len(files), _params_string(metadata, include_filter=(kind == "flat"))),
                    None,
                    False,
                )
            if any_master:
                master_fp = next((fp for fp in files if _looks_like_master(fp)), files[0])
                return (
                    SmartScanRow(kind.title() + "s", "master", len(files), _params_string(metadata, include_filter=(kind == "flat")), details=str(master_fp)),
                    str(master_fp),
                    False,
                )

        # Library fallback
        found = _find_matching_master_in_library(
            calib_root,
            kind=kind,
            exp=exp,
            gain=gain,
            binning=binning,
            temp=temp,
            flt=flt if kind == "flat" else None,
            db=db,
            search_roots=cal_roots,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
            temp_tolerance=temp_tol,
        )
        if found is None:
            return (
                SmartScanRow(kind.title() + "s", "missing", 0, _params_string(metadata, include_filter=(kind == "flat"))),
                None,
                True,
            )
        validity = masterflat_validity_days if kind == "flat" else masterdark_validity_days
        stt = get_calibration_status(
            found, kind=f"Master {kind.title()}", validity_days=validity, warnings=warnings
        )
        if stt.status == "expired":
            warnings.append(stt.message)
            return (
                SmartScanRow(kind.title() + "s", "expired", 1, _params_string(metadata, include_filter=(kind == "flat")), details=str(found)),
                str(found),
                False,
            )
        return (
            SmartScanRow(kind.title() + "s", "library", 1, _params_string(metadata, include_filter=(kind == "flat")), details=str(found)),
            str(found),
            False,
        )

    dark_row, dark_master, dark_missing = _scan_cal("dark", dark_files)
    flat_row, flat_master, flat_missing = _scan_cal("flat", flat_files)
    scan_rows.extend([dark_row, flat_row])

    # MasterDark status: match exp+gain+bin+temp (from first light meta used above)
    dark_found = _find_matching_master_in_library(
        calib_root,
        kind="dark",
        exp=exp,
        gain=gain,
        binning=binning,
        temp=temp,
        flt=None,
        db=db,
        search_roots=cal_roots,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    # Ensure downstream steps (calibration) have an actual master path if found in library
    if dark_master is None and dark_found is not None:
        dark_master = str(dark_found)
    if dark_found is None and not dark_files:
        masterdark_status = "MasterDark: [X] MISSING!"
    elif dark_found is None:
        masterdark_status = "MasterDark: raw on source (will build if requested)"
    else:
        stt_dark = get_calibration_status(
            dark_found,
            kind="Master Dark",
            validity_days=masterdark_validity_days,
            warnings=warnings,
        )
        if stt_dark.status == "expired":
            masterdark_status = f"MasterDark: ! found but expired ({stt_dark.age_days} days old)"
            warnings.append(stt_dark.message)
        else:
            masterdark_status = f"MasterDark: [OK] found ({stt_dark.age_days} days old)"

    # Decision:
    # - Missing MasterDark => full Quick Look (Draft)
    # - Missing some MasterFlats => partial draft for those filters (not full Quick Look)
    quick_look = dark_missing
    if dark_missing:
        warnings.append("No suitable MasterDark found -> Quick Look Mode (Draft).")
    return SmartImportPlan(
        source_root=str(root),
        lights_files=[str(p) for p in lights_files],
        dark_files=[str(p) for p in dark_files],
        flat_files=[str(p) for p in flat_files],
        lights_first_fits=str(first_light),
        metadata=metadata,
        scan_rows=scan_rows,
        dark_master=dark_master,
        flat_master=flat_master,
        masterflat_by_filter=masterflat_by_filter,
        masterflat_status=masterflat_status,
        missing_flat_filters=missing_flat_filters,
        masterdark_status=masterdark_status,
        quick_look=quick_look,
        detected_filters=detected_filters,
        warnings=warnings,
        observation_groups=observation_groups,
        masterflat_by_obs_key=masterflat_by_obs_key,
        dark_master_by_obs_key=dark_master_by_obs_key,
        missing_obs_keys=missing_obs_keys,
        flat_fallback_prompts=flat_fallback_prompts,
    )


def generate_master_dark_from_source_dir(
    *,
    source_dir: str | Path,
    calibration_library_root: str | Path,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> tuple[Path | None, list[str]]:
    """Master dark from raw frames (recursive, IMAGETYP): per-pixel mean stack. Filename per :func:`_write_master_to_library`."""
    messages: list[str] = []
    root = Path(source_dir)
    out_root = Path(calibration_library_root)
    if not root.is_dir():
        return None, [f"[X] Adresar neexistuje: {root}"]
    files_by_type = _collect_fits_by_type(root, db=db)
    dark_raw = [fp for fp in files_by_type.get("dark", []) if not _looks_like_master(fp)]
    if not dark_raw:
        return None, [
            "[X] V zadanom adresari sa nenasli surove dark snimky (ocakava sa IMAGETYP obsahujuci 'dark', nie hotovy master)."
        ]
    existing = _find_existing_master_for_raw_set(
        out_root,
        kind="dark",
        sample_file=dark_raw[0],
        target_binning=None,
        db=db,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    if existing is not None and _master_path_scope_conflicts(
        db,
        existing,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    ):
        existing = None
    if existing is not None:
        messages.append(f"i Master Dark uz v kniznici existuje: {existing.name}")
        if _register_master_path_in_calibration_library(
            db,
            kind="dark",
            path=existing,
            ncombine=None,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        ):
            return existing, messages
        messages.append(
            "! Subor existuje, ale registracia pre iny set - generujem novy master pre zvolenu kameru/teleskop."
        )
        existing = None
    out = _write_master_to_library(
        kind="dark",
        files=dark_raw,
        calibration_library_root=out_root,
        target_binning=None,
        db=db,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    messages.append(f"[OK] Master Dark vytvoreny: {out.name} ({len(dark_raw)} snimok)")
    return out, messages


def generate_master_flat_from_source_dir(
    *,
    source_dir: str | Path,
    calibration_library_root: str | Path,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> tuple[Path | None, list[str]]:
    """Master flat: per-pixel **median** stack, then normalization. Naming ako v kniznici."""
    messages: list[str] = []
    root = Path(source_dir)
    out_root = Path(calibration_library_root)
    if not root.is_dir():
        return None, [f"[X] Adresar neexistuje: {root}"]
    files_by_type = _collect_fits_by_type(root, db=db)
    flat_raw = [fp for fp in files_by_type.get("flat", []) if not _looks_like_master(fp)]
    if not flat_raw:
        return None, [
            "[X] V zadanom adresari sa nenasli surove flat snimky (IMAGETYP obsahujuci 'flat', nie master)."
        ]
    existing = _find_existing_master_for_raw_set(
        out_root,
        kind="flat",
        sample_file=flat_raw[0],
        target_binning=None,
        db=db,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    if existing is not None and _master_path_scope_conflicts(
        db,
        existing,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    ):
        existing = None
    if existing is not None:
        messages.append(f"i Master Flat uz v kniznici existuje: {existing.name}")
        if _register_master_path_in_calibration_library(
            db,
            kind="flat",
            path=existing,
            ncombine=None,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        ):
            return existing, messages
        messages.append(
            "! Subor existuje, ale registracia pre iny set - generujem novy master pre zvolenu kameru/teleskop."
        )
        existing = None
    out = _write_master_to_library(
        kind="flat",
        files=flat_raw,
        calibration_library_root=out_root,
        target_binning=None,
        db=db,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    messages.append(f"[OK] Master Flat vytvoreny: {out.name} ({len(flat_raw)} snimok)")
    return out, messages


def _copy_fits_folder(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for fp in _list_fits_files(src):
        shutil.copy2(fp, dst / fp.name)
        count += 1
    return count


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

    metadata = extract_fits_metadata(lights_files[0], db=pipeline.db, app_config=pipeline.config)
    from osc_extract import validate_bayer_crosscheck

    _import_warnings: list[str] = []
    for fp in lights_files[:500]:
        try:
            meta_x = extract_fits_metadata(fp, db=pipeline.db, app_config=pipeline.config)
        except Exception:  # noqa: BLE001
            continue
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
    scanning_id = pipeline.db.find_or_create_scanning_id(metadata)

    # Ingestion creates DRAFT only (Session ID created after astrometry).
    missing_obs_keys_set: set[str] = set(getattr(plan, "missing_obs_keys", []) or [])
    draft_filters: set[str] = set(getattr(plan, "missing_flat_filters", []) or [])
    draft_id = pipeline.db.create_draft(
        {
            "id_equipments": int(id_equipment),
            "id_telescope": int(id_telescope),
            "id_location": int(_id_loc),
            "id_scanning": int(scanning_id),
            "observation_start_jd": float(metadata["jd_start"]),
            "is_calibrated": (len(draft_filters) == 0 and not plan.quick_look),
        }
    )

    try:
        _comb = pipeline.db.get_combined_metadata(lights_files[0], int(draft_id))
        if _comb.get("focal_length_mm") is None or _comb.get("pixel_effective_um") is None:
            plan.warnings.append(str(DraftTechnicalMetadataError(int(draft_id))))
    except Exception:  # noqa: BLE001
        # EXC-0103: T3 -- guard suppresses appending a non-fatal DraftTechnicalMetadataError warning (EXCEPT-BULK-2 2026-07-08)
        pass

    archive_session = Path(pipeline.config.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    # Partial draft if some filters missing masterflat
    if plan.quick_look:
        target = archive_session / "non_calibrated"
        lights_root = target / "lights"
        evidence: list[dict[str, Any]] = []
        for fp in lights_files:
            flt = _sanitize_filter_folder(_read_filter(fp, db=pipeline.db))
            try:
                meta_l = extract_fits_metadata(fp, db=pipeline.db, app_config=pipeline.config)
                gk = observation_group_key(
                    flt, float(meta_l.get("exposure", 0.0)), int(meta_l.get("binning", 1) or 1)
                )
                sc_i = int(pipeline.db.find_or_create_scanning_id(meta_l))
            except Exception:  # noqa: BLE001
                gk = observation_group_key(flt, 0.0, 1)
                sc_i = None
            dst_dir = lights_root / observation_group_folder_name(gk)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = _dst_path_with_length_limit(dst_dir, fp.name)
            _safe_copy2(fp, dst_path)
            evidence.append(
                {
                    "file_path": str(dst_path),
                    "imagetyp": "light",
                    "filter": flt,
                    "observation_group_key": gk,
                    "id_scanning": sc_i,
                    "is_calibrated": 0,
                    "calib_type": "RAW_NON_CALIBRATED",
                }
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
            dark_path="",
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

    # If some filters are missing MasterFlat, import those lights into non_calibrated
    draft_root = archive_session / "non_calibrated"
    draft_lights_root = draft_root / "lights"

    # Flexible sorting: even if mixed folders exist, we sort per-file by IMAGETYP
    evidence: list[dict[str, Any]] = []
    for fp in lights_files:
        flt = _sanitize_filter_folder(_read_filter(fp, db=pipeline.db))
        try:
            meta_l = extract_fits_metadata(fp, db=pipeline.db, app_config=pipeline.config)
            gk = observation_group_key(
                flt, float(meta_l.get("exposure", 0.0)), int(meta_l.get("binning", 1) or 1)
            )
            sc_i = int(pipeline.db.find_or_create_scanning_id(meta_l))
        except Exception:  # noqa: BLE001
            gk = observation_group_key(flt, 0.0, 1)
            sc_i = None
        is_draft = gk in missing_obs_keys_set or flt in draft_filters or (
            flt == "NoFilter" and "NoFilter" in draft_filters
        )
        dst_base = draft_lights_root if is_draft else lights_root
        dst_dir = dst_base / observation_group_folder_name(gk)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = _dst_path_with_length_limit(dst_dir, fp.name)
        _safe_copy2(fp, dst_path)
        evidence.append(
            {
                "file_path": str(dst_path),
                "imagetyp": "light",
                "filter": flt,
                "observation_group_key": gk,
                "id_scanning": sc_i,
                "is_calibrated": 0 if is_draft else 1,
                "calib_type": "RAW_PENDING_CALIBRATION" if is_draft else "MASTER_PIPELINE",
            }
        )
    for fp in [Path(p) for p in plan.dark_files]:
        dst_path = _dst_path_with_length_limit(darks_dst, fp.name)
        _safe_copy2(fp, dst_path)
        evidence.append({"file_path": str(dst_path), "imagetyp": "dark", "filter": ""})
    for fp in [Path(p) for p in plan.flat_files]:
        dst_path = _dst_path_with_length_limit(flats_dst, fp.name)
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
            f"dark={plan.dark_master};flat_by_filter={plan.masterflat_by_filter};"
            f"flat_by_obs_key={getattr(plan, 'masterflat_by_obs_key', {})};"
            f"dark_by_obs_key={getattr(plan, 'dark_master_by_obs_key', {})}"
        ),
        imported_at=now_utc.isoformat(timespec="seconds"),
        import_warnings="\n".join(plan.warnings) if plan.warnings else None,
        is_calibrated=(len(draft_filters) == 0),
        archive_path=str(archive_session),
    )
    pipeline.db.insert_draft_files(draft_id, evidence)
    return ImportResult(
        draft_id=draft_id,
        observation_id=None,
        lights_path=str(lights_root),
        dark_path=str(plan.dark_master or ""),
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
    img8 = (scaled * 255).astype(np.uint8)

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

# Large stacks use a temp memmap cube + Y slabs.
_STACK_FULL_RAM_BYTES = 512 * 1024 * 1024
_STACK_LARGE_N = 8
_STACK_SLAB_RAM_BYTES = 256 * 1024 * 1024


def _to_float32_frame(data: np.ndarray) -> np.ndarray:
    """Use a float32 view when already float32; copy only for integer / other dtypes."""
    a = np.asanyarray(data)
    if a.dtype == np.float32:
        return a
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    return np.asarray(a, dtype=np.float32)


def _apply_binning_frame(arr: np.ndarray, bin_factor: int) -> np.ndarray:
    """Average-block bin ``arr`` to match light binning (integer factor >= 1)."""
    a = _to_float32_frame(arr)
    if bin_factor <= 1:
        return a
    h, w = a.shape[:2]
    h2 = h - (h % bin_factor)
    w2 = w - (w % bin_factor)
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"Cannot bin image with shape {a.shape} by factor {bin_factor}")
    if h2 != h or w2 != w:
        a = a[:h2, :w2]
        h, w = a.shape[:2]
    return a.reshape(h // bin_factor, bin_factor, w // bin_factor, bin_factor).mean(axis=(1, 3)).astype(
        np.float32
    )


def _combine_stack_mean(stack: np.ndarray) -> np.ndarray:
    """Average stack for darks: ``stack`` (n, ny, nx) -> per-pixel mean along axis 0 (float32).

    Intentionally **not** sigma-clipped - preserves hot pixels and optimizes SNR for bias/dark subtraction.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected stack (n, ny, nx), got {stack.shape}")
    out = np.nanmean(stack, axis=0).astype(np.float32)
    bad = ~np.isfinite(out)
    if np.any(bad):
        good = out[np.isfinite(out)]
        fill = float(np.median(good)) if good.size else 0.0
        out[bad] = fill
    return out


def _combine_stack_median(stack: np.ndarray) -> np.ndarray:
    """Median stack for flats: ``stack`` (n, ny, nx) -> per-pixel median along axis 0 (float32).

    Rejects transient dust/glints without sigma-clipping artefacts; vignetting is kept until
    calibrate-time normalization in :func:`calibration.get_processed_master`.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected stack (n, ny, nx), got {stack.shape}")
    out = np.nanmedian(stack, axis=0).astype(np.float32)
    bad = ~np.isfinite(out)
    if np.any(bad):
        good = out[np.isfinite(out)]
        fill = float(np.median(good)) if good.size else 0.0
        out[bad] = fill
    return out


def _stack_calibration_frames(
    files: list[Path],
    *,
    combine: str,
    bin_factor: int = 1,
    kind: str = "dark",
) -> tuple[np.ndarray, fits.Header]:
    """Stack calibration frames: **mean** for dark, **median** for flat; optional memmap cube for RAM.

    No sigma-clipping - see module comment above.
    """
    if combine not in ("mean", "median"):
        raise ValueError(f"combine must be 'mean' or 'median', got {combine!r}")
    if combine == "mean" and kind != "dark":
        raise ValueError("Mean combine is only used for master dark.")
    if combine == "median" and kind != "flat":
        raise ValueError("Median combine is only used for master flat.")
    if not files:
        raise ValueError("No FITS files provided for stacking.")

    with fits.open(files[0], memmap=False) as hdul0:
        header0 = hdul0[0].header.copy()
        a0b = _apply_binning_frame(_to_float32_frame(hdul0[0].data), bin_factor)

    n = len(files)
    h0, w0 = int(a0b.shape[0]), int(a0b.shape[1])
    cube_bytes = n * h0 * w0 * 4
    use_memmap_cube = n > _STACK_LARGE_N or cube_bytes > _STACK_FULL_RAM_BYTES

    if combine == "mean":
        log_event(
            f"Master {kind}: mean stack (per-pixel) N={n} frames, "
            f"{'memmap cube' if use_memmap_cube else 'RAM cube'} "
            f"({h0}x{w0}, bin={bin_factor})."
        )
    else:
        log_event(
            f"Master {kind}: median stack (per-pixel) N={n} frames, "
            f"{'memmap cube' if use_memmap_cube else 'RAM cube'} "
            f"({h0}x{w0}, bin={bin_factor})."
        )

    def _combine_block(block: np.ndarray) -> np.ndarray:
        if combine == "mean":
            return _combine_stack_mean(block)
        return _combine_stack_median(block)

    if not use_memmap_cube:
        stack = np.empty((n, h0, w0), dtype=np.float32)
        stack[0] = a0b
        for i in range(1, n):
            with fits.open(files[i], memmap=False) as hdul:
                stack[i] = _apply_binning_frame(_to_float32_frame(hdul[0].data), bin_factor)
        master = _combine_block(stack)
    else:
        fd, tmp_path = tempfile.mkstemp(prefix="vyvar_cal_stack_", suffix=".dat")
        os.close(fd)
        try:
            cube = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(n, h0, w0))
            cube[0] = a0b
            for i in range(1, n):
                with fits.open(files[i], memmap=False) as hdul:
                    cube[i] = _apply_binning_frame(_to_float32_frame(hdul[0].data), bin_factor)
            cube.flush()
            del cube
            cube_r = np.memmap(tmp_path, dtype=np.float32, mode="r", shape=(n, h0, w0))
            master = np.empty((h0, w0), dtype=np.float32)
            tile_h = max(1, int(_STACK_SLAB_RAM_BYTES // max(1, n * w0 * 4)))
            for y0 in range(0, h0, tile_h):
                y1 = min(h0, y0 + tile_h)
                block = np.asarray(cube_r[:, y0:y1, :], dtype=np.float32)
                master[y0:y1, :] = _combine_block(block)
            del cube_r
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if combine == "mean":
        header0["VYSTKMOD"] = ("MEAN", "Per-pixel mean along stack axis")
    else:
        header0["VYSTKMOD"] = ("MEDIAN", "Per-pixel median along stack axis")
    return master, header0


def _generate_master_dark(
    darks_files: list[Path],
    *,
    bin_factor: int = 1,
) -> tuple[np.ndarray, fits.Header]:
    """Build master dark: per-pixel mean over input frames (see :func:`_combine_stack_mean`)."""
    return _stack_calibration_frames(
        darks_files,
        combine="mean",
        bin_factor=bin_factor,
        kind="dark",
    )


def _generate_master_flat(
    flats_files: list[Path],
    *,
    bin_factor: int = 1,
) -> tuple[np.ndarray, fits.Header]:
    """Build master flat: per-pixel median over frames (normalization at calibrate; see ``VYFLNRD``)."""
    return _stack_calibration_frames(
        flats_files,
        combine="median",
        bin_factor=bin_factor,
        kind="flat",
    )


def get_calibration_status(
    path: str | Path | None,
    *,
    kind: str,
    validity_days: int,
    warnings: list[str] | None = None,
) -> CalibrationStatus:
    """Return calibration status for a given file/folder.

    - status=missing: path is None / doesn't exist / empty dir
    - status=expired: age older than validity_days (header capture date; mtime fallback)
    - status=ok: otherwise

    Boundary: age == validity_days is **ok** (same as library UI: expired only when age > limit).
    """
    if path is None:
        return CalibrationStatus(
            kind=kind,
            path=None,
            status="missing",
            last_modified_utc=None,
            age_days=None,
            validity_days=validity_days,
            message=f"{kind}: missing",
        )

    p = Path(path)
    if not p.exists():
        return CalibrationStatus(
            kind=kind,
            path=str(p),
            status="missing",
            last_modified_utc=None,
            age_days=None,
            validity_days=validity_days,
            message=f"{kind}: path not found",
        )

    check_path = p
    if check_path.is_dir():
        files = _list_fits_files(check_path)
        if not files:
            return CalibrationStatus(
                kind=kind,
                path=str(check_path),
                status="missing",
                last_modified_utc=None,
                age_days=None,
                validity_days=validity_days,
                message=f"{kind}: directory is empty",
            )
        check_path = max(files, key=lambda fp: os.path.getmtime(fp))

    age_f = _age_days(check_path, warnings=warnings)
    if age_f is None:
        return CalibrationStatus(
            kind=kind,
            path=str(p),
            status="missing",
            last_modified_utc=None,
            age_days=None,
            validity_days=validity_days,
            message=f"{kind}: cannot resolve master age",
        )

    info = resolve_master_age(check_path)
    capture = info.capture_utc
    last_str = capture.strftime("%Y-%m-%d %H:%M UTC") if capture is not None else None
    age_days = int(age_f)
    expired = age_f > validity_days

    if expired:
        return CalibrationStatus(
            kind=kind,
            path=str(p),
            status="expired",
            last_modified_utc=last_str,
            age_days=age_days,
            validity_days=validity_days,
            message=f"{kind}: expired (capture {last_str or 'unknown'}, {age_days} days old)",
        )

    return CalibrationStatus(
        kind=kind,
        path=str(p),
        status="ok",
        last_modified_utc=last_str,
        age_days=age_days,
        validity_days=validity_days,
        message=f"{kind}: ok ({age_days} days old)",
    )


def check_known_field(
    ra_deg: float,
    dec_deg: float,
    db: VyvarDatabase,
    *,
    match_radius_deg: float = 0.5,
) -> dict[str, Any] | None:
    """Return FIELD_REGISTRY match + comp-star library rows, or ``None`` (never raises)."""
    try:
        field = db.find_matching_field(ra_deg, dec_deg, match_radius_deg=match_radius_deg)
        if field is None:
            return None
        comp_stars = db.get_comp_stars_for_field(
            ra_deg,
            dec_deg,
            match_radius_deg=match_radius_deg,
            only_approved=True,
        )
        return {
            "field": field,
            "comp_stars": comp_stars,
            "n_comp_stars": len(comp_stars),
            "n_observations": int(field.get("N_OBSERVATIONS") or 1),
            "last_observation_id": field.get("LAST_OBSERVATION_ID"),
            "masterstar_path": field.get("MASTERSTAR_PATH"),
            "comparison_csv_path": field.get("COMPARISON_CSV_PATH"),
        }
    except Exception as exc:  # noqa: BLE001
        logging.warning('[EXC-0104] known-field lookup fail -> None -> field treated as new: %s', exc)
        return None

